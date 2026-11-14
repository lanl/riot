//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI.

#include "void/void.hpp"

#include <string>

#include "variables.hpp"

namespace Void {

// Volume of the void on this rank
Real ReduceVoidVolume(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto pm = md->GetParentPointer();
  const int matid = GetParams(pm).matid;

  static auto desc = parthenon::MakePackDescriptor<ccmat::rho, ccmat::volume_fraction>(
      pm->resolved_packages.get());
  auto v = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Void::ReduceVoidVolume", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ls) {
        const int mv = SliceIndex<ccmat::rho>(v, b, v.GetSize(b, ccmat::rho()), matid);
        if (mv < 0) return;
        ls += v(b, ccmat::volume_fraction(mv), k, j, i) *
              v.GetCoordinates(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(sum));
  return sum;
}

Params ParamsFromInput(ParameterInput *pin) {
  Params p;
  if (!pin->GetOrAddBoolean("physics", "void", false)) return p;
  const std::string block = "void";
  p.matid = pin->GetInteger(block, "matid");
  p.pressure = pin->GetOrAddReal(block, "pressure", 0.0, "Void pressure p_void");
  p.density = pin->GetOrAddReal(block, "density", 1.e-20, "Tracer density of the void");
  p.freeze = pin->GetOrAddBoolean(block, "freeze_pure_void", true,
                                  "Zero the velocity of pure-void cells");
  p.tol = pin->GetOrAddReal(block, "tol", 1.e-8, "Tolerance of the open-void solves");
  p.max_iter =
      pin->GetOrAddInteger(block, "max_iter", 100, "Iteration limit of open-void solves");
  return p;
}

void SetPlaceholderMaterial(ParameterInput *pin) {
  const int matid = MatidFromInput(pin);
  if (matid < 0) return;
  const std::string mat = "material" + std::to_string(matid);
  pin->SetString(mat, "eos", "void/eos");
  pin->SetString(mat, "opac", "void/eos"); // no opacity given: transparent
  pin->SetString(mat, "electron_eos", "void/electron_eos");
  pin->SetString("void/eos", "eos_type", "IdealGas");
  pin->SetReal("void/eos", "Gamma", 5.0 / 3.0);
  pin->SetReal("void/eos", "Cv", 1.0);
  pin->SetString("void/electron_eos", "eos_type", "IdealElectrons");
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("void");
  const Params p = ParamsFromInput(pin);

  const std::string mat = "material" + std::to_string(p.matid);
  PARTHENON_REQUIRE(pin->DoesBlockExist(mat), "void/matid is not a material");
  PARTHENON_REQUIRE(!pin->GetOrAddBoolean(mat, "strong", false),
                    "The void material cannot be strong");
  PARTHENON_REQUIRE(pin->GetOrAddInteger(mat, "nphase", 1) == 1,
                    "The void material must have a single phase");
  PARTHENON_REQUIRE(p.pressure >= 0.0, "void/pressure must be non-negative");
  PARTHENON_REQUIRE(p.density > 0.0, "void/density must be positive");
  pkg->AddParam("params", p);

  using parthenon::HistoryOutputVar;
  const auto HstSum = parthenon::UserHistoryOperation::sum;
  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(HistoryOutputVar(HstSum, ReduceVoidVolume, "void_volume"));
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return pkg;
}

} // namespace Void
