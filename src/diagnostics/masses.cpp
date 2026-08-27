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

#include <vector>

#include <globals.hpp>
#include <parthenon_mpi.hpp>

#include "diagnostics.hpp"
#include "materials/materials.hpp"
#include "variables.hpp"

#include "diagnostics/masses.hpp"

namespace masses {
std::vector<Real> ReduceMassesLocal(parthenon::MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccmat::rho>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  auto &materials = pm->packages.Get("materials");
  auto &mass_pkg = pm->packages.Get("masses");
  const auto &matids = materials->Param<std::vector<int>>("matids");
  int nummat = matids.size();

  std::vector<Real> mass_sum(nummat);

  for (int matid_target = 0; matid_target < nummat; ++matid_target) {
    Real mass_mat = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag,
        "Compute total masses of each material on this rank", DevExecSpace(), 0,
        v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lm) {
          const int nmat_block = v.GetSize(b, ccmat::rho());
          auto &coords = v.GetCoordinates(b);
          for (int m = 0; m < nmat_block; ++m) {
            const int matid = v(b, ccmat::rho(m)).sparse_id;
            if (matid == matid_target) {
              const Real rho = v(b, ccmat::rho(m), k, j, i);
              const Real dV = coords.CellVolume(k, j, i);
              lm += rho * dV;
            }
          }
        },
        Kokkos::Sum<Real>(mass_mat));
    mass_sum[matid_target] = mass_mat;
  }

  return mass_sum;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using parthenon::SimTime;
  auto pkg = std::make_shared<StateDescriptor>("masses");

  using parthenon::HistoryOutputVar;
  parthenon::HstVec_list hst_vecs = {};
  hst_vecs.emplace_back(parthenon::HistoryOutputVec(
      parthenon::UserHistoryOperation::sum, ReduceMassesLocal, "material_mass_sums"));
  pkg->AddParam<>(parthenon::hist_vec_param_key, hst_vecs);

  return pkg;
}

} // namespace masses
