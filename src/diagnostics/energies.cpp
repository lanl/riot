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

#include "diagnostics.hpp"
#include "materials/materials.hpp"
#include "variables.hpp"

namespace energies {

std::vector<Real> ReduceInternalEnergies(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccmat::internal_energy>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  auto &materials = pm->packages.Get("materials");
  const int nummat = materials->Param<int>("max_array_size");
  std::vector<Real> energies(nummat);

  // TODO(JMM): I hate this strategy. Surely something better is possible?
  for (int matid_target = 0; matid_target < nummat; ++matid_target) {
    Real &u = energies[matid_target];
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag,
        "Compute total total internal energy of each material on this rank",
        DevExecSpace(), 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lu) {
          const int nmat_block = v.GetSize(b, ccmat::internal_energy());
          auto &coords = v.GetCoordinates(b);
          const Real dV = coords.CellVolume(k, j, i);
          for (int m = 0; m < nmat_block; ++m) {
            const int matid = v(b, ccmat::internal_energy(m)).sparse_id;
            if (matid == matid_target) {
              lu += v(b, ccmat::internal_energy(m), k, j, i) * dV;
            }
          }
        },
        Kokkos::Sum<Real>(u));
  }

  return energies;
}

Real ReduceTotalEnergy(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc =
      MakePackDescriptor<ccbulk::total_material_energy>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Real E;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag,
      "Compute total total material energy on this rank", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lu) {
        auto &coords = v.GetCoordinates(b);
        const Real dV = coords.CellVolume(k, j, i);
        lu += v(b, ccbulk::total_material_energy(), k, j, i) * dV;
      },
      Kokkos::Sum<Real>(E));

  return E;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using parthenon::SimTime;
  auto pkg = std::make_shared<StateDescriptor>("energies");

  auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVec_list hst_vecs = {};
  hst_vecs.emplace_back(parthenon::HistoryOutputVec(HstSum, ReduceInternalEnergies,
                                                    "material_internal_energy_sums"));
  pkg->AddParam<>(parthenon::hist_vec_param_key, hst_vecs);

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(
      parthenon::HistoryOutputVar(HstSum, ReduceTotalEnergy, "total_material_energy"));
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return pkg;
}

} // namespace energies
