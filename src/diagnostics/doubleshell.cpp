//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

// C/C++ headers
#include <cmath>

// Parthenon headers
#include <sparse/sparse_management.hpp>

// Riot headers
#include "diagnostics.hpp"
#include "riot_utils/riot_loops.hpp"

namespace doubleshell {

//----------------------------------------------------------------------------------------
//! \fn  Real doubleshell::DTPeakTemperature
//! \brief User-history function that tracks peak DT temperature
Real DTPeakTemperature(MeshData<Real> *md) {
  namespace cm = cell_variables::material_averaged;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc = MakePackDescriptor<cm::temperature>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmin = -std::numeric_limits<Real>::max();
  constexpr int dt_mat_id = 5; // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real peakt = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        // sparse_id is unique per material, so at most one material matches; hoist that
        // search out of the inner reduction.
        for (int n = 0; n < vmesh.GetSize(b, cm::temperature()); n++) {
          if (vmesh(b, cm::temperature(n)).sparse_id != dt_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmax) {
            lmax = std::max(lmax, pv_n(cm::temperature(), idx));
          });
        }
      });

  return peakt;
}

//----------------------------------------------------------------------------------------
//! \fn  Real doubleshell::AddHistory
//! \brief User-history enrollment function
void AddHistory(parthenon::Params &params) {
  auto HstMax = parthenon::UserHistoryOperation::max;
  // auto HstMin = parthenon::UserHistoryOperation::min;
  // auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};

  // Leaving some of these as placeholders for now.
  // HstSum
  // hst_vars.emplace_back(HstSum, ShellMass, "ShellMass");
  // HstMax
  hst_vars.emplace_back(HstMax, DTPeakTemperature, "DTPeakTemperature");
  // HstMin
  // hst_vars.emplace_back(HstMin, SpikeBubbleMinLoc, "SpikeBubbleMinLoc");

  params.Add(parthenon::hist_param_key, hst_vars);
}

//----------------------------------------------------------------------------------------
//! \fn  Real doubleshell::Initialize
//! \brief Creates doubleshell package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("doubleshell");
  doubleshell::AddHistory(pkg->AllParams());
  return pkg;
}

} // namespace doubleshell
