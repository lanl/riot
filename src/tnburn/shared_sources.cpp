//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <string>

// Parthenon includes
#include <parthenon/package.hpp>

// Riot includes
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "tnburn.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace TNBurn {

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskStatus TNBurn::SharedSources
//! \brief This routine is supposed to take mass and energy sources and provide additional
//! terms for momentum, energy (kinetic), and material quantities
parthenon::TaskStatus SharedSources(MeshData<Real> *sourcein, MeshData<Real> *sourceout,
                                    MeshData<Real> *state) {
  auto pm = state->GetParentPointer();

  namespace c = cell_variables;
  namespace cc = c::cell_averaged;
  namespace ccbulk = cc::bulk;
  namespace ccmat = cc::mat;
  namespace cm = c::material_averaged;
  // We need to grab material density and energy sources (c.c.mat.rho and
  // c.c.mat.internal_energy) We need state variables for density, velocity and all the
  // Associated quantities (including phase fraction) We need source variables for
  // conserved/advected/sparse/
  using parthenon::MakePackDescriptor;
  auto &resolved_packages = pm->resolved_packages;

  // Get the list of all the relevant anonymous variables
  static const auto BurnFlag = Metadata::GetOrAddFlag(riot::metadata::TNBurn);
  parthenon::Metadata::FlagCollection flags{Metadata::Independent, Metadata::Advected};
  flags.Exclude(BurnFlag); // to avoid double counting isotopes
  auto [conserved_vars, prims_vars] = RiotUtils::GetAssociatedVars(sourceout, flags);
  static std::vector<bool> use_regex(conserved_vars.size(), false);

  static auto desc_anon_state =
      parthenon::MakePackDescriptor(resolved_packages.get(), prims_vars, use_regex);
  static auto desc_anon_out =
      parthenon::MakePackDescriptor(resolved_packages.get(), conserved_vars, use_regex);
  auto v_anon_state = riot::GetPack(desc_anon_state, sourcein);
  auto v_anon_out = riot::GetPack(desc_anon_out, sourceout);

  // Pack up the "known" quantities from all of the kinds of data blocks
  auto v_s_in = riot::MakePack<ccmat::rho>(sourcein);
  auto v_s_out =
      riot::MakePack<ccbulk::momentum, ccbulk::total_material_energy>(sourceout);
  auto v_state = riot::MakePack<ccmat::rho, ccbulk::velocity>(state);

  const int nblocks = state->NumBlocks();

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nummat = v_state.GetSize(b, ccmat::rho());
        const int numvel = v_state.GetSize(b, ccbulk::velocity());
        const int nanon = v_anon_state.GetUpperBound(b) + 1;

        // First the "known" quantities
        // Momentum and KE
        auto pv_out = RiotLoop::make_pack_view(idx_range, v_s_out);
        auto pv_vel = RiotLoop::make_pack_view(idx_range, v_state);
        for (int m = 0; m < nummat; m++) {
          auto rho = RiotLoop::make_var_view(idx_range, v_s_in, ccmat::rho(m));
          for (int d = 0; d < numvel; d++) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real vel = pv_vel(ccbulk::velocity(d), kji);
              pv_out(ccbulk::total_material_energy(), kji) +=
                  0.5 * rho(kji) * vel * vel;                     // Energy
              pv_out(ccbulk::momentum(d), kji) += rho(kji) * vel; // Momentum
            });
            idx_range.TeamBarrier();
          }
        }
        // Now "Anonymous" terms (including phase density)
        for (int p = 0; p < nanon; p++) {
          const int m1 = v_anon_out(b, p).sparse_id;
          auto lhs = RiotLoop::make_var_view(idx_range, v_anon_out, p);
          auto rhs = RiotLoop::make_var_view(idx_range, v_anon_state, p);
          for (int m = 0; m < nummat; m++) {
            // Add in sources if this phase is same material or if it is not sparse
            if (m1 == v_s_in(b, ccmat::rho(m)).sparse_id ||
                m1 == parthenon::InvalidSparseID) {
              auto rho = RiotLoop::make_var_view(idx_range, v_s_in, ccmat::rho(m));
              RiotLoop::inner(idx_range,
                              [&](const auto kji) { lhs(kji) += rhs(kji) * rho(kji); });
            }
            idx_range.TeamBarrier();
          }
        }
      });

  return parthenon::TaskStatus::complete;
}

} // namespace TNBurn
