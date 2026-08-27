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

#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "hydro/hydro.hpp"
#include "mix/mix.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Mix {

//----------------------------------------------------------------------------------------
//! \fn  void Mix::ComputeAnonFluxes
//  \brief Reads the diffusive mass flux deposited on the fm::diffusive_fluxes face field,
//  adds it to each material density's flux register, and advects all per-material
//  conserved quantities by that same diffusive mass flux.

template <parthenon::CoordinateDirection DIR, typename VInPack, typename ROutPack,
          typename SOutPack>
void ComputeAnonFluxesAlongDir(MeshData<Real> *md, const VInPack &v_in,
                               const ROutPack &r_out, const SOutPack &s_out) {
  namespace ccmat = cell_variables::cell_averaged::mat; // partial (conserved) density
  namespace fm = face_variables::mat;                   // diffusive mass flux register
  using TE = parthenon::TopologicalElement;

  const int nblocks = v_in.GetNBlocks();
  if (nblocks == 0) return;

  using lt = RiotUtils::LoopType<>;
  constexpr TE face = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, face);
  const auto delta = idx_space.GetDelta(DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nummat = r_out.GetSize(b, ccmat::rho());
        const int nadv = s_out.Size(b);

        // First deposit the diffusive mass flux onto each material density's flux
        // register (adds to hydro's advective mass flux, hence +=).
        for (int m = 0; m < nummat; m++) {
          auto pin = RiotLoop::make_sparse_pack_view(idx_range, v_in, m);
          auto fr = RiotLoop::make_sparse_flux_pack_view(idx_range, r_out, DIR, m);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            fr(ccmat::rho(), kji) += pin(face, fm::diffusive_fluxes(), kji);
          });
        }

        // Next, advect the mass-carried quantities by the diffusive mass flux. The map
        // from an advected variable's sparse id to its dense material index is per-block,
        // so it is resolved by a small search over materials (mirroring the hydro
        // advection kernel) rather than a scratch lookup table.
        for (int p = 0; p < nadv; p++) {
          const int sid = s_out.ConsSparseID(b, p);
          if (sid == parthenon::InvalidSparseID) continue;
          int m = -1;
          for (int mm = 0; mm < nummat; mm++) {
            if (r_out(b, ccmat::rho(mm)).sparse_id == sid) {
              m = mm;
              break;
            }
          }
          auto pin = RiotLoop::make_sparse_pack_view(idx_range, v_in, m);
          auto q = RiotLoop::make_var_view(idx_range, s_out.Prims(), p);
          auto fadv = RiotLoop::make_flux_view(idx_range, s_out.Cons(), DIR, p);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real massflux = pin(face, fm::diffusive_fluxes(), kji);
            fadv(kji) += 0.5 * (q(kji - delta) + q(kji)) * massflux;
          });
        }
      });
}

TaskStatus ComputeAnonFluxes(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat; // partial (conserved) density
  namespace fm = face_variables::mat;                   // diffusive mass flux register

  std::vector<int> matids;
  std::set<parthenon::PDOpt> withfluxes = {parthenon::PDOpt::WithFluxes};
  // v_in: the diffusive mass flux deposited by ComputeViscousFluxes (a face field).
  // r_out: the conserved partial densities, whose real flux registers receive the
  // diffusive mass flux. s_out: all anonymously advected conserved vars, advected by
  // that same diffusive mass flux.
  auto v_in = riot::MakePack<fm::diffusive_fluxes>(md);
  auto r_out = riot::MakePack<ccmat::rho>(md, matids, withfluxes);
  auto s_out = Hydro::MakeAdvectionPack(md);

  if (v_in.GetNBlocks() == 0) return TaskStatus::complete;
  const int ndim = md->GetParentPointer()->ndim;

  ComputeAnonFluxesAlongDir<X1DIR>(md, v_in, r_out, s_out);
  if (ndim > 1) ComputeAnonFluxesAlongDir<X2DIR>(md, v_in, r_out, s_out);
  if (ndim > 2) ComputeAnonFluxesAlongDir<X3DIR>(md, v_in, r_out, s_out);

  return TaskStatus::complete;
}
} // namespace Mix
