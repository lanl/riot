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

#include <cmath>
#include <memory>
#include <string>

#include <globals.hpp>
#include <parthenon/package.hpp>

#include "thinc/thinc.hpp"

namespace THINC {
using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> THINC::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto thinc = std::make_shared<StateDescriptor>("thinc");
  Params &params = thinc->AllParams();

  PARTHENON_REQUIRE(parthenon::Globals::nghost > 1,
                    "THINC requires at least two ghost cells.");

  // Steepness parameter
  params.Add("beta",
             pin->GetOrAddReal("thinc", "beta", 1.6, "Profile steepness for interfaces"));

  return thinc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus THINC::CalculateStates
//! \brief Reconstruct the interface profile of every mixed cell once per stage.
TaskStatus CalculateStates(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  constexpr int N = RiotLimits::MAX_MATERIALS;
  if (md->NumBlocks() == 0) return TaskStatus::complete;
  auto pmesh = md->GetParentPointer();
  const int ndim = pmesh->ndim;
  const Real beta = pmesh->packages.Get("thinc")->Param<Real>("beta");
  const bool viscosity =
      pmesh->packages.Get("riot")->Param<bool>("do_ionization") &&
      pmesh->packages.Get("ionization")->Param<bool>("plasma_viscosity");

  auto v = riot::MakePack<ccmat::volume_fraction, ccmat::thinc_states>(md);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  // Every cell reconstructed by the flux sweeps, see Hydro::CalculateFluxesImpl
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 2 * viscosity + 1, nblocks,
                                     md, parthenon::TopologicalElement::CC);
  const auto dx = idx_space.GetDelta(X1DIR);
  const auto dy = idx_space.GetDelta(X2DIR);
  const auto dz = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int count = v.GetSize(b, ccmat::volume_fraction());
        const auto &coords = v.GetCoordinates(b);
        RiotLoop::inner(idx_range, [&](auto kji) {
          // Pure cells keep ordinary hydro
          if (!MixedCell(v, idx_range, count, kji)) return;

          Real alpha[N], faces[N][6];
          Point gradient[N];
          int ids[N];
          for (int m = 0; m < count; ++m) {
            auto pm = RiotLoop::make_sparse_pack_view(idx_range, v, m);
            alpha[m] = pm(ccmat::volume_fraction(), kji);
            auto A = [&](auto offset) {
              return pm(ccmat::volume_fraction(), kji + offset);
            };
            gradient[m] = YoungsGradient(A, ndim, dx, dy, dz);
            ids[m] = v(b, ccmat::volume_fraction(m)).sparse_id;
          }
          const auto [k, j, i] = idx_range.GetKJI(kji);
          Real radial[4];
          RadialWeights(coords, i, radial);
          ReconstructProfile<N>(ndim, count, alpha, gradient, ids, radial, beta, faces);
          for (int m = 0; m < count; ++m) {
            auto pm = RiotLoop::make_sparse_pack_view(idx_range, v, m);
            for (int c = 0; c < 2 * ndim; ++c)
              pm(ccmat::thinc_states(c), kji) = faces[m][c];
          }
        });
      });
  return TaskStatus::complete;
}

} // namespace THINC
