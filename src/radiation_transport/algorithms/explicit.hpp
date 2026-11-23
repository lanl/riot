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
#ifndef RADIATION_TRANSPORT_ALGORITHMS_EXPLICIT_HPP_
#define RADIATION_TRANSPORT_ALGORITHMS_EXPLICIT_HPP_
// This file was made in part with generative AI.

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include "variables.hpp"

using namespace parthenon;
using namespace parthenon::package::prelude;

namespace Explicit {

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *materials);
Real EstimateTimestepMesh(MeshData<Real> *md);
Real EstimateTimestep(MeshData<Real> *md, const Real dt_ratio_hyperbolic);

//----------------------------------------------------------------------------------------
// Wrapper Explicit TaskCollection
TaskCollection ExplicitTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt);

//----------------------------------------------------------------------------------------
// Explicit Transport
TaskCollection ExplicitTransport(Mesh *pmesh, const int nsteps, const Real time,
                                 const Real dt);
TaskID CreateExplicitTaskList(const TaskID &begin, const int i, Mesh *pmesh,
                              TaskList &cycler, TaskID cycler_id,
                              parthenon::LowStorageIntegrator *integrator,
                              std::shared_ptr<MeshData<Real>> base,
                              std::shared_ptr<MeshData<Real>> r0,
                              std::shared_ptr<MeshData<Real>> r1, const int nsteps,
                              const Real dt);
TaskStatus PrepareForSubcycler(Mesh *pmesh, const Real time);
TaskStatus IncrementCounter(Mesh *pmesh, const Real dt);
TaskStatus CompletionFunction(int i, MeshData<Real> *r0, MeshData<Real> *r1,
                              const int nsteps);
TaskStatus UpdateOpacities(MeshData<Real> *md);
TaskStatus SetSource(MeshData<Real> *md);
TaskStatus CalculateFluxes(MeshData<Real> *r0);
TaskStatus Update(MeshData<Real> *r0, MeshData<Real> *r1, const Real g0, const Real g1,
                  const Real bdt);
TaskStatus ApplyRHS(MeshData<Real> *md, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn  Real SpatialFlux
//! \brief Source flux (see FaceWeights) through the lower DIR face of cell (k, j, i),
//! F = b cn I_up + (1 - b) cn (S_L + S_R) / 2, whose optical depth is the cell width dx
//! times the smaller total opacity of the two cells (zero without coupling: pure upwind)
template <int DIR, typename V1, typename V2>
KOKKOS_FORCEINLINE_FUNCTION static Real
SpatialFlux(const V1 &ii, const V2 &pack, const Real &cc, const Real &nn,
            const bool &coupling, const int &b, const int &n, const int &gg, const int &k,
            const int &j, const int &i, const Real &dx) {
  namespace ccr = cell_variables::cell_averaged::rad;

  // Indexing gymnastics
  const int koff = k - (DIR == X3DIR);
  const int joff = j - (DIR == X2DIR);
  const int ioff = i - (DIR == X1DIR);
  const Real &iil = ii(b, n, koff, joff, ioff);
  const Real &iir = ii(b, n, k, j, i);

  // Face optical depth and weights
  const Real tau =
      (coupling)
          ? dx * std::min(pack(b, ccr::aa(gg), koff, joff, ioff) +
                              pack(b, ccr::ss(gg), koff, joff, ioff),
                          pack(b, ccr::aa(gg), k, j, i) + pack(b, ccr::ss(gg), k, j, i))
          : 0.0;
  const auto [fb, fc] = FaceWeights(nn, tau);

  const Real &sfl = pack(b, ccr::source(gg), koff, joff, ioff);
  const Real &sfr = pack(b, ccr::source(gg), k, j, i);
  return cc * nn * (fb * ((nn >= 0.0) ? iil : iir) + fc * 0.5 * (sfl + sfr));
}

} // namespace Explicit

#endif // RADIATION_TRANSPORT_ALGORITHMS_EXPLICIT_HPP_
