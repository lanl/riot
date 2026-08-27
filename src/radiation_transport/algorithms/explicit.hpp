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
TaskStatus CalculateFluxes(MeshData<Real> *r0);
TaskStatus Update(MeshData<Real> *r0, MeshData<Real> *r1, const Real g0, const Real g1,
                  const Real bdt);
TaskStatus ApplyRHS(MeshData<Real> *md, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn  SpatialFlux
//! \brief
template <int DIR, FluxType FT = FluxType::hll, typename V1, typename V2>
KOKKOS_FORCEINLINE_FUNCTION static Real
SpatialFlux(const V1 &ii, const Real &cn, const int &b, const int &n, const int &k,
            const int &j, const int &i, const V2 &opac, const int &gg, const Real &beta,
            const Real dx, const Real &tmax) {
  namespace ccr = cell_variables::cell_averaged::rad;

  // Indexing gymnastics
  const bool up = (cn >= 0.0);
  const bool down = (cn < 0.0);
  const int sgnn = up - down;
  const int koff = k - (DIR == X3DIR);
  const int joff = j - (DIR == X2DIR);
  const int ioff = i - (DIR == X1DIR);
  const Real &iil = ii(b, n, koff, joff, ioff);
  const Real &iir = ii(b, n, k, j, i);

  // Flux selection
  [[maybe_unused]] auto &opac_ = opac;
  [[maybe_unused]] auto &beta_ = beta;
  [[maybe_unused]] auto &tmax_ = tmax;
  if constexpr (FT == FluxType::hll) {
    const Real mm =
        OpacityStencil((opac_(b, ccr::aa(gg), koff, joff, ioff) +
                        opac_(b, ccr::ss(gg), koff, joff, ioff)),
                       (opac_(b, ccr::aa(gg), k, j, i) + opac_(b, ccr::ss(gg), k, j, i)));
    const Real tauw = 1.0 / (1.0 + std::min(beta_ * dx * mm, tmax_));
    return 0.5 * cn * (iil + iir + tauw * sgnn * (iil - iir));
  } else if constexpr (FT == FluxType::upwind) {
    return cn * (up ? iil : iir);
  }

  return Null<Real>();
}

} // namespace Explicit

#endif // RADIATION_TRANSPORT_ALGORITHMS_EXPLICIT_HPP_
