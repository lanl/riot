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
#ifndef RADIATION_TRANSPORT_ALGORITHMS_JACOBI_HPP_
#define RADIATION_TRANSPORT_ALGORITHMS_JACOBI_HPP_
// This file was made in part with generative AI.

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include "variables.hpp"

using namespace parthenon;
using namespace parthenon::package::prelude;

namespace Jacobi {

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *materials);
Real EstimateTimestepMesh(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
// Wrapper Jacobi TaskCollection
TaskCollection JacobiTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt);

//----------------------------------------------------------------------------------------
// Jacobi Transport
TaskCollection JacobiTransport(Mesh *pmesh, const Real time, const Real dt);
TaskID CreateJacobiTaskList(const TaskID &begin, const int i, Mesh *pmesh,
                            TaskList &solver, TaskID solver_id,
                            parthenon::AllReduce<HostArray1D<Real>> *presidual,
                            std::shared_ptr<MeshData<Real>> rbase,
                            std::shared_ptr<MeshData<Real>> riter,
                            std::shared_ptr<MeshData<Real>> rout,
                            std::shared_ptr<MeshData<Real>> ubase, const Real dt);
TaskStatus CheckConvergence(HostArray1D<Real> *residual, MeshData<Real> *riter,
                            MeshData<Real> *rout);
TaskStatus IncrementCounterAndSetResidual(HostArray1D<Real> *vresidual, Mesh *pmesh);
TaskStatus CompletionFunction(int i, HostArray1D<Real> *presidual, MeshData<Real> *riter,
                              MeshData<Real> *rout);
TaskStatus SetOpacities(MeshData<Real> *rbase, MeshData<Real> *ubase);
TaskStatus ResetIterator(MeshData<Real> *rbase, const Real time, const Real dt);
TaskStatus PrepareForIterations(MeshData<Real> *rbase);
TaskStatus JacobiUpdate(MeshData<Real> *rbase, MeshData<Real> *riter,
                        MeshData<Real> *rout, const Real dt);

//----------------------------------------------------------------------------------------
// Coupling
TaskStatus UpdateOpacities(MeshData<Real> *rbase, MeshData<Real> *ubase);
TaskStatus ApplyRHS(MeshData<Real> *rbase, MeshData<Real> *rout, MeshData<Real> *ubase,
                    const Real dt);
TaskStatus JacobiFeedback(MeshData<Real> *rbase, MeshData<Real> *riter,
                          MeshData<Real> *rout, MeshData<Real> *ubase, const Real dt);

//----------------------------------------------------------------------------------------
//! \fn  Real TauWeight
//! \brief Assigns optical depth weights for Rusanov flux
KOKKOS_FORCEINLINE_FUNCTION static Real TauWeight(const Real &tau, const Real &tmax) {
  return 1.0 / (1.0 + std::min(tau, tmax));
}

//----------------------------------------------------------------------------------------
//! \fn  void GCoef
//! \brief Computes matrix coefficients for Jacobi
template <int SDIR, typename V1>
KOKKOS_FORCEINLINE_FUNCTION static void
GCoef(V1 &pack, const parthenon::Coordinates_t &coords, const ParArrayND<Real> &cp,
      const std::array<int, 3> &NDIR, const Real &ivol, const bool &split_g1,
      const int &gg, const int &aa, const int &b, const int &k, const int &j,
      const int &i, Real &g1p, Real &g1pp, Real &glra, Real &glrb) {
  namespace ccr = cell_variables::cell_averaged::rad;

  // Directions
  const Real &nn = cp(aa, NDIR[SDIR - 1]);
  const bool up = (nn >= 0.0);
  const bool down = (nn < 0.0);
  const int sgnn = up - down;
  constexpr bool x1d = (SDIR == X1DIR);
  constexpr bool x2d = (SDIR == X2DIR);
  constexpr bool x3d = (SDIR == X3DIR);

  // Compute L and R wave speeds at i face
  Real ww = ivol * nn * coords.FaceArea<SDIR>(k, j, i);
  Real tauw = std::max(pack(b, ccr::tauw(gg), k - x3d, j - x2d, i - x1d),
                       pack(b, ccr::tauw(gg), k, j, i));
  Real ss = sgnn * tauw;
  g1p += ww * (ss - (!split_g1 || down));
  g1pp += -split_g1 * up * ww;
  glra = -ww * (1.0 + ss);

  // Compute L and R wave speeds at i+1 face
  ww = ivol * nn * coords.FaceArea<SDIR>(k + x3d, j + x2d, i + x1d);
  tauw = std::max(pack(b, ccr::tauw(gg), k + x3d, j + x2d, i + x1d),
                  pack(b, ccr::tauw(gg), k, j, i));
  ss = sgnn * tauw;
  g1p += ww * (ss + (!split_g1 || up));
  g1pp += split_g1 * down * ww;
  glrb = ww * (1.0 - ss);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void GCoef
//! \brief Computes matrix coefficients for Jacobi
template <typename V1>
KOKKOS_FORCEINLINE_FUNCTION static Real
G1Coef(V1 &pack, const parthenon::Coordinates_t &coords, const ParArrayND<Real> &cp,
       const std::array<int, 3> &NDIR, const int ndim, const ParArrayND<Real> &wght,
       const ParArrayND<Real> &gflx, const ParArrayND<Real> &arcw,
       const ParArrayND<int> &numn, const Real &ivol, const Real &mcw,
       const bool &split_g1, const int &gg, const int &aa, const int &b, const int &k,
       const int &j, const int &i) {
  namespace ccr = cell_variables::cell_averaged::rad;

  // Contributions from each spatial flux direction
  Real g1p = 1.0;
  for (int dir = X1DIR; dir <= ndim; ++dir) {
    // Directions
    const Real &nn = cp(aa, NDIR[dir - 1]);
    const bool up = (nn >= 0.0);
    const bool down = (nn < 0.0);
    const int sgnn = up - down;
    const bool x1d = (dir == X1DIR);
    const bool x2d = (dir == X2DIR);
    const bool x3d = (dir == X3DIR);

    // Compute L and R wave speeds at i face
    Real ww = ivol * nn * coords.FaceArea(dir, k, j, i);
    Real tauw = std::max(pack(b, ccr::tauw(gg), k - x3d, j - x2d, i - x1d),
                         pack(b, ccr::tauw(gg), k, j, i));
    Real ss = sgnn * tauw;
    g1p += ww * (ss - (!split_g1 || down));

    // Compute L and R wave speeds at i+1 face
    ww = ivol * nn * coords.FaceArea(dir, k + x3d, j + x2d, i + x1d);
    tauw = std::max(pack(b, ccr::tauw(gg), k + x3d, j + x2d, i + x1d),
                    pack(b, ccr::tauw(gg), k, j, i));
    ss = sgnn * tauw;
    g1p += ww * (ss + (!split_g1 || up));
  }

  // Contributions from each angular flux direction
  [[maybe_unused]] auto &wght_ = wght;
  [[maybe_unused]] auto &numn_ = numn;
  [[maybe_unused]] auto &gflx_ = gflx;
  [[maybe_unused]] auto &arcw_ = arcw;
  if constexpr (do_angular_fluxes) {
    const Real mcw_over_ww = mcw / wght_(aa);
    for (int nb = 0; nb < numn_(aa); ++nb) {
      const Real na = mcw_over_ww * gflx_(aa, nb) * arcw_(aa, nb);
      g1p += (na > 0) * na;
    }
  }

  return g1p;
}

} // namespace Jacobi

#endif // RADIATION_TRANSPORT_ALGORITHMS_JACOBI_HPP_
