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
// Jacobi Transport (host-driven, subcycle-capable): solve and commit phases
TaskCollection JacobiSolve(Mesh *pmesh, const Real time, const Real dt,
                           const Real dt_ratio);
TaskCollection JacobiCommit(Mesh *pmesh, const Real dt);
TaskID CreateJacobiTaskList(const TaskID &begin, const int i, Mesh *pmesh,
                            TaskList &solver, TaskID solver_id,
                            parthenon::AllReduce<HostArray1D<Real>> *presidual,
                            std::shared_ptr<MeshData<Real>> rbase,
                            std::shared_ptr<MeshData<Real>> riter,
                            std::shared_ptr<MeshData<Real>> rout,
                            std::shared_ptr<MeshData<Real>> ubase, const Real dt);
TaskStatus CheckConvergence(HostArray1D<Real> *residual, MeshData<Real> *rbase,
                            MeshData<Real> *riter, MeshData<Real> *rout);
TaskStatus IncrementCounterAndSetResidual(HostArray1D<Real> *vresidual, Mesh *pmesh);
TaskStatus CompletionFunction(int i, HostArray1D<Real> *presidual, MeshData<Real> *riter,
                              MeshData<Real> *rout);
TaskStatus SetOpacities(MeshData<Real> *rbase, MeshData<Real> *ubase);
TaskStatus ResetIterator(MeshData<Real> *rbase, const Real time, const Real dt);
TaskStatus ExtrapolateIterate(MeshData<Real> *rbase, MeshData<Real> *riter,
                              const Real dt_ratio);
TaskStatus CopyToIterate(MeshData<Real> *src, MeshData<Real> *riter);
TaskStatus JacobiUpdate(MeshData<Real> *rbase, MeshData<Real> *riter,
                        MeshData<Real> *rout, const Real dt, const int color);

//----------------------------------------------------------------------------------------
// Coupling
TaskStatus UpdateOpacities(MeshData<Real> *rbase, MeshData<Real> *ubase);
TaskStatus ApplyRHS(MeshData<Real> *rbase, MeshData<Real> *rout, MeshData<Real> *ubase,
                    const Real dt, const int color);
TaskStatus JacobiFeedback(MeshData<Real> *rbase, MeshData<Real> *riter,
                          MeshData<Real> *rout, MeshData<Real> *ubase, const Real dt);

//----------------------------------------------------------------------------------------
//! \fn  Real TauWeight
//! \brief Assigns optical depth weights for Rusanov flux
KOKKOS_FORCEINLINE_FUNCTION static Real TauWeight(const Real &tau, const Real &tmax) {
  return 1.0 / (1.0 + std::min(tau, tmax));
}

//----------------------------------------------------------------------------------------
//! \fn  std::array<Real, 6> FaceTauWeights
//! \brief Optical depth weights on the lower and upper faces of each active direction
template <typename V1>
KOKKOS_FORCEINLINE_FUNCTION static std::array<Real, 6>
FaceTauWeights(V1 &pack, const parthenon::Coordinates_t &coords, const int ndim,
               const FluxUtils &futils, const int &gg, const int &b, const int &k,
               const int &j, const int &i) {
  namespace ccr = cell_variables::cell_averaged::rad;

  std::array<Real, 6> tw = {0.0};
  const Real sc = pack(b, ccr::aa(gg), k, j, i) + pack(b, ccr::ss(gg), k, j, i);
  for (int dir = X1DIR; dir <= ndim; ++dir) {
    const bool x1d = (dir == X1DIR);
    const bool x2d = (dir == X2DIR);
    const bool x3d = (dir == X3DIR);

    // Total opacities
    const Real bdx = futils.beta * coords.CellWidth(dir, k, j, i);
    const Real sl = pack(b, ccr::aa(gg), k - x3d, j - x2d, i - x1d) +
                    pack(b, ccr::ss(gg), k - x3d, j - x2d, i - x1d);
    const Real sr = pack(b, ccr::aa(gg), k + x3d, j + x2d, i + x1d) +
                    pack(b, ccr::ss(gg), k + x3d, j + x2d, i + x1d);

    tw[2 * (dir - 1)] = TauWeight(bdx * std::min(sl, sc), futils.taumax);
    tw[2 * (dir - 1) + 1] = TauWeight(bdx * std::min(sr, sc), futils.taumax);
  }

  return tw;
}

//----------------------------------------------------------------------------------------
//! \fn  std::array<Real, 6> FaceAreas
//! \brief Areas of the lower and upper faces of each active direction
KOKKOS_FORCEINLINE_FUNCTION static std::array<Real, 6>
FaceAreas(const parthenon::Coordinates_t &coords, const int ndim, const int &k,
          const int &j, const int &i) {
  std::array<Real, 6> fa = {0.0};
  fa[0] = coords.FaceArea<X1DIR>(k, j, i);
  fa[1] = coords.FaceArea<X1DIR>(k, j, i + 1);
  if (ndim > 1) {
    fa[2] = coords.FaceArea<X2DIR>(k, j, i);
    fa[3] = coords.FaceArea<X2DIR>(k, j + 1, i);
  }
  if (ndim > 2) {
    fa[4] = coords.FaceArea<X3DIR>(k, j, i);
    fa[5] = coords.FaceArea<X3DIR>(k + 1, j, i);
  }
  return fa;
}

//----------------------------------------------------------------------------------------
//! \fn  void GCoef
//! \brief Computes matrix coefficients for Jacobi
template <int SDIR>
KOKKOS_FORCEINLINE_FUNCTION static void
GCoef(const std::array<Real, 6> &fa, const ParArrayND<Real> &cp,
      const std::array<int, 3> &NDIR, const Real &ivol, const bool &split_g1,
      const std::array<Real, 6> &tw, const int &aa, Real &g1p, Real &g1pp, Real &glra,
      Real &glrb) {
  // Directions
  const Real &nn = cp(aa, NDIR[SDIR - 1]);
  const bool up = (nn >= 0.0);
  const bool down = (nn < 0.0);
  const int sgnn = up - down;

  // Compute L and R wave speeds at i face
  Real ww = ivol * nn * fa[2 * (SDIR - 1)];
  Real ss = sgnn * tw[2 * (SDIR - 1)];
  g1p += ww * (ss - (!split_g1 || down));
  g1pp += -split_g1 * up * ww;
  glra = -ww * (1.0 + ss);

  // Compute L and R wave speeds at i+1 face
  ww = ivol * nn * fa[2 * (SDIR - 1) + 1];
  ss = sgnn * tw[2 * (SDIR - 1) + 1];
  g1p += ww * (ss + (!split_g1 || up));
  g1pp += split_g1 * down * ww;
  glrb = ww * (1.0 - ss);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void GCoef
//! \brief Computes matrix coefficients for Jacobi
KOKKOS_FORCEINLINE_FUNCTION static Real
G1Coef(const std::array<Real, 6> &fa, const ParArrayND<Real> &cp,
       const std::array<int, 3> &NDIR, const int ndim, const ParArrayND<Real> &wght,
       const ParArrayND<Real> &gflx, const ParArrayND<Real> &arcw,
       const ParArrayND<int> &numn, const Real &ivol, const Real &mcw,
       const bool &split_g1, const std::array<Real, 6> &tw, const int &aa) {
  // Contributions from each spatial flux direction
  Real g1p = 1.0;
  for (int dir = X1DIR; dir <= ndim; ++dir) {
    // Directions
    const Real &nn = cp(aa, NDIR[dir - 1]);
    const bool up = (nn >= 0.0);
    const bool down = (nn < 0.0);
    const int sgnn = up - down;

    // Compute L and R wave speeds at i face
    Real ww = ivol * nn * fa[2 * (dir - 1)];
    Real ss = sgnn * tw[2 * (dir - 1)];
    g1p += ww * (ss - (!split_g1 || down));

    // Compute L and R wave speeds at i+1 face
    ww = ivol * nn * fa[2 * (dir - 1) + 1];
    ss = sgnn * tw[2 * (dir - 1) + 1];
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
