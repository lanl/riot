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
#ifndef LEVELSETS_LEVELSETS_HPP_
#define LEVELSETS_LEVELSETS_HPP_
// This file was made in part with generative AI.

#include <memory>

#include "parthenon/driver.hpp"
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

#include "reconstruction/reconstruction.hpp"

namespace Levelsets {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus ReinitializeStep(MeshData<Real> *mc, Real dt, int width);
TaskCollection Reinitialize(Mesh *pm, parthenon::SimTime &tm, Real dt);

void InitializeSignedDistance(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm);
TaskStatus StashZerothStep(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//! \fn  Real Levelsets::heaviside
//! \brief
KOKKOS_INLINE_FUNCTION
Real heaviside(const Real a, const Real dx) {
  Real h = 0.5 * (1. + a / dx + std::sin(M_PI * a / dx) / M_PI);
  return std::max(std::min(h, 1.), 0.);
}

//----------------------------------------------------------------------------------------
//! \fn  Real Levelsets::signh
//! \brief
template <typename Pack_t, typename VarType>
KOKKOS_INLINE_FUNCTION Real signh(const Pack_t &v, VarType var, const int k, const int j,
                                  const int i, const Real dx) {
  Real a = v(var, k, j, i);
  Real h = heaviside(a, dx);
  Real sgn = 2. * (h - 0.5);
  return sgn;
}

//----------------------------------------------------------------------------------------
//! \fn  std::array<Real, 4> Levelsets::upwind
//! \brief second order upwind reconstruction
KOKKOS_INLINE_FUNCTION
std::array<Real, 4> upwind(const std::array<Real, 5> x, const std::array<Real, 5> ux) {
  const int i = 2;
  Real sl, sr, slim;

  sl = (ux[i] - ux[i - 1]) / (x[i] - x[i - 1]) -
       (ux[i - 1] - ux[i - 2]) / (x[i - 1] - x[i - 2]);
  sr = (ux[i + 1] - ux[i]) / (x[i + 1] - x[i]) - (ux[i] - ux[i - 1]) / (x[i] - x[i - 1]);
  slim = RiotReconstruction::minmod(sl, sr);

  Real a = (ux[i] - ux[i - 1]) / (x[i] - x[i - 1]) + 0.5 * slim;
  Real am = 0.5 * (a - std::abs(a));
  Real ap = 0.5 * (a + std::abs(a));

  sl = (ux[i + 1] - ux[i]) / (x[i + 1] - x[i]) - (ux[i] - ux[i - 1]) / (x[i] - x[i - 1]);
  sr = (ux[i + 2] - ux[i + 1]) / (x[i + 2] - x[i + 1]) -
       (ux[i + 1] - ux[i]) / (x[i + 1] - x[i]);
  slim = RiotReconstruction::minmod(sl, sr);

  Real b = (ux[i + 1] - ux[i]) / (x[i + 1] - x[i]) - 0.5 * slim;
  Real bm = 0.5 * (b - std::abs(b));
  Real bp = 0.5 * (b + std::abs(b));

  return {am, ap, bm, bp};
}

} // namespace Levelsets

#endif // LEVELSETS_LEVELSETS_HPP_
