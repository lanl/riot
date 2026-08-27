//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef RIOT_PGEN_CONDUCTION_ANALYTIC_HPP_
#define RIOT_PGEN_CONDUCTION_ANALYTIC_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>
namespace conduction_analytic {
using namespace parthenon::package::prelude;
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
void ProblemModifier(parthenon::ParthenonManager *pman);
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);

/// Compute the cell average of u = beta * sin(k·x) * exp(-D |k|^2 t)
/// over a box centered at (x,y,z) with side lengths (dx,dy,dz).
///
/// Returns a tuple: (integral, average)
KOKKOS_INLINE_FUNCTION
Real cell_average_solution(const Real x, const Real y, const Real z, const Real dx,
                           const Real dy, const Real dz, const Real kx, const Real ky,
                           const Real kz, const Real beta, const Real D, const Real t) {
  // squared wavenumber and decay factor
  const double k2 = kx * kx + ky * ky + kz * kz;
  const double decay = std::exp(-D * k2 * t);

  // local phase
  const double phase = kx * x + ky * y + kz * z;
  const double sin_phase = std::sin(phase);

  // Helper: sinc(k,L) = 2*sin(kL/2)/(kL) , limit 1 if k≈0
  auto sinc = [](double k, double L) -> double {
    if (std::abs(k) < 1e-14) return 1.;
    return 2.0 * std::sin(0.5 * k * L) / (k * L);
  };

  const double Ix = sinc(kx, dx);
  const double Iy = sinc(ky, dy);
  const double Iz = sinc(kz, dz);

  // The sinc function already contains 1/dx so we already have a volume
  // average
  const double integral_average = beta * decay * Ix * Iy * Iz * sin_phase;

  return integral_average;
} // cell_average_solution

} // namespace conduction_analytic
#endif // RIOT_PGEN_CONDUCTION_ANALYTIC_HPP_
