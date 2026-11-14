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
#ifndef VOID_VOID_CLOSURE_HPP_
#define VOID_VOID_CLOSURE_HPP_
// This file was made in part with generative AI.

// Open-void state of a cell: every condensed material m (cell-averaged density rhobar_m)
// sits on its p_void isobar at a common temperature T, and alpha_m = rhobar_m / rho_m,
// alpha_V = 1 - sum_m alpha_m > 0. The void is a constant-pressure reservoir with energy
// density -p_void (zero enthalpy), so the work p_void dV is stored in it and the energy
// of the cell E = sum_m rhobar_m e_m - p_void alpha_V fixes the enthalpy of the
// materials,
//   rho_m(T): p_m(rho_m, T) = p_void,   F(T) = sum_m rhobar_m h_m(T) - (E + p_void) = 0,
// with h_m = e_m + p_void / rho_m. The inner solve is a bracketed Newton iteration in rho
// on the stable (dp/drho > 0) branch, the outer one a bracketed Newton iteration in
// log(T) with dF/dT = sum_m rhobar_m c_p,m. The void is closed if sum_m alpha_m >= 1 at
// the root.

#include <array>
#include <cmath>
#include <limits>

#include <ports-of-call/portability.hpp>

#include "riot_limits.hpp"

namespace Void {

// Relative finite-difference step for the EOS derivatives
constexpr Real fd_h = 1.e-6;

// Density rho of the material at temperature T and pressure p_void, on a stable
// (dp/drho > 0) branch, starting from rho_guess. Returns false if the solve failed.
template <typename Eos_t, typename Lambda_t>
KOKKOS_INLINE_FUNCTION bool
IsobarDensity(const Eos_t &eos, Lambda_t &&lambda, const Real p_void, const Real T,
              const Real rho_guess, const Real tol, const int max_iter, Real &rho,
              Real &dpdrho, int &neos) {
  auto P = [&](const Real r) {
    neos++;
    return eos.PressureFromDensityTemperature(r, T, lambda);
  };

  // Bracket [lo: p < p_void, hi: p > p_void], negative until found. As in singularity's
  // DensityEnergyFromPressureTemperature, the density of the pressure minimum of the
  // isotherm (if any) bounds the condensed branch: the root is above it if that pressure
  // is below p_void (so the root is the condensed one, not one in the vapor dome), else
  // below it
  Real lo = -1.0, hi = -1.0;
  Real r = rho_guess;
  const Real rmin = eos.RhoPmin(T);
  if (rmin > 0.0) {
    if (P(rmin) < p_void) {
      lo = rmin;
      r = std::max(r, rmin);
    } else {
      hi = rmin;
      r = std::min(r, rmin);
    }
  }
  Real dr_old = std::numeric_limits<Real>::infinity();
  for (int it = 0; it < max_iter; it++) {
    const Real p = P(r);
    const Real dp = (P(r * (1.0 + fd_h)) - p) / (r * fd_h);
    if (!std::isfinite(p) || !std::isfinite(dp)) return false;
    if (p < p_void) {
      lo = r;
    } else {
      hi = r;
    }
    dpdrho = dp;

    // Newton step, safeguarded by the bracket (log-bisection, also if the step does not
    // halve, e.g. at the kink where a Maxwell plateau meets the liquid), or expansion
    // by up to 10x until the bracket is found
    Real r_new = (dp > 0.0) ? r + (p_void - p) / dp : -1.0;
    const bool bracketed = lo > 0.0 && hi > 0.0;
    if (hi < 0.0) {
      if (!(r_new > r)) r_new = 10.0 * r;
      r_new = std::min(r_new, 10.0 * r);
    } else if (lo < 0.0) {
      if (!(r_new > 0.0 && r_new < r)) r_new = 0.1 * r;
      r_new = std::max(r_new, 0.1 * r);
    } else if (!(r_new > lo && r_new < hi) || std::abs(r_new - r) > 0.5 * dr_old) {
      r_new = std::sqrt(lo * hi);
    }
    if (bracketed) dr_old = std::abs(r_new - r);
    if (std::abs(r_new - r) <= tol * r || (bracketed && hi - lo <= tol * lo)) {
      rho = r_new;
      return true;
    }
    r = r_new;
  }
  return false;
}

// Open-void state of n materials. rho holds the density guesses on input and the
// isobar densities on output, T the temperature guess on input. E is the energy of the
// cell, including that of the void. Returns false if the void is closed at the energy E
// (or the solve failed); then the outputs are garbage.
// F(T) jumps where an isobar crosses a first-order phase transition (e.g. melting). The
// root is then on the transition: the bracket in T collapses onto it, and the phases
// coexist in the proportion (lever rule, linear in the specific volume and enthalpy)
// that gives F = 0.
// A cell too cold for the isobars at tfloor is clamped to the floor, as elsewhere.
template <typename EosIdx_t, typename LambdaIdx_t>
KOKKOS_INLINE_FUNCTION bool
SolveOpen(const int n, const EosIdx_t &eos, LambdaIdx_t &lambda, const Real *rhobar,
          Real *rho, const Real E, const Real p_void, const Real tfloor, Real &T,
          const Real tol, const int max_iter, int &niter, int &neos) {
  using RiotLimits::MAX_MATERIALS;
  std::array<Real, MAX_MATERIALS> drhodT, rho_lo, rho_hi;
  const Real lfloor = std::log(tfloor);
  Real lT = (T > tfloor && std::isfinite(T)) ? std::log(T) : lfloor;
  Real llo = lfloor, lhi = 0.0, F_lo = 0.0, F_hi = 0.0;
  Real dlT_old = std::numeric_limits<Real>::infinity();
  bool have_lo = false, have_hi = false;

  for (niter = 1; niter <= max_iter; niter++) {
    const Real Ti = std::exp(lT);

    // F(T) and dF/dT, with the isobar densities at T
    Real F = -(E + p_void), dFdT = 0.0, Fscale = std::abs(E + p_void);
    for (int m = 0; m < n; m++) {
      Real dpdr = 0.0;
      if (!IsobarDensity(eos[m], lambda[m], p_void, Ti, rho[m], tol, max_iter, rho[m],
                         dpdr, neos)) {
        return false;
      }
      const Real r = rho[m], hT = fd_h * Ti;
      const Real e0 = eos[m].InternalEnergyFromDensityTemperature(r, Ti, lambda[m]);
      const Real er =
          eos[m].InternalEnergyFromDensityTemperature(r * (1.0 + fd_h), Ti, lambda[m]);
      const Real eT = eos[m].InternalEnergyFromDensityTemperature(r, Ti + hT, lambda[m]);
      const Real p0 = eos[m].PressureFromDensityTemperature(r, Ti, lambda[m]);
      const Real pT = eos[m].PressureFromDensityTemperature(r, Ti + hT, lambda[m]);
      neos += 4;
      drhodT[m] = (dpdr > 0.0) ? -((pT - p0) / hT) / dpdr : 0.0;
      const Real h0 = e0 + p_void / r;
      F += rhobar[m] * h0;
      Fscale += std::abs(rhobar[m] * h0);
      dFdT += rhobar[m] *
              ((eT - e0) / hT + ((er - e0) / (r * fd_h) - p_void / (r * r)) * drhodT[m]);
    }
    if (!std::isfinite(F)) return false;

    // Converged, or clamped to the floor
    if (std::abs(F) <= tol * Fscale || (F > 0.0 && lT <= lfloor)) break;

    // Update the bracket in log(T)
    if (F < 0.0) {
      llo = lT;
      F_lo = F;
      have_lo = true;
      for (int m = 0; m < n; m++) {
        rho_lo[m] = rho[m];
      }
    } else {
      lhi = lT;
      F_hi = F;
      have_hi = true;
      for (int m = 0; m < n; m++) {
        rho_hi[m] = rho[m];
      }
    }

    // Collapsed onto a phase transition: lever rule between the two sides
    if (have_lo && have_hi && lhi - llo <= tol) {
      const Real theta = F_lo / (F_lo - F_hi);
      for (int m = 0; m < n; m++) {
        rho[m] = 1.0 / ((1.0 - theta) / rho_lo[m] + theta / rho_hi[m]);
      }
      lT = (1.0 - theta) * llo + theta * lhi;
      break;
    }

    // Newton step in log(T), of at least tol (the isobars are steep at a phase
    // transition), safeguarded by the bracket (bisection, also if the step does not
    // halve, e.g. across a phase transition), else expand by up to 2x
    Real step = (dFdT > 0.0) ? -F / (Ti * dFdT) : 0.0;
    if (step != 0.0 && std::abs(step) < tol) step = std::copysign(tol, step);
    Real lT_new = lT + step;
    const bool bracketed = have_lo && have_hi;
    if (!(step != 0.0 && std::isfinite(step) && (!have_lo || lT_new > llo) &&
          (!have_hi || lT_new < lhi)) ||
        (bracketed && std::abs(step) > 0.5 * dlT_old)) {
      if (bracketed) {
        lT_new = 0.5 * (llo + lhi);
      } else {
        lT_new = (have_lo) ? lT + std::log(2.0) : lT - std::log(2.0);
      }
    }
    if (!have_hi) lT_new = std::min(lT_new, lT + std::log(2.0));
    if (!have_lo) lT_new = std::max(lT_new, lT - std::log(2.0));
    lT_new = std::max(lT_new, lfloor);
    if (bracketed) dlT_old = std::abs(lT_new - lT);

    // Warm start the isobar densities at the new temperature
    const Real dT = std::exp(lT_new) - Ti;
    for (int m = 0; m < n; m++) {
      rho[m] = std::min(std::max(rho[m] + drhodT[m] * dT, 0.5 * rho[m]), 2.0 * rho[m]);
    }
    lT = lT_new;
  }
  if (niter > max_iter) return false;

  T = std::exp(lT);
  Real alpha_sum = 0.0;
  for (int m = 0; m < n; m++) {
    alpha_sum += rhobar[m] / rho[m];
  }
  return alpha_sum < 1.0;
}

// Isobar densities of n materials at a fixed temperature T. Returns false if the void is
// closed at T (or a solve failed).
template <typename EosIdx_t, typename LambdaIdx_t>
KOKKOS_INLINE_FUNCTION bool
SolveOpenFixedT(const int n, const EosIdx_t &eos, LambdaIdx_t &lambda, const Real *rhobar,
                Real *rho, const Real p_void, const Real T, const Real tol,
                const int max_iter, int &neos) {
  Real alpha_sum = 0.0;
  for (int m = 0; m < n; m++) {
    Real dpdr;
    if (!IsobarDensity(eos[m], lambda[m], p_void, T, rho[m], tol, max_iter, rho[m], dpdr,
                       neos)) {
      return false;
    }
    alpha_sum += rhobar[m] / rho[m];
  }
  return alpha_sum < 1.0;
}

} // namespace Void

#endif // VOID_VOID_CLOSURE_HPP_
