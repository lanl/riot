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

// Unit tests for the Riemann solvers in src/hydro/riemann.hpp.
//
// There is no closed-form reference solution for a general Riemann problem, so
// the tests here rely on properties that ANY consistent finite-volume flux
// function must satisfy, plus a couple of pinned golden values.
//
// The workhorse invariant is CONSISTENCY: when the left and right states are
// identical, the numerical flux must reduce to the exact physical (Euler) flux
// of that single state. For a state (rho, v=(v1,v2,v3), P) with conserved
// internal energy u and total energy E = u + 1/2 rho |v|^2, the flux in the
// direction DIR (normal velocity vn) is
//   f_rho   = rho * vn                      (mass)
//   f_vi    = rho * vn * vi + P * delta_i,DIR   (momentum)
//   f_eng   = vn * (E + P)                  (energy)
// and the "riemann_vel" bulk advection velocity must equal vn.
//
// The solvers are KOKKOS_FORCEINLINE_FUNCTION, so we exercise them ON DEVICE:
// each solver wrapper runs the flux computation inside a single-iteration
// Kokkos loop and copies the resulting Flux back to the host, where the
// invariants are asserted with Catch2. They are templated on the coordinate
// direction, so each invariant is checked in X1/X2/X3.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

#include "hydro/riemann.hpp"

using Catch::Approx;
using parthenon::Real;

namespace {

// A primitive+conserved fluid state used to build symmetric Riemann problems.
struct State {
  Real rho, v1, v2, v3, u, P, c;
};
template <int DIR>
Real NormalVel(const State &s) {
  if constexpr (DIR == X1DIR) return s.v1;
  if constexpr (DIR == X2DIR) return s.v2;
  if constexpr (DIR == X3DIR) return s.v3;
}
Real TotalEnergy(const State &s) {
  return s.u + 0.5 * s.rho * (s.v1 * s.v1 + s.v2 * s.v2 + s.v3 * s.v3);
}

// Bundle of solver outputs so the different call signatures can be compared
// against a single expected-physical-flux computation.
// also convenient for copying host/device
struct Flux {
  Real f_v1, f_v2, f_v3, f_eng;
  Real v1face, v2face, v3face, riemann_vel;
  Real smax; // returned max signal speed
};

// Exact physical Euler flux of a single state in direction DIR.
template <int DIR>
Flux PhysicalFlux(const State &s) {
  const Real vn = NormalVel<DIR>(s);
  const Real E = TotalEnergy(s);
  Flux f{};
  f.f_v1 = s.rho * vn * s.v1 + (DIR == X1DIR) * s.P;
  f.f_v2 = s.rho * vn * s.v2 + (DIR == X2DIR) * s.P;
  f.f_v3 = s.rho * vn * s.v3 + (DIR == X3DIR) * s.P;
  f.f_eng = vn * (E + s.P);
  f.riemann_vel = vn;
  return f;
}

// Assert that a solver's momentum/energy fluxes and advection velocity match
// the exact physical flux of the (identical) state.
void CheckConsistent(const Flux &got, const Flux &expected) {
  CHECK(got.f_v1 == Approx(expected.f_v1).margin(1e-10));
  CHECK(got.f_v2 == Approx(expected.f_v2).margin(1e-10));
  CHECK(got.f_v3 == Approx(expected.f_v3).margin(1e-10));
  CHECK(got.f_eng == Approx(expected.f_eng).margin(1e-10));
  CHECK(got.riemann_vel == Approx(expected.riemann_vel).margin(1e-10));
}

// Each wrapper runs its solver on device via RunOnDevice, which executes the
// given device-callable inside a single-iteration Kokkos::parallel_for,
// storing the outputs into a scalar Kokkos::View, then copies the resulting
// Flux (a trivially-copyable POD) back to the host.
//
// The `fill` callable receives a fresh `Flux &`, computes the fluxes, and
// stores the max signal speed into f.smax.
template <class Fill>
Flux RunOnDevice(Fill fill) {
  Kokkos::View<Flux> d_flux("riemann_flux");
  Kokkos::parallel_for(
      "run riemann solver", 1, KOKKOS_LAMBDA(const int) {
        Flux f{};
        fill(f);
        d_flux() = f;
      });
  auto h_flux = Kokkos::create_mirror_view(d_flux);
  Kokkos::deep_copy(h_flux, d_flux);
  return h_flux();
}

template <int DIR>
Flux RunHLLC(const State &l, const State &r) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    f.smax = Hydro::lr_to_flux_hllc<DIR>(
        l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3, r.v3, l.u, r.u, l.P, r.P, l.c, r.c,
        f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face, f.v3face, f.riemann_vel);
  });
}

template <int DIR>
Flux RunHLL(const State &l, const State &r) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    f.smax = Hydro::lr_to_flux_hll<DIR>(
        l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3, r.v3, l.u, r.u, l.P, r.P, l.c, r.c,
        f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face, f.v3face, f.riemann_vel);
  });
}

template <int DIR>
Flux RunFleischmann(const State &l, const State &r) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    f.smax = Hydro::lr_to_flux_fleischmann<DIR>(
        l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3, r.v3, l.u, r.u, l.P, r.P, l.c, r.c,
        f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face, f.v3face, f.riemann_vel);
  });
}

// LHLLC / CHLLC take extra dvn/dvt shock-detector arguments (non-const in/out
// references), so they need mutable copies declared inside the device kernel.
template <int DIR>
Flux RunLHLLC(const State &l, const State &r, Real dvn = 0.0, Real dvt = 0.0) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    Real dvn_ = dvn, dvt_ = dvt;
    f.smax = Hydro::lr_to_flux_lhllc<DIR>(l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3,
                                          r.v3, l.u, r.u, l.P, r.P, l.c, r.c, f.f_v1,
                                          f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face,
                                          f.v3face, f.riemann_vel, dvn_, dvt_);
  });
}

template <int DIR>
Flux RunCHLLC(const State &l, const State &r, Real dvn = 0.0, Real dvt = 0.0) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    Real dvn_ = dvn, dvt_ = dvt;
    f.smax = Hydro::lr_to_flux_chllc<DIR>(l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3,
                                          r.v3, l.u, r.u, l.P, r.P, l.c, r.c, f.f_v1,
                                          f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face,
                                          f.v3face, f.riemann_vel, dvn_, dvt_);
  });
}

// Strength solver with all deviatoric stresses and shear moduli zeroed, so it
// should reduce to the plain hydro flux.
template <int DIR>
Flux RunStrengthHydro(const State &l, const State &r) {
  return RunOnDevice(KOKKOS_LAMBDA(Flux & f) {
    constexpr Real z = 0.0;
    f.smax = Hydro::lr_to_flux_strength<DIR>(
        l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3, r.v3, l.u, r.u, l.P, r.P, l.c, r.c,
        /*gmodl*/ z, /*gmodr*/ z, /*sxx*/ z, z, /*sxy*/ z, z, /*sxz*/ z, z, /*syy*/ z, z,
        /*syz*/ z, z, f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.v1face, f.v2face, f.v3face,
        f.riemann_vel);
  });
}

// A representative moving, off-axis fluid state (nonzero in all velocity
// components so tangential-momentum terms are exercised).
constexpr State kState{/*rho*/ 2.0, /*v1*/ 0.3, /*v2*/ -0.4, /*v3*/ 0.1,
                       /*u*/ 2.5,   /*P*/ 1.5,  /*c*/ 1.2};

} // namespace

// All solvers must be consistent in a single direction: identical L/R states
// yield the exact physical flux. Called once per direction below.
template <int DIR>
void CheckConsistencyAllSolvers() {
  const State s = kState;
  const Flux expected = PhysicalFlux<DIR>(s);
  CheckConsistent(RunHLLC<DIR>(s, s), expected);
  CheckConsistent(RunHLL<DIR>(s, s), expected);
  CheckConsistent(RunFleischmann<DIR>(s, s), expected);
  CheckConsistent(RunLHLLC<DIR>(s, s), expected);
  CheckConsistent(RunCHLLC<DIR>(s, s), expected);
  // Strength solver with zero shear modulus / stresses reduces to hydro.
  CheckConsistent(RunStrengthHydro<DIR>(s, s), expected);
}

// Consistency for every solver, in every direction. This is the core
// correctness check: any consistent flux equals the physical flux when there
// is no jump across the interface.
TEST_CASE("Riemann solvers are consistent for identical L/R states",
          "[hydro][riemann][consistency]") {
  SECTION("X1") { CheckConsistencyAllSolvers<X1DIR>(); }
  SECTION("X2") { CheckConsistencyAllSolvers<X2DIR>(); }
  SECTION("X3") { CheckConsistencyAllSolvers<X3DIR>(); }
}

// Returned max signal speed must equal |vn| + c for identical states (the
// solvers estimate sl,sr = vn -/+ c and return max(|sl|,|sr|)).
template <int DIR>
void CheckSignalSpeedAllSolvers() {
  const State s = kState;
  const Real expected_smax = std::abs(NormalVel<DIR>(s)) + s.c;
  CHECK(RunHLLC<DIR>(s, s).smax == Approx(expected_smax));
  CHECK(RunHLL<DIR>(s, s).smax == Approx(expected_smax));
  CHECK(RunFleischmann<DIR>(s, s).smax == Approx(expected_smax));
  CHECK(RunLHLLC<DIR>(s, s).smax == Approx(expected_smax));
  CHECK(RunCHLLC<DIR>(s, s).smax == Approx(expected_smax));
  CHECK(RunStrengthHydro<DIR>(s, s).smax == Approx(expected_smax));
}

TEST_CASE("Riemann solvers return a physical max signal speed",
          "[hydro][riemann][signal]") {
  SECTION("X1") { CheckSignalSpeedAllSolvers<X1DIR>(); }
  SECTION("X2") { CheckSignalSpeedAllSolvers<X2DIR>(); }
  SECTION("X3") { CheckSignalSpeedAllSolvers<X3DIR>(); }
}

// A stationary contact discontinuity: equal pressures, zero normal velocity on
// both sides, differing densities. The mass flux (and hence riemann_vel) must
// vanish, and the normal momentum flux must equal the (continuous) pressure.
TEST_CASE("HLLC resolves a stationary contact with zero mass flux",
          "[hydro][riemann][contact]") {
  // Normal is X1. Zero normal velocity; nonzero equal tangential velocity.
  State l{/*rho*/ 3.0, /*v1*/ 0.0, /*v2*/ 0.5, /*v3*/ 0.0, /*u*/ 2.0, /*P*/ 1.0,
          /*c*/ 1.0};
  State r = l;
  r.rho = 1.0; // density jump across the contact
  r.u = 0.7;

  const Flux f = RunHLLC<X1DIR>(l, r);
  CHECK(f.riemann_vel == Approx(0.0).margin(1e-10)); // no mass crosses the contact
  CHECK(f.f_v1 == Approx(l.P).margin(1e-10));        // normal momentum flux == P
  CHECK(f.f_eng == Approx(0.0).margin(1e-10));       // no energy advected
}

// A symmetric problem: mirror the normal velocity (L moving +x, R moving -x,
// everything else equal). By symmetry the interface state is at rest, so the
// contact speed / advection velocity must be exactly zero and the normal
// momentum flux must be symmetric (equal pressures give the same on both).
TEST_CASE("HLLC on a symmetric compression has zero contact velocity",
          "[hydro][riemann][symmetry]") {
  State l{/*rho*/ 1.0, /*v1*/ 0.6, /*v2*/ 0.0, /*v3*/ 0.0, /*u*/ 1.0, /*P*/ 1.0,
          /*c*/ 1.4};
  State r = l;
  r.v1 = -0.6; // mirror the normal velocity

  const Flux f = RunHLLC<X1DIR>(l, r);
  CHECK(f.riemann_vel == Approx(0.0).margin(1e-10));
  CHECK(f.v1face == Approx(0.0).margin(1e-10));
}

// Golden value: HLL and HLLC agree on the total mass flux for identical states
// (both must give rho*vn), pinning the numeric result.
TEST_CASE("HLL and HLLC agree on mass flux for a uniform flow",
          "[hydro][riemann][golden]") {
  const State s = kState; // vn == v1 == 0.3, rho == 2.0  ->  rho*vn == 0.6
  CHECK(RunHLLC<X1DIR>(s, s).riemann_vel == Approx(0.3));
  CHECK(RunHLL<X1DIR>(s, s).riemann_vel == Approx(0.3));
  // mass flux frho = rho * riemann_vel = 0.6 (golden)
  CHECK(s.rho * RunHLLC<X1DIR>(s, s).riemann_vel == Approx(0.6));
}
