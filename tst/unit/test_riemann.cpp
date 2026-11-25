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

//----------------------------------------------------------------------------------------
// HLLD (ideal MHD, Heaviside-Lorentz units with magnetic pressure B^2/2).
//
// For a state with field B and normal component bn = B_DIR, the exact ideal MHD flux is
//   f_rho = rho vn
//   f_vi  = rho vn vi + (P + B^2/2) delta_i,DIR - bn Bi
//   f_eng = vn (E + P + B^2/2) - bn (v.B),   E = u + 1/2 rho |v|^2 + 1/2 B^2
// and the face EMFs returned by the solver are components of E = -v x B:
//   emf0 = E_tb and emf1 = E_ta, with ta = (DIR) % 3 and tb = (DIR + 1) % 3 (0-based).
// vrho = 2 f_rho / (rhol + rhor) is the mass-flux proxy used for CT upwinding.
namespace {

struct MHDState {
  Real rho, v1, v2, v3, u, P, c, b1, b2, b3;
  State Hydro() const { return State{rho, v1, v2, v3, u, P, c}; }
  Real V(const int d) const { return (d == 0) ? v1 : (d == 1 ? v2 : v3); }
  Real B(const int d) const { return (d == 0) ? b1 : (d == 1 ? b2 : b3); }
  Real B2() const { return b1 * b1 + b2 * b2 + b3 * b3; }
};

struct MHDFlux {
  Real f_v1, f_v2, f_v3, f_eng;
  Real v1face, v2face, v3face, riemann_vel;
  Real emf0, emf1, vrho;
  Real smax;
};

template <class Fill>
MHDFlux RunOnDeviceMHD(Fill fill) {
  Kokkos::View<MHDFlux> d_flux("hlld_flux");
  Kokkos::parallel_for(
      "run hlld solver", 1, KOKKOS_LAMBDA(const int) {
        MHDFlux f{};
        fill(f);
        d_flux() = f;
      });
  auto h_flux = Kokkos::create_mirror_view(d_flux);
  Kokkos::deep_copy(h_flux, d_flux);
  return h_flux();
}

// The normal field bn is passed separately (as in the flux kernel, where it comes from
// the face-centered field); by default it is taken from the left state.
template <int DIR>
MHDFlux RunHLLD(const MHDState &l, const MHDState &r, const Real bn) {
  return RunOnDeviceMHD(KOKKOS_LAMBDA(MHDFlux & f) {
    f.smax = Hydro::lr_to_flux_hlld<DIR>(
        l.rho, r.rho, l.v1, r.v1, l.v2, r.v2, l.v3, r.v3, l.u, r.u, l.P, r.P, l.c, r.c,
        bn, l.b1, r.b1, l.b2, r.b2, l.b3, r.b3, f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.v1face,
        f.v2face, f.v3face, f.riemann_vel, f.emf0, f.emf1, f.vrho);
  });
}
template <int DIR>
MHDFlux RunHLLD(const MHDState &l, const MHDState &r) {
  return RunHLLD<DIR>(l, r, l.B(DIR - 1));
}

template <int DIR>
MHDFlux PhysicalMHDFlux(const MHDState &s) {
  constexpr int n = DIR - 1;
  constexpr int ta = (n + 1) % 3;
  constexpr int tb = (n + 2) % 3;
  const Real vn = s.V(n);
  const Real bn = s.B(n);
  const Real pb = 0.5 * s.B2();
  const Real E = s.u + 0.5 * s.rho * (s.v1 * s.v1 + s.v2 * s.v2 + s.v3 * s.v3) + pb;
  const Real vdotb = s.v1 * s.b1 + s.v2 * s.b2 + s.v3 * s.b3;
  // E = -v x B
  auto efield = [&](const int d) {
    const int p = (d + 1) % 3, q = (d + 2) % 3;
    return -(s.V(p) * s.B(q) - s.V(q) * s.B(p));
  };
  MHDFlux f{};
  f.f_v1 = s.rho * vn * s.v1 + (n == 0) * (s.P + pb) - bn * s.b1;
  f.f_v2 = s.rho * vn * s.v2 + (n == 1) * (s.P + pb) - bn * s.b2;
  f.f_v3 = s.rho * vn * s.v3 + (n == 2) * (s.P + pb) - bn * s.b3;
  f.f_eng = vn * (E + s.P + pb) - bn * vdotb;
  f.riemann_vel = vn;
  f.emf0 = efield(tb);
  f.emf1 = efield(ta);
  f.vrho = vn;
  return f;
}

// Bulk sound speed c, fast magnetosonic speed in direction n
Real FastSpeed(const MHDState &s, const int n) {
  const Real asq = s.c * s.c;
  const Real bsq = s.B2() / s.rho;
  const Real bnsq = s.B(n) * s.B(n) / s.rho;
  return std::sqrt(0.5 *
                   (asq + bsq + std::sqrt((asq + bsq) * (asq + bsq) - 4.0 * asq * bnsq)));
}

void CheckMHDFlux(const MHDFlux &got, const MHDFlux &expected, const Real tol = 1e-10) {
  CHECK(got.f_v1 == Approx(expected.f_v1).margin(tol));
  CHECK(got.f_v2 == Approx(expected.f_v2).margin(tol));
  CHECK(got.f_v3 == Approx(expected.f_v3).margin(tol));
  CHECK(got.f_eng == Approx(expected.f_eng).margin(tol));
  CHECK(got.riemann_vel == Approx(expected.riemann_vel).margin(tol));
  CHECK(got.emf0 == Approx(expected.emf0).margin(tol));
  CHECK(got.emf1 == Approx(expected.emf1).margin(tol));
  CHECK(got.vrho == Approx(expected.vrho).margin(tol));
}

// Moving, off-axis magnetized state with every field and velocity component nonzero.
constexpr MHDState kMHDState{/*rho*/ 2.0, /*v1*/ 0.3, /*v2*/ -0.4, /*v3*/ 0.1,
                             /*u*/ 2.5,   /*P*/ 1.5,  /*c*/ 1.2,   /*b1*/ 0.7,
                             /*b2*/ -0.5, /*b3*/ 0.9};

// Builds a state from components given in the (normal, ta, tb) frame of direction DIR.
template <int DIR>
MHDState Rotated(const Real rho, const Real vn, const Real va, const Real vb,
                 const Real u, const Real P, const Real c, const Real bn, const Real ba,
                 const Real bb) {
  constexpr int n = DIR - 1;
  Real v[3], b[3];
  v[n] = vn, v[(n + 1) % 3] = va, v[(n + 2) % 3] = vb;
  b[n] = bn, b[(n + 1) % 3] = ba, b[(n + 2) % 3] = bb;
  return MHDState{rho, v[0], v[1], v[2], u, P, c, b[0], b[1], b[2]};
}

} // namespace

template <int DIR>
void CheckHLLDConsistency() {
  const MHDState s = kMHDState;
  CheckMHDFlux(RunHLLD<DIR>(s, s), PhysicalMHDFlux<DIR>(s));
  // Super-fast flow selects F_L / F_R directly
  MHDState fast = s;
  fast.v1 = fast.v2 = fast.v3 = 5.0;
  CheckMHDFlux(RunHLLD<DIR>(fast, fast), PhysicalMHDFlux<DIR>(fast));
  fast.v1 = fast.v2 = fast.v3 = -5.0;
  CheckMHDFlux(RunHLLD<DIR>(fast, fast), PhysicalMHDFlux<DIR>(fast));
  // Weak normal field (degenerate star states)
  MHDState weak = s;
  weak.b1 = weak.b2 = weak.b3 = 1.0e-12;
  CheckMHDFlux(RunHLLD<DIR>(weak, weak), PhysicalMHDFlux<DIR>(weak));
}

TEST_CASE("HLLD is consistent for identical L/R states", "[hydro][riemann][hlld]") {
  SECTION("X1") { CheckHLLDConsistency<X1DIR>(); }
  SECTION("X2") { CheckHLLDConsistency<X2DIR>(); }
  SECTION("X3") { CheckHLLDConsistency<X3DIR>(); }
}

template <int DIR>
void CheckHLLDSignalSpeed() {
  const MHDState s = kMHDState;
  const Real expected = std::abs(s.V(DIR - 1)) + FastSpeed(s, DIR - 1);
  CHECK(RunHLLD<DIR>(s, s).smax == Approx(expected));
  // For a jump, the returned speed bounds both states' fast signal speeds
  MHDState r = s;
  r.rho = 0.5, r.P = 0.3, r.c = 0.9;
  // Jump in a transverse field component only (bn is continuous)
  constexpr int ta = DIR % 3;
  if (ta == 0) r.b1 = 1.4;
  if (ta == 1) r.b2 = 1.4;
  if (ta == 2) r.b3 = 1.4;
  const Real smax = RunHLLD<DIR>(s, r, s.B(DIR - 1)).smax;
  CHECK(smax >= std::abs(s.V(DIR - 1)) + FastSpeed(s, DIR - 1) - 1e-12);
  CHECK(smax >= std::abs(r.V(DIR - 1)) + FastSpeed(r, DIR - 1) - 1e-12);
}

TEST_CASE("HLLD returns the fast magnetosonic signal speed",
          "[hydro][riemann][hlld][signal]") {
  SECTION("X1") { CheckHLLDSignalSpeed<X1DIR>(); }
  SECTION("X2") { CheckHLLDSignalSpeed<X2DIR>(); }
  SECTION("X3") { CheckHLLDSignalSpeed<X3DIR>(); }
}

// With no field HLLD reduces to HLLC (same wave speed estimates and contact speed).
template <int DIR>
void CheckHLLDReducesToHLLC() {
  const MHDState l = Rotated<DIR>(1.0, 0.2, 0.3, -0.1, 2.5, 1.0, 1.3, 0, 0, 0);
  const MHDState r = Rotated<DIR>(0.125, -0.1, -0.2, 0.4, 0.25, 0.1, 1.1, 0, 0, 0);
  const MHDFlux d = RunHLLD<DIR>(l, r);
  const Flux c = RunHLLC<DIR>(l.Hydro(), r.Hydro());
  CHECK(d.f_v1 == Approx(c.f_v1).margin(1e-12));
  CHECK(d.f_v2 == Approx(c.f_v2).margin(1e-12));
  CHECK(d.f_v3 == Approx(c.f_v3).margin(1e-12));
  CHECK(d.f_eng == Approx(c.f_eng).margin(1e-12));
  CHECK(d.v1face == Approx(c.v1face).margin(1e-12));
  CHECK(d.v2face == Approx(c.v2face).margin(1e-12));
  CHECK(d.v3face == Approx(c.v3face).margin(1e-12));
  CHECK(d.riemann_vel == Approx(c.riemann_vel).margin(1e-12));
  CHECK(d.smax == Approx(c.smax).margin(1e-12));
  CHECK(d.emf0 == Approx(0.0).margin(1e-14));
  CHECK(d.emf1 == Approx(0.0).margin(1e-14));
}

TEST_CASE("HLLD reduces to HLLC with zero field", "[hydro][riemann][hlld]") {
  SECTION("X1") { CheckHLLDReducesToHLLC<X1DIR>(); }
  SECTION("X2") { CheckHLLDReducesToHLLC<X2DIR>(); }
  SECTION("X3") { CheckHLLDReducesToHLLC<X3DIR>(); }
}

// Stationary contact (bn != 0, equal P and B, fluid at rest, density jump) and tangential
// discontinuity (bn = 0, total pressure balance, jumps in tangential v and B) are both
// exact stationary solutions that HLLD resolves exactly.
template <int DIR>
void CheckHLLDStationaryDiscontinuities() {
  constexpr int n = DIR - 1;
  {
    const MHDState l = Rotated<DIR>(3.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.8, 0.6, -0.3);
    const MHDState r = Rotated<DIR>(1.0, 0.0, 0.0, 0.0, 0.7, 1.0, 1.4, 0.8, 0.6, -0.3);
    const MHDFlux f = RunHLLD<DIR>(l, r);
    const Real fn = (n == 0) ? f.f_v1 : (n == 1 ? f.f_v2 : f.f_v3);
    CHECK(f.riemann_vel == Approx(0.0).margin(1e-12));
    CHECK(fn == Approx(l.P + 0.5 * l.B2() - 0.8 * 0.8).margin(1e-12));
    CHECK(f.f_eng == Approx(0.0).margin(1e-12));
    CHECK(f.emf0 == Approx(0.0).margin(1e-12));
    CHECK(f.emf1 == Approx(0.0).margin(1e-12));
  }
  {
    // ptot = 1 + 0.5 = 1.25 + 0.25 = 1.5
    const MHDState l = Rotated<DIR>(1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 1.2, 0.0, 1.0, 0.0);
    const MHDState r = Rotated<DIR>(0.4, 0.0, -0.3, 0.7, 1.9, 1.25, 1.5, 0.0, 0.5, 0.5);
    const MHDFlux f = RunHLLD<DIR>(l, r);
    const Real fn = (n == 0) ? f.f_v1 : (n == 1 ? f.f_v2 : f.f_v3);
    const Real fa = (n == 0) ? f.f_v2 : (n == 1 ? f.f_v3 : f.f_v1);
    const Real fb = (n == 0) ? f.f_v3 : (n == 1 ? f.f_v1 : f.f_v2);
    CHECK(f.riemann_vel == Approx(0.0).margin(1e-12));
    CHECK(fn == Approx(1.5).margin(1e-12));
    CHECK(fa == Approx(0.0).margin(1e-12));
    CHECK(fb == Approx(0.0).margin(1e-12));
    CHECK(f.f_eng == Approx(0.0).margin(1e-12));
    CHECK(f.emf0 == Approx(0.0).margin(1e-12));
    CHECK(f.emf1 == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("HLLD resolves stationary contact and tangential discontinuities",
          "[hydro][riemann][hlld][contact]") {
  SECTION("X1") { CheckHLLDStationaryDiscontinuities<X1DIR>(); }
  SECTION("X2") { CheckHLLDStationaryDiscontinuities<X2DIR>(); }
  SECTION("X3") { CheckHLLDStationaryDiscontinuities<X3DIR>(); }
}

// Brio-Wu-like jump: fluxes are finite and continuous as bn -> 0.
template <int DIR>
void CheckHLLDSmallNormalField() {
  const MHDState l = Rotated<DIR>(1.0, 0.1, 0.0, 0.0, 1.0, 1.0, 1.4, 0.0, 1.0, 0.2);
  const MHDState r = Rotated<DIR>(0.125, 0.1, 0.0, 0.0, 0.1, 0.1, 1.2, 0.0, -1.0, 0.2);
  const MHDFlux f0 = RunHLLD<DIR>(l, r, 0.0);
  for (const Real bn : {1.0e-20, 1.0e-14, 1.0e-10}) {
    const MHDFlux f = RunHLLD<DIR>(l, r, bn);
    for (const Real x : {f.f_v1, f.f_v2, f.f_v3, f.f_eng, f.riemann_vel, f.emf0, f.emf1,
                         f.vrho, f.smax}) {
      CHECK(std::isfinite(x));
    }
    CheckMHDFlux(f, f0, 1e-8);
  }
}

TEST_CASE("HLLD is finite and continuous as the normal field vanishes",
          "[hydro][riemann][hlld]") {
  SECTION("X1") { CheckHLLDSmallNormalField<X1DIR>(); }
  SECTION("X2") { CheckHLLDSmallNormalField<X2DIR>(); }
  SECTION("X3") { CheckHLLDSmallNormalField<X3DIR>(); }
}

// Mirror symmetry: reflecting x_n -> -x_n swaps L and R and flips the normal components
// of v and B. The mass, tangential momentum, energy and tangential field fluxes (hence
// the EMFs) change sign; the normal momentum flux is unchanged.
template <int DIR>
void CheckHLLDMirrorSymmetry() {
  constexpr int n = DIR - 1;
  const MHDState l =
      Rotated<DIR>(1.08, 1.2, 0.01, 0.5, 2.38, 0.95, 1.2, 0.56, 1.02, 0.56);
  const MHDState r = Rotated<DIR>(1.0, 0.0, 0.0, 0.0, 1.5, 1.0, 1.3, 0.56, 1.13, 0.56);
  auto mirror = [](MHDState s) {
    if (n == 0) s.v1 = -s.v1, s.b1 = -s.b1;
    if (n == 1) s.v2 = -s.v2, s.b2 = -s.b2;
    if (n == 2) s.v3 = -s.v3, s.b3 = -s.b3;
    return s;
  };
  const MHDFlux f = RunHLLD<DIR>(l, r);
  const MHDFlux m = RunHLLD<DIR>(mirror(r), mirror(l));
  const Real fv[3] = {f.f_v1, f.f_v2, f.f_v3};
  const Real mv[3] = {m.f_v1, m.f_v2, m.f_v3};
  for (int d = 0; d < 3; ++d) {
    const Real sign = (d == n) ? 1.0 : -1.0;
    CHECK(mv[d] == Approx(sign * fv[d]).margin(1e-12));
  }
  CHECK(m.f_eng == Approx(-f.f_eng).margin(1e-12));
  CHECK(m.vrho == Approx(-f.vrho).margin(1e-12));
  CHECK(m.emf0 == Approx(-f.emf0).margin(1e-12));
  CHECK(m.emf1 == Approx(-f.emf1).margin(1e-12));
  CHECK(m.smax == Approx(f.smax).margin(1e-12));
}

TEST_CASE("HLLD is mirror symmetric", "[hydro][riemann][hlld][symmetry]") {
  SECTION("X1") { CheckHLLDMirrorSymmetry<X1DIR>(); }
  SECTION("X2") { CheckHLLDMirrorSymmetry<X2DIR>(); }
  SECTION("X3") { CheckHLLDMirrorSymmetry<X3DIR>(); }
}
