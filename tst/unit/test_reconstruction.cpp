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

// Unit tests for the slope limiters and reconstruction stencils in
// src/reconstruction/reconstruction.hpp.
//
// As with the Riemann solvers there is no closed-form reference for a general
// reconstruction, so the tests rely on properties every consistent limiter /
// reconstruction must satisfy, plus a few pinned golden values:
//
//   * CONSTANT PRESERVATION: a constant stencil must reconstruct to that
//     constant on both cell faces (no scheme may manufacture a slope).
//   * LINEAR EXACTNESS: on a uniform linear ramp q_i = i, every scheme here
//     (PLM and the high-order PPM4/WENO5/MP5) reproduces the linear values at
//     the faces, i.e. the face outputs are q0 +/- 1/2.
//   * SYMMETRY of the centered piecewise-linear reconstruction: l + r == 2*q0.
//   * EXTREMUM CLIPPING: at a local extremum the limited slope
//     collapses to zero, so the reconstruction flattens to the cell
//     average. (For those methods that are extremum clipping.)
//
// The functions are KOKKOS_*_FUNCTION and take/return plain Real, so (matching
// the Riemann tests) we exercise them ON DEVICE: each helper runs the routine
// inside a single-iteration Kokkos loop and copies the POD result back to the
// host, where the invariants are asserted with Catch2.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

// Bring Real (and the prelude generally) into scope before the header, matching
// how reconstruction.hpp is used in the code proper.
using namespace parthenon::package::prelude;

#include "reconstruction/reconstruction.hpp"

using Catch::Approx;
using parthenon::Real;

namespace {

namespace Recon = RiotReconstruction;

// The two reconstructed values produced from a single cell i. `plus` is the
// first output argument of the routine, `minus` the second; this ordering is
// consistent across all the stencils. On a rising ramp plus == q0+dq (the
// cell's +/right edge) and minus == q0-dq (the -/left edge).
//
// NOTE on the convention: these two outputs do NOT feed the two sides of one
// face. They belong to *different* faces, which is what lets a single sweep
// over cell centers fill both sides of every face. Concretely, in the flux
// loop (calculate_fluxes.cpp, BulkRiemannFluxes) the face between cells i and
// i+1 takes its left state from plus(i) and its right state from minus(i+1).
// The tests below only inspect the per-cell numeric outputs (plus == q0+dq,
// minus == q0-dq, symmetry, clipping), so they are independent of this
// facewise bookkeeping.
struct Faces {
  Real plus, minus;
};

// Run `fill` inside a single-iteration Kokkos loop and return the POD it fills.
// `fill` is a device-callable receiving a fresh T& to populate. Using a
// function template (rather than a macro) keeps normal compiler diagnostics.
template <class T, class Fill>
T RunOnDevice(Fill fill) {
  Kokkos::View<T> d_out("recon_result");
  Kokkos::parallel_for(
      "run reconstruction", 1, KOKKOS_LAMBDA(const int) {
        T out{};
        fill(out);
        d_out() = out;
      });
  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);
  return h_out();
}

// Convenience wrappers: evaluate a scalar limiter / a face-pair reconstruction
// on device. (Named Run* to avoid colliding with parthenon's Scalar concept.)
template <class Fill>
Real RunScalar(Fill fill) {
  return RunOnDevice<Real>(fill);
}
template <class Fill>
Faces RunFaces(Fill fill) {
  return RunOnDevice<Faces>(fill);
}

constexpr Real kTol = 1e-12;

} // namespace

//----------------------------------------------------------------------------------------
// Limiters
//----------------------------------------------------------------------------------------

TEST_CASE("minmod selects the smaller-magnitude argument, zero on sign change",
          "[reconstruction][limiter][minmod]") {
  // Same sign: returns the smaller magnitude, keeping the common sign.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::minmod(2.0, 3.0); }) ==
        Approx(2.0));
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::minmod(-2.0, -3.0); }) ==
        Approx(-2.0));
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::minmod(3.0, 3.0); }) ==
        Approx(3.0));
  // Opposite signs (or a zero argument) clip to zero.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::minmod(2.0, -3.0); }) ==
        Approx(0.0).margin(kTol));
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::minmod(0.0, 5.0); }) ==
        Approx(0.0).margin(kTol));
}

TEST_CASE("mc (monotonized-central) limits the slope to alpha*min(|dm|,|dp|)",
          "[reconstruction][limiter][mc]") {
  // Balanced slopes: the central difference wins and is passed through.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::mc(1.0, 1.0, 2.0); }) ==
        Approx(1.0));
  // Central difference equals the alpha bound exactly.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::mc(1.0, 3.0, 2.0); }) ==
        Approx(2.0));
  // Steep one-sided slope is clipped to alpha*min(|dm|,|dp|) = 2*1 = 2.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::mc(1.0, 10.0, 2.0); }) ==
        Approx(2.0));
  // Sign change across the cell => zero slope (monotonicity).
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::mc(1.0, -1.0, 2.0); }) ==
        Approx(0.0).margin(kTol));
  // Negative-going data keeps the sign of the central difference.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) { x = Recon::mc(-2.0, -4.0, 2.0); }) ==
        Approx(-3.0));
}

TEST_CASE("ComputeMCSlope reproduces a linear slope and clips at extrema",
          "[reconstruction][limiter][mcslope]") {
  // Uniform linear ramp u = {0,1,2}, dx = 1 on both sides: slope == 1.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) {
          x = Recon::ComputeMCSlope(0.0, 1.0, 2.0, 1.0, 1.0);
        }) == Approx(1.0));
  // Local maximum u = {0,1,0}: forward and backward slopes disagree in sign, so
  // the limited slope is exactly zero.
  CHECK(RunScalar(KOKKOS_LAMBDA(Real & x) {
          x = Recon::ComputeMCSlope(0.0, 1.0, 0.0, 1.0, 1.0);
        }) == Approx(0.0).margin(kTol));
}

//----------------------------------------------------------------------------------------
// Reconstruction stencils
//----------------------------------------------------------------------------------------

TEST_CASE("PiecewiseConstant copies the cell average to both faces",
          "[reconstruction][constant]") {
  const Real q0 = 3.5;
  const Faces f = RunFaces(
      KOKKOS_LAMBDA(Faces & out) { Recon::PiecewiseConstant(q0, out.plus, out.minus); });
  CHECK(f.plus == Approx(q0));
  CHECK(f.minus == Approx(q0));
}

TEST_CASE("PiecewiseLinear is centered and second-order accurate",
          "[reconstruction][plm]") {
  SECTION("linear ramp: faces are q0 +/- half the (uniform) slope") {
    // qm,q0,qp = 1,2,3 -> dq = 1/2 mc(1,1,1.99) = 1/2, so l = 2.5, r = 1.5.
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::PiecewiseLinear(1.0, 2.0, 3.0, out.plus, out.minus);
    });
    CHECK(f.plus == Approx(2.5));
    CHECK(f.minus == Approx(1.5));
    // Symmetry about the cell average holds for any (non-extremal) stencil.
    CHECK(f.plus + f.minus == Approx(4.0)); // == 2*q0
  }

  SECTION("symmetry l + r == 2*q0 for a generic monotone stencil") {
    const Real q0 = -0.75;
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::PiecewiseLinear(-2.0, q0, 1.5, out.plus, out.minus);
    });
    CHECK(f.plus + f.minus == Approx(2.0 * q0));
  }

  SECTION("local extremum is flattened (slope clipped to zero)") {
    // Symmetric peak qm == qp < q0: the limiter zeroes the slope, so both faces
    // collapse to q0.
    const Real q0 = 1.0;
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::PiecewiseLinear(0.0, q0, 0.0, out.plus, out.minus);
    });
    CHECK(f.plus == Approx(q0));
    CHECK(f.minus == Approx(q0));
  }

  SECTION("PiecewiseLinearFixedSlope matches PiecewiseLinear with slope 1.99") {
    const Faces fixed = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::PiecewiseLinearFixedSlope(1.0, 2.0, 3.0, out.plus, out.minus);
    });
    CHECK(fixed.plus == Approx(2.5));
    CHECK(fixed.minus == Approx(1.5));
  }
}

TEST_CASE("PPM4 preserves constants and is exact on linear data",
          "[reconstruction][ppm4]") {
  SECTION("constant stencil") {
    const Real c = 2.0;
    const Faces f = RunFaces(
        KOKKOS_LAMBDA(Faces & out) { Recon::PPM4(c, c, c, c, c, out.plus, out.minus); });
    CHECK(f.plus == Approx(c));
    CHECK(f.minus == Approx(c));
  }
  SECTION("linear ramp {0,1,2,3,4}, center q=2") {
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::PPM4(0.0, 1.0, 2.0, 3.0, 4.0, out.plus, out.minus);
    });
    CHECK(f.plus == Approx(2.5));  // ql_ip1 (+/right edge, q0+dq)
    CHECK(f.minus == Approx(1.5)); // qr_i   (-/left edge,  q0-dq)
  }
}

TEST_CASE("WENO5 preserves constants and is exact on linear data",
          "[reconstruction][weno5]") {
  SECTION("constant stencil") {
    const Real c = -4.2;
    const Faces f = RunFaces(
        KOKKOS_LAMBDA(Faces & out) { Recon::WENO5(c, c, c, c, c, out.plus, out.minus); });
    CHECK(f.plus == Approx(c));
    CHECK(f.minus == Approx(c));
  }
  SECTION("linear ramp {0,1,2,3,4}, center q=2") {
    // For linear data all smoothness indicators are equal, so WENO5 reduces to
    // the optimal (linear-exact) fifth-order reconstruction: q0 +/- 1/2.
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::WENO5(0.0, 1.0, 2.0, 3.0, 4.0, out.plus, out.minus);
    });
    CHECK(f.plus == Approx(2.5).margin(1e-10));
    CHECK(f.minus == Approx(1.5).margin(1e-10));
  }
}

TEST_CASE("MP5 preserves constants and is exact on linear data",
          "[reconstruction][mp5]") {
  SECTION("constant stencil") {
    const Real c = 7.0;
    const Faces f = RunFaces(
        KOKKOS_LAMBDA(Faces & out) { Recon::MP5(c, c, c, c, c, out.plus, out.minus); });
    CHECK(f.plus == Approx(c));
    CHECK(f.minus == Approx(c));
  }
  SECTION("linear ramp {0,1,2,3,4}, center q=2") {
    const Faces f = RunFaces(KOKKOS_LAMBDA(Faces & out) {
      Recon::MP5(0.0, 1.0, 2.0, 3.0, 4.0, out.plus, out.minus);
    });
    CHECK(f.plus == Approx(2.5).margin(1e-10));
    CHECK(f.minus == Approx(1.5).margin(1e-10));
  }
}
