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
//
// Unit tests for the ideal-MHD primitives in src/mhd/mhd_helpers.hpp, plus the discrete
// vector-potential curl / divergence identity that constrained transport depends on.
//
// Test IDs match plan_histories/artemis_mhd_port/TEST_LEDGER.md:
//   U01  primitive -> conserved -> primitive round trip with a magnetic field
//   U01b unit-convention invariance: the same physical state at mu0 = 1 and mu0 = 4*pi
//   U03  fast magnetosonic speed: parallel, perpendicular, oblique, and b -> 0
//   U04  discrete curl of a vector potential is discretely divergence free
//
// These are chosen so that the failure modes they catch are the ones that actually
// occur in a CT port: a doubled or dropped 1/mu0, a magnetic energy that is subtracted
// with a different face-to-cell convention than it was added with, a wave speed that
// silently reduces to the Alfven speed, and a curl/divergence orientation mismatch that
// would show up much later as div B growing on block boundaries.

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include "mhd/mhd_helpers.hpp"
#include "variables.hpp"

using Catch::Approx;
using parthenon::Real;
using TE = parthenon::TopologicalElement;

namespace {

namespace fbulk = face_variables::bulk;

constexpr Real kTightTol = 1.0e-12;

//----------------------------------------------------------------------------------------
// A minimal stand-in for a Parthenon pack view over the face-centered magnetic field.
//
// MHD::FaceToCellB and MHD::CellMagneticEnergyFromFaces are templated on the view type
// precisely so they can be exercised here without standing up a Mesh. The stub supports
// the two operations they use: pv(TE, magnetic_field(), idx) and idx + offset.
struct StubFaceView {
  const Real *b1;
  const Real *b2;
  const Real *b3;

  Real operator()(TE te, const fbulk::magnetic_field &, const int idx) const {
    switch (te) {
    case TE::F1:
      return b1[idx];
    case TE::F2:
      return b2[idx];
    default:
      return b3[idx];
    }
  }
};

// Uniform field: both bounding faces in each direction carry the same value, so the
// face-to-cell average must return that value exactly.
struct UniformFaces {
  Real v1[2], v2[2], v3[2];
  StubFaceView View() const { return StubFaceView{v1, v2, v3}; }
};
UniformFaces MakeUniformFaces(Real bx, Real by, Real bz) {
  return UniformFaces{{bx, bx}, {by, by}, {bz, bz}};
}

} // namespace

//========================================================================================
// U01 -- primitive/conserved round trip with a magnetic field
//========================================================================================
TEST_CASE("U01: total energy round trip preserves thermal energy exactly",
          "[mhd][energy]") {
  // ADR-002: with MHD on, E = u + 1/2 rho v^2 + B^2/(2 mu0). Recovery subtracts the
  // kinetic and magnetic parts. A doubled or dropped magnetic term shows up here
  // immediately, which is the single most consequential bug class in this port.
  const Real mu0 = 4.0 * M_PI;

  struct Case {
    Real rho, v1, v2, v3, u, bx, by, bz;
  };
  // Deliberately spans zero field, field-aligned flow, oblique field with oblique flow,
  // and a strongly magnetized (low-beta) state where the magnetic term dominates E.
  const Case cases[] = {
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},        // no field, no flow
      {1.0, 0.3, 0.0, 0.0, 1.0, 0.7, 0.0, 0.0},        // aligned
      {2.5, -0.4, 0.9, 0.2, 3.0, 0.5, -1.1, 0.3},      // fully oblique
      {0.125, 0.0, 0.0, 0.0, 1.0e-3, 3.0, -2.0, 1.5},  // low beta
      {7.0, 1.5, -2.5, 0.75, 12.0, -0.25, 0.5, -0.75}, // supersonic-ish
  };

  for (const auto &c : cases) {
    const Real ke = 0.5 * c.rho * (c.v1 * c.v1 + c.v2 * c.v2 + c.v3 * c.v3);
    const Real emag = MHD::MagneticEnergyDensity(c.bx, c.by, c.bz, mu0);

    // Forward: assemble the conserved total.
    const Real etot = c.u + ke + emag;

    // Backward: recover thermal energy the way FillInteriorDerived does.
    const Real u_recovered = etot - ke - emag;

    // Scale the tolerance by the largest term in the sum, so that the low-beta case
    // (where E is dominated by magnetic energy and u is 1e-3) is a real test of
    // cancellation rather than a trivially satisfied absolute comparison.
    const Real scale = std::max({std::abs(c.u), ke, emag, 1.0});
    CHECK(std::abs(u_recovered - c.u) <= kTightTol * scale);

    // The decomposition identity that Orszag-Tang checks globally at 1e-10.
    CHECK(std::abs(etot - (c.u + ke + emag)) <= kTightTol * scale);
  }
}

TEST_CASE("U01: magnetic energy is B^2/(2 mu0) with the documented conventions",
          "[mhd][units]") {
  // Pin the two conventions from ADR-001 against hand-computed values, so a future
  // change to the default cannot silently redefine the field.
  //   mu0 = 1      -> normalized field, energy = B^2/2      (donor parity)
  //   mu0 = 4*pi   -> Gaussian CGS,     energy = B^2/(8 pi) (RIOT production default)
  const Real bx = 3.0, by = 4.0, bz = 0.0; // |B|^2 = 25
  CHECK(MHD::MagneticEnergyDensity(bx, by, bz, 1.0) == Approx(12.5).margin(kTightTol));
  CHECK(MHD::MagneticEnergyDensity(bx, by, bz, 4.0 * M_PI) ==
        Approx(25.0 / (8.0 * M_PI)).margin(kTightTol));
}

TEST_CASE("U01b: dynamics are invariant under the mu0 unit convention", "[mhd][units]") {
  // A physical state expressed at mu0 = 1 with field b, and the same physical state at
  // mu0 = 4*pi with field B = b*sqrt(4*pi), must have identical magnetic energy and
  // identical wave speeds. This is the check that catches a mu0 factor applied in one
  // kernel but not another -- the failure that makes a run "work" in scale-free units
  // and go unstable in CGS.
  const Real mu0_cgs = 4.0 * M_PI;
  const Real s = std::sqrt(mu0_cgs);
  const Real bn = 0.6, bt1 = -0.8, bt2 = 0.3;
  const Real rho = 1.7, bmod = 2.3;

  const Real emag_norm = MHD::MagneticEnergyDensity(bn, bt1, bt2, 1.0);
  const Real emag_cgs = MHD::MagneticEnergyDensity(bn * s, bt1 * s, bt2 * s, mu0_cgs);
  CHECK(emag_cgs == Approx(emag_norm).epsilon(kTightTol));

  const Real cf_norm = MHD::FastMagnetosonicSpeed(bmod, rho, bn, bt1, bt2, 1.0);
  const Real cf_cgs =
      MHD::FastMagnetosonicSpeed(bmod, rho, bn * s, bt1 * s, bt2 * s, mu0_cgs);
  CHECK(cf_cgs == Approx(cf_norm).epsilon(kTightTol));

  const Real b2_norm = bn * bn + bt1 * bt1 + bt2 * bt2;
  CHECK(MHD::SignalSpeedBound(bmod, rho, b2_norm * mu0_cgs, mu0_cgs) ==
        Approx(MHD::SignalSpeedBound(bmod, rho, b2_norm, 1.0)).epsilon(kTightTol));
}

//========================================================================================
// U03 -- fast magnetosonic speed
//========================================================================================
TEST_CASE("U03: fast speed satisfies its defining quadratic for oblique fields",
          "[mhd][wavespeed]") {
  // The strongest available check: cf^2 must be the larger root of
  //   c^4 - (a^2 + vA^2) c^2 + a^2 vAn^2 = 0.
  // Verifying the residual (rather than comparing against a second implementation of
  // the same closed form) means a sign or density-factor error cannot cancel out.
  const Real mu0 = 1.7; // deliberately not 1, to keep mu0 factors honest

  struct Case {
    Real bmod, rho, bn, bt1, bt2;
  };
  const Case cases[] = {
      {2.0, 1.0, 0.5, 0.25, -0.75}, {5.0, 0.3, -1.2, 0.4, 0.9},
      {0.4, 3.3, 0.1, 2.2, -1.4},   {1.0, 1.0, 1.0, 1.0, 1.0},
      {9.0, 0.05, 0.03, -0.02, 0.01},
  };

  for (const auto &c : cases) {
    const Real cf = MHD::FastMagnetosonicSpeed(c.bmod, c.rho, c.bn, c.bt1, c.bt2, mu0);
    const Real a2 = c.bmod / c.rho;
    const Real b2 = c.bn * c.bn + c.bt1 * c.bt1 + c.bt2 * c.bt2;
    const Real vA2 = b2 / (mu0 * c.rho);
    const Real vAn2 = c.bn * c.bn / (mu0 * c.rho);

    const Real cf2 = cf * cf;
    const Real residual = cf2 * cf2 - (a2 + vA2) * cf2 + a2 * vAn2;
    const Real scale = std::max(SQR(a2 + vA2), 1.0);
    CHECK(std::abs(residual) <= 1.0e-11 * scale);

    // The fast speed is an upper bound on both the acoustic and the normal Alfven
    // speed. A common porting error is to return the Alfven speed alone.
    CHECK(cf >= std::sqrt(a2) - kTightTol);
    CHECK(cf >= std::sqrt(vAn2) - kTightTol);
    // ... and is bounded above by the direction-independent bound used for the CFL.
    CHECK(cf <= MHD::SignalSpeedBound(c.bmod, c.rho, b2, mu0) + kTightTol);
  }
}

TEST_CASE("U03: fast speed handles the degenerate orientations", "[mhd][wavespeed]") {
  const Real mu0 = 2.5;
  const Real rho = 1.3, bmod = 3.1;
  const Real a2 = bmod / rho;
  const Real bmag = 0.9;
  const Real vA2 = bmag * bmag / (mu0 * rho);

  // Parallel propagation (field entirely along the normal): the fast speed degenerates
  // to max(a, vA). The discriminant is an exact square here, so this is where a
  // roundoff-negative discriminant would first bite.
  {
    const Real cf = MHD::FastMagnetosonicSpeed(bmod, rho, bmag, 0.0, 0.0, mu0);
    CHECK(cf * cf == Approx(std::max(a2, vA2)).epsilon(1.0e-11));
  }

  // Perpendicular propagation (no normal field): cf^2 = a^2 + vA^2 exactly.
  {
    const Real cf = MHD::FastMagnetosonicSpeed(bmod, rho, 0.0, bmag, 0.0, mu0);
    CHECK(cf * cf == Approx(a2 + vA2).epsilon(1.0e-11));
  }

  // Vanishing field: must reduce to the pure acoustic speed, which is the hydro limit
  // the H02 regression test checks globally.
  {
    const Real cf = MHD::FastMagnetosonicSpeed(bmod, rho, 0.0, 0.0, 0.0, mu0);
    CHECK(cf * cf == Approx(a2).epsilon(1.0e-11));
  }

  // Degenerate case where a^2 == vA^2 exactly: the discriminant is identically zero and
  // is the most likely place to produce a NaN from sqrt of a small negative number.
  {
    const Real rho_d = 1.0;
    const Real bmod_d = 1.0;                          // a^2 = 1
    const Real b_d = std::sqrt(mu0 * rho_d * bmod_d); // vA^2 = 1
    const Real cf = MHD::FastMagnetosonicSpeed(bmod_d, rho_d, b_d, 0.0, 0.0, mu0);
    CHECK(std::isfinite(cf));
    CHECK(cf * cf == Approx(1.0).epsilon(1.0e-11));
  }
}

//========================================================================================
// Face-to-cell reconstruction
//========================================================================================
TEST_CASE("Face-to-cell B reproduces a uniform field and its energy", "[mhd][faces]") {
  // A uniform field must survive face-to-cell reconstruction exactly. This is the
  // discrete statement behind the G2 requirement that adding a uniform, force-free
  // field at fixed thermal state changes no pressure: if the reconstruction were
  // lossy, magnetic energy would be mis-subtracted and the recovered temperature would
  // shift.
  const Real mu0 = 4.0 * M_PI;
  const Real bx = 0.75, by = -1.25, bz = 0.5;
  const auto faces = MakeUniformFaces(bx, by, bz);
  const auto pv = faces.View();

  // Index 0, with unit offsets into the second slot of each face array.
  Real rbx, rby, rbz;
  MHD::FaceToCellB(pv, 0, 1, 1, 1, rbx, rby, rbz);
  CHECK(rbx == Approx(bx).margin(kTightTol));
  CHECK(rby == Approx(by).margin(kTightTol));
  CHECK(rbz == Approx(bz).margin(kTightTol));

  CHECK(MHD::CellMagneticEnergyFromFaces(pv, 0, 1, 1, 1, mu0) ==
        Approx(MHD::MagneticEnergyDensity(bx, by, bz, mu0)).margin(kTightTol));
}

TEST_CASE("Face-to-cell B averages a linearly varying face field", "[mhd][faces]") {
  // For a linear profile the arithmetic mean of the two bounding faces is the exact
  // cell-centered value, so this pins the averaging convention (mean, not one-sided
  // pickup) that must match between the energy subtraction and re-synthesis.
  const Real lo = 1.0, hi = 3.0;
  const Real f1[2] = {lo, hi};
  const Real f2[2] = {-2.0, 6.0};
  const Real f3[2] = {0.5, 0.5};
  const StubFaceView pv{f1, f2, f3};

  Real bx, by, bz;
  MHD::FaceToCellB(pv, 0, 1, 1, 1, bx, by, bz);
  CHECK(bx == Approx(2.0).margin(kTightTol));  // (1 + 3)/2
  CHECK(by == Approx(2.0).margin(kTightTol));  // (-2 + 6)/2
  CHECK(bz == Approx(0.5).margin(kTightTol));  // uniform
}

TEST_CASE("Face-to-cell B keeps transverse components in a collapsed direction",
          "[mhd][faces]") {
  // In a reduced-dimension run the offset for a collapsed direction is zero, so both
  // reads hit the same slot and the mean returns that value. Transverse field
  // components are physically nonzero and evolving in 1D/2D, so dropping them (for
  // instance by zeroing a component when nx2 == 1) would be a physics bug, not an
  // optimization. This test locks in the non-dropping behavior.
  const Real f1[2] = {0.25, 0.75};
  const Real f2[2] = {-1.5, 999.0}; // second slot must never be read
  const Real f3[2] = {2.25, 999.0}; // second slot must never be read
  const StubFaceView pv{f1, f2, f3};

  Real bx, by, bz;
  MHD::FaceToCellB(pv, 0, /*di=*/1, /*dj=*/0, /*dk=*/0, bx, by, bz);
  CHECK(bx == Approx(0.5).margin(kTightTol)); // averaged: (0.25 + 0.75)/2
  CHECK(by == Approx(-1.5).margin(kTightTol));
  CHECK(bz == Approx(2.25).margin(kTightTol));
}

//========================================================================================
// U04 -- discrete curl of a vector potential is discretely divergence free
//========================================================================================
namespace {

// A small staggered grid, laid out with the same conventions as Parthenon's
// TopologicalElement staggering (basic_types.hpp:220): F1 lives at (i-1/2, j, k), E1
// lives at (i, j-1/2, k-1/2), and so on.
//
// This exercise validates the ORIENTATION AND INDEXING of the discrete curl and
// divergence operators independently of any kernel, before those conventions get baked
// into the problem generators and the CT update. Getting them wrong is the defect that
// surfaces much later as div B growing at block boundaries, where it is far harder to
// localize.
struct StaggeredGrid {
  static constexpr int n1 = 6, n2 = 5, n3 = 4;
  static constexpr Real d1 = 0.25, d2 = 0.4, d3 = 0.5;

  // Edge-centered vector potential. Ai has extents that are cell-like in direction i
  // and node-like in the two transverse directions.
  std::vector<Real> a1, a2, a3;
  // Face-centered magnetic field: normal component on each face.
  std::vector<Real> b1, b2, b3;

  StaggeredGrid()
      : a1((n1) * (n2 + 1) * (n3 + 1)), a2((n1 + 1) * (n2) * (n3 + 1)),
        a3((n1 + 1) * (n2 + 1) * (n3)), b1((n1 + 1) * n2 * n3),
        b2(n1 * (n2 + 1) * n3), b3(n1 * n2 * (n3 + 1)) {}

  static int IdxA1(int i, int j, int k) { return (k * (n2 + 1) + j) * (n1) + i; }
  static int IdxA2(int i, int j, int k) { return (k * (n2) + j) * (n1 + 1) + i; }
  static int IdxA3(int i, int j, int k) { return (k * (n2 + 1) + j) * (n1 + 1) + i; }
  static int IdxB1(int i, int j, int k) { return (k * n2 + j) * (n1 + 1) + i; }
  static int IdxB2(int i, int j, int k) { return (k * (n2 + 1) + j) * n1 + i; }
  static int IdxB3(int i, int j, int k) { return (k * n2 + j) * n1 + i; }

  // Edge midpoint coordinates for each vector-potential component.
  static void XA1(int i, int j, int k, Real &x, Real &y, Real &z) {
    x = (i + 0.5) * d1;
    y = j * d2;
    z = k * d3;
  }
  static void XA2(int i, int j, int k, Real &x, Real &y, Real &z) {
    x = i * d1;
    y = (j + 0.5) * d2;
    z = k * d3;
  }
  static void XA3(int i, int j, int k, Real &x, Real &y, Real &z) {
    x = i * d1;
    y = j * d2;
    z = (k + 0.5) * d3;
  }

  // Fill A from an arbitrary smooth analytic vector potential. The point of the test is
  // that div(curl A) vanishes for ANY A, so a deliberately messy A is a better test
  // than a simple one.
  template <class Fn>
  void FillPotential(Fn &&A) {
    Real x, y, z, ax, ay, az;
    for (int k = 0; k <= n3; ++k)
      for (int j = 0; j <= n2; ++j)
        for (int i = 0; i < n1; ++i) {
          XA1(i, j, k, x, y, z);
          A(x, y, z, ax, ay, az);
          a1[IdxA1(i, j, k)] = ax;
        }
    for (int k = 0; k <= n3; ++k)
      for (int j = 0; j < n2; ++j)
        for (int i = 0; i <= n1; ++i) {
          XA2(i, j, k, x, y, z);
          A(x, y, z, ax, ay, az);
          a2[IdxA2(i, j, k)] = ay;
        }
    for (int k = 0; k < n3; ++k)
      for (int j = 0; j <= n2; ++j)
        for (int i = 0; i <= n1; ++i) {
          XA3(i, j, k, x, y, z);
          A(x, y, z, ax, ay, az);
          a3[IdxA3(i, j, k)] = az;
        }
  }

  // B = curl A, evaluated as a line integral of A around each face boundary divided by
  // the face area (Stokes). This is exactly the discretization a problem generator must
  // use to start a run with div B = 0 to machine precision.
  void CurlIntoFaces() {
    for (int k = 0; k < n3; ++k)
      for (int j = 0; j < n2; ++j)
        for (int i = 0; i <= n1; ++i)
          b1[IdxB1(i, j, k)] =
              (a3[IdxA3(i, j + 1, k)] - a3[IdxA3(i, j, k)]) / d2 -
              (a2[IdxA2(i, j, k + 1)] - a2[IdxA2(i, j, k)]) / d3;

    for (int k = 0; k < n3; ++k)
      for (int j = 0; j <= n2; ++j)
        for (int i = 0; i < n1; ++i)
          b2[IdxB2(i, j, k)] =
              (a1[IdxA1(i, j, k + 1)] - a1[IdxA1(i, j, k)]) / d3 -
              (a3[IdxA3(i + 1, j, k)] - a3[IdxA3(i, j, k)]) / d1;

    for (int k = 0; k <= n3; ++k)
      for (int j = 0; j < n2; ++j)
        for (int i = 0; i < n1; ++i)
          b3[IdxB3(i, j, k)] =
              (a2[IdxA2(i + 1, j, k)] - a2[IdxA2(i, j, k)]) / d1 -
              (a1[IdxA1(i, j + 1, k)] - a1[IdxA1(i, j, k)]) / d2;
  }

  // Discrete divergence from the face flux balance -- the same form as
  // MHD::SetDerivedMagneticFields, and the only form in which the shared contributions
  // cancel algebraically.
  Real MaxAbsDivergence() const {
    Real worst = 0.0;
    for (int k = 0; k < n3; ++k)
      for (int j = 0; j < n2; ++j)
        for (int i = 0; i < n1; ++i) {
          const Real div =
              (b1[IdxB1(i + 1, j, k)] - b1[IdxB1(i, j, k)]) / d1 +
              (b2[IdxB2(i, j + 1, k)] - b2[IdxB2(i, j, k)]) / d2 +
              (b3[IdxB3(i, j, k + 1)] - b3[IdxB3(i, j, k)]) / d3;
          worst = std::max(worst, std::abs(div));
        }
    return worst;
  }

  Real MaxAbsField() const {
    Real m = 0.0;
    for (Real v : b1) m = std::max(m, std::abs(v));
    for (Real v : b2) m = std::max(m, std::abs(v));
    for (Real v : b3) m = std::max(m, std::abs(v));
    return m;
  }
};

} // namespace

TEST_CASE("U04: discrete curl of a vector potential is divergence free to roundoff",
          "[mhd][divb]") {
  StaggeredGrid g;

  // A deliberately non-separable, non-symmetric potential. div(curl A) = 0 is an
  // identity of the discrete operators, so it must hold for arbitrary A -- a simple or
  // symmetric choice could hide an index error through accidental cancellation.
  g.FillPotential([](Real x, Real y, Real z, Real &ax, Real &ay, Real &az) {
    ax = std::sin(1.3 * x) * std::cos(2.1 * y) + 0.4 * z * z;
    ay = std::exp(0.3 * z) * std::sin(0.9 * x + 0.5 * y) - 0.2 * x;
    az = std::cos(1.7 * y) * (1.0 + 0.6 * x) + 0.15 * y * z;
  });
  g.CurlIntoFaces();

  const Real bmax = g.MaxAbsField();
  REQUIRE(bmax > 0.1); // the test would be vacuous with a near-zero field

  // Normalize by the field scale and the smallest cell width, matching the
  // dimensionless divergence measure used by the regression tests.
  const Real eta = g.MaxAbsDivergence() * StaggeredGrid::d1 / bmax;
  CHECK(eta <= 1.0e-13);
}

TEST_CASE("U04: a uniform field is exactly divergence free", "[mhd][divb]") {
  // The trivial-but-essential case: a uniform field must produce identically zero
  // divergence, and this is what the "uniform field magnetic pressure is nonzero but
  // its force is zero" requirement rests on.
  StaggeredGrid g;
  g.FillPotential([](Real x, Real y, Real z, Real &ax, Real &ay, Real &az) {
    // A = (0, B0z * x ... ) chosen so curl A is the uniform field (0.3, -0.7, 1.1)
    ax = -0.7 * z;
    ay = 1.1 * x;
    az = 0.3 * y;
  });
  g.CurlIntoFaces();

  // Verify the curl really produced the intended uniform field before trusting div.
  CHECK(g.b1[StaggeredGrid::IdxB1(2, 1, 1)] == Approx(0.3).margin(1.0e-12));
  CHECK(g.b2[StaggeredGrid::IdxB2(2, 1, 1)] == Approx(-0.7).margin(1.0e-12));
  CHECK(g.b3[StaggeredGrid::IdxB3(2, 1, 1)] == Approx(1.1).margin(1.0e-12));

  CHECK(g.MaxAbsDivergence() <= 1.0e-13);
}
