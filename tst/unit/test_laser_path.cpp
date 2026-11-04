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

// Unit tests for the path length returned by Laser::ParticlePusher::step_to_boundary
// (src/laser/laser.hpp).
//
// step_to_boundary advances a laser ray to the boundary of the cell it is in and returns
// {path, speed}. The caller (src/laser/laser.cpp) uses the path in two places:
//
//   * the optical depth of the cell,        tau = kappa * path,
//   * the cell energy density diagnostic,   (E / (speed * dt)) * path / volume.
//
// Both only make sense if `path` is the geodesic distance the ray travelled in the cell,
// and `path / speed` is the time it spent there.
//
// Invariants checked here, all of which follow from the definitions in the code:
//
//   * DISTANCE: for a ray moving along x with a uniform density gradient (so the speed
//     changes but the direction does not), the path equals the distance between the
//     entry point and the face it exits through. This also holds when the ray is
//     accelerated or decelerated by the gradient, because velocity Verlet is exact for a
//     constant acceleration and the Verlet step travels the mean of its end speeds times
//     its duration.
//   * TRANSIT TIME: with no force, path / speed equals the time elapsed on the ray.
//   * COMPOSITION: the path of a cell crossed in two successive calls adds up to the
//     distance covered, i.e. a cell cannot "lose" length depending on where it is split.
//
// ParticlePusher only touches the cell through `ndim`, `xb`, and `grad_ne`, so a mock
// cell is enough: no mesh or Kokkos device kernel is needed. The coordinate system is
// fixed when Parthenon is configured; these tests are written for Cartesian coordinates.

#include <array>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

#include "laser/laser.hpp"

using Catch::Approx;
using parthenon::Real;

namespace {

// Minimal stand-in for Laser::CellInfo: one cell [xlo, xhi] in x with a constant
// electron-density gradient dn/dx = gx. Only the members used by ParticlePusher exist.
struct MockCell {
  static constexpr int ndim = 1;
  std::array<Real, 6> xb;
  Real gx;

  MockCell(const Real xlo, const Real xhi, const Real gradient)
      : xb{xlo, xhi, -1.0, 1.0, -1.0, 1.0}, gx(gradient) {}

  std::array<Real, 3> grad_ne(const Real, const Real, const Real) const {
    return {gx, 0.0, 0.0};
  }
};

constexpr Real kLambda = 5.3e-5;              // cm, 530 nm
constexpr Real kLight = 2.99792458e10;        // cm/s
constexpr Real kCell = 1.0e-2;                // cm

// Gradient of n_e that changes (speed)^2 of a ray by `dv2_over_c2` * c^2 across a cell of
// width kCell. In the code, the ray acceleration is a = -(c^2 / 2) grad(n_e) / n_crit
// (accel_coeff * n_crit == c^2 / 2), so a * kCell == (c^2 / 2) * dv2_over_c2 gives
// v1^2 = v0^2 - dv2_over_c2 c^2 for a ray crossing the cell along x.
Real gradient_for(const Real dv2_over_c2) {
  return dv2_over_c2 * Laser::ne_crit(kLambda) / kCell;
}

Laser::LaserParticle make_ray(const Real x, const Real v) {
  Laser::LaserParticle lp;
  lp.t = 0.0;
  lp.x = {x, 0.0, 0.0};
  lp.v = {v, 0.0, 0.0};
  lp.energy = 1.0;
  lp.lambda = kLambda;
  return lp;
}

} // namespace

TEST_CASE("ParticlePusher path length is the distance travelled, no force",
          "[laser][pusher]") {
  if (!parthenon::IsCoord<parthenon::UniformCartesian>())
    SKIP("These tests use a Cartesian mock cell");
  for (const Real v : {0.3 * kLight, kLight, 1.0}) {
    MockCell cell(0.0, kCell, 0.0);
    auto lp = make_ray(0.0, v);
    Laser::ParticlePusher pusher(lp, cell);
    const auto [path, speed] = pusher.step_to_boundary(1.0e300);
    // The ray went from x = 0 to the face at x = kCell.
    CHECK(path == Approx(kCell).epsilon(1.0e-12));
    CHECK(speed == Approx(v).epsilon(1.0e-12));
    // With a constant speed the time on the ray is the length over the speed.
    CHECK(path / speed == Approx(lp.t).epsilon(1.0e-12));
  }
}

TEST_CASE("ParticlePusher path length is the distance travelled, with a force",
          "[laser][pusher]") {
  if (!parthenon::IsCoord<parthenon::UniformCartesian>())
    SKIP("These tests use a Cartesian mock cell");
  // Ray entering a cell with n_e rising (decelerates) and falling (accelerates), and
  // leaving through either face.
  struct Case {
    Real dv2;     // change of v^2 over the cell in units of c^2 (sign of acceleration)
    Real v0;      // entry speed in units of c
    bool rightward;
  };
  const Case cases[] = {
      {+0.30, 0.80, true},  // decelerating, +x
      {+0.05, 0.50, true},  // weakly decelerating
      {-0.30, 0.80, true},  // accelerating
      {-0.60, 0.30, true},  // strongly accelerating
      {+0.30, 0.80, false}, // decelerating, -x
      {-0.30, 0.80, false}, // accelerating, -x
  };
  for (const auto &c : cases) {
    DYNAMIC_SECTION("dv2 = " << c.dv2 << " v0 = " << c.v0
                             << (c.rightward ? " +x" : " -x")) {
      // grad(n_e) as seen along the direction of travel has the sign set by dv2.
      const Real g = (c.rightward ? 1.0 : -1.0) * gradient_for(c.dv2);
      MockCell cell(0.0, kCell, g);
      const Real x0 = c.rightward ? 0.0 : kCell;
      const Real v0 = (c.rightward ? 1.0 : -1.0) * c.v0 * kLight;
      auto lp = make_ray(x0, v0);
      Laser::ParticlePusher pusher(lp, cell);
      const auto [path, speed] = pusher.step_to_boundary(1.0e300);
      CHECK(path == Approx(kCell).epsilon(1.0e-9));
      // the mean of the entry and exit speeds lies between them
      const Real vexit = std::abs(lp.v[0]);
      CHECK(speed > std::min(c.v0 * kLight, vexit) * (1.0 - 1.0e-12));
      CHECK(speed < std::max(c.v0 * kLight, vexit) * (1.0 + 1.0e-12));
    }
  }
}

TEST_CASE("ParticlePusher path lengths of adjacent cells add up", "[laser][pusher]") {
  if (!parthenon::IsCoord<parthenon::UniformCartesian>())
    SKIP("These tests use a Cartesian mock cell");
  // Cross a region of width 2 * kCell as two cells with the same gradient. The summed path
  // must be 2 * kCell regardless of the split.
  const Real g = gradient_for(0.1);
  Real total = 0.0;
  auto lp = make_ray(0.0, 0.7 * kLight);
  for (int cell = 0; cell < 2; cell++) {
    MockCell mock(cell * kCell, (cell + 1) * kCell, g);
    Laser::ParticlePusher pusher(lp, mock);
    total += pusher.step_to_boundary(1.0e300).first;
  }
  CHECK(total == Approx(2.0 * kCell).epsilon(1.0e-9));
}
