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

// Unit tests for RiotTables::UniformlyResampleTimeSeries
// (src/riot_utils/table_utils.cpp).
//
// The function resamples a (time, value) table onto a uniform grid whose
// spacing equals the minimum input spacing, using piecewise-linear
// interpolation, and returns a 1D Spiner::DataBox. It is a host-only routine
// (it builds and fills a host DataBox), so all checks run on the host.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "riot_utils/table_utils.hpp"

using Catch::Approx;
using parthenon::Real;

namespace {
// Build a HostArray2D<Real> of shape (nrows, 2) from a (time, value) list.
parthenon::HostArray2D<Real> MakeTable(const std::vector<std::pair<Real, Real>> &rows) {
  parthenon::HostArray2D<Real> table("test_table", rows.size(), 2);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    table(i, 0) = rows[i].first;
    table(i, 1) = rows[i].second;
  }
  return table;
}
} // namespace

TEST_CASE("UniformlyResampleTimeSeries reproduces already-uniform linear data",
          "[table_utils][resample]") {
  // Uniform input, min_dt == 1.0, value == 2*t. Output grid should match the
  // input grid exactly and reproduce the linear values with no error.
  auto table = MakeTable({{0.0, 0.0}, {1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}});
  auto out = RiotTables::UniformlyResampleTimeSeries(table);

  // new_nrows = (t_end - t_start)/min_dt + 1 = 3/1 + 1 = 4  (golden size).
  REQUIRE(out.size() == 4);

  auto grid = out.range(0);
  CHECK(grid.min() == Approx(0.0));
  CHECK(grid.max() == Approx(3.0));

  for (int i = 0; i < out.size(); ++i) {
    const Real t = grid.x(i);
    CHECK(out(i) == Approx(2.0 * t)); // value == 2*t everywhere
  }
}

TEST_CASE("UniformlyResampleTimeSeries refines a non-uniform grid to min spacing",
          "[table_utils][resample]") {
  // Non-uniform times: spacings are 1.0 and 2.0, so min_dt == 1.0.
  // t_start = 0, t_end = 3  ->  new_nrows = 3/1 + 1 = 4, grid = {0,1,2,3}.
  // Values are linear in t (v == t), so linear interpolation is exact.
  auto table = MakeTable({{0.0, 0.0}, {1.0, 1.0}, {3.0, 3.0}});
  auto out = RiotTables::UniformlyResampleTimeSeries(table);

  REQUIRE(out.size() == 4);
  auto grid = out.range(0);
  CHECK(grid.min() == Approx(0.0));
  CHECK(grid.max() == Approx(3.0));

  // Endpoints preserved exactly.
  CHECK(out(0) == Approx(0.0));
  CHECK(out(out.size() - 1) == Approx(3.0));

  // Interpolated interior sample at t == 2 falls between input nodes (1,1) and
  // (3,3): expected v == 2 (golden interpolated value).
  CHECK(out(2) == Approx(2.0));

  // Full linear invariant: v(t) == t on the whole resampled grid.
  for (int i = 0; i < out.size(); ++i) {
    CHECK(out(i) == Approx(grid.x(i)));
  }
}

TEST_CASE("UniformlyResampleTimeSeries interpolates a piecewise-linear kink",
          "[table_utils][resample]") {
  // Value rises then falls; slopes differ across the middle node. Uniform input
  // spacing (min_dt == 1). Interpolation is exact on each linear segment.
  auto table = MakeTable({{0.0, 0.0}, {1.0, 2.0}, {2.0, 0.0}});
  auto out = RiotTables::UniformlyResampleTimeSeries(table);

  REQUIRE(out.size() == 3);
  auto grid = out.range(0);
  CHECK(out(0) == Approx(0.0)); // node
  CHECK(out(1) == Approx(2.0)); // peak node
  CHECK(out(2) == Approx(0.0)); // node
}
