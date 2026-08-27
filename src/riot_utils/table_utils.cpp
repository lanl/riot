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

#include <ports-of-call/robust_utils.hpp>

#include <spiner/databox.hpp>

#include <kokkos_abstraction.hpp>

#include "riot_utils/table_utils.hpp"

namespace RiotTables {

Uniform1D UniformlyResampleTimeSeries(const parthenon::HostArray2D<Real> &table) {
  const int nrows = table.extent(0);
  PARTHENON_REQUIRE(nrows >= 2, "Time series must have at least 2 rows");
  PARTHENON_REQUIRE(table.extent(1) == 2, "Table must have exactly 2 columns");

  Real min_dt = std::numeric_limits<Real>::max();
  for (int i = 0; i < nrows - 1; ++i) {
    Real dt = table(i + 1, 0) - table(i, 0);
    PARTHENON_REQUIRE(dt > 0, "Time series must be monotonically increasing");
    min_dt = std::min(min_dt, dt);
  }

  const Real t_start = table(0, 0);
  const Real t_end = table(nrows - 1, 0);
  const int new_nrows = static_cast<int>((t_end - t_start) / min_dt) + 1;

  Uniform1D result(new_nrows);
  Spiner::RegularGrid1D<Real> times(t_start, t_end, new_nrows);
  result.setRange(0, times);

  int j = 0;
  for (int i = 0; i < new_nrows; ++i) {
    const Real t = times.x(i);
    while (j < nrows - 1 && table(j + 1, 0) < t) {
      ++j;
    }

    if (j >= nrows - 1) {
      result(i) = table(nrows - 1, 1);
    } else if (table(j, 0) == t) {
      result(i) = table(j, 1);
    } else {
      const Real t0 = table(j, 0);
      const Real t1 = table(j + 1, 0);
      const Real v0 = table(j, 1);
      const Real v1 = table(j + 1, 1);
      const Real alpha = (t - t0) / (t1 - t0);
      result(i) = v0 + alpha * (v1 - v0);
    }
  }

  return result;
}

} // namespace RiotTables
