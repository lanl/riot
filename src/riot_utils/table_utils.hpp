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
#ifndef RIOT_UTILS_TABLE_UTILS_HPP_
#define RIOT_UTILS_TABLE_UTILS_HPP_

#include <ports-of-call/robust_utils.hpp>

#include <spiner/databox.hpp>

#include <kokkos_abstraction.hpp>

namespace RiotTables {

using Uniform1D = Spiner::DataBox<Real>;

// Resample a time series table to uniform spacing using linear interpolation.
// Input table has first column as time and second column as values.
// Output table uses dt equal to the minimum spacing of the input table
// and is a 1D databox, ready for use.
Uniform1D UniformlyResampleTimeSeries(const parthenon::HostArray2D<Real> &table);

// Get the finite differences derivative of a spiner table
KOKKOS_INLINE_FUNCTION
Real GetFDDerivative(const Uniform1D &table, const Real x0) {
  constexpr Real eps = 3.0e-6; // Optimal for forward differences
  const Real dx = x0 * eps;
  const Real x1 = x0 + dx;
  const Real f0 = table.interpToReal(x0);
  const Real f1 = table.interpToReal(x1);
  return PortsOfCall::Robust::ratio(f1 - f0, dx);
}

} // namespace RiotTables

#endif // RIOT_UTILS_TABLE_UTILS_HPP_
