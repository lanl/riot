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
#ifndef FOREST_MAPPED_REGIONS_HPP_
#define FOREST_MAPPED_REGIONS_HPP_
// This file was made in part with generative AI.
//
// Region initialization on mapped (forest) meshes, where the subcells are on the lattice
// of the block's logical coordinates. The hooks here are no-ops unless the coordinates
// are mapped.

#include <array>
#include <cmath>

#include <coordinates/coordinates.hpp>

namespace RiotForest {

using parthenon::Real;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

// the logical position of the first face of the block
template <typename Coords>
inline void LogicalOrigin(const Coords &coords, std::array<Real, 3> &xlo) {
  if constexpr (parthenon::IsMappedCoord()) {
    xlo[0] = coords.template Xf<X1DIR>(0);
    xlo[1] = coords.template Xf<X2DIR>(0);
    xlo[2] = coords.template Xf<X3DIR>(0);
  }
}

// the physical position of the logical position (x1, x2, x3); x3 is the azimuth, and is
// left unchanged, on mapped (R, z) meshes
template <typename Coords>
inline void MapLogicalPosition(const Coords &coords, Real &x1, Real &x2, Real &x3) {
  if constexpr (parthenon::IsCoord<parthenon::MappedCartesian>()) {
    Real x, y, z;
    coords.Map(x1, x2, x3, x, y, z);
    x1 = x;
    x2 = y;
    x3 = z;
  } else if constexpr (parthenon::IsMappedCoord()) {
    Real r, z;
    coords.Map(x1, x2, r, z);
    x1 = r;
    x2 = z;
  }
}

// physical volume and first moments of the subcell with logical bounds [xlo, xhi]: the
// volume, int x dV, int y dV and int z dV (as for cylindrical, the volume, int R dV and
// int z dV on mapped (R, z) meshes)
template <typename Coords>
inline void SubcellMoments(const Coords &coords, const std::array<Real, 3> &xlo,
                           const std::array<Real, 3> &xhi, std::array<Real, 4> &moments) {
  if constexpr (parthenon::IsCoord<parthenon::MappedCartesian>()) {
    Real m[4];
    coords.LogicalBoxMoments(xlo.data(), xhi.data(), m);
    moments = {m[0], m[1], m[2], m[3]};
  } else if constexpr (parthenon::IsMappedCoord()) {
    Real m0, m1, m2;
    coords.LogicalQuadMoments(xlo[0], xlo[1], xhi[0], xhi[1], m0, m1, m2);
    moments = {2.0 * M_PI * m0, 2.0 * M_PI * m1, 2.0 * M_PI * m2, 0.0};
  }
}

} // namespace RiotForest

#endif // FOREST_MAPPED_REGIONS_HPP_
