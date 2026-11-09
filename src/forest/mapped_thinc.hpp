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
#ifndef FOREST_MAPPED_THINC_HPP_
#define FOREST_MAPPED_THINC_HPP_
// This file was made in part with generative AI.
//
// THINC on mapped (forest) meshes. THINC reconstructs the material profiles over the
// logical unit cell (the Youngs gradients and the stencils are in index space), so on
// mapped meshes only the measure of the cell differs: its Gauss points carry the volume
// density of the map, and the points of its faces the area density.

#include <cmath>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>

#include "forest/mapped_geometry.hpp"

namespace RiotForest {

// The densities of ThincDensities: 4^3 cell points and 4^2 points on each of two faces
constexpr int kThincCellPoints = 64, kThincFacePoints = 16;
constexpr int kThincDensities =
    parthenon::IsMappedCoord() ? kThincCellPoints + 2 * kThincFacePoints : 1;

//----------------------------------------------------------------------------------------
//! \brief Hook in THINC::ReconstructFractions: the volume density at the 4^ndim Gauss
//!        points (gauss_point, gauss_weight on [-1/2, 1/2]) of mapped cell (k, j, i), in
//!        THINC's order (base-4 digit d of a point selects its coordinate along d), from
//!        density[0], and the area density at the 4^(ndim - 1) points of its lower and
//!        upper faces along axis (the transverse directions in increasing order), from
//!        density[kThincCellPoints] and density[kThincCellPoints + kThincFacePoints].
//!        Each is normalized to unit mean. In (R, z) the volume density is R |det J| and
//!        the area density R |dX / dl| along the face; in 3D |det J| and the norm of the
//!        cross product of the face's tangents.
template <typename Coords>
KOKKOS_INLINE_FUNCTION void ThincDensities(const Coords &coords, const int ndim,
                                           const int axis, const int k, const int j,
                                           const int i, const Real *gauss_point,
                                           const Real *gauss_weight, Real *density) {
  if constexpr (parthenon::IsMappedCoord()) {
    using parthenon::X1DIR;
    using parthenon::X2DIR;
    using parthenon::X3DIR;
    const Real lc[3] = {coords.template Xc<X1DIR>(i), coords.template Xc<X2DIR>(j),
                        coords.template Xc<X3DIR>(k)};
    const Real dl[3] = {coords.template Dx<X1DIR>(), coords.template Dx<X2DIR>(),
                        coords.template Dx<X3DIR>()};
    // the volume density at logical l, or the area density of the face normal to axis
    auto Density = [&](const Real l[3], const int face_axis) {
      Real J[3][3];
      Real r = 1.0;
      if constexpr (IsMapped3D()) {
        coords.Patch().Jacobian(l[0], l[1], l[2], J);
      } else {
        Real z;
        coords.Map(l[0], l[1], r, z);
        coords.Patch().Jacobian(l[0], l[1], J[0][0], J[0][1], J[1][0], J[1][1]);
      }
      if (face_axis < 0) {
        const Real det = IsMapped3D() ? J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                                            J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                                            J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
                                      : J[0][0] * J[1][1] - J[0][1] * J[1][0];
        return std::abs(r * det);
      }
      if constexpr (IsMapped3D()) {
        const int a = (face_axis + 1) % 3, b = (face_axis + 2) % 3;
        Real c[3];
        for (int d = 0; d < 3; ++d)
          c[d] = J[(d + 1) % 3][a] * J[(d + 2) % 3][b] - J[(d + 2) % 3][a] * J[(d + 1) % 3][b];
        return std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
      } else {
        const int t = 1 - face_axis;
        return r * std::sqrt(J[0][t] * J[0][t] + J[1][t] * J[1][t]);
      }
    };
    const int cell_points = 1 << (2 * ndim), face_points = 1 << (2 * (ndim - 1));
    Real l[3] = {lc[0], lc[1], lc[2]};
    Real mean = 0.0;
    for (int q = 0; q < cell_points; ++q) {
      Real weight = 1.0;
      for (int d = 0, r = q; d < ndim; ++d, r /= 4) {
        l[d] = lc[d] + gauss_point[r % 4] * dl[d];
        weight *= gauss_weight[r % 4];
      }
      density[q] = Density(l, -1);
      mean += weight * density[q];
    }
    for (int q = 0; q < cell_points; ++q)
      density[q] /= mean;
    for (int s = 0; s < 2; ++s) {
      Real *face = density + kThincCellPoints + s * kThincFacePoints;
      Real fl[3] = {lc[0], lc[1], lc[2]};
      fl[axis] = lc[axis] + (s - 0.5) * dl[axis];
      Real face_mean = 0.0;
      for (int f = 0; f < face_points; ++f) {
        Real weight = 1.0;
        for (int d = 0, r = f; d < ndim; ++d) {
          if (d == axis) continue;
          fl[d] = lc[d] + gauss_point[r % 4] * dl[d];
          weight *= gauss_weight[r % 4];
          r /= 4;
        }
        face[f] = Density(fl, axis);
        face_mean += weight * face[f];
      }
      for (int f = 0; f < face_points; ++f)
        face[f] /= face_mean;
    }
  }
}

} // namespace RiotForest

#endif // FOREST_MAPPED_THINC_HPP_
