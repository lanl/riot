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
#ifndef FOREST_MAPPED_GEOMETRY_HPP_
#define FOREST_MAPPED_GEOMETRY_HPP_
// This file was made in part with generative AI.

#include <cmath>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>

namespace RiotForest {

using parthenon::Real;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

// Axisymmetric (R, z) coordinates, on rectilinear or mapped blocks
constexpr bool IsCylindricalCoord() {
  return parthenon::IsCoord<parthenon::UniformCylindrical>() ||
         parthenon::IsCoord<parthenon::MappedCylindrical>();
}

// True on mapped Cartesian (3D) meshes, whose blocks are curved in all directions
constexpr bool IsMapped3D() {
  return parthenon::IsCoord<parthenon::MappedCartesian>();
}

// Number of velocity components that are rotated into the frame of a face: (v1, v2) on
// mapped (R, z) meshes and (v1, v2, v3) on mapped Cartesian ones
constexpr int FrameDims() { return IsMapped3D() ? 3 : 2; }

// True if the DIR faces of mapped blocks are not aligned with the global basis (the
// azimuthal faces of mapped (R, z) meshes are)
template <int DIR>
constexpr bool IsMappedDir() {
  return IsMapped3D() || (parthenon::IsMappedCoord() && DIR != X3DIR);
}

//----------------------------------------------------------------------------------------
//! \fn  Real InverseRadius
//! \brief Planar area over volume per radian of a mapped (R, z) cell. It is 2/(r1+r2) on
//!        rectilinear cells, and balances the flux divergence of a uniform pressure (or
//!        isotropic radiation field) exactly on curved ones. Zero on mapped Cartesian
//!        meshes, which have no geometric source terms.
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION Real InverseRadius(const Coords &coords, const int k,
                                               const int j, const int i) {
  if constexpr (IsMapped3D()) {
    return 0.0;
  } else {
    return coords.template FaceArea<X3DIR>(k, j, i) * coords.template Dx<X3DIR>() /
           coords.CellVolume(k, j, i);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Rotation of the velocity components v[0..FrameDims()-1] into the frame of a
//!        DIR face: slot DIR-1 holds the face-normal component and the other slots the
//!        tangential ones. It is the identity except on mapped meshes, where the Riemann
//!        solvers (which take slot DIR-1 as the normal) see the rotated state and their
//!        momentum fluxes are rotated back to the global basis. Row s of m is the unit
//!        vector of slot s in the global basis.
struct FaceFrame {
  Real m[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  KOKKOS_FORCEINLINE_FUNCTION void ToFace(Real v[3]) const {
    if constexpr (IsMapped3D()) {
      const Real v0 = v[0], v1 = v[1], v2 = v[2];
      for (int s = 0; s < 3; ++s)
        v[s] = m[s][0] * v0 + m[s][1] * v1 + m[s][2] * v2;
    } else if constexpr (parthenon::IsMappedCoord()) {
      const Real a0 = v[0];
      v[0] = m[0][0] * a0 + m[0][1] * v[1];
      v[1] = m[1][0] * a0 + m[1][1] * v[1];
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION void FromFace(Real v[3]) const {
    if constexpr (IsMapped3D()) {
      const Real v0 = v[0], v1 = v[1], v2 = v[2];
      for (int d = 0; d < 3; ++d)
        v[d] = m[0][d] * v0 + m[1][d] * v1 + m[2][d] * v2;
    } else if constexpr (parthenon::IsMappedCoord()) {
      const Real a0 = v[0];
      v[0] = m[0][0] * a0 + m[1][0] * v[1];
      v[1] = m[0][1] * a0 + m[1][1] * v[1];
    }
  }
};

//! \brief Frame of the DIR face at the lower side of cell (k, j, i). The tangents point
//!        along the increasing transverse logical indices, so the frame reduces to the
//!        identity on rectilinear faces and is a reflection on mirrored patches (keeping
//!        index-space stencils such as GetVdiff mirror symmetric). In 3D the tangents are
//!        the symmetric (Lowdin) orthonormalization of the face's logical tangent
//!        vectors, which commutes with relabeling and flipping them, so the blocks on
//!        either side of a tree boundary build the same frame up to a signed permutation
//!        of its tangential slots.
template <parthenon::CoordinateDirection DIR, typename Coords>
KOKKOS_FORCEINLINE_FUNCTION FaceFrame MakeFaceFrame(const Coords &coords, const int k,
                                                    const int j, const int i) {
  FaceFrame f;
  if constexpr (IsMapped3D()) {
    constexpr int dn = DIR - 1, da = DIR % 3, db = (DIR + 1) % 3;
    Real n[3];
    coords.template FaceNormal<DIR>(k, j, i, n[0], n[1], n[2]);
    // logical tangents at the face center, projected onto the face plane
    const Real x1 = (DIR == X1DIR) ? coords.template Xf<X1DIR>(i)
                                   : coords.template Xc<X1DIR>(i);
    const Real x2 = (DIR == X2DIR) ? coords.template Xf<X2DIR>(j)
                                   : coords.template Xc<X2DIR>(j);
    const Real x3 = (DIR == X3DIR) ? coords.template Xf<X3DIR>(k)
                                   : coords.template Xc<X3DIR>(k);
    Real J[3][3];
    coords.Patch().Jacobian(x1, x2, x3, J);
    Real ta[3], tb[3], tan = 0.0, tbn = 0.0;
    for (int d = 0; d < 3; ++d) {
      tan += J[d][da] * n[d];
      tbn += J[d][db] * n[d];
    }
    for (int d = 0; d < 3; ++d) {
      ta[d] = J[d][da] - tan * n[d];
      tb[d] = J[d][db] - tbn * n[d];
    }
    // [ta tb] G^(-1/2), with G the Gram matrix of (ta, tb)
    Real gaa = 0.0, gab = 0.0, gbb = 0.0;
    for (int d = 0; d < 3; ++d) {
      gaa += ta[d] * ta[d];
      gab += ta[d] * tb[d];
      gbb += tb[d] * tb[d];
    }
    const Real s = std::sqrt(gaa * gbb - gab * gab);
    const Real t = std::sqrt(gaa + gbb + 2.0 * s);
    const Real w = 1.0 / (s * t);
    for (int d = 0; d < 3; ++d) {
      f.m[dn][d] = n[d];
      f.m[da][d] = w * ((gbb + s) * ta[d] - gab * tb[d]);
      f.m[db][d] = w * ((gaa + s) * tb[d] - gab * ta[d]);
    }
  } else if constexpr (parthenon::IsMappedCoord() && DIR != X3DIR) {
    Real n1, n2, n3;
    coords.template FaceNormal<DIR>(k, j, i, n1, n2, n3);
    const Real o = coords.Orientation();
    if constexpr (DIR == X1DIR) {
      f.m[0][0] = n1;
      f.m[0][1] = n2;
      f.m[1][0] = -o * n2;
      f.m[1][1] = o * n1;
    } else {
      f.m[0][0] = o * n2;
      f.m[0][1] = -o * n1;
      f.m[1][0] = n1;
      f.m[1][1] = n2;
    }
  }
  return f;
}

} // namespace RiotForest

#endif // FOREST_MAPPED_GEOMETRY_HPP_
