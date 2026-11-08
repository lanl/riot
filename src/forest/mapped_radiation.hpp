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
#ifndef FOREST_MAPPED_RADIATION_HPP_
#define FOREST_MAPPED_RADIATION_HPP_
// This file was made in part with generative AI.
//
// Radiation transport on mapped (forest) meshes. The hooks here are no-ops unless the
// coordinates are mapped.

#include <array>

#include <coordinates/coordinates.hpp>
#include <interface/mesh_data.hpp>
#include <kokkos_abstraction.hpp>

#include "forest/mapped_geometry.hpp"

namespace RiotForest {

using parthenon::Real;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

//----------------------------------------------------------------------------------------
//! \fn  void FaceDirection
//! \brief Hook in the Jacobi coefficients: faces need not be parallel on mapped meshes,
//!        so the direction cosine nn of angle aa (and its sign up/down/sgnn) is the
//!        projection of the angle on the normal of the SDIR face at the lower side of
//!        (k, j, i), rather than the SDIR component of the angle.
template <int SDIR, typename Coords, typename CP>
KOKKOS_FORCEINLINE_FUNCTION void
FaceDirection(const Coords &coords, const CP &cp, const std::array<int, 3> &NDIR,
              const int aa, const int k, const int j, const int i, Real &nn, bool &up,
              bool &down, int &sgnn) {
  if constexpr (IsMappedDir<SDIR>()) {
    Real n1, n2, n3;
    coords.template FaceNormal<SDIR>(k, j, i, n1, n2, n3);
    nn = n1 * cp(aa, NDIR[X1DIR - 1]) + n2 * cp(aa, NDIR[X2DIR - 1]);
    if constexpr (IsMapped3D()) nn += n3 * cp(aa, NDIR[X3DIR - 1]);
    up = (nn >= 0.0);
    down = (nn < 0.0);
    sgnn = up - down;
  }
}
template <typename Coords, typename CP>
KOKKOS_FORCEINLINE_FUNCTION void
FaceDirection(const int dir, const Coords &coords, const CP &cp,
              const std::array<int, 3> &NDIR, const int aa, const int k, const int j,
              const int i, Real &nn, bool &up, bool &down, int &sgnn) {
  if (dir == X1DIR) {
    FaceDirection<X1DIR>(coords, cp, NDIR, aa, k, j, i, nn, up, down, sgnn);
  } else if (dir == X2DIR) {
    FaceDirection<X2DIR>(coords, cp, NDIR, aa, k, j, i, nn, up, down, sgnn);
  } else if (dir == X3DIR) {
    FaceDirection<X3DIR>(coords, cp, NDIR, aa, k, j, i, nn, up, down, sgnn);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void MinCellWidth
//! \brief Hook in the Jacobi timestep: mapped cells vary within a block and the
//!        (0, 0, 0) ghost may lie outside the patch, so min_dx is replaced by the minimum
//!        over the interior cells of 1 / sum_d (1 / CellWidth_d).
void MinCellWidth(parthenon::MeshData<Real> *md, Real &min_dx);

} // namespace RiotForest

#endif // FOREST_MAPPED_RADIATION_HPP_
