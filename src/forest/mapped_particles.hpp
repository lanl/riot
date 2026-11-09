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
#ifndef FOREST_MAPPED_PARTICLES_HPP_
#define FOREST_MAPPED_PARTICLES_HPP_
// This file was made in part with generative AI.

#include <cmath>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/meshblock.hpp>

namespace RiotForest {

using parthenon::Real;

// Particles keep physical positions on mapped meshes too, but cells there are uniform
// in the logical coordinates of the tree of their block. Fields are reconstructed at a
// particle in that frame: displacements from cell centers and the distances between
// them (Dxc, which is logical on mapped meshes) are both logical.

//----------------------------------------------------------------------------------------
//! \fn  void ParticleFrame
//! \brief The position (x1, x2, x3) in the frame cells are indexed in: logical in the
//!        tree of the block of coords on mapped meshes, the position itself otherwise
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void ParticleFrame(const Coords &coords, const Real x1,
                                               const Real x2, const Real x3, Real l[3]) {
  if constexpr (parthenon::IsMappedCoord()) {
    coords.PhysicalToLogical(x1, x2, x3, l[0], l[1], l[2]);
  } else {
    l[0] = x1;
    l[1] = x2;
    l[2] = x3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real FrameXc
//! \brief Center of cell (k, j, i) along DIR in the frame of ParticleFrame
template <int DIR, typename Coords>
KOKKOS_FORCEINLINE_FUNCTION Real FrameXc(const Coords &coords, const int k, const int j,
                                         const int i) {
  if constexpr (parthenon::IsMappedCoord()) {
    return coords.template Xc<DIR>(DIR == parthenon::X1DIR   ? i
                                   : DIR == parthenon::X2DIR ? j
                                                             : k);
  } else {
    return coords.template Xc<DIR>(k, j, i);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  bool HoldsSeed
//! \brief Whether block pmb holds the particle seeded at x, so that each particle is
//!        seeded in one block. Blocks own their lower faces. On the faces between trees
//!        the logical coordinates of the trees disagree by roundoff, so there the tree
//!        with the lower id owns the face (and at an edge or node, the lowest of them).
inline bool HoldsSeed(const parthenon::MeshBlock *pmb, const Real x[3]) {
  const auto &reg = pmb->block_size;
  Real l[3];
  ParticleFrame(pmb->coords, x[0], x[1], x[2], l);
  for (int d = 0; d < pmb->pmy_mesh->ndim; d++) {
    const Real lo = reg.xmin_[d], hi = reg.xmax_[d];
    if constexpr (parthenon::IsMappedCoord()) {
      const Real tol = 1.0e-10 * (hi - lo);
      const int side = (std::abs(l[d] - lo) <= tol) ? -1 : (std::abs(l[d] - hi) <= tol);
      if (side != 0) {
        int tree = pmb->loc.tree();
        for (const auto &nb : pmb->GetNeighbors()) {
          bool face = true;
          for (int e = 0; e < 3; e++) {
            face = face && static_cast<int>(nb.offsets[e]) == ((e == d) ? side : 0);
          }
          if (face) tree = nb.loc.tree();
        }
        if (tree != pmb->loc.tree()) {
          if (tree < pmb->loc.tree()) return false;
          continue;
        }
      }
    }
    if (!(l[d] >= lo && l[d] < hi)) return false;
  }
  return true;
}

} // namespace RiotForest

#endif // FOREST_MAPPED_PARTICLES_HPP_
