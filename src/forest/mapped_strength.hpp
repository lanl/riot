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
#ifndef FOREST_MAPPED_STRENGTH_HPP_
#define FOREST_MAPPED_STRENGTH_HPP_
// This file was made in part with generative AI.
//
// Material strength on mapped (forest) meshes. The hooks here are no-ops unless the
// coordinates are mapped. The deviatoric stress and the velocity are in the global basis
// on mapped meshes, and the stored face velocities have been rotated to it by
// FluxesToGlobalFrame, so
//   - the velocity gradient is the physical (Green-Gauss) one of the face velocities,
//   - the stress is rotated into the frame of each face for the strength Riemann solver,
//     whose velocities ReconVelocityToFaceFrame has left in that frame.

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>

#include "forest/mapped_geometry.hpp"
#include "variables.hpp"

namespace RiotForest {

//----------------------------------------------------------------------------------------
//! \brief Velocity gradient g[a][b] = d v_b / d x_a of mapped cell (k, j, i), and its
//!        divergence div (the trace of the cell's volume change, sum_f (v . n) A / V). In
//!        3D it is the Green-Gauss gradient sum_f v (n A) / V. In (R, z) the meridional
//!        part is the planar Green-Gauss gradient over the polygon of the cell's vertices
//!        (the uniform-mesh differences on rectilinear cells), phi derivatives vanish,
//!        and the hoop rate v_R / R is div less the planar divergence, so the trace of g
//!        is div exactly.
template <typename Coords, typename PackView>
KOKKOS_INLINE_FUNCTION void
MappedVelocityGradient(const Coords &coords, const PackView &pv, const int k, const int j,
                       const int i, Real g[3][3], Real &div) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int nd = FrameDims();
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      g[a][b] = 0.0;
  div = 0.0;
  // the lower (s = 0) and upper (s = 1) faces of each meridional direction
  for (int d = 0; d < nd; ++d) {
    for (int s = 0; s < 2; ++s) {
      const int fi = i + s * (d == 0), fj = j + s * (d == 1), fk = k + s * (d == 2);
      const Real sgn = s ? 1.0 : -1.0;
      Real n[3];
      Real area;
      if (d == 0) {
        coords.template FaceNormal<X1DIR>(fk, fj, fi, n[0], n[1], n[2]);
        area = coords.template FaceArea<X1DIR>(fk, fj, fi);
      } else if (d == 1) {
        coords.template FaceNormal<X2DIR>(fk, fj, fi, n[0], n[1], n[2]);
        area = coords.template FaceArea<X2DIR>(fk, fj, fi);
      } else {
        coords.template FaceNormal<X3DIR>(fk, fj, fi, n[0], n[1], n[2]);
        area = coords.template FaceArea<X3DIR>(fk, fj, fi);
      }
      Real vf[3];
      for (int b = 0; b < 3; ++b)
        vf[b] = pv(ccbulk::face_velocity(3 * d + b), fk, fj, fi);
      for (int c = 0; c < nd; ++c)
        div += sgn * n[c] * vf[c] * area;
      // the (outward) planar face vector of the face
      Real sa[3] = {0.0, 0.0, 0.0};
      if constexpr (IsMapped3D()) {
        for (int a = 0; a < 3; ++a)
          sa[a] = sgn * n[a] * area;
      } else {
        // int n dl along the face is the chord turned by -90 degrees (by +90 on mirrored
        // patches), as in FaceNormal
        Real ra, za, rb, zb;
        coords.Vertex(fj, fi, ra, za);
        coords.Vertex(fj + (d == 0), fi + (d == 1), rb, zb);
        const Real o = sgn * coords.Orientation() * ((d == 0) ? 1.0 : -1.0);
        sa[0] = o * (zb - za);
        sa[1] = -o * (rb - ra);
      }
      for (int a = 0; a < nd; ++a)
        for (int b = 0; b < 3; ++b)
          g[a][b] += sa[a] * vf[b];
    }
  }
  div /= coords.CellVolume(k, j, i);
  if constexpr (IsMapped3D()) {
    const Real ivol = 1.0 / coords.CellVolume(k, j, i);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        g[a][b] *= ivol;
  } else {
    // area of the polygon of the cell's vertices
    Real r[4], z[4];
    coords.Vertex(j, i, r[0], z[0]);
    coords.Vertex(j, i + 1, r[1], z[1]);
    coords.Vertex(j + 1, i + 1, r[2], z[2]);
    coords.Vertex(j + 1, i, r[3], z[3]);
    Real area = 0.0;
    for (int v = 0; v < 4; ++v)
      area += r[v] * z[(v + 1) % 4] - r[(v + 1) % 4] * z[v];
    const Real iarea = 2.0 / (coords.Orientation() * area);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 3; ++b)
        g[a][b] *= iarea;
    g[2][2] = div - g[0][0] - g[1][1];
  }
}

//----------------------------------------------------------------------------------------
//! \brief Hook in Hydro::CalculateStrainRate: on mapped meshes the deviatoric strain
//!        rate of (k, j, i) is that of MappedVelocityGradient.
template <typename Coords, typename PackView>
KOKKOS_INLINE_FUNCTION void MappedStrainRate(const Coords &coords, const PackView &pv,
                                             const int k, const int j, const int i) {
  if constexpr (parthenon::IsMappedCoord()) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    Real g[3][3], div;
    MappedVelocityGradient(coords, pv, k, j, i, g, div);
    const Real vdiv = div / 3.0;
    pv(ccbulk::strain_rate(0), k, j, i) = g[0][0] - vdiv;
    pv(ccbulk::strain_rate(1), k, j, i) = 0.5 * (g[1][0] + g[0][1]);
    pv(ccbulk::strain_rate(2), k, j, i) = 0.5 * (g[2][0] + g[0][2]);
    pv(ccbulk::strain_rate(3), k, j, i) = g[1][1] - vdiv;
    pv(ccbulk::strain_rate(4), k, j, i) = 0.5 * (g[2][1] + g[1][2]);
    pv(ccbulk::strain_rate(5), k, j, i) = g[2][2] - vdiv;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Hook in Strength::CalculateStrengthSource: on mapped meshes the rotation
//!        tensor (wxy, wxz, wyz) is that of MappedVelocityGradient.
template <typename Coords, typename PackView>
KOKKOS_INLINE_FUNCTION void RotationTensor(const Coords &coords, const PackView &pv,
                                           const int k, const int j, const int i,
                                           Real &wxy, Real &wxz, Real &wyz) {
  if constexpr (parthenon::IsMappedCoord()) {
    Real g[3][3], div;
    MappedVelocityGradient(coords, pv, k, j, i, g, div);
    wxy = 0.5 * (g[1][0] - g[0][1]);
    wxz = 0.5 * (g[2][0] - g[0][2]);
    wyz = 0.5 * (g[2][1] - g[1][2]);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Hook in the strength Riemann solve: rotates the deviatoric stress
//!        (xx, xy, xz, yy, yz; zz = -xx - yy) of a face state into the frame of the DIR
//!        face at the lower side of (k, j, i), sigma' = M sigma M^T with M the rows of
//!        the frame. The rotation keeps the trace, so zz' = -xx' - yy' still holds.
template <parthenon::CoordinateDirection DIR, typename Coords>
KOKKOS_INLINE_FUNCTION void StressToFaceFrame(const Coords &coords, const int k,
                                              const int j, const int i, Real st[5]) {
  if constexpr (IsMappedDir<DIR>()) {
    const FaceFrame f = MakeFaceFrame<DIR>(coords, k, j, i);
    const Real S[3][3] = {
        {st[0], st[1], st[2]}, {st[1], st[3], st[4]}, {st[2], st[4], -st[0] - st[3]}};
    // rows of M; in (R, z) the frame turns only the meridional components
    Real M[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int s = 0; s < FrameDims(); ++s)
      for (int d = 0; d < FrameDims(); ++d)
        M[s][d] = f.m[s][d];
    Real T[3][3];
    for (int s = 0; s < 3; ++s)
      for (int t = 0; t < 3; ++t) {
        T[s][t] = 0.0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            T[s][t] += M[s][a] * S[a][b] * M[t][b];
      }
    st[0] = T[0][0];
    st[1] = T[0][1];
    st[2] = T[0][2];
    st[3] = T[1][1];
    st[4] = T[1][2];
  }
}

} // namespace RiotForest

#endif // FOREST_MAPPED_STRENGTH_HPP_
