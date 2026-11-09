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
#ifndef FOREST_MAPPED_RAYS_HPP_
#define FOREST_MAPPED_RAYS_HPP_
// This file was made in part with generative AI.

#include <cmath>
#include <limits>
#include <type_traits>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>

namespace RiotForest {

using parthenon::Real;

// Rays (camera rays and laser rays) are straight lines in Cartesian space. On mapped
// meshes the cells they cross are curved, but uniform in the logical coordinates of the
// tree of their block, so a ray is followed in that frame: where it is in its cell, and
// where it leaves it, come from the inverse map of the tree. Lukewarm particles keep the
// physical position, (R, z, phi) on cylindrical meshes and (x, y, z) on Cartesian ones.

// Relative to the block width, how far outside a block a ray is still on it. This is
// twice as far as LukewarmGeometry still locates a position in a block (its
// face_tolerance_), so that a ray that is done with a block is never located in it again,
// and one located in a block is on it.
constexpr Real kOnBlockTolerance = 2.0e-10;
// Relative to the cell width: where a ray leaves its cell is found to within
// kExitTolerance, and a ray that ends a step within kSnapBand of a face of its cell is
// moved kSnapNudge to the side of the face it is heading to
constexpr Real kExitTolerance = 1.0e-8;
constexpr Real kSnapBand = 1.0e-6;
constexpr Real kSnapNudge = 2.0e-6;

constexpr bool IsMappedCylindrical() {
  return parthenon::IsCoord<parthenon::MappedCylindrical>();
}
// The helpers on coords branch on its type, so they compile, and do nothing, for others
template <typename Coords>
constexpr bool kCylindrical = std::is_same_v<Coords, parthenon::MappedCylindrical>;
template <typename Coords>
constexpr bool kCartesian = std::is_same_v<Coords, parthenon::MappedCartesian>;

//----------------------------------------------------------------------------------------
//! \fn  void CartToMesh
//! \brief The physical mesh coordinates of the Cartesian point x, as lukewarm particles
//!        hold them: (R, z, phi), with phi in [0, 2 pi), on cylindrical meshes, and x
//!        itself on Cartesian ones
KOKKOS_FORCEINLINE_FUNCTION void CartToMesh(const Real x[3], Real m[3]) {
  if constexpr (IsMappedCylindrical()) {
    m[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    m[1] = x[2];
    m[2] = std::atan2(x[1], x[0]);
    m[2] += (m[2] < 0.0) * 2.0 * M_PI;
  } else {
    m[0] = x[0];
    m[1] = x[1];
    m[2] = x[2];
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshToCart
//! \brief The Cartesian point of the physical mesh coordinates m, the inverse of
//!        CartToMesh
KOKKOS_FORCEINLINE_FUNCTION void MeshToCart(const Real m[3], Real x[3]) {
  if constexpr (IsMappedCylindrical()) {
    x[0] = m[0] * std::cos(m[2]);
    x[1] = m[0] * std::sin(m[2]);
    x[2] = m[1];
  } else {
    x[0] = m[0];
    x[1] = m[1];
    x[2] = m[2];
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void CartToLogical
//! \brief The logical coordinates l of the Cartesian point x in the tree of coords. On
//!        cylindrical meshes, which are 2D, the azimuth is left at zero unless full.
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void CartToLogical(const Coords &coords, const Real x[3],
                                               Real l[3], const bool full = false) {
  if constexpr (kCylindrical<Coords>) {
    const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    Real phi = 0.0;
    if (full) {
      phi = std::atan2(x[1], x[0]);
      phi += (phi < 0.0) * 2.0 * M_PI;
    }
    // the inverse map of a polar patch, with the angle found directly rather than from
    // the middle of the patch, on the branch nearest to it
    const auto &p = coords.Patch();
    if (p.type == parthenon::MappedPatch::Polar) {
      const Real thm = 0.5 * (p.th0 + p.th1);
      Real th = std::atan2(x[2], r);
      th += 2.0 * M_PI * std::round((thm - th) / (2.0 * M_PI));
      const Real rr = std::sqrt(r * r + x[2] * x[2]);
      l[0] = (th - p.th0) / (p.th1 - p.th0);
      l[1] = p.rlog ? std::log(rr / p.r0) / std::log(p.r1 / p.r0)
                    : (rr - p.r0) / (p.r1 - p.r0);
      l[2] = phi;
      return;
    }
    coords.PhysicalToLogical(r, x[2], phi, l[0], l[1], l[2]);
  } else if constexpr (kCartesian<Coords>) {
    coords.PhysicalToLogical(x[0], x[1], x[2], l[0], l[1], l[2]);
  }
}

// The largest Newton step (in the logical coordinates of the tree) after which the
// inverse map of a Coons patch started near the point is taken to be done, as it is
// from the middle (whose last step is below 1e-14) to round-off, as a looser one moves
// pixels where layers meet
constexpr Real kNearNewtonStep = 1.0e-12;
constexpr int kNearNewtonSteps = 50;

//----------------------------------------------------------------------------------------
//! \fn  void CartToLogicalNear
//! \brief CartToLogical, with the inverse map of Coons patches, which is Newton's method
//!        from the middle of the tree, started instead from the logical point near (if
//!        not null), which is close to l, so that it takes a step or two rather than many
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void CartToLogicalNear(const Coords &coords, const Real x[3],
                                                   Real l[3], const Real *near) {
  if constexpr (kCylindrical<Coords>) {
    const auto &p = coords.Patch();
    if (near != nullptr && p.type == parthenon::MappedPatch::Coons) {
      const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      Real xi = near[0], eta = near[1];
      for (int it = 0; it < kNearNewtonSteps; ++it) {
        Real px, py, a, b, c, d;
        p.Map(xi, eta, px, py);
        p.Jacobian(xi, eta, a, b, c, d);
        const Real fx = px - r, fy = py - x[2], det = a * d - b * c;
        Real dxi = (d * fx - b * fy) / det, deta = (a * fy - c * fx) / det;
        const Real step = std::max(std::abs(dxi), std::abs(deta));
        // not a number, so start over from the middle
        if (!(step == step)) break;
        // limited far from the point, as for a ray that jumped across a block, as is the
        // inverse map from the middle
        if (step > 0.5) {
          dxi *= 0.5 / step;
          deta *= 0.5 / step;
        }
        xi -= dxi;
        eta -= deta;
        if (step < kNearNewtonStep) {
          l[0] = xi;
          l[1] = eta;
          l[2] = 0.0;
          return;
        }
      }
    }
  }
  CartToLogical(coords, x, l);
}

//----------------------------------------------------------------------------------------
//! \fn  bool FarFromBlock
//! \brief Whether the Cartesian point x is surely not on the block [lo, hi] of the tree
//!        of coords, as it is more than three times as far from the middle of the block
//!        as its farthest corner, which is farther than its curved faces bulge. Only for
//!        Coons patches of cylindrical meshes, whose inverse map is slow to find that far
//!        from where it starts; false otherwise.
template <typename Coords>
KOKKOS_INLINE_FUNCTION bool FarFromBlock(const Coords &coords, const Real x[3],
                                         const Real lo[3], const Real hi[3]) {
  if constexpr (kCylindrical<Coords>) {
    const auto &p = coords.Patch();
    if (p.type != parthenon::MappedPatch::Coons) return false;
    Real rc, zc;
    p.Map(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), rc, zc);
    Real rho2 = 0.0;
    for (int n = 0; n < 4; ++n) {
      Real rn, zn;
      p.Map(n & 1 ? hi[0] : lo[0], n & 2 ? hi[1] : lo[1], rn, zn);
      rho2 = std::max(rho2, (rn - rc) * (rn - rc) + (zn - zc) * (zn - zc));
    }
    const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    return (r - rc) * (r - rc) + (x[2] - zc) * (x[2] - zc) > 9.0 * rho2;
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn  void LogicalToCart
//! \brief The Cartesian point x of the logical l in the tree of coords. On cylindrical
//!        meshes it is at the azimuth of the Cartesian point xref, which may be x.
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void LogicalToCart(const Coords &coords, const Real l[3],
                                               const Real xref[3], Real x[3]) {
  if constexpr (kCylindrical<Coords>) {
    Real r, z;
    coords.Map(l[0], l[1], r, z);
    const Real rref = std::sqrt(xref[0] * xref[0] + xref[1] * xref[1]);
    const Real c = rref > 0.0 ? xref[0] / rref : 1.0;
    const Real s = rref > 0.0 ? xref[1] / rref : 0.0;
    x[0] = r * c;
    x[1] = r * s;
    x[2] = z;
  } else if constexpr (kCartesian<Coords>) {
    coords.Map(l[0], l[1], l[2], x[0], x[1], x[2]);
  }
}

namespace impl {
// The Jacobian J[i][d] = d m_i / d l_d of the map from logical l to physical mesh
// coordinates m, (R, z) only on cylindrical meshes, where J[2] is left as the identity
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void Jacobian(const Coords &coords, const Real l[3],
                                          Real J[3][3]) {
  if constexpr (kCylindrical<Coords>) {
    coords.Patch().Jacobian(l[0], l[1], J[0][0], J[0][1], J[1][0], J[1][1]);
    J[0][2] = J[1][2] = J[2][0] = J[2][1] = 0.0;
    J[2][2] = 1.0;
  } else if constexpr (kCartesian<Coords>) {
    coords.Patch().Jacobian(l[0], l[1], l[2], J);
  }
}

// The inverse of the 3x3 matrix J
KOKKOS_FORCEINLINE_FUNCTION void Invert(const Real J[3][3], Real Ji[3][3]) {
  Ji[0][0] = J[1][1] * J[2][2] - J[1][2] * J[2][1];
  Ji[0][1] = J[0][2] * J[2][1] - J[0][1] * J[2][2];
  Ji[0][2] = J[0][1] * J[1][2] - J[0][2] * J[1][1];
  Ji[1][0] = J[1][2] * J[2][0] - J[1][0] * J[2][2];
  Ji[1][1] = J[0][0] * J[2][2] - J[0][2] * J[2][0];
  Ji[1][2] = J[0][2] * J[1][0] - J[0][0] * J[1][2];
  Ji[2][0] = J[1][0] * J[2][1] - J[1][1] * J[2][0];
  Ji[2][1] = J[0][1] * J[2][0] - J[0][0] * J[2][1];
  Ji[2][2] = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  const Real det = J[0][0] * Ji[0][0] + J[0][1] * Ji[1][0] + J[0][2] * Ji[2][0];
  for (int i = 0; i < 3; ++i) {
    for (int d = 0; d < 3; ++d)
      Ji[i][d] /= det;
  }
}
} // namespace impl

//----------------------------------------------------------------------------------------
//! \fn  void LogicalVelocity
//! \brief The rate w = dl/dh at which the logical coordinates l of the Cartesian point x
//!        change along the Cartesian direction v. The azimuthal rate on cylindrical
//!        meshes is left at zero, as they are 2D.
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void LogicalVelocity(const Coords &coords, const Real x[3],
                                                 const Real l[3], const Real v[3],
                                                 Real w[3]) {
  Real vm[3] = {v[0], v[1], v[2]};
  if constexpr (kCylindrical<Coords>) {
    const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    // on the axis a ray can only move away from it
    vm[0] =
        r > 0.0 ? (x[0] * v[0] + x[1] * v[1]) / r : std::sqrt(v[0] * v[0] + v[1] * v[1]);
    vm[1] = v[2];
    vm[2] = 0.0;
    // the rates of the angle and the radius of polar patches, without their Jacobian
    const auto &p = coords.Patch();
    if (p.type == parthenon::MappedPatch::Polar) {
      const Real r2 = r * r + x[2] * x[2];
      w[0] = (r * vm[1] - x[2] * vm[0]) / (r2 * (p.th1 - p.th0));
      w[1] = (r * vm[0] + x[2] * vm[1]) /
             (p.rlog ? r2 * std::log(p.r1 / p.r0) : std::sqrt(r2) * (p.r1 - p.r0));
      w[2] = 0.0;
      return;
    }
  }
  Real J[3][3], Ji[3][3];
  impl::Jacobian(coords, l, J);
  impl::Invert(J, Ji);
  for (int d = 0; d < 3; ++d)
    w[d] = Ji[d][0] * vm[0] + Ji[d][1] * vm[1] + Ji[d][2] * vm[2];
}

//----------------------------------------------------------------------------------------
//! \fn  void GradientToCart
//! \brief The Cartesian gradient g of a field at the Cartesian point x, at logical l,
//!        from its gradient gl in the logical coordinates
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION void GradientToCart(const Coords &coords, const Real x[3],
                                                const Real l[3], const Real gl[3],
                                                Real g[3]) {
  // the gradient in the mesh coordinates is J^-T gl
  Real J[3][3], Ji[3][3], gm[3];
  impl::Jacobian(coords, l, J);
  impl::Invert(J, Ji);
  for (int d = 0; d < 3; ++d)
    gm[d] = Ji[0][d] * gl[0] + Ji[1][d] * gl[1] + Ji[2][d] * gl[2];
  if constexpr (kCylindrical<Coords>) {
    // the field does not vary in phi
    const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const Real c = r > 0.0 ? x[0] / r : 1.0;
    const Real s = r > 0.0 ? x[1] / r : 0.0;
    g[0] = gm[0] * c;
    g[1] = gm[0] * s;
    g[2] = gm[1];
  } else {
    g[0] = gm[0];
    g[1] = gm[1];
    g[2] = gm[2];
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real PhysicalWidth
//! \brief An estimate of the smallest physical width, over the nd active directions, of
//!        the cell at logical l with logical widths dl
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION Real PhysicalWidth(const Coords &coords, const int nd,
                                               const Real l[3], const Real dl[3]) {
  Real J[3][3];
  impl::Jacobian(coords, l, J);
  Real width = std::numeric_limits<Real>::max();
  for (int d = 0; d < nd; ++d) {
    const Real len =
        std::sqrt(J[0][d] * J[0][d] + J[1][d] * J[1][d] + J[2][d] * J[2][d]) * dl[d];
    width = len < width ? len : width;
  }
  return width;
}

namespace impl {
// The smallest positive root, where valid, of a h^2 + 2 b h + c = 0, given its
// discriminant d4 = b^2 - a c, or the largest Real if there is none
template <typename Valid>
KOKKOS_FORCEINLINE_FUNCTION Real SmallestRoot(const Real a, const Real b, const Real c,
                                              const Real d4, const Valid &valid) {
  constexpr Real big = std::numeric_limits<Real>::max();
  Real h = big;
  if (d4 < 0.0) return h;
  if (a == 0.0) {
    if (b != 0.0) {
      const Real t = -0.5 * c / b;
      if (t > 0.0 && valid(t)) h = t;
    }
    return h;
  }
  const Real q = -(b + std::copysign(std::sqrt(d4), b));
  const Real t[2] = {q / a, q != 0.0 ? c / q : big};
  for (int n = 0; n < 2; ++n) {
    if (t[n] > 0.0 && t[n] < h && valid(t[n])) h = t[n];
  }
  return h;
}

// Where the ray x + h v first meets the faces of the logical cell [lo, hi] of a polar
// patch of a cylindrical mesh: spheres |x| = r about the origin for the radial
// coordinate, and cones z cos(th) = R sin(th) about the axis for the angular one, each
// the root of a quadratic in h. Gives the distance hf to the face it meets and the
// rate |dl/dh|, in cell widths, of its logical coordinate there, or false if it meets
// none.
KOKKOS_INLINE_FUNCTION bool PolarFaces(const parthenon::MappedPatch &p, const Real x[3],
                                       const Real v[3], const Real lo[3],
                                       const Real hi[3], Real &hf, Real &rate) {
  constexpr Real big = std::numeric_limits<Real>::max();
  hf = big;
  Real hd[2][2] = {{big, big}, {big, big}};
  // radial faces
  const Real vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const Real xv = x[0] * v[0] + x[1] * v[1] + x[2] * v[2];
  const Real xx = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  for (int side = 0; side < 2; ++side) {
    const Real eta = side ? hi[1] : lo[1];
    const Real rf =
        p.rlog ? p.r0 * std::pow(p.r1 / p.r0, eta) : p.r0 + eta * (p.r1 - p.r0);
    const Real c = xx - rf * rf;
    hd[1][side] = SmallestRoot(vv, xv, c, xv * xv - vv * c, [](Real) { return true; });
  }
  // angular faces, also met by the roots on the cone reflected in z = 0, so a root is
  // kept only where z has the sign of sin(th)
  const Real ww = v[0] * v[0] + v[1] * v[1];
  const Real pw = x[0] * v[0] + x[1] * v[1];
  const Real qq = x[0] * x[0] + x[1] * x[1];
  const Real mx = v[2] * x[0] - x[2] * v[0], my = v[2] * x[1] - x[2] * v[1];
  const Real m2 = mx * mx + my * my;
  const Real l2 = (x[0] * v[1] - x[1] * v[0]) * (x[0] * v[1] - x[1] * v[0]);
  for (int side = 0; side < 2; ++side) {
    const Real th = p.th0 + (side ? hi[0] : lo[0]) * (p.th1 - p.th0);
    const Real s = std::sin(th), c = std::cos(th);
    const Real a = c * c * v[2] * v[2] - s * s * ww;
    const Real b = c * c * x[2] * v[2] - s * s * pw;
    const Real cc = c * c * x[2] * x[2] - s * s * qq;
    // b^2 - a cc, without the cancellation of its c^4 terms
    const Real d4 = s * s * (c * c * m2 - s * s * l2);
    hd[0][side] = SmallestRoot(
        a, b, cc, d4, [&](const Real t) { return (x[2] + t * v[2]) * s >= 0.0; });
  }
  int dir = -1;
  for (int d = 0; d < 2; ++d) {
    for (int side = 0; side < 2; ++side) {
      if (hd[d][side] < hf) {
        hf = hd[d][side];
        dir = d;
      }
    }
  }
  if (dir < 0) return false;
  const Real xf[3] = {x[0] + hf * v[0], x[1] + hf * v[1], x[2] + hf * v[2]};
  const Real dl = hi[dir] - lo[dir];
  if (dir == 1) {
    const Real r = std::sqrt(xf[0] * xf[0] + xf[1] * xf[1] + xf[2] * xf[2]);
    const Real dr = (xf[0] * v[0] + xf[1] * v[1] + xf[2] * v[2]) / r;
    rate = dr / (p.rlog ? r * std::log(p.r1 / p.r0) : p.r1 - p.r0);
  } else {
    const Real rc = std::sqrt(xf[0] * xf[0] + xf[1] * xf[1]);
    const Real drc = rc > 0.0 ? (xf[0] * v[0] + xf[1] * v[1]) / rc : 0.0;
    rate = (rc * v[2] - xf[2] * drc) / (rc * rc + xf[2] * xf[2]) / (p.th1 - p.th0);
  }
  rate = std::abs(rate) / dl;
  return rate > 0.0;
}

// Where the ray x + h v first meets those faces of the logical cell [lo, hi] of a Coons
// patch of a cylindrical mesh that are straight in (R, z): lines nr R + nz z = c, which
// are cones (or cylinders or planes) about the axis, each met at a root of a quadratic
// in h. The lines of constant xi are straight where the edges xi = 0 and xi = 1 are
// uniform lines, as the patch is then (1 - eta) B(xi) + eta T(xi) of its other two
// edges, and likewise for eta. Gives the distance hf to the face it meets and the
// direction dir of its logical coordinate, or false if it meets none, and along which
// directions the faces are straight.
KOKKOS_INLINE_FUNCTION bool CoonsFaces(const parthenon::MappedPatch &p, const Real x[3],
                                       const Real v[3], const Real lo[3],
                                       const Real hi[3], Real &hf, int &dir,
                                       bool straight[2]) {
  using MP = parthenon::MappedPatch;
  hf = std::numeric_limits<Real>::max();
  dir = -1;
  straight[0] = p.edge[2] == MP::Line && p.edge[3] == MP::Line;
  straight[1] = p.edge[0] == MP::Line && p.edge[1] == MP::Line;
  const Real ww = v[0] * v[0] + v[1] * v[1];
  const Real pw = x[0] * v[0] + x[1] * v[1];
  const Real qq = x[0] * x[0] + x[1] * x[1];
  const Real l2 = (x[0] * v[1] - x[1] * v[0]) * (x[0] * v[1] - x[1] * v[0]);
  for (int d = 0; d < 2; ++d) {
    if (!straight[d]) continue;
    for (int side = 0; side < 2; ++side) {
      // the face runs between its points on the two edges across it
      const Real f = side ? hi[d] : lo[d];
      Real ar, az, br, bz;
      p.EdgePoint(d == 0 ? 0 : 2, f, ar, az);
      p.EdgePoint(d == 0 ? 1 : 3, f, br, bz);
      const Real nr = az - bz, nz = br - ar;
      // nr R = u, with u = u0 + du h along the ray, squared
      const Real u0 = nr * ar + nz * (az - x[2]), du = -nz * v[2];
      const Real n2 = nr * nr;
      const Real a = n2 * ww - du * du;
      const Real b = n2 * pw - u0 * du;
      const Real cc = n2 * qq - u0 * u0;
      // b^2 - a cc, without the cancellation of its u0^2 du^2 terms
      const Real mx = u0 * v[0] - du * x[0], my = u0 * v[1] - du * x[1];
      const Real d4 = n2 * (mx * mx + my * my - n2 * l2);
      // and kept only where R = u / nr is not negative, off the reflected cone
      const Real h = SmallestRoot(
          a, b, cc, d4, [&](const Real t) { return (u0 + t * du) * nr >= 0.0; });
      if (h < hf) {
        hf = h;
        dir = d;
      }
    }
  }
  return dir >= 0;
}

// Where the ray x + h v meets the face of constant logical coordinate f along d of a
// Coons patch of a cylindrical mesh where that is curved in (R, z): the face is the
// curve Map at f and s along the other direction, and the ray is R^2 = |x + h v|^2 off
// the axis at z = x[2] + h v[2], so Newton's method finds s and h together, from where
// the ray at l with logical velocity w would meet it were the map linear. Gives the
// distance h, the point s and the rate |dl/dh|, in cell widths, of the logical
// coordinate d there, or false if the ray is not heading to the face or does not
// converge on it ahead within the cell [lo, hi].
KOKKOS_INLINE_FUNCTION bool
CoonsCurvedFace(const parthenon::MappedPatch &p, const Real x[3], const Real v[3],
                const Real l[3], const Real w[3], const Real lo[3], const Real hi[3],
                const int d, const Real f, Real &h, Real &s, Real &rate) {
  const int o = 1 - d;
  if (!((f - l[d]) * w[d] > 0.0)) return false;
  const Real wd = hi[d] - lo[d], wo = hi[o] - lo[o];
  const Real ww = v[0] * v[0] + v[1] * v[1];
  const Real pw = x[0] * v[0] + x[1] * v[1];
  const Real qq = x[0] * x[0] + x[1] * x[1];
  h = (f - l[d]) / w[d];
  s = l[o] + h * w[o];
  for (int it = 0; it < 8; ++it) {
    const Real xi = d == 0 ? f : s, eta = d == 0 ? s : f;
    Real cr, cz, r_xi, r_eta, z_xi, z_eta;
    p.Map(xi, eta, cr, cz);
    p.Jacobian(xi, eta, r_xi, r_eta, z_xi, z_eta);
    if (!(cr > 0.0)) return false;
    const Real rs = d == 0 ? r_eta : r_xi, zs = d == 0 ? z_eta : z_xi;
    const Real dq = pw + ww * h;
    const Real f1 = cr * cr - (qq + h * (2.0 * pw + ww * h)), f2 = cz - x[2] - v[2] * h;
    const Real j11 = 2.0 * cr * rs, j12 = -2.0 * dq, j21 = zs, j22 = -v[2];
    const Real det = j11 * j22 - j12 * j21;
    const Real ds = (j22 * f1 - j12 * f2) / det, dh = (j11 * f2 - j21 * f1) / det;
    s -= ds;
    h -= dh;
    // not a number, or off the cell by far, is not converging on it
    if (!(std::abs(ds) < wo)) return false;
    // the logical velocity of the ray there, from the Jacobian of the map
    const Real dr = dq / cr, jd = r_xi * z_eta - r_eta * z_xi;
    const Real wf =
        d == 0 ? (z_eta * dr - r_eta * v[2]) / jd : (r_xi * v[2] - z_xi * dr) / jd;
    rate = std::abs(wf) / wd;
    if (std::abs(dh) * rate < 1.0e-3 * kExitTolerance &&
        std::abs(ds) < 1.0e-3 * kExitTolerance * wo) {
      return h > 0.0 && rate > 0.0 && s >= lo[o] - 1.0e-2 * wo &&
             s <= hi[o] + 1.0e-2 * wo;
    }
  }
  return false;
}
} // namespace impl

//----------------------------------------------------------------------------------------
//! \fn  Real CellExit
//! \brief The distance h along the ray x + h v, in units of |v|, to where it leaves the
//!        logical cell [lo, hi] along the nd active directions: at most kExitTolerance
//!        of the cell width past the face, and never short of it, so that a slice on the
//!        face is met in this step. A step may also stop short, as below. The largest
//!        Real if it never leaves.
//!
//! The faces of curved cells can be crossed twice by a straight ray, as the radial ones
//! are around its closest approach to the origin (and to the axis on cylindrical
//! meshes), so a step stops there. From the start, Newton steps on the nearest face
//! (aimed just past it) move the ray on until it is outside the cell, after which false
//! position (with bisection to make sure it converges) closes in on where it left.
template <typename Coords>
KOKKOS_INLINE_FUNCTION Real CellExit(const Coords &coords, const int nd, const Real x[3],
                                     const Real v[3], const Real lo[3], const Real hi[3],
                                     const Real *near = nullptr) {
  constexpr Real big = std::numeric_limits<Real>::max();
  // how far outside the cell the logical point l is, in cell widths, negative inside
  auto outside = [&](const Real l[3]) {
    Real g = -big;
    for (int d = 0; d < nd; ++d) {
      const Real e = std::max(lo[d] - l[d], l[d] - hi[d]) / (hi[d] - lo[d]);
      g = e > g ? e : g;
    }
    // a failed inverse map is outside
    return g == g ? g : big;
  };
  // the distance on from the point xp, at logical lp, to just past the faces, were the
  // map linear, keeping the logical velocity w there to guess the logical point further
  // on from
  Real hw = 0.0, lw[3] = {0.0, 0.0, 0.0}, w[3] = {0.0, 0.0, 0.0};
  auto linear = [&](const Real hp, const Real xp[3], const Real lp[3]) {
    LogicalVelocity(coords, xp, lp, v, w);
    hw = hp;
    for (int d = 0; d < 3; ++d)
      lw[d] = lp[d];
    Real h = big;
    for (int d = 0; d < nd; ++d) {
      const Real past = 0.5 * kExitTolerance * (hi[d] - lo[d]);
      Real hd = big;
      if (w[d] > 0.0) hd = (hi[d] + past - lp[d]) / w[d];
      if (w[d] < 0.0) hd = (lo[d] - past - lp[d]) / w[d];
      h = hd < h ? hd : h;
    }
    return h > 0.0 ? h : 0.0;
  };

  Real l[3];
  bool known = false;
  if constexpr (kCylindrical<Coords>) {
    // on a polar patch near is the inverse map at x, which needs no guess, so the ray
    // keeps it where it is
    known = near != nullptr && coords.Patch().type == parthenon::MappedPatch::Polar;
  }
  if (known) {
    for (int d = 0; d < 3; ++d)
      l[d] = near[d];
  } else {
    CartToLogicalNear(coords, x, l, near);
  }
  Real glo = outside(l);
  if (glo >= 0.0) return 0.0;

  Real hcap = big;
  auto cap = [&](const Real num, const Real den) {
    if (den > 0.0 && num > 0.0 && num / den < hcap) hcap = num / den;
  };
  cap(-(x[0] * v[0] + x[1] * v[1] + x[2] * v[2]),
      v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if constexpr (kCylindrical<Coords>) {
    cap(-(x[0] * v[0] + x[1] * v[1]), v[0] * v[0] + v[1] * v[1]);
  }

  if constexpr (kCylindrical<Coords>) {
    // the faces of polar cells are spheres and cones, met where quadratics say, so aim
    // just past the nearest, and only search if the ray is not out there by little
    Real hf, rate;
    if (coords.Patch().type == parthenon::MappedPatch::Polar &&
        impl::PolarFaces(coords.Patch(), x, v, lo, hi, hf, rate)) {
      const Real h = std::min(hf + 0.5 * kExitTolerance / rate, hcap);
      Real xp[3], lp[3];
      for (int d = 0; d < 3; ++d)
        xp[d] = x[d] + h * v[d];
      CartToLogical(coords, xp, lp);
      const Real g = outside(lp);
      if (h < hcap ? g >= 0.0 && g <= kExitTolerance : g < 0.0) {
        return h;
      }
    }
    // likewise for the faces of Coons cells that are straight in (R, z), aiming past by
    // the rate at the start, which is all but that at the face across a cell, and for
    // those that are curved by the rate where Newton's method has the ray meet them
    if (coords.Patch().type == parthenon::MappedPatch::Coons) {
      const auto &p = coords.Patch();
      Real hs, ws[3];
      int dir;
      bool straight[2];
      const bool found = impl::CoonsFaces(p, x, v, lo, hi, hs, dir, straight);
      LogicalVelocity(coords, x, l, v, ws);
      Real rs = found ? std::abs(ws[dir]) / (hi[dir] - lo[dir]) : 0.0;
      // where the ray is when it meets the face, the logical point to start from
      Real lf[3];
      for (int d = 0; d < 3; ++d)
        lf[d] = l[d] + hs * ws[d];
      for (int d = 0; d < 2; ++d) {
        if (straight[d]) continue;
        for (int side = 0; side < 2; ++side) {
          Real hc, sc, rc;
          const Real f = side ? hi[d] : lo[d];
          if (impl::CoonsCurvedFace(p, x, v, l, ws, lo, hi, d, f, hc, sc, rc) &&
              hc < hs) {
            hs = hc;
            dir = d;
            rs = rc;
            lf[d] = f;
            lf[1 - d] = sc;
          }
        }
      }
      if (dir >= 0 && rs > 0.0) {
        const Real h = std::min(hs + 0.5 * kExitTolerance / rs, hcap);
        Real xp[3], lp[3], lg[3];
        for (int d = 0; d < 3; ++d) {
          xp[d] = x[d] + h * v[d];
          lg[d] = lf[d] + (h - hs) * ws[d];
        }
        CartToLogicalNear(coords, xp, lp, lg);
        const Real g = outside(lp);
        if (h < hcap ? g >= 0.0 && g <= kExitTolerance : g < 0.0) {
          return h;
        }
      }
    }
  }

  Real hlo = 0.0, hhi = big, ghi = 0.0;
  Real h = linear(0.0, x, l);
  if (h >= big && hcap >= big) return big;
  h = h < hcap ? h : hcap;
  for (int it = 0; it < 64; ++it) {
    Real xp[3], lp[3], lg[3];
    for (int d = 0; d < 3; ++d) {
      xp[d] = x[d] + h * v[d];
      lg[d] = lw[d] + (h - hw) * w[d];
    }
    CartToLogicalNear(coords, xp, lp, lg);
    const Real g = outside(lp);
    if (g >= 0.0) {
      hhi = h;
      ghi = g;
      if (g <= kExitTolerance) return h;
    } else {
      hlo = h;
      glo = g;
      if (h >= hcap) return h;
    }
    if (hhi >= big) {
      // not out yet
      const Real hn = h + linear(h, xp, lp);
      if (!(hn > h)) return h;
      h = hn < hcap ? hn : hcap;
    } else {
      Real hn = hlo - glo * (hhi - hlo) / (ghi - glo);
      if (!(hn > hlo && hn < hhi) || it % 4 == 3) hn = 0.5 * (hlo + hhi);
      h = hn;
    }
  }
  return hhi < big ? hhi : hlo;
}

//----------------------------------------------------------------------------------------
//! \fn  bool SnapToFace
//! \brief Move the ray at the Cartesian point x, at logical l, that is within kSnapBand
//!        of faces of the logical cell [lo, hi] kSnapNudge to the side of each it is
//!        heading to along v, so that it is found in the cell it is entering, or into the
//!        cell if it runs along the face (as CellExit would not move it on from there).
//!        Returns whether it was moved.
template <typename Coords>
KOKKOS_INLINE_FUNCTION bool SnapToFace(const Coords &coords, const int nd, Real x[3],
                                       const Real v[3], Real l[3], const Real lo[3],
                                       const Real hi[3]) {
  Real w[3];
  bool moved = false;
  for (int d = 0; d < nd; ++d) {
    const Real width = hi[d] - lo[d];
    for (int side = 0; side < 2; ++side) {
      const Real face = side ? hi[d] : lo[d];
      if (std::abs(l[d] - face) >= kSnapBand * width) continue;
      if (!moved) LogicalVelocity(coords, x, l, v, w);
      moved = true;
      const Real to = w[d] > 0.0 ? 1.0 : w[d] < 0.0 ? -1.0 : (side ? -1.0 : 1.0);
      l[d] = face + to * kSnapNudge * width;
      break;
    }
  }
  if (moved) LogicalToCart(coords, l, x, x);
  return moved;
}

// The lattice of kBoxSamples + 1 points per direction BlockBoxes samples each block at
constexpr int kBoxSamples = 16;

//----------------------------------------------------------------------------------------
//! \fn  ParArray2D<Real> BlockBoxes
//! \brief Boxes box(b, 2 d) <= m_d <= box(b, 2 d + 1) in the physical mesh coordinates m
//!        ((R, z) on cylindrical meshes, where box(b, 4..5) is left unbounded) around the
//!        local blocks b of the pack, so that a position outside the box of a block is
//!        not in it, nor within kOnBlockTolerance of it. The block is sampled on a
//!        logical lattice, whose cells are in the box of their corners grown by their
//!        longest edge (more than any curved edge bulges), and the box is grown by that.
template <typename Pack, typename Geom>
parthenon::ParArray2D<Real> BlockBoxes(const Pack &pack, const Geom &geom,
                                       const int nblocks, const int ndim) {
  using parthenon::X1DIR;
  using parthenon::X2DIR;
  using parthenon::X3DIR;
  parthenon::ParArray2D<Real> box("block boxes", nblocks, 6);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), 0,
      nblocks - 1, KOKKOS_LAMBDA(const int b) {
        const auto &coords = pack.GetCoordinates(b);
        using Coords = std::decay_t<decltype(coords)>;
        const Real lo[3] = {geom.Xmin(b, X1DIR), geom.Xmin(b, X2DIR),
                            geom.Xmin(b, X3DIR)};
        const Real hi[3] = {geom.Xmax(b, X1DIR), geom.Xmax(b, X2DIR),
                            geom.Xmax(b, X3DIR)};
        // the physical mesh coordinates of lattice point s
        auto Point = [&](const int s[3], Real m[3]) {
          Real l[3];
          for (int d = 0; d < 3; ++d)
            l[d] = d < ndim ? lo[d] + (hi[d] - lo[d]) * s[d] / kBoxSamples : lo[d];
          if constexpr (kCylindrical<Coords>) {
            coords.Map(l[0], l[1], m[0], m[1]);
            m[2] = 0.0;
          } else if constexpr (kCartesian<Coords>) {
            coords.Map(l[0], l[1], l[2], m[0], m[1], m[2]);
          }
        };
        const int nbox = kCylindrical<Coords> ? 2 : 3;
        Real mlo[3], mhi[3], grow = 0.0;
        for (int d = 0; d < 3; ++d) {
          mlo[d] = std::numeric_limits<Real>::max();
          mhi[d] = -std::numeric_limits<Real>::max();
        }
        const int n2 = ndim > 1 ? kBoxSamples : 0, n3 = ndim > 2 ? kBoxSamples : 0;
        for (int s3 = 0; s3 <= n3; ++s3) {
          for (int s2 = 0; s2 <= n2; ++s2) {
            for (int s1 = 0; s1 <= kBoxSamples; ++s1) {
              const int s[3] = {s1, s2, s3};
              Real m[3];
              Point(s, m);
              for (int d = 0; d < nbox; ++d) {
                mlo[d] = Kokkos::min(mlo[d], m[d]);
                mhi[d] = Kokkos::max(mhi[d], m[d]);
              }
              for (int e = 0; e < ndim; ++e) {
                if (s[e] == kBoxSamples) continue;
                int t[3] = {s1, s2, s3};
                ++t[e];
                Real mt[3];
                Point(t, mt);
                Real h2 = 0.0;
                for (int d = 0; d < nbox; ++d)
                  h2 += (mt[d] - m[d]) * (mt[d] - m[d]);
                grow = Kokkos::max(grow, std::sqrt(h2));
              }
            }
          }
        }
        for (int d = 0; d < 3; ++d) {
          // ... and by far more than kOnBlockTolerance of the block moves a position
          const Real g = grow + 1.0e-6 * (mhi[0] - mlo[0] + mhi[1] - mlo[1]);
          box(b, 2 * d) = d < nbox ? mlo[d] - g : -std::numeric_limits<Real>::max();
          box(b, 2 * d + 1) = d < nbox ? mhi[d] + g : std::numeric_limits<Real>::max();
        }
      });
  return box;
}

//----------------------------------------------------------------------------------------
//! \fn  int OwnerBlock
//! \brief The local block of the MeshData of geom holding the physical mesh position m,
//!        or -1 if none does, as geom.Locate(-1, m) has it, but trying only the blocks
//!        whose box (of BlockBoxes) holds m. Locate() starts from the lowest block with m
//!        in its closed logical bounds, which then holds it, and holds m in no block
//!        unless m is within its face tolerance of one, so only when m is within
//!        kOnBlockTolerance of a block other than one holding it is Locate() asked.
template <typename Pack, typename Geom, typename Boxes>
KOKKOS_INLINE_FUNCTION int OwnerBlock(const Pack &pack, const Geom &geom,
                                      const Boxes &box, const int ndim, const Real m[3]) {
  using parthenon::X1DIR;
  using parthenon::X2DIR;
  using parthenon::X3DIR;
  int inside = -1, near = 0;
  for (int b = 0; b < geom.NumBlocks(); ++b) {
    bool in_box = true;
    for (int d = 0; d < 3; ++d)
      in_box = in_box && m[d] >= box(b, 2 * d) && m[d] <= box(b, 2 * d + 1);
    if (!in_box) continue;
    // only mapped coordinates have the inverse map, and only they are asked
    Real l[3] = {m[0], m[1], m[2]};
    if constexpr (parthenon::IsMappedCoord())
      pack.GetCoordinates(b).PhysicalToLogical(m[0], m[1], m[2], l[0], l[1], l[2]);
    const Real lo[3] = {geom.Xmin(b, X1DIR), geom.Xmin(b, X2DIR), geom.Xmin(b, X3DIR)};
    const Real hi[3] = {geom.Xmax(b, X1DIR), geom.Xmax(b, X2DIR), geom.Xmax(b, X3DIR)};
    bool closed = true, is_near = true;
    for (int d = 0; d < ndim; ++d) {
      const Real tol = kOnBlockTolerance * (hi[d] - lo[d]);
      closed = closed && l[d] >= lo[d] && l[d] <= hi[d];
      is_near = is_near && l[d] >= lo[d] - tol && l[d] <= hi[d] + tol;
    }
    near += is_near;
    if (closed && inside < 0) inside = b;
  }
  if (near == 0) return -1;
  if (near == 1 && inside >= 0) return inside;
  Real x1 = m[0], x2 = m[1], x3 = m[2];
  const auto loc = geom.Locate(-1, x1, x2, x3);
  return loc.InMeshData() ? loc.Block() : -1;
}

} // namespace RiotForest

#endif // FOREST_MAPPED_RAYS_HPP_
