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
#ifndef RIOT_UTILS_RAY_TRACE_HPP_
#define RIOT_UTILS_RAY_TRACE_HPP_
// This file was made in part with generative AI.

#include <memory>
#include <tuple>
#include <utility>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

#include <variables.hpp>

using Coordinates_t = parthenon::Coordinates_t;

namespace RayTrace {

namespace rt = particles::ray_tracer;

// Machine-epsilon-scaled tolerances used by snap_to_face to handle finite
// precision in face/particle position comparisons. kFaceEps is the band within
// which a particle is treated as sitting on a cell face; kFaceNudge is how far
// it is then pushed across that face so the next cell-index calculation lands
// in the cell being entered. kFaceNudge must stay comfortably larger than
// kFaceEps.
constexpr Real kFaceEps = 10 * std::numeric_limits<Real>::epsilon();
constexpr Real kFaceNudge = 20 * kFaceEps;
//   kStepOvershoot - relative amount (times min cell size) a step is extended
//                    past a face so the crossing is unambiguous.
constexpr Real kStepOvershoot = 1.e-8;

KOKKOS_FORCEINLINE_FUNCTION
Real sign(const Real val) { return (val > 0) - (val < 0); };

// Faces of the block a ray is traced through, and of the mesh, in the mesh's coordinates
struct BlockBounds {
  BlockBounds() = default;
  // The mesh faces, with the block faces left to set on the device
  explicit BlockBounds(const parthenon::RegionSize &mesh_size)
      : x_min_global(mesh_size.xmin(X1DIR)), x_max_global(mesh_size.xmax(X1DIR)),
        y_min_global(mesh_size.xmin(X2DIR)), y_max_global(mesh_size.xmax(X2DIR)),
        z_min_global(mesh_size.xmin(X3DIR)), z_max_global(mesh_size.xmax(X3DIR)) {}

  // The same mesh faces, with the faces of local block b
  KOKKOS_INLINE_FUNCTION
  BlockBounds OnBlock(const parthenon::LukewarmGeometry &geom, const int b) const {
    BlockBounds bounds = *this;
    bounds.x_min = geom.Xmin(b, X1DIR);
    bounds.x_max = geom.Xmax(b, X1DIR);
    bounds.y_min = geom.Xmin(b, X2DIR);
    bounds.y_max = geom.Xmax(b, X2DIR);
    bounds.z_min = geom.Xmin(b, X3DIR);
    bounds.z_max = geom.Xmax(b, X3DIR);
    return bounds;
  }

  Real x_min, x_max, y_min, y_max, z_min, z_max;
  Real x_min_global, x_max_global, y_min_global, y_max_global, z_min_global,
      z_max_global;
};

struct CellInfo {
  KOKKOS_FUNCTION
  CellInfo(const Coordinates_t &coords, const int b, const int numdim)
      : coords(coords), cell_b(b), dj(numdim > 1), dk(numdim > 2), ndim(numdim) {
    min_dx = std::min(coords.Dxf<X3DIR>(0),
                      std::min(coords.Dxf<X2DIR>(0), coords.Dxf<X1DIR>(0)));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int get_index(const Real &x0, const Real xlo, const Real dx) {
    return (x0 - xlo) / dx;
  };

  KOKKOS_INLINE_FUNCTION
  void init_cell(const Real &x1, const Real &x2, const Real &x3) {
    cell_i = get_index(x1, coords.Xf<X1DIR>(0), coords.Dxf<X1DIR>(0));
    cell_j = ndim > 1 ? get_index(x2, coords.Xf<X2DIR>(0), coords.Dxf<X2DIR>(0)) : 0;
    cell_k = ndim > 2 ? get_index(x3, coords.Xf<X3DIR>(0), coords.Dxf<X3DIR>(0)) : 0;
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real xlo() const {
    return coords.Xf<dir>(cell_k, cell_j, cell_i);
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real xhi() const {
    return coords.Xf<dir>(cell_k + (dir == X3DIR), cell_j + (dir == X2DIR),
                          cell_i + (dir == X1DIR));
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real dx() const {
    return xhi<dir>() - xlo<dir>();
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real xmid() const {
    return 0.5 * (xlo<dir>() + xhi<dir>());
  }

  KOKKOS_FORCEINLINE_FUNCTION Real min_size() const { return min_dx; }

  KOKKOS_FORCEINLINE_FUNCTION Real volume() const {
    return coords.CellVolume(cell_k, cell_j, cell_i);
  }

  const Coordinates_t &coords;
  const int cell_b, dj, dk, ndim;
  int cell_i, cell_j, cell_k;
  Real min_dx;
};

template <typename VarPack_t>
class IntegratorBase : public CellInfo {
 public:
  using CellInfo::dx;
  using CellInfo::xlo;
  // xcart, where given, is the Cartesian position the ray left its last block at, which
  // spares rebuilding it from the coordinates in the pack
  template <typename Pack_t>
  KOKKOS_FUNCTION IntegratorBase(VarPack_t &vp, Pack_t &ps, const int b, const int pidx,
                                 const int ndim, const BlockBounds &bounds,
                                 const std::array<Real, 3> *xcart = nullptr)
      : CellInfo(vp.GetCoordinates(b), b, ndim), vp(vp), pidx(pidx),
        xc{ps(lukewarm::x(), pidx), ps(lukewarm::y(), pidx), ps(lukewarm::z(), pidx)},
        v{ps(rt::vx(), pidx), ps(rt::vy(), pidx), ps(rt::vz(), pidx)}, bounds(bounds) {
    init_cell(xc[0], xc[1], xc[2]);
    if (xcart != nullptr && !parthenon::IsCoord<parthenon::UniformCartesian>()) {
      // the angles in the pack along inactive directions may not have been written
      x = *xcart;
      angles_stale = ndim < 3;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      x[0] = xc[0];
      x[1] = xc[1];
      x[2] = xc[2];
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      x[0] = xc[0] * std::cos(xc[2]);
      x[1] = xc[0] * std::sin(xc[2]);
      x[2] = xc[1];
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      const Real sth = std::sin(xc[1]);
      const Real cth = std::cos(xc[1]);
      const Real sph = std::sin(xc[2]);
      const Real cph = std::cos(xc[2]);
      x[0] = xc[0] * sth * cph;
      x[1] = xc[0] * sth * sph;
      x[2] = xc[0] * cth;
    } else {
      PARTHENON_FAIL("Unsupported coordinate system.");
    }
  }

  KOKKOS_INLINE_FUNCTION
  void set_cell() { init_cell(xc[0], xc[1], xc[2]); }

  template <typename F>
  KOKKOS_INLINE_FUNCTION void loop_3d(F &&func) const {
    loop_exec<X1DIR, X3DIR>(func);
  }
  template <typename F>
  KOKKOS_INLINE_FUNCTION void loop_2d(F &&func) const {
    loop_exec<X1DIR, X2DIR>(func);
  }

  KOKKOS_INLINE_FUNCTION
  void snap_to_face() {
    // kFaceEps / kFaceNudge (defined at namespace scope) handle finite precision
    // in face/particle position comparisons. The nudge pushes the particle just a
    // bit past the face it is entering so the index calc comes out right in the
    // next cell.
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      loop_3d([&]<int d>() {
        if (std::abs(xc[d - 1] - xlo<d>()) < kFaceEps)
          xc[d - 1] = xlo<d>() + sign(v[d - 1]) * kFaceNudge;
        x[d - 1] = xc[d - 1];
        if (std::abs(xc[d - 1] - xhi<d>()) < kFaceEps)
          xc[d - 1] = xhi<d>() + sign(v[d - 1]) * kFaceNudge;
        x[d - 1] = xc[d - 1];
      });
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      Real vr = (x[0] * v[0] + x[1] * v[1]) / xc[0];
      if (std::abs(xc[0] - xlo<X1DIR>()) < kFaceEps) {
        Real correction = (xlo<X1DIR>() + sign(vr) * kFaceNudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(xc[0] - xhi<X1DIR>()) < kFaceEps) {
        Real correction = (xhi<X1DIR>() + sign(vr) * kFaceNudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(x[2] - xlo<X2DIR>()) < kFaceEps) {
        x[2] = xlo<X2DIR>() + sign(v[2]) * kFaceNudge;
        xc[1] = x[2];
      } else if (std::abs(x[2] - xhi<X2DIR>()) < kFaceEps) {
        x[2] = xhi<X2DIR>() + sign(v[2]) * kFaceNudge;
        xc[1] = x[2];
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      Real vr = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / xc[0];
      if (std::abs(xc[0] - xlo<X1DIR>()) < kFaceEps) {
        Real correction = (xlo<X1DIR>() + sign(vr) * kFaceNudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        x[2] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(xc[0] - xhi<X1DIR>()) < kFaceEps) {
        Real correction = (xhi<X1DIR>() + sign(vr) * kFaceNudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        x[2] *= correction;
        xc[0] *= correction;
      }
    } else {
      PARTHENON_FAIL("Unsupported coordinate system in IntegratorBase.");
    }
  }

  KOKKOS_INLINE_FUNCTION
  void push_to_block_boundary() {
    if constexpr (!parthenon::IsCoord<parthenon::UniformCartesian>()) {
      // Where every face the ray can leave the block through is tracked, as for 2D
      // cylindrical and 1D spherical blocks, the ray crosses the block in one go.
      // Otherwise it is left to step through.
      const Real h = block_exit_distance();
      if (h < 0.0) return;
      for (int d = 0; d < 3; d++)
        x[d] += h * v[d];
      sync_cart_to_coord();
      init_cell(xc[0], xc[1], xc[2]);
      snap_to_face();
      skipped = true;
      return;
    }
    bool use_x = std::abs(v[0]) > 0.0;
    bool use_y = std::abs(v[1]) > 0.0;
    bool use_z = std::abs(v[2]) > 0.0;
    Real h1 = use_x ? (bounds.x_min + (v[0] > 0) * (bounds.x_max - bounds.x_min) - xc[0]) / v[0]
                    : std::numeric_limits<Real>::max();
    Real h2 = use_y ? (bounds.y_min + (v[1] > 0) * (bounds.y_max - bounds.y_min) - xc[1]) / v[1]
                    : std::numeric_limits<Real>::max();
    Real h3 = use_z ? (bounds.z_min + (v[2] > 0) * (bounds.z_max - bounds.z_min) - xc[2]) / v[2]
                    : std::numeric_limits<Real>::max();
    if (h1 > h2) std::swap(h1, h2);
    if (h2 > h3) std::swap(h2, h3);
    if (h1 > h2) std::swap(h1, h2);

    h3 = h1 + kStepOvershoot * min_size();
    h1 = (h2 == std::numeric_limits<Real>::max() ? h3
                                                 : std::min(h3, 0.5 * (h1 + h2)));
    loop_3d([&]<int d>() {
      xc[d - 1] += h1 * v[d - 1];
    });
    init_cell(xc[0], xc[1], xc[2]);
    snap_to_face();
    skipped = true;
  }

  // The distance along the ray to just past where it leaves the block, where the block
  // faces it can leave through are all tracked exactly, or a negative value otherwise
  KOKKOS_INLINE_FUNCTION
  Real block_exit_distance() const {
    constexpr Real big = std::numeric_limits<Real>::max();
    Real h1 = big, h2 = big;
    // keep the two nearest faces, so the overshoot does not carry the ray past the second
    auto add_face = [&](const Real h) {
      if (h < h1) {
        h2 = h1;
        h1 = h;
      } else if (h < h2) {
        h2 = h;
      }
    };
    // the distance to the radius rlo, if the ray gets that close, and to rhi
    auto radial = [&](const Real aq, const Real bq, const Real r2, const Real rlo,
                      const Real rhi) {
      const Real disc_lo = bq * bq - aq * (r2 - rlo * rlo);
      if (bq < 0.0 && rlo > 0.0 && disc_lo >= 0.0) {
        add_face((r2 - rlo * rlo) / (-bq + std::sqrt(disc_lo)));
      } else {
        const Real sq_hi = std::sqrt(std::max(bq * bq - aq * (r2 - rhi * rhi), 0.0));
        add_face(bq > 0.0 ? (rhi * rhi - r2) / (bq + sq_hi) : (sq_hi - bq) / aq);
      }
    };
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (v[0] != 0.0) add_face(((v[0] > 0.0 ? bounds.x_max : bounds.x_min) - x[0]) / v[0]);
      if (v[1] != 0.0) add_face(((v[1] > 0.0 ? bounds.y_max : bounds.y_min) - x[1]) / v[1]);
      if (v[2] != 0.0) add_face(((v[2] > 0.0 ? bounds.z_max : bounds.z_min) - x[2]) / v[2]);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // the faces in phi are not tracked
      if (ndim > 2) return -1.0;
      const Real aq = v[0] * v[0] + v[1] * v[1];
      if (aq > 0.0) {
        radial(aq, x[0] * v[0] + x[1] * v[1], x[0] * x[0] + x[1] * x[1], bounds.x_min,
               bounds.x_max);
      }
      if (v[2] != 0.0) add_face(((v[2] > 0.0 ? bounds.y_max : bounds.y_min) - x[2]) / v[2]);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      // the faces in theta and phi are not tracked
      if (ndim > 1) return -1.0;
      radial(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], x[0] * v[0] + x[1] * v[1] + x[2] * v[2],
             x[0] * x[0] + x[1] * x[1] + x[2] * x[2], bounds.x_min, bounds.x_max);
    } else {
      return -1.0;
    }
    if (h1 == big) return -1.0;
    h1 = std::max(h1, 0.0);
    const Real h = h1 + kStepOvershoot * min_size();
    return h2 == big ? h : std::min(h, 0.5 * (h1 + h2));
  }

  KOKKOS_INLINE_FUNCTION
  bool is_active() const { return !removed; }

  // Done with this ray, which the next exchange removes
  template <typename Pack_t>
  KOKKOS_INLINE_FUNCTION void remove(Pack_t &ps) {
    removed = true;
    ps.MarkForRemoval(pidx);
  }

  KOKKOS_INLINE_FUNCTION
  bool on_block() const {
    return on_block(xc);
  }

  KOKKOS_INLINE_FUNCTION
  bool on_block(const std::array<Real, 3> &xc) const {
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (xc[0] < bounds.x_min || xc[0] > bounds.x_max) return false;
      if (xc[1] < bounds.y_min || xc[1] > bounds.y_max) return false;
      if (xc[2] < bounds.z_min || xc[2] > bounds.z_max) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (xc[0] < bounds.x_min || xc[0] > bounds.x_max) return false;
      if (xc[1] < bounds.y_min || xc[1] > bounds.y_max) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      if (xc[0] < bounds.x_min || xc[0] > bounds.x_max) return false;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  bool on_mesh() const {
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (xc[0] < bounds.x_min_global || xc[0] > bounds.x_max_global) return false;
      if (xc[1] < bounds.y_min_global || xc[1] > bounds.y_max_global) return false;
      if (xc[2] < bounds.z_min_global || xc[2] > bounds.z_max_global) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (xc[0] < bounds.x_min_global || xc[0] > bounds.x_max_global) return false;
      if (xc[1] < bounds.y_min_global || xc[1] > bounds.y_max_global) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      if (xc[0] < bounds.x_min_global || xc[0] > bounds.x_max_global) return false;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  bool stop() const {
    if (skipped) return true;
    if (!is_active()) return true;
    if (!on_block()) return true;
    return false;
  }

  // Write the ray back to the pack. Unless full is set, the angles along inactive
  // directions are left for commit_angles, as only the next block in the same trace,
  // which is handed the Cartesian position, can follow, and it does not look at them.
  template <typename Pack_t>
  KOKKOS_INLINE_FUNCTION void commit(Pack_t &ps, const bool full = true) {
    if (full) commit_angles(ps);
    ps(lukewarm::x(), pidx) = xc[0];
    ps(lukewarm::y(), pidx) = xc[1];
    ps(lukewarm::z(), pidx) = xc[2];
    ps(rt::vx(), pidx) = v[0];
    ps(rt::vy(), pidx) = v[1];
    ps(rt::vz(), pidx) = v[2];
  }

  // Write the angles along inactive directions, which commit may have left out
  template <typename Pack_t>
  KOKKOS_INLINE_FUNCTION void commit_angles(Pack_t &ps) {
    if (!angles_stale) return;
    const auto xfull = cart_to_coord(x, true);
    xc[1] = xfull[1];
    xc[2] = xfull[2];
    angles_stale = false;
    ps(lukewarm::y(), pidx) = xc[1];
    ps(lukewarm::z(), pidx) = xc[2];
  }

  // The mesh coordinates of the Cartesian point x. The data do not vary along inactive
  // directions, so unless full is set the angles along them are left at zero rather
  // than paying for the inverse trig.
  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> cart_to_coord(std::array<Real, 3> &x, const bool full = false) {
    std::array<Real, 3> xout{x[0], x[1], x[2]};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      xout[1] = x[2];
      xout[2] = 0.0;
      if (full || ndim > 2) {
        xout[2] = atan2(x[1], x[0]);
        xout[2] += (xout[2] < 0.0) * 2.0 * M_PI;
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      xout[1] = (full || ndim > 1) ? std::acos(x[2] / xout[0]) : 0.0;
      xout[2] = 0.0;
      if (full || ndim > 2) {
        xout[2] = atan2(x[1], x[0]);
        xout[2] += (xout[2] < 0.0) * 2.0 * M_PI;
      }
    }
    return xout;
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> coord_to_cart(const Real x1, const Real x2, const Real x3) {
    std::array<Real, 3> xout{x1, x2, x3};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      xout[0] = x1 * std::cos(x3);
      xout[1] = x1 * std::sin(x3);
      xout[2] = x2;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      const Real sth = std::sin(x2);
      const Real cth = std::cos(x2);
      const Real sph = std::sin(x3);
      const Real cph = std::cos(x3);
      xout[0] = x1 * sth * cph;
      xout[1] = x1 * sth * sph;
      xout[2] = x1 * cth;
    }
    return xout;
  }

  KOKKOS_INLINE_FUNCTION
  void sync_cart_to_coord() {
    auto xcoord = cart_to_coord(x);
    xc[0] = xcoord[0];
    xc[1] = xcoord[1];
    xc[2] = xcoord[2];
    if constexpr (!parthenon::IsCoord<parthenon::UniformCartesian>()) {
      angles_stale = ndim < 3;
    }
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> transform_coord_to_cart(Real n1, Real n2, Real n3) {
    std::array<Real, 3> nc{n1, n2, n3};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // the angle from the Cartesian position, which is kept even when xc[2] is not
      const Real sph = xc[0] > 0.0 ? x[1] / xc[0] : 0.0;
      const Real cph = xc[0] > 0.0 ? x[0] / xc[0] : 1.0;
      nc[0] = n1 * cph - n3 * sph;
      nc[1] = n1 * sph + n3 * cph;
      nc[2] = n2;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      // the angles from the Cartesian position, which is kept even when xc[1:] are not
      const Real rcyl = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      const Real sth = xc[0] > 0.0 ? rcyl / xc[0] : 0.0;
      const Real cth = xc[0] > 0.0 ? x[2] / xc[0] : 1.0;
      const Real sph = rcyl > 0.0 ? x[1] / rcyl : 0.0;
      const Real cph = rcyl > 0.0 ? x[0] / rcyl : 1.0;
      nc[0] = n1 * sth * cph + n2 * xc[0] * cth * cph - n3 * sth * sph;
      nc[1] = n1 * sth * sph + n2 * xc[0] * cth * sph + n3 * sth * cph;
      nc[2] = n1 * cth - n2 * xc[0] * sth;
    }
    return nc;
  }

  VarPack_t &vp;
  const int pidx;
  std::array<Real, 3> xc, x, v;
  const BlockBounds bounds;
  bool skipped = false;
  bool removed = false;
  // whether xc holds zero for the angles along inactive directions, see cart_to_coord
  bool angles_stale = false;

 private:
  template <int current, int max, typename F>
  KOKKOS_INLINE_FUNCTION void loop_exec(F &func) const {
    static_assert(current > 0 && current < 4);
    if constexpr (current == X1DIR) func.template operator()<current>();
    if constexpr (current == X2DIR) func.template operator()<current>();
    if constexpr (current == X3DIR) func.template operator()<current>();
    if constexpr (current + 1 <= max) loop_exec<current + 1, max>(func);
  }
};

// An Integrator from the leading arguments lead and then the elements of the tuple args
template <typename Integrator, typename Tuple, std::size_t... I, typename... Lead>
KOKKOS_INLINE_FUNCTION Integrator MakeIntegrator(const Tuple &args, std::index_sequence<I...>,
                                                 Lead &&...lead) {
  return Integrator(std::forward<Lead>(lead)..., std::get<I>(args)...);
}

// Trace the rays [range.s, range.e] of a lukewarm until each is done or leaves the
// MeshData of the pack, which the next exchange delivers it to. Rays just created start
// from their source block, since they may sit on its upper face.
template <template <typename> class Integrator_t, typename Pack_t, typename VarPack_t,
          typename... Args>
TaskStatus Trace(Pack_t &ps, VarPack_t &v, Mesh *pm, const IndexRange &range,
                 const bool first, Args &&...args) {
  namespace rt = particles::ray_tracer;
  using Integrator = Integrator_t<const VarPack_t>;
  if (range.size() <= 0) return TaskStatus::complete;

  const int ndim = pm->ndim;
  const BlockBounds mesh_bounds(pm->mesh_size);
  // nvcc does not let a device lambda capture a parameter pack, so the trailing
  // integrator arguments are captured as a tuple
  const auto targs = std::make_tuple(args...);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), range.s, range.e,
      KOKKOS_LAMBDA(const int n) {
        int b = first ? ps(rt::source_block(), n) : ps.GetBlockIndex(n);
        // the Cartesian position the ray left the last block at, once it has left one
        std::array<Real, 3> xcart;
        const std::array<Real, 3> *xlast = nullptr;
        // Each block is traced as one integrator, as rays used to be between exchanges
        while (true) {
          Integrator integrator = MakeIntegrator<Integrator>(
              targs, std::index_sequence_for<Args...>{}, v, ps, b, n, ndim,
              mesh_bounds.OnBlock(ps.GetGeometry(), b), xlast);
          if (integrator.is_active()) {
            while (!integrator.stop()) {
              integrator.set_cell();
              integrator.advance();
            }
          }
          integrator.commit(ps, false);
          if (!integrator.is_active()) break;
          // locating the ray only takes the coordinates along active directions
          const auto loc = ps.Locate(n);
          if (!loc.InMeshData()) {
            integrator.commit_angles(ps);
            break;
          }
          xcart = integrator.x;
          xlast = &xcart;
          b = loc.Block();
        }
      });

  return TaskStatus::complete;
}

} // namespace RayTrace

#endif // DIAGNOSTICS_RAY_TRACE_HPP_
