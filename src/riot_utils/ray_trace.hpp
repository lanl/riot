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

#include <tuple>
#include <utility>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

#include "variables.hpp"

using Coordinates_t = parthenon::Coordinates_t;

namespace RayTrace {

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

  KOKKOS_FORCEINLINE_FUNCTION Real min_size() const { return min_dx; }

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
  template <typename SwarmPack_t>
  KOKKOS_FUNCTION IntegratorBase(VarPack_t &vp, SwarmPack_t &ps, const int b,
                                 const int pidx, const int ndim,
                                 const parthenon::SwarmDeviceContext &context)
      : CellInfo(vp.GetCoordinates(b), b, ndim), vp(vp), pidx(pidx),
        xc{ps(b, swarm_position::x(), pidx), ps(b, swarm_position::y(), pidx),
           ps(b, swarm_position::z(), pidx)},
        v{ps(b, particles::ray_tracer::vx(), pidx),
          ps(b, particles::ray_tracer::vy(), pidx),
          ps(b, particles::ray_tracer::vz(), pidx)},
        context(context) {
    init_cell(xc[0], xc[1], xc[2]);
    x = coord_to_cart(xc[0], xc[1], xc[2]);
  }

  KOKKOS_INLINE_FUNCTION
  void set_cell() { init_cell(xc[0], xc[1], xc[2]); }

  template <typename F>
  KOKKOS_INLINE_FUNCTION void loop_3d(F &&func) const {
    loop_exec<X1DIR, X3DIR>(func);
  }

  KOKKOS_INLINE_FUNCTION
  void snap_to_face() {
    // kFaceEps / kFaceNudge (defined at namespace scope) handle finite precision
    // in face/particle position comparisons. The nudge pushes the particle just a
    // bit past the face it is entering so the index calc comes out right in the
    // next cell.
    // the radial face at rf, along the n Cartesian directions the radius spans. The axis
    // or origin is not a face, so a ray reaching it passes straight through.
    auto snap_radius = [&](const Real rf, const Real vr, const int n) {
      const Real correction = (rf + sign(vr) * kFaceNudge) / xc[0];
      for (int d = 0; d < n; d++)
        x[d] *= correction;
      xc[0] *= correction;
    };
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      loop_3d([&]<int d>() {
        if (std::abs(xc[d - 1] - xlo<d>()) < kFaceEps)
          xc[d - 1] = xlo<d>() + sign(v[d - 1]) * kFaceNudge;
        if (std::abs(xc[d - 1] - xhi<d>()) < kFaceEps)
          xc[d - 1] = xhi<d>() + sign(v[d - 1]) * kFaceNudge;
        x[d - 1] = xc[d - 1];
      });
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // x . v has the sign of the radial velocity
      const Real vr = x[0] * v[0] + x[1] * v[1];
      if (xlo<X1DIR>() > 0.0 && std::abs(xc[0] - xlo<X1DIR>()) < kFaceEps)
        snap_radius(xlo<X1DIR>(), vr, 2);
      if (std::abs(xc[0] - xhi<X1DIR>()) < kFaceEps) snap_radius(xhi<X1DIR>(), vr, 2);
      if (std::abs(x[2] - xlo<X2DIR>()) < kFaceEps) {
        x[2] = xlo<X2DIR>() + sign(v[2]) * kFaceNudge;
        xc[1] = x[2];
      } else if (std::abs(x[2] - xhi<X2DIR>()) < kFaceEps) {
        x[2] = xhi<X2DIR>() + sign(v[2]) * kFaceNudge;
        xc[1] = x[2];
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      // x . v has the sign of the radial velocity
      const Real vr = x[0] * v[0] + x[1] * v[1] + x[2] * v[2];
      if (xlo<X1DIR>() > 0.0 && std::abs(xc[0] - xlo<X1DIR>()) < kFaceEps)
        snap_radius(xlo<X1DIR>(), vr, 3);
      if (std::abs(xc[0] - xhi<X1DIR>()) < kFaceEps) snap_radius(xhi<X1DIR>(), vr, 3);
    } else {
      PARTHENON_FAIL("Unsupported coordinate system in IntegratorBase.");
    }
  }

  // Cross the block in one go
  KOKKOS_INLINE_FUNCTION
  void push_to_block_boundary() {
    const Real h = block_exit_distance();
    if (h < 0.0) return;
    for (int d = 0; d < 3; d++)
      x[d] += h * v[d];
    sync_cart_to_coord();
    init_cell(xc[0], xc[1], xc[2]);
    snap_to_face();
    skipped = true;
  }

  // Along the ray, the squared distance from the origin or the axis is
  // r2 + 2 bq h + aq h^2, with aq > 0. The distance to where it falls to rlo^2, or the
  // largest Real if it never gets that low.
  KOKKOS_INLINE_FUNCTION
  static Real radial_in(const Real aq, const Real bq, const Real r2, const Real rlo) {
    const Real disc = bq * bq - aq * (r2 - rlo * rlo);
    if (bq < 0.0 && disc >= 0.0) return (r2 - rlo * rlo) / (-bq + std::sqrt(disc));
    return std::numeric_limits<Real>::max();
  }
  // The distance to where it rises to rhi^2
  KOKKOS_INLINE_FUNCTION
  static Real radial_out(const Real aq, const Real bq, const Real r2, const Real rhi) {
    const Real sq = std::sqrt(std::max(bq * bq - aq * (r2 - rhi * rhi), 0.0));
    return bq > 0.0 ? (rhi * rhi - r2) / (bq + sq) : (sq - bq) / aq;
  }

  // The distance along the ray to just past where it leaves the block, or a negative
  // value if it never does
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
      const Real hlo = rlo > 0.0 ? radial_in(aq, bq, r2, rlo) : big;
      add_face(hlo < big ? hlo : radial_out(aq, bq, r2, rhi));
    };
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (v[0] != 0.0)
        add_face(((v[0] > 0.0 ? context.x_max_ : context.x_min_) - x[0]) / v[0]);
      if (v[1] != 0.0)
        add_face(((v[1] > 0.0 ? context.y_max_ : context.y_min_) - x[1]) / v[1]);
      if (v[2] != 0.0)
        add_face(((v[2] > 0.0 ? context.z_max_ : context.z_min_) - x[2]) / v[2]);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      const Real aq = v[0] * v[0] + v[1] * v[1];
      if (aq > 0.0) {
        radial(aq, x[0] * v[0] + x[1] * v[1], x[0] * x[0] + x[1] * x[1], context.x_min_,
               context.x_max_);
      }
      if (v[2] != 0.0)
        add_face(((v[2] > 0.0 ? context.y_max_ : context.y_min_) - x[2]) / v[2]);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      radial(v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
             x[0] * v[0] + x[1] * v[1] + x[2] * v[2],
             x[0] * x[0] + x[1] * x[1] + x[2] * x[2], context.x_min_, context.x_max_);
    }
    if (h1 == big) return -1.0;
    h1 = std::max(h1, 0.0);
    const Real h = h1 + kStepOvershoot * min_size();
    return h2 == big ? h : std::min(h, 0.5 * (h1 + h2));
  }

  KOKKOS_INLINE_FUNCTION
  bool is_active() const { return context.IsActive(pidx); }

  KOKKOS_INLINE_FUNCTION
  bool on_block() const { return on_block(xc); }

  // Whether the mesh coordinates xb are in the block, to within tol
  KOKKOS_INLINE_FUNCTION
  bool on_block(const std::array<Real, 3> &xb, const Real tol = 0.0) const {
    return in_box(xb, tol, context.x_min_, context.x_max_, context.y_min_, context.y_max_,
                  context.z_min_, context.z_max_);
  }

  KOKKOS_INLINE_FUNCTION
  bool on_mesh() const {
    return in_box(xc, 0.0, context.x_min_global_, context.x_max_global_,
                  context.y_min_global_, context.y_max_global_, context.z_min_global_,
                  context.z_max_global_);
  }

  KOKKOS_INLINE_FUNCTION
  bool stop() const {
    if (skipped) return true;
    if (!is_active()) return true;
    if (!on_block()) return true;
    return false;
  }

  // Write the ray back to the pack. The angles along inactive directions are filled in
  // first, since the swarm hands the ray on by its mesh coordinates.
  template <typename SwarmPack_t>
  KOKKOS_INLINE_FUNCTION void commit(SwarmPack_t &ps, const int b) {
    namespace prt = particles::ray_tracer;
    commit_angles();
    bool on_current_block;
    context.GetNeighborBlockIndex(pidx, xc[0], xc[1], xc[2], on_current_block);
    ps(b, swarm_position::x(), pidx) = xc[0];
    ps(b, swarm_position::y(), pidx) = xc[1];
    ps(b, swarm_position::z(), pidx) = xc[2];
    ps(b, prt::vx(), pidx) = v[0];
    ps(b, prt::vy(), pidx) = v[1];
    ps(b, prt::vz(), pidx) = v[2];
  }

  // Fill in the angles along inactive directions, which sync_cart_to_coord leaves out
  KOKKOS_INLINE_FUNCTION
  void commit_angles() {
    if (!angles_stale) return;
    const auto xfull = cart_to_coord(x, true);
    xc[1] = xfull[1];
    xc[2] = xfull[2];
    angles_stale = false;
  }

  // The mesh coordinates of the Cartesian point x. The data do not vary along the angles,
  // so unless full is set they are left at zero rather than paying for the inverse trig.
  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> cart_to_coord(std::array<Real, 3> &x, const bool full = false) {
    std::array<Real, 3> xout{x[0], x[1], x[2]};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      xout[1] = x[2];
      xout[2] = 0.0;
      if (full) {
        xout[2] = atan2(x[1], x[0]);
        xout[2] += (xout[2] < 0.0) * 2.0 * M_PI;
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      xout[1] = full ? std::acos(x[2] / xout[0]) : 0.0;
      xout[2] = 0.0;
      if (full) {
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
      angles_stale = true;
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
  const parthenon::SwarmDeviceContext &context;
  bool skipped = false;
  // whether xc holds zero for the angles along inactive directions, see cart_to_coord
  bool angles_stale = false;

 private:
  // Whether the mesh coordinates xb are in the box, to within tol, along the directions
  // the data vary in
  KOKKOS_INLINE_FUNCTION
  static bool in_box(const std::array<Real, 3> &xb, const Real tol, const Real x_min,
                     const Real x_max, const Real y_min, const Real y_max,
                     const Real z_min, const Real z_max) {
    if (xb[0] < x_min - tol || xb[0] > x_max + tol) return false;
    if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) return true;
    if (xb[1] < y_min - tol || xb[1] > y_max + tol) return false;
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) return true;
    if (xb[2] < z_min - tol || xb[2] > z_max + tol) return false;
    return true;
  }

  template <int current, int max, typename F>
  KOKKOS_INLINE_FUNCTION void loop_exec(F &func) const {
    static_assert(current > 0 && current < 4);
    func.template operator()<current>();
    if constexpr (current + 1 <= max) loop_exec<current + 1, max>(func);
  }
};

// An Integrator from the leading arguments lead and then the elements of the tuple args
template <typename Integrator, typename Tuple, std::size_t... I, typename... Lead>
KOKKOS_INLINE_FUNCTION Integrator MakeIntegrator(const Tuple &args,
                                                 std::index_sequence<I...>,
                                                 Lead &&...lead) {
  return Integrator(std::forward<Lead>(lead)..., std::get<I>(args)...);
}

template <template <typename> class Integrator_t,
          typename SwarmPack_t, typename VarPack_t, typename... Args>
TaskStatus Trace(SwarmPack_t &ps, VarPack_t &v, Mesh *pm, Args &&...args) {
  using Integrator = Integrator_t<const VarPack_t>;

  const int ndim = pm->ndim;
  // nvcc does not let a device lambda capture a parameter pack, so the trailing
  // integrator arguments are captured as a tuple
  const auto targs = std::make_tuple(args...);

  int num_not_done = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      ps.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ps.GetBlockParticleIndices(idx);
        auto &context = ps.GetContext(b);
        if (context.IsActive(n)) {
          Integrator integrator = MakeIntegrator<Integrator>(
              targs, std::index_sequence_for<Args...>{}, v, ps, b, n, ndim, context);
          while (!integrator.stop()) {
            integrator.set_cell();
            integrator.advance();
          }
          integrator.commit(ps, b);
          if (integrator.is_active()) {
            num_unfinished++;
          }
        }
      },
      Kokkos::Sum<int>(num_not_done));

  if (num_not_done) return TaskStatus::iterate;
  return TaskStatus::complete;
}

} // namespace RayTrace

#endif // RIOT_UTILS_RAY_TRACE_HPP_
