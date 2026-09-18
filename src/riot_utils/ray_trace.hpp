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

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

#include <variables.hpp>

using Coordinates_t = parthenon::Coordinates_t;

namespace RayTrace {

namespace rt = particles::ray_tracer;

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
  template <typename SwarmPack_t>
  KOKKOS_FUNCTION IntegratorBase(VarPack_t &vp, SwarmPack_t &ps, const int b,
                                 const int pidx, const int ndim)
      : CellInfo(vp.GetCoordinates(b), b, ndim), vp(vp), pidx(pidx),
        xc{ps(b, swarm_position::x(), pidx), ps(b, swarm_position::y(), pidx),
           ps(b, swarm_position::z(), pidx)},
        v{ps(b, rt::vx(), pidx), ps(b, rt::vy(), pidx), ps(b, rt::vz(), pidx)},
        context(ps.GetContext(b)) {
    init_cell(xc[0], xc[1], xc[2]);
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
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
    // both of these fuzz factors deal with finite precision related to face/particle
    // position comparisons and calculations.  The nudge pushes the particle just a bit
    // more into the cell it is entering to ensure the index calc comes out right in
    // the next cell the particle enters
    constexpr Real fuzzy_face = 10 * std::numeric_limits<Real>::epsilon();
    constexpr Real fuzzy_nudge = 20 * std::numeric_limits<Real>::epsilon();
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      loop_3d([&]<int d>() {
        if (std::abs(xc[d - 1] - xlo<d>()) < fuzzy_face)
          xc[d - 1] = xlo<d>() + 10 * sign(v[d - 1]) * fuzzy_nudge;
        x[d - 1] = xc[d - 1];
        if (std::abs(xc[d - 1] - xhi<d>()) < fuzzy_face)
          xc[d - 1] = xhi<d>() + 10 * sign(v[d - 1]) * fuzzy_nudge;
        x[d - 1] = xc[d - 1];
      });
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      Real vr = (x[0] * v[0] + x[1] * v[1]) / xc[0];
      if (std::abs(xc[0] - xlo<X1DIR>()) < fuzzy_face) {
        Real correction = (xlo<X1DIR>() + 10 * sign(vr) * fuzzy_nudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(xc[0] - xhi<X1DIR>()) < fuzzy_face) {
        Real correction = (xhi<X1DIR>() + 10 * sign(vr) * fuzzy_nudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(x[2] - xlo<X2DIR>()) < fuzzy_face) {
        x[2] = xlo<X2DIR>() + 10 * sign(v[2]) * fuzzy_nudge;
        xc[1] = x[2];
      } else if (std::abs(x[2] - xhi<X2DIR>()) < fuzzy_face) {
        x[2] = xhi<X2DIR>() + 10 * sign(v[2]) * fuzzy_nudge;
        xc[1] = x[2];
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      Real vr = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / xc[0];
      if (std::abs(xc[0] - xlo<X1DIR>()) < fuzzy_face) {
        Real correction = (xlo<X1DIR>() + 10 * sign(vr) * fuzzy_nudge) / xc[0];
        x[0] *= correction;
        x[1] *= correction;
        x[2] *= correction;
        xc[0] *= correction;
      }
      if (std::abs(xc[0] - xhi<X1DIR>()) < fuzzy_face) {
        Real correction = (xhi<X1DIR>() + 10 * sign(vr) * fuzzy_nudge) / xc[0];
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
  bool is_active() const { return context.IsActive(pidx); }

  KOKKOS_INLINE_FUNCTION
  bool on_block() const {
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (xc[0] < context.x_min_ || xc[0] > context.x_max_) return false;
      if (xc[1] < context.y_min_ || xc[1] > context.y_max_) return false;
      if (xc[2] < context.z_min_ || xc[2] > context.z_max_) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (xc[0] < context.x_min_ || xc[0] > context.x_max_) return false;
      if (xc[1] < context.y_min_ || xc[1] > context.y_max_) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      if (xc[0] < context.x_min_ || xc[0] > context.x_max_) return false;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  bool on_mesh() const {
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      if (xc[0] < context.x_min_global_ || xc[0] > context.x_max_global_) return false;
      if (xc[1] < context.y_min_global_ || xc[1] > context.y_max_global_) return false;
      if (xc[2] < context.z_min_global_ || xc[2] > context.z_max_global_) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (xc[0] < context.x_min_global_ || xc[0] > context.x_max_global_) return false;
      if (xc[1] < context.y_min_global_ || xc[1] > context.y_max_global_) return false;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      if (xc[0] < context.x_min_global_ || xc[0] > context.x_max_global_) return false;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  bool stop() const {
    if (!is_active()) return true;
    if (!on_block()) return true;
    return false;
  }

  template <typename SwarmPack_t>
  KOKKOS_INLINE_FUNCTION void commit(SwarmPack_t &ps, const int b) {
    bool on_current_block;
    context.GetNeighborBlockIndex(pidx, xc[0], xc[1], xc[2], on_current_block);
    ps(b, swarm_position::x(), pidx) = xc[0];
    ps(b, swarm_position::y(), pidx) = xc[1];
    ps(b, swarm_position::z(), pidx) = xc[2];
    ps(b, rt::vx(), pidx) = v[0];
    ps(b, rt::vy(), pidx) = v[1];
    ps(b, rt::vz(), pidx) = v[2];
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> cart_to_coord(std::array<Real, 3> &x) {
    std::array<Real, 3> xout{x[0], x[1], x[2]};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      xout[1] = x[2];
      xout[2] = atan2(x[1], x[0]);
      xout[2] += (xout[2] < 0.0) * 2.0 * M_PI;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      xout[0] = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      xout[1] = std::acos(x[2] / xout[0]);
      xout[2] = atan2(x[1], x[0]);
      xout[2] += (xout[2] < 0.0) * 2.0 * M_PI;
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
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> transform_coord_to_cart(Real n1, Real n2, Real n3) {
    std::array<Real, 3> nc{n1, n2, n3};
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      const Real sph = std::sin(xc[2]);
      const Real cph = std::cos(xc[2]);
      nc[0] = n1 * cph - n3 * sph;
      nc[1] = n1 * sph + n3 * cph;
      nc[2] = n2;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      const Real sth = std::sin(xc[1]);
      const Real cth = std::cos(xc[1]);
      const Real sph = std::sin(xc[2]);
      const Real cph = std::cos(xc[2]);
      nc[0] = n1 * sth * cph + n2 * xc[0] * cth * cph - n3 * sth * sph;
      nc[1] = n1 * sth * sph + n2 * xc[0] * cth * sph + n3 * sth * cph;
      nc[2] = n1 * cth - n2 * xc[0] * sth;
    }
    return nc;
  }

  VarPack_t &vp;
  const int pidx;
  std::array<Real, 3> xc, x, v;
  parthenon::SwarmDeviceContext context;

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

template <template <typename> class Integrator_t, bool RemoveComplete, bool MarkRemoval,
          typename SwarmPack_t, typename VarPack_t, typename... Args>
TaskStatus Trace(SwarmPack_t &ps, VarPack_t &v, Mesh *pm, Args &&...args) {
  namespace rt = particles::ray_tracer;

  const int ndim = pm->ndim;

  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  int num_not_done = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      ps.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ps.GetBlockParticleIndices(idx);
        Integrator_t integrator(v, ps, b, n, ndim, args...);
        while (!integrator.stop()) {
          integrator.set_cell();
          integrator.advance();
        }
        integrator.commit(ps, b);
        if (integrator.is_active()) num_unfinished++;
      },
      Kokkos::Sum<int>(num_not_done));

  if (num_not_done) return TaskStatus::iterate;
  return TaskStatus::complete;
}

} // namespace RayTrace

#endif // DIAGNOSTICS_RAY_TRACE_HPP_
