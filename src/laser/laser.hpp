//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LASER_LASER_HPP_
#define LASER_LASER_HPP_

#include <limits>
#include <memory>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;
using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

#include "laser_props.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Laser {

KOKKOS_FORCEINLINE_FUNCTION
Real sign(const Real val) { return (val > 0) - (val < 0); };

KOKKOS_FORCEINLINE_FUNCTION
Real ne_crit(const Real lambda) {
  return M_PI * pc::me * pc::c * pc::c / (lambda * lambda * pc::qe * pc::qe);
}

constexpr Real fuzzy_check_nudge = 0.0;

struct PhaseArray {
  KOKKOS_FUNCTION
  PhaseArray() : t(0.0), x({0.0, 0.0, 0.0}), v({0.0, 0.0, 0.0}) {}
  KOKKOS_FUNCTION
  PhaseArray(const Real t, const Real x1, const Real x2, const Real x3, const Real x4,
             const Real x5, const Real x6)
      : t(t), x({x1, x2, x3}), v({x4, x5, x6}) {}

  KOKKOS_INLINE_FUNCTION
  Real vmag() { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

  Real t;
  std::array<Real, 3> x, v;
};

struct LaserParticle : public PhaseArray {
  KOKKOS_FUNCTION
  LaserParticle() = default;

  template <typename T>
  KOKKOS_FUNCTION LaserParticle(T &s, const int b, const int n)
      : PhaseArray(s(b, particles::laser::t(), n), s(b, swarm_position::x(), n),
                   s(b, swarm_position::y(), n), s(b, swarm_position::z(), n),
                   s(b, particles::laser::vx(), n), s(b, particles::laser::vy(), n),
                   s(b, particles::laser::vz(), n)),
        energy(s(b, particles::laser::energy(), n)),
        lambda(s(b, particles::laser::wavelength(), n)) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION void apply_update(T &p, const int b, const int n) {
    p(b, particles::laser::t(), n) = t;
    p(b, swarm_position::x(), n) = x[0];
    p(b, swarm_position::y(), n) = x[1];
    p(b, swarm_position::z(), n) = x[2];
    p(b, particles::laser::vx(), n) = v[0];
    p(b, particles::laser::vy(), n) = v[1];
    p(b, particles::laser::vz(), n) = v[2];
    p(b, particles::laser::energy(), n) = energy;
  }

  KOKKOS_INLINE_FUNCTION
  LaserParticle &operator=(const PhaseArray &p) {
    PhaseArray::operator=(p);
    return *this;
  }

  Real energy;
  Real lambda;
};

template <typename T>
struct CellInfo {
  using TE = parthenon::TopologicalElement;
  using ne_t = cell_variables::cell_averaged::bulk::electron_number_density;
  using nen_t = node_variables::electron_number_density;
  KOKKOS_FUNCTION
  CellInfo(T &vin, const int cb, const Real x, const Real y, const Real z, const Real vx,
           const Real vy, const Real vz, const int cdj, const int cdk, const int nghost)
      : v(vin), b(cb), dj(cdj), dk(cdk), ndim(1 + cdj + cdk) {
    // assume Dxf is constant
    auto &coords = v.GetCoordinates(b);
    auto get_index = [&](const Real x0, const Real v0, const Real xlo, const Real dx) {
      int idx = (x0 - xlo) / dx;
      return idx;
    };

    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      i = get_index(x, vx, coords.template Xf<X1DIR>(0), coords.template Dxf<X1DIR>(0));
      j = ndim > 1 ? get_index(y, vy, coords.template Xf<X2DIR>(0),
                               coords.template Dxf<X2DIR>(0))
                   : 0;
      k = ndim > 2 ? get_index(z, vz, coords.template Xf<X3DIR>(0),
                               coords.template Dxf<X3DIR>(0))
                   : 0;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      auto r = std::sqrt(x * x + y * y);
      auto vr = (x * vx + y * vy) / r;
      i = get_index(r, vr, coords.template Xf<X1DIR>(0), coords.template Dxf<X1DIR>(0));
      j = get_index(z, vz, coords.template Xf<X2DIR>(0), coords.template Dxf<X2DIR>(0));
      k = 0;
    } else {
      PARTHENON_FAIL("Unsupported coordinate system for the laser package.");
    }

    xb = {coords.template Xf<X1DIR>(i), coords.template Xf<X1DIR>(i + 1),
          coords.template Xf<X2DIR>(j), coords.template Xf<X2DIR>(j + 1),
          coords.template Xf<X3DIR>(k), coords.template Xf<X3DIR>(k + 1)};

    dxf = {xb[1] - xb[0], xb[3] - xb[2], xb[5] - xb[4]};
    min_dx = std::min(std::min(dxf[0], dxf[1]), dxf[2]);
    inv_dx = 1.0 / dx<X1DIR>();
    inv_dy = 1.0 / dx<X2DIR>();
    inv_dz = 1.0 / dx<X3DIR>();
    vol = coords.CellVolume(k, j, i);
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real xlo() const {
    return xb[2 * (dir - 1)];
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real xhi() const {
    return xb[2 * (dir - 1) + 1];
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real dx() const {
    return dxf[dir - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real min_size() const { return min_dx; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real volume() const { return vol; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real ne_node(const int kk, const int jj, const int ii) const {
    return v(b, TE::NN, nen_t(), k + kk * dk, j + jj * dj, i + ii);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real ne() const { return v(b, ne_t(), k, j, i); }

  KOKKOS_INLINE_FUNCTION
  Real ne(const Real x, const Real y, const Real z) const {
    Real wxb = (x - xlo<X1DIR>()) * inv_dx;
    wxb = std::max(0.0, std::min(wxb, 1.0));
    Real wyb = (ndim > 1) * (y - xlo<X2DIR>()) * inv_dy;
    wyb = std::max(0.0, std::min(wyb, 1.0));
    Real wzb = (ndim > 2) * (z - xlo<X3DIR>()) * inv_dz;
    wzb = std::max(0.0, std::min(wzb, 1.0));

    Real val = 0.0;
    for (int kk = 0; kk <= (ndim > 2); kk++) {
      const Real wz = (1 - kk) * (1.0 - wzb) + kk * wzb;
      for (int jj = 0; jj <= (ndim > 1); jj++) {
        const Real wy = (1 - jj) * (1.0 - wyb) + jj * wyb;
        for (int ii = 0; ii <= 1; ii++) {
          const Real wx = (1 - ii) * (1.0 - wxb) + ii * wxb;
          val += wz * wy * wx * ne_node(kk, jj, ii);
        }
      }
    }

    return val;
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> grad_ne(const Real x, const Real y, const Real z) const {
    Real wxb = (x - xlo<X1DIR>()) * inv_dx;
    Real wyb = (ndim > 1) * (y - xlo<X2DIR>()) * inv_dy;
    Real wzb = (ndim > 2) * (z - xlo<X3DIR>()) * inv_dz;
    std::array<Real, 3> gne{0.0, 0.0, 0.0};
    for (int d = 0; d < ndim; d++) {
      for (int kk = 0; kk <= (ndim > 2); kk++) {
        const Real wz = (d != 2) * ((1 - kk) * (1.0 - wzb) + kk * wzb) +
                        (d == 2) * (-1 + 2 * kk) * inv_dz;
        for (int jj = 0; jj <= (ndim > 1); jj++) {
          const Real wy = (d != 1) * ((1 - jj) * (1.0 - wyb) + jj * wyb) +
                          (d == 1) * (-1 + 2 * jj) * inv_dy;
          for (int ii = 0; ii <= 1; ii++) {
            const Real wx = (d != 0) * ((1 - ii) * (1.0 - wxb) + ii * wxb) +
                            (d == 0) * (-1 + 2 * ii) * inv_dx;
            gne[d] += wz * wy * wx * ne_node(kk, jj, ii);
          }
        }
      }
    }

    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // we have {dn/dr, dn/dz, and dn/phi} from above, but we want
      // {dn/dx, dn/dy, dn/dz}, so transform
      const Real r = std::sqrt(x * x + y * y);
      gne[2] = gne[1];
      const Real gnr = gne[0] / r;
      gne[0] = gnr * x;
      gne[1] = gnr * y;
    }
    return gne;
  }

  int b, i, j, k;
  std::array<Real, 6> xb;
  std::array<Real, 3> dxf;
  Real min_dx, inv_dx, inv_dy, inv_dz, vol;
  const int dj, dk, ndim;
  T &v;
};

template <typename T>
class ParticlePusher {
 public:
  KOKKOS_FUNCTION
  ParticlePusher(LaserParticle &lpin, T &ci) : lp(lpin), c(ci) {
    using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
    accel_coeff = lp.lambda * lp.lambda * pc::qe * pc::qe / (2.0 * M_PI * pc::me);
  }

  KOKKOS_INLINE_FUNCTION
  std::pair<Real, Real> step_to_boundary(const Real tstop) {
    // use velocity verlet to integrate the particle to the next boundary
    Real dtmax = tstop - lp.t;
    // get initial accleration and velocity magnitude
    auto accel = c.grad_ne(lp.x[0], lp.x[1], lp.x[2]);
    for (int i = 0; i < 3; i++)
      accel[i] *= -accel_coeff;
    auto vmag0 = lp.vmag();

    // compute step size to put particle close to boundary
    Real h = get_converged_h(lp, accel, 5);
    lp.t += h;
    for (int i = 0; i < 3; i++) {
      // kick
      lp.v[i] += 0.5 * h * accel[i];
      // drift
      lp.x[i] += h * lp.v[i];
    }
    // get new acceleration
    accel = c.grad_ne(lp.x[0], lp.x[1], lp.x[2]);
    for (int i = 0; i < 3; i++) {
      accel[i] *= -accel_coeff;
      // kick
      lp.v[i] += 0.5 * h * accel[i];
    }
    auto vmag1 = lp.vmag();
    auto speed = 0.5 * (vmag0 + vmag1);
    Real path = 0.5 * h * speed;

    // finish it off
    path += rk1(lp, accel, vmag1);

    // and make it exactly on a boundary
    snap_to_face(lp);

    return {path, speed};
  }

 private:
  LaserParticle &lp;
  T &c;
  Real accel_coeff;

  KOKKOS_INLINE_FUNCTION
  void snap_to_face(LaserParticle &p) {
    // both of these fuzz factors deal with finite precision related to face/particle
    // position comparisons and calculations.  The nudge pushes the particle just a bit
    // more into the cell it is entering to ensure the index calc comes out right in
    // the next cell the particle enters
    constexpr Real fuzzy_face = 2 * std::numeric_limits<Real>::epsilon();
    constexpr Real fuzzy_nudge = 20 * std::numeric_limits<Real>::epsilon();
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      for (int i = 0; i < c.ndim; i++) {
        if (std::abs(p.x[i] - c.xb[2 * i]) < fuzzy_face)
          p.x[i] = c.xb[2 * i] + 10 * sign(p.v[i]) * fuzzy_nudge;
        if (std::abs(p.x[i] - c.xb[2 * i + 1]) < fuzzy_face)
          p.x[i] = c.xb[2 * i + 1] + 10 * sign(p.v[i]) * fuzzy_nudge;
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      Real r0 = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
      Real vr = (p.x[0] * p.v[0] + p.x[1] * p.v[1]) / r0;
      if (std::abs(r0 - c.xb[0]) < fuzzy_face) {
        auto r = c.xb[0] + 10 * sign(vr) * fuzzy_nudge;
        p.x[0] *= (r / r0);
        p.x[1] *= (r / r0);
      }
      if (std::abs(r0 - c.xb[1]) < fuzzy_face) {
        auto r = c.xb[1] + 10 * sign(vr) * fuzzy_nudge;
        p.x[0] *= (r / r0);
        p.x[1] *= (r / r0);
      }
      if (std::abs(p.x[2] - c.xb[2]) < fuzzy_face)
        p.x[2] = c.xb[2] + 10 * sign(p.v[2]) * fuzzy_nudge;
      if (std::abs(p.x[2] - c.xb[3]) < fuzzy_face)
        p.x[2] = c.xb[3] + 10 * sign(p.v[2]) * fuzzy_nudge;
    } else {
      PARTHENON_FAIL("Unsupported coordinate system in lasers.");
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real get_rk1_h(LaserParticle &p) {
    Real hmin = std::numeric_limits<Real>::max();
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      for (int i = 0; i < c.ndim; i++) {
        Real h = (c.xb[2 * i + (p.v[i] > 0)] - p.x[i]) / p.v[i];
        h = (p.v[i] == 0.0 ? std::numeric_limits<Real>::max() : h);
        hmin = std::min(h, hmin);
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      const Real r = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
      const Real vr = (p.x[0] * p.v[0] + p.x[1] * p.v[1]) / r;
      Real h = (c.xb[(vr > 0)] - r) / vr;
      hmin = std::min(h, hmin);
      h = (c.xb[2 + (p.v[2] > 0)] - p.x[2]) / p.v[2];
      hmin = std::min(h, hmin);
    } else {
      PARTHENON_FAIL("Unsupported coordinate system in lasers.");
    }
    return hmin;
  }

  KOKKOS_INLINE_FUNCTION
  std::pair<Real, int> get_rk1_hdir(LaserParticle &p) {
    Real hmin = std::numeric_limits<Real>::max();
    int hdir;
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      for (int i = 0; i < c.ndim; i++) {
        Real h = (c.xb[2 * i + (p.v[i] > 0)] - p.x[i]) / p.v[i];
        h = (p.v[i] == 0.0 ? std::numeric_limits<Real>::max() : h);
        if (h < hmin) {
          hmin = h;
          hdir = i;
        }
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      const Real r = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
      const Real vr = (p.x[0] * p.v[0] + p.x[1] * p.v[1]) / r;
      Real h = (c.xb[(vr > 0)] - r) / vr;
      if (h < hmin) {
        hmin = h;
        hdir = 0;
      }
      h = (c.xb[2 + (p.v[2] > 0)] - p.x[2]) / p.v[2];
      if (h < hmin) {
        hmin = h;
        hdir = 1;
      }
    } else {
      PARTHENON_FAIL("Unsupported coordinate system in lasers.");
    }
    return {hmin, hdir};
  }

  template <typename Array_t>
  KOKKOS_INLINE_FUNCTION Real get_converged_h(LaserParticle &p, Array_t &accel,
                                              const int max_iter) {
    // This is a relative tolerance parameter on step size convergence.
    // Results don't seem sensitive to this as long as it's "small" but
    // this function gets expensive when it gets too small.  1.e-3 seems
    // like a reasonable compromise in practice and I doubt we'll ever
    // need to think about this number again.
    constexpr Real h_tol = 1.e-3;
    // This tolerance allows us to skip the iteration altogether when
    // the acceleration is "small".  This corresponds to a 1% change in
    // velocity over the step, which again seems a reasonable compromise
    // and like above we'll likely never need to revisit this number.
    constexpr Real a_tol = 1.e-2;
    Real hnew = std::numeric_limits<Real>::max();
    Real dh = std::numeric_limits<Real>::max();
    int iter = 0;
    // simple estimate of step size
    auto [hk, hdir] = get_rk1_hdir(p);
    // only iterate if v will change appreciably
    Real max_a = std::abs(hk * accel[hdir] / p.v[hdir]);
    if (max_a < a_tol) return hk;
    // re-use max_a for radius if cylindrical
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>())
      max_a = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
    // now iterate using a simple fixed point
    // hnew = dx / (v + 0.5 hold a)
    do {
      if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
        for (int i = 0; i < c.ndim; i++) {
          const Real vs = p.v[i] + 0.5 * hk * accel[i];
          Real h = (c.xb[2 * i + (vs > 0)] - p.x[i]) / vs;
          h = (vs == 0.0 ? std::numeric_limits<Real>::max() : h);
          hnew = std::min(h, hnew);
        }
      } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
        const Real vsx = p.v[0] + 0.5 * hk * accel[0];
        const Real vsy = p.v[1] + 0.5 * hk * accel[1];
        const Real vr = (p.x[0] * vsx + p.x[1] * vsy) / max_a;
        Real h = (c.xb[(vr > 0)] - max_a) / vr;
        hnew = std::min(hnew, (vr == 0 ? std::numeric_limits<Real>::max() : h));
        h = (c.xb[2 + (p.v[2] > 0)] - p.x[2]) / p.v[2];
        hnew = std::min(hnew, (p.v[2] == 0 ? std::numeric_limits<Real>::max() : h));
      } else {
        PARTHENON_FAIL("Unsupported coordinate system in lasers.");
      }
      dh = std::abs(hnew - hk) / hk;
      hk = hnew;
      iter++;
    } while (iter < max_iter && dh > h_tol);
    return hnew;
  }

  template <typename Array_t>
  KOKKOS_INLINE_FUNCTION Real rk1(LaserParticle &p, Array_t &accel, const Real vmag) {
    Real h = get_rk1_h(p);
    Real path_length = vmag * h;
    p.t += h;
    for (int i = 0; i < 3; i++) {
      p.x[i] += h * p.v[i];
      p.v[i] += h * accel[i];
    }
    return path_length;
  }
};

template <typename VAR>
KOKKOS_INLINE_FUNCTION Real cell_to_node_4thorder(const VAR &field, const int k,
                                                  const int j, const int i, const int dk,
                                                  const int dj) {
  // BEWARE: this is an unlimited 4th-order interpolant, which can behave badly
  static constexpr std::array<Real, 4> wgts{343.0 / 1728.0, -49.0 / 1728.0, 7.0 / 1728.0,
                                            -1.0 / 1728.0};
  static constexpr std::array<int, 4> off_map{1, 0, 0, 1};
  Real node_val = 0.0;
  for (int kk = -2; kk < 2; kk++) {
    const int koff = off_map[kk + 2];
    for (int jj = -2; jj < 2; jj++) {
      const int joff = off_map[jj + 2];
      for (int ii = -2; ii < 2; ii++) {
        const int off = koff + joff + off_map[ii + 2];
        node_val += wgts[off] * field(k + kk * dk, j + jj * dj, i + ii);
      }
    }
  }
  return std::max(node_val, 0.0);
}

template <typename VAR>
KOKKOS_INLINE_FUNCTION Real cell_to_node_2ndorder(const VAR &field, const int k,
                                                  const int j, const int i, const int dk,
                                                  const int dj) {
  Real node_val = 0.0;
  int cnt = 0;
  for (int kk = -1 * dk; kk <= 0; kk++) {
    for (int jj = -1 * dj; jj <= 0; jj++) {
      for (int ii = -1; ii <= 0; ii++) {
        node_val += field(k + kk, j + jj, i + ii);
        cnt++;
      }
    }
  }
  return node_val / cnt;
}

template <int order>
TaskStatus SetElectronsImpl(MeshData<Real> *md) {
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  using TE = parthenon::TopologicalElement;
  using ne_t = ccbulk::electron_number_density;
  using node_ne_t = node_variables::electron_number_density;

  auto pm = md->GetMeshPointer();
  auto resolved_pkgs = pm->resolved_packages.get();
  static auto desc =
      parthenon::MakePackDescriptor<ne_t, node_ne_t, ccmat::rho>(resolved_pkgs);
  auto v = riot::GetPack(desc, md);

  int dj = pm->ndim > 1 ? 1 : 0;
  int dk = pm->ndim > 2 ? 1 : 0;

  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::NN);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto dkl = dk;
        const auto djl = dj;

        auto nodal_ne = RiotLoop::make_var_view(idx_range, v, node_ne_t());
        auto cell_ne = RiotLoop::make_var_view(idx_range, v, ne_t());
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          if constexpr (order == 2) {
            nodal_ne(k, j, i) = cell_to_node_2ndorder(cell_ne, k, j, i, dkl, djl);
          } else if constexpr (order == 4) {
            nodal_ne(k, j, i) = cell_to_node_4thorder(cell_ne, k, j, i, dkl, djl);
          }
        });
      });

  return TaskStatus::complete;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskCollection LaserUpdateTasks(Mesh *pmesh, const Real t0, const Real dt);
TaskCollection LaserDepositionTasks(Mesh *pm, const Real dt);
TaskStatus InitializeLaserSweep(MeshData<Real> *md, MeshData<Real> *md_dudt,
                                const Real t0, const Real dt);
TaskStatus Update(MeshData<Real> *md, MeshData<Real> *md_dudt, const Real t0,
                  const Real dt); //, AllReduce<int> &num_active);
TaskStatus SetElectrons(MeshData<Real> *md);
TaskStatus UpdateMatEnergy(MeshData<Real> *md, const Real dt);
bool CheckDt(Mesh *pm, Real *dt);
Real EstimateTimestepMesh(MeshData<Real> *md);

} // namespace Laser

#endif // LASER_LASER_HPP_
