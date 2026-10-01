//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see licenses/bsd_athenak.txt file for details
//========================================================================================
// This file was made in part with generative AI.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <parthenon/package.hpp>

#include "radiation_transport/angular_grids/latlon_grid.hpp"

using namespace parthenon;

//----------------------------------------------------------------------------------------
// NOTE(@pdmullen): The latlon grid infrastructure closesly resembes the AthenaK
// geodesic grid implementation (see copyrights info above).
LatLonGrid::LatLonGrid(int ntheta, int nphi, bool fv_fix)
    : ntheta_(ntheta), nphi_(nphi), num_neighbors("num_neighbors", 1),
      ind_neighbors("ind_neighbors", 1, 1),
      ind_neighbors_edges("ind_neighbors_edges", 1, 1), weights("weights", 1),
      arc_weights("arc_weights", 1, 1), cart_pos("cart_pos", 1, 1),
      cart_pos_unit("cart_pos_unit", 1, 1), cart_pos_mid("cart_pos_mid", 1, 1, 1),
      gflux("gflux", 1, 1) {

  nangles = ntheta * nphi;

  // Host Arrays
  ParArrayHost<Real> theta_v("theta_v", ntheta);
  ParArrayHost<Real> theta_f("theta_f", ntheta + 1);
  ParArrayHost<Real> costheta_v("costheta_v", ntheta);
  ParArrayHost<Real> costheta_f("costheta_f", ntheta + 1);
  ParArrayHost<Real> phi_v("phi_v", nphi);
  ParArrayHost<Real> phi_f("phi_f", nphi + 1);
  ParArrayHost<int> num_neighbors_h("num_neighbors_h", nangles);
  ParArrayHost<int> ind_neighbors_h("ind_neighbors_h", nangles, 4);
  ParArrayHost<int> ind_neighbors_edges_h("ind_neighbors_edges_h", nangles, 4);
  ParArrayHost<Real> weights_h("weights_h", nangles);
  ParArrayHost<Real> arc_weights_h("arc_weights_h", nangles, 4);
  ParArrayHost<Real> cart_pos_h("cart_pos_h", nangles, 3);
  ParArrayHost<Real> cart_pos_unit_h("cart_pos_unit_h", nangles, 3);
  ParArrayHost<Real> cart_pos_mid_h("cart_pos_mid_h", nangles, 4, 3);
  ParArrayHost<Real> gflux_h("gflux_h", nangles, 4);

  num_neighbors.Resize(nangles);
  ind_neighbors.Resize(nangles, 4);
  ind_neighbors_edges.Resize(nangles, 4);
  weights.Resize(nangles);
  arc_weights.Resize(nangles, 4);
  cart_pos.Resize(nangles, 3);
  cart_pos_unit.Resize(nangles, 3);
  cart_pos_mid.Resize(nangles, 4, 3);
  gflux.Resize(nangles, 4);

  ComputeThetaLevels(theta_v, theta_f, costheta_v, costheta_f);
  ComputePhiAngles(phi_v, phi_f);
  ComputeCartesianDirections(theta_v, phi_v, cart_pos_h, cart_pos_unit_h);
  ComputeWeights(costheta_f, phi_f, weights_h);
  ComputeNeighborConnectivity(num_neighbors_h, ind_neighbors_h, ind_neighbors_edges_h);
  ComputeArcLengths(theta_f, phi_f, theta_v, phi_v, arc_weights_h, cart_pos_mid_h,
                    ind_neighbors_h, ind_neighbors_edges_h, num_neighbors_h);
  ComputeUnitFluxes(theta_v, phi_v, theta_f, phi_f, gflux_h, cart_pos_mid_h,
                    ind_neighbors_h, ind_neighbors_edges_h, num_neighbors_h);
  if (fv_fix) {
    ApplyFiniteVolumeCorrections(theta_f, phi_f, cart_pos_h, weights_h, arc_weights_h,
                                 gflux_h, num_neighbors_h, ind_neighbors_h);
  }

  // deep copy
  Kokkos::deep_copy(num_neighbors, num_neighbors_h);
  Kokkos::deep_copy(ind_neighbors, ind_neighbors_h);
  Kokkos::deep_copy(ind_neighbors_edges, ind_neighbors_edges_h);
  Kokkos::deep_copy(weights, weights_h);
  Kokkos::deep_copy(arc_weights, arc_weights_h);
  Kokkos::deep_copy(cart_pos, cart_pos_h);
  Kokkos::deep_copy(cart_pos_unit, cart_pos_unit_h);
  Kokkos::deep_copy(cart_pos_mid, cart_pos_mid_h);
  Kokkos::deep_copy(gflux, gflux_h);
}

//----------------------------------------------------------------------------------------
LatLonGrid::~LatLonGrid() {}

//----------------------------------------------------------------------------------------
//! \fn void LatLonGrid::ApplyFiniteVolumeCorrections
//! \brief Replace the cell-centered normal directions with their exact solid-angle
//! averages <n_i> = (1/dOmega) int n_i dOmega for the finite-volume transport speeds;
//! cart_pos_unit retains the true centroid unit vectors (if needed).
//!
//! For a lat-lon cell [theta_lo,theta_hi] x [phi_lo,phi_hi] the averages are closed-form:
//!   <n_x> = [int sin^2 theta dtheta][sin phi_hi - sin phi_lo] / (dcos theta * dphi)
//!   <n_y> = [int sin^2 theta dtheta][cos phi_lo - cos phi_hi] / (dcos theta * dphi)
//!   <n_z> = 1/2 (cos theta_lo + cos theta_hi)
//! with int sin^2 theta dtheta = [theta/2 - sin(2 theta)/4].
void LatLonGrid::ApplyFiniteVolumeCorrections(
    ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
    ParArrayHost<Real> &cart_pos_h, ParArrayHost<Real> &weights_h,
    ParArrayHost<Real> &arc_weights_h, ParArrayHost<Real> &gflux_h,
    ParArrayHost<int> &num_neighbors_h, ParArrayHost<int> &ind_neighbors_h) {
  // Solid-angle-averaged cosines for every transported component.
  for (int it = 0; it < ntheta_; ++it) {
    const Real th_lo = theta_f(it), th_hi = theta_f(it + 1);
    const Real i2 =
        0.5 * (th_hi - th_lo) - 0.25 * (std::sin(2.0 * th_hi) - std::sin(2.0 * th_lo));
    const Real dcos = std::cos(th_lo) - std::cos(th_hi);
    const Real navg_z = 0.5 * (std::cos(th_lo) + std::cos(th_hi));
    for (int ip = 0; ip < nphi_; ++ip) {
      const int n = it * nphi_ + ip;
      const Real ph_lo = phi_f(ip), ph_hi = phi_f(ip + 1);
      const Real dphi = ph_hi - ph_lo;
      const Real navg_x = i2 * (std::sin(ph_hi) - std::sin(ph_lo)) / (dcos * dphi);
      const Real navg_y = i2 * (std::cos(ph_lo) - std::cos(ph_hi)) / (dcos * dphi);
      cart_pos_h(n, 0) = navg_x;
      cart_pos_h(n, 1) = navg_y;
      cart_pos_h(n, 2) = navg_z;
    }
  }

  // Consistency check: the curvature-direction cosine already stored in cart_pos_h is the
  // solid-angle average <n_curv>.  Confirm it agrees to round-off with g_a built from the
  // exact symmetrized edge arrays the divfa operator uses (both equal the same
  // divergence-theorem integral); this validates the edge gflux/arc_weights construction.
  if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>() ||
                parthenon::IsCoord<parthenon::UniformSpherical>()) {
    constexpr int curv_comp = parthenon::IsCoord<parthenon::UniformSpherical>() ? 2 : 0;
    constexpr Real kappa = parthenon::IsCoord<parthenon::UniformSpherical>() ? 2.0 : 1.0;
    for (int n = 0; n < nangles; ++n) {
      Real ga = 0.0;
      for (int nb = 0; nb < num_neighbors_h(n); ++nb)
        ga += gflux_h(n, nb) * arc_weights_h(n, nb);
      const Real ga_curv = ga / (weights_h(n) * kappa);
      PARTHENON_REQUIRE(std::abs(ga_curv - cart_pos_h(n, curv_comp)) < 1.0e-12,
                        "Lat-lon g_a disagrees with solid-angle average <n_curv>.");
    }
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeThetaLevels(ParArrayHost<Real> &theta_v,
                                    ParArrayHost<Real> &theta_f,
                                    ParArrayHost<Real> &costheta_v,
                                    ParArrayHost<Real> &costheta_f) {
  const Real dcostheta = 2.0 / ntheta_;
  for (int i = 0; i <= ntheta_; ++i) {
    costheta_f(i) = 1.0 - i * dcostheta;
    theta_f(i) = std::acos(costheta_f(i));
  }

  for (int i = 0; i < ntheta_; ++i) {
    const Real cu = costheta_f(i), cl = costheta_f(i + 1);
    const Real tu = theta_f(i), tl = theta_f(i + 1);
    theta_v(i) = ((tu * cu - tl * cl) - (std::sin(tu) - std::sin(tl))) / (cu - cl);
    costheta_v(i) = std::cos(theta_v(i));
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputePhiAngles(ParArrayHost<Real> &phi_v, ParArrayHost<Real> &phi_f) {
  const Real dphi = 2.0 * M_PI / nphi_;
  for (int i = 0; i <= nphi_; ++i)
    phi_f(i) = i * dphi;
  for (int i = 0; i < nphi_; ++i)
    phi_v(i) = 0.5 * (phi_f(i) + phi_f(i + 1));
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeCartesianDirections(ParArrayHost<Real> &theta_v,
                                            ParArrayHost<Real> &phi_v,
                                            ParArrayHost<Real> &cart_pos_h,
                                            ParArrayHost<Real> &cart_pos_unit_h) {
  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      const Real sintheta = std::sin(theta_v(it));
      const Real nx = sintheta * std::cos(phi_v(ip));
      const Real ny = sintheta * std::sin(phi_v(ip));
      const Real nz = std::cos(theta_v(it));
      cart_pos_h(idx, 0) = nx;
      cart_pos_h(idx, 1) = ny;
      cart_pos_h(idx, 2) = nz;
      cart_pos_unit_h(idx, 0) = nx;
      cart_pos_unit_h(idx, 1) = ny;
      cart_pos_unit_h(idx, 2) = nz;
    }
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeWeights(ParArrayHost<Real> &costheta_f, ParArrayHost<Real> &phi_f,
                                ParArrayHost<Real> &weights_h) {
  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      const Real omega =
          (costheta_f(it) - costheta_f(it + 1)) * (phi_f(ip + 1) - phi_f(ip));
      weights_h(idx) = omega / (4.0 * M_PI);
    }
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeNeighborConnectivity(ParArrayHost<int> &num_neighbors_h,
                                             ParArrayHost<int> &ind_neighbors_h,
                                             ParArrayHost<int> &ind_neighbors_edges_h) {
  for (int i = 0; i < nangles; ++i) {
    num_neighbors_h(i) = 0;
    for (int nb = 0; nb < 4; ++nb) {
      ind_neighbors_h(i, nb) = std::numeric_limits<int>::max();
      ind_neighbors_edges_h(i, nb) = -1;
    }
  }

  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      int nneigh = 0;

      const int j_left = it * nphi_ + ((ip - 1 + nphi_) % nphi_);
      if (j_left != idx) ind_neighbors_h(idx, nneigh++) = j_left;

      const int j_right = it * nphi_ + ((ip + 1) % nphi_);
      if (j_right != idx) ind_neighbors_h(idx, nneigh++) = j_right;

      if (it > 0) ind_neighbors_h(idx, nneigh++) = (it - 1) * nphi_ + ip;
      if (it < ntheta_ - 1) ind_neighbors_h(idx, nneigh++) = (it + 1) * nphi_ + ip;

      num_neighbors_h(idx) = nneigh;
    }
  }

  // Set up ind_neighbors_edges: for each edge, find the reciprocal neighbor index
  // This must be done after all neighbors are set up
  for (int n = 0; n < nangles; ++n) {
    for (int nb = 0; nb < num_neighbors_h(n); ++nb) {
      const int j = ind_neighbors_h(n, nb);
      // Find which neighbor index in j's list points back to n
      bool found = false;
      for (int nb_j = 0; nb_j < num_neighbors_h(j); ++nb_j) {
        if (ind_neighbors_h(j, nb_j) == n) {
          ind_neighbors_edges_h(n, nb) = nb_j;
          found = true;
          break;
        }
      }
      if (!found) {
        ind_neighbors_edges_h(n, nb) = -1;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeArcLengths(ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                                   ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                                   ParArrayHost<Real> &arc_weights_h,
                                   ParArrayHost<Real> &cart_pos_mid_h,
                                   ParArrayHost<int> &ind_neighbors_h,
                                   ParArrayHost<int> &ind_neighbors_edges_h,
                                   ParArrayHost<int> &num_neighbors_h) {
  for (int i = 0; i < nangles; ++i)
    for (int nb = 0; nb < 4; ++nb)
      arc_weights_h(i, nb) = 0.0;

  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      const Real theta_lo = theta_f(it), theta_hi = theta_f(it + 1);
      const Real phi_lo = phi_f(ip), phi_hi = phi_f(ip + 1);

      for (int nb = 0; nb < num_neighbors_h(idx); ++nb) {
        const int j = ind_neighbors_h(idx, nb);
        const int it_j = j / nphi_, ip_j = j % nphi_;

        Real theta_edge, phi_edge, sintheta, arc_weight;
        if (it_j == it && ip_j != ip) {
          theta_edge = theta_v(it);
          phi_edge = ((ip_j + nphi_ - ip) % nphi_ == 1) ? phi_hi : phi_lo;
          sintheta = std::sin(theta_edge);
          arc_weight = (std::cos(theta_lo) - std::cos(theta_hi)) / (4.0 * M_PI);
        } else {
          theta_edge = (it_j < it) ? theta_lo : theta_hi;
          phi_edge = 0.5 * (phi_lo + phi_hi);
          sintheta = std::sin(theta_edge);
          arc_weight = sintheta * (phi_hi - phi_lo) / (4.0 * M_PI);
        }

        cart_pos_mid_h(idx, nb, 0) = sintheta * std::cos(phi_edge);
        cart_pos_mid_h(idx, nb, 1) = sintheta * std::sin(phi_edge);
        cart_pos_mid_h(idx, nb, 2) = std::cos(theta_edge);
        arc_weights_h(idx, nb) = arc_weight;
      }
    }
  }

  for (int n = 0; n < nangles; ++n) {
    for (int nb = 0; nb < num_neighbors_h(n); ++nb) {
      const Real tarc = arc_weights_h(n, nb);
      const Real narc =
          arc_weights_h(ind_neighbors_h(n, nb), ind_neighbors_edges_h(n, nb));
      const Real arc_avg = 0.5 * (tarc + narc);
      arc_weights_h(n, nb) = arc_avg;
      arc_weights_h(ind_neighbors_h(n, nb), ind_neighbors_edges_h(n, nb)) = arc_avg;
    }
  }
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeUnitFluxes(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                                   ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                                   ParArrayHost<Real> &gflux_h,
                                   ParArrayHost<Real> &cart_pos_mid_h,
                                   ParArrayHost<int> &ind_neighbors_h,
                                   ParArrayHost<int> &ind_neighbors_edges_h,
                                   ParArrayHost<int> &num_neighbors_h) {
  for (int i = 0; i < nangles; ++i)
    for (int nb = 0; nb < 4; ++nb)
      gflux_h(i, nb) = 0.0;

  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      for (int nb = 0; nb < num_neighbors_h(idx); ++nb) {
        const int j = ind_neighbors_h(idx, nb);
        const int it_j = j / nphi_, ip_j = j % nphi_;
        const Real xm = cart_pos_mid_h(idx, nb, 0);
        const Real ym = cart_pos_mid_h(idx, nb, 1);
        const Real zm = cart_pos_mid_h(idx, nb, 2);

        Real unit_zeta, unit_psi;
        if (it_j == it && ip_j != ip) {
          unit_zeta = 0.0;
          unit_psi = ((ip_j + nphi_ - ip) % nphi_ == 1) ? +1.0 : -1.0;
        } else {
          unit_psi = 0.0;
          unit_zeta = (it_j < it) ? -1.0 : +1.0;
        }

        if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
          const Real sinth = std::sqrt(SQR(xm) + SQR(ym));
          const Real sin_phi = (sinth > 0.0) ? ym / sinth : 0.0;
          const Real th_lo = theta_f(it), th_hi = theta_f(it + 1);
          const Real i2 = 0.5 * (th_hi - th_lo) -
                          0.25 * (std::sin(2.0 * th_hi) - std::sin(2.0 * th_lo));
          const Real dcos = std::cos(th_lo) - std::cos(th_hi);
          gflux_h(idx, nb) = (dcos != 0.0) ? unit_psi * sin_phi * i2 / dcos : 0.0;
        } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
          const Real isz = 1.0 / std::sqrt(1.0 - SQR(zm));
          gflux_h(idx, nb) = isz * (SQR(xm) + SQR(ym)) * unit_zeta;
        }
      }
    }
  }

  for (int n = 0; n < nangles; ++n) {
    for (int nb = 0; nb < num_neighbors_h(n); ++nb) {
      const Real tgflx = gflux_h(n, nb);
      const Real ngflx = gflux_h(ind_neighbors_h(n, nb), ind_neighbors_edges_h(n, nb));
      const Real gflx_avg = 0.5 * (std::abs(tgflx) + std::abs(ngflx));
      gflux_h(n, nb) = std::copysign(gflx_avg, tgflx);
      gflux_h(ind_neighbors_h(n, nb), ind_neighbors_edges_h(n, nb)) =
          std::copysign(gflx_avg, ngflx);
    }
  }
}
