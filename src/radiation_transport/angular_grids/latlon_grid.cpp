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

LatLonGrid::LatLonGrid(int ntheta, int nphi)
    : ntheta_(ntheta), nphi_(nphi), num_neighbors("num_neighbors", 1),
      ind_neighbors("ind_neighbors", 1, 1),
      ind_neighbors_edges("ind_neighbors_edges", 1, 1), weights("weights", 1),
      arc_weights("arc_weights", 1, 1), cart_pos("cart_pos", 1, 1),
      cart_pos_mid("cart_pos_mid", 1, 1, 1), gflux("gflux", 1, 1) {

  nangles = ntheta * nphi;

  ParArrayHost<Real> theta_v("theta_v", ntheta);
  ParArrayHost<Real> theta_f("theta_f", ntheta + 1);
  ParArrayHost<Real> costheta_v("costheta_v", ntheta);
  ParArrayHost<Real> costheta_f("costheta_f", ntheta + 1);
  ParArrayHost<Real> phi_v("phi_v", nphi);
  ParArrayHost<Real> phi_f("phi_f", nphi + 1);

  ComputeThetaLevels(theta_v, theta_f, costheta_v, costheta_f);
  ComputePhiAngles(phi_v, phi_f);

  num_neighbors.Resize(nangles);
  ind_neighbors.Resize(nangles, 4);
  ind_neighbors_edges.Resize(nangles, 4);
  weights.Resize(nangles);
  arc_weights.Resize(nangles, 4);
  cart_pos.Resize(nangles, 3);
  cart_pos_mid.Resize(nangles, 4, 3);
  gflux.Resize(nangles, 4);

  ComputeCartesianDirections(theta_v, phi_v);
  ComputeWeights(costheta_f, phi_f);
  ComputeNeighborConnectivity();
  ComputeArcLengths(theta_f, phi_f, theta_v, phi_v);
  ComputeUnitFluxes(theta_v, phi_v, theta_f, phi_f);
}

//----------------------------------------------------------------------------------------
LatLonGrid::~LatLonGrid() {}

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
                                            ParArrayHost<Real> &phi_v) {
  auto cart_pos_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos);
  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      const Real sintheta = std::sin(theta_v(it));
      cart_pos_h(idx, 0) = sintheta * std::cos(phi_v(ip));
      cart_pos_h(idx, 1) = sintheta * std::sin(phi_v(ip));
      cart_pos_h(idx, 2) = std::cos(theta_v(it));
    }
  }
  Kokkos::deep_copy(cart_pos, cart_pos_h);
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeWeights(ParArrayHost<Real> &costheta_f,
                                ParArrayHost<Real> &phi_f) {
  auto weights_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

  for (int it = 0; it < ntheta_; ++it) {
    for (int ip = 0; ip < nphi_; ++ip) {
      const int idx = it * nphi_ + ip;
      const Real omega =
          (costheta_f(it) - costheta_f(it + 1)) * (phi_f(ip + 1) - phi_f(ip));
      weights_h(idx) = omega / (4.0 * M_PI);
    }
  }

  Real wsum = 0.0;
  for (int i = 0; i < nangles; ++i) {
    wsum += weights_h(i);
  }

  Kokkos::deep_copy(weights, weights_h);
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeNeighborConnectivity() {
  auto num_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), num_neighbors);
  auto ind_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors);
  auto ind_neighbors_edges_h =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors_edges);

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

  Kokkos::deep_copy(num_neighbors, num_neighbors_h);
  Kokkos::deep_copy(ind_neighbors, ind_neighbors_h);
  Kokkos::deep_copy(ind_neighbors_edges, ind_neighbors_edges_h);
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeArcLengths(ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                                   ParArrayHost<Real> &theta_v,
                                   ParArrayHost<Real> &phi_v) {
  auto arc_weights_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), arc_weights);
  auto cart_pos_mid_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos_mid);
  auto cart_pos_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos);
  auto ind_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors);
  auto num_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), num_neighbors);

  Kokkos::deep_copy(cart_pos_h, cart_pos);
  Kokkos::deep_copy(ind_neighbors_h, ind_neighbors);
  Kokkos::deep_copy(num_neighbors_h, num_neighbors);

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

  auto ind_neighbors_edges_h =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors_edges);
  Kokkos::deep_copy(ind_neighbors_edges_h, ind_neighbors_edges);

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

  Kokkos::deep_copy(arc_weights, arc_weights_h);
  Kokkos::deep_copy(cart_pos_mid, cart_pos_mid_h);
}

//----------------------------------------------------------------------------------------
void LatLonGrid::ComputeUnitFluxes(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                                   ParArrayHost<Real> &theta_f,
                                   ParArrayHost<Real> &phi_f) {
  auto gflux_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), gflux);
  auto cart_pos_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos);
  auto cart_pos_mid_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos_mid);
  auto ind_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors);
  auto num_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), num_neighbors);

  Kokkos::deep_copy(cart_pos_h, cart_pos);
  Kokkos::deep_copy(cart_pos_mid_h, cart_pos_mid);
  Kokkos::deep_copy(ind_neighbors_h, ind_neighbors);
  Kokkos::deep_copy(num_neighbors_h, num_neighbors);

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
          gflux_h(idx, nb) = ym * (SQR(xm) + SQR(ym)) * unit_psi;
        } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
          const Real isz = 1.0 / std::sqrt(1.0 - SQR(zm));
          gflux_h(idx, nb) = isz * (SQR(xm) + SQR(ym)) * unit_zeta;
        }
      }
    }
  }

  auto ind_neighbors_edges_h =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors_edges);
  Kokkos::deep_copy(ind_neighbors_edges_h, ind_neighbors_edges);

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

  Kokkos::deep_copy(gflux, gflux_h);
}
