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

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Parthenon headers
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/angular_grids/geodesic_grid.hpp"

using namespace parthenon;

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters
// NOTE(@pdmullen): The geodesic grid infrastructure is taken from the AthenaK
// implementation and ported to the Parthenon framework by @pdmullen (see copyrights
// info above)

GeodesicGrid::GeodesicGrid(int nlev, int rotate, Real zpole, Real ppole)
    : nlevel(nlev), rotate_geo(rotate), num_neighbors("num_neighbors", 1),
      ind_neighbors("ind_neighbors", 1, 1),
      ind_neighbors_edges("ind_neighbors_edges", 1, 1), weights("weights", 1),
      arc_weights("arc_weights", 1, 1), cart_pos("cart_pos", 1, 1),
      cart_pos_mid("cart_pos_mid", 1, 1, 1), gflux("gflux", 1, 1),
      amesh_normals("amesh_normals", 1, 1, 1, 1), ameshp_normals("ameshp_normals", 1, 1),
      amesh_indices("amesh_indices", 1, 1, 1), ameshp_indices("ameshp_indices", 1) {
  if (nlevel > 0) { // construct geodesic mesh
    // number of angles
    nangles = 5 * 2 * SQR(nlevel) + 2;

    // Resize Arrays
    amesh_normals.Resize(5, 2 + nlevel, 2 + 2 * nlevel, 3);
    ameshp_normals.Resize(2, 3);
    amesh_indices.Resize(5, 2 + nlevel, 2 + 2 * nlevel);
    ameshp_indices.Resize(2);
    num_neighbors.Resize(nangles);
    ind_neighbors.Resize(nangles, 6);
    ind_neighbors_edges.Resize(nangles, 6);
    weights.Resize(nangles);
    arc_weights.Resize(nangles, 6);
    cart_pos.Resize(nangles, 3);
    cart_pos_mid.Resize(nangles, 6, 3);
    gflux.Resize(nangles, 6);

    // create mirror views
    auto num_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), num_neighbors);
    auto ind_neighbors_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors);
    auto ind_neighbors_edges_h =
        Kokkos::create_mirror_view(Kokkos::HostSpace(), ind_neighbors_edges);
    auto weights_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);
    auto arc_weights_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), arc_weights);
    auto cart_pos_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos);
    auto cart_pos_mid_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos_mid);
    auto gflux_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), gflux);

    // construction parameters
    Real sin_a = 2.0 / std::sqrt(5.0);
    Real cos_a = 1.0 / std::sqrt(5.0);
    Real p1[3] = {0.0, 0.0, 1.0};
    Real p2[3] = {sin_a, 0.0, cos_a};
    Real p3[3] = {sin_a * std::cos(0.2 * M_PI), sin_a * std::sin(0.2 * M_PI), -cos_a};
    Real p4[3] = {sin_a * std::cos(-0.4 * M_PI), sin_a * std::sin(-0.4 * M_PI), cos_a};
    Real p5[3] = {sin_a * std::cos(-0.2 * M_PI), sin_a * std::sin(-0.2 * M_PI), -cos_a};
    Real p6[3] = {0.0, 0.0, -1.0};

    // set pole normal components explicitly
    auto &apnorm = ameshp_normals;
    apnorm(0, 0) = 0.0;
    apnorm(0, 1) = 0.0;
    apnorm(0, 2) = 1.0;
    apnorm(1, 0) = 0.0;
    apnorm(1, 1) = 0.0;
    apnorm(1, 2) = -1.0;

    // get normal components of all other angle centers
    // start by filling in one of the five blocks
    auto &anorm = amesh_normals;
    int row_index = 1;
    for (int l = 0; l < nlevel; ++l) {
      int col_index = 1;
      for (int m = l; m < nlevel; ++m) {
        Real x = ((m - l + 1) * p2[0] + (nlevel - m - 1) * p1[0] + l * p4[0]) / nlevel;
        Real y = ((m - l + 1) * p2[1] + (nlevel - m - 1) * p1[1] + l * p4[1]) / nlevel;
        Real z = ((m - l + 1) * p2[2] + (nlevel - m - 1) * p1[2] + l * p4[2]) / nlevel;
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0, row_index, col_index, 0) = x / norm;
        anorm(0, row_index, col_index, 1) = y / norm;
        anorm(0, row_index, col_index, 2) = z / norm;
        col_index += 1;
      }
      for (int m = nlevel - l; m < nlevel; ++m) {
        Real x = ((nlevel - l) * p2[0] + (m - nlevel + l + 1) * p5[0] +
                  (nlevel - m - 1) * p4[0]) /
                 nlevel;
        Real y = ((nlevel - l) * p2[1] + (m - nlevel + l + 1) * p5[1] +
                  (nlevel - m - 1) * p4[1]) /
                 nlevel;
        Real z = ((nlevel - l) * p2[2] + (m - nlevel + l + 1) * p5[2] +
                  (nlevel - m - 1) * p4[2]) /
                 nlevel;
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0, row_index, col_index, 0) = x / norm;
        anorm(0, row_index, col_index, 1) = y / norm;
        anorm(0, row_index, col_index, 2) = z / norm;
        col_index += 1;
      }
      for (int m = l; m < nlevel; ++m) {
        Real x = ((m - l + 1) * p3[0] + (nlevel - m - 1) * p2[0] + l * p5[0]) / nlevel;
        Real y = ((m - l + 1) * p3[1] + (nlevel - m - 1) * p2[1] + l * p5[1]) / nlevel;
        Real z = ((m - l + 1) * p3[2] + (nlevel - m - 1) * p2[2] + l * p5[2]) / nlevel;
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0, row_index, col_index, 0) = x / norm;
        anorm(0, row_index, col_index, 1) = y / norm;
        anorm(0, row_index, col_index, 2) = z / norm;
        col_index += 1;
      }
      for (int m = nlevel - l; m < nlevel; ++m) {
        Real x = ((nlevel - l) * p3[0] + (m - nlevel + l + 1) * p6[0] +
                  (nlevel - m - 1) * p5[0]) /
                 nlevel;
        Real y = ((nlevel - l) * p3[1] + (m - nlevel + l + 1) * p6[1] +
                  (nlevel - m - 1) * p5[1]) /
                 nlevel;
        Real z = ((nlevel - l) * p3[2] + (m - nlevel + l + 1) * p6[2] +
                  (nlevel - m - 1) * p5[2]) /
                 nlevel;
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0, row_index, col_index, 0) = x / norm;
        anorm(0, row_index, col_index, 1) = y / norm;
        anorm(0, row_index, col_index, 2) = z / norm;
        col_index += 1;
      }
      row_index += 1;
    }

    // fill the other four patches by rotating the first one
    for (int ptch = 1; ptch < 5; ++ptch) {
      for (int l = 1; l < 1 + nlevel; ++l) {
        for (int m = 1; m < 1 + 2 * nlevel; ++m) {
          Real x0 = anorm(0, l, m, 0);
          Real y0 = anorm(0, l, m, 1);
          Real z0 = anorm(0, l, m, 2);
          anorm(ptch, l, m, 0) =
              (x0 * std::cos(ptch * 0.4 * M_PI) + y0 * std::sin(ptch * 0.4 * M_PI));
          anorm(ptch, l, m, 1) =
              (y0 * std::cos(ptch * 0.4 * M_PI) - x0 * std::sin(ptch * 0.4 * M_PI));
          anorm(ptch, l, m, 2) = z0;
        }
      }
    }

    // fill in the ghost cells of all blocks
    for (int i = 0; i < 3; ++i) {
      for (int bl = 0; bl < 5; ++bl) {
        for (int k = 0; k < nlevel; ++k) {
          anorm(bl, 0, k + 1, i) = anorm((bl + 4) % 5, k + 1, 1, i);
          anorm(bl, 0, k + nlevel + 1, i) = anorm((bl + 4) % 5, nlevel, k + 1, i);
          anorm(bl, k + 1, 2 * nlevel + 1, i) =
              anorm((bl + 4) % 5, nlevel, k + nlevel + 1, i);
          anorm(bl, k + 2, 0, i) = anorm((bl + 1) % 5, 1, k + 1, i);
          anorm(bl, nlevel + 1, k + 1, i) = anorm((bl + 1) % 5, 1, k + nlevel + 1, i);
          anorm(bl, nlevel + 1, k + nlevel + 1, i) =
              anorm((bl + 1) % 5, k + 2, 2 * nlevel, i);
        }
        anorm(bl, 1, 0, i) = apnorm(0, i);
        anorm(bl, nlevel + 1, 2 * nlevel, i) = apnorm(1, i);
        anorm(bl, 0, 2 * nlevel + 1, i) = anorm(bl, 0, 2 * nlevel, i);
      }
    }

    // generate 2d to 1d map
    auto &apind = ameshp_indices;
    auto &aind = amesh_indices;
    apind(0) = 5 * 2 * SQR(nlevel);
    apind(1) = 5 * 2 * SQR(nlevel) + 1;
    for (int ptch = 0; ptch < 5; ++ptch) {
      for (int l = 0; l < nlevel; ++l) {
        for (int m = 0; m < 2 * nlevel; ++m) {
          aind(ptch, l + 1, m + 1) = ptch * 2 * SQR(nlevel) + l * 2 * nlevel + m;
        }
      }
    }

    // fill ghost cells
    for (int bl = 0; bl < 5; ++bl) {
      for (int k = 0; k < nlevel; ++k) {
        aind(bl, 0, k + 1) = aind((bl + 4) % 5, k + 1, 1);
        aind(bl, 0, k + nlevel + 1) = aind((bl + 4) % 5, nlevel, k + 1);
        aind(bl, k + 1, 2 * nlevel + 1) = aind((bl + 4) % 5, nlevel, k + nlevel + 1);
        aind(bl, k + 2, 0) = aind((bl + 1) % 5, 1, k + 1);
        aind(bl, nlevel + 1, k + 1) = aind((bl + 1) % 5, 1, k + nlevel + 1);
        aind(bl, nlevel + 1, k + nlevel + 1) = aind((bl + 1) % 5, k + 2, 2 * nlevel);
      }
      aind(bl, 1, 0) = apind(0);
      aind(bl, nlevel + 1, 2 * nlevel) = apind(1);
      aind(bl, 0, 2 * nlevel + 1) = aind(bl, 0, 2 * nlevel);
    }

    // set up arrays for neighbors/neighbor indexing, solid angles, and arc lengths
    auto &numn_h = num_neighbors_h;
    auto &indn_h = ind_neighbors_h;
    auto &arcl_h = arc_weights_h;
    auto &wght_h = weights_h;
    const Real ifour_pi = 1.0 / (4.0 * M_PI);
    for (int n = 0; n < nangles; ++n) {
      // find the number of neighbors and indices of neighbors
      int num_nghbr;
      int neighbors[6];
      Neighbors(anorm, apnorm, aind, n, nlevel, num_nghbr, neighbors);

      // find the solid angle and arc (edge) lengths
      Real omega;
      Real arcs[6];
      SolidAngleAndArcLengths(anorm, apnorm, aind, n, nlevel, omega, arcs);

      // store in corresponding arrays
      numn_h(n) = num_nghbr;
      wght_h(n) = omega * ifour_pi;
      for (int nb = 0; nb < 6; ++nb) {
        indn_h(n, nb) = neighbors[nb];
        arcl_h(n, nb) = arcs[nb] * ifour_pi;
      }
    }

    // set up arrays for neighbor edge indexing
    auto &indne_h = ind_neighbors_edges_h;
    for (int n = 0; n < nangles; ++n) {
      int nn = numn_h(n);
      for (int nb = 0; nb < nn; ++nb) {
        for (int nnb = 0; nnb < numn_h(indn_h(n, nb)); ++nnb) {
          if (n == indn_h(indn_h(n, nb), nnb)) {
            indne_h(n, nb) = nnb;
          }
        }
      }
      if (nn == 5) {
        indne_h(n, 5) = (INT_MAX);
      }
    }

    // correct for round-off error level diff in arc lengths among shared edges
    for (int n = 0; n < nangles; ++n) {
      for (int nb = 0; nb < numn_h(n); ++nb) {
        Real arc_avg = 0.5 * (arcl_h(n, nb) + arcl_h(indn_h(n, nb), indne_h(n, nb)));
        arcl_h(n, nb) = arc_avg;
        arcl_h(indn_h(n, nb), indne_h(n, nb)) = arc_avg;
      }
    }

    // rotate geodesic mesh
    if (rotate_geo) {
      Real rotangles[2];
      if (rotate_geo == 2) {
        rotangles[0] = zpole;
        rotangles[1] = ppole;
      } else {
        OptimalAngles(anorm, apnorm, nlevel, rotangles);
      }
      RotateGrid(anorm, apnorm, nlevel, rotangles[0], rotangles[1]);
    }

    // set grid positions
    auto &cpos_h = cart_pos_h;
    auto &cposm_h = cart_pos_mid_h;
    for (int n = 0; n < nangles; ++n) {
      Real x, y, z;
      GridCartPosition(anorm, apnorm, n, nlevel, x, y, z);
      cpos_h(n, 0) = x;
      cpos_h(n, 1) = y;
      cpos_h(n, 2) = z;
      int nn = numn_h(n);
      for (int nb = 0; nb < nn; ++nb) {
        Real xm, ym, zm;
        GridCartPositionMid(anorm, apnorm, n, indn_h(n, nb), nlevel, xm, ym, zm);
        cposm_h(n, nb, 0) = xm;
        cposm_h(n, nb, 1) = ym;
        cposm_h(n, nb, 2) = zm;
      }
      if (nn == 5) {
        cposm_h(n, 5, 0) = std::numeric_limits<Real>::quiet_NaN();
        cposm_h(n, 5, 1) = std::numeric_limits<Real>::quiet_NaN();
        cposm_h(n, 5, 2) = std::numeric_limits<Real>::quiet_NaN();
      }
    }

    // set angular unit vectors along edges of angle faces
    auto &gflx_h = gflux_h;
    for (int n = 0; n < nangles; ++n) {
      Real x, y, z;
      GridCartPosition(anorm, apnorm, n, nlevel, x, y, z);
      Real zetav = std::acos(z);
      Real psiv = std::atan2(y, x);
      for (int nb = 0; nb < numn_h(n); ++nb) {
        Real xm, ym, zm;
        GridCartPositionMid(anorm, apnorm, n, indn_h(n, nb), nlevel, xm, ym, zm);
        Real zetaf = std::acos(zm);
        Real psif = std::atan2(ym, xm);
        Real unit_zeta, unit_psi;
        UnitFluxDir(zetav, psiv, zetaf, psif, unit_zeta, unit_psi);
        if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) { // (2D RZ)
          gflx_h(n, nb) = ym * (SQR(xm) + SQR(ym)) * unit_psi;
        } else { // UniformSpherical (1D)
          const Real isz = 1.0 / std::sqrt(1.0 - SQR(zm));
          gflx_h(n, nb) = isz * (SQR(xm) + SQR(ym)) * unit_zeta;
        }
      }
    }

    // correct for round-off error level diff in unit vectors among shared edges
    for (int n = 0; n < nangles; ++n) {
      for (int nb = 0; nb < numn_h(n); ++nb) {
        Real tgflx = gflx_h(n, nb);
        Real ngflx = gflx_h(indn_h(n, nb), indne_h(n, nb));
        Real gflx_avg = 0.5 * (std::abs(tgflx) + std::abs(ngflx));
        gflx_h(n, nb) = std::copysign(gflx_avg, tgflx);
        gflx_h(indn_h(n, nb), indne_h(n, nb)) = std::copysign(gflx_avg, ngflx);
      }
    }

    // deep copy
    Kokkos::deep_copy(num_neighbors, num_neighbors_h);
    Kokkos::deep_copy(ind_neighbors, ind_neighbors_h);
    Kokkos::deep_copy(weights, weights_h);
    Kokkos::deep_copy(arc_weights, arc_weights_h);
    Kokkos::deep_copy(cart_pos, cart_pos_h);
    Kokkos::deep_copy(cart_pos_mid, cart_pos_mid_h);
    Kokkos::deep_copy(gflux, gflux_h);

  } else if (nlevel == 0) { // one angle per octant
    // number of angles
    nangles = 8;

    // reallocate geodesic mesh arrays
    weights.Resize(nangles);
    cart_pos.Resize(nangles, 3);

    // create mirror views
    auto weights_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);
    auto cart_pos_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), cart_pos);

    // set solid angles and cartesian positions
    Real zetav[2] = {M_PI / 4.0, 3.0 * M_PI / 4.0};
    Real psiv[4] = {M_PI / 4.0, 3.0 * M_PI / 4.0, 5.0 * M_PI / 4.0, 7.0 * M_PI / 4.0};
    for (int z = 0, n = 0; z < 2; ++z) {
      for (int p = 0; p < 4; ++p, ++n) {
        weights_h(n) = 1.0 / nangles;
        cart_pos_h(n, 0) = std::sin(zetav[z]) * std::cos(psiv[p]) * std::sqrt(4.0 / 3.0);
        cart_pos_h(n, 1) = std::sin(zetav[z]) * std::sin(psiv[p]) * std::sqrt(4.0 / 3.0);
        cart_pos_h(n, 2) = std::cos(zetav[z]) * std::sqrt(2.0 / 3.0);
      }
    }

    // deep copy
    Kokkos::deep_copy(weights, weights_h);
    Kokkos::deep_copy(cart_pos, cart_pos_h);

  } else { // invalid nlevel
    PARTHENON_THROW("nlevel must be >= 0");
  }
}

//----------------------------------------------------------------------------------------
//! \brief GeodesicGrid destructor

GeodesicGrid::~GeodesicGrid() {}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GridCartPosition
//! \brief find position at face center

void GeodesicGrid::GridCartPosition(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                                    int n, int nlev, Real &x, Real &y, Real &z) {
  int ibl0 = (n / (2 * nlev * nlev));
  int ibl1 = (n % (2 * nlev * nlev)) / (2 * nlev);
  int ibl2 = (n % (2 * nlev * nlev)) % (2 * nlev);
  if (ibl0 == 5) {
    x = apnorm(ibl2, 0);
    y = apnorm(ibl2, 1);
    z = apnorm(ibl2, 2);
  } else {
    x = anorm(ibl0, ibl1 + 1, ibl2 + 1, 0);
    y = anorm(ibl0, ibl1 + 1, ibl2 + 1, 1);
    z = anorm(ibl0, ibl1 + 1, ibl2 + 1, 2);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GridCartPositionMid
//! \brief find mid position between two face centers

void GeodesicGrid::GridCartPositionMid(ParArrayHost<Real> anorm,
                                       ParArrayHost<Real> apnorm, int n, int nb, int nlev,
                                       Real &x, Real &y, Real &z) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(anorm, apnorm, n, nlev, x1, y1, z1);
  GridCartPosition(anorm, apnorm, nb, nlev, x2, y2, z2);
  Real xm = 0.5 * (x1 + x2);
  Real ym = 0.5 * (y1 + y2);
  Real zm = 0.5 * (z1 + z2);
  Real norm = std::sqrt(SQR(xm) + SQR(ym) + SQR(zm));
  x = xm / norm;
  y = ym / norm;
  z = zm / norm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::Neighbors
//! \brief retrieve number of neighbors and indexing of neighbors

void GeodesicGrid::Neighbors(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                             ParArrayHost<int> aind, int n, int nlev, int &num_nghbr,
                             int neighbors[6]) {
  if (n == 5 * 2 * nlev * nlev) { // handle north pole
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = aind(bl, 1, 1);
    }
    neighbors[5] = std::numeric_limits<int>::max();
    num_nghbr = 5;
  } else if (n == 5 * 2 * nlev * nlev + 1) { // handle south pole
    for (int bl = 0; bl < 5; ++bl) {
      neighbors[bl] = aind(bl, nlev, 2 * nlev);
    }
    neighbors[5] = std::numeric_limits<int>::max();
    num_nghbr = 5;
  } else {
    int ibl0 = (n / (2 * nlev * nlev));
    int ibl1 = (n % (2 * nlev * nlev)) / (2 * nlev);
    int ibl2 = (n % (2 * nlev * nlev)) % (2 * nlev);
    neighbors[0] = aind(ibl0, ibl1 + 1, ibl2 + 2);
    neighbors[1] = aind(ibl0, ibl1 + 2, ibl2 + 1);
    neighbors[2] = aind(ibl0, ibl1 + 2, ibl2);
    neighbors[3] = aind(ibl0, ibl1 + 1, ibl2);
    neighbors[4] = aind(ibl0, ibl1, ibl2 + 1);

    if (n % (2 * nlev * nlev) == nlev - 1 || n % (2 * nlev * nlev) == 2 * nlev - 1) {
      neighbors[5] = std::numeric_limits<int>::max();
      num_nghbr = 5;
    } else {
      neighbors[5] = aind(ibl0, ibl1, ibl2 + 2);
      num_nghbr = 6;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::CircumcenterNormalized
//! \brief find circumcenter of face

void GeodesicGrid::CircumcenterNormalized(Real x1, Real x2, Real x3, Real y1, Real y2,
                                          Real y3, Real z1, Real z2, Real z3, Real &x,
                                          Real &y, Real &z) {
  Real a = std::sqrt(SQR(x3 - x2) + SQR(y3 - y2) + SQR(z3 - z2));
  Real b = std::sqrt(SQR(x1 - x3) + SQR(y1 - y3) + SQR(z1 - z3));
  Real c = std::sqrt(SQR(x2 - x1) + SQR(y2 - y1) + SQR(z2 - z1));
  Real denom = 1.0 / ((a + c + b) * (a + c - b) * (a + b - c) * (b + c - a));
  Real x_c = (x1 * (SQR(a) * (SQR(b) + SQR(c) - SQR(a))) +
              x2 * (SQR(b) * (SQR(c) + SQR(a) - SQR(b))) +
              x3 * (SQR(c) * (SQR(a) + SQR(b) - SQR(c)))) *
             denom;
  Real y_c = (y1 * (SQR(a) * (SQR(b) + SQR(c) - SQR(a))) +
              y2 * (SQR(b) * (SQR(c) + SQR(a) - SQR(b))) +
              y3 * (SQR(c) * (SQR(a) + SQR(b) - SQR(c)))) *
             denom;
  Real z_c = (z1 * (SQR(a) * (SQR(b) + SQR(c) - SQR(a))) +
              z2 * (SQR(b) * (SQR(c) + SQR(a) - SQR(b))) +
              z3 * (SQR(c) * (SQR(a) + SQR(b) - SQR(c)))) *
             denom;
  Real norm_c = std::sqrt(SQR(x_c) + SQR(y_c) + SQR(z_c));
  x = x_c / norm_c;
  y = y_c / norm_c;
  z = z_c / norm_c;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::SolidAngleAndArcLengths
//! \brief retrieve solid angles and arc lengths

void GeodesicGrid::SolidAngleAndArcLengths(ParArrayHost<Real> anorm,
                                           ParArrayHost<Real> apnorm,
                                           ParArrayHost<int> aind, int n, int nlev,
                                           Real &weight, Real length[6]) {
  int nnum;
  int nvec[6];
  Neighbors(anorm, apnorm, aind, n, nlev, nnum, nvec);
  Real x0, y0, z0;
  GridCartPosition(anorm, apnorm, n, nlev, x0, y0, z0);
  weight = 0.0;
  for (int nb = 0; nb < nnum; ++nb) {
    Real xn1, yn1, zn1;
    Real xn2, yn2, zn2;
    Real xn3, yn3, zn3;
    GridCartPosition(anorm, apnorm, nvec[(nb + nnum - 1) % nnum], nlev, xn1, yn1, zn1);
    GridCartPosition(anorm, apnorm, nvec[nb], nlev, xn2, yn2, zn2);
    GridCartPosition(anorm, apnorm, nvec[(nb + 1) % nnum], nlev, xn3, yn3, zn3);
    Real xc1, yc1, zc1;
    Real xc2, yc2, zc2;
    CircumcenterNormalized(x0, xn1, xn2, y0, yn1, yn2, z0, zn1, zn2, xc1, yc1, zc1);
    CircumcenterNormalized(x0, xn2, xn3, y0, yn2, yn3, z0, zn2, zn3, xc2, yc2, zc2);
    Real scalprod_c1 = x0 * xc1 + y0 * yc1 + z0 * zc1;
    Real scalprod_c2 = x0 * xc2 + y0 * yc2 + z0 * zc2;
    Real scalprod_12 = xc1 * xc2 + yc1 * yc2 + zc1 * zc2;
    Real numerator =
        std::abs(x0 * (yc1 * zc2 - zc1 * yc2) + y0 * (zc1 * xc2 - xc1 * zc2) +
                 z0 * (xc1 * yc2 - yc1 * xc2));
    Real denominator = 1.0 + scalprod_c1 + scalprod_c2 + scalprod_12;
    weight += 2.0 * std::atan(numerator / denominator);
    length[nb] = std::acos(scalprod_12);
  }
  if (nnum == 5) {
    length[5] = std::numeric_limits<Real>::quiet_NaN();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::ArcLength
//! \brief find arc length between two face centers

Real GeodesicGrid::ArcLength(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm, int n1,
                             int n2, int nlev) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(anorm, apnorm, n1, nlev, x1, y1, z1);
  GridCartPosition(anorm, apnorm, n2, nlev, x2, y2, z2);
  return std::acos(x1 * x2 + y1 * y2 + z1 * z2);
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::OptimalAngles
//! \brief find anorm optimal angle by which to rotate the geodesic mesh

void GeodesicGrid::OptimalAngles(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                                 int nlev, Real ang[2]) {
  int nzeta = 200;
  int npsi = 200;
  Real maxangle = ArcLength(anorm, apnorm, 0, 1, nlev);
  Real deltazeta = maxangle / nzeta;
  Real deltapsi = M_PI / npsi;
  Real vmax = 0.0;
  for (int l = 0; l < nzeta; ++l) {
    Real zeta = (l + 1) * deltazeta;
    for (int k = 0; k < npsi; ++k) {
      Real psi = (k + 1) * deltapsi;
      Real kx = -std::sin(psi);
      Real ky = std::cos(psi);
      Real vmin_curr = 1.0;
      for (int n = 0; n < nangles; ++n) {
        Real vx, vy, vz;
        GridCartPosition(anorm, apnorm, n, nlev, vx, vy, vz);
        Real vrx = vx * std::cos(zeta) + ky * vz * std::sin(zeta) +
                   kx * (kx * vx + ky * vy) * (1.0 - std::cos(zeta));
        Real vry = vy * std::cos(zeta) - kx * vz * std::sin(zeta) +
                   ky * (kx * vx + ky * vy) * (1.0 - std::cos(zeta));
        Real vrz = vz * std::cos(zeta) + (kx * vy - ky * vx) * std::sin(zeta);
        if (std::abs(vrx) < vmin_curr) {
          vmin_curr = std::abs(vrx);
        }
        if (std::abs(vry) < vmin_curr) {
          vmin_curr = std::abs(vry);
        }
        if (std::abs(vrz) < vmin_curr) {
          vmin_curr = std::abs(vrz);
        }
      }
      if (vmin_curr > vmax) {
        vmax = vmin_curr;
        ang[0] = zeta;
        ang[1] = psi;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::RotateGrid
//! \brief rotate the geodesic grid such that the north pole angle center is assigned
//  to angular coordinate znew, pnew

void GeodesicGrid::RotateGrid(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                              int nlev, Real znew, Real pnew) {
  Real kx = -std::sin(pnew);
  Real ky = std::cos(pnew);
  for (int bl = 0; bl < 5; ++bl) {
    for (int l = 0; l < nlev; ++l) {
      for (int m = 0; m < 2 * nlev; ++m) {
        Real vx = anorm(bl, l + 1, m + 1, 0);
        Real vy = anorm(bl, l + 1, m + 1, 1);
        Real vz = anorm(bl, l + 1, m + 1, 2);
        Real vrx = vx * std::cos(znew) + ky * vz * std::sin(znew) +
                   kx * (kx * vx + ky * vy) * (1.0 - std::cos(znew));
        Real vry = vy * std::cos(znew) - kx * vz * std::sin(znew) +
                   ky * (kx * vx + ky * vy) * (1.0 - std::cos(znew));
        Real vrz = vz * std::cos(znew) + (kx * vy - ky * vx) * std::sin(znew);
        anorm(bl, l + 1, m + 1, 0) = vrx;
        anorm(bl, l + 1, m + 1, 1) = vry;
        anorm(bl, l + 1, m + 1, 2) = vrz;
      }
    }
  }
  for (int pl = 0; pl < 2; ++pl) {
    Real vx = apnorm(pl, 0);
    Real vy = apnorm(pl, 1);
    Real vz = apnorm(pl, 2);
    Real vrx = vx * std::cos(znew) + ky * vz * std::sin(znew) +
               kx * (kx * vx + ky * vy) * (1.0 - std::cos(znew));
    Real vry = vy * std::cos(znew) - kx * vz * std::sin(znew) +
               ky * (kx * vx + ky * vy) * (1.0 - std::cos(znew));
    Real vrz = vz * std::cos(znew) + (kx * vy - ky * vx) * std::sin(znew);
    apnorm(pl, 0) = vrx;
    apnorm(pl, 1) = vry;
    apnorm(pl, 2) = vrz;
  }
  for (int i = 0; i < 3; ++i) {
    for (int bl = 0; bl < 5; ++bl) {
      for (int k = 0; k < nlev; ++k) {
        anorm(bl, 0, k + 1, i) = anorm((bl + 4) % 5, k + 1, 1, i);
        anorm(bl, 0, k + nlev + 1, i) = anorm((bl + 4) % 5, nlev, k + 1, i);
        anorm(bl, k + 1, 2 * nlev + 1, i) = anorm((bl + 4) % 5, nlev, k + nlev + 1, i);
        anorm(bl, k + 2, 0, i) = anorm((bl + 1) % 5, 1, k + 1, i);
        anorm(bl, nlev + 1, k + 1, i) = anorm((bl + 1) % 5, 1, k + nlev + 1, i);
        anorm(bl, nlev + 1, k + nlev + 1, i) = anorm((bl + 1) % 5, k + 2, 2 * nlev, i);
      }
      anorm(bl, 1, 0, i) = apnorm(0, i);
      anorm(bl, nlev + 1, 2 * nlev, i) = apnorm(1, i);
      anorm(bl, 0, 2 * nlev + 1, i) = anorm(bl, 0, 2 * nlev, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::RotateGrid
//! \brief find components of unit vectors along edges

void GeodesicGrid::UnitFluxDir(Real zetav, Real psiv, Real zetaf, Real psif, Real &dzeta,
                               Real &dpsi) {
  if (std::abs(psif - psiv) < 1.0e-10 ||
      std::abs(std::abs(std::cos(zetaf)) - 1.0) < 1.0e-10 ||
      std::abs(std::abs(std::cos(zetav)) - 1.0) < 1.0e-10) {
    dzeta = std::copysign(1.0, zetaf - zetav);
    dpsi = 0.0;
  } else {
    Real a_par, p_par;
    GreatCircleParam(zetav, zetaf, psiv, psif, a_par, p_par);
    Real zeta_deriv =
        (a_par * std::sin(psif - p_par) /
         (1.0 + SQR(a_par) * std::cos(psif - p_par) * std::cos(psif - p_par)));
    Real denom = 1.0 / sqrt(SQR(zeta_deriv) + SQR(std::sin(zetaf)));
    Real signfactor = std::copysign(1.0, psif - psiv) *
                      std::copysign(1.0, M_PI - std::abs(psif - psiv));
    dzeta = signfactor * zeta_deriv * denom;
    dpsi = signfactor * denom;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GreatCircleParam
//! \brief find parameters describing the great circle connecting two angular coordinates

void GeodesicGrid::GreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                                    Real &apar, Real &psi0) {
  Real atilde = (std::sin(psi2) / std::tan(zeta1) - std::sin(psi1) / std::tan(zeta2)) /
                std::sin(psi2 - psi1);
  Real btilde =
      (cos(psi2) / std::tan(zeta1) - cos(psi1) / std::tan(zeta2)) / std::sin(psi1 - psi2);
  psi0 = std::atan2(btilde, atilde);
  apar = std::sqrt(SQR(atilde) + SQR(btilde));
  return;
}