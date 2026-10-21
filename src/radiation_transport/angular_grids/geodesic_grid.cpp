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
GeodesicGrid::GeodesicGrid(int nlev, int rotate, Real zpole, Real ppole, bool fv_fix)
    : nlevel(nlev), rotate_geo(rotate), num_neighbors("num_neighbors", 1),
      ind_neighbors("ind_neighbors", 1, 1),
      ind_neighbors_edges("ind_neighbors_edges", 1, 1), weights("weights", 1),
      arc_weights("arc_weights", 1, 1), cart_pos("cart_pos", 1, 1),
      cart_pos_unit("cart_pos_unit", 1, 1), gflux("gflux", 1, 1),
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
    cart_pos_unit.Resize(nangles, 3);
    gflux.Resize(nangles, 6);

    // Host Arrays
    ParArrayHost<int> num_neighbors_h("num_neighbors_h", nangles);
    ParArrayHost<int> ind_neighbors_h("ind_neighbors_h", nangles, 6);
    ParArrayHost<int> ind_neighbors_edges_h("ind_neighbors_edges_h", nangles, 6);
    ParArrayHost<Real> weights_h("weights_h", nangles);
    ParArrayHost<Real> arc_weights_h("arc_weights_h", nangles, 6);
    ParArrayHost<Real> cart_pos_h("cart_pos_h", nangles, 3);
    ParArrayHost<Real> cart_pos_unit_h("cart_pos_unit_h", nangles, 3);
    ParArrayHost<Real> gflux_h("gflux_h", nangles, 6);

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
    for (int n = 0; n < nangles; ++n) {
      Real x, y, z;
      GridCartPosition(anorm, apnorm, n, nlevel, x, y, z);
      cpos_h(n, 0) = x;
      cpos_h(n, 1) = y;
      cpos_h(n, 2) = z;
      cart_pos_unit_h(n, 0) = x;
      cart_pos_unit_h(n, 1) = y;
      cart_pos_unit_h(n, 2) = z;
    }

    // set angular geometric flux coefficients along edges of angle faces
    constexpr int curv_comp = parthenon::IsCoord<parthenon::UniformSpherical>() ? 2 : 0;
    auto &gflx_h = gflux_h;
    for (int n = 0; n < nangles; ++n) {
      Real center[3];
      GridCartPosition(anorm, apnorm, n, nlevel, center[0], center[1], center[2]);
      const int nn = numn_h(n);
      for (int nb = 0; nb < nn; ++nb) {
        // Circumcenters bounding edge nb: shared with neighbors (nb-1) and (nb+1).
        Real pm[3], pl[3], pp[3];
        GridCartPosition(anorm, apnorm, indn_h(n, nb), nlevel, pm[0], pm[1], pm[2]);
        GridCartPosition(anorm, apnorm, indn_h(n, (nb + nn - 1) % nn), nlevel, pl[0],
                         pl[1], pl[2]);
        GridCartPosition(anorm, apnorm, indn_h(n, (nb + 1) % nn), nlevel, pp[0], pp[1],
                         pp[2]);
        Real v1[3], v2[3];
        CircumcenterNormalized(center[0], pl[0], pm[0], center[1], pl[1], pm[1],
                               center[2], pl[2], pm[2], v1[0], v1[1], v1[2]);
        CircumcenterNormalized(center[0], pm[0], pp[0], center[1], pm[1], pp[1],
                               center[2], pm[2], pp[2], v2[0], v2[1], v2[2]);
        // gflux = (edge line integral of G_curv) / arclength, so that gflux * arc_weight
        // (arc_weight = arclength/4pi) reproduces the edge integral / 4pi.
        const Real oint = GfluxEdgeIntegral(v1, v2, center, curv_comp);
        const Real arclen = std::acos(
            std::max(-1.0, std::min(1.0, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])));
        gflx_h(n, nb) = (arclen > 1.0e-14) ? oint / arclen : 0.0;
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

    if (fv_fix) {
      ApplyFiniteVolumeCorrections(cart_pos_h, weights_h, arc_weights_h, gflux_h,
                                   num_neighbors_h, ind_neighbors_h);
    }

    // deep copy
    Kokkos::deep_copy(num_neighbors, num_neighbors_h);
    Kokkos::deep_copy(ind_neighbors, ind_neighbors_h);
    Kokkos::deep_copy(weights, weights_h);
    Kokkos::deep_copy(arc_weights, arc_weights_h);
    Kokkos::deep_copy(cart_pos, cart_pos_h);
    Kokkos::deep_copy(cart_pos_unit, cart_pos_unit_h);
    Kokkos::deep_copy(gflux, gflux_h);

  } else if (nlevel == 0) { // one angle per octant
    // number of angles
    nangles = 8;

    // reallocate geodesic mesh arrays
    weights.Resize(nangles);
    cart_pos.Resize(nangles, 3);
    cart_pos_unit.Resize(nangles, 3);

    // host working arrays (host-first; single deep-copy batch at the end)
    ParArrayHost<Real> weights_h("weights_h", nangles);
    ParArrayHost<Real> cart_pos_h("cart_pos_h", nangles, 3);
    ParArrayHost<Real> cart_pos_unit_h("cart_pos_unit_h", nangles, 3);

    // set solid angles and cartesian positions
    Real zetav[2] = {M_PI / 4.0, 3.0 * M_PI / 4.0};
    Real psiv[4] = {M_PI / 4.0, 3.0 * M_PI / 4.0, 5.0 * M_PI / 4.0, 7.0 * M_PI / 4.0};
    for (int z = 0, n = 0; z < 2; ++z) {
      for (int p = 0; p < 4; ++p, ++n) {
        weights_h(n) = 1.0 / nangles;
        cart_pos_h(n, 0) = std::sin(zetav[z]) * std::cos(psiv[p]) * std::sqrt(4.0 / 3.0);
        cart_pos_h(n, 1) = std::sin(zetav[z]) * std::sin(psiv[p]) * std::sqrt(4.0 / 3.0);
        cart_pos_h(n, 2) = std::cos(zetav[z]) * std::sqrt(2.0 / 3.0);
        cart_pos_unit_h(n, 0) = cart_pos_h(n, 0);
        cart_pos_unit_h(n, 1) = cart_pos_h(n, 1);
        cart_pos_unit_h(n, 2) = cart_pos_h(n, 2);
      }
    }

    // deep copy (single batch to device)
    Kokkos::deep_copy(weights, weights_h);
    Kokkos::deep_copy(cart_pos, cart_pos_h);
    Kokkos::deep_copy(cart_pos_unit, cart_pos_unit_h);

  } else { // invalid nlevel
    PARTHENON_THROW("nlevel must be >= 0");
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::ApplyFiniteVolumeCorrections
//! \brief Replace the cell-centered normal directions with their exact solid-angle
//! averages <n_i> = (1/dOmega) int n_i dOmega for the finite-volume transport speeds;
//! cart_pos_unit retains the true centroid unit vectors (if needed).
//!
//! Each <n_i> is obtained (divergence theorem) as (1/w_a) sum_b [edge integral of the
//! flux field G_i with div_Omega G_i = n_i], summed over the cell's great-circle edges
//! between circumcenters.
void GeodesicGrid::ApplyFiniteVolumeCorrections(ParArrayHost<Real> &cart_pos_h,
                                                ParArrayHost<Real> &weights_h,
                                                ParArrayHost<Real> &arc_weights_h,
                                                ParArrayHost<Real> &gflux_h,
                                                ParArrayHost<int> &num_neighbors_h,
                                                ParArrayHost<int> &ind_neighbors_h) {
  // Solid-angle-averaged cosine for every transported component, via exact edge
  // integrals of the flux fields G_i (div_Omega G_i = n_i).
  auto &anorm = amesh_normals;
  auto &apnorm = ameshp_normals;
  for (int n = 0; n < nangles; ++n) {
    Real center[3];
    GridCartPosition(anorm, apnorm, n, nlevel, center[0], center[1], center[2]);
    const int nn = num_neighbors_h(n);
    Real navg[3] = {0.0, 0.0, 0.0};
    for (int nb = 0; nb < nn; ++nb) {
      Real pm[3], pl[3], pp[3];
      GridCartPosition(anorm, apnorm, ind_neighbors_h(n, nb), nlevel, pm[0], pm[1],
                       pm[2]);
      GridCartPosition(anorm, apnorm, ind_neighbors_h(n, (nb + nn - 1) % nn), nlevel,
                       pl[0], pl[1], pl[2]);
      GridCartPosition(anorm, apnorm, ind_neighbors_h(n, (nb + 1) % nn), nlevel, pp[0],
                       pp[1], pp[2]);
      Real v1[3], v2[3];
      CircumcenterNormalized(center[0], pl[0], pm[0], center[1], pl[1], pm[1], center[2],
                             pl[2], pm[2], v1[0], v1[1], v1[2]);
      CircumcenterNormalized(center[0], pm[0], pp[0], center[1], pm[1], pp[1], center[2],
                             pm[2], pp[2], v2[0], v2[1], v2[2]);
      for (int d = 0; d < 3; ++d) {
        navg[d] += GfluxEdgeIntegral(v1, v2, center, d);
      }
    }
    for (int d = 0; d < 3; ++d) {
      cart_pos_h(n, d) = navg[d] / (4.0 * M_PI * weights_h(n));
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
                        "Geodesic g_a disagrees with solid-angle average <n_curv>.");
    }
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
//! \fn Real GeodesicGrid::GfluxEdgeIntegral
//! \brief Line integral of a closed-form flux field G_comp along the great-circle edge
//! from circumcenter v1 to v2, with the outward (in-surface) normal relative to the cell
//! center.
Real GeodesicGrid::GfluxEdgeIntegral(const Real v1[3], const Real v2[3],
                                     const Real center[3], int comp) {
  // 16-point Gauss-Legendre nodes/weights on [-1, 1]
  constexpr int NQ = 16;
  constexpr Real gx[NQ] = {
      -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
      -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
      0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
      0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499};
  constexpr Real gw[NQ] = {
      0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
      0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
      0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
      0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541};

  const Real dot =
      std::max(-1.0, std::min(1.0, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]));
  const Real ang = std::acos(dot);
  if (ang < 1.0e-14) return 0.0;
  const Real sin_ang = std::sin(ang);

  Real total = 0.0;
  for (int q = 0; q < NQ; ++q) {
    const Real t = 0.5 * (gx[q] + 1.0); // map to [0, 1]
    const Real w = 0.5 * gw[q];
    const Real s1 = std::sin((1.0 - t) * ang) / sin_ang;
    const Real s2 = std::sin(t * ang) / sin_ang;
    Real p[3] = {s1 * v1[0] + s2 * v2[0], s1 * v1[1] + s2 * v2[1],
                 s1 * v1[2] + s2 * v2[2]};
    const Real pn = std::sqrt(SQR(p[0]) + SQR(p[1]) + SQR(p[2]));
    p[0] /= pn;
    p[1] /= pn;
    p[2] /= pn;
    // Arc tangent dP/dt (before projection), then project to sphere tangent plane.
    const Real d1 = -std::cos((1.0 - t) * ang) * ang / sin_ang;
    const Real d2 = std::cos(t * ang) * ang / sin_ang;
    Real dp[3] = {d1 * v1[0] + d2 * v2[0], d1 * v1[1] + d2 * v2[1],
                  d1 * v1[2] + d2 * v2[2]};
    const Real dpp = dp[0] * p[0] + dp[1] * p[1] + dp[2] * p[2];
    Real tang[3] = {dp[0] - dpp * p[0], dp[1] - dpp * p[1], dp[2] - dpp * p[2]};
    const Real tn = std::sqrt(SQR(tang[0]) + SQR(tang[1]) + SQR(tang[2]));
    if (tn < 1.0e-14) continue;
    tang[0] /= tn;
    tang[1] /= tn;
    tang[2] /= tn;
    // In-surface edge normal m = tang x p, oriented outward (away from center).
    Real m[3] = {tang[1] * p[2] - tang[2] * p[1], tang[2] * p[0] - tang[0] * p[2],
                 tang[0] * p[1] - tang[1] * p[0]};
    // Outward: opposite the in-plane projection of center at p.
    const Real cp = center[0] * p[0] + center[1] * p[1] + center[2] * p[2];
    const Real inw[3] = {center[0] - cp * p[0], center[1] - cp * p[1],
                         center[2] - cp * p[2]};
    if (m[0] * inw[0] + m[1] * inw[1] + m[2] * inw[2] > 0.0) {
      m[0] = -m[0];
      m[1] = -m[1];
      m[2] = -m[2];
    }
    // Flux field G_comp at p, with div_Omega G_comp = n_comp.
    Real G[3];
    if (comp == 0) { // <n_x>
      G[0] = -SQR(p[1]);
      G[1] = p[0] * p[1];
      G[2] = 0.0;
    } else if (comp == 1) { // <n_y>
      G[0] = p[0] * p[1];
      G[1] = -SQR(p[0]);
      G[2] = 0.0;
    } else { // <n_z>
      G[0] = 0.5 * p[0] * p[2];
      G[1] = 0.5 * p[1] * p[2];
      G[2] = -0.5 * (SQR(p[0]) + SQR(p[1]));
    }
    total += (G[0] * m[0] + G[1] * m[1] + G[2] * m[2]) * tn * w;
  }

  return total;
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
