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
#ifndef RADIATION_TRANSPORT_ANGULAR_GRIDS_LATLON_GRID_HPP_
#define RADIATION_TRANSPORT_ANGULAR_GRIDS_LATLON_GRID_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>

using namespace parthenon;

//----------------------------------------------------------------------------------------
// NOTE(@pdmullen): The latlon grid infrastructure closesly resembes the AthenaK
// geodesic grid implementation  (see copyrights info above)

class LatLonGrid {
 public:
  struct Resolution {
    int ntheta;
    int nphi;
  };

  static Resolution ResolutionForLevel(int nlevel) {
    PARTHENON_REQUIRE(nlevel >= 1, "Lat-lon nlevel must be at least 1.");
    const int ntheta = 2 * nlevel;
    const int nphi =
        parthenon::IsCoord<parthenon::UniformSpherical>()
            ? 1
            : (parthenon::IsCoord<parthenon::UniformCylindrical>() ? 2 * nlevel
                                                                   : 4 * nlevel);
    return {ntheta, nphi};
  }

  LatLonGrid(int nlevel, bool fv_fix)
      : LatLonGrid(ResolutionForLevel(nlevel).ntheta, ResolutionForLevel(nlevel).nphi,
                   fv_fix) {}
  LatLonGrid(int ntheta, int nphi, bool fv_fix);
  ~LatLonGrid();

  int nangles;
  int ntheta_, nphi_;
  ParArrayND<int> num_neighbors;
  ParArrayND<int> ind_neighbors;
  ParArrayND<int> ind_neighbors_edges;
  ParArrayND<Real> weights;
  ParArrayND<Real> arc_weights;
  ParArrayND<Real> cart_pos;
  ParArrayND<Real> cart_pos_unit;
  ParArrayND<Real> cart_pos_mid;
  ParArrayND<Real> gflux;

  void ApplyFiniteVolumeCorrections(
      ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
      ParArrayHost<Real> &cart_pos_h, ParArrayHost<Real> &weights_h,
      ParArrayHost<Real> &arc_weights_h, ParArrayHost<Real> &gflux_h,
      ParArrayHost<int> &num_neighbors_h, ParArrayHost<int> &ind_neighbors_h);

 private:
  void ComputeThetaLevels(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &theta_f,
                          ParArrayHost<Real> &costheta_v, ParArrayHost<Real> &costheta_f);
  void ComputePhiAngles(ParArrayHost<Real> &phi_v, ParArrayHost<Real> &phi_f);
  void ComputeCartesianDirections(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                                  ParArrayHost<Real> &cart_pos_h,
                                  ParArrayHost<Real> &cart_pos_unit_h);
  void ComputeWeights(ParArrayHost<Real> &costheta_f, ParArrayHost<Real> &phi_f,
                      ParArrayHost<Real> &weights_h);
  void ComputeNeighborConnectivity(ParArrayHost<int> &num_neighbors_h,
                                   ParArrayHost<int> &ind_neighbors_h,
                                   ParArrayHost<int> &ind_neighbors_edges_h);
  void ComputeArcLengths(ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                         ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                         ParArrayHost<Real> &arc_weights_h,
                         ParArrayHost<Real> &cart_pos_mid_h,
                         ParArrayHost<int> &ind_neighbors_h,
                         ParArrayHost<int> &ind_neighbors_edges_h,
                         ParArrayHost<int> &num_neighbors_h);
  void ComputeUnitFluxes(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                         ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                         ParArrayHost<Real> &gflux_h, ParArrayHost<Real> &cart_pos_mid_h,
                         ParArrayHost<int> &ind_neighbors_h,
                         ParArrayHost<int> &ind_neighbors_edges_h,
                         ParArrayHost<int> &num_neighbors_h);
};

#endif // RADIATION_TRANSPORT_ANGULAR_GRIDS_LATLON_GRID_HPP_
