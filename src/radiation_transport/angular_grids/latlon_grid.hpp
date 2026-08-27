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
  LatLonGrid(int ntheta, int nphi);
  ~LatLonGrid();

  int nangles;
  int ntheta_, nphi_;
  ParArrayND<int> num_neighbors;
  ParArrayND<int> ind_neighbors;
  ParArrayND<int> ind_neighbors_edges;
  ParArrayND<Real> weights;
  ParArrayND<Real> arc_weights;
  ParArrayND<Real> cart_pos;
  ParArrayND<Real> cart_pos_mid;
  ParArrayND<Real> gflux;

 private:
  void ComputeThetaLevels(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &theta_f,
                          ParArrayHost<Real> &costheta_v, ParArrayHost<Real> &costheta_f);
  void ComputePhiAngles(ParArrayHost<Real> &phi_v, ParArrayHost<Real> &phi_f);
  void ComputeCartesianDirections(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v);
  void ComputeWeights(ParArrayHost<Real> &costheta_f, ParArrayHost<Real> &phi_f);
  void ComputeNeighborConnectivity();
  void ComputeArcLengths(ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f,
                         ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v);
  void ComputeUnitFluxes(ParArrayHost<Real> &theta_v, ParArrayHost<Real> &phi_v,
                         ParArrayHost<Real> &theta_f, ParArrayHost<Real> &phi_f);
};

#endif // RADIATION_TRANSPORT_ANGULAR_GRIDS_LATLON_GRID_HPP_
