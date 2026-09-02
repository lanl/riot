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
#ifndef RADIATION_TRANSPORT_ANGULAR_GRIDS_GEODESIC_GRID_HPP_
#define RADIATION_TRANSPORT_ANGULAR_GRIDS_GEODESIC_GRID_HPP_
// This file was made in part with generative AI.

// Parthenon headers
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
//! \class GeodesicGrid
// NOTE(@pdmullen): The geodesic grid infrastructure is taken from the AthenaK
// implementation and ported to the Parthenon framework by @pdmullen (see copyrights info
// above)

class GeodesicGrid {
 public:
  GeodesicGrid(int nlev, int rotate, Real zpole, Real ppole);
  ~GeodesicGrid();

  int nangles; // number of angles (derived from nlevel, 5*(2*nlevel^2) + 2)
  ParArrayND<int> num_neighbors;       // number of neighbors
  ParArrayND<int> ind_neighbors;       // indices of neighbors
  ParArrayND<int> ind_neighbors_edges; // indices of neighbor edges
  ParArrayND<Real> weights;            // solid angles / 4pi
  ParArrayND<Real> arc_weights;        // arc lengths / 4pi
  ParArrayND<Real> cart_pos;           // coord position (cartesian) at face center
  ParArrayND<Real> cart_pos_unit;      // true unit direction cosines at face center
  ParArrayND<Real> gflux;              // flux at face edges for 1D sph and 2D RZ

  // functions
  void GridCartPosition(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm, int n,
                        int nlev, Real &x, Real &y, Real &z);
  void Neighbors(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                 ParArrayHost<int> aind, int n, int nlev, int &num_nghbr,
                 int neighbors[6]);
  void CircumcenterNormalized(Real x1, Real x2, Real x3, Real y1, Real y2, Real y3,
                              Real z1, Real z2, Real z3, Real &x, Real &y, Real &z);
  void SolidAngleAndArcLengths(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm,
                               ParArrayHost<int> aind, int n, int nlev, Real &weight,
                               Real length[6]);
  Real ArcLength(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm, int n1, int n2,
                 int nlev);
  void OptimalAngles(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm, int nlev,
                     Real ang[2]);
  void RotateGrid(ParArrayHost<Real> anorm, ParArrayHost<Real> apnorm, int nlev,
                  Real znew, Real pnew);

  Real GfluxEdgeIntegral(const Real v1[3], const Real v2[3], const Real center[3],
                         int comp);

  void ApplyFiniteVolumeCorrections(ParArrayHost<Real> &cart_pos_h,
                                    ParArrayHost<Real> &cart_pos_unit_h,
                                    ParArrayHost<Real> &weights_h,
                                    ParArrayHost<Real> &arc_weights_h,
                                    ParArrayHost<Real> &gflux_h,
                                    ParArrayHost<int> &num_neighbors_h,
                                    ParArrayHost<int> &ind_neighbors_h);

 private:
  int nlevel;     // level of the geodesic mesh (==0 is 1 angle per octant for testing)
  int rotate_geo; // flag to enable the rotation of geodesic mesh
  ParArrayHost<Real> amesh_normals;
  ParArrayHost<Real> ameshp_normals;
  ParArrayHost<int> amesh_indices;
  ParArrayHost<int> ameshp_indices;
};

#endif // RADIATION_TRANSPORT_ANGULAR_GRIDS_GEODESIC_GRID_HPP_
