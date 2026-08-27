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
#ifndef RADIATION_TRANSPORT_ANGULAR_GRIDS_ANGULAR_GRID_UTILS_HPP_
#define RADIATION_TRANSPORT_ANGULAR_GRIDS_ANGULAR_GRID_UTILS_HPP_
// This file was made in part with generative AI.

// C++ headers
#include <array>
#include <memory>
#include <string>

// Parthenon headers
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/angular_grids/geodesic_grid.hpp"
#include "radiation_transport/angular_grids/latlon_grid.hpp"

using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
//! \struct AngularGridArrays
//! \brief Bundle of the device arrays extracted from an angular grid that the radiation
//! solvers consume.  The two angular grid types (GeodesicGrid, LatLonGrid) are unrelated
//! classes but expose identically-named members; this bundle lets callers pull them in
//! one shot regardless of which grid is active.
struct AngularGridArrays {
  ParArrayND<Real> cart_pos;     // coord position (cartesian) at face center
  ParArrayND<Real> gflux;        // flux at face edges for 1D sph and 2D RZ
  ParArrayND<Real> arc_weights;  // arc lengths / 4pi
  ParArrayND<Real> weights;      // solid angles / 4pi
  ParArrayND<int> num_neighbors; // number of neighbors
  ParArrayND<int> ind_neighbors; // indices of neighbors
};

//----------------------------------------------------------------------------------------
//! \fn AngularGridArrays GetAngularGridArrays
//! \brief Extract the angular grid device arrays from a radiation package, selecting the
//! grid stored under the package's "angular_mesh" param.
inline AngularGridArrays
GetAngularGridArrays(const std::shared_ptr<StateDescriptor> &pkg) {
  auto bundle = [](const auto &agrid) {
    AngularGridArrays arrays;
    arrays.cart_pos = agrid->cart_pos;
    arrays.gflux = agrid->gflux;
    arrays.arc_weights = agrid->arc_weights;
    arrays.weights = agrid->weights;
    arrays.num_neighbors = agrid->num_neighbors;
    arrays.ind_neighbors = agrid->ind_neighbors;
    return arrays;
  };
  const auto &amesh = pkg->Param<std::string>("angular_mesh");
  if (amesh == "geodesic") {
    return bundle(pkg->Param<std::shared_ptr<GeodesicGrid>>("geodesic_grid"));
  } else if (amesh == "latlon") {
    return bundle(pkg->Param<std::shared_ptr<LatLonGrid>>("latlon_grid"));
  } else {
    PARTHENON_FAIL("Invalid angular mesh choice!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::array<int, 3> AngularFluxDirs
//! \brief Map the spatial coordinate axes (X1, X2, X3) to the angle-space direction
//! components used when computing angular fluxes.  The mapping depends on the compile-
//! time coordinate system so that the polar/azimuthal directions line up with the mesh
//! (2D RZ and 1D spherical reorder the components; Cartesian is the identity).  The
//! result is indexed by NDIR[DIR - 1].
KOKKOS_FORCEINLINE_FUNCTION std::array<int, 3> AngularFluxDirs() {
  if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) { // (2D RZ)
    return {0, 2, 1};
  } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) { // (1D sph)
    return {2, 0, 1};
  } else { // Cartesian
    return {0, 1, 2};
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real InverseRadiusForAngularFlux
//! \brief Face-averaged inverse radius used to build the angular-flux geometric
//! coefficient in curvilinear coordinates.  This is a purely spatial quantity, but it
//! only enters the angular-flux divergence term (it is 2/(r1+r2) in general, with a
//! spherical-shell correction under 1D spherical coordinates).
template <typename Coords>
KOKKOS_FORCEINLINE_FUNCTION Real InverseRadiusForAngularFlux(const Coords &coords,
                                                             const int i) {
  const Real r1 = coords.template Xf<X1DIR>(i);
  const Real r2 = coords.template Xf<X1DIR>(i + 1);
  if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
    return 1.5 * (r1 + r2) / (SQR(r1) + SQR(r2) + r1 * r2);
  }
  return 2.0 / (r1 + r2);
}

#endif // RADIATION_TRANSPORT_ANGULAR_GRIDS_ANGULAR_GRID_UTILS_HPP_
