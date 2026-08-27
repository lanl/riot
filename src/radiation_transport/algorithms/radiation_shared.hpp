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
#ifndef RADIATION_ALGORITHMS_RADIATION_SHARED_HPP_
#define RADIATION_ALGORITHMS_RADIATION_SHARED_HPP_
// This file was made in part with generative AI.

// C++ headers
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include "radiation_transport/angular_grids/geodesic_grid.hpp"
#include "radiation_transport/angular_grids/latlon_grid.hpp"
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
// Shared parameters for the radiation transport packages (explicit, jacobi).
// AddSharedParams() populates the package Params with everything common to both solvers
// in one call, so that each solver's Initialize only needs to add its algorithm-specific
// params and fields.
//----------------------------------------------------------------------------------------
namespace RadiationShared {

//----------------------------------------------------------------------------------------
//! \struct SharedParams
//! \brief The subset of the shared parameters that a solver's Initialize needs directly
//! (the rest are consumed later via the package Params).
struct SharedParams {
  int nangles; // number of discrete angles in the angular mesh
  int ngroups; // number of frequency groups (owned by the materials package)
};

// The input block from which the shared radiation parameters are read.  Algorithm-
// specific parameters are read from nested blocks (e.g. "radiation_transport/explicit").
constexpr char radiation_block[] = "radiation_transport";

// Individual shared configuration options.  Each reads one group of related parameters
// from the <radiation_transport> input block and stores them on the package.
// AddSharedParams below composes them; callers use that aggregate rather than these
// directly.
namespace ConfigOption {

//----------------------------------------------------------------------------------------
//! \fn void AddCfl
//! \brief Store the CFL number for the radiation transport integration.
inline void AddCfl(ParameterInput *pin, Params &params) {
  params.Add("cfl",
             pin->GetOrAddReal(radiation_block, "cfl", 0.8,
                               "CFL number for thermal radiation transport integration"));
}

//----------------------------------------------------------------------------------------
//! \fn void AddUnitUtils
//! \brief Build the UnitUtils (CGS by default, or a custom units system for testing) and
//! store them under the "unit_utils" param.
inline void AddUnitUtils(ParameterInput *pin, Params &params) {
  UnitUtils uutils;
  using parthenon::constants::CGS;
  using parthenon::constants::PhysicalConstants;
  uutils.c = PhysicalConstants<CGS>::speed_of_light;
  uutils.arad = (4.0 / uutils.c) * PhysicalConstants<CGS>::stefan_boltzmann;
  uutils.boltzmann = PhysicalConstants<CGS>::boltzmann;
  uutils.planck = PhysicalConstants<CGS>::planck;
  const bool units_override = pin->GetOrAddBoolean(
      radiation_block, "units_override", false,
      "Override useage of CGS units and invoke a custom units system for testing");
  if (units_override) {
    uutils.c = pin->GetOrAddReal(radiation_block, "c", 1.0,
                                 "Sets speed of light when units_override==true");
    uutils.arad = pin->GetOrAddReal(radiation_block, "arad", 1.0,
                                    "Sets radiation constant when units_override==true");
    uutils.boltzmann = pin->GetOrAddReal(
        radiation_block, "kb", 1.0, "Sets boltzmann constant when units_override==true");
    uutils.planck = pin->GetOrAddReal(radiation_block, "h", 1.0,
                                      "Sets Planck constant when units_override==true");
  }
  params.Add("unit_utils", uutils);
}

//----------------------------------------------------------------------------------------
//! \fn bool AddCouplingParams
//! \brief Read the radiation source-term coupling flags, validate them against hydro, and
//! store them.  Returns whether coupling is enabled.
inline bool AddCouplingParams(ParameterInput *pin, Params &params) {
  const bool hydro_enabled = pin->GetOrAddBoolean("physics", "hydro", true);
  const bool coupling = pin->GetOrAddBoolean(
      radiation_block, "coupling", true,
      "Enables thermal radiation source term (emission, absorption, scattering)");
  const bool affect_fluid =
      pin->GetOrAddBoolean(radiation_block, "affect_fluid", coupling,
                           "Enables radiation source term feedback on the fluid");
  if (coupling) PARTHENON_REQUIRE(hydro_enabled, "Radiation coupling requires hydro.");
  if (affect_fluid) PARTHENON_REQUIRE(coupling, "Fluid feedback requires coupling.");
  params.Add("coupling", coupling);
  params.Add("affect_fluid", affect_fluid);
  params.Add(
      "fixed_temp_rhs",
      pin->GetOrAddBoolean(
          radiation_block, "fixed_temp_rhs", false,
          "Do not solve for an advanced temperature in the radiation source term"));
  return coupling;
}

//----------------------------------------------------------------------------------------
//! \fn void AddCouplingUtils
//! \brief Build the Rusanov flux and temperature root-find utility structs (only
//! populated when coupling is enabled) and store them.
inline void AddCouplingUtils(ParameterInput *pin, Params &params, const bool coupling) {
  FluxUtils futils;
  RootUtils rutils;
  if (coupling) {
    // Flux params
    const Real beta_default = (do_angular_fluxes) ? 0.0 : 1.0;
    futils.beta = pin->GetOrAddReal(
        radiation_block, "beta", beta_default,
        "Coefficient weighting the local optical depth for the Rusanov flux");
    futils.taumax =
        pin->GetOrAddReal(radiation_block, "taumax", std::numeric_limits<Real>::max(),
                          "Optical depth max used in Rusanov flux");
    PARTHENON_REQUIRE(futils.taumax > 0.0, "Tau max must be > 0!");
    // Root find params
    rutils.tol = pin->GetOrAddReal(radiation_block, "troot_tol", 1.0e-8,
                                   "Tolerance for non-linear temperature root find ");
    rutils.titer = pin->GetOrAddInteger(
        radiation_block, "troot_max_iter", 25,
        "Maximum #iter permitted for non-linear temperature root find");
    PARTHENON_REQUIRE(rutils.tol > 0.0, "Temp rootfind tolerance must be > 0!");
    PARTHENON_REQUIRE(rutils.titer > 0, "Temp rootfind iterations must be > 0!");
  }
  params.Add("flux_utils", futils);
  params.Add("root_utils", rutils);
}

//----------------------------------------------------------------------------------------
//! \fn int AddAngularMesh
//! \brief Construct the requested angular mesh (geodesic or latlon), store it and the
//! "angular_mesh" selector, and return the number of angles.
inline int AddAngularMesh(ParameterInput *pin, Params &params) {
  const std::string amesh =
      pin->GetOrAddString(radiation_block, "angular_mesh", "geodesic",
                          "Choose which angular mesh to use for SN");
  params.Add("angular_mesh", amesh);
  int nangles = 0;
  if (amesh == "geodesic") {
    const int nlevel = pin->GetOrAddInteger(
        radiation_block, "nlevel", 1,
        "For geodesic grid, set nlevel, where nangles=10*nlevel^2 + 2");
    const int rotate =
        pin->GetOrAddInteger(radiation_block, "rotate_geo", 1,
                             "0: do not rotate geodesic grid, 1: automatically rotate "
                             "geodesic grid to dissuade alignment with spatial mesh, 2: "
                             "User supplies rotation angles for geodesic grid");
    const Real qnan = std::numeric_limits<Real>::quiet_NaN();
    const Real zpole =
        pin->GetOrAddReal(radiation_block, "zpole", qnan,
                          "Zeta rotation angle for manual geodesic grid rotation");
    const Real ppole =
        pin->GetOrAddReal(radiation_block, "ppole", qnan,
                          "Psi rotation angle for manual geodesic grid rotation");
    std::shared_ptr<GeodesicGrid> prgeo =
        std::make_unique<GeodesicGrid>(nlevel, rotate, zpole, ppole);
    params.Add("geodesic_grid", prgeo);
    nangles = prgeo->nangles;
  } else if (amesh == "latlon") {
    const int ntheta = pin->GetOrAddInteger(radiation_block, "ntheta", 8,
                                            "For latlon grid, number of latitude bins");
    const int nphi = pin->GetOrAddInteger(radiation_block, "nphi", 16,
                                          "For latlon grid, number of longitude bins");
    std::shared_ptr<LatLonGrid> prlatlon = std::make_unique<LatLonGrid>(ntheta, nphi);
    params.Add("latlon_grid", prlatlon);
    nangles = prlatlon->nangles;
  } else {
    PARTHENON_THROW("Invalid angular mesh choice!");
  }
  params.Add("nangles", nangles);
  return nangles;
}

//----------------------------------------------------------------------------------------
//! \fn int AddGroupStructure
//! \brief Source the frequency group structure from the materials package (which owns
//! it), store "ngroups" and the frequency-bound arrays ("fbnd", "fbnd_d", "fbnd_h"), and
//! return the number of groups.
//!
//! NOTE(@pdmullen): The materials package owns the group structure (i.e., the number of
//! groups and the frequency [Hz] boundaries).  It resolves this structure either from the
//! <materials> input block, from singularity-opac tables, or a gray default, and ensures
//! all material opacity models agree.  We source it here rather than constructing our
//! own, so the radiation transport is consistent with the opacities.
inline int AddGroupStructure(StateDescriptor *materials, Params &params) {
  const int ngroups = materials->Param<int>("ngroups");
  const auto group_bounds = materials->Param<std::vector<Real>>("group_bounds");
  PARTHENON_REQUIRE(static_cast<int>(group_bounds.size()) == (ngroups + 1),
                    "materials group_bounds must have ngroups+1 edges.");
  params.Add("ngroups", ngroups);

  // Group boundaries (frequency bounds in Hz, sourced from materials)
  std::vector<Real> fbnd(group_bounds);
  auto fbnd_d = ParArray1D<Real>("fbnd_d", fbnd.size());
  auto fbnd_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), fbnd_d);
  for (int bin = 0; bin < fbnd.size(); ++bin) {
    fbnd_h(bin) = fbnd[bin];
  }
  Kokkos::deep_copy(fbnd_d, fbnd_h);
  params.Add("fbnd_d", fbnd_d, true);
  params.Add("fbnd_h", fbnd_h, true);
  params.Add("fbnd", fbnd, true);
  return ngroups;
}

//----------------------------------------------------------------------------------------
//! \fn void AddFixedPgenOpac
//! \brief Store the flag that fixes opacities to the values set in the ProblemGenerator.
inline void AddFixedPgenOpac(ParameterInput *pin, Params &params) {
  params.Add("fixed_pgen_opac",
             pin->GetOrAddBoolean(
                 radiation_block, "fixed_pgen_opac", false,
                 "Fix the opacities to the values set in the ProblemGenerator"));
}

} // namespace ConfigOption

//----------------------------------------------------------------------------------------
//! \fn SharedParams AddSharedParams
//! \brief Populate a radiation package's Params with all parameters common to the
//! explicit and jacobi solvers: the sparse-physics guard, CFL number, unit system,
//! coupling flags and utils, angular mesh, group structure, and fixed-opacity flag.
//! These are all read from the shared <radiation_transport> input block and are ordering-
//! independent.  Returns the angle/group counts the caller needs for field sizing.
inline SharedParams AddSharedParams(ParameterInput *pin, StateDescriptor *materials,
                                    Params &params) {
  const bool sparse_physics = pin->GetOrAddBoolean("physics", "sparse_physics", true);
  PARTHENON_REQUIRE(!(sparse_physics), "Radiation incompatible with sparse physics.");

  ConfigOption::AddCfl(pin, params);
  ConfigOption::AddUnitUtils(pin, params);
  const bool coupling = ConfigOption::AddCouplingParams(pin, params);
  ConfigOption::AddCouplingUtils(pin, params, coupling);
  ConfigOption::AddFixedPgenOpac(pin, params);
  SharedParams shared;
  shared.nangles = ConfigOption::AddAngularMesh(pin, params);
  shared.ngroups = ConfigOption::AddGroupStructure(materials, params);
  return shared;
}

} // namespace RadiationShared

#endif // RADIATION_ALGORITHMS_RADIATION_SHARED_HPP_
