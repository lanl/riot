//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#include "region_primitives.hpp"

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t background
//! \brief
mask_func_t background(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop([=](const Real x, const Real y, const Real z) { return true; });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_sphere
//! \brief
mask_func_t inside_sphere(ParameterInput *pin, const std::string &block_name) {
  Real r = pin->GetOrAddReal(block_name, "radius", 1.0);
  Real rsq = r * r;
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0, "x-coordinate of center of sphere");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0, "y-coordinate of center of sphere");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0, "z-coordinate of center of sphere");

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    Real dx = x - x0;
    dx *= dx;
    Real dy = y - y0;
    dy *= dy;
    Real dz = z - z0;
    dz *= dz;
    return (dx + dy + dz < rsq);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_spherical_shell
//! \brief
mask_func_t inside_spherical_shell(ParameterInput *pin, const std::string &block_name) {
  Real r0 = pin->GetOrAddReal(block_name, "inner_radius", 0.0, "Radius of inner surface");
  Real r1 = pin->GetOrAddReal(block_name, "outer_radius", 1.0, "Radius of outer surface");
  Real r0sq = r0 * r0;
  Real r1sq = r1 * r1;
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of center of spherical shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of center of spherical shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of center of spherical shell");

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    Real dx = x - x0;
    dx *= dx;
    Real dy = y - y0;
    dy *= dy;
    Real dz = z - z0;
    dz *= dz;
    const Real rsq = dx + dy + dz;
    return (rsq > r0sq && rsq < r1sq);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_cylinder
//! \brief
mask_func_t inside_cylinder(ParameterInput *pin, const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of bottom of cylindrical volume");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of bottom of cylindrical volume");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of bottom of cylindrical volume");
  Real x1 = pin->GetOrAddReal(block_name, "x1", 0.0,
                              "x-coordinate of top of cylindrical volume");
  Real y1 = pin->GetOrAddReal(block_name, "y1", 0.0,
                              "y-coordinate of top of cylindrical volume");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1.0,
                              "z-coordinate of top of cylindrical volume");
  Real r = pin->GetOrAddReal(block_name, "radius", 1.0);
  Real rsq = r * r;

  Real nx = x1 - x0;
  Real ny = y1 - y0;
  Real nz = z1 - z0;
  Real dsq = nx * nx + ny * ny + nz * nz;

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    // check if point is above bottom of cylinder
    Real dx0 = x - x0;
    Real dy0 = y - y0;
    Real dz0 = z - z0;
    Real dot = nx * dx0 + ny * dy0 + nz * dz0;
    if (dot < 0.0) return false;
    // check if point is below top of cylinder
    Real dx1 = x - x1;
    Real dy1 = y - y1;
    Real dz1 = z - z1;
    dot = nx * dx1 + ny * dy1 + nz * dz1;
    if (dot > 0.0) return false;
    // now check if point is inside of cylinder
    Real cx = dy0 * dz1 - dz0 * dy1;
    Real cy = dz0 * dx1 - dx0 * dz1;
    Real cz = dx0 * dy1 - dy0 * dx1;
    Real csq = cx * cx + cy * cy + cz * cz;
    return (csq / dsq < rsq);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_cylindrical_shell
//! \brief
mask_func_t inside_cylindrical_shell(ParameterInput *pin, const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of bottom of cylindrical shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of bottom of cylindrical shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of bottom of cylindrical shell");
  Real x1 = pin->GetOrAddReal(block_name, "x1", 0.0,
                              "x-coordinate of top of cylindrical shell");
  Real y1 = pin->GetOrAddReal(block_name, "y1", 0.0,
                              "y-coordinate of top of cylindrical shell");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1.0,
                              "z-coordinate of top of cylindrical shell");
  Real r0 = pin->GetOrAddReal(block_name, "inner_radius", 0.0, "Radius of inner surface");
  Real r1 = pin->GetOrAddReal(block_name, "outer_radius", 1.0, "Radius of outer surface");
  Real r0sq = r0 * r0;
  Real r1sq = r1 * r1;

  Real nx = x1 - x0;
  Real ny = y1 - y0;
  Real nz = z1 - z0;
  Real dsq = nx * nx + ny * ny + nz * nz;

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    // check if point is above bottom of cylinder
    Real dx0 = x - x0;
    Real dy0 = y - y0;
    Real dz0 = z - z0;
    Real dot = nx * dx0 + ny * dy0 + nz * dz0;
    if (dot < 0.0) return false;
    // check if point is below top of cylinder
    Real dx1 = x - x1;
    Real dy1 = y - y1;
    Real dz1 = z - z1;
    dot = nx * dx1 + ny * dy1 + nz * dz1;
    if (dot > 0.0) return false;
    // now check if point is inside of cylindrical shell
    Real cx = dy0 * dz1 - dz0 * dy1;
    Real cy = dz0 * dx1 - dx0 * dz1;
    Real cz = dx0 * dy1 - dy0 * dx1;
    Real csq = (cx * cx + cy * cy + cz * cz) / dsq;
    return (csq > r0sq && csq < r1sq);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_ellipsoid
//! \brief
mask_func_t inside_ellipsoid(ParameterInput *pin, const std::string &block_name) {
  Real x0 =
      pin->GetOrAddReal(block_name, "x0", 0.0, "x-coordinate of center of ellipsoid");
  Real y0 =
      pin->GetOrAddReal(block_name, "y0", 0.0, "y-coordinate of center of ellipsoid");
  Real z0 =
      pin->GetOrAddReal(block_name, "z0", 0.0, "z-coordinate of center of ellipsoid");
  Real ax = pin->GetOrAddReal(block_name, "ax", 1.0, "Eccentricity parameter in x");
  Real ay = pin->GetOrAddReal(block_name, "ay", 1.0, "Eccentricity parameter in y");
  Real az = pin->GetOrAddReal(block_name, "az", 1.0, "Eccentricity parameter in z");
  Real asq = ax * ax;
  Real bsq = ay * ay;
  Real csq = az * az;

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    Real dxsq = x - x0;
    dxsq *= dxsq;
    Real dysq = y - y0;
    dysq *= dysq;
    Real dzsq = z - z0;
    dzsq *= dzsq;
    return (dxsq / asq + dysq / bsq + dzsq / csq < 1.0);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_ellipsoidal_shell
//! \brief
mask_func_t inside_ellipsoidal_shell(ParameterInput *pin, const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of center of ellipsoidal shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of center of ellipsoidal shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of center of ellipsoidal shell");
  Real ax0 = pin->GetOrAddReal(block_name, "inner_ax", 1.0,
                               "Eccentricity parameter of inner surface in x");
  Real ay0 = pin->GetOrAddReal(block_name, "inner_ay", 1.0,
                               "Eccentricity parameter of inner surface in y");
  Real az0 = pin->GetOrAddReal(block_name, "inner_az", 1.0,
                               "Eccentricity parameter of inner surface in z");
  Real ax1 = pin->GetOrAddReal(block_name, "outer_ax", 1.0,
                               "Eccentricity parameter of outer surface in x");
  Real ay1 = pin->GetOrAddReal(block_name, "outer_ay", 1.0,
                               "Eccentricity parameter of outer surface in y");
  Real az1 = pin->GetOrAddReal(block_name, "outer_az", 1.0,
                               "Eccentricity parameter of outer surface in z");

  Real a0sq = ax0 * ax0;
  Real b0sq = ay0 * ay0;
  Real c0sq = az0 * az0;
  Real a1sq = ax1 * ax1;
  Real b1sq = ay1 * ay1;
  Real c1sq = az1 * az1;

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    Real dxsq = x - x0;
    dxsq *= dxsq;
    Real dysq = y - y0;
    dysq *= dysq;
    Real dzsq = z - z0;
    dzsq *= dzsq;
    Real eq0 = dxsq / a0sq + dysq / b0sq + dzsq / c0sq;
    Real eq1 = dxsq / a1sq + dysq / b1sq + dzsq / c1sq;
    return (eq0 > 1.0 && eq1 < 1.0);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_rectangle
//! \brief
mask_func_t inside_rectangle(ParameterInput *pin, const std::string &block_name) {
  Real x0 =
      pin->GetOrAddReal(block_name, "x0", -1e300, "Left edge. Default is minus infinity");
  Real y0 = pin->GetOrAddReal(block_name, "y0", -1e300,
                              "Bottom edge. Default is minus infinity");
  Real z0 = pin->GetOrAddReal(block_name, "z0", -1e300,
                              "Vertical-bottom edge. Default is minus infinity");
  Real x1 =
      pin->GetOrAddReal(block_name, "x1", 1e300, "Right edge. Default is plus infinity");
  Real y1 =
      pin->GetOrAddReal(block_name, "y1", 1e300, "Top edge. Default is plus infinity");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1e300,
                              "Vertical-top edge. Default is plus infinity");

  return base_region_loop([=](const Real x, const Real y, const Real z) {
    bool xin = (x >= x0 && x <= x1);
    bool yin = (y >= y0 && y <= y1);
    bool zin = (z >= z0 && z <= z1);
    return (xin && yin) && zin;
  });
}
