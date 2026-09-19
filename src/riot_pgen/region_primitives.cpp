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

namespace region_primitives {
//----------------------------------------------------------------------------------------
//! \fn  mask_func_t background
//! \brief
mask_func_t background(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(background_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_sphere
//! \brief
mask_func_t inside_sphere(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_sphere_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_spherical_shell
//! \brief
mask_func_t inside_spherical_shell(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_spherical_shell_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_cylinder
//! \brief
mask_func_t inside_cylinder(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_cylinder_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_cylindrical_shell
//! \brief
mask_func_t inside_cylindrical_shell(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_cylindrical_shell_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_ellipsoid
//! \brief
mask_func_t inside_ellipsoid(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_ellipsoid_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_ellipsoidal_shell
//! \brief
mask_func_t inside_ellipsoidal_shell(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_ellipsoidal_shell_lambda(pin, block_name));
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t inside_rectangle
//! \brief
mask_func_t inside_rectangle(ParameterInput *pin, const std::string &block_name) {
  return base_region_loop(inside_rectangle_lambda(pin, block_name));
}

} // namespace region_primitives
