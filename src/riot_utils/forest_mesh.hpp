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
#ifndef RIOT_UTILS_FOREST_MESH_HPP_
#define RIOT_UTILS_FOREST_MESH_HPP_
// This file was made in part with generative AI.

#include <optional>

#include <mesh/forest/forest.hpp>
#include <parameter_input.hpp>

namespace RiotMesh {

//----------------------------------------------------------------------------------------
//! \fn  std::optional<ForestDefinition> MakeForest
//! \brief Build a forest-of-trees mesh definition from <riot/forest>, or return nullopt
//! for the default hyper-rectangular mesh.
//!
//! type = ball2d: axisymmetric (R >= 0, z) "cube-in-sphere" mesh for MappedCylindrical
//! coordinates. A half square core [0, a] x [-a, a] (two trees) is surrounded by four
//! wedge trees that blend to the circle r = r_inner, followed by polar shells (four
//! 45-degree trees each) at the radii shell_radii, which start at r_inner. Every tree has
//! logical x1 along the polar angle and x2 outward (in the core, x1 = R and x2 = |z|),
//! so the axis is always an inner-x1 boundary and the outer sphere an outer-x2 boundary.
std::optional<parthenon::forest::ForestDefinition> MakeForest(parthenon::ParameterInput *pin);

} // namespace RiotMesh

#endif // RIOT_UTILS_FOREST_MESH_HPP_
