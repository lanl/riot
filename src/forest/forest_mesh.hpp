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
#ifndef FOREST_FOREST_MESH_HPP_
#define FOREST_FOREST_MESH_HPP_
// This file was made in part with generative AI.

#include <optional>

#include <mesh/forest/forest.hpp>
#include <parameter_input.hpp>
#include <parthenon_manager.hpp>

namespace RiotForest {

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
//!
//! type = ball3d: the 3D box-in-sphere mesh for MappedCartesian coordinates. A cube
//! [-a, a]^3 (one tree) is surrounded by six wedge trees, one per cube face, that blend
//! to the sphere r = r_inner, followed by shells of six equiangular cubed-sphere trees
//! at the radii shell_radii. In the trees of each cube face, logical x3 is outward, so
//! the outer sphere is always an outer-x3 boundary. Only core_edge = line is supported.
std::optional<parthenon::forest::ForestDefinition>
MakeForest(parthenon::ParameterInput *pin);

//! \brief MakeForest from the manager's input (on restarts and analysis, that of the
//! file).
std::optional<parthenon::forest::ForestDefinition>
MakeForest(parthenon::ParthenonManager &pman);

//! \brief Radius of the sphere that bounds the ball2d (revolved about the axis) and
//! ball3d meshes, the last of riot/forest/shell_radii
parthenon::Real OuterRadius(parthenon::ParameterInput *pin);

} // namespace RiotForest

#endif // FOREST_FOREST_MESH_HPP_
