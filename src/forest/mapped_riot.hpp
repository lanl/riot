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
#ifndef FOREST_MAPPED_RIOT_HPP_
#define FOREST_MAPPED_RIOT_HPP_
// This file was made in part with generative AI.

#include <interface/state_descriptor.hpp>
#include <parameter_input.hpp>

namespace RiotForest {

//----------------------------------------------------------------------------------------
//! \fn  void Initialize
//! \brief Mapped-mesh setup of the riot package, called once its physics params are set
//! (a no-op unless the coordinates are mapped):
//!   - requires that only the physics written for mapped meshes is enabled,
//!   - marks the geometry curvilinear,
//!   - adds the param mapped_face_frame (<hydro>/mapped_velocity_frame),
//!   - adds the physical node positions n.coordinates, the output geometry.
void Initialize(parthenon::ParameterInput *pin, parthenon::StateDescriptor *riot);

} // namespace RiotForest

#endif // FOREST_MAPPED_RIOT_HPP_
