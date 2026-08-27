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
#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <parthenon/package.hpp>

#include "variables.hpp"

namespace Gravity {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus CalculateGravitySource(MeshData<Real> *state, MeshData<Real> *src);

} // namespace Gravity

#endif // GRAVITY_GRAVITY_HPP_
