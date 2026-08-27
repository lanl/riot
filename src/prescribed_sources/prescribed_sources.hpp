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
#ifndef PRESCRIBED_SOURCES_PRESCRIBED_SOURCES_HPP_
#define PRESCRIBED_SOURCES_PRESCRIBED_SOURCES_HPP_

#include <memory>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace PrescribedSources {

enum class SourceMode { Specific, Total /*, Volumetric */ };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
parthenon::TaskCollection Step(Mesh *pm, parthenon::SimTime &tm, const Real dt);
TaskStatus EnergySources(MeshData<Real> *md, parthenon::SimTime &tm, const Real dt);
Real EstimateTimestepMesh(MeshData<Real> *md);
void ReduceMasses(MeshData<Real> *md);

} // namespace PrescribedSources

#endif // PRESCRIBED_SOURCES_PRESCRIBED_SOURCES_HPP_
