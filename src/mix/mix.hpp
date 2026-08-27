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
// This file was made in part with generative AI.
#ifndef MIX_MIX_HPP_
#define MIX_MIX_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <kokkos_abstraction.hpp>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace Mix {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus CalculateMixSource(MeshData<Real> *state, MeshData<Real> *src);
TaskStatus ComputeViscousFluxes(MeshData<Real> *md);
TaskStatus ComputeStressFluxes(MeshData<Real> *md);
TaskStatus ComputeAnonFluxes(MeshData<Real> *md);
Real EstimateTimestepMesh(MeshData<Real> *rc);
void FillDerived(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//! \fn  Real Mix::three_points_to_source
//! \brief The purpose of this function is to do something like d/dx(A*d/dx(q))
KOKKOS_INLINE_FUNCTION
Real three_points_to_source(const Real ql, const Real qc, const Real qr, const Real al,
                            const Real ac, const Real ar, const Real dx) {
  // return ((qr-qc)/dx*(0.5*(ar+ac)) - (qc-ql)/dx*(0.5*(ac+al)))/dx;
  // NOTE(@chadmeyer): the following assumes geometric averaging of "a"
  return (al * (ql - qc) + ar * (qr - qc) + ac * (qr + ql - 2.0 * qc)) / (2.0 * dx * dx);
}

} // namespace Mix

#endif // MIX_MIX_HPP_
