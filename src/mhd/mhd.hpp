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
#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
// This file was made in part with generative AI.
//
// Ideal MHD with face-centered constrained transport (CT).
//
// Ported from the ideal-MHD implementation on the lanl/artemis branch dempsey/mhd
// @ 3e5aeb5 (see plan_histories/artemis_mhd_port/DONOR_DELTA.md for the file-by-file
// accounting, and the ADRs there for the units, energy, and topology contracts).
//
// SCOPE (ADR-004): Cartesian, uniform grid, single material, ideal gas, one
// temperature. Every other combination is rejected at startup in riot.cpp -- this is
// deliberate, not an oversight. Refinement metadata is registered on the face field
// because it is cheap and correct to do so, but AMR is NOT validated and is rejected.

#include <memory>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace MHD {

//! Registers the mhd package: the evolved face-centered magnetic field, its derived
//! cell-centered companions, and the mu0 normalization parameter.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//! Fills ccbulk::magnetic_field, ccbulk::magnetic_energy and
//! ccbulk::div_magnetic_field from the authoritative face state over `domain`.
//! Divergence is computed from the face flux balance, never from centered differences
//! of the cell-centered field -- the latter would not vanish for a discretely
//! divergence-free field and so would be useless as a diagnostic.
void SetDerivedMagneticFields(MeshData<Real> *md, IndexDomain domain);

//! Post-problem-generator hook, installed as StateDescriptor::PostInitializationMesh.
//! A problem generator writes only face B (plus the usual gas state); this derives
//! everything else before the first total-energy assembly.
void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

} // namespace MHD

#endif // MHD_MHD_HPP_
