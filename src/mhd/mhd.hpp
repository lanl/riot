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

//! Task-shaped wrapper for SetDerivedMagneticFields.
TaskStatus SetDerived(MeshData<Real> *md, IndexDomain domain);

//! Converts ccbulk::total_material_energy from the hydro convention to the MHD convention
//! by adding B^2/(2 mu0). MUST be called at the end of every MHD problem generator; see
//! the definition for why this cannot live in PostInitialization.
void AddMagneticEnergyToTotal(MeshBlock *pmb);

//! Post-problem-generator hook, installed as StateDescriptor::PostInitializationMesh.
//! A problem generator writes only face B (plus the usual gas state); this derives
//! everything else before the first total-energy assembly.
void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

//! Installed as StateDescriptor::UserWorkBeforeLoopMesh, which -- unlike
//! PostInitializationMesh -- also runs on a restart. Rebuilds the derived magnetic fields
//! from the checkpointed face state, which is otherwise left at zero through the first
//! post-restart stage's reconstruction. No-op on a fresh start.
void RestoreDerivedOnRestart(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm);

//! Assembles the Gardiner-Stone upwind EMF on every edge of the interior into the edge
//! flux register of the face magnetic field. Depends on the hydro flux task, which
//! writes both the transverse induction fluxes and the material mass fluxes it reads.
TaskStatus AssembleEdgeEMF(MeshData<Real> *md);

//! Constrained-transport update of the face magnetic field: the Stokes curl of the edge
//! EMFs, with the integrator's own low-storage RK coefficients applied exactly as
//! sparse_update::UpdateToNextStage applies them to cell-centered state.
//!
//! `u0md` holds the state being advanced and the edge EMF register; `u1md` is the stage
//! register. Face-centered independent state is deliberately excluded from
//! UpdateToNextStage (which is a cell-centered flux divergence) and updated here instead.
TaskStatus ApplyFaceUpdate(MeshData<Real> *u0md, MeshData<Real> *u1md, Real gam0,
                           Real gam1, Real beta_dt);

} // namespace MHD

#endif // MHD_MHD_HPP_
