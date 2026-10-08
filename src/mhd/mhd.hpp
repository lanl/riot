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

#include <cstdint>
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

//! Solver-health counters, incremented from inside the Riemann kernels.
//!
//! WHY THESE EXIST. HLLD's six degeneracy guards all fall back to HLLE. If one of them
//! were tripping on most cells of a production run, HLLD would be silently degraded to
//! HLLE and EVERY existing test would still pass -- the unit test that asserts the two
//! solvers differ catches the always-falling-back case, not the usually-falling-back one.
//! The port has no other way to answer "is the solver I selected the solver I am
//! getting?"
//!
//! Counters are cumulative over the run and are never reset, so the reported number is
//! "how often since t=0", not "since the last report".
enum SolverDiag {
  //! HLLD hit a degenerate state and used its HLLE fallback for that face.
  kDiagHlldFallback = 0,
  //! A solver clamped a non-positive reconstructed density. Routine use of this is a
  //! failed test, not a success (see the verification plan), which is why it is counted
  //! rather than silently applied.
  kDiagDensityFloor = 1,
  kNumSolverDiag = 2
};

//! Device-resident counter storage. A raw pointer -- not a View -- is what gets passed
//! into the flux kernels, because the three solvers are handed to `MHDFluxes` as plain
//! function pointers and a View in their signature would force them to become templates.
using SolverDiagView = Kokkos::View<std::int64_t *, parthenon::DevMemSpace>;

//! Installed as StateDescriptor::PostStepDiagnosticsMesh, ALWAYS.
//!
//! Two jobs with different trigger conditions, which is why they share one hook:
//!  - Solver-health counters are checked every step and reported whenever one INCREASES.
//!    Unconditional, because a silent HLLD degradation matters whether or not the user
//!    asked for diagnostics, and reporting on change rather than every step keeps it
//!    quiet when nothing is wrong.
//!  - The div B report runs only under `mhd/monitor_divb`, because unlike reading two
//!    counters it costs two mesh-wide reductions per step. It reports max |div B|, the
//!    volume-weighted mean, and the DIMENSIONLESS
//!    eta = |div B| * l_cell / max(|B|_cell, b_ref).
void PostStepDiagnostics(parthenon::SimTime const &simtime, MeshData<Real> *md);

//! The div B half of PostStepDiagnostics, separated so the reduction cost is visible at
//! the call site rather than buried behind a flag check.
void MonitorDivergence(parthenon::SimTime const &simtime, MeshData<Real> *md);

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
