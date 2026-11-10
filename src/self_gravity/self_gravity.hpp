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
#ifndef SELF_GRAVITY_SELF_GRAVITY_HPP_
#define SELF_GRAVITY_SELF_GRAVITY_HPP_
// This file was made in part with generative AI.

#include <chrono>
#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "variables.hpp"

namespace SelfGravity {
using namespace parthenon::package::prelude;

// Solver containers. The base container is the default "base" MeshData, which holds the
// (operator split) potential.
constexpr char self_gravity_u_container[] = "self_gravity_u";
constexpr char self_gravity_rhs_container[] = "self_gravity_rhs";

// Poisson solve statistics for the cycle diagnostics. The step_* members are reset at
// the start of every step, the total_* members accumulate over the run.
struct SolveStats {
  int step_solves = 0;
  int step_iterations = 0;
  int step_max_iterations = 0;
  // Solves that stopped at max_iterations or with a non-finite residual (the solver
  // does not flag either itself)
  int step_unconverged = 0;
  // NaN/inf if any solve this step had a non-finite residual
  Real step_max_residual = 0.0;
  // Wall time in the Poisson task regions on this rank, which includes time spent
  // waiting on other ranks, and its min/max over ranks (only valid on rank 0)
  Real step_time = 0.0;
  Real step_time_min = 0.0;
  Real step_time_max = 0.0;
  long total_solves = 0;
  long total_iterations = 0;
  long total_unconverged = 0;
  // Sum over steps of the slowest rank's Poisson time (only valid on rank 0)
  Real total_time_max = 0.0;
  std::chrono::steady_clock::time_point solve_start;
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus FillPoissonRHS(MeshData<Real> *md_base, MeshData<Real> *md_rhs);
TaskStatus CalculateSelfGravitySource(MeshData<Real> *state, MeshData<Real> *base,
                                      MeshData<Real> *src);
void SolvePoisson(parthenon::TaskCollection &tc, Mesh *pmesh);
void ResetStepStats(Mesh *pmesh);
void FinalizeStepStats(Mesh *pmesh);
void OutputCycleStats(Mesh *pmesh);
void ReportSolverStats(Mesh *pmesh);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
