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
// This file was made in part with generative AI.
//! \file poisson_driver.cpp
//! \brief Tasks to solve the Poisson equation for the gravitational potential (following
//! the poisson_gmg example in Parthenon).

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

#include "self_gravity/self_gravity.hpp"
#include "variables.hpp"

using namespace parthenon::driver::prelude;

namespace SelfGravity {

//----------------------------------------------------------------------------------------
//! \fn void SelfGravity::SolvePoisson
//! \brief Adds a task region that solves Laplacian(phi) = 4 pi G rho and stores phi (with
//! filled ghost zones) in the base container
void SolvePoisson(TaskCollection &tc, Mesh *pmesh) {
  using namespace parthenon;
  using phi_tl = parthenon::TypeList<phi>;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("self_gravity");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md_u = pmesh->mesh_data.Add(self_gravity_u_container, md, {phi::name()});
    auto &md_rhs = pmesh->mesh_data.Add(self_gravity_rhs_container, md, {phi::name()});

    // Set the RHS and warm start the solver from the current potential
    auto set_rhs = tl.AddTask(none, FillPoissonRHS, md.get(), md_rhs.get());
    auto set_guess = tl.AddTask(none, TF(solvers::utils::CopyData<phi_tl>), md, md_u);

    // Solve
    auto setup = psolver->AddSetupTasks(tl, set_rhs | set_guess, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Set BCs after solve
    auto bcs = parthenon::AddBoundaryExchangeTasks(solve, tl, md_u, pmesh->multilevel);

    // Move the solution back to the base container for sourcing and output
    tl.AddTask(bcs, TF(solvers::utils::CopyData<phi_tl>), md_u, md);
  }
}

} // namespace SelfGravity
