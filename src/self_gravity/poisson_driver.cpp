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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <globals.hpp>
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

  auto *stats = pkg->MutableParam<SolveStats>("solve_stats");
  const int max_iterations = pkg->Param<int>("max_iterations");
  const bool print_per_solve = pkg->Param<bool>("print_per_solve");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];

    auto start = tl.AddTask(TaskQualifier::local_sync | TaskQualifier::once_per_region,
                            none, "start solve timer", [stats]() {
                              stats->solve_start = std::chrono::steady_clock::now();
                              return TaskStatus::complete;
                            });
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md_u = pmesh->mesh_data.Add(self_gravity_u_container, md, {phi::name()});
    auto &md_rhs = pmesh->mesh_data.Add(self_gravity_rhs_container, md, {phi::name()});

    // Set the RHS and warm start the solver from the current potential
    auto set_rhs = tl.AddTask(start, FillPoissonRHS, md.get(), md_rhs.get());
    auto set_guess = tl.AddTask(start, TF(solvers::utils::CopyData<phi_tl>), md, md_u);

    // Solve
    auto setup = psolver->AddSetupTasks(tl, set_rhs | set_guess, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Set BCs after solve
    auto bcs = parthenon::AddBoundaryExchangeTasks(solve, tl, md_u, pmesh->multilevel);

    // Move the solution back to the base container for sourcing and output
    auto copy_back = tl.AddTask(bcs, TF(solvers::utils::CopyData<phi_tl>), md_u, md);

    // Record the solve once all partitions on this rank are done. The iterations and
    // residual are global, so they are the same on every rank.
    auto record = [stats, psolver, max_iterations, print_per_solve]() {
      Kokkos::fence();
      const auto end = std::chrono::steady_clock::now();
      const int iters = psolver->GetFinalIterations();
      const Real residual = psolver->GetFinalResidual();
      // The solvers test residual > tolerance, so a NaN residual stops them as if
      // converged
      const bool finite = std::isfinite(residual);
      const bool unconverged = (iters >= max_iterations) || !finite;
      stats->step_solves++;
      stats->step_iterations += iters;
      stats->step_max_iterations = std::max(stats->step_max_iterations, iters);
      stats->step_unconverged += unconverged;
      // Keep a non-finite residual once seen, so it shows up in the cycle output
      if (std::isfinite(stats->step_max_residual) &&
          (!finite || residual > stats->step_max_residual))
        stats->step_max_residual = residual;
      stats->step_time += std::chrono::duration<Real>(end - stats->solve_start).count();
      if (print_per_solve && Globals::my_rank == 0) {
        const char *note = !finite      ? " (non-finite residual)"
                           : unconverged ? " (hit max_iterations)"
                                         : "";
        printf("self_gravity: iterations = %i residual = %e%s\n", iters, residual, note);
      }
      return TaskStatus::complete;
    };
    tl.AddTask(TaskQualifier::local_sync | TaskQualifier::once_per_region, copy_back,
               "record solve stats", record);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SelfGravity::ResetStepStats
//! \brief Zeros the per-step solve statistics, called at the start of every step
void ResetStepStats(Mesh *pmesh) {
  auto *stats =
      pmesh->packages.Get("self_gravity")->MutableParam<SolveStats>("solve_stats");
  stats->step_solves = 0;
  stats->step_iterations = 0;
  stats->step_max_iterations = 0;
  stats->step_unconverged = 0;
  stats->step_max_residual = 0.0;
  stats->step_time = 0.0;
  stats->step_time_min = 0.0;
  stats->step_time_max = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void SelfGravity::FinalizeStepStats
//! \brief Reduces the per-rank Poisson time to rank 0 and accumulates the run totals,
//! called (collectively) after the step's solves
void FinalizeStepStats(Mesh *pmesh) {
  auto *stats =
      pmesh->packages.Get("self_gravity")->MutableParam<SolveStats>("solve_stats");
  stats->step_time_min = stats->step_time;
  stats->step_time_max = stats->step_time;
#ifdef MPI_PARALLEL
  // max of (t, -t) gives (max, -min) in a single reduction
  Real t[2] = {stats->step_time, -stats->step_time};
  Real tmax[2];
  PARTHENON_MPI_CHECK(
      MPI_Reduce(t, tmax, 2, MPI_PARTHENON_REAL, MPI_MAX, 0, MPI_COMM_WORLD));
  stats->step_time_max = tmax[0];
  stats->step_time_min = -tmax[1];
#endif // MPI_PARALLEL
  stats->total_solves += stats->step_solves;
  stats->total_iterations += stats->step_iterations;
  stats->total_unconverged += stats->step_unconverged;
  stats->total_time_max += stats->step_time_max;
}

//----------------------------------------------------------------------------------------
//! \fn void SelfGravity::OutputCycleStats
//! \brief Appends the step's solve statistics to the cycle diagnostics line (rank 0)
void OutputCycleStats(Mesh *pmesh) {
  const auto &stats =
      pmesh->packages.Get("self_gravity")->Param<SolveStats>("solve_stats");
  printf(" sg_solves=%i sg_iters=%i sg_max_iters=%i sg_unconverged=%i sg_res=%.2e "
         "sg_wsec=[%.2e,%.2e]",
         stats.step_solves, stats.step_iterations, stats.step_max_iterations,
         stats.step_unconverged, stats.step_max_residual, stats.step_time_min,
         stats.step_time_max);
}

//----------------------------------------------------------------------------------------
//! \fn void ReportGMGCommunication
//! \brief Prints, for each multigrid level of the current mesh, how blocks are spread
//! over ranks and how many block-to-block receives are rank-local or remote, both on
//! the level (smoothing and residual boundary exchanges) and to the next coarser level
//! (restriction and prolongation). Distinct remote ranks approximate the number of
//! messages per exchange with coalesced comms. Collective.
namespace {
void ReportGMGCommunication(Mesh *pmesh) {
  using parthenon::GridType;
  const int my_rank = parthenon::Globals::my_rank;
  const int lmin = pmesh->GetGMGMinLevel();
  const int lmax = pmesh->GetGMGMaxLevel();
  const int nlev = lmax - lmin + 1;
  // Per-level counters, summed and maxed over ranks
  enum { nblocks, same_local, same_remote, same_ranks, inter_local, inter_remote,
         inter_ranks, has_blocks, nfield };
  std::vector<long> local(nlev * nfield, 0);
  for (int level = lmin; level <= lmax; level++) {
    long *c = &local[(level - lmin) * nfield];
    const auto grid = pmesh->GetGMGGrid(level);
    std::set<int> same_ranks_set, inter_ranks_set;
    auto count = [&](const std::vector<parthenon::NeighborBlock> &nbs, long &n_local,
                     long &n_remote, std::set<int> &ranks) {
      for (const auto &nb : nbs) {
        if (nb.rank == my_rank) {
          n_local++;
        } else {
          n_remote++;
          ranks.insert(nb.rank);
        }
      }
    };
    for (const auto &part : pmesh->GetMultigridBlockPartitions(level)) {
      for (const auto &pmb : part->block_list) {
        c[nblocks]++;
        // Neighbors on this level's grid. On a two-level composite grid, blocks one
        // logical level coarser are the boundary of the grid.
        if (grid.type() == GridType::leaf) {
          count(pmb->GetNeighbors(), c[same_local], c[same_remote], same_ranks_set);
        } else if (pmb->loc.level() == grid.logical_level()) {
          count(pmb->GetGMGSameNeighbors(), c[same_local], c[same_remote],
                same_ranks_set);
        } else {
          count(pmb->GetGMGCompositeFinerNeighbors(), c[same_local], c[same_remote],
                same_ranks_set);
          continue; // boundary blocks get their inter-grid neighbors on their own level
        }
        count(pmb->GetGMGCoarserNeighbors(), c[inter_local], c[inter_remote],
              inter_ranks_set);
      }
    }
    c[same_ranks] = same_ranks_set.size();
    c[inter_ranks] = inter_ranks_set.size();
    c[has_blocks] = (c[nblocks] > 0);
  }
  std::vector<long> sum(local), max(local);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Reduce(local.data(), sum.data(), local.size(), MPI_LONG,
                                 MPI_SUM, 0, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Reduce(local.data(), max.data(), local.size(), MPI_LONG,
                                 MPI_MAX, 0, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
  if (my_rank != 0) return;
  printf("self_gravity multigrid communication on the final mesh over %i ranks, as "
         "total(max per rank):\n",
         parthenon::Globals::nranks);
  printf("  %5s %6s %12s %12s %12s %12s %12s %12s\n", "level", "ranks", "blocks",
         "same local", "same remote", "same ranks", "coarse remote", "coarse ranks");
  for (int level = lmax; level >= lmin; level--) {
    const long *s = &sum[(level - lmin) * nfield];
    const long *m = &max[(level - lmin) * nfield];
    auto pair = [](long t, long mx) {
      return std::to_string(t) + "(" + std::to_string(mx) + ")";
    };
    printf("  %5i %6li %12s %12s %12s %12s %12s %12s\n", level, s[has_blocks],
           pair(s[nblocks], m[nblocks]).c_str(),
           pair(s[same_local], m[same_local]).c_str(),
           pair(s[same_remote], m[same_remote]).c_str(),
           pair(s[same_ranks], m[same_ranks]).c_str(),
           pair(s[inter_remote], m[inter_remote]).c_str(),
           pair(s[inter_ranks], m[inter_ranks]).c_str());
  }
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void SelfGravity::ReportSolverStats
//! \brief Prints the run's solve totals and, with self_gravity/time_solver, the solver
//! task timings (per multigrid level) as min/avg/max over ranks. Collective.
void ReportSolverStats(Mesh *pmesh) {
  auto pkg = pmesh->packages.Get("self_gravity");
  const auto &stats = pkg->Param<SolveStats>("solve_stats");
  if (parthenon::Globals::my_rank == 0) {
    printf("\nself_gravity: %li solves, %li iterations (%.2f per solve), %li unconverged "
           "(max_iterations or non-finite residual), %.3e s in Poisson regions (slowest "
           "rank per step)\n",
           stats.total_solves, stats.total_iterations,
           stats.total_iterations / std::max(1.0, static_cast<Real>(stats.total_solves)),
           stats.total_unconverged, stats.total_time_max);
  }
  if (!pkg->Param<bool>("time_solver")) return;
  ReportGMGCommunication(pmesh);

  // Ranks only register timers for the levels and partitions they own, so gather each
  // rank's "label\ttime" lines rather than reducing a fixed-size array
  std::ostringstream os;
  for (const auto &[label, acc] : parthenon::solvers::SolverBase::solver_timings) {
    if (acc) os << label << '\t' << acc->GetTotalTime() << '\n';
  }
  std::string local = os.str();
  std::vector<std::string> per_rank = {local};
#ifdef MPI_PARALLEL
  const int nranks = parthenon::Globals::nranks;
  int len = local.size();
  std::vector<int> lens(nranks), displs(nranks, 0);
  PARTHENON_MPI_CHECK(
      MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
  std::string all;
  if (parthenon::Globals::my_rank == 0) {
    for (int r = 1; r < nranks; r++)
      displs[r] = displs[r - 1] + lens[r - 1];
    all.resize(displs[nranks - 1] + lens[nranks - 1]);
  }
  PARTHENON_MPI_CHECK(MPI_Gatherv(local.data(), len, MPI_CHAR, all.data(), lens.data(),
                                  displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD));
  if (parthenon::Globals::my_rank != 0) return;
  per_rank.clear();
  for (int r = 0; r < nranks; r++)
    per_rank.push_back(all.substr(displs[r], lens[r]));
#endif // MPI_PARALLEL

  // label -> time on each rank (zero where a rank never registered the label)
  const int nr = per_rank.size();
  std::map<std::string, std::vector<Real>> times;
  for (int r = 0; r < nr; r++) {
    std::istringstream is(per_rank[r]);
    std::string line;
    while (std::getline(is, line)) {
      const auto tab = line.rfind('\t');
      if (tab == std::string::npos) continue;
      auto &t = times[line.substr(0, tab)];
      t.resize(nr, 0.0);
      // strtod rather than stod, which throws on underflow (e.g. a timer printed as a
      // subnormal)
      t[r] = std::strtod(line.c_str() + tab + 1, nullptr);
    }
  }
  printf("self_gravity solver timings (s) over %i ranks, timed tasks only (boundary "
         "timers include polling for messages):\n",
         nr);
  printf("  %-48s %10s %10s %10s\n", "label", "min", "avg", "max");
  for (const auto &[label, t] : times) {
    Real tsum = 0.0;
    for (const Real v : t)
      tsum += v;
    printf("  %-48s %10.3e %10.3e %10.3e\n", label.c_str(),
           *std::min_element(t.begin(), t.end()), tsum / nr,
           *std::max_element(t.begin(), t.end()));
  }
}

} // namespace SelfGravity
