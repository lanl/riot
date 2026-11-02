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

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// Parthenon headers
#include <amr_criteria/refinement_package.hpp>
#include <globals.hpp>
#include <parthenon/package.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// Singularity headers
#include <singularity-eos/eos/eos.hpp>

// Riot headers
#include "hydro/hydro.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "microphysics/pte_closure.hpp"
#include "radiation_transport/algorithms/jacobi.hpp"
#include "radiation_transport/algorithms/radiation_bcs.hpp"
#include "radiation_transport/algorithms/radiation_init.hpp"
#include "radiation_transport/algorithms/radiation_shared.hpp"
#include "radiation_transport/angular_grids/angular_grid_utils.hpp"
#include "radiation_transport/angular_grids/geodesic_grid.hpp"
#include "radiation_transport/angular_grids/latlon_grid.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "riot_utils/sparse_update.hpp"

using namespace parthenon::package::prelude;

namespace Jacobi {
const std::string pkg_name = "jacobi";
const std::string input_block =
    std::string(RadiationShared::radiation_block) + "/" + pkg_name;

// Convergence test furthest from its target in the latest iteration (group -1 for the
// global test), with its residual and the contraction rate behind its error estimate
struct ConvergenceStatus {
  Real residual = std::numeric_limits<Real>::infinity();
  Real error = std::numeric_limits<Real>::infinity();
  Real target = 0.0;
  Real rate = std::numeric_limits<Real>::infinity();
  bool estimated = false;
  int group = -1;
};

// Red-black ordering of the interior cells: the red cells (i + j + k even) are updated
// first, and the black cells then see the updated red cells of their block
struct RedBlack {
  int color;
  IndexRange ib, jb, kb;

  RedBlack(MeshData<Real> *md, const int color_)
      : color(color_), ib(md->GetBoundsI(IndexDomain::interior)),
        jb(md->GetBoundsJ(IndexDomain::interior)),
        kb(md->GetBoundsK(IndexDomain::interior)) {}

  // Index space of the cells of this color, with ih spanning half of each row
  RiotFlatLoop::IndexSpace GetIndexSpace(const int nblocks, const int nlead = 0) const {
    return RiotFlatLoop::GetIndexSpace(nblocks, nlead, kb, jb,
                                       IndexRange{0, (ib.e - ib.s) / 2});
  }

  // Index i of the ih-th cell of this color in row (k, j), beyond ib.e if there is none
  KOKKOS_FORCEINLINE_FUNCTION int I(const int k, const int j, const int ih) const {
    return ib.s + 2 * ih + ((color + ib.s + j + k) & 1);
  }

  // Intensity at a face neighbor: updated for the interior neighbors of a black cell,
  // otherwise the previous iterate
  template <typename VI, typename VO>
  KOKKOS_FORCEINLINE_FUNCTION Real Neighbor(const VI &vi, const VO &vo, const int b,
                                            const int n, const int k, const int j,
                                            const int i) const {
    const bool updated = (color == 1) && (i >= ib.s) && (i <= ib.e) && (j >= jb.s) &&
                         (j <= jb.e) && (k >= kb.s) && (k <= kb.e);
    return updated ? vo(b, n, k, j, i) : vi(b, n, k, j, i);
  }
};

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::Initialize
//! \brief Initializes the Jacobi package
// NOTE(@pdmullen): The Jacobi TRT algorithm formulation here is largely adapted from
// Jiang (2021), ApJS, 253, 2, "An Implicit Finite Volume Scheme to Solve the
// Time-dependent Radiation Transport Equation Based on Discrete Ordinates", adapted to
// multiple material frameworks
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *materials) {
  auto jacobi_pkg = std::make_shared<StateDescriptor>(pkg_name);
  namespace ccrad = cell_variables::cell_averaged::rad;
  Params &params = jacobi_pkg->AllParams();

  // Algorithm-shared parameters (CFL, units, coupling, angular mesh, groups)
  const auto shared = RadiationShared::AddSharedParams(pin, materials, params);
  const int nangles = shared.nangles;
  const int ngroups = shared.ngroups;

  // Algorithm-shared Boundary Conditions (default, drive, reflect)
  RadiationBC::EnrollRadiationBC(jacobi_pkg.get(), pin, params);

  // Algorithm-shared Initializations (none, zero, thermal)
  RadiationInit::EnrollRadiationInit(jacobi_pkg.get(), pin, params);

  // User-Defined Metadata Flags
  auto MetadataJacobi = jacobi_pkg->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetUserFlag("OperatorSplit");

  // Jacobi iteration limit
  params.Add("niter_limit",
             pin->GetOrAddInteger(input_block, "niter_limit", 1000,
                                  "Maximum #iter permitted for Jacobi integration"));

  // Global residual error threshold
  const Real err_thr = pin->GetOrAddReal(
      input_block, "err_thr", 1.0e-8,
      "Convergence threshold of the Jacobi solve: the relative residual must fall below "
      "it or, if err_step_fraction > 0, it floors the target of the estimated error.");
  params.Add("err_thr", err_thr);

  // Optional additional per-group convergence check
  params.Add(
      "per_group_residual",
      pin->GetOrAddBoolean(input_block, "per_group_residual", false,
                           "Additionally require every group to pass the convergence "
                           "test on its own, with err_thr_group and, if "
                           "err_step_fraction > 0, its own contraction rate and step "
                           "change (so a low-energy but important group is not masked "
                           "by the global test)."));
  params.Add("err_thr_group",
             pin->GetOrAddReal(input_block, "err_thr_group", err_thr,
                               "Per-group counterpart of err_thr when per_group_residual "
                               "is true (defaults to err_thr)."));
  params.Add("per_group_residual_floor",
             pin->GetOrAddReal(input_block, "per_group_residual_floor", 1.0e-3,
                               "Floor a group's residual denominator at this fraction of "
                               "the total (all-group) denominator so a ~negligible group "
                               "cannot hold the solve hostage."));
  params.Add("err_step_fraction",
             pin->GetOrAddReal(
                 input_block, "err_step_fraction", 0.5 * do_angular_fluxes,
                 "If positive, converge when the estimated remaining error, "
                 "rho/(1-rho)*residual with rho the contraction rate of the residuals, "
                 "is below this fraction of the change in intensity over the time step, "
                 "or below err_thr (err_thr_group per group). If the residuals stop "
                 "contracting, or if set to 0, the residual itself is instead compared "
                 "against err_thr."));
  params.Add("extrapolate_guess",
             pin->GetOrAddBoolean(
                 input_block, "extrapolate_guess", false,
                 "Start each Jacobi solve from a linear extrapolation in time of the "
                 "intensity at the two previous steps rather than from the intensity at "
                 "the previous step."));

  // Minimum-iterations
  params.Add("niter_min",
             pin->GetOrAddInteger(
                 input_block, "niter_min", 0,
                 "Minimum #iter the Jacobi solver must take before it is permitted to "
                 "exit on convergence. Not needed if err_step_fraction > 0, as the "
                 "contraction rate behind the error estimate sets its own minimum."));
  const int nreduce_limit = pin->GetOrAddInteger(
      input_block, "nreduce_limit", 0,
      "Maximum number of timestep reductions permitted when the implicit Jacobi solve "
      "diverges.  On divergence the operator-split step is subcycled with dt reduced by "
      "reduce_factor and re-solved from the pristine base state.  Setting to 0 disables "
      "subcycling (a diverging solve aborts the run).");
  const int reduce_factor = pin->GetOrAddInteger(
      input_block, "reduce_factor", 2,
      "Integer factor by which the subcycle timestep is divided each time the implicit "
      "Jacobi solve diverges (e.g. 2 halves dt).");
  const int ndiverge_limit = pin->GetOrAddInteger(
      input_block, "ndiverge_limit", -1,
      "Number of consecutive Jacobi iterations without improvement on the best residual "
      "seen so far that constitutes divergence and triggers a dt reduction/subcycle.  "
      "Setting to -1 disables this detector.");
  const Real diverge_rise_frac = pin->GetOrAddReal(
      input_block, "diverge_rise_frac", -1.0,
      "Fractional rise of the residual above the best residual seen so far that "
      "constitutes divergence and triggers a dt reduction/subcycle (e.g. 0.1 flags a "
      "residual 10% above its best).  Setting to -1 disables this detector.");
  const int nrelax_steps = pin->GetOrAddInteger(
      input_block, "nrelax_steps", 0,
      "If > 0, a step starts with the subcycle count the previous step finished with, "
      "and that count is divided by reduce_factor after nrelax_steps consecutive steps "
      "without divergence.  Setting to 0 starts every step at the full dt.");
  PARTHENON_REQUIRE(nreduce_limit >= 0,
                    "radiation_transport/jacobi/nreduce_limit must be >= 0!");
  PARTHENON_REQUIRE(reduce_factor > 1,
                    "radiation_transport/jacobi/reduce_factor must be an integer > 1!");
  PARTHENON_REQUIRE(
      ndiverge_limit == -1 || ndiverge_limit > 0,
      "radiation_transport/jacobi/ndiverge_limit must be > 0, or -1 to disable");
  PARTHENON_REQUIRE(
      diverge_rise_frac == -1.0 || diverge_rise_frac > 0.0,
      "radiation_transport/jacobi/diverge_rise_frac must be > 0, or -1 to disable");
  PARTHENON_REQUIRE(nrelax_steps >= 0,
                    "radiation_transport/jacobi/nrelax_steps must be >= 0!");
  params.Add("nreduce_limit", nreduce_limit);
  params.Add("reduce_factor", reduce_factor);
  params.Add("ndiverge_limit", ndiverge_limit);
  params.Add("diverge_rise_frac", diverge_rise_frac);
  params.Add("nrelax_steps", nrelax_steps);

  // Timestep controllers
  params.Add("dt_ratio_hyperbolic",
             pin->GetOrAddReal(input_block, "dt_ratio_hyperbolic", 1.0e4),
             "Limit timestep of implicit update of thermal radiation transport by "
             "cfl * dt_ratio_hyperbolic * min_dx / c.  Setting to -1 does not permit "
             "this timestep controller to limit global timestep.");
  params.Add("split_g1", pin->GetOrAddBoolean(input_block, "split_g1", true,
                                              "Split Jacobi coefficient g1 = g1' + g1'' "
                                              "where g1' and g1'' are positive and "
                                              "negative contributions, respectively, for "
                                              "potentially more robust integration"));
  params.Add("verbose",
             pin->GetOrAddInteger(
                 input_block, "verbose", 0,
                 "Sets verbosity of implicit (Jacobi) thermal radiation transport "
                 "package.  0: Report no diagnostics, 1: Only report at final Jacobi "
                 "iteration, 2: Report every Jacobi iteration, 3: Report every Jacobi "
                 "iteration and temperature root find failures"));

  // Reducer for residual
  parthenon::AllReduce<HostArray1D<Real>> residual_reducer;
  const int nres = 3 * (ngroups + 1);
  residual_reducer.val = HostArray1D<Real>("Jacobi_pkg Residual Reducer", nres);
  for (int i = 0; i < residual_reducer.val.size(); ++i)
    residual_reducer.val(i) = 0.0;
  params.Add("jresidual_reducer", residual_reducer, true);

  // Solver mutable params
  params.Add("current_residual", std::numeric_limits<Real>::max(), true);
  params.Add("current_iter", std::numeric_limits<int>::max(), true);
  params.Add("solve_diverged", false, true);
  params.Add("best_residual", std::numeric_limits<Real>::max(), true);
  params.Add("residual_history", std::vector<Real>(4 * (ngroups + 1), 0.0), true);
  params.Add("convergence_status", ConvergenceStatus(), true);
  params.Add("residual_met", false, true);
  params.Add("nstall", 0, true);
  params.Add("ndiv_start", 1, Params::Mutability::Restart);
  params.Add("nclean", 0, Params::Mutability::Restart);
  params.Add("time", 0.0, Params::Mutability::Restart);
  params.Add("dt_previous", 0.0, true);

  // Radiation specific intensity I
  std::string control_field = ccrad::intensity::name();
  using namespace parthenon::refinement_ops;
  Metadata mi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                          Metadata::Intensive, Metadata::Conserved, MetadataJacobi,
                          MetadataOperatorSplit},
                         std::vector<int>({ngroups * nangles}));
  mi.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  jacobi_pkg->AddField<ccrad::intensity>(mi);

  // Jacobi iterate, which holds the previous intensity between solves and is carried
  // through remeshing for the extrapolated initial guess
  Metadata mr = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataOperatorSplit},
      std::vector<int>({ngroups * nangles}));
  if (params.Get<bool>("extrapolate_guess")) {
    mr.Set(Metadata::ForceRemeshComm);
    mr.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  }
  jacobi_pkg->AddField<ccrad::intensity_iterate>(mr);

  // Opacity fields
  Metadata mo = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                          Metadata::FillGhost, MetadataJacobi, MetadataOperatorSplit},
                         std::vector<int>({ngroups}));
  mo.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  jacobi_pkg->AddField<ccrad::aa>(mo);
  jacobi_pkg->AddField<ccrad::ss>(mo);

  // Moments and auxillary scratch fields
  Metadata m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                         MetadataJacobi, MetadataOperatorSplit},
                        std::vector<int>({ngroups}));
  jacobi_pkg->AddField<ccrad::moments>(m);
  jacobi_pkg->AddField<ccrad::s1>(m);
  jacobi_pkg->AddField<ccrad::s2>(m);
  jacobi_pkg->AddField<ccrad::s3>(m);

  // Advanced temperature field
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataJacobi,
                MetadataOperatorSplit});
  jacobi_pkg->AddField<ccrad::temperature>(m);

  // Jacobi Timestep
  jacobi_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Set Moments
  jacobi_pkg->UserWorkBeforeOutputMesh = SetMomentsMesh;

  return jacobi_pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::EstimateTimestepMesh
//! \brief Computes Jacobi timestep
Real EstimateTimestepMesh(MeshData<Real> *md) {
  const Real max_dt = 1.e-3 * std::numeric_limits<Real>::max();
  if (md->NumBlocks() == 0) return max_dt;

  auto pm = md->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  const Real dt_ratio_hyperbolic = jacobi_pkg->Param<Real>("dt_ratio_hyperbolic");

  // If dt_ratio_hyperbolic not active, radiation does not vote a timestep.
  if (dt_ratio_hyperbolic < 0.0) return max_dt;

  // Resolved packages and indexing
  auto resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const bool multid = ndim > 1;
  const bool threed = ndim > 2;

  // Packing and indexing
  using parthenon::MakePackDescriptor;
  const std::vector<std::string> empty;
  static auto desc = MakePackDescriptor(resolved_pkgs.get(), empty);
  auto vmesh = desc.GetPack(md);

  // Units
  const auto unit_utils = jacobi_pkg->Param<UnitUtils>("unit_utils");
  const Real cc = unit_utils.c;

  // Spatial minimum timestep
  Real min_dx = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "Jacobi::EstimateSpatialTimestep",
      Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, md->NumBlocks()),
      KOKKOS_LAMBDA(const int b, Real &ldx) {
        auto &coords = vmesh.GetCoordinates(b);
        ldx = std::min(ldx, 1.0 / (1.0 / coords.CellWidth(X1DIR, 0, 0, 0) +
                                   multid * 1.0 / coords.CellWidth(X2DIR, 0, 0, 0) +
                                   threed * 1.0 / coords.CellWidth(X3DIR, 0, 0, 0)));
      },
      Kokkos::Min<Real>(min_dx));
  const Real min_sdt = min_dx / cc;

  // Angular fluxes
  Real min_adt = std::numeric_limits<Real>::max();
  if (do_angular_fluxes) {
    // Extraction of angular grid quantities
    const auto agrid = GetAngularGridArrays(jacobi_pkg);
    const auto &cp = agrid.cart_pos_unit;
    const auto &gflx = agrid.gflux;
    const auto &numn = agrid.num_neighbors;
    const auto &indn = agrid.ind_neighbors;

    // Indexing
    const int nangles = jacobi_pkg->Param<int>("nangles");

    // Angular flux indexing
    const auto NDIR = AngularFluxDirs();
    using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto idx_space =
        rt::GetIndexSpace(IndexDomain::interior, vmesh.GetNBlocks(), nangles, md);
    min_adt = rt::five_d(
        "Jacobi::AngularTimestep", idx_space,
        KOKKOS_LAMBDA(const int b, const int aa, const int k, const int j, const int i,
                      Real &ldt) {
          auto &coords = vmesh.GetCoordinates(b);
          const Real mcw_ix1 = -cc * InverseRadiusForAngularFlux(coords, i);
          const Real &n1 = cp(aa, NDIR[X1DIR - 1]);
          const Real &n2 = cp(aa, NDIR[X2DIR - 1]);
          const Real &n3 = cp(aa, NDIR[X3DIR - 1]);

          // Angular time constraint
          for (int nb = 0; nb < numn(aa); ++nb) {
            const Real &dn1 = cp(indn(aa, nb), NDIR[X1DIR - 1]);
            const Real &dn2 = cp(indn(aa, nb), NDIR[X2DIR - 1]);
            const Real &dn3 = cp(indn(aa, nb), NDIR[X3DIR - 1]);
            const Real absna = std::abs(mcw_ix1 * gflx(aa, nb));
            const Real dang = std::acos(n1 * dn1 + n2 * dn2 + n3 * dn3);
            ldt = (absna > 0.0) ? std::min(ldt, dang / absna) : ldt;
          }
        });
  }

  // Set timestep
  const Real cfl = jacobi_pkg->Param<Real>("cfl");
  return cfl * std::min(min_sdt, min_adt) * dt_ratio_hyperbolic;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection JacobiTasks
//! \brief Allocate MeshData registers and drive the (subcycled) Jacobi update
TaskCollection JacobiTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  // Extract Jacobi package
  auto jacobi_pkg = pm->packages.Get(pkg_name);

  // Get variable names for MeshData subsets
  std::vector<std::string> jacobi_names =
      pm->GetVariableNames(jacobi_pkg->GetMetadataFlag());

  // Filter MeshData subset registers by name
  std::vector<std::string> unsplit_names;
  if (jacobi_pkg->Param<bool>("coupling")) {
    parthenon::Metadata::FlagCollection flags_unsplit;
    flags_unsplit.Exclude(parthenon::Metadata::GetUserFlag("OperatorSplit"));
    unsplit_names = pm->GetVariableNames(flags_unsplit);
  }

  // Assemble MeshData subsets once; they are reused across any subcycles this step
  namespace mdname = container_names;
  namespace ccrad = cell_variables::cell_averaged::rad;
  std::vector<std::string> intensity_names = pm->GetVariableNames(
      std::vector<std::string>{ccrad::intensity::name()}, std::vector<int>{});
  std::vector<std::string> iterate_names = pm->GetVariableNames(
      std::vector<std::string>{ccrad::intensity_iterate::name()}, std::vector<int>{});
  auto &base = pm->mesh_data.Get();
  pm->mesh_data.AddShallow(mdname::riter, base, iterate_names);
  pm->mesh_data.Add(mdname::rout, base, intensity_names);
  pm->mesh_data.AddShallow(mdname::rbase, base, jacobi_names);
  pm->mesh_data.AddShallow(mdname::ubase, base, unsplit_names);

  // Subcycle controls
  const int nreduce_limit = jacobi_pkg->Param<int>("nreduce_limit");
  const int reduce_factor = jacobi_pkg->Param<int>("reduce_factor");
  const int nrelax_steps = jacobi_pkg->Param<int>("nrelax_steps");
  const int verbose = jacobi_pkg->Param<int>("verbose");

  const bool extrapolate_guess = jacobi_pkg->Param<bool>("extrapolate_guess");

  // Adaptive subcycle loop
  const int ndiv_start = (nrelax_steps > 0) ? jacobi_pkg->Param<int>("ndiv_start") : 1;
  int ndiv = ndiv_start;
  int nremaining = ndiv_start;
  int nreduce = 0;
  int nsub = 0;
  while (nremaining > 0) {
    const Real dt_this = dt / ndiv;
    const Real t_local = dt * (ndiv - nremaining) / ndiv;

    // Report subcycle progress
    if (ndiv > 1 && Globals::my_rank == 0 && verbose >= 1) {
      printf(
          "(Jacobi) Subcycle attempt %d | dt_sub: %24.16e | committed: %6.2f%% of dt\n",
          nsub + 1, dt_this, 100.0 * t_local / dt);
    }

    // Initialize subcycler params
    jacobi_pkg->UpdateParam("solve_diverged", false);
    jacobi_pkg->UpdateParam("nstall", 0);
    jacobi_pkg->UpdateParam("best_residual", std::numeric_limits<Real>::max());

    // Solve from the current base state, extrapolated in time if possible
    const Real dt_previous = jacobi_pkg->Param<Real>("dt_previous");
    const Real dt_ratio = (dt_previous > 0.0) ? dt_this / dt_previous : 0.0;
    jacobi_pkg->UpdateParam("dt_previous", 0.0);
    const auto solve_status =
        JacobiSolve(pm, tm.time + t_local, dt_this, dt_ratio).Execute();
    PARTHENON_REQUIRE(solve_status == TaskListStatus::complete,
                      "Jacobi solve task list did not complete.");

    if (!jacobi_pkg->Param<bool>("solve_diverged")) {
      // Solve did not diverge (converged, or hit niter_limit without diverging): commit.
      // Advance the intensity and (if coupled) the fluid.
      const auto commit_status = JacobiCommit(pm, dt_this).Execute();
      PARTHENON_REQUIRE(commit_status == TaskListStatus::complete,
                        "Jacobi commit task list did not complete.");
      jacobi_pkg->UpdateParam("dt_previous", extrapolate_guess ? dt_this : 0.0);
      --nremaining;
      ++nsub;
    } else {
      // Solve diverged: back out and retry the remaining span with a reduced timestep
      ++nreduce;
      PARTHENON_REQUIRE(nreduce <= nreduce_limit,
                        "Implicit Jacobi radiation solve diverged after " +
                            std::to_string(nreduce_limit) +
                            " timestep reduction(s).  Residual: " +
                            std::to_string(jacobi_pkg->Param<Real>("current_residual")) +
                            ", dt_sub: " + std::to_string(dt_this) +
                            ".  Raise nreduce_limit, loosen "
                            "ndiverge_limit/diverge_rise_frac, or check the setup.");
      ndiv *= reduce_factor;
      nremaining *= reduce_factor;
      if (Globals::my_rank == 0 && verbose >= 1)
        printf("(Jacobi) Solve diverged; reducing dt (reduction %d/%d), "
               "new dt_sub: %24.16e\n",
               nreduce, nreduce_limit, dt / ndiv);
    }
  }

  if (ndiv > 1 && Globals::my_rank == 0 && verbose >= 1) {
    printf("(Jacobi) Completed operator-split step in %d subcycles (%d reduction(s))\n",
           nsub, nreduce);
  }

  // Carry the subcycle count into the next step, relaxing it after a clean streak
  if (nrelax_steps > 0) {
    int ndiv_next = 1;
    for (int r = 0; r < nreduce_limit && ndiv_next < ndiv; ++r)
      ndiv_next *= reduce_factor;
    ndiv_next = std::min(ndiv_next, ndiv);
    int nclean = (nreduce > 0) ? 0 : jacobi_pkg->Param<int>("nclean") + 1;
    if (ndiv_next > 1 && nclean >= nrelax_steps) {
      ndiv_next /= reduce_factor;
      nclean = 0;
    }
    jacobi_pkg->UpdateParam("ndiv_start", std::max(ndiv_next, 1));
    jacobi_pkg->UpdateParam("nclean", nclean);
  }

  return TaskCollection();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection JacobiSolve
//! \brief
TaskCollection JacobiSolve(Mesh *pmesh, const Real time, const Real dt,
                           const Real dt_ratio) {
  using namespace ::parthenon::Update;
  namespace mdname = container_names;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  // Jacobi params
  auto jacobi_pkg = pmesh->packages.Get(pkg_name);
  const int max_iters = jacobi_pkg->Param<int>("niter_limit");
  AllReduce<HostArray1D<Real>> *presidual =
      jacobi_pkg->MutableParam<AllReduce<HostArray1D<Real>>>("jresidual_reducer");

  // Construct JacobiSolve TaskList
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &rbase = pmesh->mesh_data.GetOrAdd(mdname::rbase, i);
    auto &riter = pmesh->mesh_data.GetOrAdd(mdname::riter, i);
    auto &rout = pmesh->mesh_data.GetOrAdd(mdname::rout, i);
    auto &ubase = pmesh->mesh_data.GetOrAdd(mdname::ubase, i);
    auto &tl = tr[i];

    // Start receives on MeshData registers
    auto srec = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, rbase);
    srec = tl.AddTask(srec, parthenon::StartReceiveBoundBufs<any>, rout);

    // Set opacities
    auto set_opac = tl.AddTask(none, SetOpacities, rbase.get(), ubase.get());

    // Reset iterator
    auto reset_iter =
        tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync, set_opac,
                   ResetIterator, rbase.get(), time, dt);

    // Fill ghosts with boundary conditions at the advanced time
    auto base_bc =
        parthenon::AddBoundaryExchangeTasks(reset_iter, tl, rbase, pmesh->multilevel);

    // Initialize solution in riter
    auto init_iter =
        (dt_ratio > 0.0)
            ? tl.AddTask(TaskQualifier::local_sync, base_bc, ExtrapolateIterate,
                         rbase.get(), riter.get(), dt_ratio)
            : tl.AddTask(TaskQualifier::local_sync, base_bc, CopyToIterate, rbase.get(),
                         riter.get());

    // Execute Jacobi iterations
    auto [solver, solver_id] = tl.AddSublist(init_iter, {1, max_iters});
    CreateJacobiTaskList(none, i, pmesh, solver, solver_id, presidual, rbase, riter, rout,
                         ubase, dt);
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection JacobiCommit
//! \brief
TaskCollection JacobiCommit(Mesh *pmesh, const Real dt) {
  using namespace ::parthenon::Update;
  namespace mdname = container_names;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  // Jacobi params
  auto jacobi_pkg = pmesh->packages.Get(pkg_name);
  const bool affect_fluid = jacobi_pkg->Param<bool>("affect_fluid");
  const bool extrapolate_guess = jacobi_pkg->Param<bool>("extrapolate_guess");

  // Construct JacobiCommit TaskList
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &rbase = pmesh->mesh_data.GetOrAdd(mdname::rbase, i);
    auto &riter = pmesh->mesh_data.GetOrAdd(mdname::riter, i);
    auto &rout = pmesh->mesh_data.GetOrAdd(mdname::rout, i);
    auto &ubase = pmesh->mesh_data.GetOrAdd(mdname::ubase, i);
    auto &tl = tr[i];

    // Start receives for the fluid boundary exchange (if radiation affects fluid)
    TaskID srec = none;
    if (affect_fluid) {
      srec = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, ubase);
    }

    // Apply feedback to fluid (if radiation affects fluid)
    auto feedback = tl.AddTask(srec, JacobiFeedback, rbase.get(), riter.get(), rout.get(),
                               ubase.get(), dt);

    // Keep the previous intensity in riter for the next solve's initial guess
    auto save_prev = feedback;
    if (extrapolate_guess) {
      save_prev = tl.AddTask(feedback, CopyToIterate, rbase.get(), riter.get());
    }

    // Update intensity registers
    auto copy_reg = tl.AddTask(
        save_prev, sparse_update::DeepCopyData<parthenon::MetadataFlag, MeshData<Real>>,
        std::vector<MetadataFlag>({Metadata::Independent}), rbase.get(), rout.get());

    // Update fluid state if radiation affects fluid
    if (affect_fluid) {
      auto pte = tl.AddTask(
          feedback,
          [](MeshData<Real> *md) {
            const bool use_general_pte = md->GetParentPointer()
                                             ->packages.Get("materials")
                                             ->Param<bool>("use_general_pte");
            if (use_general_pte) {
              Closure::ApplyMixedCellClosure(md, IndexDomain::interior);
            } else {
              Closure::ApplyIdealGasClosure(md, IndexDomain::interior);
            }
            return TaskStatus::complete;
          },
          ubase.get());
      auto u_bc = parthenon::AddBoundaryExchangeTasks(pte, tl, ubase, pmesh->multilevel);
      auto derive = tl.AddTask(u_bc, FillDerived<MeshData<Real>>, ubase.get());
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskID CreateJacobiTaskList
//! \brief
TaskID CreateJacobiTaskList(const TaskID &begin, const int i, Mesh *pmesh,
                            TaskList &solver, TaskID solver_id,
                            parthenon::AllReduce<HostArray1D<Real>> *presidual,
                            std::shared_ptr<MeshData<Real>> rbase,
                            std::shared_ptr<MeshData<Real>> riter,
                            std::shared_ptr<MeshData<Real>> rout,
                            std::shared_ptr<MeshData<Real>> ubase, const Real dt) {
  // Update the red cells, then the black cells
  TaskID rsrc = begin;
  for (int color = 0; color < 2; ++color) {
    auto update = solver.AddTask(rsrc, JacobiUpdate, rbase.get(), riter.get(), rout.get(),
                                 dt, color);
    rsrc =
        solver.AddTask(update, ApplyRHS, rbase.get(), rout.get(), ubase.get(), dt, color);
  }
  auto bc = parthenon::AddBoundaryExchangeTasks(rsrc, solver, rout, pmesh->multilevel);
  auto check = solver.AddTask(TaskQualifier::local_sync, rsrc, CheckConvergence,
                              &(presidual->val), rbase.get(), riter.get(), rout.get());

  // The boundary exchange completes ahead of the regional reduction, so that no list can
  // iterate (and so reset its tasks) while another is still exchanging
  auto start_reduce_residual =
      solver.AddTask(TaskQualifier::once_per_region, check | bc,
                     &AllReduce<HostArray1D<Real>>::StartReduce, presidual, MPI_SUM);
  auto finish_reduce_residual = solver.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, start_reduce_residual,
      &AllReduce<HostArray1D<Real>>::CheckReduce, presidual);
  auto pre_compl = solver.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, finish_reduce_residual,
      IncrementCounterAndSetResidual, &(presidual->val), rout->GetMeshPointer());
  auto complete = solver.AddTask(TaskQualifier::completion, pre_compl, CompletionFunction,
                                 i, &(presidual->val), riter.get(), rout.get());
  return solver_id;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus CheckConvergence
//! \brief Check for convergence in iterative solution
TaskStatus CheckConvergence(HostArray1D<Real> *presidual, MeshData<Real> *rbase,
                            MeshData<Real> *riter, MeshData<Real> *rout) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = rout->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  auto &resolved_pkgs = pm->resolved_packages;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  const int nangles = jacobi_pkg->Param<int>("nangles");

  // Packing
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc_i = MakePackDescriptor<ccrad::intensity_iterate>(resolved_pkgs.get());
  static auto desc_o = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  static auto desc_b = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  auto vi = desc_i.GetPack(riter);
  auto vo = desc_o.GetPack(rout);
  auto vb = desc_b.GetPack(rbase);

  // Residual reductions
  // NOTE(@pdmullen): When moving to the (optional) per group residual model, we must
  // track each individual group's ||\Delta I_g|| and ||I_g||.  We below opt for a for
  // loop that launches ngroup reductions (beware kernel launch latency).  Alternative
  // strategies might invoke:
  //
  //   - Hierarchical reductions (outer: blocks * ngroups)... enough threadblocks?
  //   - Single kernel with atomics?
  //   - Single reduction to a compile-time-sized ~2*MAX_GROUPS array.  Register pressure?
  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, 3>>;
  auto &residual = *presidual;
  for (int gg = 0; gg < ngroups; ++gg) {
    auto gspace =
        rt::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(), nangles, rout);
    const auto gres = rt::five_d(
        "JacobiResidualGroup", gspace,
        KOKKOS_LAMBDA(const int b, const int aa, const int k, const int j, const int i,
                      RiotUtils::array_type<Real, 3> &rsum) {
          const int n = GAI(nangles, gg, aa);
          rsum.my_array[0] += std::abs(vi(b, n, k, j, i) - vo(b, n, k, j, i));
          rsum.my_array[1] += vo(b, n, k, j, i);
          rsum.my_array[2] += std::abs(vb(b, n, k, j, i) - vo(b, n, k, j, i));
        });
    Kokkos::fence();
    for (int r = 0; r < 3; ++r) {
      residual(r) += gres.my_array[r];
      residual(3 * (gg + 1) + r) += gres.my_array[r];
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  Real ContractionRate
//! \brief Contraction rate of the residuals at iteration m, from the residual and the
//! four before it (most recent first). The rate is taken over two iterations, as Jacobi
//! pairs eigenvalues of opposite sign, and while it is still rising it is replaced by
//! its mean over the remaining iterations. Infinite until it is measurable.
static Real ContractionRate(const int m, const Real residual, const Real *history) {
  if (m < 5) return std::numeric_limits<Real>::infinity();
  const Real rho = std::sqrt(residual / std::max(history[1], 1.0e-100));
  const Real rho_prev = std::sqrt(history[1] / std::max(history[3], 1.0e-100));
  return (rho < 1.0) ? rho + 0.25 * std::max(rho - rho_prev, 0.0) / (1.0 - rho) : rho;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus IncrementCounterAndSetResidual
//! \brief Increment the iteration counter and stow/reset globally reduced residual
TaskStatus IncrementCounterAndSetResidual(HostArray1D<Real> *presidual, Mesh *pmesh) {
  // increment counter
  auto jacobi_pkg = pmesh->packages.Get(pkg_name);
  const int iter_counter = jacobi_pkg->Param<int>("current_iter");
  jacobi_pkg->UpdateParam("current_iter", iter_counter + 1);

  // update current residual
  auto &v = *presidual;
  const Real denom = std::max(v(1), 1.0e-100) + (v(1) <= 1.0e-100);
  const Real current_residual = v(0) / denom;
  jacobi_pkg->UpdateParam("current_residual", current_residual);

  // Convergence tests of an error against its target: under the step rule, the
  // estimated remaining error against a fraction of the step change, floored at the
  // threshold, otherwise (or once a residual is no smaller than that two iterations
  // earlier) the residual against the threshold. Slot 0 holds the residual history of
  // the global test, slot g + 1 that of group g.
  const Real step_fraction = jacobi_pkg->Param<Real>("err_step_fraction");
  const bool step_rule = (step_fraction > 0.0);
  auto &history = *jacobi_pkg->MutableParam<std::vector<Real>>("residual_history");
  bool residual_met = true;
  ConvergenceStatus status;
  Real worst = -std::numeric_limits<Real>::infinity();
  auto test = [&](const int slot, const Real residual, const Real step,
                  const Real threshold) {
    Real *h = &history[4 * slot];
    const Real rate = ContractionRate(iter_counter, residual, h);
    const bool contracting = (iter_counter < 5) || (residual < h[1]);
    for (int k = 3; k > 0; --k)
      h[k] = h[k - 1];
    h[0] = residual;
    const bool estimated = step_rule && contracting;
    const Real error = !estimated   ? residual
                       : rate < 1.0 ? rate / (1.0 - rate) * residual
                                    : std::numeric_limits<Real>::infinity();
    const Real target = estimated ? std::max(step_fraction * step, threshold) : threshold;
    residual_met = residual_met && (error <= target);
    const Real ratio = error / std::max(target, std::numeric_limits<Real>::min());
    if (!(ratio <= worst)) {
      worst = ratio;
      status = {residual, error, target, rate, estimated, slot - 1};
    }
  };
  test(0, current_residual, v(2) / denom, jacobi_pkg->Param<Real>("err_thr"));

  // Optional per-group tests, which only duplicate the global test for a single group
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  if (jacobi_pkg->Param<bool>("per_group_residual") && ngroups > 1) {
    const Real floor_frac = jacobi_pkg->Param<Real>("per_group_residual_floor");
    const Real denom_floor = floor_frac * std::max(v(1), 1.0e-100);
    const Real err_thr_group = jacobi_pkg->Param<Real>("err_thr_group");
    for (int gg = 0; gg < ngroups; ++gg) {
      const Real denom_g = std::max(v(3 * (gg + 1) + 1), std::max(denom_floor, 1.0e-100));
      const Real scale_g = 1.0 / (denom_g + (denom_g <= 1.0e-100));
      test(gg + 1, v(3 * (gg + 1)) * scale_g, v(3 * (gg + 1) + 2) * scale_g,
           err_thr_group);
    }
  }
  jacobi_pkg->UpdateParam("convergence_status", status);
  jacobi_pkg->UpdateParam("residual_met", residual_met);

  // Divergence detection
  const Real best_residual = jacobi_pkg->Param<Real>("best_residual");
  const int ndiverge_limit = jacobi_pkg->Param<int>("ndiverge_limit");
  const Real diverge_rise_frac = jacobi_pkg->Param<Real>("diverge_rise_frac");
  int nstall = jacobi_pkg->Param<int>("nstall");
  const bool improved = current_residual < best_residual;
  nstall = improved ? 0 : nstall + 1;
  const bool stalled =
      !residual_met && (ndiverge_limit >= 0) && (nstall >= ndiverge_limit);
  const bool rose =
      !residual_met && (diverge_rise_frac > 0.0) &&
      (current_residual - best_residual > diverge_rise_frac * best_residual);
  const bool diverged = !std::isfinite(current_residual) || stalled || rose;
  jacobi_pkg->UpdateParam("nstall", nstall);
  if (improved) jacobi_pkg->UpdateParam("best_residual", current_residual);
  if (diverged) jacobi_pkg->UpdateParam("solve_diverged", true);

  // zero residuals in prep for next iteration
  for (int r = 0; r < v.size(); ++r) {
    v(r) = 0.0;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus CompletionFunction
//! \brief Complete iteration, check for continued iterations or Jacobi finalize
TaskStatus CompletionFunction(int i, HostArray1D<Real> *presidual, MeshData<Real> *riter,
                              MeshData<Real> *rout) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = rout->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  const int iter_counter = jacobi_pkg->Param<int>("current_iter");
  const int niter_limit = jacobi_pkg->Param<int>("niter_limit");
  const int niter_min = jacobi_pkg->Param<int>("niter_min");
  const int verbose = jacobi_pkg->Param<int>("verbose");
  const bool report = (i == 0 && Globals::my_rank == 0);
  const int niter_done = iter_counter - 1;

  // Report the test furthest from convergence: its residual, the error it tests
  // (estimated, or the residual) against its target and the contraction rate behind the
  // estimate under the step rule, and its group unless it is the global test
  const auto &status = jacobi_pkg->Param<ConvergenceStatus>("convergence_status");
  const bool step_rule = (jacobi_pkg->Param<Real>("err_step_fraction") > 0.0);
  auto print = [&](const char *outcome, const bool note_niter_min) {
    char residual[16] = "--", error[16] = "--", rate[16] = "--", line[160];
    if (std::isfinite(status.residual))
      snprintf(residual, sizeof(residual), "%.3e", status.residual);
    if (std::isfinite(status.error)) snprintf(error, sizeof(error), "%.3e", status.error);
    if (std::isfinite(status.rate)) snprintf(rate, sizeof(rate), "%.3f", status.rate);
    int len = snprintf(line, sizeof(line), "(Jacobi) iter: %4d | res %9s", niter_done,
                       residual);
    if (step_rule) {
      len += snprintf(line + len, sizeof(line) - len, " | %s %9s / tgt %9.3e | rate %5s",
                      status.estimated ? "est" : "res", error, status.target, rate);
    } else {
      len += snprintf(line + len, sizeof(line) - len, " / tgt %9.3e", status.target);
    }
    if (status.group >= 0)
      len += snprintf(line + len, sizeof(line) - len, " | group %3d", status.group);
    if (note_niter_min)
      len += snprintf(line + len, sizeof(line) - len, " | niter_min %d", niter_min);
    printf("%s\n", line);
    if (outcome[0] != '\0') printf("(Jacobi) %s\n", outcome);
  };

  // Iterate or finalize
  const bool residual_met = jacobi_pkg->Param<bool>("residual_met");
  if (jacobi_pkg->Param<bool>("solve_diverged")) {
    if (report && verbose >= 1) print("Diverged!", false);
    return TaskStatus::complete;
  } else if (residual_met && niter_done >= niter_min) {
    if (report && verbose >= 1) print("Converged!", false);
    return TaskStatus::complete;
  } else if (niter_done >= niter_limit) {
    if (report && verbose >= 1) print("Hit limit!", false);
    return TaskStatus::complete;
  } else {
    if (report && verbose >= 2) print("", residual_met);
    CopyToIterate(rout, riter);
    return TaskStatus::iterate;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::SetOpacities
//! \brief Reset iterators and set matrix coefficients
TaskStatus SetOpacities(MeshData<Real> *rbase, MeshData<Real> *ubase) {
  // Return if no blocks in MeshData
  if (rbase->NumBlocks() == 0) return TaskStatus::complete;

  // Extract Jacobi package
  using parthenon::MakePackDescriptor;
  auto pm = rbase->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);

  // Indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");

  // Materials
  const bool coupling = jacobi_pkg->Param<bool>("coupling");
  const bool fixed_pgen_opac = jacobi_pkg->Param<bool>("fixed_pgen_opac");
  const Real mix_frac = jacobi_pkg->Param<Real>("mix_frac");
  const Real opac_rho_min = jacobi_pkg->Param<Real>("opac_rho_min");
  const Real opac_temp_min = jacobi_pkg->Param<Real>("opac_temp_min");
  const Real opac_rho_max = jacobi_pkg->Param<Real>("opac_rho_max");
  const Real opac_temp_max = jacobi_pkg->Param<Real>("opac_temp_max");
  ParArray1D<RiotOpacity::MeanOpacA> opac_a;
  ParArray1D<RiotOpacity::MeanOpacS> opac_s;
  ParArray1D<int> opac_from_matid;
  if (coupling) {
    auto &mat_pkg = pm->packages.Get("materials");
    opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
    opac_s = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacS>>("d.d.opac_s");
    opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");
  }

  // Packing
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  namespace ccr = cell_variables::cell_averaged::rad;
  static auto desc_u = MakePackDescriptor<cm::rho, ccmat::rho, ccmat::volume_fraction,
                                          ccbulk::temperature>(resolved_pkgs.get());
  static auto desc_r =
      MakePackDescriptor<ccr::aa, ccr::ss, ccr::temperature>(resolved_pkgs.get());
  auto vu = desc_u.GetPack(ubase);
  auto vr = desc_r.GetPack(rbase);

  if (coupling) {
    auto idx_space =
        RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vr.GetNBlocks(), rbase);
    RiotFlatLoop::four_d(
        "Coupling", idx_space,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          vr(b, ccr::temperature(), k, j, i) = vu(b, ccbulk::temperature(), k, j, i);
          if (!(fixed_pgen_opac)) {
            const Real temp = std::clamp(vu(b, ccbulk::temperature(), k, j, i),
                                         opac_temp_min, opac_temp_max);
            for (int gg = 0; gg < ngroups; ++gg) {
              Real &aa = vr(b, ccr::aa(gg), k, j, i) = 0.0;
              Real &ss = vr(b, ccr::ss(gg), k, j, i) = 0.0;
              for (int m = 0; m < vu.GetSize(b, cm::rho()); ++m) {
                const Real &rhom = vu(b, cm::rho(m), k, j, i);
                const Real &rhobarm = vu(b, ccmat::rho(m), k, j, i);
                const Real &vfracm = vu(b, ccmat::volume_fraction(m), k, j, i);
                const int &mat_id = vu(b, ccmat::rho(m)).sparse_id;
                const int &phase_id = vu(b, ccmat::rho(m)).v;
                const int opac_id = opac_from_matid(mat_id) + phase_id;
                const auto &oam = opac_a(opac_id);
                const auto &osm = opac_s(opac_id);
                const Real rm = std::clamp(rhom, opac_rho_min, opac_rho_max);
                const Real rbarm = std::clamp(rhobarm, opac_rho_min, opac_rho_max);
                aa += (rhobarm > 0.0)
                          ? MMOpacity(oam, rm, rbarm, vfracm, temp, gg, mix_frac)
                          : 0.0;
                ss += (rhobarm > 0.0)
                          ? MMOpacity(osm, rm, rbarm, vfracm, temp, gg, mix_frac)
                          : 0.0;
              }
            }
          }
        });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::ResetIterator
//! \brief Reset iterators
TaskStatus ResetIterator(MeshData<Real> *rbase, const Real time, const Real dt) {
  // Extract Jacobi package
  auto jacobi_pkg = rbase->GetParentPointer()->packages.Get(pkg_name);

  // Reset iterator and set time in preparation for iterations
  jacobi_pkg->UpdateParam("current_iter", 1);
  jacobi_pkg->UpdateParam("time", time + dt);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::ExtrapolateIterate
//! \brief Set riter to the intensity in rbase extrapolated linearly in time using the
//! previous intensity held in riter, with ghosts copied from rbase
TaskStatus ExtrapolateIterate(MeshData<Real> *rbase, MeshData<Real> *riter,
                              const Real dt_ratio) {
  if (riter->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = riter->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Packing
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc_b = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccrad::intensity_iterate>(resolved_pkgs.get());
  auto vb = desc_b.GetPack(rbase);
  auto vi = desc_i.GetPack(riter);

  const auto ib = riter->GetBoundsI(IndexDomain::interior);
  const auto jb = riter->GetBoundsJ(IndexDomain::interior);
  const auto kb = riter->GetBoundsK(IndexDomain::interior);
  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, vi.GetNBlocks(),
                                               vi.GetMaxNumberOfVars(), riter);
  RiotFlatLoop::five_d(
      "ExtrapolateIterate", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &n, const int &k, const int &j,
                    const int &i) {
        const bool interior =
            (i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s && k <= kb.e);
        const Real &in = vb(b, n, k, j, i);
        Real &ip = vi(b, n, k, j, i);
        ip = interior ? std::max(0.0, in + dt_ratio * (in - ip)) : in;
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::CopyToIterate
//! \brief Copy the intensity in src, ghosts included, into riter
TaskStatus CopyToIterate(MeshData<Real> *src, MeshData<Real> *riter) {
  if (riter->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = riter->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Packing
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc_s = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccrad::intensity_iterate>(resolved_pkgs.get());
  auto vs = desc_s.GetPack(src);
  auto vi = desc_i.GetPack(riter);

  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, vi.GetNBlocks(),
                                               vi.GetMaxNumberOfVars(), riter);
  RiotFlatLoop::five_d(
      "CopyToIterate", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &n, const int &k, const int &j,
                    const int &i) { vi(b, n, k, j, i) = vs(b, n, k, j, i); });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus JacobiUpdate
//! \brief Compute updated solution in Jacobi iteration for the cells of one color
TaskStatus JacobiUpdate(MeshData<Real> *rbase, MeshData<Real> *riter,
                        MeshData<Real> *rout, const Real dt, const int color) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;

  // Extract Jacobi package
  auto pm = rout->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  const bool split_g1 = jacobi_pkg->Param<bool>("split_g1");
  const bool coupling = jacobi_pkg->Param<bool>("coupling");

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const bool multid = ndim > 1;
  const bool threed = ndim > 2;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  const int nangles = jacobi_pkg->Param<int>("nangles");

  // Extract angular grid params
  const auto agrid = GetAngularGridArrays(jacobi_pkg);
  const auto &cp = agrid.cart_pos;
  const auto &gflx = agrid.gflux;
  const auto &arcw = agrid.arc_weights;
  const auto &wght = agrid.weights;
  const auto &numn = agrid.num_neighbors;
  const auto &indn = agrid.ind_neighbors;

  // Utilities
  const auto unit_utils = jacobi_pkg->Param<UnitUtils>("unit_utils");
  const Real cdt = unit_utils.c * dt;
  const auto futils = jacobi_pkg->Param<FluxUtils>("flux_utils");

  // Multigroup bins
  const auto fbnd = *(jacobi_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

  // Packing
  namespace ccr = cell_variables::cell_averaged::rad;
  using parthenon::MakePackDescriptor;
  static auto desc_g =
      MakePackDescriptor<ccr::aa, ccr::ss, ccr::s1, ccr::s2, ccr::s3, ccr::temperature>(
          resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccr::intensity>(resolved_pkgs.get());
  static auto desc_r = MakePackDescriptor<ccr::intensity_iterate>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(rbase);
  auto vb = desc_i.GetPack(rbase);
  auto vi = desc_r.GetPack(riter);
  auto vo = desc_i.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // The mean intensity of the iterate is summed from the initial guess on the first
  // iteration, and otherwise follows from the s1-s3 of ApplyRHS of the previous iteration
  const bool first_iter = (jacobi_pkg->Param<int>("current_iter") == 1);

  // With angular fluxes, the angles of each group are swept in upwind (descending index)
  // order
  const int nsweep = do_angular_fluxes ? nangles : 1;
  const RedBlack rb(rout, color);
  auto idx_space = rb.GetIndexSpace(vo.GetNBlocks(), ngroups * nangles / nsweep);
  RiotFlatLoop::five_d(
      "JacobiUpdate", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &nl, const int &k, const int &j,
                    const int &ih) {
        const int i = rb.I(k, j, ih);
        if (i > rb.ib.e) return;

        // Extract coordinates
        const auto &coords = vg.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);
        const auto fa = FaceAreas(coords, ndim, k, j, i);
        [[maybe_unused]] Real mcw = 0.0;
        if constexpr (do_angular_fluxes) {
          mcw = -cdt * InverseRadiusForAngularFlux(coords, i);
        }

        // Optical depth weights of the group
        const int gg = do_angular_fluxes ? nl : GI(nangles, nl);
        const auto tw = FaceTauWeights(vg, coords, ndim, futils, gg, b, k, j, i);

        // Collision source of the previous iterate, seen by the swept upwind angle
        Real cdtsigt = 0.0, sprev = 0.0;
        if (do_angular_fluxes && coupling) {
          const Real cdtsiga = cdt * vg(b, ccr::aa(nl), k, j, i);
          const Real cdtsigs = cdt * vg(b, ccr::ss(nl), k, j, i);
          const Real ee = Emissivity(nl, vg(b, ccr::temperature(), k, j, i), fbnd,
                                     ngroups, unit_utils);
          Real jj = (!first_iter) *
                    (vg(b, ccr::s1(nl), k, j, i) * ee + vg(b, ccr::s2(nl), k, j, i)) *
                    vg(b, ccr::s3(nl), k, j, i);
          if (first_iter) {
            for (int aa = 0; aa < nangles; ++aa) {
              jj += wght(aa) * vi(b, GAI(nangles, nl, aa), k, j, i);
            }
          }
          cdtsigt = cdtsiga + cdtsigs;
          sprev = cdtsigs * jj + cdtsiga * ee;
        }

        Real iu = 0.0;
        for (int m = 0; m < nsweep; ++m) {
          // Indexing
          const int n = do_angular_fluxes ? GAI(nangles, nl, nangles - 1 - m) : nl;
          const int aa = AI(nangles, n);

          // Compute Jacobi coefficients
          // clang-format off
          Real g1p = 1.0, g1pp = 0.0;
          std::array<Real, 6> gn = {0.0};
          GCoef<X1DIR>(fa, cp, NDIR, ivol, split_g1,
                       tw, aa, g1p, g1pp, gn[0], gn[1]);
          if (multid) GCoef<X2DIR>(fa, cp, NDIR, ivol, split_g1,
                                   tw, aa, g1p, g1pp, gn[2], gn[3]);
          if (threed) GCoef<X3DIR>(fa, cp, NDIR, ivol, split_g1,
                                   tw, aa, g1p, g1pp, gn[4], gn[5]);
          // clang-format on

          // Spatial flux divergence operator
          Real df = g1pp * vi(b, n, k, j, i);
          df += (gn[0] * rb.Neighbor(vi, vo, b, n, k, j, i - 1) +
                 gn[1] * rb.Neighbor(vi, vo, b, n, k, j, i + 1));
          df += multid * (gn[2] * rb.Neighbor(vi, vo, b, n, k, j - multid, i) +
                          gn[3] * rb.Neighbor(vi, vo, b, n, k, j + multid, i));
          df += threed * (gn[4] * rb.Neighbor(vi, vo, b, n, k - threed, j, i) +
                          gn[5] * rb.Neighbor(vi, vo, b, n, k + threed, j, i));

          // Angular flux divergence operator
          [[maybe_unused]] auto &wght_ = wght;
          [[maybe_unused]] auto &numn_ = numn;
          [[maybe_unused]] auto &gflx_ = gflx;
          [[maybe_unused]] auto &arcw_ = arcw;
          [[maybe_unused]] auto &indn_ = indn;
          if constexpr (do_angular_fluxes) {
            const Real mcw_ix1 = mcw / wght_(aa);
            for (int nb = 0; nb < numn_(aa); ++nb) {
              const Real na = mcw_ix1 * gflx_(aa, nb) * arcw_(aa, nb);
              g1p += (na > 0) * na;
              df += (na < 0) * na * iu;
            }
          }

          Real &vv = vo(b, n, k, j, i);
          vv = std::max((vb(b, n, k, j, i) - df) / g1p, 0.0);
          iu = std::max(vv + (sprev - cdtsigt * vv) / (g1p + cdtsigt), 0.0);
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ApplyRHS
//! \brief Computes Jacobi source term due to emission, absorption, and scattering for
//! the cells of one color
TaskStatus ApplyRHS(MeshData<Real> *rbase, MeshData<Real> *rout, MeshData<Real> *ubase,
                    const Real dt, const int color) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;
  using parthenon::MakePackDescriptor;

  // Extract Jacobi package
  auto pm = rbase->GetParentPointer();
  auto &jacobi_pkg = pm->packages.Get(pkg_name);
  const bool coupling = jacobi_pkg->Param<bool>("coupling");
  if (!(coupling)) return TaskStatus::complete;
  const bool split_g1 = jacobi_pkg->Param<bool>("split_g1");

  // troot verbosity flag
  const bool verbose = (jacobi_pkg->Param<int>("verbose") == 3);

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  const int nangles = jacobi_pkg->Param<int>("nangles");

  // Utilities
  const auto unit_utils = jacobi_pkg->Param<UnitUtils>("unit_utils");
  const auto root_utils = jacobi_pkg->Param<RootUtils>("root_utils");
  const Real cdt = unit_utils.c * dt;

  // Multigroup bins
  const auto fbnd = *(jacobi_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

  // Extract Jacobi params
  const bool affect_fluid = jacobi_pkg->Param<bool>("affect_fluid");
  const bool fixed_temp_rhs = jacobi_pkg->Param<bool>("fixed_temp_rhs");
  const auto futils = jacobi_pkg->Param<FluxUtils>("flux_utils");

  // Extract angular grid params
  const auto agrid = GetAngularGridArrays(jacobi_pkg);
  const auto &cp = agrid.cart_pos;
  const auto &gflx = agrid.gflux;
  const auto &arcw = agrid.arc_weights;
  const auto &wght = agrid.weights;
  const auto &numn = agrid.num_neighbors;

  // Hydro
  auto &hydro_pkg = pm->packages.Get("hydro");
  const Real &tflr = hydro_pkg->Param<Real>("temp_floor");

  // Materials
  auto &mat_pkg = pm->packages.Get("materials");
  const auto &eos = mat_pkg->Param<parthenon::ParArray1D<RiotEOS::EOS>>("d.d.EOS");
  const auto &eos_map = mat_pkg->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  // Packing
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc_b =
      MakePackDescriptor<ccrad::aa, ccrad::ss, ccrad::s1, ccrad::s2, ccrad::s3,
                         ccrad::temperature>(resolved_pkgs.get());
  static auto desc_u = MakePackDescriptor<ccmat::rho, cm::rho, ccbulk::temperature,
                                          ccbulk::internal_energy>(resolved_pkgs.get());
  static auto desc_o = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  auto vb = desc_b.GetPack(rbase);
  auto vu = desc_u.GetPack(ubase);
  auto vo = desc_o.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // Apply Jacobi source term
  const RedBlack rb(rbase, color);
  auto idx_space = rb.GetIndexSpace(vo.GetNBlocks());
  RiotFlatLoop::four_d(
      "ApplyRHS", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &ih) {
        const int i = rb.I(k, j, ih);
        if (i > rb.ib.e) return;

        // Extract coordinates
        auto &coords = vb.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);
        const auto fa = FaceAreas(coords, ndim, k, j, i);
        Real mcw = 0.0;
        if constexpr (do_angular_fluxes) {
          mcw = -cdt * InverseRadiusForAngularFlux(coords, i);
        }

        // Capture fluid state this MeshBlock
        const int nmat1 = vu.GetSize(b, ccmat::rho()) - 1;
        const Real &tbase = vu(b, ccbulk::temperature(), k, j, i);
        Real &tadv = vb(b, ccrad::temperature(), k, j, i);
        Real troot = tadv;

        for (int gg = 0; gg < ngroups; ++gg) {
          // Set opacity weighted timesteps
          const Real cdtsiga = cdt * vb(b, ccrad::aa(gg), k, j, i);
          const Real cdtsigs = cdt * vb(b, ccrad::ss(gg), k, j, i);
          const Real cdtsigt = cdtsiga + cdtsigs;

          // Set polynomial coefficients (A1, A2, A3)
          Real &aa1 = vb(b, ccrad::s1(gg), k, j, i) = 0.0;
          Real &aa2 = vb(b, ccrad::s2(gg), k, j, i) = 0.0;
          Real &aa3 = vb(b, ccrad::s3(gg), k, j, i) = 0.0;
          const auto tw = FaceTauWeights(vb, coords, ndim, futils, gg, b, k, j, i);
          for (int aa = 0; aa < nangles; ++aa) {
            const Real g1 = G1Coef(fa, cp, NDIR, ndim, wght, gflx, arcw, numn, ivol, mcw,
                                   split_g1, tw, aa);
            const Real &ii = vo(b, GAI(nangles, gg, aa), k, j, i);
            const Real tmp = wght(aa) / (g1 + cdtsigt);
            aa1 += tmp;
            aa2 += g1 * ii * tmp;
          }
          aa3 = 1.0 / (1.0 - aa1 * cdtsigs);
          aa1 *= cdtsiga;
        }

        // Compute new gas temperature (short-circuiting on fixed_temp_rhs)
        const bool success =
            (fixed_temp_rhs ||
             AdvancedTemperature(troot, tbase, verbose, tflr, root_utils, unit_utils, cdt,
                                 ngroups, fbnd, vb, nmat1, eos, eos_map, vu, b, k, j, i));
        tadv = (success) ? troot : tbase;

        // Update the specific intensity
        if (success) {
          for (int gg = 0; gg < ngroups; ++gg) {
            // Set opacity weighted timesteps
            const Real cdtsiga = cdt * vb(b, ccrad::aa(gg), k, j, i);
            const Real cdtsigs = cdt * vb(b, ccrad::ss(gg), k, j, i);
            const Real cdtsigt = cdtsiga + cdtsigs;

            // Calculate polynomial coefficients (A1, A2, A3)
            const Real &aa1 = vb(b, ccrad::s1(gg), k, j, i);
            const Real &aa2 = vb(b, ccrad::s2(gg), k, j, i);
            const Real &aa3 = vb(b, ccrad::s3(gg), k, j, i);

            // Set advanced stage specific intensity
            const Real ee = Emissivity(gg, tadv, fbnd, ngroups, unit_utils);
            const Real jj = (aa1 * ee + aa2) * aa3;
            const auto tw = FaceTauWeights(vb, coords, ndim, futils, gg, b, k, j, i);
            for (int aa = 0; aa < nangles; ++aa) {
              const Real g1 = G1Coef(fa, cp, NDIR, ndim, wght, gflx, arcw, numn, ivol,
                                     mcw, split_g1, tw, aa);
              Real &ii = vo(b, GAI(nangles, gg, aa), k, j, i);
              // clang-format off
              ii = std::max(ii + ((cdtsigs * jj +
                                   cdtsiga * ee -
                                   cdtsigt * ii) / (g1 + cdtsigt)), 0.0);
              // clang-format on
            }
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Jacobi::JacobiFeedback
//! \brief Difference moments to update fluid conserved variables after Jacobi update
TaskStatus JacobiFeedback(MeshData<Real> *rbase, MeshData<Real> *riter,
                          MeshData<Real> *rout, MeshData<Real> *ubase, const Real dt) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = rbase->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  if (!(jacobi_pkg->Param<bool>("affect_fluid"))) return TaskStatus::complete;
  const bool split_g1 = jacobi_pkg->Param<bool>("split_g1");

  // Indexing
  const int ndim = pm->ndim;
  const bool multid = ndim > 1;
  const bool threed = ndim > 2;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  const int nangles = jacobi_pkg->Param<int>("nangles");

  // Extract angular grid params
  const auto agrid = GetAngularGridArrays(jacobi_pkg);
  const auto &cp = agrid.cart_pos;
  const auto &gflx = agrid.gflux;
  const auto &arcw = agrid.arc_weights;
  const auto &wght = agrid.weights;
  const auto &numn = agrid.num_neighbors;
  const auto &indn = agrid.ind_neighbors;

  // Utilities
  const auto unit_utils = jacobi_pkg->Param<UnitUtils>("unit_utils");
  const Real cdt = unit_utils.c * dt;
  const auto futils = jacobi_pkg->Param<FluxUtils>("flux_utils");

  // Packing
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccr = cell_variables::cell_averaged::rad;
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_g = MakePackDescriptor<ccr::aa, ccr::ss>(resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccr::intensity>(resolved_pkgs.get());
  static auto desc_r = MakePackDescriptor<ccr::intensity_iterate>(resolved_pkgs.get());
  static auto desc_u = MakePackDescriptor<ccbulk::internal_energy>(resolved_pkgs.get());
  auto vu = desc_u.GetPack(ubase);
  auto vg = desc_g.GetPack(rbase);
  auto vb = desc_i.GetPack(rbase);
  auto vi = desc_r.GetPack(riter);
  auto vo = desc_i.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // Compute feedback with the intensities that the last JacobiUpdate of each cell saw, so
  // that it is the coupling source applied by the last ApplyRHS (consistent with the
  // advanced temperature)
  const RedBlack rb_red(rbase, 0), rb_black(rbase, 1);
  auto idx_space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(), rbase);
  RiotFlatLoop::four_d(
      "JacobiFeedback::Moments", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const auto &rb = ((i + j + k) & 1) ? rb_black : rb_red;

        // Extract coordinates
        auto &coords = vg.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);
        const auto fa = FaceAreas(coords, ndim, k, j, i);
        [[maybe_unused]] Real mcw = 0.0;
        if constexpr (do_angular_fluxes) {
          mcw = -cdt * InverseRadiusForAngularFlux(coords, i);
        }

        // Apply fluid feedback
        Real &eint = vu(b, ccbulk::internal_energy(), k, j, i);
        for (int gg = 0; gg < ngroups; ++gg) {
          Real er_old = 0.0;
          Real er_new = 0.0;
          const auto tw = FaceTauWeights(vg, coords, ndim, futils, gg, b, k, j, i);
          for (int aa = 0; aa < nangles; ++aa) {
            // Collapsed indexing
            const int n = GAI(nangles, gg, aa);

            // Compute Jacobi coefficients
            // clang-format off
            Real g1p = 1.0, g1pp = 0.0;
            std::array<Real, 6> gn = {0.0};
            GCoef<X1DIR>(fa, cp, NDIR, ivol, split_g1,
                         tw, aa, g1p, g1pp, gn[0], gn[1]);
            if (multid) GCoef<X2DIR>(fa, cp, NDIR, ivol, split_g1,
                                     tw, aa, g1p, g1pp, gn[2], gn[3]);
            if (threed) GCoef<X3DIR>(fa, cp, NDIR, ivol, split_g1,
                                     tw, aa, g1p, g1pp, gn[4], gn[5]);
            // clang-format on

            // Backtrack contributions from g2-g7 and gn
            Real df = g1pp * vi(b, n, k, j, i);
            df += (gn[0] * rb.Neighbor(vi, vo, b, n, k, j, i - 1) +
                   gn[1] * rb.Neighbor(vi, vo, b, n, k, j, i + 1));
            df += multid * (gn[2] * rb.Neighbor(vi, vo, b, n, k, j - multid, i) +
                            gn[3] * rb.Neighbor(vi, vo, b, n, k, j + multid, i));
            df += threed * (gn[4] * rb.Neighbor(vi, vo, b, n, k - threed, j, i) +
                            gn[5] * rb.Neighbor(vi, vo, b, n, k + threed, j, i));

            [[maybe_unused]] auto &wght_ = wght;
            [[maybe_unused]] auto &numn_ = numn;
            [[maybe_unused]] auto &gflx_ = gflx;
            [[maybe_unused]] auto &arcw_ = arcw;
            [[maybe_unused]] auto &indn_ = indn;
            if constexpr (do_angular_fluxes) {
              const Real mcw_ix1 = mcw / wght_(aa);
              for (int nb = 0; nb < numn_(aa); ++nb) {
                const Real na = mcw_ix1 * gflx_(aa, nb) * arcw_(aa, nb);
                g1p += (na > 0) * na;
                df += (na < 0) * na * vo(b, GAI(nangles, gg, indn_(aa, nb)), k, j, i);
              }
            }

            // Compute change in moments from coupling term
            // ...Jiang 21, Equation (22)...
            //      g1 I^{m+1} - Ic == RHS
            // ...where...
            //      Ic = I^m - (g2 I^{m+1}_{i+1} + g3 I^{m+1}_{i-1} + ...)
            // ...such that...
            //      I^{m+1} - RHS == Ic + (1 - g1) * I^{m+1}
            // ...and...
            //      \Delta Er == \int [ I^{m+1} ] dW - \int [ I^{m+1} - RHS ] dW
            const Real &iio = vo(b, n, k, j, i);
            const Real &iib = vb(b, n, k, j, i);
            er_new += wght(aa) * iio;
            er_old += wght(aa) * (iib - df + (1.0 - g1p) * iio);
          }

          eint += er_old - er_new;
        }
      });

  return TaskStatus::complete;
}

} // namespace Jacobi
