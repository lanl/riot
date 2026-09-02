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

  // Algorithm-shared Boundary Conditions (default, drive)
  RadiationBC::EnrollRadiationBC(jacobi_pkg.get(), pin, params);

  // Algorithm-shared Initializations (none, zero, thermal)
  RadiationInit::EnrollRadiationInit(jacobi_pkg.get(), pin, params);

  // User-Defined Metadata Flags
  auto MetadataJacobi = jacobi_pkg->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetUserFlag("OperatorSplit");

  // Jacobi Parameters
  params.Add("niter_limit",
             pin->GetOrAddInteger(input_block, "niter_limit", 1000,
                                  "Maximum #iter permitted for Jacobi integration"));
  params.Add("err_thr",
             pin->GetOrAddReal(input_block, "err_thr", 1.0e-8,
                               "Residual error threshold for Jacobi integration"));
  // Minimum-iterations
  // NOTE(): Set default to angular-mesh ~graph-diameter
  const int nlevel = params.Get<int>("nlevel");
  const std::string amesh = params.Get<std::string>("angular_mesh");
  int niter_min_default;
  if (amesh == "geodesic") {
    niter_min_default = 3 * nlevel;
  } else if (parthenon::IsCoord<parthenon::UniformSpherical>()) {
    niter_min_default = 2 * nlevel - 1;
  } else {
    niter_min_default = 4 * nlevel - 1;
  }
  params.Add("niter_min",
             pin->GetOrAddInteger(
                 input_block, "niter_min", do_angular_fluxes * niter_min_default,
                 "Minimum #iter the Jacobi solver must take before it is permitted to "
                 "exit on the residual threshold (even if err_thr is already met)."));
  const int nreduce_limit = pin->GetOrAddInteger(
      input_block, "nreduce_limit", 0,
      "Maximum number of timestep reductions permitted when the implicit Jacobi solve "
      "diverges.  On divergence the operator-split step is subcycled with dt reduced by "
      "reduce_factor and re-solved from the pristine base state.  Setting to 0 disables "
      "subcycling (a diverging solve aborts the run).");
  const int reduce_factor = pin->GetOrAddInteger(
      input_block, "reduce_factor", 2,
      "Integer factor by which the subcycle timestep is divided each time the implicit "
      "Jacobi solve diverges (e.g. 2 halves dt). Setting to -1 disables the stall "
      "detector entirely in which only a NaN/Inf residual can fail a solve)");
  const int ndiverge_limit = pin->GetOrAddInteger(
      input_block, "ndiverge_limit", -1,
      "Number of consecutive Jacobi iterations without improvement on the best residual "
      "seen so far that constitutes divergence and triggers a dt reduction/subcycle.");
  PARTHENON_REQUIRE(nreduce_limit >= 0,
                    "radiation_transport/jacobi/nreduce_limit must be >= 0!");
  PARTHENON_REQUIRE(reduce_factor > 1,
                    "radiation_transport/jacobi/reduce_factor must be an integer > 1!");
  PARTHENON_REQUIRE(
      ndiverge_limit == -1 || ndiverge_limit > 0,
      "radiation_transport/jacobi/ndiverge_limit must be > 0, or -1 to disable");
  params.Add("nreduce_limit", nreduce_limit);
  params.Add("reduce_factor", reduce_factor);
  params.Add("ndiverge_limit", ndiverge_limit);

  // Timestep controllers
  params.Add("dt_ratio_hyperbolic",
             pin->GetOrAddReal(input_block, "dt_ratio_hyperbolic", 1.0e4),
             "Limit timestep of implicit update of thermal radiation transport by "
             "cfl * dt_ratio_hyperbolic * min_dx / c.  Setting to -1 does not permit "
             "this timestep controller to limit global timestep.");
  params.Add(
      "dt_ratio_lag",
      pin->GetOrAddReal(
          input_block, "dt_ratio_lag", -1.0,
          "EXPERIMENTAL: Limit timestep of implicit update of thermal radiation "
          "transport to account for lagged opacities in implicit solve. Setting "
          "to -1 does not permit this timestep controller to limit global timestep."));
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
  params.Add("current_residual", std::numeric_limits<Real>::max(), true);
  params.Add("current_iter", std::numeric_limits<int>::max(), true);
  params.Add("solve_diverged", false, true);
  params.Add("best_residual", std::numeric_limits<Real>::max(), true);
  params.Add("nstall", 0, true);
  params.Add("time", 0.0, Params::Mutability::Restart);

  // Reducer for residual
  parthenon::AllReduce<HostArray1D<Real>> residual_reducer;
  residual_reducer.val = HostArray1D<Real>("Jacobi_pkg Residual Reducer", 2);
  for (int i = 0; i < residual_reducer.val.size(); ++i)
    residual_reducer.val(i) = 0.0;
  params.Add("jresidual_reducer", residual_reducer, true);

  // Radiation specific intensity I
  std::string control_field = ccrad::intensity::name();
  using namespace parthenon::refinement_ops;
  Metadata mi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                          Metadata::Intensive, Metadata::Conserved, MetadataJacobi,
                          MetadataOperatorSplit},
                         std::vector<int>({ngroups * nangles}));
  mi.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  jacobi_pkg->AddField<ccrad::intensity>(mi);

  // Opacity fields
  Metadata mo = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                          Metadata::FillGhost, MetadataJacobi, MetadataOperatorSplit},
                         std::vector<int>({ngroups}));
  mo.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
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
  jacobi_pkg->AddField<ccrad::tauw>(m);

  // Advanced temperature field
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataJacobi,
                MetadataOperatorSplit});
  jacobi_pkg->AddField<ccrad::temperature>(m);

  // Jacobi Timestep
  jacobi_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Set Moments
  jacobi_pkg->FillDerivedMesh = SetMomentsMesh;

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
  const Real dt_ratio_lag = jacobi_pkg->Param<Real>("dt_ratio_lag");
  const bool do_lag = (dt_ratio_lag > 0.0) && jacobi_pkg->Param<bool>("coupling");

  // If neither dt_ratio_hyperbolic nor dt_ratio_lag constraints are active, radiation
  // does not vote a timestep.
  if (dt_ratio_hyperbolic < 0.0 && !do_lag) return max_dt;

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
            ldt = std::min(ldt, std::acos(n1 * dn1 + n2 * dn2 + n3 * dn3) / absna);
          }
        });
  }

  // Time controller accounting for lagged opacity
  Real lag_dt = max_dt;
  if (do_lag) {
    const Real tfloor = pm->packages.Get("hydro")->Param<Real>("temp_floor");
    const int ngroups = jacobi_pkg->Param<int>("ngroups");
    const auto fbnd = *(jacobi_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

    // Opacity models
    auto &mat_pkg = pm->packages.Get("materials");
    const auto opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
    const auto opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");

    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;
    namespace ccrad = cell_variables::cell_averaged::rad;
    static auto desc_c =
        MakePackDescriptor<cm::rho, ccmat::rho, ccmat::volume_fraction, cm::specific_heat,
                           ccbulk::temperature, ccrad::moments>(resolved_pkgs.get());
    auto vc = desc_c.GetPack(md);
    const auto uu = unit_utils;

    using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto idx_space = rt::GetIndexSpace(IndexDomain::interior, vc.GetNBlocks(), md);
    const Real teq_min = rt::four_d(
        "Jacobi::EquilTimestep", idx_space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
          const Real temp = vc(b, ccbulk::temperature(), k, j, i);
          if (temp <= tfloor) return;

          // Volumetric heat capacity Cv
          const int nmat = vc.GetSize(b, cm::rho());
          Real Cv = 0.0;
          for (int m = 0; m < nmat; ++m) {
            Cv += vc(b, ccmat::rho(m), k, j, i) * vc(b, cm::specific_heat(m), k, j, i);
          }
          if (Cv <= 0.0) return;

          // Coupling rates
          Real rate_sum1 = 0.0;
          Real rate_sum2 = 0.0;
          for (int gg = 0; gg < ngroups; ++gg) {
            Real alpha = 0.0;
            Real dalpha = 0.0;
            const Real eps = Emissivity(gg, temp, fbnd, ngroups, uu);
            const Real depsdT = EmissivityDT(gg, temp, fbnd, ngroups, uu);
            for (int m = 0; m < nmat; ++m) {
              const Real rhom = vc(b, cm::rho(m), k, j, i);
              if (rhom <= 0.0) continue;
              const Real vfm = vc(b, ccmat::volume_fraction(m), k, j, i);
              const int mat_id = vc(b, cm::rho(m)).sparse_id;
              const int phase_id = vc(b, cm::rho(m)).v;
              const int opac_id = opac_from_matid(mat_id) + phase_id;
              const Real aam = opac_a(opac_id).AbsorptionCoefficient(rhom, temp, gg);
              const Real dloga =
                  opac_a(opac_id).DLogAbsorptionCoefficientDLogT(rhom, temp, gg);
              alpha += vfm * aam;
              dalpha += vfm * aam * dloga;
            }
            rate_sum1 += (vc(b, ccrad::moments(gg), k, j, i) - eps) * dalpha;
            rate_sum2 += alpha * depsdT;
          }
          const Real sum_rate = std::abs(rate_sum1) / (dt_ratio_lag * temp) - rate_sum2;
          if (sum_rate <= 0.0) return;

          const Real teq = Cv / (cc * sum_rate);
          ldt = std::min(ldt, teq);
        });

    lag_dt = teq_min;
  }

  // Set timestep
  const Real cfl = jacobi_pkg->Param<Real>("cfl");
  Real dt = max_dt;
  if (dt_ratio_hyperbolic >= 0.0)
    dt = std::min(dt, cfl * std::min(min_sdt, min_adt) * dt_ratio_hyperbolic);
  if (do_lag) dt = std::min(dt, lag_dt);
  return dt;
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
  std::vector<std::string> opac_names = pm->GetVariableNames(
      std::vector<std::string>{ccrad::aa::name(), ccrad::ss::name()}, std::vector<int>{});
  auto &base = pm->mesh_data.Get();
  pm->mesh_data.Add(mdname::riter, base, intensity_names);
  pm->mesh_data.Add(mdname::rout, base, intensity_names);
  pm->mesh_data.AddShallow(mdname::rbase, base, jacobi_names);
  pm->mesh_data.AddShallow(mdname::ubase, base, unsplit_names);
  pm->mesh_data.AddShallow(mdname::ropac, base, opac_names);

  // Subcycle controls
  const int nreduce_limit = jacobi_pkg->Param<int>("nreduce_limit");
  const int reduce_factor = jacobi_pkg->Param<int>("reduce_factor");
  const int verbose = jacobi_pkg->Param<int>("verbose");

  // Adaptive subcycle loop
  int ndiv = 1;
  int nremaining = 1;
  int nreduce = 0;
  int nsub = 0;
  while (nremaining > 0) {
    const Real dt_this = dt / ndiv;
    const Real t_local = dt * (ndiv - nremaining) / ndiv;

    // Report subcycle progress
    if (nreduce > 0 && Globals::my_rank == 0 && verbose >= 1) {
      printf(
          "(Jacobi) Subcycle attempt %d | dt_sub: %24.16e | committed: %6.2f%% of dt\n",
          nsub + 1, dt_this, 100.0 * t_local / dt);
    }

    // Initialize subcycler params
    jacobi_pkg->UpdateParam("solve_diverged", false);
    jacobi_pkg->UpdateParam("nstall", 0);
    jacobi_pkg->UpdateParam("best_residual", std::numeric_limits<Real>::max());

    // Solve from the current base state
    const auto solve_status = JacobiSolve(pm, tm.time + t_local, dt_this).Execute();
    PARTHENON_REQUIRE(solve_status == TaskListStatus::complete,
                      "Jacobi solve task list did not complete.");

    if (!jacobi_pkg->Param<bool>("solve_diverged")) {
      // Solve did not diverge (converged, or hit niter_limit without diverging): commit.
      // Advance the intensity and (if coupled) the fluid.
      const auto commit_status = JacobiCommit(pm, dt_this).Execute();
      PARTHENON_REQUIRE(commit_status == TaskListStatus::complete,
                        "Jacobi commit task list did not complete.");
      --nremaining;
      ++nsub;
    } else {
      // Solve diverged: back out and retry the remaining span with a reduced timestep
      ++nreduce;
      PARTHENON_REQUIRE(
          nreduce <= nreduce_limit,
          "Implicit Jacobi radiation solve diverged after " +
              std::to_string(nreduce_limit) + " timestep reduction(s).  Residual: " +
              std::to_string(jacobi_pkg->Param<Real>("current_residual")) +
              ", dt_sub: " + std::to_string(dt_this) +
              ".  Raise nreduce_limit, loosen ndiverge_limit, or check the setup.");
      ndiv *= reduce_factor;
      nremaining *= reduce_factor;
      if (Globals::my_rank == 0 && verbose >= 1)
        printf("(Jacobi) Solve diverged; reducing dt (reduction %d/%d), "
               "new dt_sub: %24.16e\n",
               nreduce, nreduce_limit, dt / ndiv);
    }
  }

  if (nreduce > 0 && Globals::my_rank == 0 && verbose >= 1) {
    printf("(Jacobi) Completed operator-split step in %d subcycles (%d reduction(s))\n",
           nsub, nreduce);
  }

  return TaskCollection();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection JacobiSolve
//! \brief
TaskCollection JacobiSolve(Mesh *pmesh, const Real time, const Real dt) {
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
    auto &ropac = pmesh->mesh_data.GetOrAdd(mdname::ropac, i);
    auto &tl = tr[i];

    // Start receives on MeshData registers
    auto srec = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, ropac);
    srec = tl.AddTask(srec, parthenon::StartReceiveBoundBufs<any>, rout);

    // Set opacities
    auto set_opac = tl.AddTask(none, SetOpacities, rbase.get(), ubase.get());
    auto opac_bc =
        parthenon::AddBoundaryExchangeTasks(set_opac, tl, ropac, pmesh->multilevel);

    // Initialize solution in riter
    auto init_iter = tl.AddTask(
        none, sparse_update::DeepCopyData<parthenon::MetadataFlag, MeshData<Real>>,
        std::vector<MetadataFlag>({Metadata::Independent}), riter.get(), rbase.get());

    // Reset iterator
    auto reset_iter = tl.AddTask(TaskQualifier::once_per_region, none, ResetIterator,
                                 rbase.get(), time, dt);

    // Prepare for Jacobi by setting tau weights
    auto prep_jacobi =
        tl.AddTask(TaskQualifier::local_sync, opac_bc | init_iter | reset_iter,
                   PrepareForIterations, rbase.get());

    // Execute Jacobi iterations
    auto [solver, solver_id] = tl.AddSublist(prep_jacobi, {1, max_iters});
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

    // Update intensity registers
    auto copy_reg = tl.AddTask(
        feedback, sparse_update::DeepCopyData<parthenon::MetadataFlag, MeshData<Real>>,
        std::vector<MetadataFlag>({Metadata::Independent}), rbase.get(), rout.get());

    // Update fluid state if radiation affects fluid
    TaskID aux = none;
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
      aux = tl.AddTask(u_bc, FillDerived<MeshData<Real>>, ubase.get());
    }
    auto moments = tl.AddTask(
        copy_reg | aux,
        [](MeshData<Real> *md) {
          SetMomentsMesh(md);
          return TaskStatus::complete;
        },
        rbase.get());
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
  auto update =
      solver.AddTask(begin, JacobiUpdate, rbase.get(), riter.get(), rout.get(), dt);
  auto rsrc = solver.AddTask(update, ApplyRHS, rbase.get(), rout.get(), ubase.get(), dt);
  auto bc = parthenon::AddBoundaryExchangeTasks(rsrc, solver, rout, pmesh->multilevel);
  auto check = solver.AddTask(TaskQualifier::local_sync, bc, CheckConvergence,
                              &(presidual->val), riter.get(), rout.get());
  auto start_reduce_residual =
      solver.AddTask(TaskQualifier::once_per_region, check,
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
TaskStatus CheckConvergence(HostArray1D<Real> *presidual, MeshData<Real> *riter,
                            MeshData<Real> *rout) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = rout->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  auto &resolved_pkgs = pm->resolved_packages;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");
  const int nangles = jacobi_pkg->Param<int>("nangles");

  // Packing
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc_i = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  static auto desc_o = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  auto vi = desc_i.GetPack(riter);
  auto vo = desc_o.GetPack(rout);

  // Residual reductions
  using rt =
      RiotFlatReduce::ReductionType<RiotUtils::GlobalSum<Real, Kokkos::HostSpace, 2>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(), ngroups * nangles, rout);
  const auto res = rt::five_d(
      "JacobiResidual", idx_space,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, 2> &rsum) {
        rsum.my_array[0] += std::abs(vi(b, n, k, j, i) - vo(b, n, k, j, i));
        rsum.my_array[1] += vo(b, n, k, j, i);
      });
  Kokkos::fence();

  auto &residual = *presidual;
  residual(0) += res.my_array[0];
  residual(1) += res.my_array[1];

  return TaskStatus::complete;
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
  const Real current_residual = v(0) / (std::max(v(1), 1.0e-100) + (v(1) <= 1.0e-100));
  jacobi_pkg->UpdateParam("current_residual", current_residual);

  // Divergence detection
  const Real best_residual = jacobi_pkg->Param<Real>("best_residual");
  const int ndiverge_limit = jacobi_pkg->Param<int>("ndiverge_limit");
  int nstall = jacobi_pkg->Param<int>("nstall");
  const bool improved = current_residual < best_residual;
  nstall = improved ? 0 : nstall + 1;
  const bool stalled = (ndiverge_limit >= 0) && (nstall >= ndiverge_limit);
  const bool diverged = !std::isfinite(current_residual) || stalled;
  jacobi_pkg->UpdateParam("nstall", nstall);
  if (improved) jacobi_pkg->UpdateParam("best_residual", current_residual);
  if (diverged) jacobi_pkg->UpdateParam("solve_diverged", true);

  // zero residuals in prep for next iteration
  v(0) = 0.0;
  v(1) = 0.0;

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
  const Real residual = jacobi_pkg->Param<Real>("current_residual");
  const Real err_thr = jacobi_pkg->Param<Real>("err_thr");
  const int niter_limit = jacobi_pkg->Param<int>("niter_limit");
  const int niter_min = jacobi_pkg->Param<int>("niter_min");
  const int verbose = jacobi_pkg->Param<int>("verbose");

  // Iterate or finalize
  const int niter_done = iter_counter - 1;
  if (jacobi_pkg->Param<bool>("solve_diverged")) {
    if (i == 0 && Globals::my_rank == 0 && verbose >= 1)
      printf("(Jacobi) Diverged! iter: %d err: %24.16e\n", niter_done, residual);
    return TaskStatus::complete;
  } else if (residual <= err_thr && niter_done >= niter_min) {
    if (i == 0 && Globals::my_rank == 0 && verbose >= 1)
      printf("(Jacobi) Converged! iter: %d err: %24.16e\n", niter_done, residual);
    return TaskStatus::complete;
  } else if (niter_done >= niter_limit) {
    if (i == 0 && Globals::my_rank == 0 && verbose >= 1)
      printf("(Jacobi) Reached niter_limit: %d err: %24.16e\n", niter_limit, residual);
    return TaskStatus::complete;
  } else {
    if (i == 0 && Globals::my_rank == 0 && verbose == 2) {
      if (residual <= err_thr) {
        printf("(Jacobi) iter: %d err: %24.16e (residual met, but niter_min: %d)\n",
               niter_done, residual, niter_min);
      } else {
        printf("(Jacobi) iter: %d err: %24.16e\n", niter_done, residual);
      }
    }
    auto copy_data = sparse_update::DeepCopyData<parthenon::MetadataFlag, MeshData<Real>>(
        std::vector<MetadataFlag>({Metadata::Independent}), riter, rout);
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
  static auto desc_u =
      MakePackDescriptor<cm::rho, ccmat::volume_fraction, ccbulk::temperature>(
          resolved_pkgs.get());
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
            for (int gg = 0; gg < ngroups; ++gg) {
              Real &aa = vr(b, ccr::aa(gg), k, j, i) = 0.0;
              Real &ss = vr(b, ccr::ss(gg), k, j, i) = 0.0;
              const Real &temp = vu(b, ccbulk::temperature(), k, j, i);
              for (int m = 0; m < vu.GetSize(b, cm::rho()); ++m) {
                const Real &rhom = vu(b, cm::rho(m), k, j, i);
                const Real &vfm = vu(b, ccmat::volume_fraction(m), k, j, i);
                const int &mat_id = vu(b, cm::rho(m)).sparse_id;
                const int &phase_id = vu(b, cm::rho(m)).v;
                const int opac_id = opac_from_matid(mat_id) + phase_id;
                const Real aam =
                    (rhom > 0) ? opac_a(opac_id).AbsorptionCoefficient(rhom, temp, gg)
                               : 0.0;
                const Real ssm =
                    (rhom > 0) ? opac_s(opac_id).ScatteringCoefficient(rhom, temp, gg)
                               : 0.0;
                aa += vfm * aam;
                ss += vfm * ssm;
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
//! \fn  void Jacobi::PrepareForIterations
//! \brief Reset iterators and set matrix coefficients
TaskStatus PrepareForIterations(MeshData<Real> *rbase) {
  // Return if no blocks in MeshData
  if (rbase->NumBlocks() == 0) return TaskStatus::complete;

  // Extract Jacobi package
  using parthenon::MakePackDescriptor;
  auto pm = rbase->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);

  // Indexing
  auto &resolved_pkgs = pm->resolved_packages;
  IndexRange ib = rbase->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rbase->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rbase->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const bool multid = ndim > 1;
  const bool threed = ndim > 2;
  const int ngroups = jacobi_pkg->Param<int>("ngroups");

  // Additional Jacobi package params
  const auto futils = jacobi_pkg->Param<FluxUtils>("flux_utils");
  const Real beta = futils.beta;
  const Real tmax = futils.taumax;

  // Packing
  namespace ccr = cell_variables::cell_averaged::rad;
  static auto desc_g =
      MakePackDescriptor<ccr::tauw, ccr::aa, ccr::ss>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(rbase);

  // Optical-depth weights are needed one cell into the ghost layer in each active
  // direction, so extend the interior bounds by one there.
  auto idx_space = RiotFlatLoop::GetIndexSpace(
      vg.GetNBlocks(), ngroups, {kb.s - threed, kb.e + threed},
      {jb.s - multid, jb.e + multid}, {ib.s - 1, ib.e + 1});
  RiotFlatLoop::five_d(
      "OpticalDepthWeights", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &gg, const int &k, const int &j,
                    const int &i) {
        // Extract coordinates
        auto &coords = vg.GetCoordinates(b);
        Real dxmin = coords.CellWidth<X1DIR>(k, j, i);
        if (multid) dxmin = std::min(dxmin, coords.CellWidth<X2DIR>(k, j, i));
        if (threed) dxmin = std::min(dxmin, coords.CellWidth<X3DIR>(k, j, i));

        // Optical depth coefficients
        const Real tau =
            beta * dxmin * (vg(b, ccr::aa(gg), k, j, i) + vg(b, ccr::ss(gg), k, j, i));
        vg(b, ccr::tauw(gg), k, j, i) = TauWeight(tau, tmax);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus JacobiUpdate
//! \brief Compute updated solution in Jacobi iteration
TaskStatus JacobiUpdate(MeshData<Real> *rbase, MeshData<Real> *riter,
                        MeshData<Real> *rout, const Real dt) {
  if (rout->NumBlocks() == 0) return TaskStatus::complete;

  // Extract Jacobi package
  auto pm = rout->GetParentPointer();
  auto jacobi_pkg = pm->packages.Get(pkg_name);
  const bool split_g1 = jacobi_pkg->Param<bool>("split_g1");

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

  // Packing
  namespace ccr = cell_variables::cell_averaged::rad;
  using parthenon::MakePackDescriptor;
  static auto desc_g = MakePackDescriptor<ccr::tauw>(resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccr::intensity>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(rbase);
  auto vb = desc_i.GetPack(rbase);
  auto vi = desc_i.GetPack(riter);
  auto vo = desc_i.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(),
                                               ngroups * nangles, rout);
  RiotFlatLoop::five_d(
      "JacobiUpdate", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &n, const int &k, const int &j,
                    const int &i) {
        // Extract coordinates
        const auto &coords = vg.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);

        // Indexing
        const int gg = GI(nangles, n);
        const int aa = AI(nangles, n);

        // Compute Jacobi coefficients
        // clang-format off
        Real g1p = 1.0, g1pp = 0.0;
        std::array<Real, 6> gn = {0.0};
        GCoef<X1DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                     b, k, j, i, g1p, g1pp, gn[0], gn[1]);
        if (multid) GCoef<X2DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                                 b, k, j, i, g1p, g1pp, gn[2], gn[3]);
        if (threed) GCoef<X3DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                                 b, k, j, i, g1p, g1pp, gn[4], gn[5]);
        // clang-format on

        // Spatial flux divergence operator
        Real df = g1pp * vi(b, n, k, j, i);
        df += (gn[0] * vi(b, n, k, j, i - 1) + gn[1] * vi(b, n, k, j, i + 1));
        df += multid *
              (gn[2] * vi(b, n, k, j - multid, i) + gn[3] * vi(b, n, k, j + multid, i));
        df += threed *
              (gn[4] * vi(b, n, k - threed, j, i) + gn[5] * vi(b, n, k + threed, j, i));

        // Angular flux divergence operator
        [[maybe_unused]] auto &wght_ = wght;
        [[maybe_unused]] auto &numn_ = numn;
        [[maybe_unused]] auto &gflx_ = gflx;
        [[maybe_unused]] auto &arcw_ = arcw;
        [[maybe_unused]] auto &indn_ = indn;
        if constexpr (do_angular_fluxes) {
          const Real mcw_ix1 = -cdt * InverseRadiusForAngularFlux(coords, i) / wght_(aa);
          for (int nb = 0; nb < numn_(aa); ++nb) {
            const Real na = mcw_ix1 * gflx_(aa, nb) * arcw_(aa, nb);
            g1p += (na > 0) * na;
            df += (na < 0) * na * vi(b, GAI(nangles, gg, indn_(aa, nb)), k, j, i);
          }
        }

        vo(b, n, k, j, i) = std::max((vb(b, n, k, j, i) - df) / g1p, 0.0);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ApplyRHS
//! \brief Computes Jacobi source term due to emission, absorption, and scattering
TaskStatus ApplyRHS(MeshData<Real> *rbase, MeshData<Real> *rout, MeshData<Real> *ubase,
                    const Real dt) {
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
  static auto desc_g = MakePackDescriptor<ccrad::tauw>(resolved_pkgs.get());
  static auto desc_b =
      MakePackDescriptor<ccrad::aa, ccrad::ss, ccrad::s1, ccrad::s2, ccrad::s3,
                         ccrad::temperature>(resolved_pkgs.get());
  static auto desc_u = MakePackDescriptor<ccmat::rho, cm::rho, ccbulk::temperature,
                                          ccbulk::internal_energy>(resolved_pkgs.get());
  static auto desc_o = MakePackDescriptor<ccrad::intensity>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(rbase);
  auto vb = desc_b.GetPack(rbase);
  auto vu = desc_u.GetPack(ubase);
  auto vo = desc_o.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // Apply Jacobi source term
  auto idx_space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(), rbase);
  RiotFlatLoop::four_d(
      "ApplyRHS", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        auto &coords = vg.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);
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
          for (int aa = 0; aa < nangles; ++aa) {
            const Real g1 = G1Coef(vg, coords, cp, NDIR, ndim, wght, gflx, arcw, numn,
                                   ivol, mcw, split_g1, gg, aa, b, k, j, i);
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
            for (int aa = 0; aa < nangles; ++aa) {
              const Real g1 = G1Coef(vg, coords, cp, NDIR, ndim, wght, gflx, arcw, numn,
                                     ivol, mcw, split_g1, gg, aa, b, k, j, i);
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

  // Packing
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccr = cell_variables::cell_averaged::rad;
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_g = MakePackDescriptor<ccr::tauw>(resolved_pkgs.get());
  static auto desc_i = MakePackDescriptor<ccr::intensity>(resolved_pkgs.get());
  static auto desc_u = MakePackDescriptor<ccbulk::internal_energy>(resolved_pkgs.get());
  auto vu = desc_u.GetPack(ubase);
  auto vg = desc_g.GetPack(rbase);
  auto vb = desc_i.GetPack(rbase);
  auto vi = desc_i.GetPack(riter);
  auto vo = desc_i.GetPack(rout);

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // Compute feedback
  auto idx_space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, vo.GetNBlocks(), rbase);
  RiotFlatLoop::four_d(
      "JacobiFeedback::Moments", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        auto &coords = vg.GetCoordinates(b);
        const Real ivol = 0.5 * cdt / coords.CellVolume(k, j, i);

        // Apply fluid feedback
        Real &eint = vu(b, ccbulk::internal_energy(), k, j, i);
        for (int gg = 0; gg < ngroups; ++gg) {
          Real er_old = 0.0;
          Real er_new = 0.0;
          for (int aa = 0; aa < nangles; ++aa) {
            // Collapsed indexing
            const int n = GAI(nangles, gg, aa);

            // Compute Jacobi coefficients
            // clang-format off
            Real g1p = 1.0, g1pp = 0.0;
            std::array<Real, 6> gn = {0.0};
            GCoef<X1DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                         b, k, j, i, g1p, g1pp, gn[0], gn[1]);
            if (multid) GCoef<X2DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                                     b, k, j, i, g1p, g1pp, gn[2], gn[3]);
            if (threed) GCoef<X3DIR>(vg, coords, cp, NDIR, ivol, split_g1, gg, aa,
                                     b, k, j, i, g1p, g1pp, gn[4], gn[5]);
            // clang-format on

            // Backtrack contributions from g2-g7 and gn
            Real df = g1pp * vi(b, n, k, j, i);
            df += (gn[0] * vi(b, n, k, j, i - 1) + gn[1] * vi(b, n, k, j, i + 1));
            df += multid * (gn[2] * vi(b, n, k, j - multid, i) +
                            gn[3] * vi(b, n, k, j + multid, i));
            df += threed * (gn[4] * vi(b, n, k - threed, j, i) +
                            gn[5] * vi(b, n, k + threed, j, i));

            [[maybe_unused]] auto &wght_ = wght;
            [[maybe_unused]] auto &numn_ = numn;
            [[maybe_unused]] auto &gflx_ = gflx;
            [[maybe_unused]] auto &arcw_ = arcw;
            [[maybe_unused]] auto &indn_ = indn;
            if constexpr (do_angular_fluxes) {
              const Real mcw_ix1 =
                  -cdt * InverseRadiusForAngularFlux(coords, i) / wght_(aa);
              for (int nb = 0; nb < numn_(aa); ++nb) {
                const Real na = mcw_ix1 * gflx_(aa, nb) * arcw_(aa, nb);
                g1p += (na > 0) * na;
                df += (na < 0) * na * vi(b, GAI(nangles, gg, indn_(aa, nb)), k, j, i);
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
