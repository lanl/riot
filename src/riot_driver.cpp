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

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/time.h>

// TODO(JCD): these should be exported by parthenon
#include <amr_criteria/refinement_package.hpp>
#include <globals.hpp>
#include <parthenon/package.hpp>
#include <prolong_restrict/prolong_restrict.hpp>
#include <solvers/solver_base.hpp>

// Local Includes
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "ionization/ionization.hpp"
#include "laser/laser.hpp"
#include "levelsets/levelsets.hpp"
#include "materials/materials.hpp"
#include "mix/mix.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "riot_driver.hpp"
#include "riot_utils/fpe_trap.hpp"
#include "riot_utils/sparse_update.hpp"
#include "scalars/scalars.hpp"
#include "strength/strength.hpp"
#include "tnburn/tnburn.hpp"
#include "variables.hpp"

using namespace parthenon::driver::prelude;

namespace riot {

//----------------------------------------------------------------------------------------
//! \fn  RiotDriver::RiotDriver
//! \brief
RiotDriver::RiotDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : EvolutionDriver(pin, app_in, pm), integrator(std::make_unique<Integrator_t>(pin)) {

  if (parthenon::IsCoord<parthenon::UniformSpherical>() && pm->ndim > 1) {
    PARTHENON_FAIL(
        "riot does not support more than one dimension with spherical coordinates");
  }
  if (parthenon::IsCoord<parthenon::UniformCylindrical>() && pm->ndim > 2) {
    PARTHENON_FAIL(
        "riot does not support more than two dimensions with cylindrical coordinates");
  }

  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");

  dt_init =
      pin->GetOrAddReal("parthenon/time", "dt_init", std::numeric_limits<Real>::max());
  dt_init_fact = pin->GetOrAddReal("parthenon/time", "dt_init_fact", 1.0);

  riot_pkg = pm->packages.Get("riot").get();
  do_hydro = riot_pkg->Param<bool>("do_hydro");
  do_strength = riot_pkg->Param<bool>("do_strength");
  do_mix = riot_pkg->Param<bool>("do_mix");
  do_tn = riot_pkg->Param<bool>("do_tn");
  fixed_fluid = riot_pkg->Param<bool>("fixed_fluid");
  do_multigroup_diffusion = riot_pkg->Param<bool>("do_multigroup_diffusion");
  do_levelsets = riot_pkg->Param<bool>("do_levelsets");
  do_lasers = riot_pkg->Param<bool>("do_lasers");
  do_gravity = riot_pkg->Param<bool>("do_gravity");
  do_ionization = riot_pkg->Param<bool>("do_ionization");
  curvilinear = riot_pkg->Param<bool>("curvilinear");

  // enable FPE trapping
  const bool trap_fpes = riot_pkg->Param<bool>("trap_fpes");
  const int ierr = setup_floating_point(trap_fpes);

  if (do_hydro) hydro_pkg = pm->packages.Get("hydro").get();
  if (do_hydro) mat_pkg = pm->packages.Get("materials").get();
  if (do_mix) mix_pkg = pm->packages.Get("mix").get();
  if (do_levelsets) lset_pkg = pm->packages.Get("levelsets").get();
  if (do_tn) tn_pkg = pm->packages.Get("TNBurn").get();
  if (do_lasers) laser_pkg = pm->packages.Get("laser").get();
  if (do_gravity) gravity_pkg = pm->packages.Get("gravity").get();
  if (do_strength) strength_pkg = pm->packages.Get("strength").get();
  if (do_ionization) ion_pkg = pm->packages.Get("ionization").get();

  plugins.DriverParams(pin, app_in, pm);

  sparse_dealloc = riot_pkg->Param<bool>("sparse_dealloc");
  use_general_pte = (do_hydro) ? mat_pkg->Param<bool>("use_general_pte") : false;
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::RiotStepInit
//! \brief
void RiotDriver::RiotStepInit() {
  namespace mdname = container_names;

  // set the time step appropriately, including init safety factors
  static bool first_call = true;
  Real dt_trial = tm.dt;
  if (first_call) dt_trial = std::min(dt_trial, dt_init);
  first_call = false;
  dt_trial *= dt_init_fact;
  dt_init_fact = 1.0;
  if (tm.time + dt_trial > tm.tlim) dt_trial = tm.tlim - tm.time;
  tm.dt = dt_trial;
  integrator->dt = dt_trial;

  // Assign registers with fields required in unsplit, RK integration
  static const auto names = RiotUtils::GetUnsplitVarNames(pmesh);
  auto &base = pmesh->mesh_data.Get();
  auto &u0 = pmesh->mesh_data.AddShallow(mdname::u0, base, names);
  auto &u1 = pmesh->mesh_data.Add(mdname::u1, u0);

  SetDiagnostics();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskListStatus RiotDriver::Step
//! \brief
TaskListStatus RiotDriver::Step() {
  RiotStepInit();

  if (do_lasers) {
    auto status = Laser::LaserUpdateTasks(pmesh, tm.time, integrator->dt).Execute();
    if (status != TaskListStatus::complete) return status;
    bool reset_dt = Laser::CheckDt(pmesh, &integrator->dt);
    if (reset_dt) tm.dt = integrator->dt;
  }

  auto status = RiotStepTasks().Execute();
  if (status != TaskListStatus::complete) return status;

  for (auto &fn : OperatorSplitTasks) {
    status = fn(pmesh, tm, integrator->dt).Execute();
    if (status != TaskListStatus::complete) return status;
  }

  status = RiotPostStepTasks().Execute();

  return status;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection RiotDriver::RiotStepTasks
//! \brief
TaskCollection RiotDriver::RiotStepTasks() {
  TaskCollection tc;
  // Return empty TaskCollection if hydro disabled
  if (!(do_hydro)) return tc;

  if (fixed_fluid && do_lasers)
    return Laser::LaserDepositionTasks(pmesh, integrator->dt);
  else if (fixed_fluid)
    return tc;

  using namespace ::parthenon::Update;
  namespace mdname = container_names;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // first deep copy u1 <-- u0
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd(mdname::u0, i);
    auto &mu1 = pmesh->mesh_data.GetOrAdd(mdname::u1, i);
    tl.AddTask(none, sparse_update::DeepCopyIndependentData<MeshData<Real>>, mu1.get(),
               mu0.get());
  }

  // now do multi-stage RK integration of physics
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    const Real beta = integrator->beta[stage - 1];
    const Real gam0 = integrator->gam0[stage - 1];
    const Real gam1 = integrator->gam1[stage - 1];
    const Real dt = integrator->dt;
    const Real stage_time = tm.time + integrator->c[stage - 1] * dt;

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd(mdname::u0, i);
      auto &mu1 = pmesh->mesh_data.GetOrAdd(mdname::u1, i);

      // start looking for incoming messages. These calls are not required but may
      // have an impact on MPI performance (TODO(LFR): Check if these change anything)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mu0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

      // compute hydro fluxes
      auto hydro_flx = none;
      if (do_hydro) {
        hydro_flx = tl.AddTask(none, Hydro::CalculateFluxes, mu0.get());
      }

      // calculate mix fluxes (before doing flux correction)
      auto mix_flx = none;
      if (do_mix) {
        // fluxes on momentum/total energy due to Reynolds Stress
        auto mix_flx_stresses =
            tl.AddTask(hydro_flx, Mix::ComputeStressFluxes, mu0.get());
        // viscous fluxes
        auto mix_flx_diff =
            tl.AddTask(mix_flx_stresses, Mix::ComputeViscousFluxes, mu0.get());
        // anonymous Fluxes
        mix_flx = tl.AddTask(mix_flx_diff, Mix::ComputeAnonFluxes, mu0.get());
      }

      TaskID plasma_viscosity_flx = none;
      if (do_ionization) {
        plasma_viscosity_flx = tl.AddTask(
            hydro_flx | mix_flx, Ionization::ComputePlasmaViscousFluxes, mu0.get());
      }

      // send flux corrections
      auto send_flx = tl.AddTask(hydro_flx | plasma_viscosity_flx | mix_flx,
                                 parthenon::LoadAndSendFluxCorrections, mu0);

      // geometric sources, if curvilinear
      auto geom_source = none;
      if (curvilinear) {
        geom_source =
            tl.AddTask(none, Hydro::CalculateGeometricSource, mu0.get(),
                       hydro_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get());
      }

      // compute strength sources, if enabled
      auto strength_source = none;
      if (do_strength) {
        strength_source =
            tl.AddTask(hydro_flx | mix_flx, Strength::CalculateStrengthSource, mu0.get(),
                       strength_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get());
      }

      // compute gravitational sources, if enabled
      auto gravity_source = none;
      if (do_gravity) {
        gravity_source =
            tl.AddTask(none, Gravity::CalculateGravitySource, mu0.get(),
                       gravity_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get());
      }

      auto plugin_sources = plugins.AddSources(tl, none, mu0.get(), pmesh, i, dt);

      // compute Mix sources, if enabled
      auto mix_source = none;
      if (do_mix) {
        mix_source = tl.AddTask(none, Mix::CalculateMixSource, mu0.get(),
                                mix_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get());
      }

      // compute TN sources, if enabled
      auto tn_shared_source = none;
      if (do_tn) {
        auto *pmdudt_tn = tn_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get();
        auto tn_source =
            tl.AddTask(none, TNBurn::CalculateTNBurnSource, mu0.get(), pmdudt_tn, dt);
        tn_shared_source =
            tl.AddTask(tn_source, TNBurn::SharedSources, pmdudt_tn, pmdudt_tn, mu0.get());
      }

      // compute ionization sources, if enabled
      auto ionization_source = none;
      if (do_ionization) {
        ionization_source =
            tl.AddTask(none, Ionization::CalculateElectronPDVWork, mu0.get(),
                       ion_pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get());
        // TODO(JMM): Add heat exchange sources
      }

      // receive and apply flux corrections
      auto recv_flx = tl.AddTask(start_flx_recv, parthenon::ReceiveFluxCorrections, mu0);
      auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, mu0);

      // one giant update kernel to rule them all
      // JMM: UpdateToNextStage sums every source term in dudt_args into the state
      // (weighted by beta*dt) as part of the update, so we hand it all the individual
      // source MeshData objects directly rather than pre-accumulating them into a
      // separate dudt_all object. Build the list generically from every package that
      // registered a "dudt" SubMeshData subset. The UIDs were cached when the base
      // driver allocated the subsets before this step; they are identical across
      // partitions, so the cached value is valid for this partition i.
      sparse_update::dudt_vec_t dudt_args;
      for (auto &[label, pkg] : pmesh->packages.AllPackagesWithSubMeshData("dudt")) {
        auto *smd = pkg->GetOrAddMeshDataSubset(pmesh, "dudt", i).get();
        const auto &uids = pkg->GetAllMeshDataSubsets().at("dudt").GetUids();
        dudt_args.emplace_back(smd, uids);
      }
      auto update = tl.AddTask(hydro_flx | mix_flx | set_flx | geom_source |
                                   strength_source | gravity_source | plugin_sources |
                                   mix_source | tn_shared_source | ionization_source,
                               sparse_update::UpdateToNextStage, mu0.get(), mu1.get(),
                               gam0, gam1, beta * dt, dudt_args);

      // Set maximum signal speeds
      // NOTE(@pdmullen): This task goes away if we permit Metadata::FillGhost for
      // Metadata::CellMemAligned face fields
      auto max_signal = none;
      if (do_hydro && (stage == integrator->nstages)) {
        max_signal =
            tl.AddTask(hydro_flx | set_flx, Hydro::CalculateMaxSignalSpeed, mu0.get());
      }

      // execute radial return for calculations with material strength
      // NOTE(@jonahm): Previously, we called this after FillInteriorDerived so that the
      // volume fractions and temperatures associated with PTE at the advanced stage can
      // inform the strength models. Following plastic work, PTE needed to be called again
      // after radial return.  We now move the radial return call to right after update.
      // This implies time-lagged (by 1 RK stage) primitive data, such as cm::rho, but
      // saves us from having to do an additional PTE solve. It also allows us fuse the
      // failure model with radial return.
      auto radial_return = none;
      if (do_strength) { // && (stage == integrator->nstages)) {
        radial_return = tl.AddTask(update, Strength::RadialReturn, mu0.get());
      }

      // NOTE(@swj): why aren't we doing this in PreCommFillDerived?
      // Probably because we didn't want it to be done upon a remesh, but maybe we do
      // actually... needs more thought what quantities we prolong/restrict and what is
      // derived from what?
      auto electron_entropy_to_energy = none;
      if (do_ionization) {
        electron_entropy_to_energy =
            tl.AddTask(update | radial_return, Ionization::ConvertElectronEntropyToEnergy,
                       mu0.get(), IndexDomain::interior);
      }

      // set initial guesses for volume fractions for general PTE
      auto set_vfrac = none;
      if (use_general_pte) {
        // need to get an initial guess for volume fractions
        set_vfrac = tl.AddTask(update | radial_return | electron_entropy_to_energy,
                               Hydro::GuessCellVolumeFractions, mu0.get());
      }

      // set required fields before communication
      auto int_derived =
          tl.AddTask(update | radial_return | electron_entropy_to_energy | set_vfrac,
                     PreCommFillDerived<MeshData<Real>>, mu0.get());

      // communicate boundaries
      auto set_bc = parthenon::AddBoundaryExchangeTasks(int_derived | max_signal, tl, mu0,
                                                        pmesh->multilevel);

      // fill in thermodynamic state everywhere
      auto derive = tl.AddTask(set_bc, FillDerived<MeshData<Real>>, mu0.get());
    } // npartitions
  } // nstages

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection RiotDriver::RiotPostStepTasks
//! \brief
TaskCollection RiotDriver::RiotPostStepTasks() {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto timestep = tl.AddTask(none, EstimateTimestep<MeshData<Real>>, mu0.get());
    auto refine = timestep;
    if (pmesh->adaptive) {
      refine =
          tl.AddTask(timestep, parthenon::Refinement::Tag<MeshData<Real>>, mu0.get());
    }
    auto dealloc = refine;
    if (sparse_dealloc) {
      dealloc = tl.AddTask(refine, SparseDealloc, mu0.get());
    }
    tl.AddTask(dealloc, &RiotDriver::SetBlockCosts, this, mu0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RiotDriver::SetBlockCosts
//! \brief
TaskStatus RiotDriver::SetBlockCosts(MeshData<Real> *md) {
  if (!(do_hydro)) return TaskStatus::complete;
  const auto &all_mats =
      pmesh->packages.Get("materials")->Param<std::vector<std::string>>("all_mats");
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &mbd = md->GetBlockData(b);
    int nmats = 0;
    for (const auto &mat : all_mats) {
      nmats += mbd->IsAllocated(mat);
    }
    // empirical cost model
    auto pmb = mbd->GetBlockPointer();
    pmb->SetCostForLoadBalancing(1.0); // BlockCost(nmats));
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::SetDiagnostics
//! \brief
void RiotDriver::SetDiagnostics() {
  if (!(do_hydro) || fixed_fluid) return;
  const auto &all_mats =
      pmesh->packages.Get("materials")->Param<std::vector<std::string>>("all_mats");
  int ih = nmat_hist.size();
  nmat_hist.resize(ih + all_mats.size(), 0);
  mem_usage.push_back(pmesh->GetBufferPoolSizeInBytes());
  max_rss.push_back(GetMaxRss());
  for (auto &pmb : pmesh->block_list) {
    auto &mbd = pmb->meshblock_data.Get();
    int nmats = 0;
    for (const auto &mat : all_mats) {
      nmats += mbd->IsAllocated(mat);
    }
    nmat_hist[std::max(0, ih + nmats - 1)]++;
    mem_usage.back() += pmb->ReportMemUsage();
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::ReportBlockHistogram
//! \brief
void RiotDriver::ReportBlockHistogram() {
  if (!(do_hydro) || fixed_fluid) return;
#ifdef MPI_PARALLEL
  MPI_Reduce(parthenon::Globals::my_rank == 0 ? MPI_IN_PLACE : nmat_hist.data(),
             nmat_hist.data(), static_cast<int>(nmat_hist.size()), MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
#endif // MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    const auto &all_mats =
        pmesh->packages.Get("materials")->Param<std::vector<std::string>>("all_mats");
    const int nmats = all_mats.size();
    const int ncycles = nmat_hist.size() / nmats;
    std::ofstream hist;
    hist.open("block_histogram.txt", std::ios::out | std::ios::trunc);
    int i = 0;
    for (int n = 0; n < ncycles; n++) {
      hist << n << " ";
      for (int m = 0; m < nmats; m++) {
        hist << nmat_hist[i++] << " ";
      }
      hist << std::endl;
    }
    hist.close();
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::OutputDownstreamCycleDiagnostics
//! \brief
void RiotDriver::OutputDownstreamCycleDiagnostics() {
  try {
    auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
    std::cout << " lin_sol_iters=" << std::setw(2)
              << pkg->Param<int>("step_solver_iterations");
    std::cout << " newt_iters=" << std::setw(2)
              << pkg->Param<int>("step_newt_iterations");
  } catch (...) {
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::ReportMemUsage
//! \brief
void RiotDriver::ReportMemUsage() {
  if (!(do_hydro) || fixed_fluid) return;
#ifdef MPI_PARALLEL
  std::vector<uint64_t> min_mem(mem_usage.size());
  std::vector<uint64_t> max_mem(mem_usage.size());
  std::vector<uint64_t> tot_mem(mem_usage.size());
  std::vector<uint64_t> max_rusage(max_rss.size());
  MPI_Reduce(mem_usage.data(), min_mem.data(), mem_usage.size(), MPI_UINT64_T, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(mem_usage.data(), max_mem.data(), mem_usage.size(), MPI_UINT64_T, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(mem_usage.data(), tot_mem.data(), mem_usage.size(), MPI_UINT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(max_rss.data(), max_rusage.data(), max_rss.size(), MPI_UINT64_T, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (parthenon::Globals::my_rank == 0) {
    const int ncycles = mem_usage.size();
    std::ofstream f;
    f.open("mem_usage.txt", std::ios::out | std::ios::trunc);
    for (int n = 0; n < ncycles; n++) {
      f << n << " " << min_mem[n] << " "
        << " " << max_mem[n] << " " << (1.0 * tot_mem[n]) / parthenon::Globals::nranks
        << " " << max_rusage[n] << std::endl;
    }
    f.close();
  }
#else
  const int ncycles = mem_usage.size();
  std::ofstream f;
  f.open("mem_usage.txt", std::ios::out | std::ios::trunc);
  for (int n = 0; n < ncycles; n++) {
    f << n << " " << mem_usage[n] << " " << max_rss[n] << std::endl;
  }
  f.close();
#endif // MPI_PARALLEL
}

//----------------------------------------------------------------------------------------
//! \fn  uint64_t RiotDriver::GetMaxRss
//! \brief
uint64_t RiotDriver::GetMaxRss() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024;
}

} // namespace riot
