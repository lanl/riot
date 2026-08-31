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

// Local Includes
#include "diagnostics/diagnostics.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "ionization/ionization.hpp"
#include "laser/laser.hpp"
#include "levelsets/levelsets.hpp"
#include "materials/materials.hpp"
#include "mix/mix.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "plugins.hpp"
#include "prescribed_sources/prescribed_sources.hpp"
#include "radiation_diffusion/multigroup_diffusion.hpp"
#include "radiation_transport/algorithms/explicit.hpp"
#include "radiation_transport/algorithms/jacobi.hpp"
#include "riot_driver.hpp"
#include "riot_pgen/pgen.hpp"
#include "riot_pgen/regions.hpp"
#include "scalars/scalars.hpp"
#include "strength/strength.hpp"
#include "tnburn/tnburn.hpp"
#include "tracers/tracers.hpp"
#include "variables.hpp"

namespace riot {

// need this because it's static
std::vector<TaskCollectionFnPtr> RiotDriver::OperatorSplitTasks;

//----------------------------------------------------------------------------------------
//! \fn  void RiotDriver::RegisterPgens
//  \brief Pgen registration
void RiotDriver::RegisterPgens() {
  riot::RegisterAllRiotProblems();
  // Add more registration calls below if needed. e.g.,
  // RIOT_PROBLEM(my_cool_pgen);
}

//----------------------------------------------------------------------------------------
//! \fn  Packages_t RiotDriver::ProcessPackages
//  \brief Package initializer for Parthenon
Packages_t RiotDriver::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  Packages_t packages;

  // a hack to override parthenon default before it's too late
  pin->SetString("loadbalancing", "balancer", "manual");

  // Custom Metadata flags
  parthenon::MetadataFlag MetadataOperatorSplit =
      parthenon::Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);

  // riot StateDescriptor
  packages.Add(std::make_shared<StateDescriptor>("riot"));
  auto riot = packages.Get("riot");

  // determine input file specified physics
  const bool do_hydro =
      pin->GetOrAddBoolean("physics", "hydro", true, "Enable hydrodynamics");
  const bool do_strength =
      pin->GetOrAddBoolean("physics", "strength", false, "Enable material strength");
  const bool do_mix =
      pin->GetOrAddBoolean("physics", "mix", false, "Enable BHR RANS subgrid model");
  const bool do_tn =
      pin->GetOrAddBoolean("physics", "tn", false, "Enable thermonuclear burn");
  const bool do_scalars =
      pin->GetOrAddBoolean("physics", "scalars", false, "enable passive scalars");
  const bool fixed_fluid = pin->GetOrAddBoolean("physics", "fixed_fluid", false,
                                                "Freeze the fluid background");
  const bool do_lasers = pin->GetOrAddBoolean("physics", "lasers", false,
                                              "Laser ray tracing and energy deposition");
  const bool do_multigroup_diffusion =
      pin->GetOrAddBoolean("physics", "multigroup_diffusion", false);
  const bool do_radiation_transport = pin->GetOrAddBoolean(
      "physics", "radiation_transport", false, "Enable thermal radiation transport");
  const bool do_levelsets = pin->GetOrAddBoolean("physics", "levelsets", false,
                                                 "Enable levelsets (EXPERIMENTAL)");
  const bool do_gravity = pin->GetOrAddBoolean(
      "physics", "gravity", false, "Enable a constant gravitational acceleration");
  const bool do_ionization =
      pin->GetOrAddBoolean("physics", "ionization", false, "Enable partial ionization");
  const bool do_prescribed_sources =
      pin->GetOrAddBoolean("physics", "prescribed_sources", false,
                           "Enable prescribed sources as a function of time");
  const bool do_tracers =
      pin->GetOrAddBoolean("physics", "tracers", false, "Enable tracer particles");

  // check for compatibility
  if (do_strength) PARTHENON_REQUIRE(do_hydro, "Strength requires hydro.")
  if (do_mix) PARTHENON_REQUIRE(do_hydro, "Mix requires hydro.")
  if (do_tn) PARTHENON_REQUIRE(do_hydro, "TN requires hydro.")
  if (do_scalars) PARTHENON_REQUIRE(do_hydro, "Scalars currently require hydro.")
  if (fixed_fluid) PARTHENON_REQUIRE(do_hydro, "Fixed fluid config. requires hydro.")
  if (do_multigroup_diffusion)
    PARTHENON_REQUIRE(do_hydro, "Multigroup diffusion requires hydro.")
  if (do_radiation_transport)
    PARTHENON_REQUIRE(do_hydro, "Radiation transport requires hydro.")
  PARTHENON_REQUIRE(
      !(do_multigroup_diffusion && do_radiation_transport),
      "Multigroup diffusion and radiation transport cannot both be enabled.")
  PARTHENON_REQUIRE(!(do_ionization && do_radiation_transport),
                    "Radiation transport with ionization is not yet supported.")
  if (do_levelsets) PARTHENON_REQUIRE(do_hydro, "Levelsets config. requires hydro.")
  if (do_gravity) PARTHENON_REQUIRE(do_hydro, "Gravity config. requires hydro.")
  if (do_ionization) PARTHENON_REQUIRE(do_hydro, "Ionization requires hydro.");
  if (do_lasers) PARTHENON_REQUIRE(do_ionization, "Lasers requires ionization.");
  if (do_prescribed_sources)
    PARTHENON_REQUIRE(do_hydro, "Prescribed sources requires hydro.");

  // add options to params
  riot->AddParam("do_hydro", do_hydro);
  riot->AddParam("do_strength", do_strength);
  riot->AddParam("do_mix", do_mix);
  riot->AddParam("do_tn", do_tn);
  riot->AddParam("do_scalars", do_scalars);
  riot->AddParam("do_levelsets", do_levelsets);
  riot->AddParam("do_lasers", do_lasers);
  riot->AddParam("do_gravity", do_gravity);
  riot->AddParam("curvilinear", parthenon::IsCoord<parthenon::UniformSpherical>() ||
                                    parthenon::IsCoord<parthenon::UniformCylindrical>());
  riot->AddParam("fixed_fluid", fixed_fluid);
  riot->AddParam("do_multigroup_diffusion", do_multigroup_diffusion);
  riot->AddParam("do_radiation_transport", do_radiation_transport);
  riot->AddParam("do_ionization", do_ionization);
  riot->AddParam("do_prescribed_sources", do_prescribed_sources);
  riot->AddParam("do_tracers", do_tracers);

  // determine riot verbosity
  const bool verbose = pin->GetOrAddBoolean("riot", "verbose", true);
  riot->AddParam("verbose", verbose);

  // FPE trapping
  const bool trap_fpes = pin->GetOrAddBoolean("riot", "trap_fpes", false);
  riot->AddParam("trap_fpes", trap_fpes);

  // add a field to track which blocks are active
  bool sparse_physics = pin->GetOrAddBoolean("physics", "sparse_physics", true);
  if (do_multigroup_diffusion || do_radiation_transport || do_ionization) {
    if (sparse_physics)
      PARTHENON_WARN(
          "sparse_physics being disabled because a global solver is being used.");
    sparse_physics = false;
    pin->SetBoolean("physics", "sparse_physics", sparse_physics);
  }
  riot->AddParam("sparse_physics", sparse_physics);
  const Real sparse_physics_threshold =
      pin->GetOrAddReal("physics", "sparse_physics_threshold", 1.e-12);
  riot->AddParam("sparse_physics_threshold", sparse_physics_threshold);
  Metadata m({Metadata::Cell, Metadata::OneCopy, Metadata::Sparse, Metadata::FillGhost,
              Metadata::ForceAllocOnNewBlocks});
  if (sparse_physics) {
    m.SetSparseThresholds(sparse_physics_threshold, 0.01 * sparse_physics_threshold, 0.0);
  } else {
    m.SetSparseThresholds(-1.0, -1.0, 0.0);
  }
  // TODO(JCD): make a special sentinel sparse id for this field
  riot->AddSparsePool(ccbulk::cell_delta::name(), m, ccbulk::cell_delta::name(),
                      std::vector<int>{cell_delta_id});

  // call package initializers here and enroll operator split tasks
  if (do_hydro) {
    auto mat_pkg = Materials::Initialize(pin.get());
    packages.Add(mat_pkg);
    packages.Add(Hydro::Initialize(pin.get(), mat_pkg.get()));
  }
  if (do_strength)
    packages.Add(Strength::Initialize(pin.get(), packages.Get("materials").get()));
  if (do_mix) packages.Add(Mix::Initialize(pin.get()));
  if (do_tn) {
    auto isozaids =
        packages.Get("materials")->Param<std::vector<std::vector<int>>>("Isotope Zaids");
    packages.Add(TNBurn::Initialize(pin.get(), isozaids));
  }
  if (do_scalars) packages.Add(Scalars::Initialize(pin.get()));
  if (do_multigroup_diffusion) {
    auto materials = packages.Get("materials").get();
    if (do_ionization) {
      packages.Add(
          RadiationDiffusion::MultiGroup<ccbulk::electron_temperature>::Initialize(
              pin.get(), materials));
      OperatorSplitTasks.push_back(
          RadiationDiffusion::MultiGroup<ccbulk::electron_temperature>::Step);
    } else {
      packages.Add(RadiationDiffusion::MultiGroup<ccbulk::temperature>::Initialize(
          pin.get(), materials));
      OperatorSplitTasks.push_back(
          RadiationDiffusion::MultiGroup<ccbulk::temperature>::Step);
    }
  }
  if (do_radiation_transport) {
    auto materials = packages.Get("materials").get();
    const bool do_explicit = pin->GetOrAddBoolean(
        "radiation_transport", "do_explicit", false,
        "Enable explicit integration of thermal radiation transport");
    const bool do_jacobi = pin->GetOrAddBoolean(
        "radiation_transport", "do_jacobi", false,
        "Enable implicit integration of thermal radiation transport via Jacobi");
    PARTHENON_REQUIRE(((do_explicit ^ do_jacobi) && !(do_explicit && do_jacobi)),
                      "*One* radiation algorithm must be specified if do_radiation");
    riot->AddParam("do_explicit", do_explicit);
    riot->AddParam("do_jacobi", do_jacobi);
    if (do_explicit) {
      packages.Add(Explicit::Initialize(pin.get(), materials));
      OperatorSplitTasks.push_back(&Explicit::ExplicitTasks);
    } else if (do_jacobi) {
      packages.Add(Jacobi::Initialize(pin.get(), materials));
      OperatorSplitTasks.push_back(&Jacobi::JacobiTasks);
    }
  }
  if (do_levelsets) {
    packages.Add(Levelsets::Initialize(pin.get()));
    OperatorSplitTasks.push_back(Levelsets::Reinitialize);
  }
  if (do_lasers) packages.Add(Laser::Initialize(pin.get()));
  if (do_ionization) {
    packages.Add(Ionization::Initialize(pin.get()));
    OperatorSplitTasks.push_back(Ionization::ElectronIonCouplingStep);
    OperatorSplitTasks.push_back(
        Ionization::ConductionStep<Ionization::TransportSpecies::Electron>);
    OperatorSplitTasks.push_back(
        Ionization::ConductionStep<Ionization::TransportSpecies::Ion>);
  }
  if (do_prescribed_sources) {
    packages.Add(PrescribedSources::Initialize(pin.get()));
    OperatorSplitTasks.push_back(PrescribedSources::Step);
  }
  if (do_gravity) packages.Add(Gravity::Initialize(pin.get()));

  riot_plugins::Plugins::Initialize(pin.get(), packages);

  // Problem-specific package object
  packages.Add(ProblemPackage(pin.get()));

  // sparse deallocation
  bool sparse_dealloc = pin->GetOrAddBoolean("materials", "sparse_dealloc", true);
  riot->AddParam("sparse_dealloc", sparse_dealloc);

  // set FillDerived functions where necessary
  if (do_hydro) {
    riot->PreCommFillDerivedMesh = Multiphysics::FillInteriorDerived;
    riot->PreFillDerivedMesh = Multiphysics::PostCommsFillDerived;
  }

  // setup regions for initialization
  riot->AddParam("BlockInitData",
                 std::make_shared<region_pgen::BlockInitData>(pin.get()));

  // add any diagnostic packages
  diagnostics::AddDiagnostics(pin.get(), packages);

  // Tracers must be added AFTER all other packages so it can introspect their fields
  if (do_tracers) {
    packages.Add(Tracers::Initialize(pin.get(), packages));
    OperatorSplitTasks.push_back(Tracers::PushTracers);
  }

  return packages;
}

} // namespace riot
