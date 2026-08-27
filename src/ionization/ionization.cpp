//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

#include <memory>

#include <ports-of-call/robust_utils.hpp>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <utils/error_checking.hpp>

#include <singularity-eos/eos/eos.hpp>

#include "hydro/hydro.hpp"
#include "ionization.hpp"
#include "ionization_base.hpp"
#include "materials/materials.hpp"
#include "microphysics/eos_riot.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "strength/strength.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Ionization {

using RiotLimits::MAX_MATERIALS;

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Ionization::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  auto physics = std::make_shared<StateDescriptor>("ionization");
  Params &params = physics->AllParams();

  // Root find parameters
  const Real Te_root_tol =
      pin->GetOrAddReal("ionization", "root_tol", 1e-20,
                        "Root finding tolerance for electron temperature equilibrium");
  physics->AddParam("Te_root_tol", Te_root_tol);

  // switch for full ionization
  const bool fully_ionized = pin->GetOrAddBoolean("ionization", "fully_ionized", false);
  physics->AddParam("fully_ionized", fully_ionized);

  // switch for using electron entropy advection (assumes ideal electron gas)
  const bool advect_electron_entropy =
      pin->GetOrAddBoolean("ionization", "advect_electron_entropy", false,
                           "Solve a conservation law for the electron entropy density\n"
                           "instead of using the electron internal energy equation\n"
                           "with PedV source term");
  physics->AddParam("advect_electron_entropy", advect_electron_entropy);

  // ei-coupling
  const bool electron_ion_coupling =
      pin->GetOrAddBoolean("ionization", "electron_ion_coupling", false,
                           "Solve temperature relaxation between electrons and ions");
  physics->AddParam("electron_ion_coupling", electron_ion_coupling);
  const std::string electron_ion_coupling_model =
      pin->GetOrAddString("ionization", "electron_ion_coupling_model", "landau_spitzer",
                          std::vector<std::string>{"constant", "landau_spitzer"},
                          "Model for electron-ion coupling coefficient");
  physics->AddParam("electron_ion_coupling_model", electron_ion_coupling_model);

  // ei-coupling parameter
  const Real tau_ei = pin->GetOrAddReal("ionization", "tau_ei", 0e0);
  physics->AddParam("tau_ei", tau_ei);

  // electron thermal conduction
  const bool electron_thermal_conduction =
      pin->GetOrAddBoolean("ionization", "electron_thermal_conduction", false,
                           "Solve electron thermal diffusion eqution");
  physics->AddParam("electron_thermal_conduction", electron_thermal_conduction);
  const std::string electron_conductivity_model = pin->GetOrAddString(
      "ionization", "electron_conductivity_model", "spitzer_volume_average_arithmetic",
      std::vector<std::string>{"constant", "spitzer_volume_average_arithmetic",
                               "spitzer_volume_average_harmonic",
                               "spitzer_electron_number_density_average"},
      "Model to use for electron thermal conductivity");
  physics->AddParam("electron_conductivity_model", electron_conductivity_model);
  const std::string coulomb_logarithm =
      pin->GetOrAddString("ionization", "coulomb_logarithm", "brysk",
                          std::vector<std::string>{"basic", "brysk", "lee_moore", "bps"},
                          "Model to use for Coulomb logarithm");
  physics->AddParam("coulomb_logarithm", coulomb_logarithm);
  const Real electron_conductivity =
      pin->GetOrAddReal("ionization", "electron_conductivity", 0e0,
                        "Electron thermal conductivity if constant model is selected");
  physics->AddParam("electron_conductivity", electron_conductivity);

  // timestep control. options:
  // explicit: explicit diffusion timescale dt ~ (dx)^2 / D
  // relative: limit relative change in electron temperature
  std::string timestep_control =
      pin->GetOrAddString("ionization", "timestep_control", "relative",
                          "Conduction timestep control: 'explicit' or 'relative'.");
  physics->AddParam<>("timestep_control", timestep_control);

  Real T_scale_floor =
      pin->GetOrAddReal("ionization", "T_scale_floor", 1.0,
                        "Limit for temperature scale in 'relative' timestep control/");
  Real fractional_change_scale = pin->GetOrAddReal(
      "ionization", "fractional_change_scale", 0.1,
      "Allowable fractional change in electron temperature for conduction timestep.");
  physics->AddParam<>("fractional_change_scale", fractional_change_scale);
  physics->AddParam<>("T_scale_floor", T_scale_floor);

  // ion thermal conduction
  const bool ion_thermal_conduction =
      pin->GetOrAddBoolean("ionization", "ion_thermal_conduction", false,
                           "Solve ion thermal diffusion eqution");
  physics->AddParam("ion_thermal_conduction", ion_thermal_conduction);

  const std::string ion_conductivity_model =
      pin->GetOrAddString("ionization", "ion_conductivity_model", "braginskii",
                          std::vector<std::string>{"constant", "braginskii"},
                          "Model to use for ion thermal conductivity");
  physics->AddParam("ion_conductivity_model", ion_conductivity_model);

  const Real ion_conductivity =
      pin->GetOrAddReal("ionization", "ion_conductivity", 0e0,
                        "Ion thermal conductivity if constant model is selected");
  physics->AddParam("ion_conductivity", ion_conductivity);

  const Real zbar_floor =
      pin->GetOrAddReal("ionization", "zbar_floor", 1e-6,
                        "Minimum value for zbar to be used in microphysics "
                        "calculations including Spitzer-Harm conductivities.");
  physics->AddParam("zbar_floor", zbar_floor);

  const Real ion_number_density_floor =
      pin->GetOrAddReal("ionization", "ion_number_density_floor", 1e11,
                        "Minimum value for ion number density to be used in microphysics "
                        "calculations including Spitzer-Harm conductivities.");
  physics->AddParam("ion_number_density_floor", ion_number_density_floor);

  // plasma viscosity
  const bool plasma_viscosity =
      pin->GetOrAddBoolean("ionization", "plasma_viscosity", false,
                           "Add plasma viscous fluxes to momentum and energy equations");
  physics->AddParam("plasma_viscosity", plasma_viscosity);
  const std::string ion_viscosity_model =
      pin->GetOrAddString("ionization", "ion_viscosity_model", "fokker_planck_landau",
                          std::vector<std::string>{"constant", "fokker_planck_landau"},
                          "Model to use for ion viscosity");
  physics->AddParam("ion_viscosity_model", ion_viscosity_model);
  const Real ion_shear_viscosity =
      pin->GetOrAddReal("ionization", "ion_shear_viscosity", 1e0,
                        "Ion shear viscosity if constant model is selected");
  physics->AddParam("ion_shear_viscosity", ion_shear_viscosity);
  const Real ion_bulk_viscosity =
      pin->GetOrAddReal("ionization", "ion_bulk_viscosity", 0e0,
                        "Ion bulk viscosity if constant model is selected");
  physics->AddParam("ion_bulk_viscosity", ion_bulk_viscosity);

  using namespace parthenon::refinement_ops;
  using solver_t = parthenon::solvers::BiCGSTABSolver<LinearizedDiffusionEquation<delta>>;
  const std::string conduction_base_container = "conduction_base";
  const std::string conduction_u_container = "conduction_u";
  const std::string conduction_rhs_container = "conduction_rhs";

  auto bicg_solver = std::make_shared<solver_t>(
      conduction_base_container, conduction_u_container, conduction_rhs_container, pin,
      "ionization/linear_solver_params");
  physics->AddParam<>("MGBiCGSTABsolver", bicg_solver);

  // Store the previous timestep, since for 'relative' timestep control,
  // the updated timestep change is based on the size of the previous timestep.
  Real dt = -1.0;
  physics->AddParam<>("dt", dt, parthenon::Params::Mutability::Mutable);

  static const parthenon::MetadataFlag OperatorSplit =
      Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);

  auto m_delta = Metadata({Metadata::Cell, Metadata::Derived, Metadata::FillGhost,
                           Metadata::WithFluxes, Metadata::GMGRestrict,
                           Metadata::GMGProlongate, OperatorSplit});
  m_delta.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  physics->AddField<delta>(m_delta);

  auto m_Dcell =
      Metadata({Metadata::Derived, Metadata::OneCopy, Metadata::Cell, OperatorSplit});
  physics->AddField<Dcell>(m_Dcell);

  auto m_D = Metadata({Metadata::Derived, Metadata::OneCopy, Metadata::Face,
                       Metadata::GMGRestrict, Metadata::CellMemAligned, OperatorSplit});
  m_D.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  physics->AddField<D>(m_D);

  auto m_c = Metadata({Metadata::Derived, Metadata::OneCopy, Metadata::Cell,
                       Metadata::GMGRestrict, OperatorSplit});
  m_c.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  physics->AddField<diag_loc>(m_c);

  // JMM: electron internal energy by volume u_e is advected with the bulk
  // note that electron specific internal energy is
  // ue / rho_bulk,
  // implying the specific energy is specific to ION density!
  Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                         Metadata::FillGhost, Metadata::Advected, Metadata::WithFluxes});
  physics->AddField<ccbulk::electron_internal_energy>(m);

  // electron entropy density s_e is advected with the bulk
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::FillGhost, Metadata::Advected, Metadata::WithFluxes,
                Metadata::Conserved});
  physics->AddField<ccbulk::electron_entropy>(m);

  m = Metadata(
      {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  // JMM: electron temperature is computed in post-comms fill derived
  physics->AddField<ccbulk::electron_temperature>(m);

  // JMM: electron bulk number density. materials.cpp will also
  // provide per-material. Bulk is needed for things like the laser
  // package. Per-material is needed for the temperature coupling
  // timescale Cv/tau and for manual Spitzer opacity. It is CURRNETLY
  // not needed for the EOS, which incorporates ionization implicitly.
  // We may wish to change this in the future and pass in ionization
  // as a lambda.
  physics->AddField<ccbulk::electron_number_density>(m);
  physics->AddField<ccbulk::electron_pressure>(m);
  physics->AddField<ccbulk::electron_bulk_modulus>(m);
  physics->AddField<ccbulk::electron_gruneisen_parameter>(m);

  // ion shear viscosity... is this needed in ghosts? Yes but it is derived so
  // it should be calculated in ghosts not communicated. What it is calculated
  // _from_ must be available in ghosts however
  m = Metadata(
      {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  physics->AddField<ccbulk::ion_shear_viscosity>(m);

  // timestep controls
  auto m_no_ghost =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit});
  physics->AddField<temperature_old>(m);
  physics->AddField<temp_tstep_criterion>(m);

  // JMM: Ionization state can be computed after comms
  physics->FillDerivedMesh = FillDerivedIonization;

  physics->UserWorkBeforeLoopMesh = ConvertElectronEnergyToEntropyMesh;

  physics->EstimateTimestepMesh = EstimateTimestepMesh;

  // Source term dU/dt variables
  physics->RegisterMeshDataSubset(
      "dudt",
      RiotUtils::MakePackageDudtRequirements({ccbulk::electron_internal_energy::name()}));

  return physics;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskCollection Ionization::ElectronIonCouplingStep
//! \brief local electron-ion coupling/relaxation.
parthenon::TaskCollection ElectronIonCouplingStep(Mesh *pm, parthenon::SimTime &tm,
                                                  const Real dt) {

  parthenon::TaskCollection tc;
  parthenon::TaskID none(0);

  auto &options = pm->packages.Get("ionization");
  const bool do_electron_ion_coupling = options->Param<bool>("electron_ion_coupling");

  auto *dt_conduction = options->MutableParam<Real>("dt");
  *dt_conduction = dt;

  if (!do_electron_ion_coupling) return tc;

  const int num_partitions = pm->DefaultNumPartitions();
  parthenon::TaskRegion &tr = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &mdpart = pm->mesh_data.GetOrAdd("base", i);
    auto couple = tl.AddTask(none, ElectronIonEquilibration, mdpart.get(), dt);
    auto int_derived = tl.AddTask(
        couple, parthenon::Update::PreCommFillDerived<MeshData<Real>>, mdpart.get());

    namespace mdname = riot::container_names;
    auto &mu0 = pm->mesh_data.GetOrAdd(mdname::u0, i);
    bool multilevel = pm->multilevel;
    auto set_bc = parthenon::AddBoundaryExchangeTasks<parthenon::BoundaryType::any>(
        int_derived, tl, mu0, multilevel);
    auto derive =
        tl.AddTask(set_bc, parthenon::Update::FillDerived<MeshData<Real>>, mdpart.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::ElectronIonEquilibration
//! \brief
TaskStatus ElectronIonEquilibration(MeshData<Real> *md, Real dt) {
  using unit_system = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const bool do_electron_ion_coupling = options->Param<bool>("electron_ion_coupling");
  const std::string electron_ion_coupling_model =
      options->Param<std::string>("electron_ion_coupling_model");
  const Real tau_ei_const = options->Param<Real>("tau_ei");
  const std::string coulomb_logarithm = options->Param<std::string>("coulomb_logarithm");

  // pull out material eos
  auto &materials = pm->packages.Get("materials");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &electron_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  // pack up vars we need
  auto v = riot::MakePack<cm::rho, cm::lr_cache, cm::lT_cache, ccmat::rho, ccbulk::rho,
                          ccbulk::electron_temperature, ccbulk::temperature,
                          ccbulk::electron_number_density, cm::ionization_zbar,
                          ccbulk::electron_internal_energy, ccmat::internal_energy,
                          ccbulk::internal_energy, ccmat::volume_fraction>(md);

  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  CoulombLogarithmKind logKind = ParseCoulombLogarithmKind(coulomb_logarithm);
  bool is_constant = (electron_ion_coupling_model == "constant");

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        std::array<int, MAX_MATERIALS> eos_map;
        RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, b, k, j, i);
          ElectronIonEquilibrationOne(ion_eos, electron_eos, eos_map, lambda, dt,
                                      is_constant, tau_ei_const, logKind, v, b, k, j, i);
        });
      });

  return TaskStatus::complete;

} // ElectronIonCouplingStep

void FillDerivedIonization(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const bool do_plasma_viscosity = options->Param<bool>("plasma_viscosity");

  ComputeFreeElectronNumberDensity(md);
  if (do_plasma_viscosity) CalculatePlasmaViscosity(md);
}

//----------------------------------------------------------------------------------------
//! \fn  void Ionization::ComputeFreeElectronNumberDensity
//! \brief
void ComputeFreeElectronNumberDensity(MeshData<Real> *md) {
  using PortsOfCall::Robust::ratio;
  using unit_system = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const bool fully_ionized = options->Param<bool>("fully_ionized");

  // pull out material eos
  auto &materials = pm->packages.Get("materials");
  const auto &eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  // pack up vars we need
  auto v =
      riot::MakePack<cm::rho, ccmat::rho, ccbulk::electron_temperature,
                     ccbulk::electron_number_density, ccbulk::ionization_zbar,
                     ccmat::ionization_zbar, cm::ionization_zbar, ccmat::volume_fraction>(
          md);

  const int nblocks = v.GetNBlocks();
  if (nblocks < 1) return;

  // SWJ: this needs to be computed over entire mesh because thermal diffusion
  // coefficient is calculated in the ghost zones
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        std::array<int, MAX_MATERIALS> eos_map;
        RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

        // compute ionization state for each material
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto ne =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_number_density());
        for (int m = 0; m < nmat; m++) {
          auto cm_rho = RiotLoop::make_var_view(idx_range, v, cm::rho(m));
          auto ccmat_rho = RiotLoop::make_var_view(idx_range, v, ccmat::rho(m));
          auto cm_zbar = RiotLoop::make_var_view(idx_range, v, cm::ionization_zbar(m));
          auto ccmat_zbar =
              RiotLoop::make_var_view(idx_range, v, ccmat::ionization_zbar(m));
          auto &eosm = eos(eos_map[m]);
          const Real anuc = eosm.MeanAtomicMass();
          const Real one_over_mnuc = 1.0 / (anuc * unit_system::amu);
          const Real znuc = eosm.MeanAtomicNumber();
          RiotLoop::inner(idx_range, [&](const auto kji) {
            Real zbar = cm_zbar(kji);
            ComputeIonizationState(anuc, znuc, cm_rho(kji),
                                   pv(ccbulk::electron_temperature(), kji), zbar,
                                   fully_ionized);
            cm_zbar(kji) = zbar;
            ccmat_zbar(kji) = ccmat_rho(kji) * zbar;
            ne(kji) = (m == 0) ? (ccmat_zbar(kji) * one_over_mnuc)
                               : (ne(kji) + ccmat_zbar(kji) * one_over_mnuc);
          });
          if (v.Contains(b, ccbulk::ionization_zbar())) {
            idx_range.TeamBarrier();
            RiotLoop::inner(idx_range, [&](const auto kji) {
              pv(ccbulk::ionization_zbar(), kji) =
                  (m == 0) ? (ccmat_zbar(kji) * cm_zbar(kji) * one_over_mnuc)
                           : (pv(ccbulk::ionization_zbar(), kji) +
                              ccmat_zbar(kji) * cm_zbar(kji) * one_over_mnuc);
            });
          }
          idx_range.TeamBarrier();
        } // mats

        if (v.Contains(b, ccbulk::ionization_zbar())) {
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv(ccbulk::ionization_zbar(), kji) =
                ratio(pv(ccbulk::ionization_zbar(), kji), ne(kji));
          });
        }
      });

} // ComputeFreeElectronNumberDensity

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::CalculateElectronPDVWork
//! \brief
TaskStatus CalculateElectronPDVWork(MeshData<Real> *u0, MeshData<Real> *dudt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;

  auto v = riot::MakePack<ccbulk::electron_pressure, ccbulk::face_velocity>(u0);
  auto dv = riot::MakePack<ccbulk::electron_internal_energy>(dudt);
  const int nblocks = v.GetNBlocks();

  auto &options = pm->packages.Get("ionization");
  const bool advect_electron_entropy = options->Param<bool>("advect_electron_entropy");
  if (advect_electron_entropy) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, u0,
                                     parthenon::TopologicalElement::CC);
  // Memory offsets to the "plus" face neighbor in each direction (zero for collapsed
  // dimensions), used to reach the i+1/j+1/k+1 face velocity via flat pack-view indexing.
  auto di = idx_space.GetDelta(X1DIR);
  auto dj = idx_space.GetDelta(X2DIR);
  auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto pdv = RiotLoop::make_pack_view(idx_range, dv);

        // Compute velocity divergence via divergence theorem. Face velocities are read
        // through flat pack views (neighbor via MemoryOffset); face areas and cell
        // volume need (k, j, i), read once per cell from the cached coords.
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);
          const Real area1_lo = coords.template FaceArea<X1DIR>(k, j, i);
          const Real area1_hi = coords.template FaceArea<X1DIR>(k, j, i + 1);
          Real div = area1_lo * pv(ccbulk::face_velocity(0), kji) -
                     area1_hi * pv(ccbulk::face_velocity(0), kji + di); // -PdV: -1 moved

          if (ndim > 1) {
            const Real area2_lo = coords.template FaceArea<X2DIR>(k, j, i);
            const Real area2_hi = coords.template FaceArea<X2DIR>(k, j + 1, i);
            div += area2_lo * pv(ccbulk::face_velocity(4), kji) -
                   area2_hi * pv(ccbulk::face_velocity(4), kji + dj);
          }

          if (ndim > 2) {
            const Real area3_lo = coords.template FaceArea<X3DIR>(k, j, i);
            const Real area3_hi = coords.template FaceArea<X3DIR>(k + 1, j, i);
            div += area3_lo * pv(ccbulk::face_velocity(8), kji) -
                   area3_hi * pv(ccbulk::face_velocity(8), kji + dk);
          }

          pdv(ccbulk::electron_internal_energy(), kji) =
              pv(ccbulk::electron_pressure(), kji) * div / coords.CellVolume(k, j, i);
        });
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Ionization::ConvertElectronEnergyEntropyWork
//! \brief
void ConvertElectronEnergyEntropyWork(MeshData<Real> *md, const EntropyDirection dir,
                                      const IndexDomain domain) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto riot = pm->packages.Get("riot");
  const bool do_ionization = riot->Param<bool>("do_ionization");
  if (!do_ionization) return;

  auto &options = pm->packages.Get("ionization");
  const bool advect_electron_entropy = options->Param<bool>("advect_electron_entropy");
  // if ((!advect_electron_entropy) && (dir == EntropyDirection::ToEnergy)) return;
  if (!advect_electron_entropy) return;

  auto v =
      riot::MakePack<ccbulk::electron_pressure, ccbulk::rho, ccbulk::electron_entropy,
                     ccbulk::electron_internal_energy, ccmat::rho,
                     ccbulk::electron_temperature, ccbulk::electron_bulk_modulus,
                     ccbulk::electron_gruneisen_parameter>(md);
  const int nblocks = v.GetNBlocks();

  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(domain, 0, nblocks, md, parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto rho = RiotLoop::make_var_view(idx_range, v, ccbulk::rho());

        // Compute bulk rho by summing material densities
        for (int m = 0; m < nmat; m++) {
          auto rmat = RiotLoop::make_var_view(idx_range, v, ccmat::rho(m));
          RiotLoop::inner(idx_range, [&](const auto kji) {
            rho(kji) = (m == 0) ? rmat(kji) : (rho(kji) + rmat(kji));
          });
          idx_range.TeamBarrier();
        }

        auto s_e = RiotLoop::make_var_view(idx_range, v, ccbulk::electron_entropy());
        auto uu_e =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_internal_energy());
        auto p_e = RiotLoop::make_var_view(idx_range, v, ccbulk::electron_pressure());
        auto bmod_e =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_bulk_modulus());
        auto Gamma_e =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_gruneisen_parameter());

        if (dir == EntropyDirection::ToEntropy) {
          // from electron internal energy density to electron entropy density
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real gam = bmod_e(kji) / (p_e(kji) + 1e-16);
            s_e(kji) = uu_e(kji) * Gamma_e(kji) / std::pow(rho(kji), gam); // specific
            s_e(kji) *= rho(kji);                                          // density
          });
        } else { // from electron entropy density to electron internal energy density
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real gam = bmod_e(kji) / (p_e(kji) + 1e-16);
            s_e(kji) /= rho(kji); // convert to specific
            uu_e(kji) = s_e(kji) * std::pow(rho(kji), gam) / Gamma_e(kji);
          });
        }
      });
} // ConvertElectronEnergyEntropyWork

//----------------------------------------------------------------------------------------
//! \fn  void Ionization::ConvertElectronEnergyToEntropyMesh
//! \brief
void ConvertElectronEnergyToEntropyMesh(Mesh *pm, ParameterInput *pin,
                                        parthenon::SimTime &tm) {
  auto md = pm->mesh_data.Get();
  ConvertElectronEnergyEntropyWork(md.get(), EntropyDirection::ToEntropy,
                                   IndexDomain::entire);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::ConvertElectronEnergyToEntropy
//! \brief
TaskStatus ConvertElectronEnergyToEntropy(MeshData<Real> *u0, const IndexDomain domain) {
  ConvertElectronEnergyEntropyWork(u0, EntropyDirection::ToEntropy, domain);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::ConvertElectronEntropyToEnergy
//! \brief
TaskStatus ConvertElectronEntropyToEnergy(MeshData<Real> *u0, const IndexDomain domain) {
  ConvertElectronEnergyEntropyWork(u0, EntropyDirection::ToEnergy, domain);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskCollection Ionization::ConductionStep
//! \brief thermal conduction
template <TransportSpecies Species>
parthenon::TaskCollection ConductionStep(Mesh *pmesh, parthenon::SimTime &tm,
                                         const Real dt) {

  parthenon::TaskCollection tc;
  parthenon::TaskID none(0);

  auto &pkg = pmesh->packages.Get("ionization");

  if (Species == TransportSpecies::Electron) {
    const bool electron_thermal_conduction =
        pkg->Param<bool>("electron_thermal_conduction");
    if (!electron_thermal_conduction) return tc;
  } else {
    const bool ion_thermal_conduction = pkg->Param<bool>("ion_thermal_conduction");
    if (!ion_thermal_conduction) return tc;
  }

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const std::string conduction_base_container = "conduction_base";
  const std::string conduction_u_container = "conduction_u";
  const std::string conduction_rhs_container = "conduction_rhs";

  using solver_t = parthenon::solvers::BiCGSTABSolver<LinearizedDiffusionEquation<delta>>;
  using conduction_aux_vars_tl = parthenon::TypeList<D, diag_loc, delta>;

  using all_vars_tl = parthenon::concatenate_type_lists_t<
      parthenon::TypeList<ccbulk::electron_temperature, ccbulk::temperature>,
      conduction_aux_vars_tl>;

  std::shared_ptr<solver_t> bicgstab_solver =
      pkg->Param<std::shared_ptr<solver_t>>("MGBiCGSTABsolver");

  const int num_partitions = pmesh->DefaultNumPartitions();
  parthenon::TaskRegion &tr = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &md_base = pmesh->mesh_data.GetOrAdd("base", i);

    // core container used by solver
    auto &md = pmesh->mesh_data.Add(conduction_base_container, md_base,
                                    parthenon::GetNames<all_vars_tl>());
    auto &md_u = pmesh->mesh_data.Add(conduction_u_container, md, {delta::name()});

    // holds solution
    auto &md_rhs = pmesh->mesh_data.Add(conduction_rhs_container, md, {delta::name()});

    parthenon::TaskID compute_conductivity;
    if constexpr (Species == TransportSpecies::Electron) {
      compute_conductivity =
          tl.AddTask(none, CalculateElectronThermalDiffusionCoefficient, md_base.get(),
                     md.get(), dt);
    } else {
      compute_conductivity = tl.AddTask(none, CalculateIonThermalDiffusionCoefficient,
                                        md_base.get(), md.get(), dt);
    }

    auto set_local = tl.AddTask(none, SetLocal<Species>, md_base.get(), md.get());

    // add flux divergence to rhs
    parthenon::TaskID calc_flux;
    if constexpr (Species == TransportSpecies::Electron) {
      calc_flux = tl.AddTask(set_local | compute_conductivity,
                             CalculateRHSFluxes<ccbulk::electron_temperature, D, delta>,
                             md, md_rhs);
    } else {
      calc_flux =
          tl.AddTask(set_local | compute_conductivity,
                     CalculateRHSFluxes<ccbulk::temperature, D, delta>, md, md_rhs);
    }

    auto zero_rhs = tl.AddTask(
        calc_flux, parthenon::solvers::utils::SetToZero<parthenon::TypeList<delta>>,
        md_rhs);
    auto flx_res = AddFluxContribution<delta>(tl, zero_rhs, md, md_rhs);

    // Solve the matrix equation for \delta T = T^{n+1} - T^{n}
    auto init_solve = bicgstab_solver->AddSetupTasks(tl, flx_res, i, pmesh);
    auto solve = bicgstab_solver->AddTasks(tl, init_solve, i, pmesh);

    // Update electron temperature and energy
    auto update = tl.AddTask(solve, UpdateStateFromConduction<Species>, md_base.get(),
                             md_u.get(), dt);

    auto int_derived = tl.AddTask(
        update, parthenon::Update::PreCommFillDerived<MeshData<Real>>, md_base.get());

    namespace mdname = riot::container_names;
    auto &mu0 = pmesh->mesh_data.GetOrAdd(mdname::u0, i);
    bool multilevel = pmesh->multilevel;
    auto set_bc = parthenon::AddBoundaryExchangeTasks<parthenon::BoundaryType::any>(
        int_derived, tl, mu0, multilevel);
    auto derive =
        tl.AddTask(set_bc, parthenon::Update::FillDerived<MeshData<Real>>, md_base.get());
  }

  return tc;
} // ConductionStep

template parthenon::TaskCollection
ConductionStep<TransportSpecies::Electron>(Mesh *pmesh, parthenon::SimTime &tm,
                                           const Real dt);

template parthenon::TaskCollection
ConductionStep<TransportSpecies::Ion>(Mesh *pmesh, parthenon::SimTime &tm, const Real dt);

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::CalculateElectronThermalDiffusionCoefficient
//! \brief Electron thermal diffusion coefficient
TaskStatus CalculateElectronThermalDiffusionCoefficient(MeshData<Real> *md,
                                                        MeshData<Real> *md_out,
                                                        const Real dt) {
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const Real k_e_const = options->Param<Real>("electron_conductivity");
  const std::string electron_conductivity_model =
      options->Param<std::string>("electron_conductivity_model");
  const ElectronThermalConductivityModel electron_model =
      ElectronConductivityModelEnumFromString(electron_conductivity_model);
  const std::string coulomb_logarithm = options->Param<std::string>("coulomb_logarithm");
  const Real zbar_floor = options->Param<Real>("zbar_floor");
  const Real ion_number_density_floor = options->Param<Real>("ion_number_density_floor");

  auto &materials = pm->packages.Get("materials");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  auto v = riot::MakePack<cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccbulk::rho,
                          ccbulk::electron_temperature, ccbulk::temperature,
                          cm::ionization_zbar, ccmat::volume_fraction,
                          ccbulk::electron_number_density, Ionization::Dcell>(md);
  auto v_out = riot::MakePack<Ionization::D>(md_out);
  const int nblocks = v.GetNBlocks();
  const int ndim = pm->ndim;

  CoulombLogarithmKind logKind = ParseCoulombLogarithmKind(coulomb_logarithm);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md, TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        std::array<int, MAX_MATERIALS> eos_map;
        RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto d_cell = RiotLoop::make_var_view(idx_range, v, Ionization::Dcell());
        auto tele = RiotLoop::make_var_view(idx_range, v, ccbulk::electron_temperature());
        auto tion = RiotLoop::make_var_view(idx_range, v, ccbulk::temperature());
        auto ccbulk_ne =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_number_density());

        // Compute conductivity for each material and accumulate
        for (int m = 0; m < nmat; ++m) {
          auto rhom = RiotLoop::make_var_view(idx_range, v, cm::rho(m));
          auto fvm = RiotLoop::make_var_view(idx_range, v, ccmat::volume_fraction(m));
          auto zbarm = RiotLoop::make_var_view(idx_range, v, cm::ionization_zbar(m));
          const Real mu_m = ion_eos(eos_map[m]).MeanAtomicMass();
          const Real mi = CouplingModelConstants::amu * mu_m;

          if (electron_model ==
              ElectronThermalConductivityModel::SpitzerVolumeAverageHarmonic) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ni = rhom(kji) / mi;
              const Real nem = rhom(kji) * zbarm(kji);
              const Real k_mat =
                  SpitzerHarmConductivity(tele(kji), tion(kji), nem, ni, mi, zbarm(kji),
                                          logKind, zbar_floor, ion_number_density_floor);
              d_cell(kji) =
                  (m == 0) ? (fvm(kji) / k_mat) : (d_cell(kji) + fvm(kji) / k_mat);
            });
          } else if (electron_model ==
                     ElectronThermalConductivityModel::SpitzerVolumeAverageArithmetic) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ni = rhom(kji) / mi;
              const Real nem = zbarm(kji) * ni;
              const Real k_mat =
                  SpitzerHarmConductivity(tele(kji), tion(kji), nem, ni, mi, zbarm(kji),
                                          logKind, zbar_floor, ion_number_density_floor);
              d_cell(kji) = (m == 0) ? (dt * fvm(kji) * k_mat)
                                     : (d_cell(kji) + dt * fvm(kji) * k_mat);
            });
          } else if (electron_model == ElectronThermalConductivityModel::
                                           SpitzerElectronNumberDensityAverage) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ni = rhom(kji) / mi;
              const Real nem = zbarm(kji) * ni;
              const Real k_mat =
                  SpitzerHarmConductivity(tele(kji), tion(kji), nem, ni, mi, zbarm(kji),
                                          logKind, zbar_floor, ion_number_density_floor);
              d_cell(kji) =
                  (m == 0) ? (dt * fvm(kji) * nem * k_mat / ccbulk_ne(kji))
                           : (d_cell(kji) + dt * fvm(kji) * nem * k_mat / ccbulk_ne(kji));
            });
          } else { // constant
            RiotLoop::inner(idx_range, [&](const auto kji) {
              d_cell(kji) = (m == 0) ? (dt * fvm(kji) * k_e_const) // folding dt in here
                                     : (d_cell(kji) + dt * fvm(kji) * k_e_const);
            });
          } // electron conductivity model
          idx_range.TeamBarrier();
        } // nmat

        // Finalize harmonic average
        if (electron_model ==
            ElectronThermalConductivityModel::SpitzerVolumeAverageHarmonic) {
          RiotLoop::inner(idx_range,
                          [&](const auto kji) { d_cell(kji) = dt / d_cell(kji); });
        }
      });

  // --- Average kcell to faces. Dcell is cell-centered while D lives on faces, and the
  // stencil reaches the low-side neighbor, so this loop uses (k, j, i).
  for (int dim = 0; dim < ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    const int ioff = dim == 0;
    const int joff = dim == 1;
    const int koff = dim == 2;
    using lt = RiotUtils::LoopType<>;
    auto idx_space_face = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, te);
    RiotLoop::outer(
        idx_space_face, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            v_out(b, te, Ionization::D(), k, j, i) =
                0.5 * (v(b, Ionization::Dcell(), k, j, i) +
                       v(b, Ionization::Dcell(), k - koff, j - joff, i - ioff));
          });
        });
  }
  return TaskStatus::complete;
} // CalculateElectronThermalDiffusionCoefficient

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::CalculateIonThermalDiffusionCoefficient
//! \brief Sets the local diagonal for the solve and stores the old temperature
//! for the timestep control.
//! Ion thermal diffusion coefficient
TaskStatus CalculateIonThermalDiffusionCoefficient(MeshData<Real> *md,
                                                   MeshData<Real> *md_out,
                                                   const Real dt) {
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const Real k_i_const = options->Param<Real>("ion_conductivity");
  const std::string ion_conductivity_model =
      options->Param<std::string>("ion_conductivity_model");
  const IonThermalConductivityModel ion_model =
      IonConductivityModelEnumFromString(ion_conductivity_model);
  const std::string coulomb_logarithm = options->Param<std::string>("coulomb_logarithm");
  const Real zbar_floor = options->Param<Real>("zbar_floor");
  const Real ion_number_density_floor = options->Param<Real>("ion_number_density_floor");

  auto &materials = pm->packages.Get("materials");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  auto v = riot::MakePack<cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccbulk::rho,
                          ccbulk::electron_temperature, ccbulk::temperature,
                          cm::ionization_zbar, ccmat::volume_fraction,
                          ccbulk::electron_number_density, Ionization::Dcell>(md);
  auto v_out = riot::MakePack<Ionization::D>(md_out);
  const int nblocks = v.GetNBlocks();
  const int ndim = pm->ndim;

  CoulombLogarithmKind logKind = ParseCoulombLogarithmKind(coulomb_logarithm);

  // ion thermal conductivities
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md, TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        std::array<int, MAX_MATERIALS> eos_map;
        RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

        auto d_cell = RiotLoop::make_var_view(idx_range, v, Ionization::Dcell());
        auto tion = RiotLoop::make_var_view(idx_range, v, ccbulk::temperature());
        auto tele = RiotLoop::make_var_view(idx_range, v, ccbulk::electron_temperature());

        for (int m = 0; m < nmat; ++m) {
          auto rhom = RiotLoop::make_var_view(idx_range, v, cm::rho(m));
          auto fvm = RiotLoop::make_var_view(idx_range, v, ccmat::volume_fraction(m));
          auto zbarm = RiotLoop::make_var_view(idx_range, v, cm::ionization_zbar(m));
          const Real mu_m = ion_eos(eos_map[m]).MeanAtomicMass();
          const Real mi = CouplingModelConstants::amu * mu_m;

          if (ion_model == IonThermalConductivityModel::Braginskii) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ni = rhom(kji) / mi;
              const Real nem = zbarm(kji) * ni;
              const Real k_mat =
                  BraginskiiConductivity(tele(kji), nem, tion(kji), ni, mi, zbarm(kji),
                                         zbar_floor, ion_number_density_floor);
              d_cell(kji) = (m == 0) ? (fvm(kji) * dt * k_mat) // folding dt in here
                                     : (d_cell(kji) + fvm(kji) * dt * k_mat);
            });
          } else { // constant
            RiotLoop::inner(idx_range, [&](const auto kji) {
              d_cell(kji) = (m == 0) ? (fvm(kji) * dt * k_i_const) // folding dt in here
                                     : (d_cell(kji) + fvm(kji) * dt * k_i_const);
            });
          }
          idx_range.TeamBarrier();
        }
      });

  // --- Average kcell to faces. Dcell is cell-centered while D lives on faces, and the
  // stencil reaches the low-side neighbor, so this loop uses (k, j, i).
  for (int dim = 0; dim < ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    const int ioff = dim == 0;
    const int joff = dim == 1;
    const int koff = dim == 2;
    using lt = RiotUtils::LoopType<>;
    auto idx_space_face = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, te);
    RiotLoop::outer(
        idx_space_face, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            v_out(b, te, Ionization::D(), k, j, i) =
                0.5 * (v(b, Ionization::Dcell(), k, j, i) +
                       v(b, Ionization::Dcell(), k - koff, j - joff, i - ioff));
          });
        });
  }
  return TaskStatus::complete;
} // CalculateIonThermalDiffusionCoefficient

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::SetLocal
//! \brief
template <TransportSpecies Species>
TaskStatus SetLocal(MeshData<Real> *md, MeshData<Real> *md_out) {
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");

  auto &materials = pm->packages.Get("materials");
  const auto &electron_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  auto v = riot::MakePack<cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccbulk::rho,
                          ccbulk::electron_temperature, ccbulk::temperature,
                          cm::ionization_zbar, ccmat::volume_fraction,
                          Ionization::temperature_old>(md);
  auto v_out = riot::MakePack<Ionization::diag_loc>(md_out);
  const int nblocks = v.GetNBlocks();

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md, TE::CC);
  if (Species == TransportSpecies::Electron) {
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const int nmat = v.GetSize(b, ccmat::rho());
          std::array<int, MAX_MATERIALS> eos_map;
          RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real tele = v(b, ccbulk::electron_temperature(), k, j, i);
            Real rhocve_sum = 0.0;
            RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, b, k, j, i);
            for (int m = 0; m < nmat; ++m) {
              auto &eose = electron_eos(eos_map[m]);
              const Real rhom = v(b, cm::rho(m), k, j, i);
              const Real fvm = v(b, ccmat::volume_fraction(m), k, j, i);
              const Real cvem =
                  eose.SpecificHeatFromDensityTemperature(rhom, tele, lambda[m]);

              rhocve_sum += rhom * cvem * fvm;
            }
            v_out(b, Ionization::diag_loc(), k, j, i) = rhocve_sum;
            v(b, Ionization::temperature_old(), k, j, i) = tele;
          });
        });
  } else {
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const int nmat = v.GetSize(b, ccmat::rho());
          std::array<int, MAX_MATERIALS> eos_map;
          RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real tion = v(b, ccbulk::temperature(), k, j, i);
            Real rhocvi_sum = 0.0;
            RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, b, k, j, i);
            for (int m = 0; m < nmat; ++m) {
              auto &eosi = ion_eos(eos_map[m]);
              const Real rhom = v(b, cm::rho(m), k, j, i);
              const Real fvm = v(b, ccmat::volume_fraction(m), k, j, i);
              const Real cvim =
                  eosi.SpecificHeatFromDensityTemperature(rhom, tion, lambda[m]);

              rhocvi_sum += rhom * cvim * fvm;
            }
            v_out(b, Ionization::diag_loc(), k, j, i) = rhocvi_sum;
          });
        });
  }
  return TaskStatus::complete;
} // SetLocal

template TaskStatus SetLocal<TransportSpecies::Electron>(MeshData<Real> *md,
                                                         MeshData<Real> *md_out);
template TaskStatus SetLocal<TransportSpecies::Ion>(MeshData<Real> *md,
                                                    MeshData<Real> *md_out);

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::InitializeConductionQuantities
//! \brief
TaskStatus InitializeConductionQuantities(MeshData<Real> *md_base, MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pm = md->GetParentPointer();

  auto v_base = riot::MakePack<ccbulk::electron_temperature>(md_base);
  auto v = riot::MakePack<ccbulk::electron_temperature>(md);
  const int nblocks = v.GetNBlocks();

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto tele_base =
            RiotLoop::make_var_view(idx_range, v_base, ccbulk::electron_temperature());
        auto tele_out =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_temperature());
        RiotLoop::inner(idx_range,
                        [&](const auto kji) { tele_out(kji) = tele_base(kji); });
      });
  return TaskStatus::complete;
} // InitializeConductionQuantities

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Ionization::UpdateStateFromConduction
//! \brief
template <TransportSpecies Species>
TaskStatus UpdateStateFromConduction(MeshData<Real> *md_base, MeshData<Real> *md_u,
                                     const Real dt) {
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md_base->GetParentPointer();
  auto &options = pm->packages.Get("ionization");

  // timestep controls
  const Real T_scale_floor = options->Param<Real>("T_scale_floor");
  const auto fractional_change_scale = options->Param<Real>("fractional_change_scale");

  auto &materials = pm->packages.Get("materials");
  const auto &electron_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  auto v = riot::MakePack<Ionization::delta>(md_u);
  auto v_base =
      riot::MakePack<cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccbulk::rho,
                     ccbulk::electron_temperature, ccbulk::temperature,
                     cm::ionization_zbar, ccmat::volume_fraction,
                     ccbulk::total_material_energy, ccbulk::electron_internal_energy,
                     Ionization::delta, Ionization::temperature_old,
                     Ionization::temp_tstep_criterion>(md_base);
  const int nblocks = v.GetNBlocks();

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md_base, TE::CC);
  if (Species == TransportSpecies::Electron) {
    // electron conduction
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const int nmat = v_base.GetSize(b, ccmat::rho());
          std::array<int, MAX_MATERIALS> eos_map;
          RiotEOS::FillEosMap<ccmat::rho>(v_base, b, nmat, eos_from_matid, nphase,
                                          eos_map);

          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real delta = v(b, Ionization::delta(), k, j, i);
            Real tele = v_base(b, ccbulk::electron_temperature(), k, j, i);
            const Real tele_old = v_base(b, Ionization::temperature_old(), k, j, i);
            Real rhocve_sum = 0.0;
            RiotEOS::LambdaIndexerMulti<decltype(v_base)> lambda(v_base, b, k, j, i);
            for (int m = 0; m < nmat; m++) {
              auto &eose = electron_eos(eos_map[m]);
              const Real rhom =
                  v_base(b, cm::rho(m), k, j, i); // physical density (not cell averaged)
              const Real fvm = v_base(b, ccmat::volume_fraction(m), k, j, i);
              const Real cvem =
                  eose.SpecificHeatFromDensityTemperature(rhom, tele, lambda[m]);

              rhocve_sum += fvm * rhom * cvem;
            }
            tele -= delta;
            const Real dE = -rhocve_sum * delta;
            v_base(b, ccbulk::electron_temperature(), k, j, i) = tele;
            v_base(b, ccbulk::electron_internal_energy(), k, j, i) += dE;
            v_base(b, ccbulk::total_material_energy(), k, j, i) += dE;
            // Store the ~logarithmic derivative in del for timestep control
            const Real Tscale = std::max(tele, tele_old);
            Real del = Tscale > T_scale_floor
                           ? std::abs(tele - tele_old) /
                                 ((Tscale + T_scale_floor) * fractional_change_scale)
                           : 0.0;
            v_base(b, Ionization::temp_tstep_criterion(), k, j, i) =
                std::min(del, 2.0) / dt;
          });
        });
  } else {
    // ion conduction
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const int nmat = v_base.GetSize(b, ccmat::rho());
          std::array<int, MAX_MATERIALS> eos_map;
          RiotEOS::FillEosMap<ccmat::rho>(v_base, b, nmat, eos_from_matid, nphase,
                                          eos_map);

          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real delta = v(b, Ionization::delta(), k, j, i);
            Real tion = v_base(b, ccbulk::temperature(), k, j, i);
            Real rhocvi_sum = 0.0;
            RiotEOS::LambdaIndexerMulti<decltype(v_base)> lambda(v_base, b, k, j, i);
            for (int m = 0; m < nmat; m++) {
              auto &eosi = ion_eos(eos_map[m]);
              const Real rhom =
                  v_base(b, cm::rho(m), k, j, i); // physical density (not cell averaged)
              const Real fvm = v_base(b, ccmat::volume_fraction(m), k, j, i);
              const Real cvim =
                  eosi.SpecificHeatFromDensityTemperature(rhom, tion, lambda[m]);

              rhocvi_sum += fvm * rhom * cvim;
            }
            tion -= delta;
            const Real dE = -rhocvi_sum * delta;
            v_base(b, ccbulk::temperature(), k, j, i) = tion;
            v_base(b, ccbulk::total_material_energy(), k, j, i) += dE;
          });
        });
  }

  return TaskStatus::complete;
} // UpdateStateFromConduction

template TaskStatus UpdateStateFromConduction<TransportSpecies::Electron>(
    MeshData<Real> *md_base, MeshData<Real> *md_u, const Real dt);
template TaskStatus
UpdateStateFromConduction<TransportSpecies::Ion>(MeshData<Real> *md_base,
                                                 MeshData<Real> *md_u, const Real dt);

// Plasma viscosity calculation
// See Arnault (2013), High Energy Density Physics
// Volume 9, Issue 4, December 2013, Pages 711-721
// and
// E. L. Vold, R. M. Rauenzahn, C. H. Aldrich, K. Molvig, A. N. Simakov, and B.
// M. Haines. Plasma transport in an Eulerian AMR code. Physics of Plasmas,
// 24(4):042702, April 2017. ISSN 1070-664X. doi: 10.1063/1.4979171. URL
// https://doi.org/10.1063/1.4979171.
void CalculatePlasmaViscosity(MeshData<Real> *md) {
  using parthenon::ScratchPad1D;
  using namespace CouplingModelConstants;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &options = pm->packages.Get("ionization");
  const Real ion_shear_viscosity = options->Param<Real>("ion_shear_viscosity");
  const Real ion_bulk_viscosity = options->Param<Real>("ion_bulk_viscosity");
  const std::string ion_viscosity_model =
      options->Param<std::string>("ion_viscosity_model");
  const PlasmaViscosityModel viscosity_model =
      PlasmaViscosityEnumFromString(ion_viscosity_model);
  const Real zbar_floor = options->Param<Real>("zbar_floor");
  const Real ion_number_density_floor = options->Param<Real>("ion_number_density_floor");

  auto v =
      riot::MakePack<cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccbulk::rho,
                     ccbulk::electron_temperature, ccbulk::temperature,
                     cm::ionization_zbar, ccmat::volume_fraction,
                     ccbulk::electron_number_density, ccbulk::ion_shear_viscosity>(md);
  const int nblocks = v.GetNBlocks();

  auto &materials = pm->packages.Get("materials");
  const auto &ion_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const int max_array_size = materials->Param<int>("max_array_size");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");

  using lt = RiotUtils::LoopType<>;
  const int nhalo = 1;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, nhalo, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(1);

  const Real alpha_ij = 1.0;  // viscosity transport coefficients
  const Real R_alpha = 0.965; // Arnault's relaxation correction factors

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto nui = RiotLoop::GetPerPointScratch<Real>(idx_range);
        const int nmat = v.GetSize(b, ccmat::rho());
        std::array<int, MAX_MATERIALS> eos_map;
        RiotEOS::FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);

        // zero out eta
        RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
          v(b, ccbulk::ion_shear_viscosity(), k, j, i) = 0.0;
        });
        idx_range.TeamBarrier();

        if (viscosity_model == PlasmaViscosityModel::FokkerPlanckLandau) {
          for (int m1 = 0; m1 < nmat; ++m1) {
            // zero out nui
            RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
              nui(k, j, i) = 0.0;
            });
            idx_range.TeamBarrier();

            // species 1 mass
            const Real mu_m1 = ion_eos(eos_map[m1]).MeanAtomicMass();
            const Real mi1 = CouplingModelConstants::amu * mu_m1;

            // here we calculate nui = sum_j(alpha_ij^V * nu_ij)
            for (int m2 = 0; m2 < nmat; ++m2) {

              // species 2 mass
              const Real mu_m2 = ion_eos(eos_map[m2]).MeanAtomicMass();
              const Real mi2 = CouplingModelConstants::amu * mu_m2;

              RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
                // species 1
                const Real rhom1 = v(b, ccmat::rho(m1), k, j, i);
                const Real fvm1 = v(b, ccmat::volume_fraction(m1), k, j, i);
                const Real zbarm1 = v(b, cm::ionization_zbar(m1), k, j, i);

                // species 2
                const Real rhom2 = v(b, ccmat::rho(m2), k, j, i);
                const Real fvm2 = v(b, ccmat::volume_fraction(m2), k, j, i);
                const Real zbarm2 = v(b, cm::ionization_zbar(m2), k, j, i);

                // bulk
                const Real tele = v(b, ccbulk::electron_temperature(), k, j, i);
                const Real tion = v(b, ccbulk::temperature(), k, j, i);
                const Real ccbulk_ne = v(b, ccbulk::electron_number_density(), k, j, i);

                const Real ni1 = rhom1 / mi1;
                const Real ni2 = rhom2 / mi2;
                const Real nu12 = IonIonMomentumExchangeRate(
                    tele, tion, ccbulk_ne, ni1, ni2, mi1, mi2, zbarm1, zbarm2, zbar_floor,
                    ion_number_density_floor);
                nui(k, j, i) += nu12;
              }); // inner
              idx_range.TeamBarrier();
            } // nmat2

            RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
              const Real rhom1 = v(b, ccmat::rho(m1), k, j, i);
              const Real ni1 = rhom1 / mi1;
              const Real tion = v(b, ccbulk::temperature(), k, j, i);
              v(b, ccbulk::ion_shear_viscosity(), k, j, i) +=
                  alpha_ij * ni1 * kberg * tion / (nui(k, j, i) + 1e-100);
            }); // inner
          } // nmat1
        } else { // constant viscosity
          RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
            v(b, ccbulk::ion_shear_viscosity(), k, j, i) = ion_shear_viscosity;
          });
        } // viscosity model
      }); // outer
} // CalculatePlasmaViscosity

//----------------------------------------------------------------------------------------
//! \fn  Real Ionization::EstimateTimestepMesh
//! \brief Calculates the numerical timestep associated with electron thermal conduction
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pm = md->GetParentPointer();
  auto &hydro_pkg = pm->packages.Get("hydro");
  auto &hydro_params = hydro_pkg->AllParams();
  auto &ionization_pkg = pm->packages.Get("ionization");
  auto &ionization_params = ionization_pkg->AllParams();

  const bool plasma_viscosity = ionization_params.Get<bool>("plasma_viscosity");
  const bool electron_thermal_conduction =
      ionization_params.Get<bool>("electron_thermal_conduction");

  Real dt_conduction = 1e20;
  Real dt_viscosity = 1e20;
  const auto timestep_control = ionization_params.Get<std::string>("timestep_control");

  auto v = riot::MakePack<Ionization::Dcell, Ionization::temp_tstep_criterion,
                          ccbulk::rho, ccbulk::ion_shear_viscosity>(md);
  const int ndim = pm->ndim;

  const auto &cfl = hydro_params.Get<Real>("cfl");

  //===========================
  // conduction time step limit
  //===========================

  if (electron_thermal_conduction) {
    if (timestep_control == "explicit") {
      // SWJ: not sure why dt is included in the diffusion coefficient, but pass dt
      // = 1
      CalculateElectronThermalDiffusionCoefficient(md, md, 1.0);

      using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
      auto idx_space = rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md,
                                         parthenon::TopologicalElement::CC);
      const Real min_dt = RiotLoop::outer_reduce(
          idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
            auto pv = RiotLoop::make_pack_view(idx_range, v);
            auto &coords = v.GetCoordinates(b);
            RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &ldt) {
              const auto [k, j, i] = idx_range.GetKJI(idx);
              const Real d_cell = pv(Ionization::Dcell(), idx);
              for (int d = 0; d < ndim; d++) {
                const Real dx = coords.Dxc(d + 1, k, j, i);
                ldt = std::min(ldt, 0.5 * dx * dx / (2.0 * d_cell));
              }
            });
          });
      const auto &cfl = hydro_params.Get<Real>("cfl");
      dt_conduction = cfl * min_dt;
    } else if (timestep_control == "relative") {
      auto *dt = ionization_pkg->MutableParam<Real>("dt");
      if (*dt < 0.0) {
        dt_conduction = 1.e12;
      } else {
        using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
        auto idx_space = rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md,
                                           parthenon::TopologicalElement::CC);
        const Real max_delta = RiotLoop::outer_reduce(
            idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
              auto pv = RiotLoop::make_pack_view(idx_range, v);
              RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &mdt) {
                mdt = std::max(mdt, std::abs(pv(temp_tstep_criterion(), idx)));
              });
            });
        dt_conduction = 1.0 / max_delta;
      }
    }
  }

  //===========================
  // viscosity time step limit
  //===========================

  if (plasma_viscosity) {
    using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
    auto idx_space = rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md,
                                       parthenon::TopologicalElement::CC);
    const Real min_dt = RiotLoop::outer_reduce(
        idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
          auto pv = RiotLoop::make_pack_view(idx_range, v);
          auto &coords = v.GetCoordinates(b);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &ldt) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real rho = pv(ccbulk::rho(), k, j, i);
            const Real eta = pv(ccbulk::ion_shear_viscosity(), k, j, i);
            for (int d = 0; d < ndim; d++) {
              const Real dx = coords.Dxc(d + 1, k, j, i);
              ldt = std::min(ldt, rho * dx * dx / (eta + 1e-15));
            }
          });
        });
    const auto &cfl = hydro_params.Get<Real>("cfl");
    dt_viscosity = cfl * min_dt;
  }

  return std::min(dt_conduction, dt_viscosity);
}

// Compute fluxes to momentum and energy from plasma viscosity
TaskStatus ComputePlasmaViscousFluxes(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();

  auto &ionization_pkg = pm->packages.Get("ionization");
  auto &ionization_params = ionization_pkg->AllParams();

  const bool plasma_viscosity = ionization_params.Get<bool>("plasma_viscosity");
  if (!plasma_viscosity) return TaskStatus::complete;

  // Pack up fields
  std::set<parthenon::PDOpt> opts = {parthenon::PDOpt::WithFluxes};
  auto vb = riot::MakePack<ccbulk::momentum, ccbulk::total_material_energy,
                           ccbulk::face_velocity, ccbulk::strain_rate,
                           ccbulk::ion_shear_viscosity>(md, std::vector<int>{}, opts);

  // don't launch kernel if there are no blocks
  const int nblocks = vb.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  const Real bulk_viscosity = ionization_params.Get<Real>("ion_bulk_viscosity");

  // Calculate the strain rate tensor.
  Hydro::CalculateStrainRate(md, vb);

  const int ndim = pm->ndim;

  using lt = RiotUtils::LoopType<>;
  const int nhalo = 1;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, nhalo, nblocks, md,
                                     parthenon::TopologicalElement::CC);

  // TODO(@SWJ): This currently only works for planar geometries
  // X1 fluxes
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pvb = RiotLoop::make_pack_view(idx_range, vb);
        RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
          Real &eta = pvb(ccbulk::ion_shear_viscosity(), k, j, i);
          Real &etam = pvb(ccbulk::ion_shear_viscosity(), k, j, i - 1);

          Real &vxx = pvb(ccbulk::face_velocity(0), k, j, i);
          Real &vyx = pvb(ccbulk::face_velocity(1), k, j, i);
          Real &vzx = pvb(ccbulk::face_velocity(2), k, j, i);

          Real &exx = pvb(ccbulk::strain_rate(0), k, j, i);
          Real &exxm = pvb(ccbulk::strain_rate(0), k, j, i - 1);
          Real &exy = pvb(ccbulk::strain_rate(1), k, j, i);
          Real &exym = pvb(ccbulk::strain_rate(1), k, j, i - 1);
          Real &exz = pvb(ccbulk::strain_rate(2), k, j, i);
          Real &exzm = pvb(ccbulk::strain_rate(2), k, j, i - 1);
          Real &eyy = pvb(ccbulk::strain_rate(3), k, j, i);
          Real &eyym = pvb(ccbulk::strain_rate(3), k, j, i - 1);
          Real &eyz = pvb(ccbulk::strain_rate(4), k, j, i);
          Real &ezz = pvb(ccbulk::strain_rate(5), k, j, i);
          Real &ezzm = pvb(ccbulk::strain_rate(5), k, j, i - 1);

          // these should use Riot::MakeFluxPack objects
          Real &fmomxx = vb.flux(b, X1DIR, ccbulk::momentum(0), k, j, i);
          Real &fmomyx = vb.flux(b, X1DIR, ccbulk::momentum(1), k, j, i);
          Real &fmomzx = vb.flux(b, X1DIR, ccbulk::momentum(2), k, j, i);
          Real &fEx = vb.flux(b, X1DIR, ccbulk::total_material_energy(), k, j, i);

          // compute the fluxes
          const Real lambda_visc = bulk_viscosity - 2. / 3. * eta;
          const Real lambda_viscm = bulk_viscosity - 2. / 3. * etam;
          // velocity divergence
          const Real vdiv = exx + eyy + ezz;
          const Real vdivm = exxm + eyym + ezzm;
          // stress tensor components
          const Real sxx = 2.0 * eta * exx + lambda_visc * vdiv;
          const Real sxxm = 2.0 * etam * exxm + lambda_viscm * vdivm;
          const Real syx = 2.0 * eta * exy;
          const Real syxm = 2.0 * etam * exym;
          const Real szx = 2.0 * eta * exz;
          const Real szxm = 2.0 * etam * exzm;
          // add the viscous fluxes
          // momentum
          fmomxx -= 0.5 * (sxx + sxxm);
          fmomyx -= 0.5 * (syx + syxm);
          fmomzx -= 0.5 * (szx + szxm);
          // total energy
          fEx -= 0.5 * (sxx + sxxm) * vxx + 0.5 * (syx + syxm) * vyx +
                 0.5 * (szx + szxm) * vzx;
        }); // inner
      });   // outer X1 fluxes

  if (ndim > 1) {
    // X2 fluxes
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          auto pvb = RiotLoop::make_pack_view(idx_range, vb);
          RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
            Real &eta = pvb(ccbulk::ion_shear_viscosity(), k, j, i);
            Real &etam = pvb(ccbulk::ion_shear_viscosity(), k, j - 1, i);

            Real &vxy = pvb(ccbulk::face_velocity(3), k, j, i);
            Real &vyy = pvb(ccbulk::face_velocity(4), k, j, i);
            Real &vzy = pvb(ccbulk::face_velocity(5), k, j, i);

            Real &exx = pvb(ccbulk::strain_rate(0), k, j, i);
            Real &exxm = pvb(ccbulk::strain_rate(0), k, j - 1, i);
            Real &exy = pvb(ccbulk::strain_rate(1), k, j, i);
            Real &exym = pvb(ccbulk::strain_rate(1), k, j - 1, i);
            Real &exz = pvb(ccbulk::strain_rate(2), k, j, i);
            Real &exzm = pvb(ccbulk::strain_rate(2), k, j - 1, i);
            Real &eyy = pvb(ccbulk::strain_rate(3), k, j, i);
            Real &eyym = pvb(ccbulk::strain_rate(3), k, j - 1, i);
            Real &eyz = pvb(ccbulk::strain_rate(4), k, j, i);
            Real &eyzm = pvb(ccbulk::strain_rate(4), k, j - 1, i);
            Real &ezz = pvb(ccbulk::strain_rate(5), k, j, i);
            Real &ezzm = pvb(ccbulk::strain_rate(5), k, j - 1, i);

            // these should use Riot::MakeFluxPack objects
            Real &fmomxy = vb.flux(b, X2DIR, ccbulk::momentum(0), k, j, i);
            Real &fmomyy = vb.flux(b, X2DIR, ccbulk::momentum(1), k, j, i);
            Real &fmomzy = vb.flux(b, X2DIR, ccbulk::momentum(2), k, j, i);
            Real &fEy = vb.flux(b, X2DIR, ccbulk::total_material_energy(), k, j, i);

            // compute the fluxes
            const Real lambda_visc = bulk_viscosity - 2. / 3. * eta;
            const Real lambda_viscm = bulk_viscosity - 2. / 3. * etam;
            // velocity divergence
            const Real vdiv = exx + eyy + ezz;
            const Real vdivm = exxm + eyym + ezzm;
            // stress tensor components
            const Real syy = 2.0 * eta * eyy + lambda_visc * vdiv;
            const Real syym = 2.0 * etam * eyym + lambda_viscm * vdivm;
            const Real sxy = 2.0 * eta * exy;
            const Real sxym = 2.0 * etam * exym;
            const Real szy = 2.0 * eta * eyz;
            const Real szym = 2.0 * etam * eyzm;
            // add the viscous fluxes
            // momentum
            fmomxy -= 0.5 * (sxy + sxym);
            fmomyy -= 0.5 * (syy + syym);
            fmomzy -= 0.5 * (szy + szym);
            // total energy
            fEy -= 0.5 * (sxy + sxym) * vxy + 0.5 * (syy + syym) * vyy +
                   0.5 * (szy + szym) * vzy;
          }); // inner
        });   // Outer X2 fluxes
  } // if (ndim > 1)

  if (ndim > 2) {
    // X3 fluxes
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          auto pvb = RiotLoop::make_pack_view(idx_range, vb);
          RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
            Real &eta = pvb(ccbulk::ion_shear_viscosity(), k, j, i);
            Real &etam = pvb(ccbulk::ion_shear_viscosity(), k, j - 1, i);

            Real &vxz = pvb(ccbulk::face_velocity(6), k, j, i);
            Real &vyz = pvb(ccbulk::face_velocity(7), k, j, i);
            Real &vzz = pvb(ccbulk::face_velocity(8), k, j, i);

            Real &exx = pvb(ccbulk::strain_rate(0), k, j, i);
            Real &exxm = pvb(ccbulk::strain_rate(0), k - 1, j, i);
            Real &exy = pvb(ccbulk::strain_rate(1), k, j, i);
            Real &exym = pvb(ccbulk::strain_rate(1), k - 1, j, i);
            Real &exz = pvb(ccbulk::strain_rate(2), k, j, i);
            Real &exzm = pvb(ccbulk::strain_rate(2), k - 1, j, i);
            Real &eyy = pvb(ccbulk::strain_rate(3), k, j, i);
            Real &eyym = pvb(ccbulk::strain_rate(3), k - 1, j, i);
            Real &eyz = pvb(ccbulk::strain_rate(4), k, j, i);
            Real &eyzm = pvb(ccbulk::strain_rate(4), k - 1, j, i);
            Real &ezz = pvb(ccbulk::strain_rate(5), k, j, i);
            Real &ezzm = pvb(ccbulk::strain_rate(5), k - 1, j, i);

            // these should use Riot::MakeFluxPack objects
            Real &fmomxz = vb.flux(b, X3DIR, ccbulk::momentum(0), k, j, i);
            Real &fmomyz = vb.flux(b, X3DIR, ccbulk::momentum(1), k, j, i);
            Real &fmomzz = vb.flux(b, X3DIR, ccbulk::momentum(2), k, j, i);
            Real &fEz = vb.flux(b, X3DIR, ccbulk::total_material_energy(), k, j, i);

            // compute the fluxes
            const Real lambda_visc = bulk_viscosity - 2. / 3. * eta;
            const Real lambda_viscm = bulk_viscosity - 2. / 3. * etam;
            // velocity divergence
            const Real vdiv = exx + eyy + ezz;
            const Real vdivm = exxm + eyym + ezzm;
            // stress tensor components
            const Real szz = 2.0 * eta * ezz + lambda_visc * vdiv;
            const Real szzm = 2.0 * etam * ezzm + lambda_viscm * vdivm;
            const Real sxz = 2.0 * eta * exz;
            const Real sxzm = 2.0 * etam * exzm;
            const Real syz = 2.0 * eta * eyz;
            const Real syzm = 2.0 * etam * eyzm;
            // add the viscous fluxes
            // momentum
            fmomxz -= 0.5 * (sxz + sxzm);
            fmomyz -= 0.5 * (syz + syzm);
            fmomzz -= 0.5 * (szz + szzm);
            // total energy
            fEz -= 0.5 * (sxz + sxzm) * vxz + 0.5 * (syz + syzm) * vyz +
                   0.5 * (szz + szzm) * vzz;
          }); // inner
        });   // outer X3 fluxes
  } // if (ndim > 2)

  return TaskStatus::complete;
} // ComputeViscousFluxes

//----------------------------------------------------------------------------------------
//! \fn  ElectronThermalConductivityModel
//! Ionization::ElectronConductivityModelEnumFromString
//! \brief
ElectronThermalConductivityModel
ElectronConductivityModelEnumFromString(const std::string &model) {
  if (model == "constant") {
    return ElectronThermalConductivityModel::Constant;
  } else if (model == "spitzer_volume_average_arithmetic") {
    return ElectronThermalConductivityModel::SpitzerVolumeAverageArithmetic;
  } else if (model == "spitzer_volume_average_harmonic") {
    return ElectronThermalConductivityModel::SpitzerVolumeAverageHarmonic;
  } else if (model == "spitzer_electron_number_density_average") {
    return ElectronThermalConductivityModel::SpitzerElectronNumberDensityAverage;
  } else {
    PARTHENON_FAIL("Invalid choice for ionization/electron_conductivity_model");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  IonThermalConductivityModel Ionization::IonConductivityModelEnumFromString
//! \brief
IonThermalConductivityModel IonConductivityModelEnumFromString(const std::string &model) {
  if (model == "constant") {
    return IonThermalConductivityModel::Constant;
  } else if (model == "braginskii") {
    return IonThermalConductivityModel::Braginskii;
  } else {
    PARTHENON_FAIL("Invalid choice for ionization/ion_conductivity_model");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  CoulombLogarithmKind Ionization::ParseCoulombLogarithmKind
//! \brief
CoulombLogarithmKind ParseCoulombLogarithmKind(std::string coulomb_logarithm) {
  if (coulomb_logarithm == "lee_moore") {
    return CoulombLogarithmKind::LeeMoore;
  } else if (coulomb_logarithm == "basic") {
    return CoulombLogarithmKind::Basic;
  } else if (coulomb_logarithm == "bps") {
    return CoulombLogarithmKind::BPS;
  } else {
    // Brysk as default since it is the most tested.
    return CoulombLogarithmKind::Brysk;
  }
} // ParseCoulombLogarithmKind

PlasmaViscosityModel PlasmaViscosityEnumFromString(const std::string &model) {
  if (model == "constant") {
    return PlasmaViscosityModel::Constant;
  } else if (model == "fokker_planck_landau") {
    return PlasmaViscosityModel::FokkerPlanckLandau;
  } else {
    PARTHENON_FAIL("Invalid choice for ionization/plasma_viscosity_model");
  }
}

} // namespace Ionization
