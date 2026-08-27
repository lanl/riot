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

// C++ includes
#include <cmath>
#include <limits>

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>

// Riot includes
#include "hydro/hydro.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "radiation_diffusion/diffusion_equation.hpp"
#include "radiation_diffusion/material_helpers.hpp"
#include "radiation_diffusion/multigroup_diffusion.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"

using namespace parthenon::driver::prelude;

namespace {
constexpr IndexDomain GetDomain(parthenon::CoordinateDirection dir,
                                parthenon::BoundaryFunction::BCSide side) {
  using namespace parthenon;
  using namespace parthenon::BoundaryFunction;
  if (side == BCSide::Inner) {
    if (dir == X1DIR)
      return IndexDomain::inner_x1;
    else if (dir == X2DIR)
      return IndexDomain::inner_x2;
    else if (dir == X3DIR)
      return IndexDomain::inner_x3;
  } else {
    if (dir == X1DIR)
      return IndexDomain::outer_x1;
    else if (dir == X2DIR)
      return IndexDomain::outer_x2;
    else if (dir == X3DIR)
      return IndexDomain::outer_x3;
  }
  return IndexDomain::inner_x1;
}

template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE,
          class F>
void RadiationBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
                       F temperature_function, bool zero_flux) {
  using namespace parthenon;
  using namespace parthenon::BoundaryFunction;
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  static std::vector<MetadataFlag> flags{Metadata::FillGhost, Metadata::Cell};
  static auto desc =
      MakePackDescriptor<RadiationDiffusion::MultiGroupVars::Egroup>(rc.get(), flags);
  static auto desc_coarse =
      MakePackDescriptor<RadiationDiffusion::MultiGroupVars::Egroup>(
          rc.get(), flags, std::set<PDOpt>{PDOpt::Coarse});
  auto q = coarse ? desc_coarse.GetPack(rc.get()) : desc.GetPack(rc.get());

  const int b = 0;
  auto nb = IndexRange{0, 0};

  constexpr IndexDomain domain = GetDomain(DIR, SIDE);
  auto pmb = rc->GetBlockPointer();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = (DIR == X1DIR)   ? bounds.GetBoundsI(IndexDomain::interior)
                      : (DIR == X2DIR) ? bounds.GetBoundsJ(IndexDomain::interior)
                                       : bounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->pmy_mesh->packages.Get("multigroup_diffusion_package");

  const auto sim_time = pkg->Param<parthenon::SimTime>("sim_time");
  RadiationDiffusion::BlackBodyHelper bb_helper(pmb->pmy_mesh);
  const Real tnp1 = sim_time.time + sim_time.dt;
  const Real T0 = temperature_function(tnp1);

  const int offset = (SIDE == BCSide::Inner) ? 1 : -1;
  const int ioff = (DIR == X1DIR) * offset;
  const int joff = (DIR == X2DIR) * offset;
  const int koff = (DIR == X3DIR) * offset;
  pmb->par_for_bndry(
      "Robin BC", nb, domain, TE::CC, coarse, false,
      KOKKOS_LAMBDA(const int & /*l*/, const int &k, const int &j, const int &i) {
        int sg = q.GetLowerBound(b, RadiationDiffusion::MultiGroupVars::Egroup());
        int eg = q.GetUpperBound(b, RadiationDiffusion::MultiGroupVars::Egroup());
        for (int g = 0; g <= eg - sg; ++g) {
          const auto [B, dBdT] = bb_helper.GetBB(g, T0);
          Real Ebl = q(b, RadiationDiffusion::MultiGroupVars::Egroup(g), k + koff,
                       j + joff, i + ioff);
          q(b, RadiationDiffusion::MultiGroupVars::Egroup(g), k, j, i) =
              zero_flux ? Ebl : 2 * B - Ebl;
        }
      });
}

// Have to do this with a named functor rather than a lambda for things
// to work with NVCC
struct constantT_functor {
  constantT_functor() : Tbound(-1.0) {}
  explicit constantT_functor(Real Tbound) : Tbound(Tbound) {}
  const Real Tbound;
  Real operator()(Real /*time*/) const { return Tbound; }
};

struct doubleshell_boundary_functor {
  static constexpr Real evToK = 11604.5250061657;
  static constexpr Real t1 = 0.5e-9;
  static constexpr Real T1 = 10.0;
  static constexpr Real t2 = 1.9e-9;
  static constexpr Real T2 = 175.0;
  static constexpr Real t3 = 3e-9;
  static constexpr Real T3 = 260.0;
  static constexpr Real t4 = 5.5e-9;
  static constexpr Real T4 = 290.0;
  static constexpr Real Tfinal = 100.0;
  static constexpr Real tau = 1.e-8 - t4;

  Real operator()(Real tnp1) const {
    Real T0;
    if (tnp1 < t1) {
      T0 = T1;
    } else if (tnp1 < t2) {
      T0 = T1 + (tnp1 - t1) / (t2 - t1) * (T2 - T1);
    } else if (tnp1 < t3) {
      T0 = T2 + (tnp1 - t2) / (t3 - t2) * (T3 - T2);
    } else if (tnp1 < t4) {
      T0 = T3 + (tnp1 - t3) / (t4 - t3) * (T4 - T3);
    } else {
      T0 = Tfinal + (T4 - Tfinal) * exp(-(tnp1 - t4) / tau);
    }
    T0 *= evToK;
    return T0;
  }
};

template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE,
          class F>
inline auto GetRadBC(F temperature_function) {
  return [temperature_function](std::shared_ptr<MeshBlockData<Real>> &rc,
                                bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    RadiationBoundary<DIR, SIDE>(rc, coarse, temperature_function, false);
  };
}

template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
inline auto GetConstantFluxRadBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    RadiationBoundary<DIR, SIDE>(rc, coarse, constantT_functor(), true);
  };
}

} // unnamed namespace

namespace RadiationDiffusion {

template <class temperature>
std::shared_ptr<StateDescriptor>
MultiGroup<temperature>::Initialize(ParameterInput *pin, StateDescriptor *materials) {
  auto pkg = std::make_shared<StateDescriptor>("multigroup_diffusion_package");

  pkg->EstimateTimestepMesh = EstimateTimestepMesh;
  pkg->PostProblemGeneratorMesh = MeshPostProblemGenerator;

  using namespace parthenon;
  using namespace parthenon::BoundaryFunction;
  using namespace MultiGroupVars;

  std::string boundary_condition =
      pin->GetOrAddString("diffusion", "boundary_condition", "constant_temperature",
                          {"constant_temperature", "zero_flux", "double_shell"},
                          "Boundary condition for diffusion.");
  if (boundary_condition == "constant_temperature") {
    auto Tbounds = pin->GetOrAddVector<Real>("diffusion", "boundary_T", {1.e5},
                                             "Boundary temperatures.");
    if (Tbounds.size() == 1) Tbounds = std::vector<Real>(6, Tbounds[0]);
    if (Tbounds.size() < 6)
      PARTHENON_FAIL("Need a boundary temperature for each side of each dimension.");

    pkg->UserBoundaryFunctions[BoundaryFace::inner_x1].push_back(
        GetRadBC<X1DIR, BCSide::Inner>(constantT_functor(Tbounds[0])));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x1].push_back(
        GetRadBC<X1DIR, BCSide::Outer>(constantT_functor(Tbounds[1])));
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x2].push_back(
        GetRadBC<X2DIR, BCSide::Inner>(constantT_functor(Tbounds[2])));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x2].push_back(
        GetRadBC<X2DIR, BCSide::Outer>(constantT_functor(Tbounds[3])));
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x3].push_back(
        GetRadBC<X3DIR, BCSide::Inner>(constantT_functor(Tbounds[4])));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x3].push_back(
        GetRadBC<X3DIR, BCSide::Outer>(constantT_functor(Tbounds[5])));
  } else if (boundary_condition == "zero_flux") {
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x1].push_back(
        GetConstantFluxRadBC<X1DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x1].push_back(
        GetConstantFluxRadBC<X1DIR, BCSide::Outer>());
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x2].push_back(
        GetConstantFluxRadBC<X2DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x2].push_back(
        GetConstantFluxRadBC<X2DIR, BCSide::Outer>());
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x3].push_back(
        GetConstantFluxRadBC<X3DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x3].push_back(
        GetConstantFluxRadBC<X3DIR, BCSide::Outer>());
  } else if (boundary_condition == "double_shell") {
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x1].push_back(
        GetRadBC<X1DIR, BCSide::Inner>(doubleshell_boundary_functor()));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x1].push_back(
        GetRadBC<X1DIR, BCSide::Outer>(doubleshell_boundary_functor()));
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x2].push_back(
        GetRadBC<X2DIR, BCSide::Inner>(doubleshell_boundary_functor()));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x2].push_back(
        GetRadBC<X2DIR, BCSide::Outer>(doubleshell_boundary_functor()));
    pkg->UserBoundaryFunctions[BoundaryFace::inner_x3].push_back(
        GetRadBC<X3DIR, BCSide::Inner>(doubleshell_boundary_functor()));
    pkg->UserBoundaryFunctions[BoundaryFace::outer_x3].push_back(
        GetRadBC<X3DIR, BCSide::Outer>(doubleshell_boundary_functor()));
  }
  // Set boundary conditions for Poisson variables

  bool flux_limit =
      pin->GetOrAddBoolean("diffusion", "flux_limit", true, "Limit the diffusive flux");
  pkg->AddParam<>("flux_limit", flux_limit);

  bool update_temperature =
      pin->GetOrAddBoolean("diffusion", "update_temperature", false,
                           "Update the temperature due to radiation energy exchange.");
  pkg->AddParam<>("update_temperature", update_temperature);

  // Extract group structure
  const int ngroup = materials->Param<int>("ngroups");
  const auto group_bounds_hz = materials->Param<std::vector<Real>>("group_bounds");
  PARTHENON_REQUIRE(ngroup <= MAX_GROUPS,
                    "Number of radiation groups exceeds MAX_GROUPS compile-time limit");
  PARTHENON_REQUIRE(static_cast<int>(group_bounds_hz.size()) == ngroup + 1,
                    "materials group_bounds must have ngroups+1 edges.");
  pkg->AddParam<>("ngroup", ngroup);

  // Convert the shared Hz group bounds to the diffusion package's native units (K)
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  constexpr Real hz_to_K = pc::h / pc::kb;
  parthenon::ParArray1D<Real> group_energies_in_K("Radiation group energies", ngroup + 1);
  auto group_energies_h =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), group_energies_in_K);
  for (int g = 0; g < ngroup; ++g) {
    group_energies_h(g) = group_bounds_hz[g] * hz_to_K;
  }
  group_energies_h(ngroup) = 1.e40; // Big number for the last (open) group
  Kokkos::deep_copy(group_energies_in_K, group_energies_h);
  pkg->AddParam<>("group_energies_in_K", group_energies_in_K);

  const Real mintol = 1.0e5 / std::numeric_limits<Real>::max();
  Real diff_rho_min = pin->GetOrAddReal("diffusion", "opacity_rho_min", mintol,
                                        "Density floor for calculating opacities.");
  Real diff_temp_min = pin->GetOrAddReal("diffusion", "opacity_temp_min", mintol,
                                         "Temperature floor for calculating opacities.");
  pkg->AddParam<>("diff_rho_min", diff_rho_min);
  pkg->AddParam<>("diff_temp_min", diff_temp_min);

  int nriter = pin->GetOrAddInteger("diffusion", "nriter", 3,
                                    "Maximum number of Newton-Raphson iterations.");
  pkg->AddParam<>("nriter", nriter);
  int local_nriter = pin->GetOrAddInteger(
      "diffusion", "local_nriter", 0, "Number of zone-local Newton-Raphson iterations.");
  pkg->AddParam<>("local_nriter", local_nriter);

  Real nr_tolerance = pin->GetOrAddReal("diffusion", "nr_tolerance", 1.e-5,
                                        "Tolerance for Newton-Raphson convergence.");
  pkg->AddParam<>("nr_tolerance", nr_tolerance);

  bool print_per_nr_step =
      pin->GetOrAddBoolean("diffusion", "print_per_nr_step", false,
                           "Print Newton-Raphson convergence to screen.");
  pkg->AddParam<>("print_per_nr_step", print_per_nr_step);

  pkg->AddParam<>("TimingAccumulatorDict",
                  std::make_shared<TimingAccumulatorDictionary>());

  // NOTE(@pdmullen): RIOT's singularity-opac config always assumes CGS units, however, we
  // only ever make calls to opac.AbsorptionCoefficient(), thereby permitting us to
  // (slimily) side-step interference between singularity-opac's unit system and RIOT's
  // diffusion unit system... If we ever expand the radiation module to make calls to
  // e.g., the emissivity, we would have a units mismatch if a_radiation or c_light are
  // manually specified to non-CGS values.  The team-agreed best solution here is to
  // (eventually) flesh out RIOT's handling of units... (pc is aliased above.)
  Real a_rad = pin->GetOrAddReal("diffusion", "a_radiation", pc::ar);
  Real c_l = pin->GetOrAddReal("diffusion", "c_light", pc::c);
  pkg->AddParam<>("a_radiation", a_rad);
  pkg->AddParam<>("c_light", c_l);

  // Radiation diffusion timestep controls, by default we set the cfl and the minimum
  // considered temperature to large values so that diffusion does not provide a
  // timestep constraint
  Real cfl =
      pin->GetOrAddReal("diffusion", "cfl", 1.e-3 * std::numeric_limits<Real>::max(),
                        "CFL number for P1 timestep vote.");
  pkg->AddParam<>("cfl", cfl);

  Real timestep_min_temperature = pin->GetOrAddReal(
      "diffusion", "timestep_min_temperature", 1.e-3 * std::numeric_limits<Real>::max(),
      "Minimum zone temperature to consider when suggesting the timestep.");
  pkg->AddParam<>("timestep_min_temperature", timestep_min_temperature);

  Real temperature_fractional_change_target = pin->GetOrAddReal(
      "diffusion", "temperature_fractional_change_target", 0.1,
      "Fractional change in temperature to target when setting the timestep.");
  pkg->AddParam<>("temperature_fractional_change_target",
                  temperature_fractional_change_target);

  Real timestep_temperature_scale =
      pin->GetOrAddReal("diffusion", "timestep_temperature_scale", 0.0,
                        "Additive temperature scale parameter for timestep calculation.");
  pkg->AddParam<>("timestep_temperature_scale", timestep_temperature_scale);

  Real maximum_timestep_reduction_factor =
      pin->GetOrAddReal("diffusion", "maximum_timestep_reduction_factor", 2.0,
                        "Maximum factor that timestep can be reduced due to radiation "
                        "vote from one step to the next.");
  pkg->AddParam<>("maximum_timestep_reduction_factor", maximum_timestep_reduction_factor);

  // Radiation AMR controls
  bool use_rad_amr = pin->GetOrAddBoolean("diffusion", "use_amr_criteria", false,
                                          "Use the radiation diffusion AMR criteria.");
  if (use_rad_amr) {
    pkg->CheckRefinementBlock = CheckRefinement;
    Real amr_min_temperature = pin->GetOrAddReal(
        "diffusion", "amr_min_temperature", 0.0,
        "Minimum temperature to consider when checking for required refinement.");
    pkg->AddParam<>("amr_min_temperature", amr_min_temperature);
    Real amr_min_density = pin->GetOrAddReal(
        "diffusion", "amr_min_density", 0.0,
        "Minimum density to consider when checking for required refinement.");
    pkg->AddParam<>("amr_min_density", amr_min_density);
    Real amr_threshold = pin->GetOrAddReal("diffusion", "amr_threshold", 0.0,
                                           "Dimensionless threshold for refinement.");
    pkg->AddParam<>("amr_threshold", amr_threshold);
    Real derefine_radius =
        pin->GetOrAddReal("diffusion", "derefine_radius", -1.0,
                          "Vote for derefinement outside this radius.");
    pkg->AddParam<>("derefine_radius", derefine_radius);
  }
  std::vector<int> sol_shape{ngroup};

  std::shared_ptr<parthenon::solvers::SolverBase> solver =
      GetSolverSptr<temperature>(pin);
  solver->initial_guess_is_zero = true;
  pkg->AddParam<>("solver", solver);

  pkg->AddParam<>("sim_time", parthenon::SimTime(),
                  parthenon::Params::Mutability::Mutable);
  pkg->AddParam<int>("step_solver_iterations", 0, parthenon::Params::Mutability::Mutable);
  pkg->AddParam<int>("step_newt_iterations", 0, parthenon::Params::Mutability::Mutable);

  bool report_timings =
      pin->GetOrAddBoolean("diffusion", "report_timings", false,
                           "Report different timings of the radiation diffusion solver "
                           "at the end of the calculation.");
  pkg->AddParam<>("report_timings", report_timings);
  if (parthenon::Globals::my_rank == 0 && report_timings) {
    parthenon::Task::enable_timing = true;
    parthenon::Task::enable_timing_chunks = false;
  }
  pkg->AddParam<Real>("total_operator_split", 0, parthenon::Params::Mutability::Mutable);
  pkg->AddParam<Real>("total_hydro", 0, parthenon::Params::Mutability::Mutable);
  pkg->AddParam<Real>("this_operator_split", 0, parthenon::Params::Mutability::Mutable);
  pkg->AddParam<Real>("this_hydro", 0, parthenon::Params::Mutability::Mutable);

  using namespace parthenon::refinement_ops;
  using namespace MultiGroupVars;
  static const parthenon::MetadataFlag OperatorSplit =
      Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);

  auto m_egroup =
      Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                Metadata::WithFluxes, Metadata::GMGRestrict, Metadata::CommunicateOne,
                Metadata::GMGProlongate, OperatorSplit},
               sol_shape);
  m_egroup.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  pkg->AddField<Egroup>(m_egroup);

  auto m_fgroup = Metadata({Metadata::Face, Metadata::Independent,
                            Metadata::CellMemAligned, Metadata::Flux, OperatorSplit},
                           sol_shape);
  m_fgroup.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  pkg->AddField<Fgroup>(m_fgroup);

  auto m_D = Metadata({Metadata::Independent, Metadata::OneCopy, Metadata::Face,
                       Metadata::GMGRestrict, Metadata::CellMemAligned, OperatorSplit},
                      sol_shape);
  m_D.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField<D>(m_D);

  auto m_c = Metadata({Metadata::Independent, Metadata::OneCopy, Metadata::Cell,
                       Metadata::GMGRestrict, OperatorSplit},
                      sol_shape);
  m_c.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField<diag_loc>(m_c);
  pkg->AddField<sigma>(m_c);
  pkg->AddField<dSdT>(m_c);

  auto m_kappa_cell = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit}, sol_shape);
  pkg->AddField<kappa_cell>(m_kappa_cell);

  auto m_kappa_face = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                                Metadata::CellMemAligned, OperatorSplit},
                               sol_shape);
  pkg->AddField<kappa_face>(m_kappa_face);

  auto m_T0 =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit});
  pkg->AddField<temperature0>(m_T0);

  auto m_dTc =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit});
  pkg->AddField<dTc>(m_dTc);

  // Cached geometry
  auto m_face = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                          Metadata::CellMemAligned, OperatorSplit});
  pkg->AddField<face_area>(m_face);
  pkg->AddField<DeltaX>(m_face);
  auto m_cell =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit});
  pkg->AddField<volume>(m_cell);

  return pkg;
}

template <class temperature>
TaskCollection MultiGroup<temperature>::Step(Mesh *pmesh, parthenon::SimTime &tm,
                                             const Real dt) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  using Tasks = MultiGroupTasks<temperature>;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  auto solver = pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver");
  auto timings = pkg->Param<std::shared_ptr<parthenon::TimingAccumulatorDictionary>>(
      "TimingAccumulatorDict");

  auto *solver_iters = pkg->MutableParam<int>("step_solver_iterations");
  *solver_iters = 0;
  auto *nr_iters = pkg->MutableParam<int>("step_newt_iterations");
  *nr_iters = 0;

  const int nriter = pkg->Param<int>("nriter");
  const int local_nriter = pkg->Param<int>("local_nriter");
  pkg->UpdateParam("sim_time", tm);

  const int num_partitions = pmesh->DefaultNumPartitions();

  TaskCollection tc;
  TaskRegion &region = tc.AddRegion(num_partitions);
  auto step_change_norm = std::make_shared<AllReduce<Real>>();

  bool multilevel = pmesh->multilevel;

  for (int partition = 0; partition < num_partitions; ++partition) {
    TaskList &tl = region[partition];

    namespace ccbulk = cell_variables::cell_averaged::bulk;
    using all_vars_tl = parthenon::TypeList<Egroup, D, diag_loc>;

    auto partitions = pmesh->GetDefaultBlockPartitions();

    // This is the container that should have data from the end of the previous operator
    // split update for hydro or whatever. It is where we will apply the NR updates
    // that are stored in the container md_u
    auto &md_base = pmesh->mesh_data.GetOrAdd("base", partition);
    auto &md_base_comm = pmesh->mesh_data.AddShallow(
        "base_P1_comm", md_base, {Egroup::name(), temperature::name()});
    // md is the "base" container used in the linear solver, where all of the
    // data required to (implicitly) reconstruct the matrix is set and stored previous
    // to calling the linear solver.
    auto &md_matrix = pmesh->mesh_data.AddShallow(multigroup_base_container, md_base);
    // We make a shallow copy for the solution vector since we want to start with the
    // old value of Egroup as a guess and we want to store the solution in the base
    // container
    auto &md_u = pmesh->mesh_data.Add(multigroup_u_container, md_base, {Egroup::name()});
    // Need a unique container to store the rhs vector, but this only needs to contain the
    // NR update field
    auto &md_rhs =
        pmesh->mesh_data.Add(multigroup_rhs_container, md_base, {Egroup::name()});

    // Container to save the star state for calculating residuals
    auto &md_star =
        pmesh->mesh_data.Add("multigroup_star", md_base,
                             {Egroup::name(), temperature0::name(), Fgroup::name()});
    auto timer_guard_total =
        parthenon::TimingAccumulatorGuard(timings->GetOrAddAndRegister("Total", tl));
    // Save geometry
    auto geom_time = timings->GetOrAddAndRegister("Geometry", tl);
    geom_time->StartCollectingTasks();
    auto geometry_setup = none;
    for (int level = pmesh->GetGMGMinLevel(); level <= pmesh->GetGMGMaxLevel(); ++level) {
      auto partitions = pmesh->GetMultigridBlockPartitions(level);
      if (partition >= partitions.size()) continue;
      auto &md_base_mg = pmesh->mesh_data.Add("base", partitions[partition]);
      auto &md_matrix_mg =
          pmesh->mesh_data.AddShallow(multigroup_base_container, md_base_mg);
      geometry_setup =
          geometry_setup | tl.AddTask(none, Tasks::InitializeGeometry, md_matrix_mg);
    }
    geom_time->StopCollectingTasks();

    // First save the star state
    auto setup_time = timings->GetOrAddAndRegister("Setup", tl);
    setup_time->StartCollectingTasks();
    auto set_star = tl.AddTask(geometry_setup, Tasks::SaveStarState, md_base, md_star);
    auto [itl, nr_id] = tl.AddSublist(set_star, {1, nriter});
    { // Iterative tasks
      auto init = itl.AddTask(none, Tasks::InitializeRadiationQuantities, md_base,
                              md_star, md_matrix, md_rhs, dt);
      init = itl.AddTask(init, CalculateFluxes<Egroup, D, D>, md_matrix, md_base);
      init = itl.AddTask(init, CorrectRefinementBoundaryFluxes<Egroup>, md_base);
      init = AddFluxContribution<Egroup>(itl, init, md_matrix, md_base, md_rhs, -1.0);
      init =
          itl.AddTask(init, Tasks::SetLaggedRadiationMomentumFlux, md_star, md_base, dt);
      init = AddFluxContribution<Egroup>(itl, init, md_matrix, md_base, md_rhs, -1.0);
      setup_time->StopCollectingTasks();

      auto solve_setup_time = timings->GetOrAddAndRegister("Solve Setup", itl);
      solve_setup_time->StartCollectingTasks();
      auto init_mat = solver->AddSetupTasks(itl, init, partition, pmesh);
      solve_setup_time->StopCollectingTasks();
      auto solve_time = timings->GetOrAddAndRegister("Solve", itl);
      solve_time->StartCollectingTasks();
      auto solve = solver->AddTasks(itl, init_mat, partition, pmesh);
      solve_time->StopCollectingTasks();

      setup_time->StartCollectingTasks();
      auto update = itl.AddTask(solve, Tasks::ApplyUpdate, md_star, md_u, md_base);
      auto convergence = parthenon::solvers::utils::DotProduct<parthenon::TypeList<dTc>>(
          update, itl, step_change_norm.get(), md_base, md_base);

      auto comm = AddBoundaryExchangeTasks<parthenon::BoundaryType::any>(
          convergence, itl, md_base_comm, multilevel);
      auto sync = itl.AddTask(TaskQualifier::local_sync, comm,
                              []() { return TaskStatus::complete; });
      auto check = itl.AddTask(TaskQualifier::completion, sync, Tasks::Completion,
                               step_change_norm, pmesh, partition);
      setup_time->StopCollectingTasks();
    }

    auto final_time = timings->GetOrAddAndRegister("Finalize", tl);
    final_time->StartCollectingTasks();

    auto local_update =
        tl.AddTask(nr_id, Tasks::LocalSolve, md_star, md_base, dt, local_nriter);
    auto update_fluid =
        false ? local_update
              : tl.AddTask(local_update, Tasks::CorrectTotalEnergy, md_base);

    auto update_momentum =
        tl.AddTask(update_fluid, CalculateFluxes<Egroup, D, D>, md_matrix, md_base);
    update_momentum =
        tl.AddTask(update_momentum, CorrectRefinementBoundaryFluxes<Egroup>, md_base);
    update_momentum =
        tl.AddTask(update_momentum, Tasks::UpdateRadiationMomentum, md_star, md_base, dt);
    final_time->StopCollectingTasks();

    // Call con2prim
    auto c2p_time = timings->GetOrAddAndRegister("C2P", tl);
    c2p_time->StartCollectingTasks();
    namespace mdname = riot::container_names;
    auto &md = pmesh->mesh_data.GetOrAdd(mdname::u0, partition);
    auto int_derived = tl.AddTask(
        update_fluid, parthenon::Update::PreCommFillDerived<MeshData<Real>>, md.get());
    auto set_bc = AddBoundaryExchangeTasks<parthenon::BoundaryType::any>(int_derived, tl,
                                                                         md, multilevel);
    auto derive =
        tl.AddTask(set_bc, parthenon::Update::FillDerived<MeshData<Real>>, md.get());
    c2p_time->StopCollectingTasks();
  }

  return tc;
}

template <class temperature>
void MultiGroup<temperature>::MeshPostProblemGenerator(parthenon::Mesh *mesh,
                                                       parthenon::ParameterInput *pin,
                                                       parthenon::MeshData<Real> *md) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  static const auto desc = parthenon::MakePackDescriptor<temperature, Egroup>(md);
  auto pack = desc.GetPack(md);

  Multiphysics::FillInteriorDerived(md);

  BlackBodyHelper bb_helper(md->GetMeshPointer());
  const int ngroup = pack.GetSizeHost(0, Egroup());
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          for (int g = 0; g < ngroup; ++g) {
            const auto [B, dBdT] = bb_helper.GetBB(g, pack(b, temperature(), k, j, i));
            pack(b, Egroup(g), k, j, i) = B;
          }
        });
      });
}

template <class temperature>
Real MultiGroup<temperature>::EstimateTimestepMesh(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using namespace MultiGroupVars;

  auto pmesh = md->GetMeshPointer();

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const Real cfl = pkg->Param<Real>("cfl");
  const Real c_light = pkg->Param<Real>("c_light");
  const auto sim_time = pkg->Param<parthenon::SimTime>("sim_time");
  const Real dt_old = sim_time.dt;

  static const auto desc = parthenon::MakePackDescriptor<temperature, temperature0>(md);
  auto pack = desc.GetPack(md);
  const Real min_T = pkg->Param<Real>("timestep_min_temperature");
  const Real pct_change_target = pkg->Param<Real>("temperature_fractional_change_target");
  const Real Toffset = pkg->Param<Real>("timestep_temperature_scale");
  const Real min_F = 1.0 / pkg->Param<Real>("maximum_timestep_reduction_factor");

  const int ndim = pmesh->ndim;
  const Real sqrt3 = sqrt(3.0);
  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(), md, TE::CC);
  return RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, pack);
        const auto &coords = pack.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &mdt) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);

          mdt = std::min(mdt, sqrt3 * cfl * dx1 / c_light);
          if (ndim > 1) mdt = std::min(mdt, sqrt3 * cfl * dx2 / c_light);
          if (ndim > 2) mdt = std::min(mdt, sqrt3 * cfl * dx3 / c_light);

          const Real Told = pv(temperature0(), idx);
          const Real Tnew = pv(temperature(), idx);
          const Real dT = std::abs(Told - Tnew);
          const Real F = std::max((Tnew + Toffset) * pct_change_target / dT, min_F);
          if (dt_old > 0.0 && Tnew > min_T) mdt = std::min(mdt, dt_old * F);
        });
      });
}

template <class temperature>
AmrTag MultiGroup<temperature>::CheckRefinement(MeshBlockData<Real> *rc) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using namespace MultiGroupVars;
  static const auto desc =
      parthenon::MakePackDescriptor<temperature, ccbulk::rho, Egroup>(rc);
  auto pack = desc.GetPack(rc);
  auto pmesh = rc->GetMeshPointer();

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const Real amr_min_temperature = pkg->Param<Real>("amr_min_temperature");
  const Real amr_min_density = pkg->Param<Real>("amr_min_density");
  const Real amr_threshold = pkg->Param<Real>("amr_threshold");
  const Real derefine_radius = pkg->Param<Real>("derefine_radius");

  const int ndim = pmesh->ndim;
  // Reduce over the boundary-inset interior of this single block. The inset by one ghost
  // layer in each active direction is exactly IndexDomain::entire shrunk by halo = -1
  // (the md-constructor gates the halo on ndim). The reduction range stays halo-free
  // (halo_t == none_t), so inner_reduce accepts it, and the ±1 neighbor reads below
  // remain inside the entire domain.
  using TE = parthenon::TopologicalElement;
  using base_lt = RiotUtils::LoopType<LoopConstraint::SingleBlock>;
  auto base_space =
      base_lt::GetIndexSpace(IndexDomain::entire, -1, pack.GetNBlocks(), rc, TE::CC);
  // Memory offsets to the +/- neighbor in each direction (zero for collapsed dims), used
  // for the centered temperature differences via flat pack-view indexing.
  const auto di = base_space.GetDelta(X1DIR);
  const auto dj = base_space.GetDelta(X2DIR);
  const auto dk = base_space.GetDelta(X3DIR);

  auto minmax_space = base_space.template WithReducer<Kokkos::MinMax<Real>>();
  using minmax_rist = decltype(minmax_space);
  const auto minmax = RiotLoop::outer_reduce(
      minmax_space,
      KOKKOS_LAMBDA(const minmax_rist::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, pack);
        RiotLoop::inner_reduce(
            idx_range,
            [&](const auto idx, typename Kokkos::MinMax<Real>::value_type &lminmax) {
              const auto T = pv(temperature(), idx);
              const auto rho = pv(ccbulk::rho(), idx);
              const auto dTd1 = pv(temperature(), idx + di) - pv(temperature(), idx - di);
              const auto dTd2 = pv(temperature(), idx + dj) - pv(temperature(), idx - dj);
              const auto dTd3 = pv(temperature(), idx + dk) - pv(temperature(), idx - dk);
              const auto fig = sqrt(dTd1 * dTd1 + dTd2 * dTd2 + dTd3 * dTd3) / T;
              if (T > amr_min_temperature && rho > amr_min_density)
                lminmax.max_val = std::max(fig, lminmax.max_val);
            });
      });

  if (derefine_radius > 0.0) {
    auto max_space = base_space.template WithReducer<Kokkos::Max<int>>();
    using max_rist = decltype(max_space);
    const int max = RiotLoop::outer_reduce(
        max_space, KOKKOS_LAMBDA(const max_rist::idx_range_t &idx_range, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, int &lmax) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x1 = coords.template Xc<X1DIR>(k, j, i);
            const Real x2 = ndim > 1 ? coords.template Xc<X2DIR>(k, j, i) : 0.0;
            const Real x3 = ndim > 2 ? coords.template Xc<X3DIR>(k, j, i) : 0.0;
            if (sqrt(x1 * x1 + x2 * x2 + x3 * x3) > derefine_radius)
              lmax = std::max(-1, lmax);
          });
        });
    if (max < 0) return AmrTag::derefine;
  }

  if (minmax.max_val > amr_threshold) return AmrTag::refine;
  return AmrTag::derefine;
}

template struct MultiGroup<cell_variables::cell_averaged::bulk::temperature>;
template struct MultiGroup<cell_variables::cell_averaged::bulk::electron_temperature>;

} // namespace RadiationDiffusion
