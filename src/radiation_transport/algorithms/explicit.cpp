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
#include "radiation_transport/algorithms/explicit.hpp"
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

namespace Explicit {
const std::string pkg_name = "explicit";
const std::string input_block =
    std::string(RadiationShared::radiation_block) + "/" + pkg_name;
//----------------------------------------------------------------------------------------
//! \fn  void Explicit::Initialize
//! \brief Initializes the Explicit package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *materials) {
  auto explicit_pkg = std::make_shared<StateDescriptor>(pkg_name);
  namespace ccrad = cell_variables::cell_averaged::rad;
  Params &params = explicit_pkg->AllParams();

  // Algorithm-shared parameters (CFL, units, coupling, angular mesh, groups)
  const auto shared = RadiationShared::AddSharedParams(pin, materials, params);
  const int nangles = shared.nangles;
  const int ngroups = shared.ngroups;

  // Algorithm-shared Boundary Conditions (default, drive)
  RadiationBC::EnrollRadiationBC(explicit_pkg.get(), pin, params);

  // Algorithm-shared Initializations (none, zero, thermal)
  RadiationInit::EnrollRadiationInit(explicit_pkg.get(), pin, params);

  // User-Defined Metadata Flags
  auto MetadataExplicit = explicit_pkg->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetUserFlag("OperatorSplit");

  // Maximum permitted dt_ratio_hyperbolic for sub-cycled explicit integration
  params.Add(
      "dt_ratio_hyperbolic", pin->GetOrAddReal(input_block, "dt_ratio_hyperbolic", -1.0),
      "Limit timestep of explicit subcycling of thermal radiation transport by "
      "cfl * dt_ratio_hyperbolic * min_dx / c.  Setting to -1 does not permit thermal "
      "radiation transport to limit global timestep.");

  // RK subcycler
  const std::string integrator = pin->GetOrAddString(
      input_block, "integrator", "rk2",
      "Temporal integrator for explicit thermal radiation transport subcycler");
  params.Add("integrator", integrator);

  // Subcycling
  params.Add(
      "verbose", pin->GetOrAddInteger(input_block, "verbose", 0),
      "Sets verbosity of explicit thermal radiation transport package.  0: Report no "
      "diagnostics, 1: Only report at final subcycle, 2: Report every subcycle, 3: "
      "Report every subcycle and temperature root find failures");
  params.Add("current_iter", std::numeric_limits<int>::max(), true);
  params.Add("time", 0.0, Params::Mutability::Restart);

  // Radiation specific intensity I
  std::string control_field = ccrad::intensity::name();
  using namespace parthenon::refinement_ops;
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
              Metadata::WithFluxes, Metadata::Intensive, Metadata::Conserved,
              MetadataExplicit, MetadataOperatorSplit},
             std::vector<int>({ngroups * nangles}));
  explicit_pkg->AddField<ccrad::intensity>(m);

  // Divergence of angular fluxes (for non-Cartesian geometries)
  if (do_angular_fluxes) {
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataExplicit,
                  MetadataOperatorSplit},
                 std::vector<int>({ngroups * nangles}));
    explicit_pkg->AddField<ccrad::divfa>(m);
  }

  // Opacity fields
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                MetadataExplicit, MetadataOperatorSplit},
               std::vector<int>({ngroups}));
  explicit_pkg->AddField<ccrad::aa>(m);
  explicit_pkg->AddField<ccrad::ss>(m);

  // Moments and auxillary scratch fields
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataExplicit,
                MetadataOperatorSplit},
               std::vector<int>({ngroups}));
  explicit_pkg->AddField<ccrad::moments>(m);
  explicit_pkg->AddField<ccrad::s1>(m);
  explicit_pkg->AddField<ccrad::s2>(m);
  explicit_pkg->AddField<ccrad::s3>(m);

  // Explicit Timestep
  explicit_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Set Moments
  explicit_pkg->FillDerivedMesh = SetMomentsMesh;

  return explicit_pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  void Explicit::EstimateTimestep
//! \brief Estimate Timestep wrapper
Real EstimateTimestepMesh(MeshData<Real> *md) {
  const Real max_dt = 1.e-3 * std::numeric_limits<Real>::max();
  if (md->NumBlocks() == 0) return max_dt;

  auto pm = md->GetParentPointer();
  const Real dt_ratio_hyperbolic =
      pm->packages.Get(pkg_name)->Param<Real>("dt_ratio_hyperbolic");
  return (dt_ratio_hyperbolic < 0) ? max_dt : EstimateTimestep(md, dt_ratio_hyperbolic);
}

//----------------------------------------------------------------------------------------
//! \fn  void Explicit::EstimateTimestepMesh
//! \brief Computes Explicit timestep
Real EstimateTimestep(MeshData<Real> *md, const Real dt_ratio_hyperbolic) {
  auto pm = md->GetParentPointer();

  // Extract explicit package
  auto explicit_pkg = pm->packages.Get(pkg_name);

  // Resolved packages and indexing
  auto resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const int multid = ndim > 1;
  const int threed = ndim > 2;

  // Packing and indexing
  using parthenon::MakePackDescriptor;
  const std::vector<std::string> empty;
  static auto desc = MakePackDescriptor(resolved_pkgs.get(), empty);
  auto vmesh = desc.GetPack(md);

  // Units
  const auto unit_utils = explicit_pkg->Param<UnitUtils>("unit_utils");
  const Real cc = unit_utils.c;

  // Spatial minimum timestep
  Real min_dx = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "Explicit::EstimateSpatialTimestep",
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
    const auto agrid = GetAngularGridArrays(explicit_pkg);
    const auto &gflx = agrid.gflux;
    const auto &cp = agrid.cart_pos;
    const auto &numn = agrid.num_neighbors;
    const auto &indn = agrid.ind_neighbors;

    // Indexing
    const int nangles = explicit_pkg->Param<int>("nangles");

    // Angular flux indexing
    const auto NDIR = AngularFluxDirs();

    using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto idx_space =
        rt::GetIndexSpace(IndexDomain::interior, vmesh.GetNBlocks(), nangles, md);
    min_adt = rt::five_d(
        "Explicit::AngularTimestep", idx_space,
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

  // Set timestep
  const Real min_dt = std::min(min_sdt, min_adt);
  const Real cfl = explicit_pkg->Param<Real>("cfl");
  return cfl * min_dt * dt_ratio_hyperbolic;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection ExplicitTasks
//! \brief Assemble MeshData registers and create/return Explicit TaskCollection
TaskCollection ExplicitTasks(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  // Extract Explicit package
  auto explicit_pkg = pm->packages.Get(pkg_name);

  // Get variable names for MeshData subsets
  std::vector<std::string> explicit_names =
      pm->GetVariableNames(explicit_pkg->GetMetadataFlag());

  // Get unsplit variable names if coupling enabled
  std::vector<std::string> unsplit_names;
  if (explicit_pkg->Param<bool>("coupling")) {
    parthenon::Metadata::FlagCollection flags_unsplit;
    flags_unsplit.Exclude(parthenon::Metadata::GetUserFlag("OperatorSplit"));
    unsplit_names = pm->GetVariableNames(flags_unsplit);
  }

  // Assemble MeshData subsets
  namespace mdname = container_names;
  namespace ccrad = cell_variables::cell_averaged::rad;
  std::vector<std::string> opac_names = pm->GetVariableNames(
      std::vector<std::string>{ccrad::aa::name(), ccrad::ss::name()}, std::vector<int>{});
  auto &base = pm->mesh_data.Get();
  auto &r0 = pm->mesh_data.AddShallow(mdname::r0, base, explicit_names);
  auto &r1 = pm->mesh_data.Add(mdname::r1, r0);
  auto &a0 = pm->mesh_data.AddShallow(mdname::a0, base, unsplit_names);
  auto &ro = pm->mesh_data.AddShallow(mdname::ro, base, opac_names);

  // Compute number of subcycles
  Real min_dt = EstimateTimestep(base.get(), 1.0);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &min_dt, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif
  const int nsteps = static_cast<int>(std::ceil(dt / min_dt));
  const Real scdt = dt / nsteps;

  return ExplicitTransport(pm, nsteps, tm.time, scdt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection ExplicitTransport
//! \brief Generates TaskCollection for Explicit Transport
TaskCollection ExplicitTransport(Mesh *pmesh, const int nsteps, const Real time,
                                 const Real dt) {
  using namespace ::parthenon::Update;

  TaskCollection tc;
  namespace mdname = container_names;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Extract Explicit params
  auto explicit_pkg = pmesh->packages.Get(pkg_name);
  const bool affect_fluid = explicit_pkg->Param<bool>("affect_fluid");

  // Subcyler integrator
  auto int_string = explicit_pkg->Param<std::string>("integrator");
  static std::unique_ptr<parthenon::LowStorageIntegrator> integrator =
      std::make_unique<parthenon::LowStorageIntegrator>(int_string);

  // Construct ExplicitTransport TaskList
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto &r0 = pmesh->mesh_data.GetOrAdd(mdname::r0, i);
    auto &r1 = pmesh->mesh_data.GetOrAdd(mdname::r1, i);
    auto &a0 = pmesh->mesh_data.GetOrAdd(mdname::a0, i);
    auto &ro = pmesh->mesh_data.GetOrAdd(mdname::ro, i);

    // Start receives on MeshData registers
    auto srec = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, ro);
    srec = tl.AddTask(srec, parthenon::StartReceiveBoundBufs<any>, r0);
    if (affect_fluid) srec = tl.AddTask(srec, parthenon::StartReceiveBoundBufs<any>, a0);
    srec = tl.AddTask(srec, parthenon::StartReceiveFluxCorrections, r0);

    // Set opacities
    auto set_opac = tl.AddTask(srec, UpdateOpacities, base.get());
    auto opac_bc =
        parthenon::AddBoundaryExchangeTasks(set_opac, tl, ro, pmesh->multilevel);

    // Prepare for iterations
    auto prep_sc = tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                              opac_bc, PrepareForSubcycler, pmesh, time);

    // Execute RK subcycler
    // NOTE(@pdmullen): Should never reach nsteps + 1 limit
    auto [cycler, cycler_id] = tl.AddSublist(prep_sc, {1, nsteps + 1});
    auto subcycle = CreateExplicitTaskList(none, i, pmesh, cycler, cycler_id,
                                           integrator.get(), base, r0, r1, nsteps, dt);

    // Update fluid state if radiation affects fluid
    TaskID aux = none;
    if (affect_fluid) {
      auto pte = tl.AddTask(
          cycler_id,
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
          a0.get());
      auto u_bc = parthenon::AddBoundaryExchangeTasks(pte, tl, a0, pmesh->multilevel);
      aux = tl.AddTask(u_bc, FillDerived<MeshData<Real>>, a0.get());
    }

    // Update moments
    auto moments = tl.AddTask(
        cycler_id | aux,
        [](MeshData<Real> *md) {
          SetMomentsMesh(md);
          return TaskStatus::complete;
        },
        r0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskID CreateExplicitTaskList
//! \brief
TaskID CreateExplicitTaskList(const TaskID &begin, const int i, Mesh *pmesh,
                              TaskList &cycler, TaskID cycler_id,
                              parthenon::LowStorageIntegrator *integrator,
                              std::shared_ptr<MeshData<Real>> base,
                              std::shared_ptr<MeshData<Real>> r0,
                              std::shared_ptr<MeshData<Real>> r1, const int nsteps,
                              const Real dt) {
  const auto any = parthenon::BoundaryType::any;

  // r1 <-- r0
  auto init_rk = cycler.AddTask(
      TaskQualifier::local_sync, begin,
      sparse_update::DeepCopyData<parthenon::MetadataFlag, MeshData<Real>>,
      std::vector<MetadataFlag>({Metadata::Independent}), r1.get(), r0.get());

  // RK integration
  TaskID rk = init_rk;
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    auto flx = cycler.AddTask(rk, CalculateFluxes, r0.get());
    auto send_flx = cycler.AddTask(flx | rk, parthenon::LoadAndSendFluxCorrections, r0);
    auto recv_flx = cycler.AddTask(rk, parthenon::ReceiveFluxCorrections, r0);
    auto set_flx = cycler.AddTask(recv_flx | rk, parthenon::SetFluxCorrections, r0);
    auto update = cycler.AddTask(flx | set_flx | rk, Update, r0.get(), r1.get(),
                                 integrator->gam0[stage - 1], integrator->gam1[stage - 1],
                                 integrator->beta[stage - 1] * dt);
    auto rhs = cycler.AddTask(update | rk, ApplyRHS, base.get(),
                              integrator->beta[stage - 1] * dt);
    auto set_bc =
        parthenon::AddBoundaryExchangeTasks(rhs | rk, cycler, r0, pmesh->multilevel);
    rk = cycler.AddTask(TaskQualifier::local_sync, set_bc | rk,
                        []() { return TaskStatus::complete; });
  }

  // Increment subcycle counter and evaluation completion function
  auto inc = cycler.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                            rk, IncrementCounter, pmesh, dt);
  auto complete = cycler.AddTask(TaskQualifier::completion, inc, CompletionFunction, i,
                                 r0.get(), r1.get(), nsteps);
  return cycler_id;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Explicit::UpdateOpacities
//! \brief Set bulk opacities across the mesh
TaskStatus UpdateOpacities(MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return TaskStatus::complete;

  auto pm = md->GetParentPointer();
  auto explicit_pkg = pm->packages.Get(pkg_name);
  const bool coupling = explicit_pkg->Param<bool>("coupling");
  const bool fixed_pgen_opac = explicit_pkg->Param<bool>("fixed_pgen_opac");
  if (!(coupling) || fixed_pgen_opac) return TaskStatus::complete;

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ngroups = explicit_pkg->Param<int>("ngroups");

  // Opacity parameters
  auto &mat_pkg = pm->packages.Get("materials");
  const auto &opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
  const auto &opac_s = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacS>>("d.d.opac_s");
  const auto &opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");

  // Packing
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  namespace ccrad = cell_variables::cell_averaged::rad;
  static auto desc =
      MakePackDescriptor<ccrad::aa, ccrad::ss, cm::rho, ccmat::volume_fraction,
                         ccbulk::temperature>(resolved_pkgs.get());
  auto pack = desc.GetPack(md);

  // Set bulk opacities
  auto idx_space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, pack.GetNBlocks(), ngroups, md);
  RiotFlatLoop::five_d(
      "UpdateOpacities", idx_space,
      KOKKOS_LAMBDA(const int &b, const int gg, const int &k, const int &j,
                    const int &i) {
        Real &aa = pack(b, ccrad::aa(gg), k, j, i) = 0.0;
        Real &ss = pack(b, ccrad::ss(gg), k, j, i) = 0.0;
        const Real &temp = pack(b, ccbulk::temperature(), k, j, i);
        for (int m = 0; m < pack.GetSize(b, cm::rho()); ++m) {
          const Real &rhom = pack(b, cm::rho(m), k, j, i);
          const Real &vfracm = pack(b, ccmat::volume_fraction(m), k, j, i);
          const int &mat_id = pack(b, cm::rho(m)).sparse_id;
          const int &phase_id = pack(b, cm::rho(m)).v;
          const int opac_id = opac_from_matid(mat_id) + phase_id;
          const Real aam =
              (rhom > 0) ? opac_a(opac_id).AbsorptionCoefficient(rhom, temp, gg) : 0.0;
          const Real ssm =
              (rhom > 0) ? opac_s(opac_id).ScatteringCoefficient(rhom, temp, gg) : 0.0;
          aa += vfracm * aam;
          ss += vfracm * ssm;
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus CalculateFluxes
//! \brief Computes fluxes
TaskStatus CalculateFluxes(MeshData<Real> *r0) {
  if (r0->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = r0->GetParentPointer();
  auto explicit_pkg = pm->packages.Get(pkg_name);

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const int ndir = 2 * ndim;
  const int multid = ndim > 1;
  const int threed = ndim > 2;
  const int ngroups = explicit_pkg->Param<int>("ngroups");
  const int nangles = explicit_pkg->Param<int>("nangles");
  IndexRange ib = r0->GetBoundsI(IndexDomain::interior);
  IndexRange jb = r0->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = r0->GetBoundsK(IndexDomain::interior);

  // Additional extractions
  const auto unit_utils = explicit_pkg->Param<UnitUtils>("unit_utils");
  const Real cc = unit_utils.c;
  const auto agrid = GetAngularGridArrays(explicit_pkg);
  const auto &cp = agrid.cart_pos;
  const auto &gflx = agrid.gflux;
  const auto &arcw = agrid.arc_weights;
  const auto &wght = agrid.weights;
  const auto &numn = agrid.num_neighbors;
  const auto &indn = agrid.ind_neighbors;

  // Spatial fluxes
  auto futils = explicit_pkg->Param<FluxUtils>("flux_utils");
  const Real beta = futils.beta;
  const Real tmax = futils.taumax;

  // Angular flux indexing
  const auto NDIR = AngularFluxDirs();

  // Packing
  namespace ccr = cell_variables::cell_averaged::rad;
  static auto desc_ii = MakePackDescriptor<ccr::intensity>(
      pm->resolved_packages.get(), {}, {parthenon::PDOpt::WithFluxes});
  static auto desc_da = MakePackDescriptor<ccr::divfa>(pm->resolved_packages.get());
  static auto desc_op = MakePackDescriptor<ccr::aa, ccr::ss>(pm->resolved_packages.get());
  auto ii = desc_ii.GetPack(r0);
  auto da = desc_da.GetPack(r0);
  auto op = desc_op.GetPack(r0);

  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, ii.GetNBlocks(),
                                               ngroups * nangles, r0);
  RiotFlatLoop::five_d(
      "CalculateFluxes", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &n, const int &k, const int &j,
                    const int &i) {
        // Extract coordinates
        auto &coords = ii.GetCoordinates(b);

        // Indexing gymnastics
        const int gg = GI(nangles, n);
        const int aa = AI(nangles, n);

        // Calculate fluxes
        Real cn = cc * cp(aa, NDIR[X1DIR - 1]);
        Real dx = coords.CellWidth<X1DIR>(k, j, i);
        ii.flux(b, X1DIR, n, k, j, i) =
            SpatialFlux<X1DIR>(ii, cn, b, n, k, j, i, op, gg, beta, dx, tmax);
        if (i == ib.e) {
          dx = coords.CellWidth<X1DIR>(k, j, i + 1);
          ii.flux(b, X1DIR, n, k, j, i + 1) =
              SpatialFlux<X1DIR>(ii, cn, b, n, k, j, i + 1, op, gg, beta, dx, tmax);
        }

        if (multid) { // Cartesian or RZ
          cn = cc * cp(aa, NDIR[X2DIR - 1]);
          dx = coords.CellWidth<X2DIR>(k, j, i);
          ii.flux(b, X2DIR, n, k, j, i) =
              SpatialFlux<X2DIR>(ii, cn, b, n, k, j, i, op, gg, beta, dx, tmax);
          if (j == jb.e) {
            dx = coords.CellWidth<X2DIR>(k, j + 1, i);
            ii.flux(b, X2DIR, n, k, j + 1, i) =
                SpatialFlux<X2DIR>(ii, cn, b, n, k, j + 1, i, op, gg, beta, dx, tmax);
          }
        }

        if (threed) { // 3D Cartesian only
          cn = cc * cp(aa, NDIR[X3DIR - 1]);
          dx = coords.CellWidth<X3DIR>(k, j, i);
          ii.flux(b, X3DIR, n, k, j, i) =
              SpatialFlux<X3DIR>(ii, cn, b, n, k, j, i, op, gg, beta, dx, tmax);
          if (k == kb.e) {
            const Real &dx3p = coords.CellWidth<X3DIR>(k + 1, j, i);
            ii.flux(b, X3DIR, n, k + 1, j, i) =
                SpatialFlux<X3DIR>(ii, cn, b, n, k + 1, j, i, op, gg, beta, dx, tmax);
          }
        }

        // Compute angular flux divergence
        [[maybe_unused]] auto &da_ = da;
        [[maybe_unused]] auto &wght_ = wght;
        [[maybe_unused]] auto &numn_ = numn;
        [[maybe_unused]] auto &gflx_ = gflx;
        [[maybe_unused]] auto &indn_ = indn;
        [[maybe_unused]] auto &arcw_ = arcw;
        if constexpr (do_angular_fluxes) {
          Real &divfa = da_(b, n, k, j, i) = 0.0;
          const Real mcw_ix1 = -cc * InverseRadiusForAngularFlux(coords, i) / wght_(aa);
          for (int nb = 0; nb < numn_(aa); ++nb) {
            const Real na = mcw_ix1 * gflx_(aa, nb);
            const Real &iu = (na > 0) ? ii(b, n, k, j, i)
                                      : ii(b, GAI(nangles, gg, indn_(aa, nb)), k, j, i);
            divfa += na * iu * arcw_(aa, nb);
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Update
//! \brief Apply RK update logic
TaskStatus Update(MeshData<Real> *r0, MeshData<Real> *r1, const Real g0, const Real g1,
                  const Real bdt) {
  if (r0->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = r0->GetParentPointer();
  auto explicit_pkg = pm->packages.Get(pkg_name);

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ndim = pm->ndim;
  const int ndir = 2 * ndim;
  const int multid = ndim > 1;
  const int threed = ndim > 2;
  const int ngroups = explicit_pkg->Param<int>("ngroups");
  const int nangles = explicit_pkg->Param<int>("nangles");

  // Packing
  namespace ccr = cell_variables::cell_averaged::rad;
  static auto desc0 = MakePackDescriptor<ccr::intensity>(pm->resolved_packages.get(), {},
                                                         {parthenon::PDOpt::WithFluxes});
  static auto desc1 = MakePackDescriptor<ccr::intensity>(pm->resolved_packages.get());
  static auto desca = MakePackDescriptor<ccr::divfa>(pm->resolved_packages.get());
  auto i0 = desc0.GetPack(r0);
  auto i1 = desc1.GetPack(r1);
  auto da = desca.GetPack(r0);

  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, i0.GetNBlocks(),
                                               ngroups * nangles, r0);
  RiotFlatLoop::five_d(
      "Explicit::RKUpdate", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &n, const int &k, const int &j,
                    const int &i) {
        // Extract coordinates
        auto &coords = i0.GetCoordinates(b);

        // Flux divergence
        // clang-format off
        const Real &ap1 = coords.FaceArea<X1DIR>(k, j, i + 1);
        const Real &a1 = coords.FaceArea<X1DIR>(k, j, i);
        Real df = (i0.flux(b, X1DIR, n, k, j, i + 1) * ap1 -
                   i0.flux(b, X1DIR, n, k, j, i    ) * a1);
        if (multid) { // Cartesian or RZ
          const Real &ap2 = coords.FaceArea<X2DIR>(k, j + 1, i);
          const Real &a2 = coords.FaceArea<X2DIR>(k, j, i);
          df += (i0.flux(b, X2DIR, n, k, j + 1, i) * ap2 -
                 i0.flux(b, X2DIR, n, k, j    , i) * a2);
        }
        if (threed) { // 3D Cartesian only
          const Real &ap3 = coords.FaceArea<X3DIR>(k + 1, j, i);
          const Real &a3 = coords.FaceArea<X3DIR>(k, j, i);
          df += (i0.flux(b, X3DIR, n, k + 1, j, i) * ap3 -
                 i0.flux(b, X3DIR, n, k    , j, i) * a3);
        }
        df /= coords.CellVolume(k, j, i);
        [[maybe_unused]] auto &da_ = da;
        if constexpr (do_angular_fluxes) df += da_(b, n, k, j, i);
        // clang-format on

        // Apply RK logic
        Real &ii = i0(b, n, k, j, i);
        ii = std::max(g0 * ii + g1 * i1(b, n, k, j, i) - bdt * df, 0.0);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ApplyRHS
//! \brief Computes explicit source term due to emission, absorption, and scattering
TaskStatus ApplyRHS(MeshData<Real> *md, const Real dt) {
  if (md->NumBlocks() == 0) return TaskStatus::complete;
  using parthenon::MakePackDescriptor;

  // Extract explicit package
  auto pm = md->GetParentPointer();
  auto &explicit_pkg = pm->packages.Get(pkg_name);
  const bool coupling = explicit_pkg->Param<bool>("coupling");
  if (!(coupling)) return TaskStatus::complete;

  // troot verbosity flag
  const bool verbose = (explicit_pkg->Param<int>("verbose") == 3);

  // Resolved packages and indexing
  auto &resolved_pkgs = pm->resolved_packages;
  const int ngroups = explicit_pkg->Param<int>("ngroups");
  const int nangles = explicit_pkg->Param<int>("nangles");

  // Utilities
  const auto unit_utils = explicit_pkg->Param<UnitUtils>("unit_utils");
  const auto root_utils = explicit_pkg->Param<RootUtils>("root_utils");
  const Real cdt = unit_utils.c * dt;

  // Multigroup bins
  const auto fbnd = *(explicit_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

  // Extract explicit params
  const bool affect_fluid = explicit_pkg->Param<bool>("affect_fluid");
  const bool fixed_temp_rhs = explicit_pkg->Param<bool>("fixed_temp_rhs");
  const auto agrid = GetAngularGridArrays(explicit_pkg);
  const auto &cp = agrid.cart_pos;
  const auto &wght = agrid.weights;

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
  static auto desc =
      MakePackDescriptor<ccmat::rho, cm::rho, ccbulk::temperature,
                         ccbulk::internal_energy, ccrad::aa, ccrad::ss, ccrad::s1,
                         ccrad::s2, ccrad::s3, ccrad::intensity>(resolved_pkgs.get());
  auto pack = desc.GetPack(md);

  // Apply explicit source term
  auto idx_space =
      RiotFlatLoop::GetIndexSpace(IndexDomain::interior, pack.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "ApplyRHS", idx_space,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Capture fluid state this MeshBlock
        const int nmat1 = pack.GetSize(b, ccmat::rho()) - 1;
        Real &tbase = pack(b, ccbulk::temperature(), k, j, i);
        Real tadv = tbase;

        for (int gg = 0; gg < ngroups; ++gg) {
          // Set opacity weighted timesteps
          const Real cdtsiga = cdt * pack(b, ccrad::aa(gg), k, j, i);
          const Real cdtsigs = cdt * pack(b, ccrad::ss(gg), k, j, i);
          const Real cdtsigt = cdtsiga + cdtsigs;

          // Set polynomial coefficients (A1, A2, A3)
          Real &aa1 = pack(b, ccrad::s1(gg), k, j, i) = 0.0;
          Real &aa2 = pack(b, ccrad::s2(gg), k, j, i) = 0.0;
          Real &aa3 = pack(b, ccrad::s3(gg), k, j, i) = 0.0;
          for (int aa = 0; aa < nangles; ++aa) {
            const int n = GAI(nangles, gg, aa);
            const Real &ii = pack(b, ccrad::intensity(n), k, j, i);
            const Real tmp = wght(aa) / (1.0 + cdtsigt);
            aa1 += tmp;
            aa2 += ii * tmp;
          }
          aa3 = 1.0 / (1.0 - aa1 * cdtsigs);
          aa1 *= cdtsiga;
        }

        // Compute new gas temperature
        const bool success =
            (fixed_temp_rhs ||
             AdvancedTemperature<false>(tadv, tbase, verbose, tflr, root_utils,
                                        unit_utils, cdt, ngroups, fbnd, pack, nmat1, eos,
                                        eos_map, pack, b, k, j, i));
        tadv = (success) ? tadv : tbase;

        // Update the specific intensity
        if (success) {
          Real delta_er = 0.0;
          for (int gg = 0; gg < ngroups; ++gg) {
            // Set opacity weighted timesteps
            const Real cdtsiga = cdt * pack(b, ccrad::aa(gg), k, j, i);
            const Real cdtsigs = cdt * pack(b, ccrad::ss(gg), k, j, i);
            const Real cdtsigt = cdtsiga + cdtsigs;

            // Calculate polynomial coefficients (A1, A2, A3)
            const Real &aa1 = pack(b, ccrad::s1(gg), k, j, i);
            const Real &aa2 = pack(b, ccrad::s2(gg), k, j, i);
            const Real &aa3 = pack(b, ccrad::s3(gg), k, j, i);

            // Set advanced stage specific intensity
            Real er_old = 0.0;
            Real er_new = 0.0;
            const Real ee = Emissivity(gg, tadv, fbnd, ngroups, unit_utils);
            const Real jj = (aa1 * ee + aa2) * aa3;
            for (int aa = 0; aa < nangles; ++aa) {
              const int n = GAI(nangles, gg, aa);
              Real &ii = pack(b, ccrad::intensity(n), k, j, i);
              // clang-format off
              er_old += ii * wght(aa); // pre-coupling 0th moment contribution
              ii = std::max(ii + ((cdtsigs * jj +
                                   cdtsiga * ee -
                                   cdtsigt * ii) / (1.0 + cdtsigt)), 0.0);
              er_new += ii * wght(aa); // post-coupling 0th moment contribution
              // clang-format on
            }
            delta_er += er_old - er_new;
          }

          // Feedback
          if (affect_fluid) {
            tbase = tadv; // overwritten upon PTE
            pack(b, ccbulk::internal_energy(), k, j, i) += delta_er;
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus PrepareForSubcycler
//! \brief Reset the subcycler counter
TaskStatus PrepareForSubcycler(Mesh *pmesh, const Real time) {
  // reset counter
  auto explicit_pkg = pmesh->packages.Get(pkg_name);
  explicit_pkg->UpdateParam("current_iter", 1);
  explicit_pkg->UpdateParam("time", time);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus IncrementCounter
//! \brief Increment the subcycler counter
TaskStatus IncrementCounter(Mesh *pmesh, const Real dt) {
  // increment counter
  auto explicit_pkg = pmesh->packages.Get(pkg_name);
  const int iter_counter = explicit_pkg->Param<int>("current_iter");
  explicit_pkg->UpdateParam("current_iter", iter_counter + 1);

  // Update time
  // TODO(@pdmullen): Fix time centering...
  const Real time = explicit_pkg->Param<Real>("time");
  explicit_pkg->UpdateParam("time", time + dt);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus CompletionFunction
//! \brief Complete iteration, check for continued subcycling or finalize
TaskStatus CompletionFunction(int i, MeshData<Real> *r0, MeshData<Real> *r1,
                              const int nsteps) {
  if (r0->NumBlocks() == 0) return TaskStatus::complete;

  // Extract iter/iter_limit and residual/error threshhold
  auto pm = r0->GetParentPointer();
  auto explicit_pkg = pm->packages.Get(pkg_name);
  const int iter_counter = explicit_pkg->Param<int>("current_iter");
  const int verbose = explicit_pkg->Param<int>("verbose");
  PARTHENON_REQUIRE(iter_counter - 1 <= nsteps, "Too many iterations in subcycler!");

  // Report
  const bool report = (verbose) && ((iter_counter - 1 == nsteps) || (verbose == 2));
  if (i == 0 && Globals::my_rank == 0 && report) {
    printf("(Explicit) subcycle: %d nsteps: %d\n", iter_counter - 1, nsteps);
  }

  // Complete or iterate (thereby deep-copying r1 <-- r0)
  if (iter_counter - 1 == nsteps) {
    return TaskStatus::complete;
  } else {
    return TaskStatus::iterate;
  }
}

} // namespace Explicit
