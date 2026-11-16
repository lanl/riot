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

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ports-of-call/robust_utils.hpp>

#include <singularity-eos/eos/eos.hpp>

#include <globals.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>

#include "hydro.hpp"
#include "materials/materials.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/pte_closure.hpp"
#include "microphysics/strength_models.hpp"
#include "reconstruction/reconstruction.hpp"
#include "riemann.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "riot_utils/sparse_update.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexSplit;
using parthenon::ParArray1D;
using parthenon::team_mbr_t;

namespace Hydro {

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Initialize
//! \brief Initializes the hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *mat_pkg) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  const int nghost = parthenon::Globals::nghost;
  auto hydro = std::make_shared<StateDescriptor>("hydro");
  Params &params = hydro->AllParams();

  const bool do_ionization = pin->GetOrAddBoolean("physics", "ionization", false);
  const bool do_mhd = pin->GetOrAddBoolean(
      "hydro", "mhd", false, "Evolve the ideal MHD equations with constrained transport");
  params.Add("do_mhd", do_mhd);
  bool do_plasma_viscosity = false;
  if (do_ionization) {
    do_plasma_viscosity = pin->GetOrAddBoolean("ionization", "plasma_viscosity", false);
  }

  // Reconstruction Algorithm
  std::vector<std::string> recon_name_list;
  for (const auto &[k, v] : RiotReconstruction::NAME_MAP) {
    auto name = k;
    RiotUtils::ToLower(name);
    recon_name_list.push_back(name);
  }
  std::string recon_name = pin->GetOrAddString("hydro", "recon", "plm", recon_name_list,
                                               "Reconstruction method to use");
  std::string vfrac_recon_name =
      pin->GetOrAddString("hydro", "vfrac_recon", recon_name, recon_name_list,
                          "Reconstruction to use for the volume fractions, if different "
                          "from the other variables");
  auto GetReconTag = [&](std::string &recon_name) {
    RiotUtils::ToUpper(recon_name); // make case insensitive
    if (RiotReconstruction::NAME_MAP.count(recon_name) == 0) {
      PARTHENON_THROW("Unknown reconstruction " + recon_name);
    }
    auto recon_tag = RiotReconstruction::NAME_MAP.at(recon_name);
    int stencil_width = RiotReconstruction::STENCIL_WIDTH.at(recon_tag);
    stencil_width += (do_ionization && do_plasma_viscosity);
    // CT corner EMFs need face EMFs from one transverse ghost row
    stencil_width += do_mhd;
    PARTHENON_REQUIRE(nghost > stencil_width,
                      "Must have enough ghosts for reconstruction stencil " + recon_name);
    return std::make_pair(stencil_width, recon_tag);
  };
  auto [recon_width, recon_tag] = GetReconTag(recon_name);
  auto [vfrac_recon_width, vfrac_recon_tag] = GetReconTag(vfrac_recon_name);
  int stencil_width = std::max(recon_width, vfrac_recon_width);
  params.Add("recon", recon_tag);
  params.Add("vfrac_recon", vfrac_recon_tag);
  params.Add("stencil_width", stencil_width);

  // PLM slope limit theta in slope = minmod(central, theta * one-sided): 1 is minmod,
  // 2 is MC. Applies to the bulk flow variables and magnetic field; material density
  // and internal energy use min(theta, 1.5).
  const Real plm_slope_limit = pin->GetOrAddReal(
      "hydro", "plm_slope_limit", THETA,
      "PLM slope limit in [1, 2] (1 = minmod, 2 = MC) for the flow variables");
  PARTHENON_REQUIRE_THROWS(plm_slope_limit >= 1.0 && plm_slope_limit <= 2.0,
                           "hydro/plm_slope_limit must be in [1, 2]");
  params.Add("plm_slope_limit", plm_slope_limit);

  // CFL is 0.8 by default
  Real cfl = pin->GetOrAddReal("hydro", "cfl", 0.8, "Courant-Friedrichs-Lewy number");
  params.Add("cfl", cfl);

  // Refinement Strategy
  std::string refinement = pin->GetOrAddString("parthenon/mesh", "refinement", "none");
  bool adaptive = (!refinement.compare("adaptive") ? true : false);
  bool amr_int = pin->GetOrAddBoolean("hydro", "amr_interface", true,
                                      "Refine on material interfaces");

  // Choose Riemann Solver
  // TODO(JMM): Move Carbuncle correction into HLLC solver
  const bool user_set_solver = pin->DoesParameterExist("hydro", "riemann");
  std::string solver = pin->GetOrAddString(
      "hydro", "riemann", "hllc",
      std::vector<std::string>{"hllc", "hllcf", "chllc", "lhllc", "hll", "hlld"},
      "Riemann solver to use (forced to hlld when MHD is enabled)");
  if (do_mhd) {
    if (solver != "hlld" && user_set_solver && parthenon::Globals::my_rank == 0) {
      PARTHENON_WARN("MHD is enabled, so the hlld Riemann solver will be used instead "
                     "of " +
                     solver);
    }
    solver = "hlld";
  }
  if (solver == "hlld") {
    PARTHENON_REQUIRE_THROWS(do_mhd, "The hlld Riemann solver requires <hydro>/mhd=true");
    params.Add("riemann_solver", RiemannSolver::hlld);
  } else if (solver == "hllc") {
    params.Add("riemann_solver", RiemannSolver::hllc);
  } else if (solver == "hllcf") {
    if (parthenon::Globals::my_rank == 0) {
      PARTHENON_WARN("The low-Mach correction in HLLCF is still experimental and may "
                     "cause stability issues");
    }
    params.Add("riemann_solver", RiemannSolver::hllcf);
  } else if (solver == "chllc") {
    params.Add("riemann_solver", RiemannSolver::chllc);
  } else if (solver == "lhllc") {
    if (parthenon::Globals::my_rank == 0) {
      PARTHENON_WARN("The low-Mach correction in LHLLC is still experimental and may "
                     "cause stability issues");
    }
    params.Add("riemann_solver", RiemannSolver::lhllc);
  } else if (solver == "hll") {
    params.Add("riemann_solver", RiemannSolver::hll);
  } else {
    PARTHENON_THROW("Invalid Riemann solver option");
  }

  // Thornber's low-Mach correction
  bool lm_correction =
      pin->GetOrAddBoolean("hydro", "lm_correction", false,
                           "Apply the Thornber low-Mach correction. WARNING: "
                           "experimental and may cause stability issues.");
  if (lm_correction) {
    if (parthenon::Globals::my_rank == 0) {
      PARTHENON_WARN("The Thornber low-Mach correction is still experimental and may "
                     "cause stability issues");
    }
  }
  params.Add("lm_correction", lm_correction);

  // Bulk density
  Metadata m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Conserved,
                         Metadata::Derived, Metadata::OneCopy});
  hydro->AddField<ccbulk::rho>(m);

  // Bulk total material energy
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::WithFluxes});
  m.Associate(ccbulk::internal_energy::name());
  hydro->AddField<ccbulk::total_material_energy>(m);

  // Bulk Momentum
  std::vector<int> vel_arr_size(1, 3);
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Vector, Metadata::WithFluxes},
               vel_arr_size);
  m.Associate(ccbulk::velocity::name());
  hydro->AddField<ccbulk::momentum>(m);

  // Bulk Velocity
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector, Metadata::Derived,
                Metadata::OneCopy, Metadata::FillGhost},
               vel_arr_size);
  hydro->AddField<ccbulk::velocity>(m);

  // Bulk Temperature
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::ForceRemeshComm, Metadata::Restart});
  hydro->AddField<ccbulk::temperature>(m);

  // Bulk Pressure
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy,
                Metadata::FillGhost});
  hydro->AddField<ccbulk::pressure>(m);

  // Bulk Volumetric Internal Energy and Bulk Modulus
  m = Metadata(
      {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  hydro->AddField<ccbulk::internal_energy>(m);
  hydro->AddField<ccbulk::bulk_modulus>(m);

  // Face Signal Speeds
  m = Metadata({Metadata::Face, Metadata::Derived, Metadata::Flux, Metadata::OneCopy,
                Metadata::CellMemAligned});
  hydro->AddField<ccbulk::face_signal>(m);

  // Max Signal Speed
  std::vector<int> sig_size(1, 3);
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                Metadata::ForceRemeshComm, Metadata::Restart},
               sig_size);
  hydro->AddField<ccbulk::max_signal>(m);

  // Strength Fields
  const bool do_strength =
      pin->GetOrAddBoolean("physics", "strength", false, "Enable material strength");
  if (do_strength) {
    // Bulk Shear Modulus
    m = Metadata(
        {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
    hydro->AddField<ccbulk::shear_modulus>(m);
  }

  if (do_strength || do_ionization) {
    // strain-rate tensor
    std::vector<int> strain_rate_shape(1, 6);
    m = Metadata(
        {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy},
        strain_rate_shape);
    hydro->AddField<ccbulk::strain_rate>(m);
  }

  const bool store_vf = do_ionization || do_strength;
  params.Add("store_vf", store_vf);
  if (store_vf) {
    // Face Velocities (to compute strain rate)
    std::vector<int> face_vel_shape(1, 9);
    m = Metadata(
        {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy},
        face_vel_shape);
    hydro->AddField<ccbulk::face_velocity>(m);
  }

  if (do_mhd) {
    PARTHENON_REQUIRE_THROWS(parthenon::IsCoord<parthenon::UniformCartesian>() ||
                                 parthenon::IsCoord<parthenon::UniformCylindrical>(),
                             "MHD is only implemented for uniform Cartesian and "
                             "cylindrical coordinates");
    PARTHENON_REQUIRE_THROWS(!do_strength, "MHD is incompatible with material strength");
    PARTHENON_REQUIRE_THROWS(!do_ionization, "MHD is incompatible with ionization");

    // Face-centered magnetic field, evolved with constrained transport. The edge fluxes
    // automatically allocated by WithFluxes hold the EMFs. Node memory layout is
    // required for FillGhost on face fields, so this field must never be accessed with
    // the flat (cell-aligned) pack views.
    m = Metadata({Metadata::Face, Metadata::Independent, Metadata::WithFluxes,
                  Metadata::FillGhost, Metadata::Vector});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                            parthenon::refinement_ops::RestrictAverage,
                            parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
    hydro->AddField<ccbulk::bfield>(m);

    // Cell-centered magnetic field (average of the face fields)
    m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                  Metadata::Derived, Metadata::OneCopy},
                 vel_arr_size);
    hydro->AddField<ccbulk::bcc>(m);

    // Divergence of B (diagnostic)
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    hydro->AddField<ccbulk::divb>(m);

    // Cell-aligned face scratch written by the Riemann solve: normal B, upwind mass
    // flux proxy, and the two transverse EMFs (slot0 = E_tbc, slot1 = E_tac)
    m = Metadata(
        {Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::CellMemAligned});
    hydro->AddField<ccbulk::bface_aligned>(m);
    hydro->AddField<ccbulk::face_vrho>(m);
    m = Metadata(
        {Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::CellMemAligned},
        std::vector<int>{2});
    hydro->AddField<ccbulk::face_emf>(m);
  }

  // Hydro Timestep
  hydro->EstimateTimestepMesh = EstimateTimestepMesh;
  if (amr_int) {
    hydro->CheckRefinementBlock = CheckRefinement;
  }

  // Mass/Volume Fraction Thresholds
  params.Add(
      "vol_frac_thresh",
      pin->GetOrAddReal("hydro", "vol_frac_thresh", 1.0e-12,
                        "Below this volume fraction, materials are deleted in a cell."));
  params.Add("mass_frac_thresh",
             pin->GetOrAddReal("hydro", "mass_frac_thresh", 1.0e-12,
                               "Below this mass fraction, materials are excluded from "
                               "PTE in a cell and deleted."));

  // Temperature floor used in general PTE
  params.Add("temp_floor", pin->GetOrAddReal("hydro", "temp_floor", 1.0,
                                             "The minimum temperature alllowed."));

  // History for total kinetic energy
  bool history_tot_ke =
      pin->GetOrAddBoolean("hydro", "track_total_kinetic_energy", false,
                           "Output to the total kinetic energy in the history file. "
                           "Useful for turbulence studies.");
  params.Add("track_total_kinetic_energy", history_tot_ke);
  parthenon::HstVar_list scalar_histories{};
  if (history_tot_ke) {
    scalar_histories.emplace_back(parthenon::UserHistoryOperation::sum,
                                  total_kinetic_energy, "KE");
  }
  if (do_mhd) {
    scalar_histories.emplace_back(parthenon::UserHistoryOperation::sum,
                                  total_magnetic_energy, "ME");
    scalar_histories.emplace_back(parthenon::UserHistoryOperation::max, max_divb,
                                  "max_divB_dx");
  }
  if (!scalar_histories.empty()) {
    hydro->AddParam<>(parthenon::hist_param_key, scalar_histories, true);
  }

  // Geometric source term dU/dt variables (only in curvilinear geometry)
  if (!parthenon::IsCoord<parthenon::UniformCartesian>()) {
    hydro->RegisterMeshDataSubset(
        "dudt", RiotUtils::MakePackageDudtRequirements({ccbulk::momentum::name()}));
  }

  return hydro;
}

//----------------------------------------------------------------------------------------
//! \fn  AmrTag Hydro::CheckRefinement
//! \brief Material Interface Refinement Strategy
AmrTag CheckRefinement(MeshBlockData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto pmb = md->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
  const int current_level = pmb->loc.level() - pm->GetRootLevel();
  auto mat = pmb->packages.Get("materials");
  auto max_lev_bnd = mat->Param<parthenon::ParArray1D<int>>("max_lev_bnd");
  auto max_lev_mat = mat->Param<parthenon::ParArray1D<int>>("max_lev_mat");

  auto v = riot::MakePack<ccmat::volume_fraction>(md);
  if (v.GetNBlocks() == 0) return AmrTag::same;

  // static auto desc_bulk = MakePackDescriptor<ccbulk::rho,
  //                                            ccbulk::pressure>(resolved_pgks.get());
  // auto vbulk = desc_bulk.GetPack(md);

  const Real vfrac_refine_thresh = 0.01;
  const Real vfrac_derefine_thresh = 0.001;
  // const Real d2press_ref_thresh = 0.5;
  // const Real d2press_deref_thresh = 0.1;
  // const Real d2rho_ref_thresh = 0.5;
  // const Real d2rho_deref_thresh = 0.1;
  // const Real norm_eps = 0.01;
  // const Real vfrac_mat_thresh = 0.5;

  const int b = 0;
  const int Nm = v.GetMaxNumberOfVars();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);
  int kl = 0, ku = 0, jl = 0, ju = 0, il = 0, iu = 0;
  auto bf = parthenon::BoundaryFlag::block;
  int ndim = 1;
  if (ibe.s != ibe.e) {
    il = (pmb->boundary_flag[0] == bf ? ibe.s + 1 : ib.s + 1);
    iu = (pmb->boundary_flag[1] == bf ? ibe.e - 1 : ib.e - 1);
  }
  if (jbe.s != jbe.e) {
    jl = (pmb->boundary_flag[2] == bf ? jbe.s + 1 : jb.s + 1);
    ju = (pmb->boundary_flag[3] == bf ? jbe.e - 1 : jb.e - 1);
    ndim = 2;
  }
  if (kbe.s != kbe.e) {
    kl = (pmb->boundary_flag[4] == bf ? kbe.s + 1 : kb.s + 1);
    ku = (pmb->boundary_flag[5] == bf ? kbe.e - 1 : kb.e - 1);
    ndim = 3;
  }

  // Reduction over blocks (nblocks == 1 here, since md is a single MeshBlockData). The
  // material loop sits between outer_reduce and inner_reduce -- one inner_reduce per
  // material, all joining the same accumulator. The logical bounds are the boundary-aware
  // interior computed above; the memory extent is fixed by Parthenon (CC layout). Each
  // leaf casts a refine/derefine/same vote (+1/-1/0); the block's tag is the max.
  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<int>, LoopConstraint::SingleBlock>;
  auto idx_space =
      rt::GetIndexSpace(v.GetNBlocks(), {kl, ku}, {jl, ju}, {il, iu}, md, TE::CC);
  // Memory offsets to the +/- face neighbor in each direction (zero for collapsed dims),
  // used to reach the i+/-1, j+/-1, k+/-1 volume fractions via flat pack-view indexing.
  const auto di = idx_space.GetDelta(X1DIR);
  const auto dj = idx_space.GetDelta(X2DIR);
  const auto dk = idx_space.GetDelta(X3DIR);
  const int refine_flag = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        for (int m = 0; m < Nm; m++) {
          auto pv = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          const int max_bnd = max_lev_bnd(v(b, m).sparse_id);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, int &lmax) {
            // check for interfaces
            int l;
            if (current_level > max_bnd) {
              l = -1;
            } else {
              const int refine_flag = current_level < max_bnd;
              const int derefine_flag = 1 - (current_level > max_bnd);
              // Centered volume-fraction jump across a cell -> material interface vote.
              auto vote = [&](const Real qm, const Real qp) {
                Real d = 0.5 * std::abs(qp - qm);
                d *= derefine_flag;
                return (d > vfrac_refine_thresh) * refine_flag -
                       (d < vfrac_derefine_thresh);
              };
              l = vote(pv(ccmat::volume_fraction(), idx - di),
                       pv(ccmat::volume_fraction(), idx + di));
              if (ndim > 1)
                l = std::max(l, vote(pv(ccmat::volume_fraction(), idx - dj),
                                     pv(ccmat::volume_fraction(), idx + dj)));
              if (ndim > 2)
                l = std::max(l, vote(pv(ccmat::volume_fraction(), idx - dk),
                                     pv(ccmat::volume_fraction(), idx + dk)));
            }
            lmax = std::max(lmax, l);

            // // now check for pressure and density based triggers: a second-derivative
            // // (smoothness) detector on the bulk pack vbulk, ported to the flat
            // style.
            // // Each (m, k, j, i) leaf votes per field, gated by this cell's vol frac.
            // {
            //   const int refine_flag = current_level < max_mat;
            //   const int derefine_flag = 1 - (current_level > max_mat);
            //   auto d2vote = [&](const Real fm, const Real f0, const Real fp,
            //                     const Real ref_thresh, const Real deref_thresh) {
            //     Real d = std::abs(fm - 2.0 * f0 + fp) /
            //              (std::abs(fm - f0) + std::abs(f0 - fp) +
            //               norm_eps * (std::abs(fm) + 2.0 * std::abs(f0) +
            //               std::abs(fp)));
            //     d *= derefine_flag;
            //     return (d > ref_thresh) * refine_flag - (d < deref_thresh);
            //   };
            //   for (int id = 0; id < 2; id++) {
            //     const Real rt = (id == 0 ? d2rho_ref_thresh : d2press_ref_thresh);
            //     const Real drt = (id == 0 ? d2rho_deref_thresh :
            //     d2press_deref_thresh); int l2 = d2vote(vbulk(b, id, k, j, i - 1),
            //     vbulk(b, id, k, j, i),
            //                     vbulk(b, id, k, j, i + 1), rt, drt);
            //     if (ndim > 1)
            //       l2 = std::max(l2, d2vote(vbulk(b, id, k, j - 1, i), vbulk(b, id, k,
            //       j, i),
            //                                vbulk(b, id, k, j + 1, i), rt, drt));
            //     if (ndim > 2)
            //       l2 = std::max(l2, d2vote(vbulk(b, id, k - 1, j, i), vbulk(b, id, k,
            //       j, i),
            //                                vbulk(b, id, k + 1, j, i), rt, drt));
            //     if (v(b, m, k, j, i) > vfrac_mat_thresh) lmax = std::max(lmax, l2);
            //   }
            // }
          });
        }
      });

  if (refine_flag == 1) return AmrTag::refine;
  if (refine_flag == -1) return AmrTag::derefine;
  return AmrTag::same;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::GuessCellVolumeFractions
//! \brief Construct Volume Fraction Guesses for PTE Solver
TaskStatus GuessCellVolumeFractions(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &riot = pm->packages.Get("riot");
  const int ndim = pm->ndim;

  auto &hydro = pm->packages.Get("hydro");
  auto &materials = pm->packages.Get("materials");
  const auto &eos = materials->Param<parthenon::ParArray1D<RiotEOS::EOS>>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");
  const bool do_ionization = riot->Param<bool>("do_ionization");

  auto v = riot::MakePack<ccmat::rho, ccmat::volume_fraction, cm::rho, ccbulk::pressure,
                          ccbulk::electron_pressure, ccbulk::temperature, cm::lr_cache,
                          cm::lT_cache, cm::ionization_zbar>(md);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  using TE = parthenon::TopologicalElement;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, TE::CC);
  idx_space.template AddPerPointScratch<Real>(1);

  // Neighbor-cell offsets (zero in collapsed dimensions) for the flat-indexed max-density
  // stencil.
  auto di = idx_space.GetDelta(X1DIR);
  auto dj = idx_space.GetDelta(X2DIR);
  auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        auto rho_max = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto pv = RiotLoop::make_pack_view(idx_range, v);

        for (int m = 0; m < nmat; m++) {
          const int mat_id = v(b, ccmat::rho(m)).sparse_id;
          const int phase_id = v(b, ccmat::rho(m)).v;
          auto &eosm = eos(eos_from_matid(mat_id) + phase_id);
          auto pvm = RiotLoop::make_sparse_pack_view(idx_range, v, m);

          // Seed rho_max with this material's density; where a cell is present but has no
          // density, borrow the larger of the two X1 neighbors.
          RiotLoop::inner(idx_range, [&](const auto kji) {
            rho_max(kji) = pvm(cm::rho(), kji);
            if (pvm(ccmat::rho(), kji) > 0.0 && rho_max(kji) <= 0.0) {
              rho_max(kji) = std::max(pvm(cm::rho(), kji - di), pvm(cm::rho(), kji + di));
            }
          });
          idx_range.TeamBarrier();

          if (ndim > 1) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              if (pvm(ccmat::rho(), kji) > 0.0 && rho_max(kji) <= 0.0) {
                rho_max(kji) =
                    std::max(pvm(cm::rho(), kji - dj), pvm(cm::rho(), kji + dj));
              }
            });
            idx_range.TeamBarrier();
          }
          if (ndim > 2) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              if (pvm(ccmat::rho(), kji) > 0.0 && rho_max(kji) <= 0.0) {
                rho_max(kji) =
                    std::max(pvm(cm::rho(), kji - dk), pvm(cm::rho(), kji + dk));
              }
            });
            idx_range.TeamBarrier();
          }

          RiotLoop::inner(idx_range, [&](const auto kji) {
            if (pvm(ccmat::rho(), kji) > 0.0) {
              if (rho_max(kji) > 0.0) {
                pvm(ccmat::volume_fraction(), kji) =
                    pvm(ccmat::rho(), kji) / (rho_max(kji) + 1.e-30);
              } else {
                Real P = pv(ccbulk::pressure(), kji);
                // NOTE(JMM): ccbulk_pressure is TOTAL pressure, but only ions
                // participate in PTE.
                if (do_ionization) {
                  P -= pv(ccbulk::electron_pressure(), kji);
                }
                RiotEOS::LambdaIndexerSingle lambda(pvm, kji);
                pvm(cm::rho(), kji) = RiotEOS::rho_from_P_T(
                    eosm, P, pv(ccbulk::temperature(), kji), 20.0, lambda);
                pvm(cm::rho(), kji) = (pvm(cm::rho(), kji) > 0.0) * pvm(cm::rho(), kji) +
                                      (pvm(cm::rho(), kji) < 0.0) * 20.0;
                pvm(ccmat::volume_fraction(), kji) =
                    pvm(ccmat::rho(), kji) / pvm(cm::rho(), kji);
              }
            } else {
              pvm(ccmat::volume_fraction(), kji) = 0.0;
            }
          });
          // rho_max scratch is reused by the next material; fence the read above before
          // it is overwritten.
          idx_range.TeamBarrier();
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::CalculateGeometricSource
//! \brief Calculates "Geometric Source Terms" for 1D Spherical and 2D RZ Geometries.
//!        In cylindrical (r, z, phi) coordinates, the divergence of the momentum flux
//!        tensor T = rho v v + (p + B^2/2) I - B B - sigma has the geometric parts
//!          S_r = T_phiphi / r,  S_phi = -T_rphi / r,
//!        i.e. pressure, deviatoric/viscous hoop stress, centrifugal and magnetic hoop
//!        stress in S_r, and Coriolis and magnetic tension in S_phi.
TaskStatus CalculateGeometricSource(MeshData<Real> *state, MeshData<Real> *src) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  // just return immediately for Cartesian
  if (parthenon::IsCoord<parthenon::UniformCartesian>()) return TaskStatus::complete;

  auto pm = state->GetParentPointer();
  const bool do_strength = pm->packages.Get("riot")->Param<bool>("do_strength");
  auto &materials = pm->packages.Get("materials");
  auto &strength_mats = materials->Param<std::vector<int>>("strength_mats");

  const bool do_ionization = pm->packages.Get("riot")->Param<bool>("do_ionization");
  const bool do_mhd = pm->packages.Get("hydro")->Param<bool>("do_mhd");
  bool do_plasma_viscosity = false;
  if (do_ionization) {
    do_plasma_viscosity = pm->packages.Get("ionization")->Param<bool>("plasma_viscosity");
  }

  auto v = riot::MakePack<ccbulk::pressure, ccbulk::rho, ccbulk::velocity, ccbulk::bcc,
                          ccmat::volume_fraction, cm::deviatoric_stress,
                          ccbulk::ion_shear_viscosity, ccbulk::strain_rate>(
      state, strength_mats);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;
  auto vsrc = riot::MakePack<ccbulk::momentum>(src);

  using lt = RiotUtils::LoopType<>;
  using TE = parthenon::TopologicalElement;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state, TE::CC);
  idx_space.template AddPerPointScratch<Real>(1);
  idx_space.template AddPerPointScratch<Real>(1);
  idx_space.template AddPerPointScratch<Real>(1);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        // stress = T_phiphi (T_thetatheta + T_phiphi in spherical), shear = T_rphi
        auto stress = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto shear = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto geom = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto pvsrc = RiotLoop::make_pack_view(idx_range, vsrc);

        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);
          const Real dr = coords.Dxf<X1DIR>(i);
          const Real r0 = coords.Xf<X1DIR>(i) + 0.5 * dr;
          if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
            const Real inv_rc = 12.0 * r0 / (dr * dr + 12.0 * r0 * r0);
            geom(kji) = 2.0 * inv_rc;
          } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
            geom(kji) = 1.0 / r0;
          } else {
            PARTHENON_FAIL("Unknown coordinate system!");
          }
        });

        const int nstr_mat = v.GetSize(b, ccmat::volume_fraction());
        RiotLoop::inner(idx_range, [&](const auto kji) {
          stress(kji) = pv(ccbulk::pressure(), kji);
          shear(kji) = 0.0;
          if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
            const Real rho = pv(ccbulk::rho(), kji);
            const Real vr = pv(ccbulk::velocity(0), kji);
            const Real vphi = pv(ccbulk::velocity(2), kji);
            stress(kji) += rho * vphi * vphi;
            shear(kji) = rho * vr * vphi;
          }
        });

        if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
          if (do_mhd) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real br = pv(ccbulk::bcc(0), kji);
              const Real bz = pv(ccbulk::bcc(1), kji);
              const Real bphi = pv(ccbulk::bcc(2), kji);
              stress(kji) += 0.5 * (br * br + bz * bz - bphi * bphi);
              shear(kji) -= br * bphi;
            });
          }
        }

        if (do_strength) {
          for (int n = 0; n < nstr_mat; n++) {
            auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
            idx_range.TeamBarrier();
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real vfrac = pv_n(ccmat::volume_fraction(), kji);
              const Real sxx = pv_n(cm::deviatoric_stress(0), kji);
              const Real syy = pv_n(cm::deviatoric_stress(3), kji);
              if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
                stress(kji) += 0.5 * vfrac * sxx;
              } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
                stress(kji) += vfrac * (sxx + syy);
              }
            });
          }
        }

        if (do_plasma_viscosity) {
          auto pv = RiotLoop::make_pack_view(idx_range, v);
          idx_range.TeamBarrier();
          RiotLoop::inner(idx_range, [&](const auto kji) {
            if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
              const Real Pi_tt = 2.0 * pv(ccbulk::strain_rate(3), kji) *
                                 pv(ccbulk::ion_shear_viscosity(), kji);
              stress(kji) -= Pi_tt;
            } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
              const Real Pi_pp = 2.0 * pv(ccbulk::strain_rate(5), kji) *
                                 pv(ccbulk::ion_shear_viscosity(), kji);
              stress(kji) -= Pi_pp;
            }
          });
        }

        idx_range.TeamBarrier();
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pvsrc(ccbulk::momentum(0), kji) = stress(kji) * geom(kji);
          pvsrc(ccbulk::momentum(1), kji) = 0.0;
          pvsrc(ccbulk::momentum(2), kji) = -shear(kji) * geom(kji);
        });
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateMaxSignalSpeed
//! \brief Calculates the maximum signal speed given face signal speeds
TaskStatus CalculateMaxSignalSpeed(MeshData<Real> *state) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  using parthenon::MakePackDescriptor;

  if (state->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = state->GetParentPointer();
  const int ndim = pm->ndim;
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);

  // Pack up bulk signal speed
  auto vstate = riot::MakePack<ccbulk::face_signal, ccbulk::max_signal>(state);
  const int nblocks = vstate.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state,
                                     parthenon::TopologicalElement::CC);
  auto di = idx_space.GetDelta(X1DIR);
  auto dj = idx_space.GetDelta(X2DIR);
  auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, vstate);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv(ccbulk::max_signal(0), kji) =
              std::max(pv(TE::F1, ccbulk::face_signal(), kji),
                       pv(TE::F1, ccbulk::face_signal(), kji + di));
          if (multi_d) {
            pv(ccbulk::max_signal(1), kji) =
                std::max(pv(TE::F2, ccbulk::face_signal(), kji),
                         pv(TE::F2, ccbulk::face_signal(), kji + dj));
          }
          if (three_d) {
            pv(ccbulk::max_signal(2), kji) =
                std::max(pv(TE::F3, ccbulk::face_signal(), kji),
                         pv(TE::F3, ccbulk::face_signal(), kji + dk));
          }
        });
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::EstimateTimestepMesh
//! \brief Calculates the numerical timestep associated with hydro
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using PortsOfCall::Robust::ratio;
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("hydro");
  auto &params = pkg->AllParams();

  // If the fluid is fixed, hydro imposes no timestep constraints. We set the
  // hydro timestep vote to a large value, but a slightly smaller one than
  // the default value of parthenon/time/dt_max
  const bool fixed_fluid = pm->packages.Get("riot")->Param<bool>("fixed_fluid");
  if (fixed_fluid) return 1.e-3 * std::numeric_limits<Real>::max();

  // strength params
  const bool do_strength = pm->packages.Get("riot")->Param<bool>("do_strength");
  const bool do_mhd = params.Get<bool>("do_mhd");
  auto v = riot::MakePack<ccbulk::bulk_modulus, ccbulk::shear_modulus, ccbulk::rho,
                          ccbulk::velocity, ccbulk::max_signal, ccbulk::bcc>(md);
  const int ndim = pm->ndim;

  using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC);
  const Real min_dt = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &ldt) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          const Real rho = pv(ccbulk::rho(), idx);
          const Real bulk_modulus = pv(ccbulk::bulk_modulus(), idx);
          const Real shear_modulus =
              (do_strength) ? pv(ccbulk::shear_modulus(), idx) : 0.0;
          const Real a2 = ratio((bulk_modulus + (4.0 / 3.0) * shear_modulus), rho);
          const Real cs = std::sqrt(a2);
          Real b2 = 0.0;
          if (do_mhd) {
            b2 = SQR(pv(ccbulk::bcc(0), idx)) + SQR(pv(ccbulk::bcc(1), idx)) +
                 SQR(pv(ccbulk::bcc(2), idx));
          }
          Real denom = 0.0;
          for (int d = 0; d < ndim; d++) {
            const Real vphys =
                pv(ccbulk::velocity(d), idx) * coords.Scale<TE::CC>(d + 1, k, j, i);
            Real csig = std::abs(vphys) + cs;
            if (do_mhd) {
              // Fast magnetosonic speed along d (Heaviside-Lorentz units)
              const Real va2 = ratio(b2, rho);
              const Real vad2 = ratio(SQR(pv(ccbulk::bcc(d), idx)), rho);
              const Real s = a2 + va2;
              const Real cf2 =
                  0.5 * (s + std::sqrt(std::max(SQR(s) - 4.0 * a2 * vad2, 0.0)));
              csig = std::abs(vphys) + std::sqrt(cf2);
            }
            const Real max_sig = std::max(csig, pv(ccbulk::max_signal(d), idx));
            denom += max_sig / coords.CellWidth(d + 1, k, j, i);
          }
          ldt = std::min(ldt, 1.0 / denom);
        });
      });

  const auto &cfl = params.Get<Real>("cfl");
  return cfl * min_dt;
}

//----------------------------------------------------------------------------------------
//! \fn  PrimFluxPack Hydro::MakeAdvectionPack
//! \brief Generates meshblock pack containing advected variables
PrimFluxPack MakeAdvectionPack(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using parthenon::PDOpt;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  parthenon::Metadata::FlagCollection all_advected_flags{Metadata::Advected};

  // For advection, the conserved vars (containing the fluxes) are associated with the
  // primitives. e.g., momentum->getAssociated() == velocity so flxs are the flagged vars
  // and vars are the assoc_vars

  // Static here implies the assumption that the list of vars doesn't change during the
  // life of the simulation
  // TODO(JMM): can't do static vars with structured binding.
  auto [flxs, vars] = RiotUtils::GetAssociatedVars(md, all_advected_flags);
  static std::vector<bool> use_regex(flxs.size(), false);

  static auto desc_vars = MakePackDescriptor(resolved_pkgs.get(), vars, use_regex);
  static auto desc_flxs =
      MakePackDescriptor(resolved_pkgs.get(), flxs, use_regex, {}, {PDOpt::WithFluxes});
  auto pack_vars = riot::GetPack(desc_vars, md);
  auto pack_flxs = riot::GetPack(desc_flxs, md);

  PrimFluxPack pack(pack_vars, pack_flxs);
  return pack;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::total_kinetic_energy
//! \brief Computes total kinetic energy for history output
Real total_kinetic_energy(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();

  auto v = riot::MakePack<ccbulk::rho, ccbulk::velocity>(md);
  const int ndim = pm->ndim;

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC);
  const Real ke = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &ke) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          Real cell_ke = 0.0;
          for (int d = 0; d < ndim; d++) {
            cell_ke += 0.5 * pv(ccbulk::rho(), idx) * SQR(pv(ccbulk::velocity(d), idx));
          }
          ke += cell_ke * coords.CellVolume(k, j, i);
        });
      });

  const parthenon::RegionSize &mesh_size = pm->mesh_size;
  const Real lx1 = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));
  const Real lx2 = (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  const Real lx3 = (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
  return ke / (lx1 * lx2 * lx3);
}

} // namespace Hydro
