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

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "mhd/emf.hpp"
#include "mhd/mhd.hpp"
#include "mhd/mhd_helpers.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace MHD {

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> MHD::Initialize
//! \brief Registers the ideal-MHD package.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace fbulk = face_variables::bulk;
  using parthenon::Metadata;
  using namespace parthenon::refinement_ops;

  auto mhd = std::make_shared<StateDescriptor>("mhd");
  Params &params = mhd->AllParams();

  // Published limitations, printed as well as documented (Gate G5). The startup
  // rejections in riot.cpp already refuse everything outside the certified matrix, but a
  // rejection only fires for a configuration the user actually asked for. This banner
  // states the boundary of what has been VALIDATED even when the run is inside it, so a
  // user does not have to infer the scope from the absence of an error. It is not a
  // warning and does not indicate a problem.
  if (parthenon::Globals::my_rank == 0) {
    printf("MHD: ideal MHD with face-centered constrained transport is enabled.\n"
           "MHD: validated for Cartesian, uniform-grid, single-material, ideal-gas,\n"
           "MHD:   single-temperature problems only; see the MHD chapter of the docs.\n"
           "MHD: no resistivity, Hall term, Biermann battery, or anisotropic transport.\n"
           "MHD: not run on GPU. total_material_energy INCLUDES B^2/(2*mu0).\n");
  }

  // Magnetic normalization. See ADR-001: magnetic energy density is B^2/(2 mu0), so
  // mu0 = 4*pi is Gaussian CGS (B in Gauss, energy B^2/8pi), consistent with the CGS
  // units RIOT uses throughout for EOS and opacity. mu0 = 1 reproduces the donor's
  // scale-free normalized field exactly and is what the regression tests use, so that
  // comparison against Artemis needs no unit reconciliation.
  const Real mu0 =
      pin->GetOrAddReal("mhd", "mu0", 4.0 * M_PI,
                        "Magnetic permeability in code units; magnetic energy density "
                        "is B^2/(2*mu0). Default 4*pi is Gaussian CGS (B in Gauss). "
                        "Set to 1 for a scale-free normalized field.");
  PARTHENON_REQUIRE(mu0 > 0.0, "mhd/mu0 must be positive");
  params.Add("mu0", mu0);

  const bool monitor_divb = pin->GetOrAddBoolean(
      "mhd", "monitor_divb", false,
      "Report max and volume-weighted-mean |div B| plus the dimensionless eta once per "
      "step, after the update (diagnostic; costs two mesh-wide reductions per step)");
  params.Add("monitor_divb", monitor_divb);

  // Solver-health counters (MHD::SolverDiag). Allocated unconditionally and zero-filled:
  // Kokkos::View zero-initializes by default, and the counters must exist before the
  // first flux kernel runs. `params.Add` stores the View by value, so the reference count
  // keeps the device allocation alive for the run.
  SolverDiagView solver_diag("mhd solver diagnostics", kNumSolverDiag);
  params.Add("solver_diag", solver_diag);
  // Host-side mirror of the last reported values, so PostStepDiagnostics can report only
  // when a counter INCREASES rather than printing every step. Mutable because it is state
  // that the diagnostic updates as the run proceeds.
  params.Add("solver_diag_reported", std::vector<std::int64_t>(kNumSolverDiag, 0), true);

  // Escape hatch for MHD::RestoreDerivedOnRestart's zero-field check. Registered here
  // rather than on the restart path so it appears in the parameter table of every MHD
  // run.
  params.Add("allow_zero_field_restart",
             pin->GetOrAddBoolean(
                 "mhd", "allow_zero_field_restart", false,
                 "Permit restarting an MHD run from a checkpoint whose magnetic field is "
                 "identically zero everywhere. Off by default, because that is exactly "
                 "what restarting from a HYDRO checkpoint by mistake looks like."));

  // ---------------------------------------------------------------------------------
  // The evolved magnetic field: the component normal to each face.
  //
  // Metadata rationale (ADR-003), each flag load-bearing:
  //   Face        - genuine face topology, so storage carries the extra element per
  //                 normal direction. NOTE the deliberate ABSENCE of CellMemAligned,
  //                 which every other face field in RIOT carries; that flag forces a
  //                 cell-aligned layout and would break comms and refinement here.
  //   Independent - this is evolved state, so it joins the u0/u1 stage registers and
  //                 is written to restart files.
  //   Conserved   - the transported quantity is the magnetic flux through the face.
  //   WithFluxes  - makes Parthenon allocate an EDGE-centered flux register
  //                 automatically (interface/metadata.cpp:189-197). That edge register
  //                 IS the EMF; there is no separate EMF field, and this is also what
  //                 makes the standard flux-correction tasks average EMFs correctly at
  //                 coarse/fine boundaries.
  //   FillGhost   - face ghosts ride the driver's existing boundary exchange, so no
  //                 new communication task is needed. Shared faces between blocks are
  //                 single-valued via Parthenon's block-ownership masking, which is
  //                 what makes div B identical on both sides of a block boundary.
  //
  // Refinement operators are registered because they are the correct ones and cost
  // nothing when unused: divergence-preserving Toth & Roe (2002) internal-face
  // prolongation, with area-weighted face restriction. AMR is nonetheless REJECTED at
  // startup: registering an operator is not evidence that refinement works, and the
  // donor's own MHD tests never exercise AMR, so there is no oracle for it (ADR-004).
  Metadata m = Metadata({Metadata::Face, Metadata::Independent, Metadata::Conserved,
                         Metadata::WithFluxes, Metadata::FillGhost});
  m.RegisterRefinementOps<ProlongateSharedMinMod, RestrictAverage,
                          ProlongateInternalTothAndRoe>();
  mhd->AddField<fbulk::magnetic_field>(m);

  // ---------------------------------------------------------------------------------
  // Derived cell-centered companions.
  //
  // magnetic_field and magnetic_energy are Derived but carry WithFluxes so that their
  // flux registers can serve as stage scratch: the transverse induction fluxes from
  // the Riemann solve, and the face magnetic pressure respectively. Being Derived is
  // what keeps sparse_update::UpdateToNextStage (which requires Independent) from
  // applying a spurious flux divergence to them.
  //
  // FillGhost on the cell-centered field: the reconstruction stencil reads it in the
  // ghost zones, and it is cheaper to communicate it than to re-derive it there.
  std::vector<int> bfield_arr_size(1, 3);
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::WithFluxes, Metadata::Vector},
               bfield_arr_size);
  mhd->AddField<ccbulk::magnetic_field>(m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::WithFluxes});
  mhd->AddField<ccbulk::magnetic_energy>(m);

  // Pure diagnostic. Not FillGhost: it is never read by a kernel, only reduced and
  // written to output.
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  mhd->AddField<ccbulk::div_magnetic_field>(m);

  // A problem generator supplies face B only; derive the rest before the first
  // total-energy assembly.
  mhd->PostInitializationMesh = PostInitialization;

  // PostInitializationMesh does not run on a restart (see RestoreDerivedOnRestart), so
  // the same derived state has to be rebuilt from the checkpointed face field here.
  mhd->UserWorkBeforeLoopMesh = RestoreDerivedOnRestart;

  // Installed unconditionally. The solver-health counters must be watched in every MHD
  // run -- a silent HLLD degradation is not something a user opts into noticing -- and
  // reading two int64s per step is free. The expensive half (the div B reductions) is
  // gated on `monitor_divb` inside.
  mhd->PostStepDiagnosticsMesh = PostStepDiagnostics;

  return mhd;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SetDerivedMagneticFields
//! \brief Derives cell-centered B, magnetic energy, and discrete div B from face state.
//!
//! This is the ONLY writer of the three derived magnetic fields. Centralizing it is
//! what guarantees that the energy subtraction in FillInteriorDerived and the energy
//! re-synthesis in PostCommsFillDerived cannot disagree about the face-to-cell
//! convention (ADR-002).
//!
//! The caller owns the ordering contract: face state must be valid on `domain` before
//! this runs. On IndexDomain::interior that holds after the CT face update; on
//! IndexDomain::entire it holds only after the boundary exchange.
void SetDerivedMagneticFields(MeshData<Real> *md, IndexDomain domain) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace fbulk = face_variables::bulk;
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetParentPointer();
  const Real mu0 = pm->packages.Get("mhd")->template Param<Real>("mu0");
  const int ndim = pm->ndim;
  const int multi_d = (ndim > 1);
  const int three_d = (ndim > 2);

  auto v = riot::MakePack<fbulk::magnetic_field, ccbulk::magnetic_field,
                          ccbulk::magnetic_energy, ccbulk::div_magnetic_field>(md);
  if (v.GetNBlocks() == 0) return;

  // The cell-centered writes go through the flat pack view; the face reads go through
  // the pack with logical coordinates (see MHD::FaceB for why face storage must never
  // be addressed with the cell-shaped memory indexer). Because no *view* here mixes the
  // two layouts, the loop keeps the default (fast) inner contract.
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(domain, 0, v.GetNBlocks(), md, TE::CC);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);

        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);

          // multi_d / three_d are zero in a collapsed dimension, which makes the two
          // face reads coincide: the face-to-cell average degenerates to the single
          // available value, and the corresponding divergence term cancels identically.
          // That is the correct behavior -- a transverse field component still evolves
          // in a reduced-dimension run.
          Real bx, by, bz;
          FaceToCellB(v, b, k, j, i, multi_d, three_d, bx, by, bz);
          pv(ccbulk::magnetic_field(0), kji) = bx;
          pv(ccbulk::magnetic_field(1), kji) = by;
          pv(ccbulk::magnetic_field(2), kji) = bz;
          pv(ccbulk::magnetic_energy(), kji) = MagneticEnergyDensity(bx, by, bz, mu0);

          // Discrete divergence as the signed sum of magnetic FLUXES through the cell
          // faces, divided by cell volume. Computing it this way -- rather than from
          // centered differences of the cell-centered field -- is essential: it is the
          // quantity that constrained transport holds at roundoff, and it is the only
          // form in which the shared-edge EMF contributions cancel algebraically.
          Real divb =
              coords.template FaceArea<X1DIR>(k, j, i + 1) *
                  FaceB(v, b, TE::F1, k, j, i + 1) -
              coords.template FaceArea<X1DIR>(k, j, i) * FaceB(v, b, TE::F1, k, j, i);
          if (multi_d) {
            divb +=
                coords.template FaceArea<X2DIR>(k, j + 1, i) *
                    FaceB(v, b, TE::F2, k, j + 1, i) -
                coords.template FaceArea<X2DIR>(k, j, i) * FaceB(v, b, TE::F2, k, j, i);
          }
          if (three_d) {
            divb +=
                coords.template FaceArea<X3DIR>(k + 1, j, i) *
                    FaceB(v, b, TE::F3, k + 1, j, i) -
                coords.template FaceArea<X3DIR>(k, j, i) * FaceB(v, b, TE::F3, k, j, i);
          }
          pv(ccbulk::div_magnetic_field(), kji) = divb / coords.CellVolume(k, j, i);
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus MHD::SetDerived
//! \brief Task-shaped wrapper for SetDerivedMagneticFields.
TaskStatus SetDerived(MeshData<Real> *md, IndexDomain domain) {
  SetDerivedMagneticFields(md, domain);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::PostInitialization
//! \brief Derives magnetic state after a problem generator has written face B.
//!
//! A problem generator is responsible for face B ONLY -- ideally as a discrete curl of
//! a vector potential, so that div B vanishes to machine precision at t = 0. Everything
//! else follows from here, which keeps every MHD problem generator from having to
//! reimplement the face-to-cell and energy conventions.
void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  // Derived state is needed on `entire` because the first total-energy assembly and the
  // first reconstruction both read ghost zones.
  SetDerivedMagneticFields(md, IndexDomain::entire);
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RestoreDerivedOnRestart
//! \brief Rebuilds the derived magnetic fields when a run resumes from a checkpoint.
//!
//! WHY THIS IS NEEDED, since it looks redundant next to PostInitialization. Parthenon
//! calls `Mesh::Initialize(!is_restart, ...)`, and EVERYTHING in the `if (init_problem)`
//! block -- the problem generator, every PostInitialization hook, and the
//! PreCommFillDerived/communicate/FillDerived cycle that follows them
//! (`mesh.cpp:872-953`) -- is skipped on a restart. So `MHD::PostInitialization` never
//! runs, and the three derived magnetic fields stay at their default zero.
//!
//! That is not merely a cosmetic problem for the first output. `ccbulk::magnetic_field`
//! is reconstructed to get the TRANSVERSE field in the Riemann solve, and the per-stage
//! `MHD::SetDerived` task runs AFTER the update, not before the fluxes. So without this
//! the first post-restart stage computes its fluxes from a zero transverse field: the run
//! continues, stays divergence-free, and is permanently wrong. Measured as a 3.7e-06
//! deviation from the uninterrupted trajectory (TEST_LEDGER G5.6).
//!
//! `UserWorkBeforeLoopMesh` is the right hook because it is one of the few that runs
//! regardless of restart, and it runs before the first output and the first step.
//!
//! Restart-only on purpose. On a fresh start, PostInitialization has already derived this
//! state and `Mesh::Initialize` has since communicated boundaries, so the cell-centered
//! ghosts hold values received from neighbouring interiors. Recomputing them here from
//! face ghosts could only replace communicated values with locally derived ones and would
//! risk perturbing the verified G2-G5 results for no benefit.
//!
//! `entire`, not `interior`, and that is measured rather than cautious. Nothing
//! communicates the cell-centered ghosts between here and the first stage's
//! reconstruction, so they have to be derived locally. Swapping this one argument to
//! `interior` makes the restarted trajectory diverge from the uninterrupted one
//! by 1.4e-10 in time by cycle 40, where `entire` keeps it bitwise (TEST_LEDGER G5.6).
void RestoreDerivedOnRestart(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  if (!parthenon::Globals::is_restart) return;
  auto &base = pm->mesh_data.Add("base", pm->GetBasePartition());
  SetDerivedMagneticFields(base.get(), IndexDomain::entire);

  // ------------------------------------------------------------------------------------
  // Second half of the "did this checkpoint actually come from an MHD run" test, and the
  // half that cannot be defeated by a command-line override.
  //
  // riot.cpp rejects an MHD/hydro checkpoint mismatch by looking for `mhd/mu0` in the
  // input deck that Parthenon embeds in every checkpoint. That works, but a user who
  // passes `mhd/mu0=...` on the command line creates the parameter themselves and masks
  // it -- and passing mu0 is a natural thing to do when turning MHD on. So the deck test
  // alone leaves a hole exactly where a user is most likely to be experimenting.
  //
  // This closes it from the DATA instead: a checkpoint written without MHD has no face
  // field, so the read zero-fills it and the magnetic energy is identically zero on every
  // block. Reducing over `ccbulk::magnetic_energy` rather than the face field is
  // deliberate -- it was just computed above, it is a sum of squares (so it vanishes if
  // and only if every face component does), and it lives in the cell-shaped pack, which
  // keeps this away from the face-addressing hazard documented on MHD::FaceB.
  //
  // The reduction must be GLOBAL. A localized field such as the field-loop test leaves
  // whole blocks -- hence whole ranks -- with zero magnetic energy, so a rank-local test
  // would abort a perfectly good restart.
  //
  // An escape hatch exists because a genuinely unmagnetized MHD run is legitimate: it is
  // how the MHD-reduces-to-hydro check (H02) is posed. That case is rare enough to be
  // worth an explicit opt-in and common enough as a MISTAKE to be worth stopping.
  // Read from the package, not from `pin`. GetOrAdd is still legal after FinalizeParsing,
  // but a parameter added only on the restart path would be missing from the table that
  // `parthenon/job/output_params_and_exit` prints for a fresh run -- so a user hitting
  // the rejection below could not discover the escape hatch that the message names.
  if (pm->packages.Get("mhd")->template Param<bool>("allow_zero_field_restart")) return;

  auto v = riot::MakePack<ccbulk::magnetic_energy>(base.get());
  Real emag_max = 0.0;
  if (v.GetNBlocks() > 0) {
    using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
    auto idx_space = rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(),
                                       base.get(), parthenon::TopologicalElement::CC);
    emag_max = RiotLoop::outer_reduce(
        idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
          auto pv = RiotLoop::make_pack_view(idx_range, v);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &m) {
            m = std::max(m, std::abs(pv(ccbulk::magnetic_energy(), idx)));
          });
        });
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &emag_max, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  PARTHENON_REQUIRE(
      emag_max > 0.0,
      "Restarted an MHD run from a checkpoint whose magnetic field is identically zero "
      "everywhere. This is what restarting from a HYDRO checkpoint looks like: the face "
      "field is absent from the file and is silently zero-filled, so the run would "
      "continue as unmagnetized hydrodynamics without saying so. Restart from an MHD "
      "checkpoint, or set <mhd>/allow_zero_field_restart = true if a zero field really "
      "is "
      "intended.");
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::PostStepDiagnostics
//! \brief Per-step MHD health report: solver counters always, div B on request.
void PostStepDiagnostics(parthenon::SimTime const &simtime, MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto pkg = pm->packages.Get("mhd");

  // ------------------------------------------------------------------------------------
  // Solver-health counters. Unconditional; see MHD::SolverDiag for why.
  //
  // Reported only when a counter has GONE UP since the last report. Printing every step
  // would bury the signal, and printing only at the end of the run would lose which step
  // it happened on -- which is the piece of information that makes the number actionable.
  auto diag = pkg->template Param<SolverDiagView>("solver_diag");
  auto h_diag = Kokkos::create_mirror_view(diag);
  Kokkos::deep_copy(h_diag, diag);

  auto *reported =
      pkg->template MutableParam<std::vector<std::int64_t>>("solver_diag_reported");

  // The counters are per-rank, because each rank only solves its own faces. Sum them, or
  // a fallback confined to one rank's blocks would be invisible on every other rank and
  // the report would depend on which rank happened to print.
  std::vector<std::int64_t> totals(kNumSolverDiag);
  for (int n = 0; n < kNumSolverDiag; ++n)
    totals[n] = h_diag(n);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, totals.data(), kNumSolverDiag,
                                    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
#endif

  bool grew = false;
  for (int n = 0; n < kNumSolverDiag; ++n)
    grew |= (totals[n] > (*reported)[n]);
  if (grew) {
    if (parthenon::Globals::my_rank == 0) {
      // Raw counts are not interpretable on their own -- the same count means very
      // different things on a 64^3 mesh and a 512^3 one -- so the fallback count is also
      // reported PER CELL PER STEP. A face is solved once per direction per RK stage, so
      // for rk2 in 3D a value near 6 means essentially every face fell back, i.e. HLLD is
      // running as HLLE. Anything near zero means HLLD is genuinely being used.
      //
      // The denominator is the mesh cell count, which is exact and free, rather than a
      // true face-solve count: counting faces would need an atomic on every face, which
      // would cost far more than the diagnostic is worth.
      const std::int64_t ncells = pm->GetTotalCells();
      const std::int64_t d_fb =
          totals[kDiagHlldFallback] - (*reported)[kDiagHlldFallback];
      const double per_cell =
          (ncells > 0) ? static_cast<double>(d_fb) / static_cast<double>(ncells) : 0.0;
      printf("MHD solver diagnostics at cycle=%d t=%.6e (cumulative): "
             "hlld_fallback=%lld (%.2f per cell this step) density_floor=%lld\n",
             simtime.ncycle + 1, simtime.time + simtime.dt,
             static_cast<long long>(totals[kDiagHlldFallback]), per_cell,
             static_cast<long long>(totals[kDiagDensityFloor]));
      // A density floor is not a diagnostic curiosity. It means the reconstruction
      // produced a non-positive density and the solver silently repaired it, which the
      // verification plan classes as a failed test rather than a success.
      if (totals[kDiagDensityFloor] > (*reported)[kDiagDensityFloor]) {
        printf("MHD WARNING: a non-positive reconstructed density was clamped. This is a "
               "repaired invalid state, not a benign event -- treat the result as "
               "suspect.\n");
      }
    }
    for (int n = 0; n < kNumSolverDiag; ++n)
      (*reported)[n] = totals[n];
  }

  // ------------------------------------------------------------------------------------
  // div B, on request only: two mesh-wide reductions per step.
  if (pkg->template Param<bool>("monitor_divb")) MonitorDivergence(simtime, md);
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MonitorDivergence
//! \brief Per-step div B diagnostic, enabled by `mhd/monitor_divb`.
//!
//! WHAT IS REPORTED, and why it is three numbers rather than one:
//!
//!  - `max|divB|` is the headline number, but on its own it is NOT interpretable. It
//!    carries units and scales like 1/dx, so the same physical quality of solution
//!    produces a larger number on a finer mesh. Reporting only this invites the reader to
//!    conclude a refinement broke constrained transport.
//!  - `mean|divB|` is volume weighted. A handful of bad cells and a uniformly poor
//!    solution give very different ratios of max to mean, which is the difference between
//!    a localized indexing bug and a systematically wrong stencil.
//!  - `eta = |divB| * l_cell / max(|B|_cell, b_ref)` is DIMENSIONLESS and is the number
//!  to
//!    judge. It is the fractional field imbalance per cell, so "at roundoff" means
//!    ~1e-16 independent of mesh, units, and field strength. This is the quantity the
//!    analysis scripts already use, deliberately, so the runtime monitor and the test
//!    harness cannot disagree about what "divergence free" means.
//!
//! Two conventions worth stating because they are choices, not derivations:
//!
//!  - `l_cell` is the LARGEST active cell width, not the smallest. Both are defensible;
//!    the largest is the conservative one, because it makes eta bigger and so cannot
//!    under-report a problem.
//!  - The denominator is `max(|B|_cell, b_ref)` with `b_ref = kEtaFieldFloor * max|B|`,
//!    i.e. LOCAL where the field is strong and floored where it is not. Normalizing on
//!    the local field is the sensitive choice: a divergence error confined to a
//!    weak-field region is invisible under a global normalization, which is exactly where
//!    such an error is most likely to hide. The floor exists only so a field-free region
//!    divides roundoff by a scale instead of by zero. Computing `max|B|` is the second
//!    reduction, and the reason this whole diagnostic is opt-in.
//!
//! Consequence worth knowing when comparing against the analysis scripts:
//! `analyze_field_loop.py` normalizes on the GLOBAL max |B|, so its eta is a lower bound
//! on the one printed here. They agree when the field is roughly uniform and diverge when
//! it is not. Neither is wrong; this one is the stricter statement.
//!
//! Reads the ALREADY COMPUTED `ccbulk::div_magnetic_field` rather than re-deriving the
//! face stencil. That field comes from the face flux balance in SetDerivedMagneticFields
//! and is refreshed every stage, so reusing it guarantees the monitor reports the same
//! divergence the tests measure. Re-implementing the stencil here would create a second
//! definition that could drift from the first.
void MonitorDivergence(parthenon::SimTime const &simtime, MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  auto pm = md->GetParentPointer();
  auto v = riot::MakePack<ccbulk::div_magnetic_field, ccbulk::magnetic_field>(md);
  if (v.GetNBlocks() == 0) return;

  using rmax = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  using rsum = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  const auto te = parthenon::TopologicalElement::CC;
  const int nb = v.GetNBlocks();

  // Pass 1: the field scale that makes eta dimensionless.
  auto max_space = rmax::GetIndexSpace(IndexDomain::interior, 0, nb, md, te);
  Real bmax = RiotLoop::outer_reduce(
      max_space, KOKKOS_LAMBDA(const rmax::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &m) {
          const Real bsq = SQR(pv(ccbulk::magnetic_field(0), idx)) +
                           SQR(pv(ccbulk::magnetic_field(1), idx)) +
                           SQR(pv(ccbulk::magnetic_field(2), idx));
          m = std::max(m, std::sqrt(bsq));
        });
      });
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &bmax, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif
  // A completely unmagnetized run has nothing to report and no scale to report it
  // against; bail rather than print zeros over a divide-by-zero.
  if (!(bmax > 0.0)) return;

  const int ndim = pm->ndim;
  // Weak-field floor for the eta denominator. 1e-6 of the peak field means a genuinely
  // field-free cell cannot manufacture a huge eta from roundoff, while still leaving eta
  // local -- and therefore sensitive -- across the six orders of magnitude of field
  // strength above the floor. The value is a choice, not a derivation; it is named so it
  // is visible rather than buried as a literal.
  constexpr Real kEtaFieldFloor = 1.0e-6;
  const Real b_ref = kEtaFieldFloor * bmax;

  Real divb_max = RiotLoop::outer_reduce(
      max_space, KOKKOS_LAMBDA(const rmax::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &m) {
          m = std::max(m, std::abs(pv(ccbulk::div_magnetic_field(), idx)));
        });
      });

  Real eta_max = RiotLoop::outer_reduce(
      max_space, KOKKOS_LAMBDA(const rmax::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &m) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          // Largest ACTIVE width: a collapsed direction has no gradient in it and must
          // not be allowed to set the length scale.
          Real len = coords.template Dxc<X1DIR>(k, j, i);
          if (ndim > 1) len = std::max(len, coords.template Dxc<X2DIR>(k, j, i));
          if (ndim > 2) len = std::max(len, coords.template Dxc<X3DIR>(k, j, i));
          const Real bsq = SQR(pv(ccbulk::magnetic_field(0), idx)) +
                           SQR(pv(ccbulk::magnetic_field(1), idx)) +
                           SQR(pv(ccbulk::magnetic_field(2), idx));
          const Real bloc = std::max(std::sqrt(bsq), b_ref);
          m = std::max(m, std::abs(pv(ccbulk::div_magnetic_field(), idx)) * len / bloc);
        });
      });

  auto sum_space = rsum::GetIndexSpace(IndexDomain::interior, 0, nb, md, te);
  Real divb_vol = RiotLoop::outer_reduce(
      sum_space, KOKKOS_LAMBDA(const rsum::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &s) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          s += std::abs(pv(ccbulk::div_magnetic_field(), idx)) *
               coords.CellVolume(k, j, i);
        });
      });
  Real vol_tot = RiotLoop::outer_reduce(
      sum_space, KOKKOS_LAMBDA(const rsum::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &s) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          s += coords.CellVolume(k, j, i);
        });
      });

#ifdef MPI_PARALLEL
  Real mx[2] = {divb_max, eta_max};
  Real sm[2] = {divb_vol, vol_tot};
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, mx, 2, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, sm, 2, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
  divb_max = mx[0];
  eta_max = mx[1];
  divb_vol = sm[0];
  vol_tot = sm[1];
#endif

  if (parthenon::Globals::my_rank == 0) {
    const Real divb_mean = (vol_tot > 0.0) ? divb_vol / vol_tot : 0.0;
    // ncycle + 1 and time + dt, NOT simtime's own values. Parthenon calls this hook after
    // Step() but BEFORE `tm.ncycle++; tm.time += tm.dt` (driver.cpp:185-190), so the
    // state being measured here is the state at the END of the step while `simtime` still
    // holds the labels from its START. Printing simtime directly reports the first step's
    // result as "cycle=0 t=0", which reads as an initial-condition diagnostic and would
    // send anyone comparing this against an output file's cycle number chasing an
    // off-by-one.
    printf("divB: cycle=%d t=%.6e  max=%.6e  mean=%.6e  eta=%.6e  |B|max=%.6e\n",
           simtime.ncycle + 1, simtime.time + simtime.dt, divb_max, divb_mean, eta_max,
           bmax);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::AddMagneticEnergyToTotal
//! \brief Promotes ccbulk::total_material_energy from the hydro convention (u + kinetic)
//!        to the MHD convention (u + kinetic + B^2/(2 mu0)). Called by every MHD problem
//!        generator as its LAST step.
//!
//! WHY THIS LIVES IN THE PROBLEM GENERATOR and not in PostInitialization, which is where
//! it naturally belongs: Mesh::Initialize runs a full PreCommFillDerived / communicate /
//! FillDerived cycle on the freshly generated state BEFORE any PostInitialization hook
//! (mesh.cpp, the `do {} while (!init_done)` loop). Multiphysics::FillInteriorDerived in
//! that cycle recovers the thermal energy by subtracting B^2/(2 mu0) -- reading face B,
//! which the pgen has already written -- so a hydro-convention total energy has the
//! magnetic part removed once before any later correction can be applied.
//!
//! That is not recoverable afterwards. The subtract/re-add pair is self-inverse only
//! while the intermediate thermal energy stays positive; where it goes negative it is
//! clipped, and the information is gone. Concretely, in Brio-Wu the right state (u = 0.1,
//! B^2/2 = 0.78125) went to u = -0.68125, was clipped to 0, and the state settled at
//! u = 0.78125 with P = 0.78125 instead of 0.1 -- while the left state, where no clipping
//! occurred, came out exactly right. A bug that corrupts only the states where a floor
//! triggers is precisely the kind that survives a casual look at a plot.
//!
//! So the conversion has to happen before that first cycle, i.e. inside the pgen. It
//! still goes through the same MHD::CellMagneticEnergyFromFaces helper that
//! FillInteriorDerived subtracts and PostCommsFillDerived re-adds, which is what ADR-002
//! actually requires: one definition of the magnetic energy, not one call site.
void AddMagneticEnergyToTotal(MeshBlock *pmb) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace fbulk = face_variables::bulk;
  using parthenon::MakePackDescriptor;

  auto &rc = pmb->meshblock_data.Get();
  const Real mu0 = pmb->packages.Get("mhd")->template Param<Real>("mu0");
  const int multi_d = (pmb->pmy_mesh->ndim > 1);
  const int three_d = (pmb->pmy_mesh->ndim > 2);

  static auto desc =
      MakePackDescriptor<ccbulk::total_material_energy, fbulk::magnetic_field>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // IndexDomain::entire: the pgen initializes ghosts too, and the first derived pass
  // subtracts the magnetic energy over the interior only -- but a subsequent one runs on
  // `entire`, so the ghosts must carry the same convention.
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "MHD::AddMagneticEnergyToTotal", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, ccbulk::total_material_energy(), k, j, i) +=
            CellMagneticEnergyFromFaces(v, 0, k, j, i, multi_d, three_d, mu0);
      });
}

namespace impl {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::impl::AssembleOneEdgeEMF
//! \brief Write the EMF on every edge along direction E1 of the interior.
//!
//! Templated on E1 and on whether the edge basis' E3 direction is collapsed, so that the
//! KOKKOS_LAMBDA sits directly in a function body (an extended device lambda nested
//! inside another lambda is not portable) and so the collapsed branch costs nothing at
//! runtime.
//!
//! The loop bounds come straight from Parthenon's edge bounds, which already carry the
//! reduced-dimension behavior this needs: TopologicalOffsetI/J/K
//! (basic_types.hpp:296-304) add the extra element only on the two axes transverse to the
//! edge, and IndexShape suppresses the offset entirely on a collapsed axis
//! (domain.hpp:228-252). So in 2D the E3 edge range is (js..je+1, is..ie+1) with a single
//! k plane, with no special-casing here.
//!
//! Written with RiotFlatLoop rather than the loop abstraction on purpose: every access in
//! this kernel is logical-coordinate (the EMF destination is edge-shaped storage and must
//! not be reached through a flat cell-shaped view -- see MHD::FaceB), there is no scratch
//! and no halo, so the loop abstraction would add machinery without buying anything.
template <int E1, bool E3_COLLAPSED, typename Pack_t>
void AssembleOneEdgeEMF(MeshData<Real> *md, const Pack_t &v, const int nblocks) {
  namespace fbulk = face_variables::bulk;
  using TE = parthenon::TopologicalElement;
  constexpr TE edge_te = (E1 == X1DIR) ? TE::E1 : ((E1 == X2DIR) ? TE::E2 : TE::E3);

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, nblocks, md, edge_te);
  RiotFlatLoop::four_d(
      "MHD::AssembleEdgeEMF", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real emf = E3_COLLAPSED ? CollapsedEdgeEMF<E1>(v, b, k, j, i)
                                      : UpwindEMF<E1>(v, b, k, j, i);
        v.flux(b, edge_te, fbulk::magnetic_field(), k, j, i) = emf;
      });
}
} // namespace impl

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus MHD::AssembleEdgeEMF
//! \brief Builds the edge EMFs from the face fluxes the hydro flux task produced.
//!
//! Which edges need which formula follows from one rule: an edge along E1 uses the full
//! upwind reconstruction when the E3 direction of its basis is active, the collapsed form
//! when E3 is collapsed but E2 is active, and is skipped when both are collapsed. Applied
//! to the three edge directions that is exactly the donor's table
//! (DONOR_KERNELS.md section 5):
//!
//!   E3 edge (E2 = X1DIR, E3 = X2DIR): upwind if multi_d,  else collapsed
//!   E2 edge (E2 = X1DIR, E3 = X3DIR): upwind if three_d,  else collapsed
//!   E1 edge (E2 = X2DIR, E3 = X3DIR): upwind if three_d,  collapsed if multi_d,
//!                                     otherwise SKIPPED
//!
//! Skipping E1 in 1D is correct rather than a gap: the B1 face update differences E3
//! across x2 and E2 across x3, both collapsed, so it is identically zero -- and it must
//! be, since div B = d_1 B1 = 0 forces B1 uniform in 1D.
TaskStatus AssembleEdgeEMF(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace fbulk = face_variables::bulk;

  if (md->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = md->GetParentPointer();
  const int ndim = pm->ndim;
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);

  // ccmat::rho is packed for its mass flux (the upwind selector); ccbulk::magnetic_field
  // for its induction flux registers; ccbulk::velocity and ccbulk::magnetic_field for the
  // cell-centered EMFs; fbulk::magnetic_field for the edge register being written.
  auto v = riot::MakePack<ccbulk::velocity, ccbulk::magnetic_field, ccmat::rho,
                          fbulk::magnetic_field>(
      md, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  if (multi_d) {
    impl::AssembleOneEdgeEMF<X3DIR, false>(md, v, nblocks);
  } else {
    impl::AssembleOneEdgeEMF<X3DIR, true>(md, v, nblocks);
  }

  if (three_d) {
    impl::AssembleOneEdgeEMF<X2DIR, false>(md, v, nblocks);
  } else {
    impl::AssembleOneEdgeEMF<X2DIR, true>(md, v, nblocks);
  }

  if (three_d) {
    impl::AssembleOneEdgeEMF<X1DIR, false>(md, v, nblocks);
  } else if (multi_d) {
    impl::AssembleOneEdgeEMF<X1DIR, true>(md, v, nblocks);
  }

  return TaskStatus::complete;
}

namespace impl {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::impl::ApplyOneFaceUpdate
//! \brief Constrained-transport update of one face component.
//!
//! Implements d_t(B_n A_f) = -\oint E . dl with right-handed orientation, i.e.
//!
//!   d_t(B1 A1) = -[ E3 L3 |_{j+1} - E3 L3 |_j ] + [ E2 L2 |_{k+1} - E2 L2 |_k ]
//!   d_t(B2 A2) = -[ E1 L1 |_{k+1} - E1 L1 |_k ] + [ E3 L3 |_{i+1} - E3 L3 |_i ]
//!   d_t(B3 A3) = -[ E2 L2 |_{i+1} - E2 L2 |_i ] + [ E1 L1 |_{j+1} - E1 L1 |_j ]
//!
//! re-derived independently of the donor and then checked against it term by term
//! (DONOR_KERNELS.md section 6a). This is the cyclic pattern d_t B_a = -(d_b E_c - d_c
//! E_b).
//!
//! Why the flux-balance form (edge length times stored EMF, then differenced) rather than
//! a divided difference: an edge is shared by four faces, and div B is preserved at
//! roundoff only if those four contributions cancel ALGEBRAICALLY. Differencing the same
//! stored edge value multiplied by the same edge length guarantees that two neighboring
//! faces subtract exactly the same floating-point number. A form that recomputed the
//! quotient per face would cancel only to roundoff, which accumulates.
//!
//! Uniform Cartesian: edge lengths are the constant coords.Dx and face areas the constant
//! coords.FaceArea. (The donor evaluates its length factors at slightly inconsistent
//! neighbor cells -- dq3p from coords_kp paired with the E3 flux at j+1, for instance --
//! which is exact on a uniform grid and would need re-deriving otherwise.)
//!
//! The collapsed-direction offsets make the corresponding difference identically zero,
//! which is what keeps B1 constant in 1D and B3's out-of-plane evolution correct in 2D.
template <parthenon::TopologicalElement FACE_TE, typename Pack_t>
void ApplyOneFaceUpdate(MeshData<Real> *md, const Pack_t &v0, const Pack_t &v1,
                        const Real gam0, const Real gam1, const Real beta_dt,
                        const int multi_d, const int three_d, const int nblocks) {
  namespace fbulk = face_variables::bulk;
  using TE = parthenon::TopologicalElement;

  auto space = RiotFlatLoop::GetIndexSpace(IndexDomain::interior, nblocks, md, FACE_TE);
  RiotFlatLoop::four_d(
      "MHD::ApplyFaceUpdate", space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &coords = v0.GetCoordinates(b);
        const Real l1 = coords.template Dx<X1DIR>();
        const Real l2 = coords.template Dx<X2DIR>();
        const Real l3 = coords.template Dx<X3DIR>();

        // Edge EMFs. Reading them through the pack with logical coordinates is required:
        // the register is edge-shaped storage (see MHD::FaceB).
        auto emf = [&](const TE te, const int kk, const int jj, const int ii) {
          return v0.flux(b, te, fbulk::magnetic_field(), kk, jj, ii);
        };

        Real curl = 0.0;
        Real area = 1.0;
        if constexpr (FACE_TE == TE::F1) {
          area = coords.template FaceArea<X1DIR>(k, j, i);
          curl = l3 * (emf(TE::E3, k, j + multi_d, i) - emf(TE::E3, k, j, i)) -
                 l2 * (emf(TE::E2, k + three_d, j, i) - emf(TE::E2, k, j, i));
        } else if constexpr (FACE_TE == TE::F2) {
          area = coords.template FaceArea<X2DIR>(k, j, i);
          curl = l1 * (emf(TE::E1, k + three_d, j, i) - emf(TE::E1, k, j, i)) -
                 l3 * (emf(TE::E3, k, j, i + 1) - emf(TE::E3, k, j, i));
        } else {
          area = coords.template FaceArea<X3DIR>(k, j, i);
          curl = l2 * (emf(TE::E2, k, j, i + 1) - emf(TE::E2, k, j, i)) -
                 l1 * (emf(TE::E1, k, j + multi_d, i) - emf(TE::E1, k, j, i));
        }

        // Low-storage RK combination, identical in form to the cell-centered update in
        // sparse_update::UpdateToNextStage. NOTE that the u1 read covers EVERY face,
        // including the last plane in the normal direction -- which is why the stage
        // register copy must be face-aware (sparse_update::DeepCopyFaceData); a truncated
        // copy is invisible in a first RK stage and corrupts the second.
        Real &b0 = v0(b, FACE_TE, fbulk::magnetic_field(), k, j, i);
        const Real b1 = v1(b, FACE_TE, fbulk::magnetic_field(), k, j, i);
        b0 = gam0 * b0 + gam1 * b1 - (beta_dt / area) * curl;
      });
}
} // namespace impl

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus MHD::ApplyFaceUpdate
//! \brief Advances the face magnetic field by the Stokes curl of the edge EMFs.
//!
//! All three components are updated unconditionally, including one whose own normal
//! direction is collapsed. That is required, not wasteful: the transverse field
//! components are physically nonzero and evolving in a reduced-dimension run (B2 and B3
//! in 1D, B3 in 2D). Their loop bounds degenerate to a single plane and the collapsed
//! differences vanish on their own.
TaskStatus ApplyFaceUpdate(MeshData<Real> *u0md, MeshData<Real> *u1md, const Real gam0,
                           const Real gam1, const Real beta_dt) {
  namespace fbulk = face_variables::bulk;
  using TE = parthenon::TopologicalElement;

  if (u0md->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = u0md->GetParentPointer();
  const int multi_d = (pm->ndim > 1);
  const int three_d = (pm->ndim > 2);

  // One descriptor, two containers -- the same arrangement UpdateToNextStage uses, so the
  // pack indices line up between the state being advanced and the stage register.
  auto v0 = riot::MakePack<fbulk::magnetic_field>(
      u0md, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  auto v1 = riot::MakePack<fbulk::magnetic_field>(
      u1md, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  const int nblocks = v0.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  impl::ApplyOneFaceUpdate<TE::F1>(u0md, v0, v1, gam0, gam1, beta_dt, multi_d, three_d,
                                   nblocks);
  impl::ApplyOneFaceUpdate<TE::F2>(u0md, v0, v1, gam0, gam1, beta_dt, multi_d, three_d,
                                   nblocks);
  impl::ApplyOneFaceUpdate<TE::F3>(u0md, v0, v1, gam0, gam1, beta_dt, multi_d, three_d,
                                   nblocks);

  return TaskStatus::complete;
}

} // namespace MHD
