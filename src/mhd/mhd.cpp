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
      "Report max |div B| before and after each step (diagnostic; costs a reduction)");
  params.Add("monitor_divb", monitor_divb);

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
