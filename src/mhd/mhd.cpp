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
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::WithFluxes,
                Metadata::Vector},
               bfield_arr_size);
  mhd->AddField<ccbulk::magnetic_field>(m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::WithFluxes});
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
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);

  auto v = riot::MakePack<fbulk::magnetic_field, ccbulk::magnetic_field,
                          ccbulk::magnetic_energy, ccbulk::div_magnetic_field>(md);
  if (v.GetNBlocks() == 0) return;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(domain, 0, v.GetNBlocks(), md, TE::CC);

  // Offsets to the upper face in each direction. These are zero in a collapsed
  // dimension, which makes the two face reads coincide: the face-to-cell average
  // degenerates to the single available value, and the corresponding divergence term
  // cancels identically. That is the correct behavior -- a transverse field component
  // still evolves in a reduced-dimension run.
  auto di = idx_space.GetDelta(X1DIR);
  auto dj = idx_space.GetDelta(X2DIR);
  auto dk = idx_space.GetDelta(X3DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);

        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);

          Real bx, by, bz;
          FaceToCellB(pv, kji, di, dj, dk, bx, by, bz);
          pv(ccbulk::magnetic_field(0), kji) = bx;
          pv(ccbulk::magnetic_field(1), kji) = by;
          pv(ccbulk::magnetic_field(2), kji) = bz;
          pv(ccbulk::magnetic_energy(), kji) =
              MagneticEnergyDensity(bx, by, bz, mu0);

          // Discrete divergence as the signed sum of magnetic FLUXES through the cell
          // faces, divided by cell volume. Computing it this way -- rather than from
          // centered differences of the cell-centered field -- is essential: it is the
          // quantity that constrained transport holds at roundoff, and it is the only
          // form in which the shared-edge EMF contributions cancel algebraically.
          Real divb = coords.template FaceArea<X1DIR>(k, j, i + 1) *
                          pv(TE::F1, fbulk::magnetic_field(), kji + di) -
                      coords.template FaceArea<X1DIR>(k, j, i) *
                          pv(TE::F1, fbulk::magnetic_field(), kji);
          if (multi_d) {
            divb += coords.template FaceArea<X2DIR>(k, j + 1, i) *
                        pv(TE::F2, fbulk::magnetic_field(), kji + dj) -
                    coords.template FaceArea<X2DIR>(k, j, i) *
                        pv(TE::F2, fbulk::magnetic_field(), kji);
          }
          if (three_d) {
            divb += coords.template FaceArea<X3DIR>(k + 1, j, i) *
                        pv(TE::F3, fbulk::magnetic_field(), kji + dk) -
                    coords.template FaceArea<X3DIR>(k, j, i) *
                        pv(TE::F3, fbulk::magnetic_field(), kji);
          }
          pv(ccbulk::div_magnetic_field(), kji) = divb / coords.CellVolume(k, j, i);
        });
      });
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
  // Derived state is needed on `entire` because the first total-energy assembly and the
  // first reconstruction both read ghost zones.
  SetDerivedMagneticFields(md, IndexDomain::entire);
}

} // namespace MHD
