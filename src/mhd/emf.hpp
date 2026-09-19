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
#ifndef MHD_EMF_HPP_
#define MHD_EMF_HPP_
// This file was made in part with generative AI.
//
// Gardiner & Stone (2005) upwind edge-centered EMFs for constrained transport.
//
// Adapted from src/mhd/emf.hpp on the lanl/artemis branch dempsey/mhd @ 3e5aeb5. The
// donor kernel is quoted verbatim, with the index decoding worked out and the Cartesian
// reduction derived, in plan_histories/artemis_mhd_port/DONOR_KERNELS.md sections 2-5.
//
// This is the Cartesian reduction (section 4b), not a transcription. In Cartesian every
// scale factor is unity and the half-cell distance from each face centroid to the edge
// equals the grid spacing that appears in the denominator of the corresponding transverse
// gradient, so the two cancel exactly and the kernel collapses to plain differences of
// EMFs -- no cell widths, no scale factors, no coordinate objects. Writing that form
// directly also avoids importing an O(delta) curvilinear inconsistency present in the
// donor's scale-factor pairing (DONOR_KERNELS.md section 4c); if geometry support is ever
// added, re-derive the metric factors from the Stokes line integral rather than
// transcribing them.
//
// Sign convention, verified independently in DONOR_KERNELS.md section 6a:
//
//   E = -(v x B)          =>   d_t B = -curl(E)
//   Fx(By) = -Ez   Fy(Bx) = +Ez   Fx(Bz) = +Ey
//   Fz(Bx) = -Ey   Fy(Bz) = -Ex   Fz(By) = +Ex

#include <parthenon/package.hpp>

#include "mhd/mhd_helpers.hpp"
#include "variables.hpp"

namespace MHD {

using parthenon::Real;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

//----------------------------------------------------------------------------------------
//! \struct  MHD::EdgeBasis
//! \brief Compile-time index bookkeeping for the EMF on an edge along direction E1.
//!
//! Reproduces the donor's constexpr preamble (emf.hpp:77-82) exactly. E2 and E3 are the
//! two directions transverse to the edge; the EMF on the edge is assembled from the four
//! face fluxes on the E2 and E3 faces that touch it.
//!
//!   E1 = X1DIR  ->  E2 = X2DIR, E3 = X3DIR
//!   E1 = X2DIR  ->  E2 = X1DIR, E3 = X3DIR
//!   E1 = X3DIR  ->  E2 = X1DIR, E3 = X2DIR
//!
//! `e2_bcomp` is the magnetic-field component whose E2-directed flux carries the E1 EMF,
//! and the signs come from Fa(Bb) = +/- Ec with the cyclic orientation above. Note that
//! E2/E3 are NOT in cyclic order for E1 = X2DIR (that is the donor's choice and the sign
//! factors compensate), which is why these are tabulated rather than derived from a
//! cyclic permutation.
template <int E1>
struct EdgeBasis {
  static constexpr int e2 = (E1 == X1DIR) ? X2DIR : X1DIR;
  static constexpr int e3 = (E1 == X3DIR) ? X2DIR : X3DIR;
  static constexpr int e2_bcomp = e3 - 1;
  static constexpr int e3_bcomp = e2 - 1;
  static constexpr int e2_sign = 1 - 2 * (E1 % 2);
  static constexpr int e3_sign = -e2_sign;
};

//! Unit index offsets along the axis of a coordinate direction.
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION constexpr int OffK() {
  return (DIR == X3DIR);
}
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION constexpr int OffJ() {
  return (DIR == X2DIR);
}
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION constexpr int OffI() {
  return (DIR == X1DIR);
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::CellEMF
//! \brief Cell-centered EMF component along EDGE, i.e. -(v x B) evaluated at a cell
//!        center from the cell-averaged velocity and cell-centered magnetic field.
//!
//! Ported verbatim from artemis/src/mhd/emf.hpp:22-34. Note the sign: for EDGE == X1DIR
//! this returns v3*b2 - v2*b3, which is -(v x B)_1, not +(v x B)_1.
//!
//! There is no 1/mu0 here. The EMF has units of velocity times field, and the mu0 factor
//! enters the induction equation only through the Lorentz force in the momentum and
//! energy fluxes -- so a stray 1/mu0 in this function would break the field evolution
//! while leaving the energy budget looking self-consistent.
template <int EDGE, typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real CellEMF(const Pack_t &v, const int b, const int k,
                                         const int j, const int i) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const Real v1 = v(b, ccbulk::velocity(0), k, j, i);
  const Real v2 = v(b, ccbulk::velocity(1), k, j, i);
  const Real v3 = v(b, ccbulk::velocity(2), k, j, i);
  const Real b1 = v(b, ccbulk::magnetic_field(0), k, j, i);
  const Real b2 = v(b, ccbulk::magnetic_field(1), k, j, i);
  const Real b3 = v(b, ccbulk::magnetic_field(2), k, j, i);
  if constexpr (EDGE == X1DIR) return v3 * b2 - v2 * b3;
  if constexpr (EDGE == X2DIR) return v1 * b3 - v3 * b1;
  return v2 * b1 - v1 * b2;
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::UpwindEMFGradient
//! \brief Donor-cell selection of a transverse EMF derivative by the sign of the mass
//!        flux at the same face. Ported verbatim from artemis/src/mhd/emf.hpp:38-42.
//!
//! mass_flux > 0 takes `left`, < 0 takes `right`, and exactly 0 takes the arithmetic
//! mean. Branch-free because this is evaluated four times per edge.
//!
//! This upwinding is what makes the scheme reduce to the correct 1D solution when the
//! flow is grid-aligned, and it is the only place the mass flux is needed -- no stored
//! face velocities and no upwind flags.
KOKKOS_FORCEINLINE_FUNCTION Real UpwindEMFGradient(const Real mass_flux, const Real left,
                                                   const Real right) {
  const int sign = (mass_flux > 0.0) - (mass_flux < 0.0);
  return 0.5 * ((1.0 + sign) * left + (1.0 - sign) * right);
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::BulkMassFlux
//! \brief Bulk mass flux through a face, summed over every packed material and phase.
//!
//! Only the SIGN of this is used (by UpwindEMFGradient), which is why no separate storage
//! is needed for it: `riemann_vel` is block-local scratch inside CalculateFluxesImpl and
//! is gone by the time the EMF task runs, but the ccmat::rho flux register persists, and
//! summing it over materials IS the bulk mass flux. Summing over the whole packed range
//! rather than assuming a single slot keeps this correct if a material carries multiple
//! phases, even though the MHD support gate currently allows only one material.
template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real BulkMassFlux(const Pack_t &v, const int b, const int dir,
                                              const int k, const int j, const int i) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  Real flux = 0.0;
  const int lo = v.GetLowerBound(b, ccmat::rho());
  const int hi = v.GetUpperBound(b, ccmat::rho());
  for (int n = lo; n <= hi; ++n)
    flux += v.flux(b, dir, n, k, j, i);
  return flux;
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::FaceEMF
//! \brief The E1 EMF carried by a face flux of the transverse magnetic field.
//!
//! `DIR` is the face direction (E2 or E3) and `BCOMP` the field component; the sign comes
//! from EdgeBasis. This reads the flux register of ccbulk::magnetic_field, which
//! MHDFluxes wrote -- a cell-centered variable, so its flux array is cell-shaped and
//! logical indexing is straightforward.
template <int DIR, int BCOMP, int SIGN, typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real FaceEMF(const Pack_t &v, const int b, const int k,
                                         const int j, const int i) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  return SIGN * v.flux(b, DIR, ccbulk::magnetic_field(BCOMP), k, j, i);
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::UpwindEMF
//! \brief Gardiner & Stone (2005) Eq. 51 upwind EMF on the edge along E1 at (k, j, i).
//!
//! Each of the four face EMFs surrounding the edge is extrapolated a half cell to the
//! edge using a donor-cell transverse derivative selected by the mass flux at that same
//! face, and the four results are averaged. In Cartesian the half-cell distance cancels
//! the gradient denominator exactly, so this is the difference form of
//! DONOR_KERNELS.md section 4b; the 0.25 times half-differences is the 1/8 of the
//! textbook Gardiner-Stone expression.
//!
//! Requires face fluxes on one extra transverse layer relative to the edge being written
//! (the pm/qm/mm offsets below reach one cell back in both transverse directions), which
//! is what Hydro::MHDFluxIndexSpace provides.
template <int E1, typename Pack_t>
KOKKOS_INLINE_FUNCTION Real UpwindEMF(const Pack_t &v, const int b, const int k,
                                      const int j, const int i) {
  using EB = EdgeBasis<E1>;

  // pp is the cell diagonally "up" from the edge; pm/qm step back one cell along the E2
  // and E3 axes respectively, and mm steps back along both. The edge sits at the common
  // corner of these four cells.
  constexpr int pm_k = -OffK<EB::e2>(), pm_j = -OffJ<EB::e2>(), pm_i = -OffI<EB::e2>();
  constexpr int qm_k = -OffK<EB::e3>(), qm_j = -OffJ<EB::e3>(), qm_i = -OffI<EB::e3>();
  constexpr int mm_k = pm_k + qm_k, mm_j = pm_j + qm_j, mm_i = pm_i + qm_i;

  // The four face EMFs. ea_* live on E2 faces and are offset along the E3 axis; eb_* live
  // on E3 faces and are offset along the E2 axis.
  const Real ea_lo =
      FaceEMF<EB::e2, EB::e2_bcomp, EB::e2_sign>(v, b, k + qm_k, j + qm_j, i + qm_i);
  const Real ea_hi = FaceEMF<EB::e2, EB::e2_bcomp, EB::e2_sign>(v, b, k, j, i);
  const Real eb_lo =
      FaceEMF<EB::e3, EB::e3_bcomp, EB::e3_sign>(v, b, k + pm_k, j + pm_j, i + pm_i);
  const Real eb_hi = FaceEMF<EB::e3, EB::e3_bcomp, EB::e3_sign>(v, b, k, j, i);

  // The four cell-centered EMFs.
  const Real emm = CellEMF<E1>(v, b, k + mm_k, j + mm_j, i + mm_i);
  const Real epm = CellEMF<E1>(v, b, k + pm_k, j + pm_j, i + pm_i);
  const Real eqm = CellEMF<E1>(v, b, k + qm_k, j + qm_j, i + qm_i);
  const Real epp = CellEMF<E1>(v, b, k, j, i);

  // Mass fluxes selecting the upwind side of each extrapolation.
  const Real fa_lo = BulkMassFlux(v, b, EB::e2, k + qm_k, j + qm_j, i + qm_i);
  const Real fa_hi = BulkMassFlux(v, b, EB::e2, k, j, i);
  const Real fb_lo = BulkMassFlux(v, b, EB::e3, k + pm_k, j + pm_j, i + pm_i);
  const Real fb_hi = BulkMassFlux(v, b, EB::e3, k, j, i);

  const Real ga_lo = UpwindEMFGradient(fa_lo, eb_lo - emm, eb_hi - eqm);
  const Real ga_hi = UpwindEMFGradient(fa_hi, epm - eb_lo, epp - eb_hi);
  const Real gb_lo = UpwindEMFGradient(fb_lo, ea_lo - emm, ea_hi - epm);
  const Real gb_hi = UpwindEMFGradient(fb_hi, eqm - ea_lo, epp - ea_hi);

  return 0.25 * ((ea_lo + ga_lo) + (ea_hi - ga_hi) + (eb_lo + gb_lo) + (eb_hi - gb_hi));
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::CollapsedEdgeEMF
//! \brief Edge EMF when the E3 direction of the edge basis is collapsed.
//!
//! With no E3 direction there is no transverse direction to upwind along and no second
//! pair of face EMFs, so the edge EMF is simply the single face EMF at that location.
//! This is the donor's reduced-dimension shortcut (DONOR_KERNELS.md section 5) and it is
//! load-bearing, not an optimization: in 1D the transverse field components still evolve
//! and in 2D the out-of-plane component still evolves. Dropping this branch would
//! silently freeze physically evolving field components.
//!
//! For every case the donor tabulates, the collapsed direction is E3:
//!   1D,       E3 edge (E1 = X3DIR): E2 = X1DIR active, E3 = X2DIR collapsed
//!   2D,       E2 edge (E1 = X2DIR): E2 = X1DIR active, E3 = X3DIR collapsed
//!   2D,       E1 edge (E1 = X1DIR): E2 = X2DIR active, E3 = X3DIR collapsed
//! In 1D the E1 edge has both transverse directions collapsed; its EMF is never needed,
//! because the B1 face update differences it across collapsed directions and so is
//! identically zero -- which is correct, since div B = d_1 B1 = 0 forces B1 uniform.
template <int E1, typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real CollapsedEdgeEMF(const Pack_t &v, const int b,
                                                  const int k, const int j, const int i) {
  using EB = EdgeBasis<E1>;
  return FaceEMF<EB::e2, EB::e2_bcomp, EB::e2_sign>(v, b, k, j, i);
}

} // namespace MHD

#endif // MHD_EMF_HPP_
