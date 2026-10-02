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
#ifndef MHD_RIEMANN_MHD_HPP_
#define MHD_RIEMANN_MHD_HPP_
// This file was made in part with generative AI.
//
// Ideal-MHD Riemann solvers.
//
// Adapted from src/utils/fluxes/riemann/hlle.hpp on the lanl/artemis branch dempsey/mhd
// @ 3e5aeb5. The donor kernel is quoted verbatim, with the term-by-term correspondence
// to this implementation, in plan_histories/artemis_mhd_port/DONOR_KERNELS.md section 7.
//
// These follow RIOT's hydro solver conventions (src/hydro/riemann.hpp), NOT the donor's
// functor interface: free functions templated on the sweep direction, taking and
// returning scalars, returning the maximum signal speed. Two convention differences from
// the donor are deliberate and are documented at the point of use below:
//
//   1. Pressure is carried INSIDE the normal momentum flux, as RIOT's hydro solvers do
//      (riemann.hpp:157-159), rather than in a separate face-pressure register applied
//      later as a gradient. The two are algebraically identical; see the note on
//      `ptot_flux` below.
//   2. Everything is written in GLOBAL vector components rather than a
//      (normal, transverse, transverse) permutation. The ideal-MHD momentum and
//      induction fluxes are expressible componentwise with only the scalar normal field
//      `bn` appearing, so no cyclic index bookkeeping is needed -- which removes the
//      most error-prone part of a port like this. Verified against the donor
//      term-for-term and by unit test U02 in all three directions.

#include <cmath>

#include <parthenon/package.hpp>

#include "mhd/mhd.hpp"
#include "mhd/mhd_helpers.hpp"

namespace MHD {

using parthenon::Real;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::lr_to_flux_mhd_hlle
//! \brief Ideal-MHD HLLE (Harten--Lax--van Leer--Einfeldt) flux in direction DIR.
//!
//! \param[in]  bn        magnetic field component NORMAL to the interface. This is a
//!                       SINGLE shared value, taken from the face-centered state and
//!                       supplied identically to both sides -- it is never
//!                       reconstructed twice. Constrained transport depends on this.
//! \param[in]  b1l..b3r  transverse magnetic field components; the entries in the
//!                       normal slot are ignored, `bn` is authoritative.
//! \param[in]  cl, cr    acoustic sound speeds sqrt(bmod/rho), as for the hydro solvers.
//!                       The bulk modulus is recovered as rho*c^2 so that the general-EOS
//!                       acoustic derivative RIOT already aggregates is what enters the
//!                       fast speed -- no constant-gamma assumption is introduced.
//! \param[in]  ul, ur    VOLUMETRIC internal energy (matching RIOT's hydro solvers),
//!                       so total energy is u + 1/2 rho |v|^2 + B^2/(2 mu0).
//! \param[out] f_b1..3   induction fluxes. The normal component is set to EXACTLY 0.0.
//! \return               max(|sl|, |sr|), the signal speed for the CFL vote.
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_mhd_hlle(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, const Real bn, const Real b1l,
    const Real b1r, const Real b2l, const Real b2r, const Real b3l, const Real b3r,
    const Real mu0, Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &f_b1,
    Real &f_b2, Real &f_b3, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel,
    std::int64_t *diag) {

  // Counted, not silently applied: routine floor activity is a failed test, not a
  // success, and without a counter the only way to notice is to inspect output positivity
  // after the fact. See MHD::SolverDiag.
  if (rhol <= 0.0 || rhor <= 0.0) {
    Kokkos::atomic_add(&diag[kDiagDensityFloor], static_cast<std::int64_t>(1));
  }
  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);

  // Normal velocity, selected branch-free exactly as the hydro solvers do.
  const Real vnl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  const Real vnr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  // Use the shared normal field in place of whatever sits in the normal slot of the
  // reconstructed triples, so that the two sides cannot disagree even if a caller
  // forgets the substitution.
  const Real b1lc = (DIR == X1DIR) ? bn : b1l;
  const Real b2lc = (DIR == X2DIR) ? bn : b2l;
  const Real b3lc = (DIR == X3DIR) ? bn : b3l;
  const Real b1rc = (DIR == X1DIR) ? bn : b1r;
  const Real b2rc = (DIR == X2DIR) ? bn : b2r;
  const Real b3rc = (DIR == X3DIR) ? bn : b3r;

  const Real bsql = SQR(b1lc) + SQR(b2lc) + SQR(b3lc);
  const Real bsqr = SQR(b1rc) + SQR(b2rc) + SQR(b3rc);

  // Magnetic pressure B^2/(2 mu0), and the TOTAL pressure that appears in the normal
  // momentum flux.
  const Real pbl = MagneticEnergyDensity(b1lc, b2lc, b3lc, mu0);
  const Real pbr = MagneticEnergyDensity(b1rc, b2rc, b3rc, mu0);
  const Real ptot_l = Pl + pbl;
  const Real ptot_r = Pr + pbr;

  // Conserved total energy, INCLUDING magnetic energy (ADR-002).
  const Real el = ul + 0.5 * rhol * (SQR(v1l) + SQR(v2l) + SQR(v3l)) + pbl;
  const Real er = ur + 0.5 * rhor * (SQR(v1r) + SQR(v2r) + SQR(v3r)) + pbr;

  // Fast magnetosonic speeds. bmod = rho c^2 recovers RIOT's bulk modulus.
  const Real cfl = FastMagnetosonicSpeed(rhol * cl * cl, rhol, bsql, bn, mu0);
  const Real cfr = FastMagnetosonicSpeed(rhor * cr * cr, rhor, bsqr, bn, mu0);

  // Einfeldt (1988) Eq. 5.7 wave-speed estimate, on the Roe-averaged normal velocity.
  // Note the fast speeds -- not the acoustic speeds -- enter here; using the Alfven
  // speed alone or the sound speed alone are both classic porting errors.
  const Real sqrtdl = std::sqrt(rhol);
  const Real sqrtdr = std::sqrt(rhor);
  const Real isdlpdr = 1.0 / (sqrtdl + sqrtdr);
  const Real wroe_vn = (sqrtdl * vnl + sqrtdr * vnr) * isdlpdr;
  const Real ngam = 0.5 * sqrtdl * sqrtdr * SQR(isdlpdr);
  const Real aroe = std::sqrt((SQR(cfl) * sqrtdl + SQR(cfr) * sqrtdr) * isdlpdr +
                              ngam * SQR(vnr - vnl));
  const Real sl = std::min(wroe_vn - aroe, vnl - cfl);
  const Real sr = std::max(wroe_vn + aroe, vnr + cfr);

  // Clamped away from zero to avoid a 0/0 in converging supersonic flow, following the
  // donor (and Athena before it).
  const Real bp = (sr > 0.0) ? sr : 1.0e-20;
  const Real bm = (sl < 0.0) ? sl : -1.0e-20;

  // L/R fluxes along the bounding characteristics: F - S*U.
  const Real qa = vnl - bm;
  const Real qb = vnr - bp;

  const Real fl_d = rhol * qa;
  const Real fr_d = rhor * qb;

  // Momentum: rho v_i v_n - b_n b_i / mu0. The isotropic total-pressure term is added
  // to the normal component separately, below.
  const Real fl_m1 = rhol * v1l * qa - bn * b1lc / mu0;
  const Real fr_m1 = rhor * v1r * qb - bn * b1rc / mu0;
  const Real fl_m2 = rhol * v2l * qa - bn * b2lc / mu0;
  const Real fr_m2 = rhor * v2r * qb - bn * b2rc / mu0;
  const Real fl_m3 = rhol * v3l * qa - bn * b3lc / mu0;
  const Real fr_m3 = rhor * v3r * qb - bn * b3rc / mu0;

  // Energy: (E + p + B^2/2mu0) v_n - b_n (v.B)/mu0, in F - S*U form.
  const Real vdbl = v1l * b1lc + v2l * b2lc + v3l * b3lc;
  const Real vdbr = v1r * b1rc + v2r * b2rc + v3r * b3rc;
  const Real fl_e = el * qa + ptot_l * vnl - bn * vdbl / mu0;
  const Real fr_e = er * qb + ptot_r * vnr - bn * vdbr / mu0;

  // Induction: F_i(B) = v_n b_i - v_i b_n, again in F - S*U form with U = b_i.
  const Real fl_b1 = (vnl * b1lc - v1l * bn) - bm * b1lc;
  const Real fr_b1 = (vnr * b1rc - v1r * bn) - bp * b1rc;
  const Real fl_b2 = (vnl * b2lc - v2l * bn) - bm * b2lc;
  const Real fr_b2 = (vnr * b2rc - v2r * bn) - bp * b2rc;
  const Real fl_b3 = (vnl * b3lc - v3l * bn) - bm * b3lc;
  const Real fr_b3 = (vnr * b3rc - v3r * bn) - bp * b3rc;

  // HLLE combination. With w = (bp+bm)/(2(bp-bm)),
  //   0.5(fL+fR) + w(fL-fR) == (bp fL - bm fR)/(bp-bm),
  // which is the standard HLLE flux once fL/fR carry their -S*U shifts. Written in this
  // algebraically equivalent form because it is what the donor uses, so the two codes
  // agree to roundoff rather than merely to truncation error.
  const Real w = (bp != bm) ? 0.5 * (bp + bm) / (bp - bm) : 0.0;
  auto hlle = [&](const Real fl, const Real fr) {
    return 0.5 * (fl + fr) + w * (fl - fr);
  };

  const Real frho = hlle(fl_d, fr_d);

  // The pressure part of the momentum flux. Applying the same HLLE weighting to the
  // total pressure is exactly the (bp ptot_l - bm ptot_r)/(bp-bm) term that would arise
  // if pressure had been carried in fL/fR from the start -- so this reproduces the
  // donor's separate face-pressure register while keeping RIOT's pressure-in-flux
  // convention.
  const Real ptot_flux = hlle(ptot_l, ptot_r);

  f_v1 = hlle(fl_m1, fr_m1) + (DIR == X1DIR) * ptot_flux;
  f_v2 = hlle(fl_m2, fr_m2) + (DIR == X2DIR) * ptot_flux;
  f_v3 = hlle(fl_m3, fr_m3) + (DIR == X3DIR) * ptot_flux;
  f_eng = hlle(fl_e, fr_e);

  // The normal induction flux vanishes identically under HLLE (both states carry the
  // same bn, so the shifts cancel), but it is set to exactly 0.0 rather than left to
  // roundoff: a nonzero value here would leak into the EMF assembly and break the
  // divergence-free property that constrained transport exists to maintain.
  f_b1 = (DIR == X1DIR) ? 0.0 : hlle(fl_b1, fr_b1);
  f_b2 = (DIR == X2DIR) ? 0.0 : hlle(fl_b2, fr_b2);
  f_b3 = (DIR == X3DIR) ? 0.0 : hlle(fl_b3, fr_b3);

  // Upwind face state, selected by the sign of the mass flux (HLLE has no contact wave
  // to key off, unlike HLLC). The normal component is the mass-flux velocity, which is
  // what the material-density and advection loops consume, and whose SIGN is what the
  // Gardiner-Stone EMF upwinding needs.
  const Real l_flag = 1.0 * (frho >= 0.0);
  const Real r_flag = 1.0 - l_flag;
  const Real rho_up = l_flag * rhol + r_flag * rhor;
  riemann_vel = frho / rho_up;

  v1face = (DIR == X1DIR) ? riemann_vel : (l_flag * v1l + r_flag * v1r);
  v2face = (DIR == X2DIR) ? riemann_vel : (l_flag * v2l + r_flag * v2r);
  v3face = (DIR == X3DIR) ? riemann_vel : (l_flag * v3l + r_flag * v3r);

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::lr_to_flux_mhd_hlld
//! \brief Ideal-MHD HLLD flux (Miyoshi & Kusano 2005) in direction DIR.
//!
//! Ported from artemis/src/utils/fluxes/riemann/hlld.hpp lines 40-430 (the
//! single-material branch; the donor's species >= 1 HLLC fallback is not carried over).
//! The argument list is identical to `lr_to_flux_mhd_hlle` so the two are interchangeable
//! at the call site.
//!
//! HLLD resolves five waves -- two fast, two rotational (Alfven), and the contact --
//! instead of HLLE's two, so contact and rotational discontinuities are captured without
//! the numerical diffusion that HLLE spreads over several cells. That is what makes it
//! the donor's default and what the Brio-Wu compound structure is sensitive to.
//!
//! TWO STRUCTURAL DEPARTURES FROM THE HLLE ABOVE, both forced by the algorithm rather
//! than chosen:
//!
//!   1. **It works in the rotated (normal, t_a, t_b) frame**, not in global components.
//!   The
//!      rotational-discontinuity jump mixes the two transverse components through
//!      `sgn(b_n)`, so the transverse pair has to be treated as an oriented 2-vector;
//!      there is no componentwise form. The permutation is the cyclic right-handed one,
//!      (nc, (nc+1)%3, (nc+2)%3), which is the same convention as
//!      `RiotUtils::DirBasis` (`src/riot_utils/riot_loops.hpp:64-90`) and as the donor's
//!      `((dir-1)+1)%3` indices. Getting this wrong is the single most likely porting
//!      error, so unit tests exercise all three directions and all component
//!      permutations.
//!   2. **Its wave-speed estimate is `min/max(v_n -+ c_f)`, not the Roe average** used by
//!   the
//!      HLLE above. That is the donor's choice and is kept deliberately: HLLD's star
//!      states are derived assuming sl and sr bracket all five waves, and the simple
//!      bound is the one the derivation and its degeneracy guards were tuned against. Do
//!      not "improve" this to the Einfeldt estimate without re-deriving the guards.
//!
//! ROBUSTNESS. Every guard from DONOR_KERNELS.md section 13 is preserved verbatim,
//! including the exact constants. They are not defensive clutter -- they are the
//! difference between a solver that survives low-beta and near-degenerate states and one
//! that emits NaNs. Any guard that trips falls back to the HLLE flux computed alongside
//! (which is why that is evaluated unconditionally rather than lazily: it is needed on
//! every path where HLLD's intermediate states are not trustworthy).
//!
//! Restricted to an ideal gas, as the donor is; the caller enforces this.
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_mhd_hlld(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, const Real bn, const Real b1l,
    const Real b1r, const Real b2l, const Real b2r, const Real b3l, const Real b3r,
    const Real mu0, Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &f_b1,
    Real &f_b2, Real &f_b3, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel,
    std::int64_t *diag) {

  // Counted, not silently applied: routine floor activity is a failed test, not a
  // success, and without a counter the only way to notice is to inspect output positivity
  // after the fact. See MHD::SolverDiag.
  if (rhol <= 0.0 || rhor <= 0.0) {
    Kokkos::atomic_add(&diag[kDiagDensityFloor], static_cast<std::int64_t>(1));
  }
  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);

  // Cyclic right-handed permutation: normal component, then the two transverse ones.
  constexpr int nc = DIR - 1;
  constexpr int tac = (nc + 1) % 3;
  constexpr int tbc = (nc + 2) % 3;

  const Real vlg[3] = {v1l, v2l, v3l};
  const Real vrg[3] = {v1r, v2r, v3r};
  // The normal slot is overwritten with the shared face value, exactly as in the HLLE
  // above, so a caller that forgot the substitution still cannot make the two sides
  // disagree about b_n.
  const Real blg[3] = {(DIR == X1DIR) ? bn : b1l, (DIR == X2DIR) ? bn : b2l,
                       (DIR == X3DIR) ? bn : b3l};
  const Real brg[3] = {(DIR == X1DIR) ? bn : b1r, (DIR == X2DIR) ? bn : b2r,
                       (DIR == X3DIR) ? bn : b3r};

  const Real vxl = vlg[nc], vyl = vlg[tac], vzl = vlg[tbc];
  const Real vxr = vrg[nc], vyr = vrg[tac], vzr = vrg[tbc];
  const Real byl = blg[tac], bzl = blg[tbc];
  const Real byr = brg[tac], bzr = brg[tbc];

  constexpr Real small = 1.0e-20;
  constexpr Real eps = 1.0e-12;

  // The donor forms bxi as the average of the two normal slots; with the substitution
  // above both equal bn, so bxi == bn identically. Kept explicit for traceability.
  const Real bxi = bn;
  const Real sqrt_mu0 = std::sqrt(mu0);
  const Real inv_sqrt_mu0 = 1.0 / sqrt_mu0;

  // HLLD's internal normalization is B/sqrt(mu0), which makes magnetic pressure 0.5*b_n^2
  // and puts the Alfven speed at |b_n|/sqrt(rho). Induction fluxes stay in PHYSICAL B
  // units, which is why the star-state field corrections below are multiplied back by
  // sqrt(mu0).
  const Real bxi_n = bxi * inv_sqrt_mu0;
  const Real byl_n = byl * inv_sqrt_mu0;
  const Real bzl_n = bzl * inv_sqrt_mu0;
  const Real byr_n = byr * inv_sqrt_mu0;
  const Real bzr_n = bzr * inv_sqrt_mu0;
  const Real bxsq_n = SQR(bxi_n);

  const Real pbl = 0.5 * (bxsq_n + SQR(byl_n) + SQR(bzl_n));
  const Real pbr = 0.5 * (bxsq_n + SQR(byr_n) + SQR(bzr_n));
  const Real ptl = Pl + pbl;
  const Real ptr = Pr + pbr;
  const Real vdotBl = vxl * bxi_n + vyl * byl_n + vzl * bzl_n;
  const Real vdotBr = vxr * bxi_n + vyr * byr_n + vzr * bzr_n;

  // ul/ur are VOLUMETRIC internal energies in RIOT (the donor carries specific internal
  // energy and multiplies by density here).
  const Real el = ul + 0.5 * rhol * (SQR(vxl) + SQR(vyl) + SQR(vzl)) + pbl;
  const Real er = ur + 0.5 * rhor * (SQR(vxr) + SQR(vyr) + SQR(vzr)) + pbr;

  const Real fl_d = rhol * vxl;
  const Real fl_mx = rhol * SQR(vxl) - bxsq_n;
  const Real fl_my = rhol * vxl * vyl - bxi_n * byl_n;
  const Real fl_mz = rhol * vxl * vzl - bxi_n * bzl_n;
  const Real fl_e = (el + ptl) * vxl - bxi_n * vdotBl;
  const Real fl_by = vxl * byl - bxi * vyl;
  const Real fl_bz = vxl * bzl - bxi * vzl;

  const Real fr_d = rhor * vxr;
  const Real fr_mx = rhor * SQR(vxr) - bxsq_n;
  const Real fr_my = rhor * vxr * vyr - bxi_n * byr_n;
  const Real fr_mz = rhor * vxr * vzr - bxi_n * bzr_n;
  const Real fr_e = (er + ptr) * vxr - bxi_n * vdotBr;
  const Real fr_by = vxr * byr - bxi * vyr;
  const Real fr_bz = vxr * bzr - bxi * vzr;

  // bmod = rho c^2, so the general-EOS acoustic derivative is what enters the fast speed.
  const Real cfl = FastMagnetosonicSpeed(rhol * cl * cl, rhol, bxi, byl, bzl, mu0);
  const Real cfr = FastMagnetosonicSpeed(rhor * cr * cr, rhor, bxi, byr, bzr, mu0);

  const Real sl = std::min(vxl - cfl, vxr - cfr);
  const Real sr = std::max(vxl + cfl, vxr + cfr);

  // HLLE flux, evaluated unconditionally: it is the fallback for every degeneracy guard
  // below, and branching around it would not save work on the paths that matter.
  const Real hlle_qa = (sr > 0.0) ? sr : small;
  const Real hlle_qb = (sl < 0.0) ? sl : -small;
  const Real hlle_qc = 0.5 * (hlle_qa + hlle_qb) / (hlle_qa - hlle_qb);
  auto hlle = [&](const Real a, const Real b) {
    return 0.5 * (a + b) + hlle_qc * (a - b);
  };
  const Real qal = vxl - hlle_qb;
  const Real qar = vxr - hlle_qa;
  const Real hlle_p = hlle(Pl, Pr);
  const Real hlle_pb = hlle(pbl, pbr);
  const Real hlle_frho = hlle(qal * rhol, qar * rhor);
  const Real hlle_fmx = hlle(qal * rhol * vxl - bxsq_n, qar * rhor * vxr - bxsq_n);
  const Real hlle_fmy =
      hlle(qal * rhol * vyl - bxi_n * byl_n, qar * rhor * vyr - bxi_n * byr_n);
  const Real hlle_fmz =
      hlle(qal * rhol * vzl - bxi_n * bzl_n, qar * rhor * vzr - bxi_n * bzr_n);
  const Real hlle_fe =
      hlle(el * qal + ptl * vxl - bxi_n * vdotBl, er * qar + ptr * vxr - bxi_n * vdotBr);
  const Real hlle_fby = hlle(fl_by - hlle_qb * byl, fr_by - hlle_qa * byr);
  const Real hlle_fbz = hlle(fl_bz - hlle_qb * bzl, fr_bz - hlle_qa * bzr);

  const Real sdl = sl - vxl;
  const Real sdr = sr - vxr;
  const Real denom = rhor * sdr - rhol * sdl;

  // Guard 1: a vanishing normal field makes the rotational waves degenerate with the
  // contact and the star-state algebra singular; a near-zero denominator or a non-finite
  // wave speed is unusable outright.
  bool use_hlle = (std::abs(bxi) <= eps) || (std::abs(denom) <= small) ||
                  !std::isfinite(sl) || !std::isfinite(sr);

  Real sm = 0.0, ptst = 0.0, dlst = 0.0, drst = 0.0, sal = 0.0, sar = 0.0;
  Real vlst_y = 0.0, vlst_z = 0.0, blst_y = 0.0, blst_z = 0.0, elst = 0.0;
  Real vrst_y = 0.0, vrst_z = 0.0, brst_y = 0.0, brst_z = 0.0, erst = 0.0;
  Real vdst_y = 0.0, vdst_z = 0.0, bdst_y = 0.0, bdst_z = 0.0;
  Real eldst = 0.0, erdst = 0.0;

  if (!use_hlle) {
    sm = (rhor * sdr * vxr - rhol * sdl * vxl + ptl - ptr) / denom;
    const Real ptstl = ptl + rhol * sdl * (sm - vxl);
    const Real ptstr = ptr + rhor * sdr * (sm - vxr);
    ptst = 0.5 * (ptstl + ptstr);

    const Real sdml = sl - sm;
    const Real sdmr = sr - sm;
    // Guard 2: the contact coinciding with a fast wave.
    if (std::abs(sdml) <= small || std::abs(sdmr) <= small) {
      use_hlle = true;
    } else {
      dlst = rhol * sdl / sdml;
      drst = rhor * sdr / sdmr;

      // Guard 3: an unphysical star state. Falling back is correct here rather than
      // clipping -- a negative star density means the wave pattern assumed by the
      // derivation does not hold, so the derived fluxes are meaningless, not merely
      // out of range.
      if (dlst <= 0.0 || drst <= 0.0 || !std::isfinite(dlst) || !std::isfinite(drst) ||
          ptst <= 0.0 || !std::isfinite(ptst)) {
        use_hlle = true;
      } else {
        const Real dsl = rhol * sdl * sdml - bxsq_n;
        const Real dsr = rhor * sdr * sdmr - bxsq_n;
        // Guard 4: rotational degeneracy. The tolerance is RELATIVE to the star total
        // pressure, which is what makes it scale-free -- an absolute tolerance would
        // misfire at extreme field strengths.
        const Real deg_tol = 1.0e-4 * ptst;

        if (std::abs(dsl) < deg_tol) {
          vlst_y = vyl;
          vlst_z = vzl;
          blst_y = byl;
          blst_z = bzl;
        } else {
          const Real inv_dsl = 1.0 / dsl;
          const Real mfact = bxi_n * (sm - vxl) * inv_dsl;
          const Real bfact = (rhol * SQR(sdl) - bxsq_n) * inv_dsl;
          vlst_y = vyl - byl_n * mfact;
          vlst_z = vzl - bzl_n * mfact;
          blst_y = byl_n * bfact;
          blst_z = bzl_n * bfact;
        }

        if (std::abs(dsr) < deg_tol) {
          vrst_y = vyr;
          vrst_z = vzr;
          brst_y = byr;
          brst_z = bzr;
        } else {
          const Real inv_dsr = 1.0 / dsr;
          const Real mfact = bxi_n * (sm - vxr) * inv_dsr;
          const Real bfact = (rhor * SQR(sdr) - bxsq_n) * inv_dsr;
          vrst_y = vyr - byr_n * mfact;
          vrst_z = vzr - bzr_n * mfact;
          brst_y = byr_n * bfact;
          brst_z = bzr_n * bfact;
        }

        const Real vbstl = sm * bxi_n + vlst_y * blst_y + vlst_z * blst_z;
        const Real vbstr = sm * bxi_n + vrst_y * brst_y + vrst_z * brst_z;
        elst = (sdl * el - ptl * vxl + ptst * sm + bxi_n * (vdotBl - vbstl)) / sdml;
        erst = (sdr * er - ptr * vxr + ptst * sm + bxi_n * (vdotBr - vbstr)) / sdmr;

        sal = sm - std::abs(bxi_n) / std::sqrt(dlst);
        sar = sm + std::abs(bxi_n) / std::sqrt(drst);

        const Real sqrtdlst = std::sqrt(dlst);
        const Real sqrtdrst = std::sqrt(drst);
        const Real denom_dst = sqrtdlst + sqrtdrst;
        if (std::abs(denom_dst) <= small || !std::isfinite(denom_dst)) {
          use_hlle = true;
        } else {
          const Real sgnbx = (bxi_n >= 0.0) ? 1.0 : -1.0;
          const Real inv_dst = 1.0 / denom_dst;

          // Guard 5: when the normal field is too weak for the two rotational waves to
          // separate, the double-star state collapses onto the single-star one.
          if (0.5 * bxsq_n < deg_tol) {
            vdst_y = vlst_y;
            vdst_z = vlst_z;
            bdst_y = blst_y;
            bdst_z = blst_z;
            eldst = elst;
            erdst = erst;
          } else {
            vdst_y = (sqrtdlst * vlst_y + sqrtdrst * vrst_y + (brst_y - blst_y) * sgnbx) *
                     inv_dst;
            vdst_z = (sqrtdlst * vlst_z + sqrtdrst * vrst_z + (brst_z - blst_z) * sgnbx) *
                     inv_dst;
            bdst_y = (sqrtdlst * brst_y + sqrtdrst * blst_y +
                      sqrtdlst * sqrtdrst * (vrst_y - vlst_y) * sgnbx) *
                     inv_dst;
            bdst_z = (sqrtdlst * brst_z + sqrtdrst * blst_z +
                      sqrtdlst * sqrtdrst * (vrst_z - vlst_z) * sgnbx) *
                     inv_dst;

            const Real vbdst = sm * bxi_n + vdst_y * bdst_y + vdst_z * bdst_z;
            eldst = elst - sqrtdlst * (vbstl - vbdst) * sgnbx;
            erdst = erst + sqrtdrst * (vbstr - vbdst) * sgnbx;
          }

          // Guard 6: any non-finite intermediate state invalidates the whole pattern.
          use_hlle = !std::isfinite(sal) || !std::isfinite(sar) || !std::isfinite(elst) ||
                     !std::isfinite(erst) || !std::isfinite(eldst) ||
                     !std::isfinite(erdst);
        }
      }
    }
  }

  // Sample the fan. `pface`/`pmag_face` are the donor's separate gas- and
  // magnetic-pressure face registers; RIOT carries total pressure inside the normal
  // momentum flux, so their SUM is added to `fmx` at the end. In the star branches that
  // sum is `ptst`, which cancels the explicit `-ptst` in `fmx` -- the two conventions
  // really are the same flux.
  Real frho, fmx, fmy, fmz, fe, fby, fbz, pface, pmag_face;

  if (use_hlle) {
    // The single most important counter in the port. Every one of the six degeneracy
    // guards above lands here, and a guard that tripped on most cells would silently turn
    // the user's HLLD into HLLE while every existing test still passed. See
    // MHD::kDiagHlldFallback.
    Kokkos::atomic_add(&diag[kDiagHlldFallback], static_cast<std::int64_t>(1));
    frho = hlle_frho;
    fmx = hlle_fmx;
    fmy = hlle_fmy;
    fmz = hlle_fmz;
    fe = hlle_fe;
    fby = hlle_fby;
    fbz = hlle_fbz;
    pface = hlle_p;
    pmag_face = hlle_pb;
  } else if (sl >= 0.0) {
    frho = fl_d;
    fmx = fl_mx;
    fmy = fl_my;
    fmz = fl_mz;
    fe = fl_e;
    fby = fl_by;
    fbz = fl_bz;
    pface = Pl;
    pmag_face = pbl;
  } else if (sal >= 0.0) {
    frho = fl_d + sl * (dlst - rhol);
    fmx = fl_mx + (ptl - ptst) + sl * (dlst * sm - rhol * vxl);
    fmy = fl_my + sl * (dlst * vlst_y - rhol * vyl);
    fmz = fl_mz + sl * (dlst * vlst_z - rhol * vzl);
    fe = fl_e + sl * (elst - el);
    fby = fl_by + sl * (sqrt_mu0 * blst_y - byl);
    fbz = fl_bz + sl * (sqrt_mu0 * blst_z - bzl);
    pmag_face = 0.5 * (bxsq_n + SQR(blst_y) + SQR(blst_z));
    pface = ptst - pmag_face;
  } else if (sm >= 0.0) {
    frho = fl_d + sl * (dlst - rhol);
    fmx = fl_mx + (ptl - ptst) + sl * (dlst * sm - rhol * vxl);
    fmy = fl_my + sl * (dlst * vlst_y - rhol * vyl) + sal * dlst * (vdst_y - vlst_y);
    fmz = fl_mz + sl * (dlst * vlst_z - rhol * vzl) + sal * dlst * (vdst_z - vlst_z);
    fe = fl_e + sl * (elst - el) + sal * (eldst - elst);
    fby = fl_by + sl * (sqrt_mu0 * blst_y - byl) + sal * sqrt_mu0 * (bdst_y - blst_y);
    fbz = fl_bz + sl * (sqrt_mu0 * blst_z - bzl) + sal * sqrt_mu0 * (bdst_z - blst_z);
    pmag_face = 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
    pface = ptst - pmag_face;
  } else if (sar > 0.0) {
    frho = fr_d + sr * (drst - rhor);
    fmx = fr_mx + (ptr - ptst) + sr * (drst * sm - rhor * vxr);
    fmy = fr_my + sr * (drst * vrst_y - rhor * vyr) + sar * drst * (vdst_y - vrst_y);
    fmz = fr_mz + sr * (drst * vrst_z - rhor * vzr) + sar * drst * (vdst_z - vrst_z);
    fe = fr_e + sr * (erst - er) + sar * (erdst - erst);
    fby = fr_by + sr * (sqrt_mu0 * brst_y - byr) + sar * sqrt_mu0 * (bdst_y - brst_y);
    fbz = fr_bz + sr * (sqrt_mu0 * brst_z - bzr) + sar * sqrt_mu0 * (bdst_z - brst_z);
    pmag_face = 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
    pface = ptst - pmag_face;
  } else if (sr > 0.0) {
    frho = fr_d + sr * (drst - rhor);
    fmx = fr_mx + (ptr - ptst) + sr * (drst * sm - rhor * vxr);
    fmy = fr_my + sr * (drst * vrst_y - rhor * vyr);
    fmz = fr_mz + sr * (drst * vrst_z - rhor * vzr);
    fe = fr_e + sr * (erst - er);
    fby = fr_by + sr * (sqrt_mu0 * brst_y - byr);
    fbz = fr_bz + sr * (sqrt_mu0 * brst_z - bzr);
    pmag_face = 0.5 * (bxsq_n + SQR(brst_y) + SQR(brst_z));
    pface = ptst - pmag_face;
  } else {
    frho = fr_d;
    fmx = fr_mx;
    fmy = fr_my;
    fmz = fr_mz;
    fe = fr_e;
    fby = fr_by;
    fbz = fr_bz;
    pface = Pr;
    pmag_face = pbr;
  }

  // Rotate back to global components, folding total pressure into the normal momentum.
  Real fm[3], fb[3];
  fm[nc] = fmx + (pface + pmag_face);
  fm[tac] = fmy;
  fm[tbc] = fmz;
  // Exactly 0.0, not merely small: a nonzero normal induction flux would enter the EMF
  // assembly and destroy the divergence-free property that CT exists to maintain.
  fb[nc] = 0.0;
  fb[tac] = fby;
  fb[tbc] = fbz;

  f_v1 = fm[0];
  f_v2 = fm[1];
  f_v3 = fm[2];
  f_eng = fe;
  f_b1 = fb[0];
  f_b2 = fb[1];
  f_b3 = fb[2];

  // Upwind face state, keyed on the sign of the mass flux -- the same convention as the
  // HLLE above, and the sign the Gardiner-Stone EMF upwinding consumes.
  const Real l_flag = 1.0 * (frho >= 0.0);
  const Real r_flag = 1.0 - l_flag;
  riemann_vel = frho / (l_flag * rhol + r_flag * rhor);

  v1face = (DIR == X1DIR) ? riemann_vel : (l_flag * v1l + r_flag * v1r);
  v2face = (DIR == X2DIR) ? riemann_vel : (l_flag * v2l + r_flag * v2r);
  v3face = (DIR == X3DIR) ? riemann_vel : (l_flag * v3l + r_flag * v3r);

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::lr_to_flux_mhd_llf
//! \brief Ideal-MHD local Lax-Friedrichs (Rusanov) flux in direction DIR.
//!
//! Ported from artemis/src/utils/fluxes/riemann/llf.hpp, the `FLUID_TYPE != radiation`
//! specialization, restricted to its `do_mhd` branch. Reference: Toro, "Riemann Solvers
//! and Numerical Methods for Fluid Dynamics", 2nd ed., ch. 10 (Eq. 10.43 for the speed).
//!
//! The whole scheme is
//!   F = 1/2 (F_L + F_R) - 1/2 a (U_R - U_L),   a = max(|v_n| + c_f)
//! i.e. a centred flux plus the most diffusive stable amount of upwinding. There are no
//! intermediate states, so -- unlike HLLD -- there is nothing to degenerate and no
//! degeneracy guards exist. Like HLLE and unlike HLLD it needs only the bulk modulus, so
//! it carries NO ideal-gas restriction.
//!
//! WHY KEEP IT, given it is strictly less accurate than both others: it is the fallback
//! that has no failure mode. When a problem misbehaves under HLLD or HLLE, running it
//! under LLF separates "the solver's intermediate states are breaking down" from "the
//! reconstruction, CT, or initial condition is wrong", because LLF has no intermediate
//! states to break down. That diagnostic is the reason the donor ships all three.
//!
//! The argument list is identical to `lr_to_flux_mhd_hlle` and `..._hlld`, so all three
//! are interchangeable at the call site.
//!
//! \return  `a`, the single signal speed, for the CFL vote. LLF is symmetric
//!          (effectively sl = -a, sr = +a), so this plays the same role as
//!          max(|sl|, |sr|) does for the other two.
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_mhd_llf(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, const Real bn, const Real b1l,
    const Real b1r, const Real b2l, const Real b2r, const Real b3l, const Real b3r,
    const Real mu0, Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &f_b1,
    Real &f_b2, Real &f_b3, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel,
    std::int64_t *diag) {

  // Counted, not silently applied: routine floor activity is a failed test, not a
  // success, and without a counter the only way to notice is to inspect output positivity
  // after the fact. See MHD::SolverDiag.
  if (rhol <= 0.0 || rhor <= 0.0) {
    Kokkos::atomic_add(&diag[kDiagDensityFloor], static_cast<std::int64_t>(1));
  }
  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);

  const Real vnl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  const Real vnr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  // Shared normal field substituted into the normal slot, exactly as in the HLLE. This
  // is what makes the normal induction flux vanish IDENTICALLY below rather than
  // approximately.
  const Real b1lc = (DIR == X1DIR) ? bn : b1l;
  const Real b2lc = (DIR == X2DIR) ? bn : b2l;
  const Real b3lc = (DIR == X3DIR) ? bn : b3l;
  const Real b1rc = (DIR == X1DIR) ? bn : b1r;
  const Real b2rc = (DIR == X2DIR) ? bn : b2r;
  const Real b3rc = (DIR == X3DIR) ? bn : b3r;

  const Real bsql = SQR(b1lc) + SQR(b2lc) + SQR(b3lc);
  const Real bsqr = SQR(b1rc) + SQR(b2rc) + SQR(b3rc);

  const Real pbl = MagneticEnergyDensity(b1lc, b2lc, b3lc, mu0);
  const Real pbr = MagneticEnergyDensity(b1rc, b2rc, b3rc, mu0);
  const Real ptot_l = Pl + pbl;
  const Real ptot_r = Pr + pbr;

  // Conserved total energy, INCLUDING magnetic energy (ADR-002). The donor reaches the
  // same value by forming the hydro energy first and adding pb later; the difference is
  // presentation, and both the energy flux and the dissipation term below use the total.
  const Real el = ul + 0.5 * rhol * (SQR(v1l) + SQR(v2l) + SQR(v3l)) + pbl;
  const Real er = ur + 0.5 * rhor * (SQR(v1r) + SQR(v2r) + SQR(v3r)) + pbr;

  const Real cfl = FastMagnetosonicSpeed(rhol * cl * cl, rhol, bsql, bn, mu0);
  const Real cfr = FastMagnetosonicSpeed(rhor * cr * cr, rhor, bsqr, bn, mu0);

  // Toro Eq. 10.43. Note |v_n| rather than the signed velocity, and the FAST speed
  // rather than the acoustic one -- with c_s here the scheme would be unstable wherever
  // the field dominates, which is the classic way this solver gets ported wrong.
  const Real a = std::max(std::abs(vnl) + cfl, std::abs(vnr) + cfr);

  const Real vdbl = v1l * b1lc + v2l * b2lc + v3l * b3lc;
  const Real vdbr = v1r * b1rc + v2r * b2rc + v3r * b3rc;

  // Sum of the two PHYSICAL fluxes. Momentum carries -b_n b_i/mu0; the isotropic total
  // pressure is added to the normal component below, per RIOT's pressure-in-flux
  // convention (the donor instead routes it to a separate face-pressure register).
  const Real fsum_d = rhol * vnl + rhor * vnr;
  const Real fsum_m1 = rhol * v1l * vnl + rhor * v1r * vnr - bn * (b1lc + b1rc) / mu0;
  const Real fsum_m2 = rhol * v2l * vnl + rhor * v2r * vnr - bn * (b2lc + b2rc) / mu0;
  const Real fsum_m3 = rhol * v3l * vnl + rhor * v3r * vnr - bn * (b3lc + b3rc) / mu0;
  const Real fsum_e =
      (el + ptot_l) * vnl + (er + ptot_r) * vnr - bn * (vdbl + vdbr) / mu0;
  const Real fsum_b1 = (vnl * b1lc - v1l * bn) + (vnr * b1rc - v1r * bn);
  const Real fsum_b2 = (vnl * b2lc - v2l * bn) + (vnr * b2rc - v2r * bn);
  const Real fsum_b3 = (vnl * b3lc - v3l * bn) + (vnr * b3rc - v3r * bn);

  // Dissipation: a * (U_R - U_L) on the CONSERVED variables. Pressure gets no such term
  // because it is not a conserved variable -- it appears only in the physical flux, which
  // is why the normal momentum flux below picks up a plain average of the total pressure.
  const Real du_d = a * (rhor - rhol);
  const Real du_m1 = a * (rhor * v1r - rhol * v1l);
  const Real du_m2 = a * (rhor * v2r - rhol * v2l);
  const Real du_m3 = a * (rhor * v3r - rhol * v3l);
  const Real du_e = a * (er - el);
  const Real du_b1 = a * (b1rc - b1lc);
  const Real du_b2 = a * (b2rc - b2lc);
  const Real du_b3 = a * (b3rc - b3lc);

  const Real ptot_flux = 0.5 * (ptot_l + ptot_r);

  const Real frho = 0.5 * (fsum_d - du_d);

  f_v1 = 0.5 * (fsum_m1 - du_m1) + (DIR == X1DIR) * ptot_flux;
  f_v2 = 0.5 * (fsum_m2 - du_m2) + (DIR == X2DIR) * ptot_flux;
  f_v3 = 0.5 * (fsum_m3 - du_m3) + (DIR == X3DIR) * ptot_flux;
  f_eng = 0.5 * (fsum_e - du_e);

  // Both fsum and du vanish identically in the normal slot once bn has been substituted
  // above, so this assignment is not a correction -- but it is written explicitly for the
  // same reason as in the HLLE: anything nonzero here leaks into the EMF and destroys the
  // divergence-free property that constrained transport exists to maintain.
  f_b1 = (DIR == X1DIR) ? 0.0 : 0.5 * (fsum_b1 - du_b1);
  f_b2 = (DIR == X2DIR) ? 0.0 : 0.5 * (fsum_b2 - du_b2);
  f_b3 = (DIR == X3DIR) ? 0.0 : 0.5 * (fsum_b3 - du_b3);

  // Upwind face state on the sign of the mass flux, identical to the other two solvers.
  const Real l_flag = 1.0 * (frho >= 0.0);
  const Real r_flag = 1.0 - l_flag;
  riemann_vel = frho / (l_flag * rhol + r_flag * rhor);

  v1face = (DIR == X1DIR) ? riemann_vel : (l_flag * v1l + r_flag * v1r);
  v2face = (DIR == X2DIR) ? riemann_vel : (l_flag * v2l + r_flag * v2r);
  v3face = (DIR == X3DIR) ? riemann_vel : (l_flag * v3l + r_flag * v3r);

  return a;
}

} // namespace MHD

#endif // MHD_RIEMANN_MHD_HPP_
