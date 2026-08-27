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
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see licenses/bsd_athenapp.txt file for details
//========================================================================================

#ifndef HYDRO_RIEMANN_HPP_
#define HYDRO_RIEMANN_HPP_
// This file was made in part with generative AI.

namespace Hydro {

enum RiemannSolver { hllc, hllcf, chllc, lhllc, hll, strong };

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_fleischmann
//! \brief
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_fleischmann(
    Real rhol, Real rhor, Real v1l, Real v1r, Real v2l, Real v2r, Real v3l, Real v3r,
    const Real ul, const Real ur, const Real Pl, const Real Pr, const Real cl,
    const Real cr, Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &v1face,
    Real &v2face, Real &v3face, Real &riemann_vel) {
  // implements the shock-stable, low-Mach corrected solver of
  // Fleischmann, Adami, & Adams, 2020, J Comp Phys, 423, 109762
  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);
  Real vpl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  Real vpr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  const Real sl = std::min(vpl - cl, vpr - cr);
  const Real sr = std::max(vpl + cl, vpr + cr);

  const Real rhocsl = rhol * (sl - vpl);
  const Real rhocsr = rhor * (sr - vpr);

  const Real Ma_l = std::abs(vpl / cl);
  const Real Ma_r = std::abs(vpr / cr);
  Real Ma_local = std::max(Ma_l, Ma_r);
  Real phi = std::sin(std::min(1.0, Ma_local / 0.1) * 0.5 * M_PI);
  Real ss = (sl >= 0) * vpl + (sr <= 0) * vpr +
            (sl * sr < 0) * (rhocsl * vpl - rhocsr * vpr + (Pr - Pl)) / (rhocsl - rhocsr);

  const Real rhos_rhol = (sl - vpl) / (sl - ss);
  const Real rhos_rhor = (sr - vpr) / (sr - ss);
  const Real rho_upwind = (ss >= 0.0) * rhol + (ss < 0.0) * rhor;

  const Real v1fl = (DIR == X1DIR ? ss : v1l);
  const Real v1fr = (DIR == X1DIR ? ss : v1r);
  const Real v2fl = (DIR == X2DIR ? ss : v2l);
  const Real v2fr = (DIR == X2DIR ? ss : v2r);
  const Real v3fl = (DIR == X3DIR ? ss : v3l);
  const Real v3fr = (DIR == X3DIR ? ss : v3r);

  v1face = (DIR == X1DIR ? ss : (ss >= 0.0) * v1l + (ss < 0.0) * v1r);
  v2face = (DIR == X2DIR ? ss : (ss >= 0.0) * v2l + (ss < 0.0) * v2r);
  v3face = (DIR == X3DIR ? ss : (ss >= 0.0) * v3l + (ss < 0.0) * v3r);

  // now rescale sl and sr
  const Real slp = sl * phi;
  const Real srp = sr * phi;
  const Real super_l = 0.5 * (1 + (sl >= 0.0) - (sr <= 0.0));
  const Real super_r = 0.5 * (1 + (sr <= 0.0) - (sl >= 0.0));
  const Real super_beta = 2.0 * super_l * super_r;

  const Real frho = super_l * rhol * vpl + super_r * rhor * vpr +
                    super_beta * (slp * rhol * (rhos_rhol - 1.0) +
                                  std::abs(ss) * (rhos_rhol * rhol - rhos_rhor * rhor) +
                                  srp * rhor * (rhos_rhor - 1.0));
  riemann_vel = frho / rho_upwind;

  f_v1 =
      super_l * (rhol * vpl * v1l + (DIR == X1DIR) * Pl) +
      super_r * (rhor * vpr * v1r + (DIR == X1DIR) * Pr) +
      super_beta * (slp * rhol * (rhos_rhol * v1fl - v1l) +
                    std::abs(ss) * (rhos_rhol * rhol * v1fl - rhos_rhor * rhor * v1fr) +
                    srp * rhor * (rhos_rhor * v1fr - v1r));
  f_v2 =
      super_l * (rhol * vpl * v2l + (DIR == X2DIR) * Pl) +
      super_r * (rhor * vpr * v2r + (DIR == X2DIR) * Pr) +
      super_beta * (slp * rhol * (rhos_rhol * v2fl - v2l) +
                    std::abs(ss) * (rhos_rhol * rhol * v2fl - rhos_rhor * rhor * v2fr) +
                    srp * rhor * (rhos_rhor * v2fr - v2r));
  f_v3 =
      super_l * (rhol * vpl * v3l + (DIR == X3DIR) * Pl) +
      super_r * (rhor * vpr * v3r + (DIR == X3DIR) * Pr) +
      super_beta * (slp * rhol * (rhos_rhol * v3fl - v3l) +
                    std::abs(ss) * (rhos_rhol * rhol * v3fl - rhos_rhor * rhor * v3fr) +
                    srp * rhor * (rhos_rhor * v3fr - v3r));

  const Real El = ul + 0.5 * rhol * (v1l * v1l + v2l * v2l + v3l * v3l);
  const Real Esl = rhos_rhol * (El + (ss - vpl) * (rhol * ss + Pl / (sl - vpl)));
  const Real Er = ur + 0.5 * rhor * (v1r * v1r + v2r * v2r + v3r * v3r);
  const Real Esr = rhos_rhor * (Er + (ss - vpr) * (rhor * ss + Pr / (sr - vpr)));
  f_eng = super_l * vpl * (El + Pl) + super_r * vpr * (Er + Pr) +
          super_beta * (slp * (Esl - El) + std::abs(ss) * (Esl - Esr) + srp * (Esr - Er));

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_hllc
//! \brief
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_hllc(
    Real rhol, Real rhor, Real v1l, Real v1r, Real v2l, Real v2r, Real v3l, Real v3r,
    const Real ul, const Real ur, const Real Pl, const Real Pr, const Real cl,
    const Real cr, Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &v1face,
    Real &v2face, Real &v3face, Real &riemann_vel) {

  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);
  Real vpl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  Real vpr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  const Real sl = std::min(vpl - cl, vpr - cr);
  const Real sr = std::max(vpl + cl, vpr + cr);

  const Real rhocsl = rhol * (sl - vpl);
  const Real rhocsr = rhor * (sr - vpr);

  Real ss = (sl >= 0) * vpl + (sr <= 0) * vpr +
            (sl * sr < 0) * (rhocsl * vpl - rhocsr * vpr + Pr - Pl) / (rhocsl - rhocsr);
  const Real l_flag = 1.0 * (ss >= 0);
  const Real r_flag = 1.0 - l_flag;
  const Real rho = l_flag * rhol + r_flag * rhor;
  const Real s = l_flag * sl + r_flag * sr;
  const Real v1 = l_flag * v1l + r_flag * v1r;
  const Real v2 = l_flag * v2l + r_flag * v2r;
  const Real v3 = l_flag * v3l + r_flag * v3r;
  const Real vp = l_flag * vpl + r_flag * vpr;
  const Real u = l_flag * ul + r_flag * ur;
  const Real P = l_flag * Pl + r_flag * Pr;
  v1face = (DIR == X1DIR ? ss : v1);
  v2face = (DIR == X2DIR ? ss : v2);
  v3face = (DIR == X3DIR ? ss : v3);

  const Real rhos_rho = (s - vp) / (s - ss);
  const Real frho = rho * (vp + s * (rhos_rho - 1.0));
  riemann_vel = frho / rho;

  f_v1 = rho * (vp * v1 + s * (rhos_rho * v1face - v1));
  f_v2 = rho * (vp * v2 + s * (rhos_rho * v2face - v2));
  f_v3 = rho * (vp * v3 + s * (rhos_rho * v3face - v3));
  if constexpr (DIR == X1DIR) f_v1 += P;
  if constexpr (DIR == X2DIR) f_v2 += P;
  if constexpr (DIR == X3DIR) f_v3 += P;

  const Real ske = v1 * v1 + v2 * v2 + v3 * v3;
  const Real E = u + 0.5 * rho * ske;
  const Real Es = rhos_rho * (E + (ss - vp) * (rho * ss + P / (s - vp)));
  f_eng = vp * (E + P) + s * (Es - E);

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_lhllc_base
//! \brief
//! NOTE(@pdmullen): The below LHLLC solver is largely borrowed from Athena++, adapted for
//! multiple materials...
//! NOTE(JMM): I have used this as the base for two hllc-based solvers:
//!   chllc is carbuncle corrected hllc
//!   lhllc is low-Mach corrected and carbuncle corrected hllc
//! Relevant reference is Minoshima and Miyoshi, 2021
template <int DIR, bool enable_low_mach>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_lhllc_base(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, Real &f_v1, Real &f_v2, Real &f_v3,
    Real &f_eng, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel, Real &dvn,
    Real &dvt) {

  // Extract L/R states
  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);
  // Set normal components
  const Real vpl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  const Real vpr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  const Real sl = std::min(vpl - cl, vpr - cr);
  const Real sr = std::max(vpl + cl, vpr + cr);

  const Real el = ul / rhol;
  const Real er = ur / rhor;
  const Real vsql = SQR(v1l) + SQR(v2l) + SQR(v3l);
  const Real vsqr = SQR(v1r) + SQR(v2r) + SQR(v3r);

  // Shock detector
  const Real cmax = std::max(cl, cr);
  const Real rt6th =
      std::min(1.0, (cmax - std::min(dvn, 0.0)) / (cmax - std::min(dvt, 0.0)));
  Real th = SQR(rt6th * rt6th * rt6th);

  // Determine the contact wave speed and the pressure at the contact surface
  const Real rhocsl = rhol * (sl - vpl);
  const Real rhocsr = rhor * (sr - vpr);
  const Real irhocslr = 1.0 / (rhocsl - rhocsr);
  const Real am = irhocslr * (rhocsl * vpl - rhocsr * vpr + th * (Pr - Pl));
  // JMM: this low-Mach correction causes odd-even decoupling breaks symmetry
  Real phi = 1.0;
  if constexpr (enable_low_mach) {
    static constexpr Real SAFETY = 1e-4;
    const Real chi =
        std::max(SAFETY, std::min(1.0, std::sqrt(std::max(vsql, vsqr)) / cmax));
    phi = chi * (2.0 - chi);
  }
  const Real cp =
      irhocslr * (rhocsl * Pr - rhocsr * Pl - phi * rhocsr * rhocsl * (vpr - vpl));

  // Assign face velocities
  const Real ss = (sl >= 0.0) * vpl + (sr <= 0.0) * vpr + (sl * sr < 0.0) * am;
  const Real l_flag = 1.0 * (ss >= 0);
  const Real r_flag = 1.0 - l_flag;
  const Real v1 = l_flag * v1l + r_flag * v1r;
  const Real v2 = l_flag * v2l + r_flag * v2r;
  const Real v3 = l_flag * v3l + r_flag * v3r;
  v1face = v1;
  v2face = v2;
  v3face = v3;
  if constexpr (DIR == X1DIR) v1face = ss;
  if constexpr (DIR == X2DIR) v2face = ss;
  if constexpr (DIR == X3DIR) v3face = ss;

  // Compute L/R fluxes
  const Real bp = (sr > 0.0) * sr + (sr <= 0.0) * 1.0e-20;
  const Real bm = (sl < 0.0) * sl + (sl >= 0.0) * -1.0e-20;
  const Real flIDN = rhol * (vpl - bm);
  const Real frIDN = rhor * (vpr - bp);
  const Real flIEN = (el + 0.5 * vsql) * flIDN + Pl * vpl;
  const Real frIEN = (er + 0.5 * vsqr) * frIDN + Pr * vpr;
  Real flIVX = flIDN * v1l;
  Real flIVY = flIDN * v2l;
  Real flIVZ = flIDN * v3l;
  Real frIVX = frIDN * v1r;
  Real frIVY = frIDN * v2r;
  Real frIVZ = frIDN * v3r;
  if constexpr (DIR == X1DIR) flIVX += Pl;
  if constexpr (DIR == X2DIR) flIVY += Pl;
  if constexpr (DIR == X3DIR) flIVZ += Pl;
  if constexpr (DIR == X1DIR) frIVX += Pr;
  if constexpr (DIR == X2DIR) frIVY += Pr;
  if constexpr (DIR == X3DIR) frIVZ += Pr;

  // Compute HLLC flux weights
  const Real wl = (am >= 0.0) * am / (am - bm);
  const Real wr = (am < 0.0) * -am / (bp - am);
  const Real sm = (am >= 0.0) * (-bm / (am - bm)) + (am < 0.0) * (bp / (bp - am));

  // Compute the HLLC momentum and total energy fluxes
  f_v1 = wl * flIVX + wr * frIVX;
  f_v2 = wl * flIVY + wr * frIVY;
  f_v3 = wl * flIVZ + wr * frIVZ;
  f_eng = wl * flIEN + wr * frIEN + sm * cp * am;
  if constexpr (DIR == X1DIR) f_v1 += sm * cp;
  if constexpr (DIR == X2DIR) f_v2 += sm * cp;
  if constexpr (DIR == X3DIR) f_v3 += sm * cp;

  // Compute the HLLC bulk mass flux to assign the Riemann velocity
  const Real frho = wl * flIDN + wr * frIDN;
  const Real rho = l_flag * rhol + r_flag * rhor;
  riemann_vel = frho / rho;

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_lhllc
//! \brief
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_lhllc(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, Real &f_v1, Real &f_v2, Real &f_v3,
    Real &f_eng, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel, Real &dvn,
    Real &dvt) {
  return lr_to_flux_lhllc_base<DIR, true>(rhol, rhor, v1l, v1r, v2l, v2r, v3l, v3r, ul,
                                          ur, Pl, Pr, cl, cr, f_v1, f_v2, f_v3, f_eng,
                                          v1face, v2face, v3face, riemann_vel, dvn, dvt);
}
//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_chllc
//! \brief
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real lr_to_flux_chllc(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, Real &f_v1, Real &f_v2, Real &f_v3,
    Real &f_eng, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel, Real &dvn,
    Real &dvt) {
  return lr_to_flux_lhllc_base<DIR, false>(rhol, rhor, v1l, v1r, v2l, v2r, v3l, v3r, ul,
                                           ur, Pl, Pr, cl, cr, f_v1, f_v2, f_v3, f_eng,
                                           v1face, v2face, v3face, riemann_vel, dvn, dvt);
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_hll
//! \brief
template <int DIR>
KOKKOS_FORCEINLINE_FUNCTION Real
lr_to_flux_hll(Real rhol, Real rhor, Real v1l, Real v1r, Real v2l, Real v2r, Real v3l,
               Real v3r, const Real ul, const Real ur, const Real Pl, const Real Pr,
               const Real cl, const Real cr, Real &f_v1, Real &f_v2, Real &f_v3,
               Real &f_eng, Real &v1face, Real &v2face, Real &v3face, Real &riemann_vel) {

  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);
  Real vpl = (DIR == X1DIR) * v1l + (DIR == X2DIR) * v2l + (DIR == X3DIR) * v3l;
  Real vpr = (DIR == X1DIR) * v1r + (DIR == X2DIR) * v2r + (DIR == X3DIR) * v3r;

  const Real sl = std::min(vpl - cl, vpr - cr);
  const Real sr = std::max(vpl + cl, vpr + cr);

  const Real rhocsl = rhol * (sl - vpl);
  const Real rhocsr = rhor * (sr - vpr);

  const Real isrsl = 1.0 / (sr - sl);
  const Real frho = (sl * rhocsr - sr * rhocsl) * isrsl;
  const Real l_flag = 1.0 * (frho >= 0.0);
  const Real r_flag = 1.0 - l_flag;
  riemann_vel = frho / (l_flag * rhol + r_flag * rhor);
  // TODO(jcd): figure out the best velocities to use here
  //            these are not consistent with Uhll, but maybe that's OK
  v1face = (DIR == X1DIR ? riemann_vel : l_flag * v1l + r_flag * v1r);
  v2face = (DIR == X2DIR ? riemann_vel : l_flag * v2l + r_flag * v2r);
  v3face = (DIR == X3DIR ? riemann_vel : l_flag * v3l + r_flag * v3r);

  f_v1 = (sl * rhocsr * v1r - sr * rhocsl * v1l) * isrsl;
  f_v2 = (sl * rhocsr * v2r - sr * rhocsl * v2l) * isrsl;
  f_v3 = (sl * rhocsr * v3r - sr * rhocsl * v3l) * isrsl;
  if constexpr (DIR == X1DIR) f_v1 += (sr * Pl - sl * Pr) * isrsl;
  if constexpr (DIR == X2DIR) f_v2 += (sr * Pl - sl * Pr) * isrsl;
  if constexpr (DIR == X3DIR) f_v3 += (sr * Pl - sl * Pr) * isrsl;

  const Real El = ul + 0.5 * rhol * (SQR(v1l) + SQR(v2l) + SQR(v3l));
  const Real Er = ur + 0.5 * rhor * (SQR(v1r) + SQR(v2r) + SQR(v3r));
  f_eng = (sr * (El + Pl) * vpl - sl * (Er + Pr) * vpr + sr * sl * (Er - El)) * isrsl;

  return std::max(std::abs(sl), std::abs(sr));
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::lr_to_flux_strength
//! \brief
template <int DIR>
KOKKOS_INLINE_FUNCTION Real lr_to_flux_strength(
    Real rhol, Real rhor, const Real v1l, const Real v1r, const Real v2l, const Real v2r,
    const Real v3l, const Real v3r, const Real ul, const Real ur, const Real Pl,
    const Real Pr, const Real cl, const Real cr, const Real gmodl, const Real gmodr,
    const Real sxxl, const Real sxxr, const Real sxyl, const Real sxyr, const Real sxzl,
    const Real sxzr, const Real syyl, const Real syyr, const Real syzl, const Real syzr,
    Real &f_v1, Real &f_v2, Real &f_v3, Real &f_eng, Real &v1face, Real &v2face,
    Real &v3face, Real &riemann_vel) {
  // get szz assuming s is trace free
  const Real szzl = -sxxl - syyl;
  const Real szzr = -sxxr - syyr;

  // set normal components
  Real vpl, vpr, sddl, sddr;
  if constexpr (DIR == X1DIR) {
    vpl = v1l;
    vpr = v1r;
    sddl = sxxl;
    sddr = sxxr;
  }
  if constexpr (DIR == X2DIR) {
    vpl = v2l;
    vpr = v2r;
    sddl = syyl;
    sddr = syyr;
  }
  if constexpr (DIR == X3DIR) {
    vpl = v3l;
    vpr = v3r;
    sddl = szzl;
    sddr = szzr;
  }

  rhol = std::max(rhol, 1.e-100);
  rhor = std::max(rhor, 1.e-100);
  // const Real gl = std::sqrt(4.0/3.0*gmodl/rhol);
  // const Real gr = std::sqrt(4.0/3.0*gmodr/rhor);
  // Real gmax = std::max(gl,gr);
  // speed of transverse modes.  set to above when there's no support for shear stress
  // TODO(JCD): what happens when gmodl or gmodr > 0 but not both???
  // const Real tl = (gmodl > 0) ? std::sqrt(gmodl / rhol) : cl;
  // const Real tr = (gmodr > 0) ? std::sqrt(gmodr / rhor) : cr;
  // longitudinal signal speed
  const Real sl = std::min(vpl - cl, vpr - cr);
  const Real sr = std::max(vpl + cl, vpr + cr);
  // transverse signal speed
  // TODO(JCD): are these reasonable estimates?
  // const Real tsl = vpl - tl;
  // const Real tsr = vpr + tr;

  Real rhocsl = rhol * (sl - vpl);
  Real rhocsr = rhor * (sr - vpr);
  const Real irhocslr = 1.0 / (rhocsl - rhocsr);
  Real ss =
      (sl >= 0) * vpl + (sr <= 0) * vpr +
      (sl * sr < 0) * (rhocsl * vpl - rhocsr * vpr + Pr - sddr - Pl + sddl) * irhocslr;

  // flag to determine if we're inside the longitudinal waves
  const Real is_subsonic = 1.0 * (sl * sr < 0);
  // flag to determine if we should modify star states for strength
  const bool is_strong = (gmodl + gmodr > 0) * is_subsonic;

  const Real l_flag = 1.0 * (ss >= 0);
  const Real r_flag = 1.0 - l_flag;
  const Real rho = l_flag * rhol + r_flag * rhor;
  const Real s = l_flag * sl + r_flag * sr;
  const Real rcs = l_flag * rhocsl + r_flag * rhocsr;
  const Real v1 = l_flag * v1l + r_flag * v1r;
  const Real v2 = l_flag * v2l + r_flag * v2r;
  const Real v3 = l_flag * v3l + r_flag * v3r;
  const Real vp = l_flag * vpl + r_flag * vpr;
  const Real u = l_flag * ul + r_flag * ur;
  const Real P = l_flag * Pl + r_flag * Pr;
  const Real sxx = l_flag * sxxl + r_flag * sxxr;
  const Real sxy = l_flag * sxyl + r_flag * sxyr;
  const Real sxz = l_flag * sxzl + r_flag * sxzr;
  const Real syy = l_flag * syyl + r_flag * syyr;
  const Real syz = l_flag * syzl + r_flag * syzr;
  const Real szz = l_flag * szzl + r_flag * szzr;

  // In the supersonic case, vp == ss, so rhos/rho = 1
  const Real rhos_rho = (s - vp) / (s - ss);
  const Real frho = rho * (vp + s * (rhos_rho - 1.0));
  riemann_vel = frho / rho;

  const Real E = u + 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3);
  const Real ircs = 1.0 / rcs;
  Real sf1, sf2, sf3, sf1s, sf2s, sf3s;
  if constexpr (DIR == X1DIR) {
    sf1 = sxx - P;
    sf2 = sxy;
    sf3 = sxz;
    sf1s = (rhocsl * (sxxr - Pr) - rhocsr * (sxxl - Pl) + rhocsl * rhocsr * (v1r - v1l)) *
           irhocslr;
    sf2s = is_strong * (rhocsl * sxyr - rhocsr * sxyl + rhocsl * rhocsr * (v2r - v2l)) *
           irhocslr;
    sf3s = is_strong * (rhocsl * sxzr - rhocsr * sxzl + rhocsl * rhocsr * (v3r - v3l)) *
           irhocslr;
    v1face = ss;
    v2face = v2 + is_strong * (sxy - sf2s) * ircs;
    v3face = v3 + is_strong * (sxz - sf3s) * ircs;
  }
  if constexpr (DIR == X2DIR) {
    sf1 = sxy;
    sf2 = syy - P;
    sf3 = syz;
    sf1s = is_strong * (rhocsl * sxyr - rhocsr * sxyl + rhocsl * rhocsr * (v1r - v1l)) *
           irhocslr;
    sf2s = (rhocsl * (syyr - Pr) - rhocsr * (syyl - Pl) + rhocsl * rhocsr * (v2r - v2l)) *
           irhocslr;
    sf3s = is_strong * (rhocsl * syzr - rhocsr * syzl + rhocsl * rhocsr * (v3r - v3l)) *
           irhocslr;
    v1face = v1 + is_strong * (sxy - sf1s) * ircs;
    v2face = ss;
    v3face = v3 + is_strong * (syz - sf3s) * ircs;
  }
  if constexpr (DIR == X3DIR) {
    sf1 = sxz;
    sf2 = syz;
    sf3 = szz - P;
    sf1s = is_strong * (rhocsl * sxzr - rhocsr * sxzl + rhocsl * rhocsr * (v1r - v1l)) *
           irhocslr;
    sf2s = is_strong * (rhocsl * syzr - rhocsr * syzl + rhocsl * rhocsr * (v2r - v2l)) *
           irhocslr;
    sf3s = (rhocsl * (szzr - Pr) - rhocsr * (szzl - Pl) + rhocsl * rhocsr * (v3r - v3l)) *
           irhocslr;
    v1face = v1 + is_strong * (sxz - sf1s) * ircs;
    v2face = v2 + is_strong * (syz - sf2s) * ircs;
    v3face = ss;
  }
  const Real se = sf1 * v1 + sf2 * v2 + sf3 * v3;
  const Real ses = sf1s * v1face + sf2s * v2face + sf3s * v3face;
  const Real Es = rhos_rho * (E + is_subsonic * rho * (se - ses) * ircs);

  f_v1 = rho * (vp * v1 + s * (rhos_rho * v1face - v1)) - sf1;
  f_v2 = rho * (vp * v2 + s * (rhos_rho * v2face - v2)) - sf2;
  f_v3 = rho * (vp * v3 + s * (rhos_rho * v3face - v3)) - sf3;
  f_eng = vp * E - se + s * (Es - E);

  return std::max(std::abs(sl), std::abs(sr));
}

} // namespace Hydro

#endif // HYDRO_RIEMANN_HPP_
