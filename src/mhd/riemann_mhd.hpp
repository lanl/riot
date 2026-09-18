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
    Real &f_b2, Real &f_b3, Real &v1face, Real &v2face, Real &v3face,
    Real &riemann_vel) {

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

} // namespace MHD

#endif // MHD_RIEMANN_MHD_HPP_
