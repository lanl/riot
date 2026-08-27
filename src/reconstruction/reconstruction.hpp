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
#ifndef RECONSTRUCTION_RECONSTRUCTION_HPP_
#define RECONSTRUCTION_RECONSTRUCTION_HPP_
// This file was made in part with generative AI.

#include <algorithm>
#include <string>
#include <unordered_map>

#include <basic_types.hpp>
#include <kokkos_abstraction.hpp>

#include "riot_utils/riot_utils.hpp"

namespace RiotReconstruction {

enum class Type { CONSTANT, PLM, PPM4, WENO5, MP5 };
// JMM: Would be more efficient to use an array, but this is more
// convenient and it doesn't matter. This is only used at
// initialization.
const std::unordered_map<Type, int> STENCIL_WIDTH = {{Type::CONSTANT, 0},
                                                     {Type::PLM, 1},
                                                     {Type::PPM4, 2},
                                                     {Type::WENO5, 2},
                                                     {Type::MP5, 2}};
// Note these must be explicitly upper-case
const std::unordered_map<std::string, Type> NAME_MAP = {{"CONSTANT", Type::CONSTANT},
                                                        {"PLM", Type::PLM},
                                                        {"PPM4", Type::PPM4},
                                                        {"WENO5", Type::WENO5},
                                                        {"MP5", Type::MP5}};

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::mc
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm / dp : 0.0);
  return std::max(0.0, std::min(2.0, std::min(2 * r, 0.5 * (1 + r))));
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::minmod
//! \brief
KOKKOS_INLINE_FUNCTION
Real minmod(const Real a, const Real b) {
  // return (a * b > 0.) ? std::min(std::abs(a), std::abs(b)) * std::copysign(1., a) : 0.;
  return (a * b > 0.) * std::min(std::abs(a), std::abs(b)) * std::copysign(1., a);
};

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::ltanh
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real ltanh(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.0) return 2.0 * std::tanh(dm / dp);
  return 2.0;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::barth_jespersen
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real barth_jespersen(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm / dp : 3.0);
  return std::min(std::min(1.0, 4.0 / (1.0 + r)), 4.0 * r / (1.0 + r));
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::overbee_alpha
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real overbee_alpha(const Real dm, const Real dp, const Real alpha) {
  if (alpha < 1.0 || alpha > 2.0) {
    PARTHENON_FAIL("alpha not between 1 and 2");
  }
  Real dc = std::abs(0.5 * (dp - dm));
  Real dcent = std::min(dc, 2.0 * std::min(std::abs(dm), std::abs(dp)));
  Real dmin = alpha * std::min(std::abs(dm), std::abs(dp));
  dmin = std::max(dcent, dmin);
  if (std::abs(dp) > 0.) return dmin / std::abs(dp);
  return alpha;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::overbee
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real overbee(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.) return 2.0 * std::min(std::abs(dm), std::abs(dp)) / std::abs(dp);
  return 2.0;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::superbee
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real superbee(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm / dp : 2.0);
  // superbee
  return std::max(std::max(0.0, std::min(2 * r, 1.0)), std::min(r, 2.0));
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::nolim
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real nolim(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.) return 0.5 * (dp - dm) / dp;
  return 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::phifunc
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real phifunc(const Real mind, const Real maxd, const Real gx, const Real gy,
             const Real dx, const Real dy) {
  Real delta = gx * dx + gy * dy;
  if (delta > 0.) {
    return std::min(1.0, maxd / delta);
  } else if (delta < 0.) {
    return std::min(1.0, mind / delta);
  }
  return 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::mc
//! \brief
#pragma omp declare simd
KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp, const Real alpha) {
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return std::copysign(
      std::min(std::fabs(dc), alpha * std::min(std::fabs(dm), std::fabs(dp))), dc);
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotReconstruction::PiecewiseLinearSlope
//! \brief
KOKKOS_INLINE_FUNCTION
Real PiecewiseLinearSlope(const Real qm, const Real q0, const Real qp,
                          const Real slope_limit = 1.99) {
  Real dq = qp - q0;
  // const Real slope_limit = 0.99 / ((wgt <= 0.5)*wgt + (wgt > 0.5)*(1.0-wgt));
  return 0.5 * mc(q0 - qm, dq, slope_limit);
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::PiecewiseLinear
//! \brief
KOKKOS_INLINE_FUNCTION
void PiecewiseLinear(const Real qm, const Real q0, const Real qp, Real &l, Real &r,
                     const Real slope_limit = 1.99) {
  Real dq = qp - q0;
  // const Real slope_limit = 0.99 / ((wgt <= 0.5)*wgt + (wgt > 0.5)*(1.0-wgt));
  dq = 0.5 * mc(q0 - qm, dq, slope_limit);
  l = q0 + dq; // wgt *dq;
  r = q0 - dq; //(1.0 - wgt) * dq;
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::PiecewiseLinearFixedSlope
//! \brief
// NOTE(): Default arguments confuse template resolution
KOKKOS_INLINE_FUNCTION
void PiecewiseLinearFixedSlope(const Real qm, const Real q0, const Real qp, Real &l,
                               Real &r) {
  return PiecewiseLinear(qm, q0, qp, l, r, 1.99);
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::PiecewiseConstant
//! \brief
KOKKOS_INLINE_FUNCTION
void PiecewiseConstant(const Real q0, Real &l, Real &r) {
  l = q0;
  r = q0;
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::PPM4
//! \brief
KOKKOS_INLINE_FUNCTION
void PPM4(const Real q_im2, const Real q_im1, const Real q_i, const Real q_ip1,
          const Real q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Interpolate L/R values (CS eqn 16, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7. * (q_i + q_im1) - (q_im2 + q_ip1)) / 12.0;
  Real qrv = (7. * (q_i + q_ip1) - (q_im1 + q_ip2)) / 12.0;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = std::max(qlv, std::min(q_i, q_im1));
  qlv = std::min(qlv, std::max(q_i, q_im1));
  qrv = std::max(qrv, std::min(q_i, q_ip1));
  qrv = std::min(qrv, std::max(q_i, q_ip1));

  //--- monotonize interpolated L/R states (CS eqns 14, 15) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  /*
  if ((qc*qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (fabs(qc) >= 2.0*fabs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (fabs(qd) >= 2.0*fabs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }
  */
  bool same_sign = (qc * qd) >= 0.0;
  bool qc_big = std::abs(qc) >= 2.0 * std::abs(qd);
  bool qd_big = std::abs(qd) >= 2.0 * std::abs(qc);
  qlv = (same_sign)*q_i + (!same_sign) * ((qd_big) * (q_i - 2.0 * qc) + (!qd_big) * qlv);
  qrv = (same_sign)*q_i + (!same_sign) * ((qc_big) * (q_i - 2.0 * qd) + (!qc_big) * qrv);

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i = qlv;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::WENO5Z
//! \brief
#pragma omp declare simd
KOKKOS_INLINE_FUNCTION void WENO5Z(const Real q0, const Real q1, const Real q2,
                                   const Real q3, const Real q4, Real &ql, Real &qr) {
  constexpr Real w5alpha[3][3] = {{1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0},
                                  {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
                                  {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  constexpr Real eps = 1e-100;
  constexpr Real thirteen_thirds = 13.0 / 3.0;

  Real a = q0 - 2 * q1 + q2;
  Real b = q0 - 4.0 * q1 + 3.0 * q2;
  Real beta0 = thirteen_thirds * a * a + b * b + eps;
  a = q1 - 2.0 * q2 + q3;
  b = q3 - q1;
  Real beta1 = thirteen_thirds * a * a + b * b + eps;
  a = q2 - 2.0 * q3 + q4;
  b = q4 - 4.0 * q3 + 3.0 * q2;
  Real beta2 = thirteen_thirds * a * a + b * b + eps;
  const Real tau5 = std::fabs(beta2 - beta0);

  beta0 = (beta0 + tau5) / beta0;
  beta1 = (beta1 + tau5) / beta1;
  beta2 = (beta2 + tau5) / beta2;

  Real w0 = w5gamma[0] * beta0 + eps;
  Real w1 = w5gamma[1] * beta1 + eps;
  Real w2 = w5gamma[2] * beta2 + eps;
  Real wsum = 1.0 / (w0 + w1 + w2);
  ql = w0 * (w5alpha[0][0] * q0 + w5alpha[0][1] * q1 + w5alpha[0][2] * q2);
  ql += w1 * (w5alpha[1][0] * q1 + w5alpha[1][1] * q2 + w5alpha[1][2] * q3);
  ql += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q3 + w5alpha[2][2] * q4);
  ql *= wsum;
  const Real alpha_l =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  w0 = w5gamma[0] * beta2 + eps;
  w1 = w5gamma[1] * beta1 + eps;
  w2 = w5gamma[2] * beta0 + eps;
  wsum = 1.0 / (w0 + w1 + w2);
  qr = w0 * (w5alpha[0][0] * q4 + w5alpha[0][1] * q3 + w5alpha[0][2] * q2);
  qr += w1 * (w5alpha[1][0] * q3 + w5alpha[1][1] * q2 + w5alpha[1][2] * q1);
  qr += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q1 + w5alpha[2][2] * q0);
  qr *= wsum;
  const Real alpha_r =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  Real dq = q3 - q2;
  dq = mc(q2 - q1, dq, 2.0);

  const Real alpha_lin = 2.0 * alpha_l * alpha_r / (alpha_l + alpha_r);
  ql = alpha_lin * ql + (1.0 - alpha_lin) * (q2 + 0.5 * dq);
  qr = alpha_lin * qr + (1.0 - alpha_lin) * (q2 - 0.5 * dq);
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::WENO5
//! \brief
#pragma omp declare simd
KOKKOS_FORCEINLINE_FUNCTION void WENO5(const Real q0, const Real q1, const Real q2,
                                       const Real q3, const Real q4, Real &ql, Real &qr) {
  WENO5Z(q0, q1, q2, q3, q4, ql, qr);
}

//----------------------------------------------------------------------------------------
//! \fn  double RiotReconstruction::Median
//! \brief
#define MINMOD(a, b) ((a) * (b) > 0.0 ? (fabs(a) < fabs(b) ? (a) : (b)) : 0.0)
KOKKOS_INLINE_FUNCTION
double Median(double a, double b, double c) { return (a + MINMOD(b - a, c - a)); }

//----------------------------------------------------------------------------------------
//! \fn  double RiotReconstruction::mp5_subcalc
//! \brief
// NOTE(): Lifted shamelessly from nubhlight, which was lifted shamelessly from PLUTO
#pragma omp declare simd
KOKKOS_INLINE_FUNCTION
double mp5_subcalc(double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2) {
  double f, d2, d2p, d2m;
  double dMMm, dMMp;
  double scrh1, scrh2, Fmin, Fmax;
  double fAV, fMD, fLC, fUL, fMP;
  constexpr double alpha = 4.0, epsm = 1.e-12;

  f = 2.0 * Fjm2 - 13.0 * Fjm1 + 47.0 * Fj + 27.0 * Fjp1 - 3.0 * Fjp2;
  f /= 60.0;

  fMP = Fj + MINMOD(Fjp1 - Fj, alpha * (Fj - Fjm1));

  if ((f - Fj) * (f - fMP) <= epsm) return f;

  d2m = Fjm2 + Fj - 2.0 * Fjm1; // Eqn. 2.19
  d2 = Fjm1 + Fjp1 - 2.0 * Fj;
  d2p = Fj + Fjp2 - 2.0 * Fjp1; // Eqn. 2.19

  scrh1 = MINMOD(4.0 * d2 - d2p, 4.0 * d2p - d2);
  scrh2 = MINMOD(d2, d2p);
  dMMp = MINMOD(scrh1, scrh2); // Eqn. 2.27
  scrh1 = MINMOD(4.0 * d2m - d2, 4.0 * d2 - d2m);
  scrh2 = MINMOD(d2, d2m);
  dMMm = MINMOD(scrh1, scrh2); // Eqn. 2.27

  fUL = Fj + alpha * (Fj - Fjm1);                   // Eqn. 2.8
  fAV = 0.5 * (Fj + Fjp1);                          // Eqn. 2.16
  fMD = fAV - 0.5 * dMMp;                           // Eqn. 2.28
  fLC = 0.5 * (3.0 * Fj - Fjm1) + 4.0 / 3.0 * dMMm; // Eqn. 2.29

  scrh1 = fmin(Fj, Fjp1);
  scrh1 = fmin(scrh1, fMD);
  scrh2 = fmin(Fj, fUL);
  scrh2 = fmin(scrh2, fLC);
  Fmin = fmax(scrh1, scrh2); // Eqn. (2.24a)

  scrh1 = fmax(Fj, Fjp1);
  scrh1 = fmax(scrh1, fMD);
  scrh2 = fmax(Fj, fUL);
  scrh2 = fmax(scrh2, fLC);
  Fmax = fmin(scrh1, scrh2); // Eqn. 2.24b

  f = Median(f, Fmin, Fmax); // Eqn. 2.26
  return f;
}
#undef MINMOD

//----------------------------------------------------------------------------------------
//! \fn  void RiotReconstruction::MP5
//! \brief
#pragma omp declare simd
KOKKOS_INLINE_FUNCTION
void MP5(const Real q0, const Real q1, const Real q2, const Real q3, const Real q4,
         Real &ql, Real &qr) {
  ql = mp5_subcalc(q0, q1, q2, q3, q4);
  qr = mp5_subcalc(q4, q3, q2, q1, q0);
}

//----------------------------------------------------------------------------------------
//! \fn  Real ComputeMCSlope
//  \brief Compute MC-limited slope in one direction
KOKKOS_INLINE_FUNCTION
Real ComputeMCSlope(const Real u_im1, const Real u_i, const Real u_ip1,
                    const Real dx_minus, const Real dx_plus) {
  const Real slope_central = (u_ip1 - u_im1) / (dx_minus + dx_plus);
  const Real slope_forward = (u_ip1 - u_i) / dx_plus;
  const Real slope_backward = (u_i - u_im1) / dx_minus;
  return minmod(minmod(slope_forward, slope_backward), slope_central);
}

} // namespace RiotReconstruction

#endif // RECONSTRUCTION_RECONSTRUCTION_HPP_
