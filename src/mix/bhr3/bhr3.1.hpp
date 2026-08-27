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
#ifndef BHR3_BHR_3_1_HPP_
#define BHR3_BHR_3_1_HPP_
// This file was made in part with generative AI.

#include <cassert>
#include <cmath>

namespace MixModel {

using namespace parthenon::package::prelude;

class BHR3_1 {
 public:
  //--------------------------------------------------------------------------------------
  //! \fn  MixModel::BHR3_1::BHR3_1
  //! \brief
  KOKKOS_FUNCTION BHR3_1() {}

  //--------------------------------------------------------------------------------------
  //! \fn  MixModel::BHR3_1::BHR3_1
  //! \brief
  KOKKOS_FUNCTION
  BHR3_1(const Real c_1, const Real c_2, const Real c_3, const Real c_4, const Real c_a1,
         const Real c_a2, const Real c_a3, const Real c_b2, const Real c_r1,
         const Real c_r2, const Real c_r4, const Real c_ap, const Real c_ar,
         const Real c_au, const Real sigma_c, const Real sigma_a, const Real sigma_b,
         const Real sigma_k, const Real sigma_visc, const Real sigma_epsilon,
         const Real c_1v, const Real c_2v, const Real c_3v, const Real c_4v,
         const Real c_mu, const Real tke_0, const Real S0)
      : c_1_{c_1}, c_2_{c_2}, c_3_{c_3}, c_4_{c_4}, c_a1_{c_a1}, c_a2_{c_a2}, c_a3_{c_a3},
        c_b2_{c_b2}, c_r1_{c_r1}, c_r2_{c_r2}, c_r4_{c_r4}, c_ap_{c_ap}, c_ar_{c_ar},
        c_au_{c_au}, sigma_c_{sigma_c}, sigma_a_{sigma_a}, sigma_b_{sigma_b},
        sigma_k_{sigma_k}, sigma_visc_{sigma_visc}, sigma_epsilon_{sigma_epsilon},
        c_1v_{c_1v}, c_2v_{c_2v}, c_3v_{c_3v}, c_4v_{c_4v}, c_mu_{c_mu}, tke_0_{tke_0},
        S0_{S0}, sigma_min_{std::min(
                     {sigma_c, sigma_a, sigma_b, sigma_k, sigma_visc, sigma_epsilon})},
        c_max_{std::max({c_a1, c_b2, c_3, c_3v, c_r4})} {}

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::mut
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real mut(const Real rho, const Real ST, const Real sqrtk) {
    return c_mu_ * rho * ST * sqrtk;
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::ST_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real ST_source(const Real ST, const Real K, const Real sqrtk, const Real rho,
                 const Real rho_R_grad_u, const Real a_dot_grad_p, const Real div_u) {
    // NOTE(@chadmeyer): This could be overwritten by initializing
    // this object with tke_0 = 0)
    return -ST / std::max(K, tke_0_) *
               ((1.5 - c_1_) * rho_R_grad_u - (1.5 - c_4_) * a_dot_grad_p) -
           (1.5 - c_2_) * rho * sqrtk - c_3_ * rho * ST * div_u;
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::SD_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real SD_source(const Real SD, const Real K, const Real sqrtk, const Real rho,
                 const Real rho_R_grad_u, const Real a_dot_grad_p, const Real div_u) {
    return -SD / std::max(K, tke_0_) *
               ((1.5 - c_1v_) * rho_R_grad_u - (1.5 - c_4v_) * a_dot_grad_p) -
           (1.5 - c_2v_) * rho * sqrtk - c_3v_ * rho * SD * div_u;
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::tke
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real tke(const Real Rxx, const Real Ryy, const Real Rzz) {
    assert(Rxx + Ryy + Rzz >= 0.0);
    return 0.5 * (Rxx + Ryy + Rzz);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::rho_R_grad_u
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real rho_R_grad_u(const Real rho, const Real Rxx, const Real Ryy, const Real Rzz,
                    const Real Rxy, const Real Rxz, const Real Ryz, const Real ux_x,
                    const Real ux_y, const Real ux_z, const Real uy_x, const Real uy_y,
                    const Real uy_z, const Real uz_x, const Real uz_y, const Real uz_z) {
    return rho_R_grad_u_3d(rho, Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, ux_x, ux_y, ux_z, uy_x,
                           uy_y, uy_z, uz_x, uz_y, uz_z);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::b_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real b_source(const Real b, const Real b_x, const Real b_y, const Real b_z,
                const Real ax, const Real ay, const Real az, const Real rho,
                const Real rho_x, const Real rho_y, const Real rho_z, const Real SD,
                const Real sqrtk) {
    return b_source_3d(b, b_x, b_y, b_z, ax, ay, az, rho, rho_x, rho_y, rho_z, SD, sqrtk);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::ai_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real ai_source(const Real ai, const Real P_i, const Real rho, const Real sqrtk,
                 const Real SD, const Real b, const Real div_aa, const Real ax,
                 const Real ay, const Real az, const Real Rix, const Real Riy,
                 const Real Riz, const Real rho_x, const Real rho_y, const Real rho_z,
                 const Real ui_x, const Real ui_y, const Real ui_z, const Real div_a) {
    return ai_source_3d(ai, P_i, rho, sqrtk, SD, b, div_aa, ax, ay, az, Rix, Riy, Riz,
                        rho_x, rho_y, rho_z, ui_x, ui_y, ui_z, div_a);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::Rii_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real Rii_source(const Real Rii, const Real ai, const Real P_i, const Real rho,
                  const Real k, const Real sqrtk, const Real SD, const Real a_dot_grad_p,
                  const Real rho_R_grad_u, const Real Rix, const Real Riy, const Real Riz,
                  const Real ui_x, const Real ui_y, const Real ui_z) {
    return Rii_source_3d(Rii, ai, P_i, rho, k, sqrtk, SD, a_dot_grad_p, rho_R_grad_u, Rix,
                         Riy, Riz, ui_x, ui_y, ui_z);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::Rij_source
  //! \brief
  KOKKOS_INLINE_FUNCTION
  Real Rij_source(const Real Rij, const Real ai, const Real aj, const Real P_i,
                  const Real P_j, const Real rho, const Real sqrtk, const Real SD,
                  const Real Rix, const Real Riy, const Real Riz, const Real Rjx,
                  const Real Rjy, const Real Rjz, const Real ui_x, const Real ui_y,
                  const Real ui_z, const Real uj_x, const Real uj_y, const Real uj_z) {
    return Rij_source_3d(Rij, ai, aj, P_i, P_j, rho, sqrtk, SD, Rix, Riy, Riz, Rjx, Rjy,
                         Rjz, ui_x, ui_y, ui_z, uj_x, uj_y, uj_z);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_d
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_d() { return sigma_visc_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_t
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_t() { return sigma_epsilon_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_b
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_b() { return sigma_b_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_a
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_a() { return sigma_a_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_k
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_k() { return sigma_k_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_c
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_c() { return sigma_c_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::sigma_min
  //! \brief
  KOKKOS_INLINE_FUNCTION Real sigma_min() { return sigma_min_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::c_max
  //! \brief
  // NOTE(): The following is the most restrective of the homogeneous source term
  //         coefficients
  KOKKOS_INLINE_FUNCTION Real c_max() { return c_max_; }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::tke_0
  //! \brief
  KOKKOS_INLINE_FUNCTION Real tke_0() { return tke_0_; }

 private:
  Real c_1_, c_2_, c_3_, c_4_, c_a1_, c_a2_, c_a3_, c_b2_, c_r1_, c_r2_, c_r4_, c_ap_,
      c_ar_, c_au_, sigma_c_, sigma_a_, sigma_b_, sigma_k_, sigma_visc_, sigma_epsilon_,
      c_1v_, c_2v_, c_3v_, c_4v_, c_mu_, tke_0_, S0_;
  Real sigma_min_, c_max_;

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::rho_R_grad_u_3d
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real rho_R_grad_u_3d(const Real rho, const Real Rxx, const Real Ryy, const Real Rzz,
                       const Real Rxy, const Real Rxz, const Real Ryz, const Real ux_x,
                       const Real ux_y, const Real ux_z, const Real uy_x, const Real uy_y,
                       const Real uy_z, const Real uz_x, const Real uz_y,
                       const Real uz_z) {
    return rho * (Rxx * ux_x + Ryy * uy_y + Rzz * uz_z + Rxy * (ux_y + uy_x) +
                  Rxz * (ux_z + uz_x) + Ryz * (uy_z + uz_y));
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::a_dot_grad_P_3d
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real a_dot_grad_P_3d(const Real ax, const Real ay, const Real az, const Real P_x,
                       const Real P_y, const Real P_z) {
    return ax * P_x + ay * P_y + az * P_z;
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::div_u_3d
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real div_u_3d(const Real ux_x, const Real uy_y, const Real uz_z) {
    return ux_x + uy_y + uz_z;
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::b_source_3d
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real b_source_3d(const Real b, const Real b_x, const Real b_y, const Real b_z,
                   const Real ax, const Real ay, const Real az, const Real rho,
                   const Real rho_x, const Real rho_y, const Real rho_z, const Real SD,
                   const Real sqrtk) {
    return -2.0 * (b + 1.0) * (ax * rho_x + ay * rho_y + az * rho_z) +
           2.0 * rho * (ax * b_x + ay * b_y + az * b_z) -
           c_b2_ * rho * b * sqrtk * (SD > 0.0 ? 1.0 / SD : 0.0);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::ai_source_3d
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real ai_source_3d(const Real ai, const Real P_i, const Real rho, const Real sqrtk,
                    const Real SD, const Real b, const Real div_aa, const Real ax,
                    const Real ay, const Real az, const Real Rix, const Real Riy,
                    const Real Riz, const Real rho_x, const Real rho_y, const Real rho_z,
                    const Real ui_x, const Real ui_y, const Real ui_z, const Real div_a) {
    // NOTE(@chadmeyer): The 3rd term, final parenthesis, should be equivalent to
    // a_k ubar_i;k = a_k (u_i;k - a_i;k) = a_k u_i;k - a_k a_i;k =
    //                                      a_k u_i;k - (a_i a_k)_;k + a_i*a_k;k
    return b * P_i * (1.0 - c_ap_) - (Rix * rho_x + Riy * rho_y + Riz * rho_z) +
           (c_au_ - 1.0) * rho *
               (ax * ui_x + ay * ui_y + az * ui_z - div_aa + ai * div_a) +
           c_a3_ * rho * div_aa - c_a1_ * rho * sqrtk * ai * (SD > 0.0 ? 1.0 / SD : 0.0);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::Rij_source_3d
  //! \brief Off diagonal components
  KOKKOS_FORCEINLINE_FUNCTION
  Real Rij_source_3d(const Real Rij, const Real ai, const Real aj, const Real P_i,
                     const Real P_j, const Real rho, const Real sqrtk, const Real SD,
                     const Real Rix, const Real Riy, const Real Riz, const Real Rjx,
                     const Real Rjy, const Real Rjz, const Real ui_x, const Real ui_y,
                     const Real ui_z, const Real uj_x, const Real uj_y, const Real uj_z) {
    return (1.0 - c_r1_) * (ai * P_j + aj * P_i) +
           rho * (c_r2_ - 1.0) *
               (Rix * uj_x + Riy * uj_y + Riz * uj_z + Rjx * ui_x + Rjy * ui_y +
                Rjz * ui_z) -
           c_r4_ * rho * sqrtk * Rij * (SD > 0.0 ? 1.0 / SD : 0.0);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  Real MixModel::BHR3_1::Rii_source_3d
  //! \brief Diagonal components
  KOKKOS_FORCEINLINE_FUNCTION
  Real Rii_source_3d(const Real Rii, const Real ai, const Real P_i, const Real rho,
                     const Real k, const Real sqrtk, const Real SD,
                     const Real a_dot_grad_p, const Real rho_R_grad_u, const Real Rix,
                     const Real Riy, const Real Riz, const Real ui_x, const Real ui_y,
                     const Real ui_z) {
    Real ssterms = (1.0 - c_r1_) * 2.0 * ai * P_i +
                   rho * (c_r2_ - 1.0) * 2.0 * (Rix * ui_x + Riy * ui_y + Riz * ui_z) +
                   2.0 / 3.0 * (c_r1_ * a_dot_grad_p - c_r2_ * rho_R_grad_u);
    // NOTE(@chadmeyer): If Rii is bigger than 1e-4 of its initial
    // value, use the source as written, else scale it
    if (ssterms < 0) ssterms *= Rii / std::max(Rii, 1.0e-4 * 2.0 / 3.0 * tke_0_ * rho);
    return ssterms + (2.0 / 3.0 * (c_r4_ - 1.0) * k - c_r4_ * Rii) * rho * sqrtk *
                         (SD > 0.0 ? 1.0 / SD : 0.0);
    // return (1.0 - c_r1_) * 2.0 * ai * P_i +
    //        rho * (c_r2_ - 1.0) * 2.0 * (Rix * ui_x + Riy * ui_y + Riz * ui_z) +
    //        2.0 / 3.0 *
    //            (c_r1_ * a_dot_grad_p - c_r2_ * rho_R_grad_u +
    //             (c_r4_ - 1.0) * rho * k * sqrtk * (SD > 0.0 ? 1.0 / SD : 0.0)) -
    //        c_r4_ * rho * sqrtk * Rii * (SD > 0.0 ? 1.0 / SD : 0.0);
  }
};

} // namespace MixModel

#endif // BHR3_BHR_3_1_HPP_
