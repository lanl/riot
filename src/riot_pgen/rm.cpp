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

#include <singularity-eos/eos/eos.hpp>

#include "rm.hpp"

namespace rm {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void rm::SetPostShock
//! \brief
void SetPostShock(const Real rho0, const Real P0, const Real gam, const Real pratio,
                  Real &rhos, Real &Ps, Real &vs) {
  const Real c0 = std::sqrt(gam * P0 / rho0);
  const Real s0 = 1.0 / rho0;
  Ps = P0 * pratio;
  const Real ss =
      s0 * ((gam + 1.0) * P0 + (gam - 1.0) * Ps) / ((gam - 1.0) * P0 + (gam + 1.0) * Ps);
  rhos = 1.0 / ss;
  const Real us = c0 * std::sqrt(1.0 + (gam + 1.0) / (2.0 * gam) * (pratio - 1.0));
  const Real u2 = -us * rho0 / rhos;
  vs = u2 + us;
}

//----------------------------------------------------------------------------------------
//! \fn  void rm::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  Interface surface(pin);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  const int ni = ib.e - ib.s + 1;
  const int nj = jb.e - jb.s + 1;
  const int nk = kb.e - kb.s + 1;
  const int ntot = ni * nj * nk;

  ParArray3D<Real> vfrac("vfrac", nk, nj, ni);
  auto &coords = pmb->coords;
  Real total_vfrac = 0;
  pmb->par_reduce(
      "ProblemGenerator::rm::vfrac", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &vsum) {
        vfrac(k, j, i) = surface.Vfrac(coords.Xf<X1DIR>(i), coords.Xf<X1DIR>(i + 1),
                                       coords.Xf<X2DIR>(j), coords.Xf<X2DIR>(j + 1),
                                       coords.Xf<X3DIR>(k), coords.Xf<X3DIR>(k + 1));
        vsum += vfrac(k, j, i);
      },
      Kokkos::Sum<Real>(total_vfrac));

  bool include_left_mat = total_vfrac > 0.0;
  bool include_right_mat = 1.0 * ntot - total_vfrac > 1.e-15;

  auto &rc = pmb->meshblock_data.Get();
  auto &rho_left = rc->Get(ccmat::rho::name(), 0);
  if (include_left_mat) {
    if (!rho_left.IsAllocated()) pmb->AllocateSparse(rho_left.label());
  } else {
    if (rho_left.IsAllocated()) pmb->DeallocateSparse(rho_left.label());
  }
  auto &rho_right = rc->Get(ccmat::rho::name(), 1);
  if (include_right_mat) {
    if (!rho_right.IsAllocated()) pmb->AllocateSparse(rho_right.label());
  } else {
    if (rho_right.IsAllocated()) pmb->DeallocateSparse(rho_right.label());
  }

  // now get the pack
  auto resolved_pkgs = pmb->resolved_packages;
  static auto desc =
      riot::MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                               ccbulk::total_material_energy, ccbulk::momentum>(
          resolved_pkgs.get(), {0, 1});
  auto v = riot::GetPack(desc, rc.get());

  // Problem Parameters
  const Real Tl = pin->GetOrAddReal("rm", "T_l", 300.0);
  const Real Pl = pin->GetOrAddReal("rm", "P_l", 1.0e6);
  const Real gammal = pin->GetReal("material0", "Gamma");
  const Real cvl = pin->GetReal("material0", "Cv");
  const Real rhol = Pl / (Tl * cvl * (gammal - 1.0));
  const Real Tr = pin->GetOrAddReal("rm", "T_r", 300.0);
  const Real Pr = pin->GetOrAddReal("rm", "P_r", 1.0e6);
  const Real gammar = pin->GetReal("material1", "Gamma");
  const Real cvr = pin->GetReal("material1", "Cv");
  const Real rhor = Pr / (Tr * cvr * (gammar - 1.0));
  const Real shock_x = pin->GetOrAddReal("rm", "shock_x", 0.1);

  const Real pressure_jump = pin->GetOrAddReal("rm", "pressure_jump", 2.0);
  Real rho_shock, Pshock, vshock;
  SetPostShock(rhol, Pl, gammal, pressure_jump, rho_shock, Pshock, vshock);
  const Real Tshock = Pshock / (rho_shock * cvl * (gammal - 1.0));

  pmb->par_for(
      "ProblemGenerator::kh", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<X1DIR>(i);
        v(0, ccbulk::total_material_energy(), k, j, i) = 0.0;
        if (include_left_mat) {
          v(0, ccmat::rho(0), k, j, i) = x < shock_x ? rho_shock : vfrac(k, j, i) * rhol;
          v(0, ccmat::volume_fraction(0), k, j, i) = vfrac(k, j, i);
          v(0, ccbulk::total_material_energy(), k, j, i) =
              x < shock_x ? rho_shock * (cvl * Tshock + 0.5 * vshock * vshock)
                          : vfrac(k, j, i) * rhol * cvl * Tl;
        }
        if (include_right_mat) {
          v(0, ccmat::rho(include_left_mat), k, j, i) = (1.0 - vfrac(k, j, i)) * rhor;
          v(0, ccmat::volume_fraction(include_left_mat), k, j, i) = 1.0 - vfrac(k, j, i);
          v(0, ccbulk::total_material_energy(), k, j, i) +=
              (1.0 - vfrac(k, j, i)) * rhor * cvr * Tr;
        }
        v(0, ccbulk::momentum(0), k, j, i) = x < shock_x ? rho_shock * vshock : 0.0;
        v(0, ccbulk::momentum(1), k, j, i) = 0.0;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;
      });

  return;
}

} // namespace rm