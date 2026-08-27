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
// This file was made in part with generative AI.

#include <cmath>
#include <cstdio>

#include "riot_pgen/pgen.hpp"
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

namespace gresho {
using parthenon::ParArray1D;
using namespace RiotEOS;

// Gresho vortex a la Miczek 2013.  Input Mach number allows probe of low Mach flows

//----------------------------------------------------------------------------------------
//! \fn  void gresho::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim == 2, "Gresho problem only works in 2D.");

  auto &rc = pmb->meshblock_data.Get();
  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc = MakePackDescriptor<ccmat::rho, ccmat::volume_fraction,
                                        ccbulk::total_material_energy, ccbulk::momentum>(
      (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  PARTHENON_REQUIRE(v.GetLowerBoundHost(0, ccmat::rho()) ==
                        v.GetUpperBoundHost(0, ccmat::rho()),
                    "Gresho vortex is only for one mat");

  const Real Ma = pin->GetOrAddReal("gresho", "Ma", 0.1);
  auto mat_pkg = pmb->packages.Get("materials");
  auto eos_host = mat_pkg->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  Real gm1 = eos_host[0].GruneisenParamFromDensityTemperature(1., 1.);
  Real gamma = gm1 + 1.0;
  const Real rho0 = 1.0;
  const Real P0 = rho0 / (gamma * Ma * Ma);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");
  auto &coords = pmb->coords;
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, ccmat::volume_fraction(), k, j, i) = 1.0;
        v(0, ccmat::rho(), k, j, i) = rho0;

        const Real x = coords.Xc<parthenon::X1DIR>(i);
        const Real y = coords.Xc<parthenon::X2DIR>(j);
        const Real r = std::sqrt(x * x + y * y);
        Real uphi, p;
        if (r < 0.2) {
          uphi = 5.0;
          p = P0 + 12.5 * r * r;
        } else if (r < 0.4) {
          uphi = 2.0 / r - 5.0;
          p = P0 + 12.5 * r * r + 4.0 * (1.0 - 5.0 * r - std::log(0.2) + std::log(r));
        } else {
          uphi = 0.0;
          p = P0 - 2.0 + 4.0 * std::log(2.0);
        }

        const Real u = energy_from_rho_P(eos_vec(0), rho0, p);
        // Momenta
        const Real vx = -uphi * y;
        const Real vy = uphi * x;
        v(0, ccbulk::momentum(0), k, j, i) = rho0 * vx;
        v(0, ccbulk::momentum(1), k, j, i) = rho0 * vy;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;

        // Total energy
        const Real ekin = 0.5 * rho0 * (vx * vx + vy * vy);
        v(0, ccbulk::total_material_energy(), k, j, i) = u + ekin;
      });
  return;
}

} // namespace gresho
