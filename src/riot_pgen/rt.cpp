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

// Single mode Rayleigh-Taylor instability from Athena code test page
// https://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html

#include <singularity-eos/eos/eos.hpp>

#include "microphysics/eos_riot.hpp"
#include "riot_pgen/pgen.hpp"

namespace rt {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void rt::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  using RiotEOS::EOS_Array_t;

  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  auto v = riot::MakePack<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                          ccbulk::total_material_energy, ccbulk::momentum>(rc.get());

  // Adiabatic index (assuming single material)
  auto gravity_pkg = pmb->packages.Get("gravity");
  auto mat_pkg = pmb->packages.Get("materials");
  const auto &eos = mat_pkg->Param<EOS_Array_t>("d.d.EOS");
  auto gdim = gravity_pkg->Param<int>("gravity_dim");
  auto gacc = gravity_pkg->Param<Real>("gravity_g");
  const int ndim = pmb->pmy_mesh->ndim;

  // Problem Parameters
  const Real mx = 4.;
  const Real my = 3.;
  // pressure at interface
  const Real amplitude = pin->GetOrAddReal("rayleightaylor", "amp_v", 0.01);
  const Real P0 = pin->GetOrAddReal("rayleightaylor", "P0", 2.5);
  const Real rhol = pin->GetOrAddReal("rayleightaylor", "rho_low", 1.0);
  const Real rhoh = pin->GetOrAddReal("rayleightaylor", "rho_high", 2.0 * rhol);

  // Indexing
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  PARTHENON_REQUIRE_THROWS(gdim < 2, "Cannot accelerate in z direction");

  pmb->par_for(
      "ProblemGenerator::rt", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<parthenon::X1DIR>(i);
        Real y = coords.Xc<parthenon::X2DIR>(j);
        Real z = coords.Xc<parthenon::X3DIR>(k);

        const Real cond_coord = (gdim == 0) ? x : y;
        const Real orthog_coord = (gdim == 0) ? y : x;
        Real rho = (cond_coord < 0.) ? rhol : rhoh;
        v(0, ccmat::rho(0), k, j, i) = (cond_coord < 0.) ? rho : 0.;
        v(0, ccmat::rho(1), k, j, i) = (cond_coord < 0.) ? 0. : rho;
        v(0, ccmat::volume_fraction(0), k, j, i) = (cond_coord < 0.) ? 1.0 : 0.;
        v(0, ccmat::volume_fraction(1), k, j, i) =
            1.0 - v(0, ccmat::volume_fraction(0), k, j, i);

        // hydrostatic pressure
        Real p = P0 + gacc * rho * cond_coord;

        // internal energy density
        int matid = (cond_coord < 0) ? 0 : 1;
        Real u = RiotEOS::energy_from_rho_P(eos(matid), rho, p);

        // zero out velocities
        for (int dim = 0; dim < ndim; dim++) {
          v(0, ccbulk::momentum(dim), k, j, i) = 0.;
        }

        // perturbation to y momentum
        Real vmod = amplitude * (1 + std::cos(mx * M_PI * orthog_coord)) *
                    (1 + std::cos(my * M_PI * cond_coord)) / 4.;
        v(0, ccbulk::momentum(gdim), k, j, i) = rho * vmod;

        // total material energy perturbation
        Real ekin = 0.5 * rho * vmod * vmod;
        v(0, ccbulk::total_material_energy(), k, j, i) = u + ekin;
      });

  return;
}

} // namespace rt
