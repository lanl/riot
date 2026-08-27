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

// ionized shock tube

#include <singularity-eos/eos/eos.hpp>

#include "riot_pgen/pgen.hpp"

namespace ionized_shocktube_analytic {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void ionized_shocktube_analytic::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc = MakePackDescriptor<
      ccmat::rho, ccmat::internal_energy, ccbulk::total_material_energy, ccbulk::momentum,
      ccbulk::temperature, ccbulk::electron_temperature, ccbulk::electron_internal_energy,
      ccbulk::electron_pressure, ccbulk::pressure, ccbulk::electron_number_density,
      ccmat::ionization_zbar>((pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // Adiabatic index (assuming single material)
  auto hydro_pkg = pmb->packages.Get("hydro");
  auto mat_pkg = pmb->packages.Get("materials");
  auto eos_vec = mat_pkg->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  Real gm1 = eos_vec[0].GruneisenParamFromDensityTemperature(1., 1.);
  Real gamma = gm1 + 1.0;
  const Real cv = pin->GetReal("material0", "Cv");
  const Real abar = pin->GetReal("material0", "mean_atomic_mass");
  const Real zbar = pin->GetReal("material0", "mean_atomic_number");

  printf("RIOT_PGEN: Cv = %8.2e abar = %8.2e zbar = %8.2e gm1 = %8.2e\n", cv, abar, zbar,
         gm1);

  // Indexing
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  const Mesh *pmesh = pmb->pmy_mesh;

  const parthenon::RegionSize &mesh_size = pmesh->mesh_size;
  const Real dx1_mesh = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));

  // problem parameters
  const Real rhol = 2e-2;
  const Real rhor = 2e-3;
  const Real Tl = 1e6;
  const Real Tr = 5e5;

  pmb->par_for(
      "ProblemGenerator::ionized_shocktube_analytic", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<parthenon::X1DIR>(i);
        Real y = coords.Xc<parthenon::X2DIR>(j);
        Real z = coords.Xc<parthenon::X3DIR>(k);

        const Real halfdx = 0.5 * coords.Dxf<parthenon::X1DIR>(i);
        const Real left = (x + halfdx < 0.);
        const Real right = 1. - left;

        const Real rho = left * rhol + right * rhor;
        v(0, ccmat::rho(0), k, j, i) = rho;

        const Real isplit = 1. / (1.0 + zbar);
        const Real esplit = zbar * isplit;

        // electron temperature
        const Real Te = left * Tl + right * Tr;
        v(0, ccbulk::electron_temperature(), k, j, i) = Te;

        // ion temperature
        const Real Ti = left * Tl + right * Tr;
        v(0, ccbulk::temperature(), k, j, i) = Ti;

        // internal energy density
        const Real ue = esplit * rho * cv * Te;
        const Real ui = isplit * rho * cv * Ti;

        // electron and total material energy
        v(0, ccbulk::electron_internal_energy(), k, j, i) = ue;
        v(0, ccbulk::total_material_energy(), k, j, i) = ui + ue;
        // v(0, ccmat::internal_energy(0), k, j, i) = ui + ue;

        const Real ne = rho / abar / 1.66054e-24 * zbar;
        v(0, ccbulk::electron_number_density(), k, j, i) = ne;
        v(0, ccmat::ionization_zbar(0), k, j, i) = rho * zbar;

        // electron and total pressure
        const Real pe = ue * gm1;
        const Real pi = ui * gm1;
        v(0, ccbulk::electron_pressure(), k, j, i) = pe;
        v(0, ccbulk::pressure(), k, j, i) = pi + pe;
      });

  return;
}

} // namespace ionized_shocktube_analytic
