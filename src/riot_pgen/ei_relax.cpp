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

// electron-ion relaxation with SESAME eos

#include <singularity-eos/eos/eos.hpp>

#include "riot_pgen/pgen.hpp"

namespace ei_relax {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void ei_relax::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc = MakePackDescriptor<
      cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccmat::internal_energy,
      ccmat::volume_fraction, ccbulk::total_material_energy, ccbulk::momentum,
      ccbulk::temperature, ccbulk::electron_temperature, ccbulk::electron_internal_energy,
      ccbulk::electron_pressure, ccbulk::pressure, ccbulk::electron_number_density,
      ccmat::ionization_zbar, cm::ionization_zbar>((pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // pull EoS pointers for electrons and ions (assumes single material)
  auto hydro_pkg = pmb->packages.Get("hydro");
  auto mat_pkg = pmb->packages.Get("materials");
  const auto &ion_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &electron_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &eos_from_matid =
      mat_pkg->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  const int mat_id = 0;

  // Indexing
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  const Mesh *pmesh = pmb->pmy_mesh;

  const parthenon::RegionSize &mesh_size = pmesh->mesh_size;
  const Real dx1_mesh = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));

  // problem parameters
  const Real rho = pin->GetReal("ei_relax", "rho");
  const Real Te = pin->GetReal("ei_relax", "Te0");
  const Real Ti = pin->GetReal("ei_relax", "Ti0");

  pmb->par_for(
      "ProblemGenerator::ei_relax", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<parthenon::X1DIR>(i);
        Real y = coords.Xc<parthenon::X2DIR>(j);
        Real z = coords.Xc<parthenon::X3DIR>(k);

        // Pull abar, zbar and EoS pointers and create lambda indexer so that
        // we can call the EoSs
        const Real abar = ion_eos[mat_id].MeanAtomicMass();
        const Real zbar = ion_eos[mat_id].MeanAtomicNumber();
        const int eos_id = eos_from_matid(mat_id);
        auto &eose = electron_eos(eos_id);
        auto &eosi = ion_eos(eos_id);
        RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, 0, k, j, i);

        // density and volume fraction
        v(0, ccmat::rho(0), k, j, i) = rho;
        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;

        // electron temperature
        v(0, ccbulk::electron_temperature(), k, j, i) = Te;

        // ion temperature
        v(0, ccbulk::temperature(), k, j, i) = Ti;

        // assume fully ionized, set ne and zbar
        const Real ne = rho / abar / 1.66054e-24 * zbar;
        v(0, ccbulk::electron_number_density(), k, j, i) = ne;
        v(0, ccmat::ionization_zbar(0), k, j, i) = rho * zbar;
        // have to set prim zbar because it's used in eos calls below
        v(0, cm::ionization_zbar(0), k, j, i) = zbar;

        // internal energy density
        const Real sie_e =
            eose.InternalEnergyFromDensityTemperature(rho, Te, lambda[mat_id]);
        const Real sie_i =
            eosi.InternalEnergyFromDensityTemperature(rho, Ti, lambda[mat_id]);
        const Real ue = sie_e * rho;
        const Real ui = sie_i * rho;

        // electron and total material energy
        v(0, ccbulk::electron_internal_energy(), k, j, i) = ue;
        v(0, ccbulk::total_material_energy(), k, j, i) = ui + ue;

        // electron and total pressure
        const Real pe = eose.PressureFromDensityTemperature(rho, Te, lambda[mat_id]);
        const Real pi = eosi.PressureFromDensityTemperature(rho, Ti, lambda[mat_id]);
        v(0, ccbulk::electron_pressure(), k, j, i) = pe;
        v(0, ccbulk::pressure(), k, j, i) = pi + pe;
      });

  return;
}

} // namespace ei_relax
