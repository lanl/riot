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

#include "riot_pgen/pgen.hpp"

#include <utils/constants.hpp>

namespace gaussian_conduction {

//----------------------------------------------------------------------------------------
//! \fn  void gaussian_conduction::ProblemGenerator
//! \brief
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  /* radiation_diffusion is not ported to the new loop abstractions; disabled for now.
  using namespace RadiationDiffusion;
  */
  using namespace parthenon;
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  riot::AllocateAllSparse(md);

  const Real T0 = pin->GetOrAddReal("gaussian_conduction", "T0", 1.0);
  const Real rho0 = pin->GetOrAddReal("gaussian_conduction", "rho0", 1.0);
  const Real Te0 = pin->GetOrAddReal("gaussian_conduction", "Te0", 1.0);
  const double t0 = 0.001;

  const auto &ionization = pmesh->packages.Get("ionization");
  const Real Ke = ionization->Param<Real>("electron_conductivity");
  const auto conductivity_model =
      ionization->Param<std::string>("electron_conductivity_model");
  PARTHENON_REQUIRE_THROWS(
      conductivity_model == "constant",
      "Problem 'gaussian_conduction' requires constant conductivity model!");

  auto &materials = pmesh->packages.Get("materials");
  const auto &ion_eos = materials->Param<parthenon::ParArray1D<RiotEOS::EOS>>("d.d.EOS");
  const auto &electron_eos = materials->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  using TE = parthenon::TopologicalElement;
  constexpr TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire, te);
  static auto desc = MakePackDescriptor<
      cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccmat::internal_energy,
      ccmat::volume_fraction, ccbulk::total_material_energy, ccbulk::momentum,
      ccbulk::temperature, ccbulk::electron_temperature, ccbulk::electron_internal_energy,
      ccbulk::electron_pressure, ccbulk::pressure, ccbulk::electron_number_density,
      cm::ionization_zbar>(md);
  auto pack = desc.GetPack(md);

  constexpr int mat_id = 0;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::conduction_analytic", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x1 = coords.X<1, te>(i);

        // EoS parameters, lambdas and electron specific heat
        const Real abar = ion_eos[mat_id].MeanAtomicMass();
        const Real zbar = ion_eos[mat_id].MeanAtomicNumber();
        const int eos_id = eos_from_matid(mat_id);
        auto &eose = electron_eos(eos_id);
        auto &eosi = ion_eos(eos_id);
        RiotEOS::LambdaIndexerMulti<decltype(pack)> lambda(pack, 0, k, j, i);
        // specific heat of electrons - should be constant so just pass any
        // temperature
        const Real cve =
            eose.SpecificHeatFromDensityTemperature(rho0, 1., lambda[mat_id]);

        // diffusion coefficient
        const Real D = Ke / (rho0 * cve);

        // density and vol frac
        pack(b, ccmat::rho(0), k, j, i) = rho0;
        pack(b, ccmat::volume_fraction(mat_id), k, j, i) = 1.0;

        const double c = 0.0;
        const Real Te = 0.1 + std::exp(-(x1 - c) * (x1 - c) / (4.0 * D * t0));
        const double Ti = Te;
        pack(b, ccbulk::electron_temperature(), k, j, i) = Te;
        pack(b, ccbulk::temperature(), k, j, i) = Ti;
        pack(b, cm::rho(), k, j, i) = rho0;

        // assume fully ionized
        const Real ne = rho0 / abar / 1.66054e-24 * zbar;
        pack(b, ccbulk::electron_number_density(), k, j, i) = ne;
        pack(b, cm::ionization_zbar(mat_id), k, j, i) = zbar;

        // internal energy density
        const Real sie_e =
            eose.InternalEnergyFromDensityTemperature(rho0, Te, lambda[mat_id]);
        const Real sie_i =
            eosi.InternalEnergyFromDensityTemperature(rho0, Ti, lambda[mat_id]);
        const Real ue = sie_e * rho0;
        const Real ui = sie_i * rho0;
        pack(b, ccbulk::electron_internal_energy(), k, j, i) = ue;
        pack(b, ccbulk::total_material_energy(), k, j, i) = ui + ue;

        // electron and total pressure
        const Real pe = eose.PressureFromDensityTemperature(rho0, Te, lambda[mat_id]);
        const Real pi = eosi.PressureFromDensityTemperature(rho0, Ti, lambda[mat_id]);
        pack(b, ccbulk::electron_pressure(), k, j, i) = pe;
        pack(b, ccbulk::pressure(), k, j, i) = pi + pe;
      });

  return;
}

} // namespace gaussian_conduction
