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

#include <cmath>
#include <cstdio>

#include "riot_pgen/pgen.hpp"
#include "tnburn/tnburn.hpp"
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

/*
  This problem is a basic test for TN burn, starting with a
  1:1 DT mix (at ~5 gm/cc) at 1 keV.

  Ways this could be extended:
    1. Add different kinds of material (P, D, T, He3, He4)
    2. Allow multiple materials, and multiphase materials (Would be good to exercise this)
*/
namespace tn_test {
using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void tn_test::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using unit_system = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  constexpr Real ergs_per_MeV{1.0e6 * unit_system::eV};
  constexpr Real MeV_to_kelvin = ergs_per_MeV / unit_system::boltzmann;
  constexpr Real temperature = 0.001 * MeV_to_kelvin; // 1 keV in kelvin
  // constexpr Real amu = unit_system::amu;
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                         ccbulk::total_material_energy, ccbulk::momentum, ccmat::iso>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  const int numiso =
      v.GetUpperBoundHost(0, ccmat::iso()) - v.GetLowerBoundHost(0, ccmat::iso()) + 1;
  PARTHENON_REQUIRE(nummat == 1, "This TN test is only set-up for one mat.");
  PARTHENON_REQUIRE(numiso >= 2, "This TN test requires at least two isotopes.");

  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");

  auto materials = pmb->packages.Get("materials");
  materials->PostInitializationMesh = nullptr;

  // Get the masses of D and T from the isotope data if TN is active, otherwise use
  // approximate values
  auto do_tn = pmb->packages.Get("riot")->Param<bool>("do_tn");
  const std::array<int, 2> d_and_t{1002, 1003};
  std::array<Real, 2> masses{0.0, 0.0};
  if (do_tn) {
    auto const &tnburn = pmb->packages.Get("TNBurn"); // for isotope masses
    // get masses of D and T for input parameters
    auto const isotopes = tnburn->Param<std::vector<int>>("all_isotopes");
    auto const isotope_masses = tnburn->Param<std::vector<Real>>("isotope_masses");
    // Find the D+T reaction
    for (int inx = 0; inx < 2; inx++) {
      Real &mass = masses[inx];
      const int &zaid = d_and_t[inx];
      for (int iso = 0; iso < isotopes.size(); iso++) {
        if (zaid == isotopes[iso]) {
          mass = isotope_masses[iso];
          break;
        }
      }
      if (mass == 0.0) PARTHENON_FAIL("Could not find a needed isotope's mass");
    }
  } else {
    masses = {2.0141, 3.0155};
  }

  // We also have to figure out which materials are d and t
  auto const &iso_names = pmb->packages.Get("materials")
                              ->Param<std::vector<std::vector<int>>>("Isotope Zaids");
  std::array<int, 2> DT_iso_inx = {-1, -1};
  for (int iso_num = 0; iso_num < 2; iso_num++) {
    auto const myzaid = d_and_t[iso_num];
    for (int i = 0; i < iso_names[0].size(); i++) {
      if (iso_names[0][i] == myzaid) {
        DT_iso_inx[iso_num] = i;
        break;
      }
    }
    if (DT_iso_inx[iso_num] == -1)
      PARTHENON_FAIL(
          ("Did not find isotope " + std::to_string(myzaid) + " in material 0.").c_str());
  }

  const Real D_mass = masses[0];
  const Real T_mass = masses[1];

  // initial conditions
  const Real density = D_mass + T_mass; // ~ 5 g/cc

  pmb->par_for(
      "ProblemGenerator::tn_test", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, ccmat::volume_fraction(), k, j, i) = 1.0;

        // rho, u
        const Real u =
            eos_vec(0).InternalEnergyFromDensityTemperature(density, temperature);
        v(0, ccmat::rho(), k, j, i) = density;
        v(0, ccmat::internal_energy(), k, j, i) = density * u;

        // Momenta
        v(0, ccbulk::momentum(0), k, j, i) = 0.0;
        v(0, ccbulk::momentum(1), k, j, i) = 0.0;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;

        // Total energy
        v(0, ccbulk::total_material_energy(), k, j, i) = density * u;

        // Isotopics
        for (int p = 0; p < numiso; p++) {
          v(0, ccmat::iso(p), k, j, i) = 0.0;
        }
        v(0, ccmat::iso(DT_iso_inx[0]), k, j, i) = D_mass;
        v(0, ccmat::iso(DT_iso_inx[1]), k, j, i) = T_mass;
      });
  return;
}

} // namespace tn_test
