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

// hydrostatic equilibrium stratification

#include <singularity-eos/eos/eos.hpp>

#include "riot_pgen/pgen.hpp"

namespace hse {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void hse::ProblemGenerator
//! \brief
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  riot::AllocateAllSparse(md);

  auto v = riot::MakePack<ccmat::rho, ccmat::internal_energy,
                          ccbulk::total_material_energy, ccbulk::momentum>(md);

  // Adiabatic index (assuming single material)
  auto gravity_pkg = pmesh->packages.Get("gravity");
  auto mat_pkg = pmesh->packages.Get("materials");
  auto eos_vec = mat_pkg->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  const Real gm1 = eos_vec[0].GruneisenParamFromDensityTemperature(1., 1.);
  auto gdim = gravity_pkg->Param<int>("gravity_dim");
  auto gacc = gravity_pkg->Param<Real>("gravity_g");
  const int ndim = pmesh->ndim;

  // Problem Parameters
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0); // background density
  const Real P0 = pin->GetOrAddReal("problem", "P0", 2.5);     // background pressure

  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::hse", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);
        const Real x2 = coords.Xc<parthenon::X2DIR>(j);

        v(b, ccmat::rho(0), k, j, i) = rho0;

        // zero out momenta
        for (int dim = 0; dim < ndim; dim++) {
          v(b, ccbulk::momentum(dim), k, j, i) = 0.;
        }

        // internal energy density
        const Real pres = P0 + gacc * rho0 * x2;
        const Real uu = pres / gm1;

        // total material energy
        v(b, ccbulk::total_material_energy(), k, j, i) = uu;
      });

  return;
}

} // namespace hse
