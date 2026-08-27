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

// gravitational acceleration of a sinusoidal density profile

#include <singularity-eos/eos/eos.hpp>

#include "riot_pgen/pgen.hpp"

namespace gacc {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void gacc::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::volume_fraction, ccmat::internal_energy,
                         ccbulk::total_material_energy, ccbulk::momentum>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // Adiabatic index (assuming single material)
  auto gravity_pkg = pmb->packages.Get("gravity");
  auto mat_pkg = pmb->packages.Get("materials");
  auto eos_vec = mat_pkg->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  Real gm1 = eos_vec[0].GruneisenParamFromDensityTemperature(1., 1.);
  Real gamma = gm1 + 1.0;
  auto gdim = gravity_pkg->Param<int>("gravity_dim");
  auto gacc = gravity_pkg->Param<Real>("gravity_g");
  const int ndim = pmb->pmy_mesh->ndim;

  // Problem Parameters
  const Real P0 = pin->GetOrAddReal("problem", "P0", 2.5); // uniform pressure
  const Real vy = pin->GetOrAddReal("problem", "vy", 0.0); // initial velocity-y

  // Indexing
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator::gacc", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<parthenon::X1DIR>(i);
        Real y = coords.Xc<parthenon::X2DIR>(j);
        Real z = coords.Xc<parthenon::X3DIR>(k);

        // sinusoidal density profile, evaluated pointwise at cell centroid
        Real rho = 1. + 0.1 * std::sin(2. * y * M_PI);
        v(0, ccmat::rho(0), k, j, i) = rho;
        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;

        // zero out velocities
        for (int dim = 0; dim < ndim; dim++) {
          v(0, ccbulk::momentum(dim), k, j, i) = 0.;
        }

        // initial velocity
        v(0, ccbulk::momentum(1), k, j, i) = rho * vy;

        // internal energy density
        Real u = P0 / (gamma - 1);

        // kinetic energy density
        Real ekin = 0.5 * rho * vy * vy;

        // total material energy
        v(0, ccbulk::total_material_energy(), k, j, i) = u + ekin;
      });

  return;
}

} // namespace gacc
