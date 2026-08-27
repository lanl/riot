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

#include "Kokkos_Random.hpp"
#include "riot_pgen/pgen.hpp"
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

namespace quirk {
using parthenon::ParArray1D;
using namespace RiotEOS;
typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

//----------------------------------------------------------------------------------------
//! \fn  void quirk::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim == 2, "Quirk carbuncle test is for 2D.");

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
                    "Quirk carbuncle test is only for one mat");

  // case = 0 has a Ma = 6 shock
  // case = 1 has a Ma = 20 shock
  const int Ma_case = pin->GetOrAddInteger("quirk", "case", 0);
  const Real perturb_amp = pin->GetOrAddReal("quirk", "perturb_amp", 1.0e-2);
  const Real rho_preshock = 1.0;
  const Real press_preshock = 1.0;
  Real rho_postshock, press_postshock, v_postshock;
  if (Ma_case == 0) {
    rho_postshock = 216.0 / 41.0;
    press_postshock = 251.0 / 6.0;
    v_postshock = 35.0 * std::sqrt(35.0) / 36.0;
  } else if (Ma_case == 1) {
    rho_postshock = 160.0 / 27.0;
    press_postshock = 466.5;
    v_postshock = 133.0 * std::sqrt(1.4) / 8.0;
  } else {
    PARTHENON_THROW("Unsupported Ma case. Supported values are 0 and 1.");
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");
  auto &coords = pmb->coords;

  auto rng_pool =
      RNGPool(pmb->gid); // Seed is meshblock gid for consistency across MPI decomposition

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();
        auto perturb = [&]() {
          return rng_gen.drand(-0.5 * perturb_amp, 0.5 * perturb_amp);
        };
        const Real x = coords.Xc<parthenon::X1DIR>(i);
        Real rho, press, vx, vy;
        if (x > 5.0) {
          rho = rho_preshock + perturb();
          press = press_preshock + perturb();
          vx = perturb();
          vy = perturb();
        } else {
          rho = rho_postshock;     // + perturb();
          press = press_postshock; // + perturb();
          vx = v_postshock;        // + perturb();
          vy = 0.0;
        }

        v(0, ccmat::volume_fraction(), k, j, i) = 1.0;
        v(0, ccmat::rho(0), k, j, i) = rho;
        v(0, ccbulk::momentum(0), k, j, i) = rho * vx;
        v(0, ccbulk::momentum(1), k, j, i) = rho * vy;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;

        Real u = energy_from_rho_P(eos_vec(0), rho, press);

        // Total energy
        const Real ekin = 0.5 * rho * (vx * vx + vy * vy);
        v(0, ccbulk::total_material_energy(), k, j, i) = u + ekin;

        rng_pool.free_state(rng_gen);
      });
  return;
}

} // namespace quirk
