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

#include <singularity-eos/eos/eos.hpp>

namespace shock_tube {
using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void shock_tube::ProblemGenerator
//! \brief
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  riot::AllocateAllSparse(md);

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                         ccbulk::total_material_energy, ccbulk::momentum>(md);
  auto v = desc.GetPack(md);

  const Real rhol = pin->GetOrAddReal("shock_tube", "rho_l", 1.0);
  const Real Pl = pin->GetOrAddReal("shock_tube", "P_l", 1.0);
  const Real vl = pin->GetOrAddReal("shock_tube", "v_l", 0.0);
  const Real rhor = pin->GetOrAddReal("shock_tube", "rho_r", 0.125);
  const Real Pr = pin->GetOrAddReal("shock_tube", "P_r", 0.1);
  const Real vr = pin->GetOrAddReal("shock_tube", "v_r", 0.0);

  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  PARTHENON_REQUIRE(nummat == 1, "shock_tube setup is only for one mat");

  auto eos_vec = pmesh->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::shock_tube", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);
        // assume this is single material for now
        const Real x1c = coords.Xc<parthenon::X1DIR>(i);
        const bool lhs = x1c < 0.5;

        const Real vfrac = 1.0;
        const Real rho = (lhs) ? rhol : rhor;
        const Real vel = (lhs) ? vl : vr;
        const Real pres = (lhs) ? Pl : Pr;
        const Real uu = energy_from_rho_P(eos_vec(0), rho, pres);

        v(b, ccmat::volume_fraction(0), k, j, i) = vfrac;
        v(b, ccmat::rho(0), k, j, i) = rho;
        v(b, ccbulk::momentum(0), k, j, i) = rho * vel;
        v(b, ccbulk::momentum(1), k, j, i) = 0.0;
        v(b, ccbulk::momentum(2), k, j, i) = 0.0;
        v(b, ccmat::internal_energy(0), k, j, i) = uu;
        v(b, ccbulk::total_material_energy(), k, j, i) = uu + 0.5 * rho * SQR(vel);
      });

  return;
}

} // namespace shock_tube
