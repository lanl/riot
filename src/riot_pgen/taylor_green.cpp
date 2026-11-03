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
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

namespace taylor_green {
using parthenon::ParArray1D;
using namespace RiotEOS;

// Taylor, G. I. and Green, A. E., Mechanism of the Production of Small Eddies from Large
// Ones Proc. R. Soc. Lond. A, 158, 499–521 (1937).

//----------------------------------------------------------------------------------------
//! \fn  void taylor_green::ProblemGenerator
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

  const Real rho0 = pin->GetOrAddReal("taylor_green", "rho0", 1);
  const Real P0 = pin->GetOrAddReal("taylor_green", "P0", 1);
  const int kx = pin->GetOrAddInteger("taylor_green", "wave_number_x", 1);
  const int ky = pin->GetOrAddInteger("taylor_green", "wave_number_y", 1);
  const int kz = pin->GetOrAddInteger("taylor_green", "wave_number_z", 1);
  const Real Ax = pin->GetOrAddReal("taylor_green", "amplitude_x", 1.0);
  const Real Ay = pin->GetOrAddReal("taylor_green", "amplitude_y", -0.5);
  const Real Az = pin->GetOrAddReal("taylor_green", "amplitude_z", -0.5);

  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  PARTHENON_REQUIRE(nummat == 1, "Taylor-Green vortex is only for one mat");

  auto eos_vec = pmesh->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");

  const parthenon::RegionSize &mesh_size = pmesh->mesh_size;
  const Real dx1_mesh = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));
  const Real dx2_mesh = (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  const Real dx3_mesh = (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
  const Real fx = 2 * M_PI * kx / dx1_mesh;
  const Real fy = 2 * M_PI * ky / dx2_mesh;
  const Real fz = 2 * M_PI * kz / dx3_mesh;
  PARTHENON_REQUIRE(std::abs(Ax * fx + Ay * fy + Az * fz) <= 1e-5, "div(momentum) == 0");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::taylor_green", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);
        v(b, ccmat::volume_fraction(), k, j, i) = 1.0;

        const Real x = coords.Xc<parthenon::X1DIR>(i);
        const Real y = coords.Xc<parthenon::X2DIR>(j);
        const Real z = coords.Xc<parthenon::X3DIR>(k);

        // rho, u
        Real p = P0 + (rho0 / 16) * ((std::cos(2 * fx * x) + std::cos(2 * fy * y)) *
                                         (std::cos(2 * fz * z) + 2) -
                                     2);
        const Real u = energy_from_rho_P(eos_vec(0), rho0, p);
        v(b, ccmat::rho(), k, j, i) = rho0;
        v(b, ccmat::internal_energy(), k, j, i) = u;
        // Momenta
        const Real vx = Ax * std::sin(fx * x) * std::cos(fy * y) * std::cos(fz * z);
        const Real vy = Ay * std::cos(fx * x) * std::sin(fy * y) * std::cos(fz * z);
        const Real vz = Az * std::cos(fx * x) * std::cos(fy * y) * std::sin(fz * z);
        v(b, ccbulk::momentum(0), k, j, i) = rho0 * vx;
        v(b, ccbulk::momentum(1), k, j, i) = rho0 * vy;
        v(b, ccbulk::momentum(2), k, j, i) = rho0 * vz;

        // Total energy
        const Real ekin = 0.5 * rho0 * (vx * vx + vy * vy + vz * vz);
        v(b, ccbulk::total_material_energy(), k, j, i) = u + ekin;
      });
  return;
}

} // namespace taylor_green
