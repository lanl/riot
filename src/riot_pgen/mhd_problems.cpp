//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
//
// Ideal-MHD problem generators, ported from artemis/tst/scripts/mhd and the corresponding
// donor problem setups on lanl/artemis branch dempsey/mhd @ 3e5aeb5.
//
// CONTRACT FOR EVERY MHD PROBLEM GENERATOR (ADR-003):
//
//   1. Write ONLY the face-centered magnetic field, never the cell-centered one. The
//      cell-centered field, the magnetic energy, and div B are all derived from face state
//      by MHD::PostInitialization, which is installed as the mhd package's
//      PostInitializationMesh hook. A pgen that wrote cell-centered B directly would
//      produce a state that is not representable on the staggered mesh, and the first CT
//      update would silently overwrite it.
//
//   2. Set total_material_energy to the HYDRO total (u + kinetic), exactly as every other
//      RIOT pgen does, then call MHD::AddMagneticEnergyToTotal(pmb) as the LAST step. That
//      helper adds B^2/(2 mu0) through the same shared definition that
//      Multiphysics::FillInteriorDerived subtracts (ADR-002), so the conventions cannot
//      drift. It has to happen inside the pgen rather than in PostInitialization -- see the
//      comment on that function for the initialization-ordering reason. Do not open-code
//      the magnetic term here, and do not call the helper twice.
//
//   3. Where the field is not trivially divergence free by construction, initialize it as
//      a DISCRETE CURL OF A VECTOR POTENTIAL, using the same difference stencil that the
//      CT update and the div B diagnostic use. Then div B is zero to machine precision at
//      t = 0 as an algebraic identity, not as an approximation -- which matters because a
//      nonzero initial divergence is indistinguishable, later, from a defective CT
//      implementation.
//
// Staggering convention for the vector potential: A_d lives on the edge along direction d,
// so A1 sits at (x1c, x2f, x3f), A2 at (x1f, x2c, x3f), and A3 at (x1f, x2f, x3c). Then
//
//   B1(k,j,i) = [A3(k,j+1,i) - A3(k,j,i)]/dx2 - [A2(k+1,j,i) - A2(k,j,i)]/dx3
//   B2(k,j,i) = [A1(k+1,j,i) - A1(k,j,i)]/dx3 - [A3(k,j,i+1) - A3(k,j,i)]/dx1
//   B3(k,j,i) = [A2(k,j,i+1) - A2(k,j,i)]/dx1 - [A1(k,j+1,i) - A1(k,j,i)]/dx2
//
// which is B = curl A on the staggered mesh, and whose divergence telescopes to zero.

#include "riot_pgen/pgen.hpp"

#include <singularity-eos/eos/eos.hpp>

#include "mhd/mhd.hpp"

namespace {

using parthenon::ParArray1D;
using TE = parthenon::TopologicalElement;

//----------------------------------------------------------------------------------------
//! Allocate every sparse variable on this block, as the other pgens do.
void AllocateAll(MeshBlock *pmb, std::shared_ptr<parthenon::MeshBlockData<Real>> &rc) {
  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
}

//! Guard shared by every MHD pgen: these setups only make sense with the mhd package on,
//! and the failure mode without it (an unallocated face field) is obscure, so it is caught
//! here with a message that says what to do.
void RequireMHD(MeshBlock *pmb, const char *problem) {
  PARTHENON_REQUIRE(pmb->packages.AllPackages().count("mhd") == 1,
                    std::string(problem) + " requires MHD; set <physics>/mhd = true.");
}

} // namespace

//========================================================================================
// Brio & Wu (1988) MHD shock tube
//========================================================================================
namespace mhd_shock_tube {

//----------------------------------------------------------------------------------------
//! \fn  void mhd_shock_tube::ProblemGenerator
//! \brief The Brio-Wu coplanar MHD Riemann problem.
//!
//! Left  (x < x0): rho = 1,     P = 1,   By = +1
//! Right (x > x0): rho = 0.125, P = 0.1, By = -1
//! Bx = 0.75 everywhere, Bz = 0, v = 0, gamma = 2.
//!
//! The reference solution at t = 0.08 includes a compound structure (a slow shock attached
//! to a rotational discontinuity) that is sensitive to the solver, which is what makes this
//! a real test rather than a smoke test. artemis/tst/scripts/mhd/athena_bw.std is an
//! independent Athena++ solution for it.
//!
//! No vector potential is needed: the field is uniform along x, so div B = d_1 B1 = 0
//! holds exactly for any By(x) profile. The transverse discontinuity is placed on the same
//! x0 as the fluid discontinuity.
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace fbulk = face_variables::bulk;
  using namespace RiotEOS;

  RequireMHD(pmb, "mhd_shock_tube");
  auto &rc = pmb->meshblock_data.Get();
  AllocateAll(pmb, rc);

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                         ccbulk::total_material_energy, ccbulk::momentum,
                         fbulk::magnetic_field>((pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  const Real x0 = pin->GetOrAddReal("mhd_shock_tube", "x0", 0.0);
  const Real rhol = pin->GetOrAddReal("mhd_shock_tube", "rho_l", 1.0);
  const Real Pl = pin->GetOrAddReal("mhd_shock_tube", "P_l", 1.0);
  const Real rhor = pin->GetOrAddReal("mhd_shock_tube", "rho_r", 0.125);
  const Real Pr = pin->GetOrAddReal("mhd_shock_tube", "P_r", 0.1);
  const Real bx = pin->GetOrAddReal("mhd_shock_tube", "bx", 0.75);
  const Real byl = pin->GetOrAddReal("mhd_shock_tube", "by_l", 1.0);
  const Real byr = pin->GetOrAddReal("mhd_shock_tube", "by_r", -1.0);

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  PARTHENON_REQUIRE(nummat == 1, "mhd_shock_tube is a single-material setup");

  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");
  auto &coords = pmb->coords;

  // Gas state, on cell centers.
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "ProblemGenerator::mhd_shock_tube", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const bool lhs = coords.Xc<parthenon::X1DIR>(i) < x0;
        const Real rho = lhs ? rhol : rhor;
        const Real pres = lhs ? Pl : Pr;
        const Real uu = energy_from_rho_P(eos_vec(0), rho, pres);

        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
        v(0, ccmat::rho(0), k, j, i) = rho;
        v(0, ccbulk::momentum(0), k, j, i) = 0.0;
        v(0, ccbulk::momentum(1), k, j, i) = 0.0;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;
        v(0, ccmat::internal_energy(0), k, j, i) = uu;
        // Hydro total only; the magnetic part is added by PostCommsFillDerived.
        v(0, ccbulk::total_material_energy(), k, j, i) = uu;
      });

  // Face-centered field, one loop per topological element so each gets its own bounds.
  // B1 is uniform, which is the discrete statement of div B = 0 for this setup, and is
  // also the invariant the regression test checks (|Bx - 0.75| <= 1e-12 for all time).
  auto f1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "ProblemGenerator::mhd_shock_tube_b1", kb.s, kb.e, jb.s, jb.e, f1.s, f1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, TE::F1, fbulk::magnetic_field(), k, j, i) = bx;
      });

  auto j2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "ProblemGenerator::mhd_shock_tube_b2", kb.s, kb.e, j2.s, j2.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // The x2 faces of a given cell column share the cell's x1 centroid, so the jump
        // sits at the same x0 as the fluid jump.
        v(0, TE::F2, fbulk::magnetic_field(), k, j, i) =
            (coords.Xc<parthenon::X1DIR>(i) < x0) ? byl : byr;
      });

  auto k3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "ProblemGenerator::mhd_shock_tube_b3", k3.s, k3.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, TE::F3, fbulk::magnetic_field(), k, j, i) = 0.0;
      });

  // Promote the conserved total energy to the MHD convention. Must be last: it reads the
  // face field written above.
  MHD::AddMagneticEnergyToTotal(pmb);
}

} // namespace mhd_shock_tube

//========================================================================================
// Gardiner & Stone (2005) advected field loop
//========================================================================================
namespace mhd_field_loop {

//----------------------------------------------------------------------------------------
//! \fn  void mhd_field_loop::ProblemGenerator
//! \brief A weak magnetic field loop advected diagonally across a periodic box.
//!
//! A3 = a0 * max(0, r_loop - r) with r measured from the loop center, so
//! B = curl A is a purely in-plane loop of field confined to r < r_loop, embedded in a
//! uniform, uniformly moving gas. The field is dynamically negligible (beta ~ 1e6), so the
//! exact solution is pure advection: after one crossing time the loop must return to its
//! initial position with its shape and magnetic energy intact.
//!
//! This is the sharpest available test of the CT implementation, because the two failure
//! modes it exposes are ones no 1D test can see: an EMF averaging error diffuses the loop
//! (energy decays) and an orientation or upwinding error distorts it anisotropically even
//! though the advection velocity is uniform.
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace fbulk = face_variables::bulk;
  using namespace RiotEOS;

  RequireMHD(pmb, "mhd_field_loop");
  auto &rc = pmb->meshblock_data.Get();
  AllocateAll(pmb, rc);

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                         ccbulk::total_material_energy, ccbulk::momentum,
                         fbulk::magnetic_field>((pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  const Real rho0 = pin->GetOrAddReal("mhd_field_loop", "rho", 1.0);
  const Real p0 = pin->GetOrAddReal("mhd_field_loop", "P", 1.0);
  const Real v1 = pin->GetOrAddReal("mhd_field_loop", "v1", 2.0);
  const Real v2 = pin->GetOrAddReal("mhd_field_loop", "v2", 1.0);
  const Real v3 = pin->GetOrAddReal("mhd_field_loop", "v3", 0.0);
  const Real a0 = pin->GetOrAddReal("mhd_field_loop", "amp", 1.0e-3);
  const Real rloop = pin->GetOrAddReal("mhd_field_loop", "r_loop", 0.3);
  const Real x1c0 = pin->GetOrAddReal("mhd_field_loop", "x1_center", 0.0);
  const Real x2c0 = pin->GetOrAddReal("mhd_field_loop", "x2_center", 0.0);

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  PARTHENON_REQUIRE(nummat == 1, "mhd_field_loop is a single-material setup");

  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");
  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;
  PARTHENON_REQUIRE(ndim >= 2, "mhd_field_loop needs at least two dimensions");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "ProblemGenerator::mhd_field_loop", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real uu = energy_from_rho_P(eos_vec(0), rho0, p0);
        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
        v(0, ccmat::rho(0), k, j, i) = rho0;
        v(0, ccbulk::momentum(0), k, j, i) = rho0 * v1;
        v(0, ccbulk::momentum(1), k, j, i) = rho0 * v2;
        v(0, ccbulk::momentum(2), k, j, i) = rho0 * v3;
        v(0, ccmat::internal_energy(0), k, j, i) = uu;
        v(0, ccbulk::total_material_energy(), k, j, i) =
            uu + 0.5 * rho0 * (SQR(v1) + SQR(v2) + SQR(v3));
      });

  // A3 on the E3 edge at (x1f(i), x2f(j)); zero outside the loop. A1 = A2 = 0, so
  // B1 = dA3/dx2, B2 = -dA3/dx1, B3 = 0.
  auto a3 = KOKKOS_LAMBDA(const int i, const int j) {
    const Real dx = coords.Xf<parthenon::X1DIR>(i) - x1c0;
    const Real dy = coords.Xf<parthenon::X2DIR>(j) - x2c0;
    const Real r = std::sqrt(SQR(dx) + SQR(dy));
    return (r < rloop) ? a0 * (rloop - r) : 0.0;
  };

  const Real dx1 = coords.Dx<parthenon::X1DIR>();
  const Real dx2 = coords.Dx<parthenon::X2DIR>();

  // The x1 faces run to ib.e + 1; the extra plane is why each element gets its own bounds.
  auto f1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b1", kb.s, kb.e, jb.s, jb.e, f1.s, f1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, TE::F1, fbulk::magnetic_field(), k, j, i) = (a3(i, j + 1) - a3(i, j)) / dx2;
      });

  auto j2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b2", kb.s, kb.e, j2.s, j2.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, TE::F2, fbulk::magnetic_field(), k, j, i) = -(a3(i + 1, j) - a3(i, j)) / dx1;
      });

  auto k3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b3", k3.s, k3.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, TE::F3, fbulk::magnetic_field(), k, j, i) = 0.0;
      });

  MHD::AddMagneticEnergyToTotal(pmb);
}

} // namespace mhd_field_loop
