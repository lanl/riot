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
//      cell-centered field, the magnetic energy, and div B are all derived from face
//      state by MHD::PostInitialization, which is installed as the mhd package's
//      PostInitializationMesh hook. A pgen that wrote cell-centered B directly would
//      produce a state that is not representable on the staggered mesh, and the first CT
//      update would silently overwrite it.
//
//   2. Set total_material_energy to the HYDRO total (u + kinetic), exactly as every other
//      RIOT pgen does, then call MHD::AddMagneticEnergyToTotal(pmb) as the LAST step.
//      That helper adds B^2/(2 mu0) through the same shared definition that
//      Multiphysics::FillInteriorDerived subtracts (ADR-002), so the conventions cannot
//      drift. It has to happen inside the pgen rather than in PostInitialization -- see
//      the comment on that function for the initialization-ordering reason. Do not
//      open-code the magnetic term here, and do not call the helper twice.
//
//   3. Where the field is not trivially divergence free by construction, initialize it as
//      a DISCRETE CURL OF A VECTOR POTENTIAL, using the same difference stencil that the
//      CT update and the div B diagnostic use. Then div B is zero to machine precision at
//      t = 0 as an algebraic identity, not as an approximation -- which matters because a
//      nonzero initial divergence is indistinguishable, later, from a defective CT
//      implementation.
//
// Staggering convention for the vector potential: A_d lives on the edge along direction
// d, so A1 sits at (x1c, x2f, x3f), A2 at (x1f, x2c, x3f), and A3 at (x1f, x2f, x3c).
// Then
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
//! and the failure mode without it (an unallocated face field) is obscure, so it is
//! caught here with a message that says what to do.
void RequireMHD(MeshBlock *pmb, const char *problem) {
  PARTHENON_REQUIRE(pmb->packages.AllPackages().count("mhd") == 1,
                    std::string(problem) + " requires MHD; set <physics>/mhd = true.");
}

//----------------------------------------------------------------------------------------
//! \brief The field-loop vector potential, evaluated at a NODE, for a loop whose axis is
//!        along `axis`. Only the component along the axis is nonzero, and it is a
//!        function of the two perpendicular node coordinates only.
//!
//! A plain functor with a KOKKOS_INLINE_FUNCTION call operator rather than a lambda,
//! because it is invoked from inside a KOKKOS_LAMBDA and nesting an extended device
//! lambda inside another is not portable.
struct LoopPotential {
  parthenon::Coordinates_t coords;
  Real amp, rloop, c_a, c_b;
  int axis;

  //! Perpendicular node coordinates for each axis, taken in cyclic order so the sign
  //! conventions of the curl below stay right-handed: axis 1 -> (x2, x3),
  //! axis 2 -> (x3, x1), axis 3 -> (x1, x2).
  KOKKOS_INLINE_FUNCTION Real operator()(const int k, const int j, const int i) const {
    Real pa, pb;
    if (axis == 1) {
      pa = coords.Xf<parthenon::X2DIR>(j);
      pb = coords.Xf<parthenon::X3DIR>(k);
    } else if (axis == 2) {
      pa = coords.Xf<parthenon::X3DIR>(k);
      pb = coords.Xf<parthenon::X1DIR>(i);
    } else {
      pa = coords.Xf<parthenon::X1DIR>(i);
      pb = coords.Xf<parthenon::X2DIR>(j);
    }
    const Real r = std::sqrt(SQR(pa - c_a) + SQR(pb - c_b));
    return (r < rloop) ? amp * (rloop - r) : 0.0;
  }
};

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
//! The reference solution at t = 0.08 includes a compound structure (a slow shock
//! attached to a rotational discontinuity) that is sensitive to the solver, which is what
//! makes this a real test rather than a smoke test. artemis/tst/scripts/mhd/athena_bw.std
//! is an independent Athena++ solution for it.
//!
//! No vector potential is needed: the field is uniform along x, so div B = d_1 B1 = 0
//! holds exactly for any By(x) profile. The transverse discontinuity is placed on the
//! same x0 as the fluid discontinuity.
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
//! \brief A weak magnetic field loop advected across a periodic box.
//!
//! The vector potential has a single nonzero component along the loop axis,
//! A_axis = a0 * max(0, r_loop - r) with r measured from the axis, so B = curl A is a
//! loop of field in the plane perpendicular to the axis, confined to r < r_loop and
//! embedded in a uniform, uniformly moving gas. The field is dynamically negligible (beta
//! ~ 1e6), so the exact solution is pure advection: after one crossing time the loop must
//! return to its initial position with its shape and magnetic energy intact.
//!
//! In 2D this is the sharpest available test of the CT implementation, because the two
//! failure modes it exposes are ones no 1D test can see: an EMF averaging error diffuses
//! the loop (energy decays) and an orientation or upwinding error distorts it
//! anisotropically even though the advection velocity is uniform.
//!
//! ** In 3D WITH A VELOCITY COMPONENT ALONG THE LOOP AXIS it becomes something stronger.
//! ** This is Gardiner & Stone (2005) section 5.4. Take the axis to be x3, so B = (B1,
//! B2, 0) and v = (v1, v2, v3) with v3 != 0. Then
//!
//!   E1 = v3 B2,   E2 = -v3 B1,   E3 = v2 B1 - v1 B2
//!
//! are all nonzero, and the axial field evolves as
//!
//!   d_t B3 = -(d_1 E2 - d_2 E1) = v3 (d_1 B1 + d_2 B2) = v3 * div B = 0.
//!
//! So B3 must remain EXACTLY zero -- but only because the two transverse EMFs cancel
//! against each other. Any inconsistency between how E1 and E2 are assembled shows up
//! immediately as a spurious axial field, with nothing to hide behind. That makes
//! `max abs(B3)` a direct, quantitative check on `MHD::UpwindEMF<X1DIR>` and `<X2DIR>`,
//! which are unreachable in 2D (there the collapsed branch is used instead). `loop_axis`
//! rotates the whole setup so each of the three edge directions can be checked in turn.
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
  // Loop axis. The vector potential and the two nonzero face components rotate with it,
  // so axis = 1/2/3 puts the same physical setup on a different pair of edge directions.
  // Used to check all three permutations of the EMF kernel; see the note above on why the
  // 3D case with a velocity along the axis is the load-bearing test.
  const int axis = pin->GetOrAddInteger("mhd_field_loop", "loop_axis", 3);
  PARTHENON_REQUIRE(axis >= 1 && axis <= 3,
                    "mhd_field_loop/loop_axis must be 1, 2, or 3");
  // Loop center, in the two coordinates perpendicular to the axis, cyclic order.
  const Real ax1c = pin->GetOrAddReal("mhd_field_loop", "center_a", 0.0);
  const Real ax2c = pin->GetOrAddReal("mhd_field_loop", "center_b", 0.0);

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

  // The vector potential, evaluated at the node (x1f(i), x2f(j), x3f(k)). Only the
  // component along `axis` is nonzero, and it depends only on the two coordinates
  // perpendicular to the axis -- so evaluating it at the node and then differencing it
  // along one perpendicular direction gives exactly the value it would have on the edge
  // along `axis`. That is what makes the two face components below a discrete curl of a
  // single-valued potential, and hence div B = 0 as an algebraic identity.
  //
  // A POD functor rather than a captured lambda: a KOKKOS_LAMBDA defined inside another
  // KOKKOS_LAMBDA is not portable (nvcc rejects a nested extended device lambda), and
  // this is called from inside the par_for bodies below.
  LoopPotential apot{coords, a0, rloop, ax1c, ax2c, axis};

  const Real dx1 = coords.Dx<parthenon::X1DIR>();
  const Real dx2 = coords.Dx<parthenon::X2DIR>();
  const Real dx3 = coords.Dx<parthenon::X3DIR>();

  // B = curl A with only A_axis nonzero, cyclically:
  //   axis 3:  B1 = +dA3/dx2,  B2 = -dA3/dx1,  B3 = 0
  //   axis 1:  B2 = +dA1/dx3,  B3 = -dA1/dx2,  B1 = 0
  //   axis 2:  B3 = +dA2/dx1,  B1 = -dA2/dx3,  B2 = 0
  // Each face element gets its own bounds because the face normal direction carries one
  // extra plane.
  auto f1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b1", kb.s, kb.e, jb.s, jb.e, f1.s, f1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real b1 = 0.0;
        if (axis == 3) b1 = (apot(k, j + 1, i) - apot(k, j, i)) / dx2;
        if (axis == 2) b1 = -(apot(k + 1, j, i) - apot(k, j, i)) / dx3;
        v(0, TE::F1, fbulk::magnetic_field(), k, j, i) = b1;
      });

  auto j2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b2", kb.s, kb.e, j2.s, j2.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real b2 = 0.0;
        if (axis == 3) b2 = -(apot(k, j, i + 1) - apot(k, j, i)) / dx1;
        if (axis == 1) b2 = (apot(k + 1, j, i) - apot(k, j, i)) / dx3;
        v(0, TE::F2, fbulk::magnetic_field(), k, j, i) = b2;
      });

  auto k3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "ProblemGenerator::mhd_field_loop_b3", k3.s, k3.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real b3 = 0.0;
        if (axis == 1) b3 = -(apot(k, j + 1, i) - apot(k, j, i)) / dx2;
        if (axis == 2) b3 = (apot(k, j, i + 1) - apot(k, j, i)) / dx1;
        v(0, TE::F3, fbulk::magnetic_field(), k, j, i) = b3;
      });

  MHD::AddMagneticEnergyToTotal(pmb);
}

} // namespace mhd_field_loop
