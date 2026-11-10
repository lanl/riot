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
//! \file self_gravity.cpp
//! \brief Self-gravity package: the Poisson solve for the potential, and the momentum
//! and energy sources (no Jeans swindle).

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <utils/constants.hpp>

#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "self_gravity/poisson_equation.hpp"
#include "self_gravity/self_gravity.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace SelfGravity {

using parthenon::BoundaryFunction::BCSide;
using parthenon::BoundaryFunction::BCType;

// Homogeneous Dirichlet (phi = 0 on the domain face)
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto DirZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    parthenon::BoundaryFunction::GenericBC<DIR, SIDE, BCType::FixedFace, phi>(rc, coarse,
                                                                             0.0);
  };
}

// Homogeneous Neumann (dphi/dn = 0 on the domain face)
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto NeuZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    parthenon::BoundaryFunction::GenericBC<DIR, SIDE, BCType::Outflow, phi>(rc, coarse,
                                                                           0.0);
  };
}

//----------------------------------------------------------------------------------------
//! \fn GetSolverSptr
//! \brief Build the Poisson solver selected by <block>/solver: pure geometric multigrid
//! ("MG") or a Krylov method ("BiCGSTAB", "CG") preconditioned by one MG V-cycle.
//! Every option reads its tolerances and MG parameters from the same input block.
std::shared_ptr<parthenon::solvers::SolverBase>
GetSolverSptr(ParameterInput *pin, const std::string &block_name) {
  using namespace parthenon::solvers;
  using PoissEq = PoissonEquation<phi>;
  using prolongator_t = ProlongationBlockInteriorZeroDirichlet;
  using restrictor_t = RestrictionCombined;
  using mg_t = MGSolver<PoissEq, prolongator_t, restrictor_t>;
  const std::string solver = pin->GetOrAddString(
      block_name, "solver", "MG", {"MG", "BiCGSTAB", "CG"}, "Poisson solver");
  const std::string base = "base";
  if (solver == "MG") {
    // Standalone MG with flux correction only converges with
    // parthenon/mesh/base_block_coarsenings >= 1
    return std::make_shared<MGSolver<PoissEq, prolongator_t>>(
        base, self_gravity_u_container, self_gravity_rhs_container, pin, block_name,
        PoissEq());
  } else if (solver == "BiCGSTAB") {
    return std::make_shared<BiCGSTABSolver<PoissEq, mg_t>>(
        base, self_gravity_u_container, self_gravity_rhs_container, pin, block_name,
        PoissEq());
  }
  return std::make_shared<CGSolver<PoissEq, mg_t>>(base, self_gravity_u_container,
                                                    self_gravity_rhs_container, pin,
                                                    block_name, PoissEq());
}

//----------------------------------------------------------------------------------------
//! \fn  void SelfGravity::Initialize
//! \brief Initializes the self-gravity package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
  auto self_gravity = std::make_shared<StateDescriptor>("self_gravity");
  Params &params = self_gravity->AllParams();
  const std::string block_name = "self_gravity";

  // Check coordinates compatibility
  PARTHENON_REQUIRE(parthenon::IsCoord<parthenon::UniformCartesian>(),
                    "Self-gravity only supports UniformCartesian coordinates");

  // Units and constants
  Real four_pi_G = 4.0 * M_PI * pc::g_newt;
  if (pin->GetOrAddBoolean(block_name, "units_override", false,
                           "Override the CGS gravitational constant with four_pi_G")) {
    four_pi_G = pin->GetOrAddReal(block_name, "four_pi_G", 1.0,
                                  "4 pi G, only used if units_override is true");
  }
  params.Add("four_pi_G", four_pi_G);

  // A material that neither sources nor feels gravity (e.g. a background standing in for
  // vacuum, which then stays static at uniform pressure), -1 for none
  params.Add("inert_matid",
             pin->GetOrAddInteger(block_name, "inert_matid", -1,
                                  "Material excluded from self-gravity (-1 for none)"));

  // Without a Jeans swindle the Poisson problem is ill-posed for fully periodic domains
  const int ndim = 1 + (pin->GetOrAddInteger("parthenon/mesh", "nx2", 1) > 1) +
                   (pin->GetOrAddInteger("parthenon/mesh", "nx3", 1) > 1);
  bool fully_periodic = true;
  for (int d = 1; d <= ndim; d++) {
    const std::string sd = std::to_string(d);
    fully_periodic =
        fully_periodic &&
        (pin->GetOrAddString("parthenon/mesh", "ix" + sd + "_bc", "outflow") ==
         "periodic") &&
        (pin->GetOrAddString("parthenon/mesh", "ox" + sd + "_bc", "outflow") ==
         "periodic");
  }
  PARTHENON_REQUIRE(!fully_periodic,
                    "Self-gravity does not support fully periodic domains (no Jeans "
                    "swindle).");

  // Enroll DirZ or NeuZ BCs if specified.  "default" defers to the mesh BC.
  using BF = parthenon::BoundaryFace;
  constexpr auto LL = BCSide::Inner;
  constexpr auto RR = BCSide::Outer;
  const std::vector<std::string> bc_names = {"ix1_bc", "ox1_bc", "ix2_bc",
                                             "ox2_bc", "ix3_bc", "ox3_bc"};
  std::vector<std::string> grav_bc(bc_names.size());
  for (std::size_t n = 0; n < bc_names.size(); n++) {
    grav_bc[n] = pin->GetOrAddString(block_name, bc_names[n], "default",
                                     {"default", "zero", "neumann"},
                                     "Poisson BC: default (mesh BC), zero, or neumann");
  }
  auto &ubf = self_gravity->UserBoundaryFunctions;
  if (grav_bc[0] == "zero") ubf[BF::inner_x1].push_back(DirZ<X1DIR, LL>());
  if (grav_bc[1] == "zero") ubf[BF::outer_x1].push_back(DirZ<X1DIR, RR>());
  if (grav_bc[2] == "zero") ubf[BF::inner_x2].push_back(DirZ<X2DIR, LL>());
  if (grav_bc[3] == "zero") ubf[BF::outer_x2].push_back(DirZ<X2DIR, RR>());
  if (grav_bc[4] == "zero") ubf[BF::inner_x3].push_back(DirZ<X3DIR, LL>());
  if (grav_bc[5] == "zero") ubf[BF::outer_x3].push_back(DirZ<X3DIR, RR>());
  if (grav_bc[0] == "neumann") ubf[BF::inner_x1].push_back(NeuZ<X1DIR, LL>());
  if (grav_bc[1] == "neumann") ubf[BF::outer_x1].push_back(NeuZ<X1DIR, RR>());
  if (grav_bc[2] == "neumann") ubf[BF::inner_x2].push_back(NeuZ<X2DIR, LL>());
  if (grav_bc[3] == "neumann") ubf[BF::outer_x2].push_back(NeuZ<X2DIR, RR>());
  if (grav_bc[4] == "neumann") ubf[BF::inner_x3].push_back(NeuZ<X3DIR, LL>());
  if (grav_bc[5] == "neumann") ubf[BF::outer_x3].push_back(NeuZ<X3DIR, RR>());

  // The potential and Poisson RHS are excluded from the RK registers
  using namespace parthenon::refinement_ops;
  static const parthenon::MetadataFlag OperatorSplit =
      Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);

  // Gravitational potential
  auto m_phi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                         Metadata::WithFluxes, Metadata::GMGRestrict,
                         Metadata::GMGProlongate, Metadata::CommunicateOne,
                         OperatorSplit});
  m_phi.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  self_gravity->AddField<phi>(m_phi);

  // Solver
  PARTHENON_REQUIRE(pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false),
                    "Self-gravity requires parthenon/mesh/multigrid = true");
  auto psolver = GetSolverSptr(pin, block_name);
  // We warm start from the potential stored in the base container
  psolver->initial_guess_is_zero = false;
  params.Add("solver_pointer", psolver);

  // Source term dU/dt variables
  self_gravity->RegisterMeshDataSubset(
      "dudt", RiotUtils::MakePackageDudtRequirements(
                  {ccbulk::momentum::name(), ccbulk::total_material_energy::name()}));

  return self_gravity;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SelfGravity::FillPoissonRHS
//! \brief Sets the RHS of the Poisson equation, 4 pi G rho, in the solver RHS container,
//! where rho is the summed (non-inert) material density
TaskStatus FillPoissonRHS(MeshData<Real> *md_base, MeshData<Real> *md_rhs) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto pm = md_base->GetParentPointer();
  auto &sg = pm->packages.Get("self_gravity");
  const Real four_pi_G = sg->Param<Real>("four_pi_G");
  const int inert = sg->Param<int>("inert_matid");

  auto vmat = riot::MakePack<ccmat::rho>(md_base);
  auto vrhs = riot::MakePack<phi>(md_rhs);
  const int nblocks = vmat.GetNBlocks();
  if ((nblocks != vrhs.GetNBlocks()) || (nblocks == 0)) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md_base,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv_rhs = RiotLoop::make_pack_view(idx_range, vrhs);
        RiotLoop::inner(idx_range, [&](const auto kji) { pv_rhs(phi(), kji) = 0.0; });
        const int nmat = vmat.GetSize(b, ccmat::rho());
        for (int m = 0; m < nmat; m++) {
          if (vmat(b, ccmat::rho(m)).sparse_id == inert) continue;
          idx_range.TeamBarrier();
          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, vmat, m);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv_rhs(phi(), kji) += four_pi_G * std::max(pv_m(ccmat::rho(), kji), 0.0);
          });
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SelfGravity::CalculateSelfGravitySource
//! \brief Calculates momentum and total energy sources due to self-gravity, summed over
//! the (non-inert) materials and phases.  The momentum source is each material's density
//! times the centered potential gradient, and the energy source uses its face mass
//! fluxes rather than rho v, so that it is consistent with the discrete mass transport.
//! These are the stage's corrected fluxes in state, the ones UpdateToNextStage applies,
//! and phi was solved from the same stage's density (base shares state's fields).
TaskStatus CalculateSelfGravitySource(MeshData<Real> *state, MeshData<Real> *base,
                                      MeshData<Real> *src) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  using TE = parthenon::TopologicalElement;

  if (state->NumBlocks() == 0) return TaskStatus::complete;
  const int ndim = state->GetParentPointer()->ndim;
  const int inert =
      state->GetParentPointer()->packages.Get("self_gravity")->Param<int>("inert_matid");

  // Pack up material densities (with their mass fluxes) and the potential
  auto vflx = riot::MakePack<ccmat::rho>(
      state, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  auto vphi = riot::MakePack<phi>(base);

  // Pack up fields for which this package produces source terms
  auto vsrc = riot::MakePack<ccbulk::momentum, ccbulk::total_material_energy>(src);
  const int nblocks = vflx.GetNBlocks();
  if ((nblocks != vphi.GetNBlocks()) || (nblocks != vsrc.GetNBlocks()) ||
      (nblocks == 0))
    return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state, TE::CC);
  // Memory offsets to the "plus" neighbor in each direction, which is also the "plus"
  // face of the cell for the flux views
  const auto di = idx_space.GetDelta(X1DIR);
  const auto dj = idx_space.GetDelta(X2DIR);
  const auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        // Uniform Cartesian only (see Initialize), so one spacing per direction
        const auto &coords = vflx.GetCoordinates(b);
        const Real dx[3] = {coords.Dxc<X1DIR>(0), coords.Dxc<X2DIR>(0),
                            coords.Dxc<X3DIR>(0)};
        const decltype(di) off[3] = {di, dj, dk};

        auto pv_phi = RiotLoop::make_pack_view(idx_range, vphi);
        auto pv_src = RiotLoop::make_pack_view(idx_range, vsrc);

        RiotLoop::inner(idx_range, [&](const auto kji) {
          for (int d = 0; d < 3; d++) {
            pv_src(ccbulk::momentum(d), kji) = 0.0;
          }
          pv_src(ccbulk::total_material_energy(), kji) = 0.0;
        });
        idx_range.TeamBarrier();

        // The inert material neither feels gravity nor does work against it
        const int nmat = vflx.GetSize(b, ccmat::rho());
        for (int m = 0; m < nmat; m++) {
          if (vflx(b, ccmat::rho(m)).sparse_id == inert) continue;

          // Momentum source: the material density times the centered potential gradient
          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, vflx, m);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real rho_m = std::max(pv_m(ccmat::rho(), kji), 0.0);
            for (int d = 0; d < ndim; d++) {
              pv_src(ccbulk::momentum(d), kji) -=
                  rho_m * (pv_phi(phi(), kji + off[d]) - pv_phi(phi(), kji - off[d])) /
                  (2.0 * dx[d]);
            }
          });
          idx_range.TeamBarrier();

          // Energy source: on each face, the material's mass flux times minus the
          // potential difference across the face, half to each of the two cells
          for (int d = 0; d < ndim; d++) {
            auto f = RiotLoop::make_flux_view(idx_range, vflx, X1DIR + d, ccmat::rho(m));
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real phic = pv_phi(phi(), kji);
              pv_src(ccbulk::total_material_energy(), kji) -=
                  (f(kji) * (phic - pv_phi(phi(), kji - off[d])) +
                   f(kji + off[d]) * (pv_phi(phi(), kji + off[d]) - phic)) /
                  (2.0 * dx[d]);
            });
            idx_range.TeamBarrier();
          }
        }
      });

  return TaskStatus::complete;
} // CalculateSelfGravitySource

} // namespace SelfGravity
