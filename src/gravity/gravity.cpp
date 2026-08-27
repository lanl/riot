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

#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "gravity/gravity.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Gravity {

//----------------------------------------------------------------------------------------
//! \fn  void Gravity::Initialize
//! \brief Initializes the constant-acceleration gravity package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto gravity = std::make_shared<StateDescriptor>("gravity");
  Params &params = gravity->AllParams();

  // Gravity
  int gravity_dim =
      pin->GetOrAddInteger("gravity", "gravity_dim", X2DIR,
                           "Direction to apply gravity, if gravity is enabled");
  params.Add("gravity_dim", gravity_dim);
  Real gravity_g =
      pin->GetOrAddReal("gravity", "gravity_g", -9.799785002124e2,
                        "Gravitational constant g, only relevant if gravity is enabled");
  params.Add("gravity_g", gravity_g);

  // Source term dU/dt variables
  gravity->RegisterMeshDataSubset(
      "dudt", RiotUtils::MakePackageDudtRequirements(
                  {ccbulk::momentum::name(), ccbulk::total_material_energy::name()}));

  return gravity;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::CalculateGravitySource
//! \brief Calculates source terms associated with uniform gravitational acceleration
TaskStatus CalculateGravitySource(MeshData<Real> *state, MeshData<Real> *src) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  if (state->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = state->GetParentPointer();

  auto &grav_pkg = pm->packages.Get("gravity");
  auto gdim = grav_pkg->Param<int>("gravity_dim");
  auto gacc = grav_pkg->Param<Real>("gravity_g");

  // Pack up material state fields
  auto vstate = riot::MakePack<ccbulk::rho, ccbulk::velocity>(state);

  // Pack up fields for which this package produces source terms
  auto vsrc = riot::MakePack<ccbulk::momentum, ccbulk::total_material_energy>(src);
  if ((vstate.GetNBlocks() != vsrc.GetNBlocks()) || (vstate.GetNBlocks() == 0))
    return TaskStatus::complete;

  // loop over block and compute source term
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, vstate.GetNBlocks(), state,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv_state = RiotLoop::make_pack_view(idx_range, vstate);
        auto pv_src = RiotLoop::make_pack_view(idx_range, vsrc);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv_src(ccbulk::momentum(gdim), kji) = pv_state(ccbulk::rho(), kji) * gacc;
          pv_src(ccbulk::total_material_energy(), kji) =
              pv_state(ccbulk::rho(), kji) * pv_state(ccbulk::velocity(gdim), kji) * gacc;
        });
      });

  return TaskStatus::complete;

} // CalculateGravitySource

} // namespace Gravity
