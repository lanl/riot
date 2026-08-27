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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "parthenon/driver.hpp"
#include <parthenon/package.hpp>

#include "levelsets.hpp"
#include "reconstruction/reconstruction.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

namespace Levelsets {

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Levelsets::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ls = Levelsets;
  auto levelsets = std::make_shared<StateDescriptor>("levelsets");
  Params &params = levelsets->AllParams();

  // add levelset field - this currently supports only one levelset
  Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                         Metadata::FillGhost, Metadata::Advected, Metadata::WithFluxes});
  levelsets->AddField<ls::levelset>(m);

  // add levelset0 field - sloppy for now
  m = Metadata(
      {Metadata::OneCopy, Metadata::Cell, Metadata::Independent, Metadata::Intensive});
  levelsets->AddField<ls::levelset0>(m);

  // add levelset dudt field - sloppy for now
  m = Metadata(
      {Metadata::OneCopy, Metadata::Cell, Metadata::Independent, Metadata::Intensive});
  levelsets->AddField<ls::dudt_reinitialize>(m);

  auto sharp_mat = pin->GetOrAddInteger("levelsets", "sharp_mat", 0,
                                        "Which material is the sharp one?");
  params.Add("sharp_mat", sharp_mat);

  auto reinit_modcyc =
      pin->GetOrAddInteger("levelsets", "reinit_modcyc", 25,
                           "How often to reinitialize the level set in number of cycles");
  params.Add("reinit_modcyc", reinit_modcyc);

  auto reinit_width = pin->GetOrAddInteger(
      "levelsets", "reinit_width", 4,
      "Half-width of the region bounding the material interface (in units of dx) for "
      "which the level set should be a signed distance function. Outside of this region, "
      "the levelset assumes a constant value in space.");
  params.Add("reinit_width", reinit_width);

  // how many pseudo-timesteps to reinitialize level set?
  // The levelset is reinitialized to be a signed distance function by solving
  // a hyperbolic version of the Eikonal equation using a pseudo-time-stepping
  // method that propagates the correction approximately one cell per pseudo
  // time step (see, e.g., Sussman & Fatemi // SIAM J. SCI. COMPUT. Vol. 20,
  // No. 4, pp. 1165–1191.
  // Therefore, the number of pseudo time steps should ideally be greater than
  // the width of the region in which the levelset is a signed distance
  // function in units of dx.
  auto reinit_nstep =
      pin->GetOrAddInteger("levelsets", "reinit_nstep", 15,
                           "How many pseudo-timesteps to reinitialize the level set?");
  params.Add("reinit_nstep", reinit_nstep);

  levelsets->UserWorkBeforeLoopMesh = InitializeSignedDistance;

  return levelsets;
}

//----------------------------------------------------------------------------------------
//! \fn  void Levelsets::InitializeSignedDistance
//! \brief
void InitializeSignedDistance(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm) {
  Real dt = 0.;
  if (parthenon::Globals::is_restart) return;
  Reinitialize(pm, tm, dt).Execute();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Levelsets::StashZerothStep
//! \brief
TaskStatus StashZerothStep(MeshData<Real> *md) {
  namespace ls = Levelsets;

  auto v = riot::MakePack<ls::levelset, ls::levelset0>(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  // stash 0th step
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv(ls::levelset0(), kji) = pv(ls::levelset(), kji);
        });
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Levelsets::Reinitialize
//! \brief
TaskCollection Reinitialize(Mesh *pm, parthenon::SimTime &tm, Real dt) {
  using namespace parthenon::Update;
  using parthenon::PackIdx;
  TaskCollection tc;
  TaskID none(0);

  auto &levelset_pkg = pm->packages.Get("levelsets");
  Params &params = levelset_pkg->AllParams();

  const auto &modcyc = params.Get<int>("reinit_modcyc");
  if (tm.ncycle % std::max(modcyc, 1) > 0) return tc;

  auto &md = pm->mesh_data.Get();

  // set pseudo time step for reinitialization
  if (md->NumBlocks() == 0) return tc;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  double dx = pmb->coords.Dxc<X1DIR>(0, 0, 0);
  Real dt_reinit = 0.5 * dx;

  const int &num_steps = params.Get<int>("reinit_nstep");
  const int &width = params.Get<int>("reinit_width");

  const int num_partitions = pm->DefaultNumPartitions();

  // stash zero step level set field needed for RHS
  // TODO: @swj should put this into a function and put it inside the task
  // region
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &mdpart = pm->mesh_data.GetOrAdd("base", i);
    auto stash = tl.AddTask(none, StashZerothStep, mdpart.get());

    for (int pseudostep = 0; pseudostep < num_steps; pseudostep++) {
      constexpr auto any = parthenon::BoundaryType::any;

      auto rcvbuf = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, md);

      // do a reinitialization step
      auto &mdpart = pm->mesh_data.GetOrAdd("base", i);
      auto reinitializestep =
          tl.AddTask(stash, ReinitializeStep, mdpart.get(), dt_reinit, width);

      parthenon::AddBoundaryExchangeTasks(reinitializestep, tl, md, pm->multilevel);
    }

  } // end of pseudostep loop

  return tc;
} // Reinitialize task collection

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Levelsets::ReinitializeStep
//! \brief
//! TODO @swj: this should be anchoring but may not be
//! TODO @swj: consider replacing with fast marching method
TaskStatus ReinitializeStep(MeshData<Real> *mc, Real dt, int width) {
  namespace ls = Levelsets;

  const auto pmesh = mc->GetParentPointer();

  auto v = riot::MakePack<ls::levelset, ls::levelset0, ls::dudt_reinitialize>(mc);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  auto pmb = mc->GetBlockData(0)->GetBlockPointer();
  const double dx = pmb->coords.Dxc<X1DIR>(0, 0, 0);
  const double sqrtndimdx = std::sqrt(pmesh->ndim) * dx;

  // Get the re-initialization RHS. The upwind reconstruction reads a 5-point stencil in
  // each direction and needs cell-center coordinates, so this loop uses (k, j, i).
  using ltng = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  auto idx_space_rhs = ltng::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), mc,
                                           parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space_rhs, KOKKOS_LAMBDA(const ltng::idx_range_t &idx_range, const int b) {
        const auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);
          double gradpmag = 0.;
          double gradmmag = 0.;
          std::array<Real, 5> x, ux;
          std::array<Real, 4> res;
          for (int dim = 0; dim < pmesh->ndim; dim++) {
            for (int idx = 0; idx < 5; idx++) {
              if (dim == 0) {
                x[idx] = coords.Xc<1>(i - 2 + idx);
                ux[idx] = pv(ls::levelset(), k, j, i - 2 + idx);
              } else if (dim == 1) {
                x[idx] = coords.Xc<2>(j - 2 + idx);
                ux[idx] = pv(ls::levelset(), k, j - 2 + idx, i);
              } else if (dim == 2) {
                x[idx] = coords.Xc<3>(k - 2 + idx);
                ux[idx] = pv(ls::levelset(), k - 2 + idx, j, i);
              }
            }
            res = upwind(x, ux);
            const double am = res[0];
            const double ap = res[1];
            const double bm = res[2];
            const double bp = res[3];
            gradpmag += std::max(ap * ap, bm * bm);
            gradmmag += std::max(am * am, bp * bp);
          }
          gradpmag = RiotUtils::SafeSqrt(gradpmag);
          gradmmag = RiotUtils::SafeSqrt(gradmmag);

          const Real sgn = signh(pv, ls::levelset0(), k, j, i, dx);

          const int a = (pv(ls::levelset0(), k, j, i) > 0.);
          const double grad = (a * gradpmag + (1 - a) * gradmmag);
          pv(ls::dudt_reinitialize(), k, j, i) = sgn * (1. - grad);
          pv(ls::dudt_reinitialize(), k, j, i) *=
              (std::abs(pv(ls::levelset0(), k, j, i)) > sqrtndimdx);
        });
      });

  // Push the re-initialization (pure same-cell update: flat indexing).
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), mc,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv(ls::levelset(), kji) += pv(ls::dudt_reinitialize(), kji) * dt;
          pv(ls::levelset(), kji) =
              std::copysign(std::min(std::abs(pv(ls::levelset(), kji)), width * dx),
                            pv(ls::levelset0(), kji));
        });
      });

  return TaskStatus::complete;
}

} // namespace Levelsets
