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
#ifndef RIOT_UTILS_SPARSE_UPDATE_HPP_
#define RIOT_UTILS_SPARSE_UPDATE_HPP_
// This file was made in part with generative AI.

#include <algorithm>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace sparse_update {

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

using dudt_pair_t = std::pair<MeshData<Real> *, std::vector<Uid_t>>;
using dudt_vec_t = std::vector<dudt_pair_t>;

namespace impl {
using PackView_t = Kokkos::View<SparsePack<> *>;

//----------------------------------------------------------------------------------------
//! \fn  auto sparse_update::impl::MakePairDescriptors
//! \brief
inline auto MakePairDescriptors(MeshData<Real> *md, const dudt_vec_t &dudt_array) {
  std::vector<SparsePack<>::Descriptor> descriptors;
  for (const auto &[smd, uids] : dudt_array) {
    if (smd == nullptr) continue;
    descriptors.emplace_back(riot::MakePackDescriptor(uids, md));
  }
  return descriptors;
}

//----------------------------------------------------------------------------------------
//! \fn  auto sparse_update::impl::MakeDudtPacks
//! \brief
inline auto MakeDudtPacks(MeshData<Real> *md, const dudt_vec_t &dudt_array) {
  auto descriptors = MakePairDescriptors(md, dudt_array);

  // Count active sources at runtime
  int num_dudt = 0;
  for (const auto &[smd, uids] : dudt_array) {
    if (smd != nullptr) num_dudt++;
  }

  // SparsePack derives from SparsePackBase, which has a `virtual
  // ~SparsePackBase()`. The default ViewOfViewAlloc<DevMemSpace>
  // allocator initializes the View, which makes Kokkos construct
  // *and* register an on-device destructor for each element. Running
  // a virtual ctor/dtor on device dispatches through a vtable pointer
  // that only exists on the host (and gets bit-copied here by
  // deep_copy), which segfaults on GPU. WithoutInitializing skips
  // both on-device construct and destruct, so the packs are only ever
  // touched through their non-virtual KOKKOS_INLINE_FUNCTION
  // accessors on device. Contrast VariablePack (mesh_data.hpp), which
  // uses the initializing allocator safely precisely because it is
  // non-polymorphic.
  PackView_t upacks(Kokkos::view_alloc(Kokkos::WithoutInitializing, "upacks"), num_dudt);
  PackView_t dudt_packs(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dudt_packs"),
                        num_dudt);

  // Host mirrors are ALWAYS a fresh, sequentially-initialized
  // allocation. We use create_mirror (not create_mirror_view): in a
  // host-only build DevMemSpace == HostSpace, and create_mirror_view
  // would then alias the device View itself -- i.e. alias the
  // uninitialized storage above -- so the fill assignments below
  // would run SparsePack::operator= over garbage inner
  // Views. create_mirror always allocates, and SequentialHostInit
  // default-constructs each pack on the host so the inner Views start
  // empty and the (reference-counted) assignment/destruction below is
  // well-defined.  deep_copy then bit-copies the fully-formed host
  // packs into device memory; it does not bump refcounts, and the
  // device View never destructs, so ownership stays entirely with
  // these host mirrors, which release it sequentially at scope exit.
  auto upacks_h =
      Kokkos::create_mirror(Kokkos::view_alloc(Kokkos::SequentialHostInit), upacks);
  auto dudt_packs_h =
      Kokkos::create_mirror(Kokkos::view_alloc(Kokkos::SequentialHostInit), dudt_packs);

  // Fill on host
  int idx = 0;
  for (const auto &[smd, uids] : dudt_array) {
    if (smd == nullptr) continue;
    auto &desc = descriptors[idx];
    upacks_h(idx) = riot::GetPack(desc, md);
    dudt_packs_h(idx) = riot::GetPack(desc, smd);
    idx++;
  }

  // Validation
  for (int i = 1; i < num_dudt; ++i) {
    PARTHENON_DEBUG_REQUIRE(upacks_h(i).GetNBlocks() == upacks_h(i - 1).GetNBlocks(),
                            "All packs must have the same number of active blocks");
  }
  const int nblocks = (num_dudt > 0) ? upacks_h(0).GetNBlocks() : 0;

  // Deep copy to device (only copies View metadata/handles, not underlying data)
  Kokkos::deep_copy(upacks, upacks_h);
  Kokkos::deep_copy(dudt_packs, dudt_packs_h);

  return std::make_tuple(upacks, dudt_packs, num_dudt, nblocks);
}

//----------------------------------------------------------------------------------------
//! \fn  void sparse_update::impl::SumTermsHelper
//! \brief Add the remaining dudt source terms into the state, one flat pack view per
//!        (pack, var). Pure same-cell accumulation.
template <typename IndexRangeType>
KOKKOS_INLINE_FUNCTION void
SumTermsHelper(const IndexRangeType &idx_range, const int b, const PackView_t upacks,
               const PackView_t dudt_packs, const int num_dudt, const Real weight = 1.0) {
  for (int l = 0; l < num_dudt; l++) {
    auto v1 = upacks(l);
    auto dv1 = dudt_packs(l);
    for (int n = v1.GetLowerBound(b); n <= v1.GetUpperBound(b); n++) {
      auto u = RiotLoop::make_var_view(idx_range, v1, n);
      auto dudt = RiotLoop::make_var_view(idx_range, dv1, n);
      RiotLoop::inner(idx_range, [&](auto kji) { u(kji) += weight * dudt(kji); });
    }
  }
}
} // namespace impl

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus sparse_update::UpdateToNextStage
//! \brief With kinematic advection, only KinematicAdvected fields are updated and no
//!        dudt sources are applied, freezing the bulk state.
inline TaskStatus UpdateToNextStage(MeshData<Real> *umd, MeshData<Real> *u0md,
                                    const Real gam0, const Real gam1, const Real beta_dt,
                                    const dudt_vec_t &dudt_array) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::PDOpt;
  using parthenon::variable_names::any;

  auto pm = umd->GetParentPointer();
  const int ndim = pm->ndim;
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);
  auto &riot_pkg = pm->packages.Get("riot");
  bool sparse_physics = riot_pkg->template Param<bool>("sparse_physics");
  const bool kinematic = riot_pkg->template Param<bool>("kinematic_advection");

  // Cell fields only: face fields with fluxes (the CT magnetic field) have node memory
  // layout and are updated separately (Hydro::UpdateFaceFields).
  std::vector<MetadataFlag> flags(
      {kinematic ? Metadata::GetUserFlag(riot::metadata::KinematicAdvected)
                 : Metadata::WithFluxes,
       Metadata::Cell});

  // make packs for the full state vector
  auto desc = parthenon::MakePackDescriptor<any>(umd, flags, {PDOpt::WithFluxes});
  auto v = riot::GetPack(desc, umd);
  const auto v0 = riot::GetPack(desc, u0md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  auto delta = riot::MakePack<ccbulk::cell_delta>(umd);

  auto pack_data = impl::MakeDudtPacks(umd, kinematic ? dudt_vec_t{} : dudt_array);
  auto upacks = std::get<0>(pack_data);
  auto dudt_packs = std::get<1>(pack_data);
  const int num_dudt = std::get<2>(pack_data);

  const int nblocks = v.GetNBlocks();
  using lt = RiotUtils::LoopType<>;
  using TE = parthenon::TopologicalElement;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, umd, TE::CC);

  // Memory offsets to the "plus" face neighbor in each direction (zero for collapsed
  // dimensions), used to reach the i+1/j+1/k+1 face flux via flat pack-view indexing.
  auto di = idx_space.GetDelta(X1DIR);
  auto dj = idx_space.GetDelta(X2DIR);
  auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto vd = RiotLoop::make_var_view(idx_range, delta, 0);

        // Fused flux divergence and time integration update, one field at a time. Flux
        // and state are accessed through flat pack views; geometry (face areas, cell
        // volume) needs (k, j, i), read once per cell from the cached coords.
        for (int n = v.GetLowerBound(b); n <= v.GetUpperBound(b); n++) {
          auto u = RiotLoop::make_var_view(idx_range, v, n);
          auto u0 = RiotLoop::make_var_view(idx_range, v0, n);
          auto f1 = RiotLoop::make_flux_view(idx_range, v, X1DIR, n);
          auto f2 = RiotLoop::make_flux_view(idx_range, v, X2DIR, n);
          auto f3 = RiotLoop::make_flux_view(idx_range, v, X3DIR, n);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const auto [k, j, i] = idx_range.GetKJI(kji);
            const Real ivol = 1.0 / coords.CellVolume(k, j, i);

            const Real flux_x_left =
                Kokkos::fma(coords.template FaceArea<X1DIR>(k, j, i), f1(kji), Real(0.0));
            const Real flux_x_right = Kokkos::fma(
                coords.template FaceArea<X1DIR>(k, j, i + 1), f1(kji + di), Real(0.0));
            const Real dudt_x = flux_x_left - flux_x_right;
            Real dudt_y = 0.0;
            if (multi_d) {
              const Real flux_y_left = Kokkos::fma(
                  coords.template FaceArea<X2DIR>(k, j, i), f2(kji), Real(0.0));
              const Real flux_y_right = Kokkos::fma(
                  coords.template FaceArea<X2DIR>(k, j + 1, i), f2(kji + dj), Real(0.0));
              dudt_y = flux_y_left - flux_y_right;
            }
            Real dudt = dudt_x + dudt_y;
            if (three_d) {
              const Real flux_z_left = Kokkos::fma(
                  coords.template FaceArea<X3DIR>(k, j, i), f3(kji), Real(0.0));
              const Real flux_z_right = Kokkos::fma(
                  coords.template FaceArea<X3DIR>(k + 1, j, i), f3(kji + dk), Real(0.0));
              dudt += flux_z_left - flux_z_right;
            }

            // the literature writes this as: u0 = gam0 u0 + gam1 u1 + beta dt dudt
            // here, u is u0, and u0 is u1 (which is just the initial state of u)
            u(kji) = gam0 * u(kji) + gam1 * u0(kji) + beta_dt * ivol * dudt;
          });
        }

        impl::SumTermsHelper(idx_range, b, upacks, dudt_packs, num_dudt, beta_dt);

        // fill in delta
        if (sparse_physics) {
          for (int n = v.GetLowerBound(b); n <= v.GetUpperBound(b); n++) {
            const Real mask = 1.0 * (n > v.GetLowerBound(b));
            auto u = RiotLoop::make_var_view(idx_range, v, n);
            auto u0 = RiotLoop::make_var_view(idx_range, v0, n);
            RiotLoop::inner(idx_range, [&](const auto kji) {
              vd(kji) =
                  mask * vd(kji) + std::abs(u(kji) - u0(kji)) / (std::abs(u(kji)) + 1.0);
            });
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus sparse_update::DeepCopyData
//! \brief Deep copy the fields selected by flags over the entire domain. TT selects the
//!        topological type (cell, face, edge or node); each of its elements is copied
//!        through a flat view of the whole allocation. The selected fields must share one
//!        memory layout (all Metadata::CellMemAligned or none).
//  TODO(@jdolence): do we really want this templated?
template <typename F, typename T,
          parthenon::TopologicalType TT = parthenon::TopologicalType::Cell>
TaskStatus DeepCopyData(std::vector<F> flags, T *to, T *from) {
  using TE = parthenon::TopologicalElement;
  constexpr bool cell = (TT == parthenon::TopologicalType::Cell);
  Kokkos::Profiling::pushRegion(cell ? "Task_DeepCopy" : "Task_DeepCopyNonCell");
  TE layout = TE::CC;
  if constexpr (!cell) {
    flags.push_back(parthenon::TopologicalTypeToMetaData(TT));
    int naligned = 0, nselected = 0;
    for (const auto &[id, m] : from->GetMeshPointer()->resolved_packages->AllFields()) {
      if (std::all_of(flags.begin(), flags.end(),
                      [&](const auto &f) { return m.IsSet(f); })) {
        nselected++;
        naligned += m.IsSet(Metadata::CellMemAligned);
      }
    }
    PARTHENON_REQUIRE(naligned == 0 || naligned == nselected,
                      "DeepCopyData fields must share one memory layout");
    layout = (naligned > 0) ? TE::CC : TE::NN;
  }

  auto desc = riot::MakePackDescriptor(flags, to);
  const auto &dst = riot::GetPack(desc, to);
  const auto &src = riot::GetPack(desc, from);

  const int nblocks = src.GetNBlocks();
  if (nblocks > 0 && src.GetMaxNumberOfVars() > 0) {
    using lt = RiotUtils::LoopType<>;
    auto idx_space =
        lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, from, layout, layout);

    // @NOTE(JMM): Tested against memcpy, Kokkos::deep_copy, Kokkos::local_deep_copy,
    // WeightedSumData. This was the fastest and most portable.
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          for (int var = src.GetLowerBound(b); var <= src.GetUpperBound(b); var++) {
            if constexpr (cell) {
              auto vsrc = RiotLoop::make_var_view(idx_range, src, var);
              auto vdst = RiotLoop::make_var_view(idx_range, dst, var);
              RiotLoop::inner(idx_range, [&](auto kji) { vdst(kji) = vsrc(kji); });
            } else {
              constexpr int nel = parthenon::NumberOfTopologicalElements(TT);
              for (int d = 0; d < nel; d++) {
                const TE te = parthenon::GetTopologicalElementInDir(TT, d);
                auto vsrc = RiotLoop::make_var_view(idx_range, src, te, var);
                auto vdst = RiotLoop::make_var_view(idx_range, dst, te, var);
                RiotLoop::inner(idx_range, [&](auto kji) { vdst(kji) = vsrc(kji); });
              }
            }
          }
        });
  }
  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus sparse_update::DeepCopyIndependentData
//! \brief
template <typename T>
TaskStatus DeepCopyIndependentData(T *to, T *from) {
  // return AverageIndependentData(to, from, 0);
  // Cell and face fields are copied separately, since they index different elements.
  DeepCopyData(std::vector<MetadataFlag>({Metadata::Independent, Metadata::Cell}), to,
               from);
  return DeepCopyData<MetadataFlag, T, parthenon::TopologicalType::Face>(
      {Metadata::Independent}, to, from);
}

} // namespace sparse_update

#endif // RIOT_UTILS_SPARSE_UPDATE_HPP_
