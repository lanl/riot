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

#include "riot_utils/sparse_update.hpp"

namespace sparse_update {
namespace impl {

template <typename T, parthenon::TopologicalType TT>
void DeepCopyLayout(const std::vector<Uid_t> &uids,
                    const parthenon::TopologicalElement layout, T *to, T *from) {
  using TE = parthenon::TopologicalElement;
  auto desc = riot::MakePackDescriptor(uids, to);
  const auto &dst = desc.GetPack(to);
  const auto &src = desc.GetPack(from);

  const int nblocks = src.GetNBlocks();
  if (nblocks == 0 || src.GetMaxNumberOfVars() == 0) return;
  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, from, layout, layout);
  constexpr int nel = parthenon::NumberOfTopologicalElements(TT);

  // @NOTE(JMM): Tested against memcpy, Kokkos::deep_copy, Kokkos::local_deep_copy,
  // WeightedSumData. This was the fastest and most portable.
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        for (int var = src.GetLowerBound(b); var <= src.GetUpperBound(b); var++) {
          for (int d = 0; d < nel; d++) {
            const TE te = parthenon::GetTopologicalElementInDir(TT, d);
            auto vsrc = RiotLoop::make_var_view(idx_range, src, te, var);
            auto vdst = RiotLoop::make_var_view(idx_range, dst, te, var);
            RiotLoop::inner(idx_range, [&](auto kji) { vdst(kji) = vsrc(kji); });
          }
        }
      });
}

template void DeepCopyLayout<MeshData<Real>, parthenon::TopologicalType::Cell>(
    const std::vector<Uid_t> &, const parthenon::TopologicalElement, MeshData<Real> *,
    MeshData<Real> *);
template void DeepCopyLayout<MeshData<Real>, parthenon::TopologicalType::Face>(
    const std::vector<Uid_t> &, const parthenon::TopologicalElement, MeshData<Real> *,
    MeshData<Real> *);

} // namespace impl
} // namespace sparse_update
