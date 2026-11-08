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

#include <algorithm>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "forest/mapped_radiation.hpp"
#include "riot_utils/riot_loops.hpp"

namespace RiotForest {

void MinCellWidth(parthenon::MeshData<Real> *md, Real &min_dx) {
  if constexpr (parthenon::IsMappedCoord()) {
    auto pm = md->GetParentPointer();
    const bool multid = pm->ndim > 1;
    const bool threed = pm->ndim > 2;
    static auto desc = parthenon::MakePackDescriptor(pm->resolved_packages.get(),
                                                     std::vector<std::string>{});
    auto vmesh = desc.GetPack(md);
    using rt = RiotFlatReduce::ReductionType<Kokkos::Min<Real>>;
    auto idx_space =
        rt::GetIndexSpace(parthenon::IndexDomain::interior, vmesh.GetNBlocks(), md);
    min_dx = rt::four_d(
        "RiotForest::MinCellWidth", idx_space,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldx) {
          auto &coords = vmesh.GetCoordinates(b);
          ldx = std::min(ldx, 1.0 / (1.0 / coords.CellWidth(X1DIR, k, j, i) +
                                     multid * 1.0 / coords.CellWidth(X2DIR, k, j, i) +
                                     threed * 1.0 / coords.CellWidth(X3DIR, k, j, i)));
        });
  }
}

} // namespace RiotForest
