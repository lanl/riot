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
#ifndef MICROPHYSICS_PTE_CLOSURE_HPP_
#define MICROPHYSICS_PTE_CLOSURE_HPP_
// This file was made in part with generative AI.

// singularity includes
#include <ports-of-call/portability.hpp>

// parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include <memory>
#include <string>
#include <utility>

#include <iostream> // debug

namespace Closure {

void ApplyMixedCellClosure(MeshData<Real> *md, IndexDomain domain);
void ApplyIdealGasClosure(MeshData<Real> *md, IndexDomain domain);

namespace closure_impl {
using parthenon::ParArray1D;

// EOS indexer for the per-cell PTE solve, driven by raw index arrays living on the stack
// (the local pte2slice/eos_map arrays). PTE slice index m maps through pte2slice to a
// material index, then through eos_map to the EOS array slot.
template <typename T>
struct LocalEosIndexer {
  KOKKOS_INLINE_FUNCTION
  LocalEosIndexer(const int *pte2slice, const int *eos_map, const ParArray1D<T> &eos)
      : pte2slice_(pte2slice), eos_map_(eos_map), eos_(eos) {}
  KOKKOS_INLINE_FUNCTION
  auto &operator[](const int m) const { return eos_(eos_map_[pte2slice_[m]]); }
  const int *pte2slice_;
  const int *eos_map_;
  const ParArray1D<T> &eos_;
};

} // namespace closure_impl
} // namespace Closure

#endif // MICROPHYSICS_PTE_CLOSURE_HPP_
