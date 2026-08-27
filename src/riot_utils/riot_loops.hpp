//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef RIOT_UTILS_RIOT_LOOPS_HPP_
#define RIOT_UTILS_RIOT_LOOPS_HPP_
// This file was made in part with generative AI.

#include <string>

#include <loop_abstraction/loop_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace RiotLoop = parthenon::loop_abstraction;
enum class LoopConstraint { NoGhost, DifferentMemSpaces, SingleBlock };
template <LoopConstraint Needle, LoopConstraint... Haystack>
inline constexpr bool has_loop_constraint = ((Needle == Haystack) || ...);

namespace RiotUtils {

//----------------------------------------------------------------------------------------
//! \namespace RiotUtils::halo
//! \brief Symmetric (+/-1) inner-range halo tags for centered/diffusive stencils.
//!
//! The loop-abstraction ships only single-offset halo tags (halo::minus_i_t etc., each
//! npoints == 2). A three-point stencil d/dx(A d/dx(q)) needs the neighbor on *both*
//! sides, so we declare symmetric {-1, 0, +1} halos here. Each must satisfy the halo
//! contract (a unique identity offset, offsets sorted strictly ascending by
//! (dk, dj, di)); the loop abstraction static_asserts this. Offsets n = 0,1,2 map to
//! -1, 0, +1 so the identity sits in the middle and the ordering is monotone.
namespace halo {
struct pm_i_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n - 1; }
};
struct pm_j_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return n - 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};
struct pm_k_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return n - 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};
} // namespace halo

// Right-handed coordinate system representing directions normal and tangent to a face
// NOTE(): Directional basis for a sweep/face direction DIR (X1DIR/X2DIR/X3DIR). Holds the
// per-axis unit offsets (from IndexSpace::GetDelta, which are zero for collapsed
// dimensions) and exposes them permuted into (normal, transverse-a, transverse-b)
// order. The transverse pair is in CYCLIC (right-handed) order: for normal
// component c, transverse components are (c+1)%3 then (c+2)%3. This is the physically
// meaningful convention (curl/EMF signs for CT-MHD); it is also safe for symmetric
// reductions like GetVdiff's transverse min.
template <parthenon::CoordinateDirection DIR, typename Offset>
struct DirBasis {
  // Component indices (0/1/2) for the normal and the two cyclic transverse dirs.
  static constexpr int nc = static_cast<int>(DIR) - 1; // X1DIR->0, X2DIR->1, X3DIR->2
  static constexpr int tac = (nc + 1) % 3;
  static constexpr int tbc = (nc + 2) % 3;

  Offset off[3]; // off[c] = unit offset along axis of component c (i=0, j=1, k=2)

  KOKKOS_INLINE_FUNCTION
  DirBasis(Offset di, Offset dj, Offset dk) : off{di, dj, dk} {}

  KOKKOS_FORCEINLINE_FUNCTION const Offset &normal() const { return off[nc]; }
  KOKKOS_FORCEINLINE_FUNCTION const Offset &trans_a() const { return off[tac]; }
  KOKKOS_FORCEINLINE_FUNCTION const Offset &trans_b() const { return off[tbc]; }
};

// Build a DirBasis<DIR> from an index space (pulls the three GetDelta offsets).
template <parthenon::CoordinateDirection DIR, typename IdxSpace>
auto MakeDirBasis(IdxSpace &idx_space) {
  auto di = idx_space.GetDelta(parthenon::X1DIR);
  return DirBasis<DIR, decltype(di)>(di, idx_space.GetDelta(parthenon::X2DIR),
                                     idx_space.GetDelta(parthenon::X3DIR));
}

//----------------------------------------------------------------------------------------
//! \brief True if an offset (MemoryOffset or Index3) is the zero offset, which the
//!        IndexSpace returns for a collapsed (size-1) dimension.
KOKKOS_FORCEINLINE_FUNCTION bool IsZeroOffset(const RiotLoop::MemoryOffset &o) {
  return o.dk == 0 && o.dj == 0 && o.di == 0;
}
KOKKOS_FORCEINLINE_FUNCTION bool IsZeroOffset(const RiotLoop::Index3 &o) {
  return o.k == 0 && o.j == 0 && o.i == 0;
}

template <LoopConstraint... Cs>
constexpr RiotLoop::loop_tag GetLoopTag() {
  // A SingleBlock loop has just one block, so the host bvoi shape -- one team per block,
  // whole-block scratch with vectorized inner sweeps -- would launch a single team and
  // leave the machine idle. Use the point-wise boiv shape on both backends so the one
  // block's cells parallelize across threads.
  if constexpr (has_loop_constraint<LoopConstraint::SingleBlock, Cs...>) {
    return RiotLoop::loop_tag::boiv;
  } else {
    return (RiotLoop::default_loop_backend_v == RiotLoop::loop_backend::kokkos)
               ? RiotLoop::loop_tag::boiv
               : RiotLoop::loop_tag::bvoi;
  }
}

template <LoopConstraint... Cs>
constexpr RiotLoop::inner_tag GetInnerTag() {
  constexpr auto ltag = GetLoopTag<Cs...>();
  if constexpr (has_loop_constraint<LoopConstraint::DifferentMemSpaces, Cs...>) {
    return RiotLoop::inner_tag::logical_coords;
  } else if constexpr (has_loop_constraint<LoopConstraint::NoGhost, Cs...>) {
    return RiotLoop::inner_tag::logical_flat;
  } else {
    if constexpr (ltag == RiotLoop::loop_tag::boiv)
      return RiotLoop::inner_tag::logical_flat;
    return RiotLoop::inner_tag::memory;
  }
}

template <LoopConstraint... Cs, class MeshDataOrMeshBlockData>
inline auto RiotIndexSpace(parthenon::IndexDomain domain, int halo, int nblocks,
                           const MeshDataOrMeshBlockData *md,
                           parthenon::TopologicalElement domain_te) {
  // Chunk one (halo-extended) ij-plane per inner iteration. The loop abstraction
  // resolves this against the extended indexer, so chunks stay plane-aligned in
  // every sweep direction (a bare cell count would misalign once a halo is added).
  const RiotLoop::NInner ninner = RiotLoop::chunk_shape::ij_slab;

  // Select the loop tag by backend: on device (GPU) use boiv -- point-wise
  // parallelism, one cell per thread, which is the performant GPU shape. On host
  // use bvoi -- whole-block scratch with vectorized inner sweeps. (Hierarchical
  // bvoi/bovi on GPU is sub-optimal in our experience.)
  constexpr RiotLoop::loop_tag ltag = GetLoopTag<Cs...>();

  // Inner tag selection depends on what the loop allows
  constexpr RiotLoop::inner_tag itag = GetInnerTag<Cs...>();
  return RiotLoop::IndexSpace<ltag, itag>(ninner, domain, halo, nblocks, md, domain_te);
}

template <LoopConstraint... Cs>
struct LoopType {
  using idx_space_t = RiotLoop::IndexSpace<GetLoopTag<Cs...>(), GetInnerTag<Cs...>()>;
  using idx_range_t = RiotLoop::InnerIndexRange<idx_space_t>;
  using inner_idx_t = typename idx_space_t::inner_idx_t;

  template <class MeshDataOrMeshBlockData>
  static idx_space_t GetIndexSpace(parthenon::IndexDomain domain, int halo, int nblocks,
                                   const MeshDataOrMeshBlockData *md,
                                   parthenon::TopologicalElement domain_te) {
    return RiotIndexSpace<Cs...>(domain, halo, nblocks, md, domain_te);
  }
};

//----------------------------------------------------------------------------------------
//! \struct  RiotUtils::ReductionType
//! \brief Reduction analog of LoopType: bundles the loop-tag/inner-tag selection with a
//! Kokkos reducer (e.g. Kokkos::Min<Real>) so callers get a reduction IndexSpace ready
//! for RiotLoop::outer_reduce/inner_reduce. Kept separate from LoopType so a reduction
//! can pick a loop tag independent of the plain par_for path if we later find a different
//! shape is faster for reductions. idx_range_t names the outer-body parameter and value_t
//! is the reduced/returned type.
template <class Reducer, LoopConstraint... Cs>
struct ReductionType {
  using idx_space_t = RiotLoop::IndexSpace<GetLoopTag<Cs...>(), GetInnerTag<Cs...>(),
                                           RiotLoop::default_loop_backend_v, Reducer>;
  using idx_range_t = typename idx_space_t::idx_range_t;
  using value_t = typename idx_space_t::value_t;

  // Standard (domain, halo) mesh reduction space. halo != 0 insets/extends the base
  // logical domain (halo < 0 insets); the range stays halo-free so inner_reduce accepts
  // it.
  template <class MeshDataOrMeshBlockData>
  static idx_space_t GetIndexSpace(parthenon::IndexDomain domain, int halo, int nblocks,
                                   const MeshDataOrMeshBlockData *md,
                                   parthenon::TopologicalElement domain_te) {
    return RiotIndexSpace<Cs...>(domain, halo, nblocks, md, domain_te)
        .template WithReducer<Reducer>();
  }

  // Explicit-logical-ranges reduction space for a non-standard iteration box (e.g. a
  // leading dimension that is not blocks). nouter fills the block slot; kb/jb/ib are the
  // logical cells to visit. The memory extent stays fixed by Parthenon (entire bounds for
  // memory_te's layout); the caller only selects CC vs NN.
  template <class MeshDataOrMeshBlockData>
  static idx_space_t GetIndexSpace(
      int nouter, const parthenon::IndexRange &kb, const parthenon::IndexRange &jb,
      const parthenon::IndexRange &ib, const MeshDataOrMeshBlockData *md,
      parthenon::TopologicalElement memory_te = parthenon::TopologicalElement::CC) {
    using base_t = RiotLoop::IndexSpace<GetLoopTag<Cs...>(), GetInnerTag<Cs...>()>;
    return base_t(nouter, kb, jb, ib, md, memory_te).template WithReducer<Reducer>();
  }
};

//----------------------------------------------------------------------------------------
//! \struct  ParticleIndexSpace
//! \brief
struct ParticleIndexSpace {
  int begin;
  int end;
};

//----------------------------------------------------------------------------------------
//! \fn ParticleIndexSpace GetParticleIndexSpace(const Pack &pack)
//! \brief
template <typename Pack>
ParticleIndexSpace GetParticleIndexSpace(const Pack &pack) {
  return {0, pack.GetMaxFlatIndex()};
}

} // namespace RiotUtils

namespace RiotParticleLoop {

//----------------------------------------------------------------------------------------
//! \fn void particles(const ParticleIndexSpace &space, Functor &&func)
//! \brief
template <typename Functor>
void flat(const RiotUtils::ParticleIndexSpace &space, Functor func) {
  parthenon::par_for(DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(),
                     space.begin, space.end, func);
}
} // namespace RiotParticleLoop

//----------------------------------------------------------------------------------------
//! \namespace RiotFlatLoop
//! \brief Lightweight, named entry points (and their index space) for flat
//! multidimensional par_for loops. Like RiotParticleLoop, these carry no
//! performance-portability machinery -- each maps identically to the parthenon::par_for
//! it replaces -- they just bundle the mdrange bounds so call sites read uniformly and
//! stop repeating the DEFAULT_LOOP_PATTERN / DevExecSpace() boilerplate. The loop name
//! counts the total loop indices: four_d over (b, k, j, i), five_d over (b, n, k, j, i)
//! where n is a flattened leading dimension.
namespace RiotFlatLoop {

//----------------------------------------------------------------------------------------
//! \struct  RiotFlatLoop::IndexSpace
//! \brief Plain (block, [lead,] k, j, i) iteration space. nlead is an optional flattened
//! leading dimension in front of (k, j, i); nlead <= 0 means there is no leading
//! dimension.
struct IndexSpace {
  int nblocks;
  int nlead;
  parthenon::IndexRange kb, jb, ib;
};

//----------------------------------------------------------------------------------------
//! \fn  IndexSpace RiotFlatLoop::GetIndexSpace
//! \brief Build an IndexSpace from a domain. Argument order mirrors RiotReduce's
//! GetIndexSpace / the loop abstraction's (domain, ..., nblocks, md, te): nblocks stays
//! an explicit count because call sites loop over the pack's block count, which need not
//! equal md->NumBlocks(). The nlead overload prepends a flattened leading dimension of
//! count nlead.
template <class MD>
inline IndexSpace
GetIndexSpace(parthenon::IndexDomain domain, const int nblocks, const MD *md,
              parthenon::TopologicalElement te = parthenon::TopologicalElement::CC) {
  return {nblocks, 0, md->GetBoundsK(domain, te), md->GetBoundsJ(domain, te),
          md->GetBoundsI(domain, te)};
}
template <class MD>
inline IndexSpace
GetIndexSpace(parthenon::IndexDomain domain, const int nblocks, const int nlead,
              const MD *md,
              parthenon::TopologicalElement te = parthenon::TopologicalElement::CC) {
  return {nblocks, nlead, md->GetBoundsK(domain, te), md->GetBoundsJ(domain, te),
          md->GetBoundsI(domain, te)};
}

//----------------------------------------------------------------------------------------
//! \fn  IndexSpace RiotFlatLoop::GetIndexSpace
//! \brief Build an IndexSpace directly from bounds, for loops whose iteration space is
//! not a standard (block, IndexDomain) box -- e.g. a boundary-extended stencil range. The
//! nlead overload prepends a flattened leading dimension of count nlead. For a
//! single-block loop, nblocks is 1.
inline IndexSpace GetIndexSpace(const int nblocks, const parthenon::IndexRange &kb,
                                const parthenon::IndexRange &jb,
                                const parthenon::IndexRange &ib) {
  return {nblocks, 0, kb, jb, ib};
}
inline IndexSpace GetIndexSpace(const int nblocks, const int nlead,
                                const parthenon::IndexRange &kb,
                                const parthenon::IndexRange &jb,
                                const parthenon::IndexRange &ib) {
  return {nblocks, nlead, kb, jb, ib};
}

//----------------------------------------------------------------------------------------
//! \fn void RiotFlatLoop::four_d
//! \brief Mesh-wide (b, k, j, i) loop. The functor takes (b, k, j, i).
template <typename Functor>
void four_d(const std::string &name, const IndexSpace &space, Functor func) {
  parthenon::par_for(DEFAULT_LOOP_PATTERN, name, DevExecSpace(), 0, space.nblocks - 1,
                     space.kb.s, space.kb.e, space.jb.s, space.jb.e, space.ib.s,
                     space.ib.e, func);
}

//----------------------------------------------------------------------------------------
//! \fn void RiotFlatLoop::five_d
//! \brief Mesh-wide (b, n, k, j, i) loop, where n is the flattened leading dimension in
//! [0, nlead). The functor takes (b, n, k, j, i).
template <typename Functor>
void five_d(const std::string &name, const IndexSpace &space, Functor func) {
  parthenon::par_for(DEFAULT_LOOP_PATTERN, name, DevExecSpace(), 0, space.nblocks - 1, 0,
                     space.nlead - 1, space.kb.s, space.kb.e, space.jb.s, space.jb.e,
                     space.ib.s, space.ib.e, func);
}
} // namespace RiotFlatLoop

//----------------------------------------------------------------------------------------
//! \namespace RiotFlatReduce
//! \brief Reduction analog of RiotFlatLoop, and the flat counterpart to
//! RiotUtils::ReductionType. Like the non-flat reductions, the user picks a Kokkos
//! reducer once in a `using rt = ReductionType<...>` alias, builds an index space with
//! rt::GetIndexSpace(...), and gets the reduced value back from the reduce call. The
//! differences from the non-flat path are only the essential ones: the functor is flat
//! (b, [n,] k, j, i, accumulator&) rather than nested outer_reduce/inner_reduce, and the
//! reduce verb is named for the total loop-index count (four_d / five_d) to match
//! RiotFlatLoop.
//!
//! The reducer is passed as a *fully-specified* type (e.g. Kokkos::Min<Real>,
//! Kokkos::MinMax<Real>, or RiotUtils::GlobalSum<Real, Kokkos::HostSpace, 2>); the
//! reduced value type is deduced from Reducer::value_type, so array- and pair-valued
//! reductions need no special-casing. As with parthenon::par_reduce, the caller is
//! responsible for any Kokkos::fence() needed before reading a host-space result.
namespace RiotFlatReduce {

template <class Reducer>
struct ReductionType {
  //! The reduced/returned type (Real for Min/Max/Sum, a {min,max} pair for MinMax, an
  //! array_type for GlobalSum, ...). Names the accumulator the functor receives.
  using value_t = typename Reducer::value_type;

  //! Index-space factories. The reducer does not affect the iteration bounds, so these
  //! simply delegate to RiotFlatLoop::GetIndexSpace; they are provided so call sites read
  //! like the non-flat rt::GetIndexSpace idiom. See RiotFlatLoop::GetIndexSpace for the
  //! argument conventions (nblocks/nlead are counts; kb/jb/ib are inclusive).
  template <class MD>
  static RiotFlatLoop::IndexSpace
  GetIndexSpace(parthenon::IndexDomain domain, const int nblocks, const MD *md,
                parthenon::TopologicalElement te = parthenon::TopologicalElement::CC) {
    return RiotFlatLoop::GetIndexSpace(domain, nblocks, md, te);
  }
  template <class MD>
  static RiotFlatLoop::IndexSpace
  GetIndexSpace(parthenon::IndexDomain domain, const int nblocks, const int nlead,
                const MD *md,
                parthenon::TopologicalElement te = parthenon::TopologicalElement::CC) {
    return RiotFlatLoop::GetIndexSpace(domain, nblocks, nlead, md, te);
  }
  static RiotFlatLoop::IndexSpace GetIndexSpace(const int nblocks,
                                                const parthenon::IndexRange &kb,
                                                const parthenon::IndexRange &jb,
                                                const parthenon::IndexRange &ib) {
    return RiotFlatLoop::GetIndexSpace(nblocks, kb, jb, ib);
  }
  static RiotFlatLoop::IndexSpace GetIndexSpace(const int nblocks, const int nlead,
                                                const parthenon::IndexRange &kb,
                                                const parthenon::IndexRange &jb,
                                                const parthenon::IndexRange &ib) {
    return RiotFlatLoop::GetIndexSpace(nblocks, nlead, kb, jb, ib);
  }

  //! Mesh-wide (b, k, j, i) reduction. The functor takes (b, k, j, i, value_t&); the
  //! seeded accumulator is returned.
  template <typename Functor>
  static value_t four_d(const std::string &name, const RiotFlatLoop::IndexSpace &space,
                        const Functor &f) {
    value_t result;
    parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, name, DevExecSpace(), 0,
                          space.nblocks - 1, space.kb.s, space.kb.e, space.jb.s,
                          space.jb.e, space.ib.s, space.ib.e, f, Reducer(result));
    return result;
  }

  //! Mesh-wide (b, n, k, j, i) reduction, n the flattened leading dimension in [0,
  //! nlead). The functor takes (b, n, k, j, i, value_t&).
  template <typename Functor>
  static value_t five_d(const std::string &name, const RiotFlatLoop::IndexSpace &space,
                        const Functor &f) {
    value_t result;
    parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, name, DevExecSpace(), 0,
                          space.nblocks - 1, 0, space.nlead - 1, space.kb.s, space.kb.e,
                          space.jb.s, space.jb.e, space.ib.s, space.ib.e, f,
                          Reducer(result));
    return result;
  }
};
} // namespace RiotFlatReduce

#endif // RIOT_UTILS_RIOT_LOOPS_HPP_
