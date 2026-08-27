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
#ifndef RIOT_UTILS_RIOT_UTILS_HPP_
#define RIOT_UTILS_RIOT_UTILS_HPP_
// This file was made in part with generative AI.

#include <algorithm>
#include <ranges>
#include <string>
#include <utility>

#include "kokkos_abstraction.hpp"
#include <globals.hpp>
#include <parthenon/package.hpp>

// TODO(JMM): This is a circular dependency waiting to happen
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace RiotUtils {

using VarNamePairList = std::pair<std::vector<std::string>, std::vector<std::string>>;

void ToUpper(std::string &s);
void ToLower(std::string &s);

template <typename T, typename... Extra>
auto VectorToViewPair(const std::vector<T, Extra...> &v, const std::string &name) {
  auto device_data = parthenon::ParArray1D<T>(name, v.size());
  auto host_data = Kokkos::create_mirror_view(Kokkos::HostSpace(), device_data);
  std::ranges::copy(v, host_data.data());
  Kokkos::deep_copy(device_data, host_data);
  return std::make_pair(device_data, host_data);
}
template <typename Dst_t, typename T, typename... Extra>
void DeepCopyVectorToDevice(Dst_t &dst, std::vector<T, Extra...> &src) {
  using UnManaged_t = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using mirror_t = Kokkos::View<T *, Kokkos::HostSpace, UnManaged_t>;
  auto mirror = mirror_t(src.data(), src.size());
  Kokkos::deep_copy(dst, mirror);
}
template <typename T, typename... Extra>
auto VectorToDevice(std::vector<T, Extra...> &v, const std::string &name) {
  auto device_data = parthenon::ParArray1D<T>(name, v.size());
  DeepCopyVectorToDevice(device_data, v);
  return device_data;
}

//----------------------------------------------------------------------------------------
// NOTE(@pdmullen): The following is taken from the Kokkos wiki for custom reductions:
// https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/
template <class ScalarType, int N>
struct array_type {
  ScalarType my_array[N];

  KOKKOS_INLINE_FUNCTION
  array_type() { init(); }

  KOKKOS_INLINE_FUNCTION
  array_type(const array_type &rhs) {
    for (int i = 0; i < N; i++) {
      my_array[i] = rhs.my_array[i];
    }
  }

  // initialize my_array to 0
  KOKKOS_INLINE_FUNCTION
  void init() {
    for (int i = 0; i < N; i++) {
      my_array[i] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  array_type &operator+=(const array_type &src) {
    for (int i = 0; i < N; i++) {
      my_array[i] += src.my_array[i];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile array_type &src) volatile {
    for (int i = 0; i < N; i++) {
      my_array[i] += src.my_array[i];
    }
  }
};

template <class T, class Space, int N>
struct GlobalSum {
 public:
  // Required
  typedef GlobalSum reducer;
  typedef array_type<T, N> value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

 private:
  value_type &value;

 public:
  KOKKOS_INLINE_FUNCTION
  GlobalSum(value_type &value_) : value(value_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const { dest += src; }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type &dest, const volatile value_type &src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const { val.init(); }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const { return value; }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value, 1); }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }
};

inline void BcastBytes(void *buffer, std::size_t nbytes, int root, MPI_Comm comm) {
  auto *p = static_cast<unsigned char *>(buffer);

  while (nbytes > 0) {
    int chunk = static_cast<int>(std::min<std::size_t>(nbytes, INT_MAX));

    MPI_Bcast(p, chunk, MPI_BYTE, root, comm);

    p += chunk;
    nbytes -= chunk;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotUtils::SafeSqrt
//! \brief
KOKKOS_INLINE_FUNCTION
Real SafeSqrt(const Real &a) { return std::sqrt((a > 0) * std::abs(a)); };

// For a flag collection, generate the list of vars associated with each other.
// Returns a std::pair<std::vector<std::string>, std::vector<std::string>>
// The vars in the second vector are associated with the vars in the first.
// The associated machinery is used to tie primitive vars to conserved for anonymous
// advection
VarNamePairList GetAssociatedVars(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Metadata::FlagCollection &flags);
VarNamePairList GetAssociatedVars(parthenon::MeshBlockData<parthenon::Real> *bd,
                                  const parthenon::Metadata::FlagCollection &flags);

std::vector<std::string> GetUnsplitVarNames(Mesh *pmesh);

//----------------------------------------------------------------------------------------
//! \fn  parthenon::MeshDataDescriptor RiotUtils::MakePackageDudtRequirements
//! \brief Build a source-term (dU/dt) MeshDataDescriptor for a package.
//!
//! Always carries the block-active flag (cell_delta) so that riot::GetPack indexes
//! active blocks consistently when summing right-hand sides. cell_delta is only added
//! if it isn't already requested, so it is never duplicated.
inline parthenon::MeshDataDescriptor
MakePackageDudtRequirements(std::vector<std::string> varnames,
                            parthenon::Metadata::FlagCollection flags = {},
                            std::vector<int> sparse_ids = {}) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  parthenon::MeshDataDescriptor req;
  const std::string cell_delta_name = ccbulk::cell_delta::name();
  if (std::find(varnames.begin(), varnames.end(), cell_delta_name) == varnames.end()) {
    varnames.push_back(cell_delta_name);
    // When filtering sparse pools by id, cell_delta's id must survive the filter.
    // (An empty sparse_ids set takes all ids of a pool, so only extend a non-empty set.)
    if (!sparse_ids.empty()) sparse_ids.push_back(cell_delta_id);
  }
  req.varnames = std::move(varnames);
  req.flags = std::move(flags);
  req.sparse_ids = std::move(sparse_ids);
  return req;
}

} // namespace RiotUtils

#endif // RIOT_UTILS_RIOT_UTILS_HPP_
