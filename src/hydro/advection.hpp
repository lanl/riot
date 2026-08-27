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
#ifndef HYDRO_ADVECTION_HPP_
#define HYDRO_ADVECTION_HPP_
// This file was made in part with generative AI.

// TODO(JMM): Should this be a separate package?

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include "variables.hpp"

namespace Hydro {

class PrimFluxPack {
 public:
  using Pack_t = parthenon::SparsePack<>;

  PrimFluxPack() = default;
  PrimFluxPack(const Pack_t &prims, const Pack_t &cons) : prims_(prims), cons_(cons) {}

  //--------------------------------------------------------------------------------------
  //! \fn  Real& Hydro::PrimFluxPack::operator()
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int b, const int var, const int k, const int j,
                   const int i) const {
    return prims_(b, var, k, j, i);
  }
  //--------------------------------------------------------------------------------------
  //! \fn  Real& Hydro::PrimFluxPack::flux
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  Real &flux(const int b, const int d, const int var, const int k, const int j,
             const int i) const {
    return cons_.flux(b, d, var, k, j, i);
  }

  //--------------------------------------------------------------------------------------
  //! \fn  auto Hydro::PrimFluxPack::ConsSparseID
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  auto ConsSparseID(const int b, const int var) const { return cons_(b, var).sparse_id; }

  //--------------------------------------------------------------------------------------
  //! \fn  int Hydro::PrimFluxPack::Size
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  int Size(const int b) const {
    return cons_.GetUpperBound(b) - cons_.GetLowerBound(b) + 1;
  }
  //--------------------------------------------------------------------------------------
  //! \fn  int Hydro::PrimFluxPack::MaxSize
  //! \brief
  KOKKOS_FORCEINLINE_FUNCTION
  int MaxSize() const { return prims_.GetMaxNumberOfVars(); }

  //--------------------------------------------------------------------------------------
  //! \fn  const Pack_t& Hydro::PrimFluxPack::Prims / Cons
  //! \brief Underlying packs, so the loop abstraction's single-variable views can bind
  //!        directly: make_var_view over the primitive state, make_flux_view over the
  //!        conserved fluxes. Both index by the same anonymous integer var used above.
  KOKKOS_FORCEINLINE_FUNCTION const Pack_t &Prims() const { return prims_; }
  KOKKOS_FORCEINLINE_FUNCTION const Pack_t &Cons() const { return cons_; }

 private:
  Pack_t prims_;
  Pack_t cons_;
};

} // namespace Hydro

#endif // HYDRO_ADVECTION_HPP_
