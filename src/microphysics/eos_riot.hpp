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
#ifndef MICROPHYSICS_EOS_RIOT_HPP_
#define MICROPHYSICS_EOS_RIOT_HPP_
// This file was made in part with generative AI.

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <tuple>
// parthenon infrastructure includes
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singularity EOS
#include <singularity-eos/eos/eos.hpp>
#include <singularity-eos/eos/eos_builder.hpp>
#include <singularity-utils/indexable_types.hpp>
#include <singularity-utils/variadic_utils.hpp>

#include "ionization/ionization_base.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace RiotEOS {

// The magic to automatically make zsplit of all EOS's available.
// TODO(JMM): Thread this through the cmake.
// TODO(JMM): Add shift and scale eventually when we need them
namespace impl {
// Type list of EOS types
template <typename... Ts>
using tl = singularity::variadic_utils::type_list<Ts...>;
// Type list of modifier types
template <template <typename> class... Ts>
using al = singularity::variadic_utils::adapt_list<Ts...>;
// Variadic utils
using singularity::variadic_utils::concat;
using singularity::variadic_utils::transform_variadic_list;

// The EOS's that can be zsplit
static constexpr const auto zsplitable =
    tl<singularity::IdealGas, singularity::Gruneisen
#ifdef SPINER_USE_HDF
       ,
       singularity::SpinerEOSDependsRhoT, singularity::SpinerEOSDependsRhoSie
#endif
       >{};
// EOS's that should not be zsplit
static constexpr const auto not_splitable =
    tl<singularity::JWL, singularity::DavisProducts, singularity::DavisReactants,
       singularity::IdealElectrons>{};
// A list of EOS's of the form ZSplitE<eos>...
static constexpr const auto zsplite =
    transform_variadic_list(zsplitable, al<singularity::ZSplitE>{});
// A list of EOS's of the form ZSplitI<eos>...
static constexpr const auto zspliti =
    transform_variadic_list(zsplitable, al<singularity::ZSplitI>{});
// Unmodified, ZSPlitE<eos>, ZSPlitI<eos>, IdealElectrons
static constexpr const auto all_eoses =
    concat(zsplitable, zsplite, zspliti, not_splitable);
} // namespace impl

using EOS = typename decltype(singularity::tl_to_Variant(impl::all_eoses))::vt;
using EOS_Array_t = parthenon::ParArray1D<EOS>;

class LambdaIndexerSimple {
 public:
  static constexpr bool is_type_indexable = true;
  LambdaIndexerSimple() = delete;
  KOKKOS_INLINE_FUNCTION
  LambdaIndexerSimple(Real zbar = 0.0) : zbar_(zbar) {}

  KOKKOS_FORCEINLINE_FUNCTION const Real &
  operator[](const singularity::IndexableTypes::MeanIonizationState &t) const {
    return zbar_;
  }
  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::MeanIonizationState &t) {
    return zbar_;
  }

 private:
  Real zbar_;
};

// (sparse pack view, kji)-driven lambda indexer for the loop abstraction. operator[]
// returns a live reference into the lr/lT/zbar cache fields in grid memory, so EOS
// write-backs (root-find guesses, solved zbar) persist across calls.
template <typename PackView, typename Index>
class LambdaIndexerSingle {
 public:
  static constexpr bool is_type_indexable = true;

  KOKKOS_FORCEINLINE_FUNCTION
  LambdaIndexerSingle(const PackView &pv, const Index &kji) : pv_(&pv), kji_(kji) {}

  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::LogDensity &) const {
    return (*pv_)(cell_variables::material_averaged::lr_cache(), kji_);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::LogTemperature &) const {
    return (*pv_)(cell_variables::material_averaged::lT_cache(), kji_);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::MeanIonizationState &) const {
    return (*pv_)(cell_variables::material_averaged::ionization_zbar(), kji_);
  }

 private:
  const PackView *pv_;
  Index kji_;
};

// (pack, b, m, k, j, i)-driven variant of LambdaIndexerSingle. Same semantics, but
// driven by explicit coordinates -- for callers that loop over materials inside a
// per-cell solve (e.g. the electron temperature root find), where (k, j, i) is already
// computed for the data reads, so no per-material sparse pack view is needed.
template <typename Pack_t>
class LambdaIndexerSingleCoord {
 public:
  static constexpr bool is_type_indexable = true;

  KOKKOS_FORCEINLINE_FUNCTION
  LambdaIndexerSingleCoord(const Pack_t &pack, const int b, const int m, const int k,
                           const int j, const int i)
      : pack_(&pack), b_(b), m_(m), k_(k), j_(j), i_(i) {}

  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::LogDensity &) const {
    return (*pack_)(b_, cell_variables::material_averaged::lr_cache(m_), k_, j_, i_);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::LogTemperature &) const {
    return (*pack_)(b_, cell_variables::material_averaged::lT_cache(m_), k_, j_, i_);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real &
  operator[](const singularity::IndexableTypes::MeanIonizationState &) const {
    return (*pack_)(b_, cell_variables::material_averaged::ionization_zbar(m_), k_, j_,
                    i_);
  }

 private:
  const Pack_t *pack_;
  int b_, m_, k_, j_, i_;
};

// Per-cell multi-material lambda indexer for the PTE solver: operator[](m) maps PTE slice
// index m through a (stack-resident) pte2slice array to a material index, and returns a
// LambdaIndexerSingleCoord bound to that material's cache fields. Mirrors the EOS-side
// LocalEosIndexer used by the same solve, and is the pack-view analogue of
// LambdaIndexerMulti (which remaps through a ScratchPad2D instead).
template <typename Pack_t>
class LambdaIndexerMultiCoord {
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  LambdaIndexerMultiCoord(const Pack_t &pack, const int *pte2slice, const int b,
                          const int k, const int j, const int i)
      : pack_(&pack), pte2slice_(pte2slice), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto operator[](const int m) const {
    return LambdaIndexerSingleCoord<Pack_t>(*pack_, b_, pte2slice_[m], k_, j_, i_);
  }

 private:
  const Pack_t *pack_;
  const int *pte2slice_;
  int b_, k_, j_, i_;
};

template <typename Pack_t>
class LambdaIndexerMulti {
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  LambdaIndexerMulti(const Pack_t &v, parthenon::ScratchPad2D<int> &pte2slice, int b,
                     int k, int j, int i)
      : v_(v), pte2slice_(pte2slice), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  LambdaIndexerMulti(const Pack_t &v, int b, int k, int j, int i)
      : v_(v), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto operator[](int m) {
    // Materials may be remapped through a dense PTE slice; otherwise m is the material
    // index directly. Returns a pack-view-backed single-material lambda indexer so
    // EOS write-backs land in the live cache fields (lr/lT/zbar) in grid memory.
    const int idx = (pte2slice_.size() > 0) ? pte2slice_(i_, m) : m;
    return LambdaIndexerSingleCoord<Pack_t>(v_, b_, idx, k_, j_, i_);
  }

 private:
  const Pack_t &v_;
  int b_, k_, j_, i_;
  parthenon::ScratchPad2D<int> pte2slice_;
};

template <typename Container_t = std::vector<std::string>, typename... Ts>
Container_t GetEOSNames(const singularity::Variant<Ts...> &eos) {
  return Container_t{Ts::EosType()...};
}

template <typename Lambda_t = Real *>
KOKKOS_INLINE_FUNCTION Real
energy_from_rho_P(const EOS &eos, const Real rho, const Real P,
                  Real sie_guess = std::numeric_limits<Real>::lowest(),
                  Lambda_t &&lambda = static_cast<Real *>(nullptr)) {
  using namespace RootFinding1D;
  Real sie;
  auto f = [&](const Real sie) {
    return eos.PressureFromDensityInternalEnergy(rho, sie, lambda);
  };
  const Real sie_min = eos.InternalEnergyFromDensityTemperature(rho, 1.e-50, lambda);
  const Real sie_max = eos.InternalEnergyFromDensityTemperature(rho, 1.e20, lambda);
  sie_guess =
      (sie_guess == std::numeric_limits<Real>::lowest() ? 0.5 * (sie_min + sie_max)
                                                        : sie_guess);
  auto status = regula_falsi(f, P, sie_guess, sie_min, sie_max, 1.e-12, 1.e-12, sie);
  if (status == Status::FAIL) {
    PARTHENON_DEBUG_WARN("energy_from_rho_P failed.");
    return std::numeric_limits<Real>::lowest();
  }
  return rho * sie;
}

//----------------------------------------------------------------------------------------
//! \fn  Real RiotEOS::rho_from_P_T
//! \brief
template <typename Lambda_t = Real *>
KOKKOS_INLINE_FUNCTION Real
rho_from_P_T(const EOS &eos, const Real P, const Real T, const Real rho_guess = 1.0,
             Lambda_t &&lambda = static_cast<Real *>(nullptr)) {
  using namespace RootFinding1D;
  Real rho;
  auto f = [&](const Real r) { return eos.PressureFromDensityTemperature(r, T, lambda); };
  const Real rho_min = 1.e-10;
  const Real rho_max = 1.e10;
  auto status = regula_falsi(f, P, rho_guess, rho_min, rho_max, 1.e-12, 1.e-12, rho);
  if (status == Status::FAIL) {
    PARTHENON_DEBUG_WARN("rho_from_P_T failed.");
    return -1.0;
  }
  return rho;
}

//----------------------------------------------------------------------------------------
//! \fn  auto RiotEOS::temperature_zbar_from_rho_P
//! \brief
KOKKOS_INLINE_FUNCTION
auto temperature_zbar_from_rho_P(const EOS &ion_eos, const EOS &electron_eos,
                                 const Real rho, const Real Ptot, Real T_guess,
                                 bool fully_ionized) {
  using namespace RootFinding1D;
  Real anuc = ion_eos.MeanAtomicMass();
  Real znuc = ion_eos.MeanAtomicNumber();
  Real zbar;
  auto f = [&](const Real T) {
    zbar = znuc;
    if (!fully_ionized)
      Ionization::ComputeIonizationState(anuc, znuc, rho, T, zbar, fully_ionized);
    LambdaIndexerSimple lambda(zbar);
    return ion_eos.PressureFromDensityTemperature(rho, T, lambda) +
           electron_eos.PressureFromDensityTemperature(rho, T, lambda);
  };
  const Real T_min = 1.e-10;
  const Real T_max = 1.e10;
  Real temperature = T_guess;
  auto status = regula_falsi(f, Ptot, T_guess, T_min, T_max, 1.e-12, 1.e-12, temperature);
  zbar = znuc;
  if (!fully_ionized)
    Ionization::ComputeIonizationState(anuc, znuc, rho, temperature, zbar, fully_ionized);
  return std::make_tuple(temperature, zbar);
}

//----------------------------------------------------------------------------------------
//! \fn  auto RiotEOS::temperature_zbar_from_rho_P_Te
//! \brief
KOKKOS_INLINE_FUNCTION
auto temperature_zbar_from_rho_P_Te(const EOS &ion_eos, const EOS &electron_eos,
                                    const Real rho, const Real Ptot, const Real Te,
                                    Real T_guess, bool fully_ionized) {
  using namespace RootFinding1D;
  Real anuc = ion_eos.MeanAtomicMass();
  Real znuc = ion_eos.MeanAtomicNumber();
  Real zbar = znuc;
  if (!fully_ionized)
    Ionization::ComputeIonizationState(anuc, znuc, rho, Te, zbar, fully_ionized);
  LambdaIndexerSimple lambda(zbar);
  auto f = [&](const Real T) {
    return ion_eos.PressureFromDensityTemperature(rho, T, lambda) +
           electron_eos.PressureFromDensityTemperature(rho, Te, lambda);
  };
  const Real T_min = 1.e-10;
  const Real T_max = 1.e10;
  Real temperature = T_guess;
  auto status = regula_falsi(f, Ptot, T_guess, T_min, T_max, 1.e-12, 1.e-12, temperature);
  return std::make_tuple(temperature, zbar);
}

//----------------------------------------------------------------------------------------
//! \fn  auto RiotEOS::density_zbar_from_P_temperature
//! \brief
KOKKOS_INLINE_FUNCTION
auto density_zbar_from_P_temperature(const EOS &ion_eos, const EOS &electron_eos,
                                     const Real Ptot, const Real T, Real rho_guess,
                                     bool fully_ionized) {
  using namespace RootFinding1D;
  Real anuc = ion_eos.MeanAtomicMass();
  Real znuc = ion_eos.MeanAtomicNumber();
  Real zbar = znuc;
  auto f = [&](const Real rho) {
    if (!fully_ionized)
      Ionization::ComputeIonizationState(anuc, znuc, rho, T, zbar, fully_ionized);
    LambdaIndexerSimple lambda(zbar);
    return ion_eos.PressureFromDensityTemperature(rho, T, lambda) +
           electron_eos.PressureFromDensityTemperature(rho, T, lambda);
  };
  const Real rho_min = ion_eos.RhoPmin(T);
  const Real rho_max = 1.0e6;
  Real density = rho_guess;
  auto status =
      regula_falsi(f, Ptot, rho_guess, rho_min, rho_max, 1.e-12, 1.e-12, density);
  zbar = znuc;
  if (!fully_ionized)
    Ionization::ComputeIonizationState(anuc, znuc, density, T, zbar, fully_ionized);
  return std::make_tuple(density, zbar);
}

//----------------------------------------------------------------------------------------
//! \fn  auto RiotEOS::density_zbar_from_P_Ti_Te
//! \brief
KOKKOS_INLINE_FUNCTION
auto density_zbar_from_P_Ti_Te(const EOS &ion_eos, const EOS &electron_eos,
                               const Real Ptot, const Real Ti, const Real Te,
                               Real rho_guess, bool fully_ionized) {
  using namespace RootFinding1D;
  Real anuc = ion_eos.MeanAtomicMass();
  Real znuc = ion_eos.MeanAtomicNumber();
  Real zbar = znuc;
  auto f = [&](const Real rho) {
    if (!fully_ionized)
      Ionization::ComputeIonizationState(anuc, znuc, rho, Te, zbar, fully_ionized);
    LambdaIndexerSimple lambda(zbar);
    return ion_eos.PressureFromDensityTemperature(rho, Ti, lambda) +
           electron_eos.PressureFromDensityTemperature(rho, Te, lambda);
  };
  const Real rho_min = ion_eos.RhoPmin(Ti);
  const Real rho_max = 1.0e6;
  Real density = rho_guess;
  auto status =
      regula_falsi(f, Ptot, rho_guess, rho_min, rho_max, 1.e-12, 1.e-12, density);
  zbar = znuc;
  if (!fully_ionized)
    Ionization::ComputeIonizationState(anuc, znuc, density, Te, zbar, fully_ionized);
  return std::make_tuple(density, zbar);
}

//----------------------------------------------------------------------------------------
//! \fn  EOS RiotEOS::Shift
//! \brief
template <typename T>
EOS Shift(T &&eos, bool shift, const Real reference_energy) {
  // if (shift) {
  // const Real std_temp = 273.15;
  // const Real std_pres = 1.013e6;
  // const Real rho0 = rho_from_P_T(eos, std_pres, std_temp, 10.0);
  // const Real sie0 = eos.InternalEnergyFromDensityTemperature(rho0, std_temp);
  // printf("sie0 = %g\n", sie0);
  // const Real sie_shift = reference_energy - sie0;
  // return singularity::ShiftedEOS<T>(std::forward<T>(eos), sie_shift);
  //}
  return eos;
}

enum class ZSplitting { None, Electrons, Ions };
//----------------------------------------------------------------------------------------
//! \fn  auto RiotEOS::ZSplit
//! \brief
template <typename T>
auto ZSplit(T eos, const ZSplitting s) {
  if (s == ZSplitting::Electrons) {
    return singularity::EOSBuilder::Modify<singularity::ZSplitE>(eos);
  } else if (s == ZSplitting::Ions) {
    return singularity::EOSBuilder::Modify<singularity::ZSplitI>(eos);
  } else {
    return eos;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotEOS::FillEosMap
//! \brief Fills a std::array material->EOS-index map.
template <typename T, typename V, typename M, std::size_t N>
KOKKOS_INLINE_FUNCTION void FillEosMap(V &v, const int b, const int nmat,
                                       M &eos_from_matid, M &nphase,
                                       std::array<int, N> &map) {
  for (int m = 0; m < nmat; m++) {
    const int sparse_id = v(b, T(m)).sparse_id;
    map[m] = eos_from_matid(sparse_id);
    for (int n = 1; n < nphase(sparse_id); n++) {
      m++;
      map[m] = eos_from_matid(sparse_id) + n;
    }
  }
}

EOS InitializeEOS(ParameterInput *pin, const std::string &block_name, bool is_electron,
                  bool actually_load);

} // namespace RiotEOS

#endif // MICROPHYSICS_EOS_RIOT_HPP_
