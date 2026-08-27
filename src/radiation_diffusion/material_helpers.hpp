//========================================================================================
// (C) (or copyright) 2025-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef RADIATION_DIFFUSION_MATERIAL_HELPERS_HPP_
#define RADIATION_DIFFUSION_MATERIAL_HELPERS_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>
#include <ports-of-call/robust_utils.hpp>

#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "microphysics/pte_closure.hpp"
#include "polylog.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace RadiationDiffusion {
struct MaterialHelpers {
  template <class T>
  using pa1d_t = parthenon::ParArray1D<T>;

  // A tagged wrapper around a loop-abstraction per-point scratch view. The compile-time
  // Tag type selects which physical quantity the helper accumulates into this scratch, so
  // a single variadic helper can fill different subsets of outputs at each call site. The
  // scratch view type is deduced (1D per-point for EOS quantities, or per-point-by-group
  // for multigroup opacities).
  struct EgasTag {};
  struct CvTag {};
  struct AlphaAbsMGTag {};
  struct dAlphaAbsMGdTTag {};
  struct AlphaTotMGTag {};

  template <class Tag, class ScratchT>
  struct ScratchTag {
    using tag = Tag;
    KOKKOS_INLINE_FUNCTION explicit ScratchTag(const ScratchT &scratch)
        : scratch(scratch) {}
    const ScratchT &scratch;
  };

  // Factory helpers so call sites read the same as before, e.g. EgasScratch(egas_view).
  template <class ScratchT>
  KOKKOS_INLINE_FUNCTION static auto EgasScratch(const ScratchT &s) {
    return ScratchTag<EgasTag, ScratchT>(s);
  }
  template <class ScratchT>
  KOKKOS_INLINE_FUNCTION static auto CvScratch(const ScratchT &s) {
    return ScratchTag<CvTag, ScratchT>(s);
  }
  template <class ScratchT>
  KOKKOS_INLINE_FUNCTION static auto AlphaAbsMGScratch(const ScratchT &s) {
    return ScratchTag<AlphaAbsMGTag, ScratchT>(s);
  }
  template <class ScratchT>
  KOKKOS_INLINE_FUNCTION static auto dAlphaAbsMGdTScratch(const ScratchT &s) {
    return ScratchTag<dAlphaAbsMGdTTag, ScratchT>(s);
  }
  template <class ScratchT>
  KOKKOS_INLINE_FUNCTION static auto AlphaTotMGScratch(const ScratchT &s) {
    return ScratchTag<AlphaTotMGTag, ScratchT>(s);
  }

  template <class temperature_t>
  static auto GetEOSs(parthenon::Mesh *pmesh, temperature_t) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    if constexpr (std::is_same_v<temperature_t, ccbulk::temperature>) {
      return pmesh->packages.Get("materials")->Param<pa1d_t<RiotEOS::EOS>>("d.d.EOS");
    } else if constexpr (std::is_same_v<temperature_t, ccbulk::electron_temperature>) {
      return pmesh->packages.Get("materials")
          ->Param<pa1d_t<RiotEOS::EOS>>("d.d.electron_EOS");
    } else {
      PARTHENON_FAIL("Bad temperature type.");
      return pmesh->packages.Get("materials")->Param<pa1d_t<RiotEOS::EOS>>("d.d.EOS");
    }
  }

  template <class temperature_t>
  MaterialHelpers(parthenon::Mesh *pmesh, const std::string &rad_package, temperature_t T)
      : eos(GetEOSs(pmesh, T)),
        eos_from_matid(
            pmesh->packages.Get("materials")->Param<pa1d_t<int>>("d.EOS_from_matid")),
        opac_a(pmesh->packages.Get("materials")
                   ->Param<pa1d_t<RiotOpacity::MeanOpacA>>("d.d.opac_a")),
        opac_s(pmesh->packages.Get("materials")
                   ->Param<pa1d_t<RiotOpacity::MeanOpacS>>("d.d.opac_s")),
        opac_from_matid(
            pmesh->packages.Get("materials")->Param<pa1d_t<int>>("d.opac_from_matid")) {
    const auto &radiation_pkg = pmesh->packages.Get(rad_package);
    const auto &rad_params = pmesh->packages.Get(rad_package)->AllParams();
    rhomin = rad_params.Get<parthenon::Real>("diff_rho_min");
    tempmin = rad_params.Get<parthenon::Real>("diff_temp_min");
    ngroup = rad_params.hasKey("ngroup") ? rad_params.Get<int>("ngroup") : 0;
  }

  KOKKOS_DEFAULTED_FUNCTION
  MaterialHelpers() = default;

  KOKKOS_DEFAULTED_FUNCTION
  MaterialHelpers(const MaterialHelpers &) = default;

  const pa1d_t<RiotEOS::EOS> eos;
  const pa1d_t<int> eos_from_matid;
  const pa1d_t<RiotOpacity::MeanOpacA> opac_a;
  const pa1d_t<RiotOpacity::MeanOpacS> opac_s;
  const pa1d_t<int> opac_from_matid;
  Real rhomin, tempmin;
  int ngroup;

  template <class temperature_t, class idx_range_t, class pack_t,
            class pack_temperature_t, class... output_ts>
  KOKKOS_INLINE_FUNCTION void CalculateEos(const idx_range_t &idx_range,
                                           const pack_t &pack,
                                           const pack_temperature_t &pack_temperature,
                                           int b, output_ts... outputs) const {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;

    idx_range.TeamBarrier();
    RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
      ([&] { outputs.scratch(k, j, i) = 0.0; }(), ...);
    });

    using idx_t = typename idx_range_t::inner_idx_t;
    const int nmat = pack.GetSize(b, ccmat::rho());
    auto T = make_var_view(idx_range, pack_temperature, temperature_t());
    for (int m = 0; m < nmat; ++m) {
      const int mat_id = pack(b, cm::rho(m)).sparse_id;
      const int phase_id = pack(b, cm::rho(m)).v;
      const int eos_id = eos_from_matid(mat_id) + phase_id;
      const auto &eosm = eos(eos_id);
      idx_range.TeamBarrier();
      auto rhom = make_var_view(idx_range, pack, cm::rho(m));
      auto rhobarm = make_var_view(idx_range, pack, ccmat::rho(m));
      RiotLoop::inner(idx_range, [&](const idx_t kji) {
        (
            [&] {
              if constexpr (std::is_same_v<typename output_ts::tag, EgasTag>) {
                outputs.scratch(kji) +=
                    rhobarm(kji) *
                    eosm.InternalEnergyFromDensityTemperature(rhom(kji), T(kji));
              } else if constexpr (std::is_same_v<typename output_ts::tag, CvTag>) {
                outputs.scratch(kji) +=
                    rhobarm(kji) *
                    eosm.SpecificHeatFromDensityTemperature(rhom(kji), T(kji));
              } else {
                PARTHENON_FAIL("Unknown type.");
              }
            }(),
            ...);
      });
    }
  }

  template <class temperature_t, class idx_range_t, class pack_t,
            class pack_temperature_t, class... output_ts>
  KOKKOS_INLINE_FUNCTION void
  CalculateMultiGroupOpacities(const idx_range_t &idx_range, const pack_t &pack,
                               const pack_temperature_t &pack_temperature, int b,
                               output_ts... outputs) const {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;

    constexpr bool calculate_a = true;
    constexpr bool calculate_s = true;

    using idx_t = typename idx_range_t::inner_idx_t;
    const int ng = ngroup;
    idx_range.TeamBarrier();
    RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
      for (int g = 0; g < ng; ++g) {
        ([&] { outputs.scratch(g, k, j, i) = 0.0; }(), ...);
      }
    });

    const int nmat = pack.GetSize(b, cm::rho());
    auto T = make_var_view(idx_range, pack_temperature, temperature_t());
    for (int m = 0; m < nmat; ++m) {
      const int mat_id = pack(b, cm::rho(m)).sparse_id;
      const int phase_id = pack(b, cm::rho(m)).v;
      const int opac_id = opac_from_matid(mat_id) + phase_id;
      const auto &opac_am = opac_a(opac_id);
      const auto &opac_sm = opac_s(opac_id);
      auto rhom = make_var_view(idx_range, pack, cm::rho(m));
      auto vfracm = make_var_view(idx_range, pack, ccmat::volume_fraction(m));
      for (int g = 0; g < ng; ++g) {
        idx_range.TeamBarrier();
        RiotLoop::inner(idx_range, [&](const idx_t kji) {
          const Real rrm = std::max(rhom(kji), rhomin);
          const Real ttm = std::max(T(kji), tempmin);
          const Real aa_am =
              calculate_a ? opac_am.AbsorptionCoefficient(rrm, ttm, g) : 0.0;
          const Real aa_sm =
              calculate_s ? opac_sm.ScatteringCoefficient(rrm, ttm, g) : 0.0;

          (
              [&] {
                if constexpr (std::is_same_v<typename output_ts::tag, AlphaAbsMGTag>) {
                  outputs.scratch(g, kji) += vfracm(kji) * aa_am;
                } else if constexpr (std::is_same_v<typename output_ts::tag,
                                                    dAlphaAbsMGdTTag>) {
                  // d(alpha_g)/dT = (alpha_g / T) * d(log alpha_g)/d(log T)
                  const Real dlogadlogT =
                      opac_am.DLogAbsorptionCoefficientDLogT(rrm, ttm, g);
                  const Real daadTm = (ttm > tempmin) ? aa_am / ttm * dlogadlogT : 0.0;
                  outputs.scratch(g, kji) += vfracm(kji) * daadTm;
                } else if constexpr (std::is_same_v<typename output_ts::tag,
                                                    AlphaTotMGTag>) {
                  outputs.scratch(g, kji) += vfracm(kji) * (aa_am + aa_sm);
                } else {
                  PARTHENON_FAIL("Unknown type.");
                }
              }(),
              ...);
        });
      }
    }
  }
};

struct BlackBodyHelper {
  KOKKOS_DEFAULTED_FUNCTION
  BlackBodyHelper() = default;

  BlackBodyHelper(Mesh *pmesh)
      : a(pmesh->packages.Get("multigroup_diffusion_package")
              ->Param<Real>("a_radiation")),
        ngroup(pmesh->packages.Get("multigroup_diffusion_package")->Param<int>("ngroup")),
        group_energies_in_K(
            pmesh->packages.Get("multigroup_diffusion_package")
                ->Param<parthenon::ParArray1D<Real>>("group_energies_in_K")) {}

  Real a;
  const parthenon::ParArray1D<Real> group_energies_in_K;
  int ngroup;

  KOKKOS_INLINE_FUNCTION
  auto GetBB(int g, Real T) const {
    using PortsOfCall::Robust::ratio;
    const Real aT3 = a * T * T * T;
    auto [bl, dbldT] = IncompleteBose3FromZero<5>(ratio(group_energies_in_K(g), T));
    auto [bu, dbudT] = IncompleteBose3FromZero<5>(ratio(group_energies_in_K(g + 1), T));
    dbldT *= -ratio(group_energies_in_K(g), (T * T));
    dbudT *= -ratio(group_energies_in_K(g + 1), (T * T));
    if (g == 0) {
      bl = 0.0;
      dbldT = 0.0;
    }
    if (g == ngroup - 1) {
      bu = 1.0;
      dbudT = 0.0;
    }
    const Real B = aT3 * T * (bu - bl);
    const Real dBdT = 4.0 * aT3 * (bu - bl) + aT3 * T * (dbudT - dbldT);
    return std::make_tuple(B, dBdT);
  }
};

struct SourceHelper {
  SourceHelper(Mesh *pmesh, Real dt)
      : bb_helper(pmesh),
        dtcl(dt *
             pmesh->packages.Get("multigroup_diffusion_package")->Param<Real>("c_light")),
        ngroup(
            pmesh->packages.Get("multigroup_diffusion_package")->Param<int>("ngroup")) {}

  BlackBodyHelper bb_helper;
  Real dtcl;
  int ngroup;

  template <class temperature_t, class idx_range_t, class pack_egroup_t,
            class pack_temperature_t, class scratch_mg_t>
  KOKKOS_INLINE_FUNCTION void
  CalculateSource(const idx_range_t &idx_range, const pack_egroup_t &pack_egroup,
                  const pack_temperature_t &pack_temperature, const scratch_mg_t &aa,
                  const scratch_mg_t &daadT, int b, scratch_mg_t &S,
                  scratch_mg_t &dSdT) const {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;

    using idx_t = typename idx_range_t::inner_idx_t;
    const int ng = ngroup;
    idx_range.TeamBarrier();
    auto T = make_var_view(idx_range, pack_temperature, temperature_t());
    for (int g = 0; g < ng; ++g) {
      auto Eg = make_var_view(idx_range, pack_egroup, MultiGroupVars::Egroup(g));
      RiotLoop::inner(idx_range, [&](const idx_t kji) {
        const auto [B, dBdT] = bb_helper.GetBB(g, T(kji));
        S(g, kji) = dtcl * aa(g, kji) * (B - Eg(kji));
        dSdT(g, kji) = dtcl * daadT(g, kji) * (B - Eg(kji)) + dtcl * aa(g, kji) * dBdT;
      });
    }
  }
};

} // namespace RadiationDiffusion

#endif // RADIATION_DIFFUSION_MATERIAL_HELPERS_HPP_
