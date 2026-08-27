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
#ifndef MICROPHYSICS_STRENGTH_MODELS_HPP_
#define MICROPHYSICS_STRENGTH_MODELS_HPP_
// This file was made in part with generative AI.

#include <ports-of-call/variant.hpp>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include <singularity-eos/eos/eos.hpp>

namespace Strength {

// enum class stress_model {epp, sg, invalid};
enum class stress_model { epp, invalid };

//----------------------------------------------------------------------------------------
//! \fn  Real Strength::radial_return
//! \brief
template <typename T, class... Args>
KOKKOS_INLINE_FUNCTION Real radial_return(Real &sxx, Real &sxy, Real &sxz, Real &syy,
                                          Real &syz, Real &eps, T &model, Real &j2,
                                          Args &&...args) {
  const Real szz = -sxx - syy;
  const Real sdots =
      sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
  j2 = 0.5 * sdots;

  // shear modulus and yield stress
  const Real G = model.shear_modulus(std::forward<Args>(args)...);
  const Real Y = model.yield(std::forward<Args>(args)...);

  // equivalent plastic strain
  Real eps_fact = 1.0;
  if (Y == 0.0) eps_fact = 0.0;

  // yield function
  Real yield_func = (2.0 / 3.0) * Y * Y / (sdots + 1.e-16);
  // return if the stress is inside the yield surface
  if (yield_func >= 1) return 0.0;

  // correction factor
  Real f = yield_func;
  j2 *= f;
  // TODO(JCD): this should be a root find to put sij on the yield surface,
  // accounting for how increments in eps modify the shear mod and yield stress.
  // if we end up incrementing the thermal energy from plastic work, the
  // temperature/energy dependence of the shear mod and yield stress should
  // also be captured in the root find.
  f = std::sqrt(f);
  sxx *= f;
  sxy *= f;
  sxz *= f;
  syy *= f;
  syz *= f;

  // increment equivalent plastic strain
  f = 1.0 - f;
  const Real delta_eps = eps_fact * (f / (G + 1.e-16)) * std::sqrt(sdots / 6.0);
  eps += delta_eps;

  // plastic work source term
  eps *= eps_fact;
  const Real deltaW = eps_fact * (f * sdots / (2.0 * G + 1.e-16));
  return deltaW;
}

template <typename T>
class StrengthBase {
 public:
  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real shear_modulus(const Real rho, Args &&...args) const {
    return (rho > rho_fail_) * static_cast<const T *>(this)->shear_modulus_impl(
                                   rho, std::forward<Args>(args)...);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real yield(const Real rho, Args &&...args) const {
    return (rho > rho_fail_) *
           static_cast<const T *>(this)->yield_impl(rho, std::forward<Args>(args)...);
  }

  KOKKOS_INLINE_FUNCTION
  Real failure(const Real rho) const { return std::min(rho / rho_fail_, 1.0); }
  void set_rho_fail(const Real rho_fail) { rho_fail_ = rho_fail; }

 private:
  Real rho_fail_ = 0.0;
};

class EPP : public StrengthBase<EPP> {
 public:
  KOKKOS_FUNCTION EPP(const Real G, const Real Y) : G_(G), Y_(Y) {}
  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real shear_modulus_impl(Args &&...args) const {
    return G_;
  }
  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real yield_impl(Args &&...args) const {
    return Y_;
  }
  stress_model type() const { return stress_model::epp; }

 private:
  Real G_, Y_;
};

// class SG : public StrengthBase<SG> {
//  public:
//   SG(const Real G0, const Real Y0, const Real rho0, const Real alpha, const Real beta,
//      const Real gamma, const Real gammap, const Real gammaT, const Real gammaTp,
//      const Real ep0, const Real Troom, const Real Tmelt)
//     : G0_(G0), Y0_(Y0), rho0_(rho0), alpha_(alpha), beta_(beta), gamma_(gamma),
//       gammap_(gammap), gammaT_(gammaT), gammaTp_(gammaTp), ep0_(ep0),
//       Troom_(Troom), Tmelt_(Tmelt) {}

//   template <class... Args>
//   KOKKOS_INLINE_FUNCTION
//   Real shear_modulus_impl(Real rho, Real P, Real T, Args &&... args) const {
//     const Real Pp = std::max(P,0.0);
//     const Real pcor = 1.0 + gammap_*Pp*std::cbrt(rho0_/rho);
//     const Real fmelt = gammaTp_ * (T - Troom_);
//     return (T < Tmelt_) * G0_ * (pcor - fmelt);
//   }

//   KOKKOS_INLINE_FUNCTION
//   Real yield_impl(Real rho, Real P, Real T, Real eps, Args &&... args) const {
//     const Real Pp = std::max(P,0.0);
//     const Real pcor = 1.0 + gamma_*Pp*std::cbrt(rho0_/rho);
//     const Real fmelt = gammaT_ * (T - Troom_);
//     const Real Y = Y0_ * std::pow(1.0 + alpha_*(ep0_ + eps), beta_) * (pcor - fmelt);
//     return (T < Tmelt_) * Y;
//   }

//   stress_model type() const { return stress_model::sg; }

//  private:
//   Real G0_, Y0_, rho0_, alpha_, beta_, gamma_, gammap_;
//   Real gammaT_, gammaTp_, ep0_, Troom_, Tmelt_;
// };

template <typename... StressModels>
class StressModelVariant {
 public:
  KOKKOS_FUNCTION StressModelVariant() : sm_(EPP(0.0, 0.0)) {}
  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real shear_modulus(Args &&...args) {
    return PortsOfCall::visit(
        [args = std::make_tuple(std::forward<Args>(args)...)](auto &model) {
          return std::apply(
              [&model](auto &&...args) { return model.shear_modulus(args...); },
              std::move(args));
        },
        sm_);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real yield(Args &&...args) {
    return PortsOfCall::visit(
        [args = std::make_tuple(std::forward<Args>(args)...)](auto &model) {
          return std::apply([&model](auto &&...args) { return model.yield(args...); },
                            std::move(args));
        },
        sm_);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION Real failure(const Real rho) {
    return PortsOfCall::visit([&rho](const auto &model) { return model.failure(rho); },
                              sm_);
  }

  stress_model type() const {
    return PortsOfCall::visit([](const auto &sm) { return sm.type(); }, sm_);
  }

  template <
      typename Model,
      typename std::enable_if<
          !std::is_same<StressModelVariant, typename std::decay<Model>::type>::value,
          bool>::type = true>
  KOKKOS_FUNCTION StressModelVariant(Model &&m) : sm_(m) {}

  template <
      typename Model,
      typename std::enable_if<
          !std::is_same<StressModelVariant, typename std::decay<Model>::type>::value,
          bool>::type = true>
  KOKKOS_INLINE_FUNCTION Model &get() {
    return PortsOfCall::get<Model>(sm_);
  }

 private:
  PortsOfCall::variant<StressModels...> sm_;
};

using StressModel = StressModelVariant<EPP>;

//----------------------------------------------------------------------------------------
//! \fn  StressModel Strength::InitStressModel
//! \brief
inline StressModel InitStressModel(ParameterInput *pin, const std::string &block) {
  auto model = pin->GetString(block, "modelname");

  const Real rho_fail = pin->GetOrAddReal(block, "rho_fail", 0.0);
  if (model == "epp") {
    const Real G0 = pin->GetReal(block, "G0");
    const Real Y0 = pin->GetReal(block, "Y0");
    EPP sm(G0, Y0);
    sm.set_rho_fail(rho_fail);
    return StressModel(sm);
  }
  // else if (model == "sg") {
  //   const Real G0 = pin->GetReal(block, "G0");
  //   const Real Y0 = pin->GetReal(block, "Y0");
  //   ...
  //   ...
  //   MSG sm = MSG(G0, Y0, rho0, alpha, beta, gamma, gammap, delta, ep0, emelt);
  //   sm.set_rho_fail(rho_fail);
  //   return StressModel(sm);
  // }
  else {
    std::stringstream ss;
    ss << "Invalid strength model in input block " << block << ". " << model
       << " is not one of {epp}";
    PARTHENON_THROW(ss);
  }

  return StressModel();
}

} // namespace Strength

#endif // MICROPHYSICS_STRENGTH_MODELS_HPP_
