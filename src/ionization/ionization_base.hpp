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
#ifndef IONIZATION_IONIZATION_BASE_HPP_
#define IONIZATION_IONIZATION_BASE_HPP_
// This file was made in part with generative AI.

// Formulae and constants for the Ionization package go here.

namespace Ionization {
enum class EntropyDirection { ToEntropy, ToEnergy };
enum class EntropyForm { Primitive, Conserved };
enum class ElectronThermalConductivityModel {
  Constant,
  SpitzerVolumeAverageArithmetic,
  SpitzerVolumeAverageHarmonic,
  SpitzerElectronNumberDensityAverage
};
enum class IonThermalConductivityModel { Constant, Braginskii };
enum class CoulombLogarithmKind : int { LeeMoore, Brysk, BPS, Basic };
enum class TransportSpecies { Electron, Ion };
enum class PlasmaViscosityModel { Constant, FokkerPlanckLandau };

namespace IonizationModelConstants {
// Thomas-Fermi model, fit by R. More (1981, UCRL-84991), after
// Salzmann (1998) Atomic Physics in Hot Plasmas. Number v.97 in International
// Series of Monographs on
// Physics Ser. Oxford University Press, Incorporated, Cary, 1998. ISBN
// 978-0-19-510930-6 978-0-19-535515-4
namespace ThomasFermiMore {
// ThomasFermiMore() {}
constexpr Real me = 9.10938e-28;      // g
constexpr Real planck = 6.6261e-27;   // erg/s
constexpr Real kbev = 8.617333262e-5; // eV/K
constexpr Real kberg = 1.3807e-16;    // erg/K
constexpr Real eh = 2.17e-11;         // erg
constexpr Real amu = 1.66054e-24;     // g
constexpr Real a1 = 3.323e-3;
constexpr Real a2 = 0.971832;
constexpr Real a3 = 9.26148e-5;
constexpr Real a4 = 3.10165;
constexpr Real b0 = -1.7630;
constexpr Real b1 = 1.43175;
constexpr Real b2 = 0.315463;
constexpr Real c1 = -0.366667;
constexpr Real c2 = 0.983333;
constexpr Real alpha = 14.3139;
constexpr Real beta = 0.6624;
}; // namespace ThomasFermiMore
} // namespace IonizationModelConstants

namespace CouplingModelConstants {
constexpr Real me = 9.10938e-28;                 // g
constexpr Real sqrtme = 3.018174945227662e-14;   // sqrt(g)
constexpr Real planck = 6.6261e-27;              // erg/s
constexpr Real hbar = 1.05457266e-27;            // erg/s
constexpr Real kbev = 8.617333262e-5;            // eV/K
constexpr Real kberg = 1.3807e-16;               // erg/K
constexpr Real kberg72 = 3.0927700922056064e-56; // (erg/K)^(7/2)
constexpr Real eh = 2.17e-11;                    // erg
constexpr Real amu = 1.66054e-24;                // g
constexpr Real qe = 4.8e-10;                     // esu (statcoulombs)
constexpr Real qe2 = 2.304e-19;                  // esu (statcoulombs^2)
constexpr Real qe4 = 5.308416e-38;               // esu (statcoulombs^4)
constexpr Real pi = 3.141592653589793;
constexpr Real geiconst = 2.1871301561467949e-13; // 8*sqrt(2*pi)*qe**4/(3*kberg**1.5)
constexpr Real euler = 0.5772156649;
// constexpr Real kberg_ov_4piqe2 = 2.498455858643382e-35;
constexpr Real kberg_ov_4piqe2 = 47.687766911236956;
constexpr Real qe2_ov_3kberg = 5.562395886144709e-4;
constexpr Real eight_ov_pi_32 = 4.063592699791422;
constexpr Real sqrtpi = 1.7724538509055159;
} // namespace CouplingModelConstants

//----------------------------------------------------------------------------------------
//! \fn  Real CoulombLogarithmBPS
//! \brief Coulomb Logarithm of Brown, Preston and Singleton
//! Charged particle motion in a highly ionized plasma. Physics Reports,
//! 410:237–333, 2005
KOKKOS_FORCEINLINE_FUNCTION
Real CoulombLogarithmBPS(const Real &tele, const Real &tion, const Real &ne,
                         const Real &ni, const Real &mi, const Real &zbar) {
  using namespace CouplingModelConstants;
  static constexpr Real FOURPI = 4.0 * pi;

  Real we2 = FOURPI * qe2 * ne / me;

  Real clog = 8.0 * std::pow((kberg * tele / hbar), 2) / (we2 + 1e-15);
  clog = std::log(1.0 + clog) - 1.0 - euler;
  clog = std::max(1.0, 0.5 * clog);
  return clog;

} // CoulombLogarithmBPS

//----------------------------------------------------------------------------------------
//! \fn  Real CoulombLogarithmLeeMoore
//! \brief Coulomb Logarithm of Lee & Moore 1984
//! This should be similar to the form below at high, HEDP relevant temperatures
//! but can deviate at mid-lower temperatures.
KOKKOS_FORCEINLINE_FUNCTION
Real CoulombLogarithmLeeMoore(const Real &tele, const Real &tion, const Real &ne,
                              const Real &ni, const Real &mi, const Real &zbar) {
  using namespace CouplingModelConstants;
  static constexpr Real FOURPI = 4.0 * pi;

  // min impact parameter is that of a 90 degree scattering, which is
  const Real bmin = std::max((zbar * qe2) / (3.0 * kberg * tele),
                             planck / (2.0 * std::sqrt(3.0 * kberg * tele * me)));
  const Real deb = std::sqrt((kberg * tele / (FOURPI * ne * qe2 + 1e-15)) +
                             kberg * tion /
                                 (FOURPI * ni * qe2 * zbar * zbar +
                                  1e-15)); // debye length including ionization state
  const Real bmax = deb; // take debye length as maximum impact parameter
  const Real ratio = bmax / (bmin + 1e-15);

  // Enforce clog to be >= 2.0. (Higher floor suggested by Lee & Moore)
  // TODO(blb): It is perhaps worthwhile to fold the floor into an input param.
  return std::max(2.0, 0.5 * std::log(1.0 + ratio * ratio));
} // CoulombLogarithmLeeMoore

//----------------------------------------------------------------------------------------
//! \fn  Real CoulombLogarithmBasic
//! \brief basic Coulomb Logarithm
KOKKOS_FORCEINLINE_FUNCTION
Real CoulombLogarithmBasic(const Real &tele, const Real &tion, const Real &ne,
                           const Real &ni, const Real &mi, const Real &zbar) {
  using namespace CouplingModelConstants;

  const Real vion =
      std::sqrt(kberg * tion / (mi + 1.e-15)); // ion thermal velocity a la boltzmann
  const Real vele =
      std::sqrt(kberg * tele / me); // electron thermal velocity a la boltzmann
  // const Real vrel = vele - vion;    // relative thermal velocity

  // const Real mu = 1.0 / (1.0 / me + 1.0 / mi); // reduced mass of system

  // min impact parameter is that of a 90 degree scattering, which is
  // bmin = b0 = q1q2/(mu * urel**2) (Sivukhin 1966)
  // Real bmin = zbar * qe2 / mu / (vrel * vrel + 1e-15); // min impact parameter
  const Real bmin = std::max((zbar * qe2) / (3.0 * kberg * tele),
                             planck / (2.0 * std::sqrt(3.0 * kberg * tele * me)));
  const Real edb =
      planck / std::sqrt(2.0 * pi * me * kberg *
                         tele); // electron thermal de Broglie wavelength in cm
  const Real deb =
      std::sqrt(kberg * tele / (4.0 * pi * ne * qe2 + 1e-15)); // debye length
  const Real bmax = deb; // take debye length as maximum impact parameter

  // Enforce clog to be >= 1.0.
  // TODO(blb): It is perhaps worthwhile to fold the floor into an input param.
  return std::max(1.0, std::log(bmax / (bmin + 1e-15)));
} // CoulombLogarithmBasic

//----------------------------------------------------------------------------------------
//! \fn  Real CoulombLogarithmBrysk
//! \brief Coulomb Logarithm a la Brysk
//! H Brysk 1974 Plasma Physics 16 927
KOKKOS_FORCEINLINE_FUNCTION
Real CoulombLogarithmBrysk(const Real &tele, const Real &tion, const Real &ne,
                           const Real &ni, const Real &mi, const Real &zbar) {
  using namespace CouplingModelConstants;
  const Real bmax =
      std::sqrt(kberg_ov_4piqe2 * tele / (ne + 1e-15)); // max impact parameter
  Real bmin = qe2_ov_3kberg * zbar / tele;              // min impact parameter
  bmin = std::max(bmin, 0.5 * hbar / std::sqrt(3.0 * kberg * tele * me));

  return std::max(1.0, std::log(bmax / (bmin + 1e-15)));
} // CoulombLogarithmBrysk

//----------------------------------------------------------------------------------------
//! \fn  Real CoulombLogarithmIonIon
//! \brief Coulomb Logarithm for ion-ion collisions
//! a la Vold et al. Phys. Plasmas 24, 042702 (2017)
KOKKOS_FORCEINLINE_FUNCTION
Real CoulombLogarithmIonIon(const Real &tele, const Real &ne, const Real &tion,
                            const Real &mi1, const Real mi2, const Real &zbar1,
                            const Real &zbar2) {
  using namespace CouplingModelConstants;

  const Real kdebye = std::sqrt(kberg * tele / (4.0 * M_PI * qe2 * ne));
  const Real kperp = 3.0 * kberg * tele * kdebye / (zbar1 * zbar2 * qe2);
  return std::log(1.0 + 0.7 * kperp);

} // CoulombLogarithmIonIon

//----------------------------------------------------------------------------------------
//! \fn  Real EvaluateCoulombLogarithm
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real EvaluateCoulombLogarithm(const Real &tele, const Real &tion, const Real &ne,
                              const Real &ni, const Real &mi, const Real &zbar,
                              CoulombLogarithmKind logKind) {
  switch (logKind) {
  case CoulombLogarithmKind::LeeMoore:
    return CoulombLogarithmLeeMoore(tele, tion, ne, ni, mi, zbar);

  case CoulombLogarithmKind::BPS:
    return CoulombLogarithmBPS(tele, tion, ne, ni, mi, zbar);

  case CoulombLogarithmKind::Basic:
    return CoulombLogarithmBasic(tele, tion, ne, ni, mi, zbar);

  case CoulombLogarithmKind::Brysk:
  default:
    return CoulombLogarithmBrysk(tele, tion, ne, ni, mi, zbar);
  }
}

// Fokker-Planck-Landau symmetrical momentum exchange rate for ion species i
// scattering off of a background species of ions j
// see
// E. L. Vold, R. M. Rauenzahn, C. H. Aldrich, K. Molvig, A. N. Simakov, and B.
// M. Haines. Plasma transport in an Eulerian AMR code. Physics of Plasmas,
// 24(4):042702, April 2017. ISSN 1070-664X. doi: 10.1063/1.4979171. URL
// https://doi.org/10.1063/1.4979171.
KOKKOS_FORCEINLINE_FUNCTION
Real IonIonMomentumExchangeRate(const Real tele, const Real tion, const Real ne,
                                const Real ni1, const Real ni2, const Real mi1,
                                const Real mi2, const Real zbar1, const Real zbar2,
                                const Real zbar_floor,
                                const Real ion_number_density_floor) {
  using namespace CouplingModelConstants;
  const Real zbar1f = std::max(zbar1, zbar_floor);
  const Real zbar2f = std::max(zbar2, zbar_floor);
  const Real nef = std::max(ne, zbar_floor * ion_number_density_floor);
  const Real telef = std::max(tele, 10.0);
  const Real tionf = std::max(tion, 10.0);
  const Real clog = CoulombLogarithmIonIon(telef, nef, tionf, mi1, mi2, zbar1f, zbar2f);
  const Real mu12 = mi1 * mi2 / (mi1 + mi2 + 1e-100);

  // constant for momentum exchange rate
  // constexpr Real C = 4.0 * std::sqrt(2.0 * M_PI) / 3.0;
  // deflection frequency
  // constexpr Real C = std::sqrt(2.0) * M_PI;
  constexpr Real C = 4.442882938158366; // sqrt(2) * pi
  Real nuij = C * std::sqrt(mu12 / std::pow(kberg * tionf, 3)) / mi1 * zbar1f * zbar1f *
              zbar2f * zbar2f * qe4 * ni2 * clog;

  return nuij;
}

//----------------------------------------------------------------------------------------
//! \fn  Real TauEILandauSpitzer
//! \brief Electron-ion coupling timescale in classical, non-degenerate
//! Landau-Spitzer limit
//! C. Blancard et al. / High Energy Density Physics 9 (2013) 247e250
KOKKOS_FORCEINLINE_FUNCTION
Real TauEILandauSpitzer(const Real &tele, const Real &tion, const Real &ne,
                        const Real &ni, const Real &mi, const Real &zbar,
                        const CoulombLogarithmKind logKind) {
  using namespace CouplingModelConstants;

  const Real clog = EvaluateCoulombLogarithm(tele, tion, ne, ni, mi, zbar, logKind);

  Real tau_ei = 8.0 * std::sqrt(2.0 * pi) * ni * zbar * zbar * qe4;
  tau_ei /= 3.0 * me * mi;
  tau_ei *= clog * std::pow(kberg * tele / me + kberg * tion / mi, -1.5);
  tau_ei = 1.0 / (tau_ei + 1e-15);

  return tau_ei;
} // TauEILandauSpitzer

//----------------------------------------------------------------------------------------
//! \fn  Real SpitzerHarmConductivity
//! \brief Electron thermal conductivity a la Spitzer-Harm
//! see, e.g., Kim Molvig, Andrei N. Simakov, and Erik L. Vold. Classical
//! transport equations for burning gas-metal
//! plasmas. Physics of Plasmas, 21(9):092709, September 2014. ISSN 1070-664X.
//! doi: 10.1063/1.4895666.
//! URL https://doi.org/10.1063/1.4895666
KOKKOS_FORCEINLINE_FUNCTION
Real SpitzerHarmConductivity(const Real &tele, const Real &tion, const Real &ne,
                             const Real &ni, const Real &mi, const Real &zbar,
                             const CoulombLogarithmKind logKind, const Real zbar_floor,
                             const Real ion_number_density_floor) {
  using namespace CouplingModelConstants;

  // floor number densities and zbar
  const Real zbarf = std::max(zbar, zbar_floor);
  const Real nif = std::max(ni, ion_number_density_floor);
  const Real nef = std::max(ne, zbar_floor * ion_number_density_floor);

  const Real clog = EvaluateCoulombLogarithm(tele, tion, nef, nif, mi, zbarf, logKind);

  constexpr Real prefac = eight_ov_pi_32 * kberg72 / qe4 / sqrtme;
  const Real alf = 1. / (1. + 3.3 / zbarf);
  const Real coeff = alf * prefac / zbarf;
  const Real Ke = coeff * std::pow(tele, 5. / 2.) / (clog + 1e-15);

  return Ke;
} // SpitzerHarmConductivity

//----------------------------------------------------------------------------------------
//! \fn  Real BraginskiiConductivity
//! \brief Ion thermal conductivity a la Braginskii
// Ion thermal conductivity a la Braginskii
// Note that this is conductivity arising from ion self-collisions, so should be called
// using a bulk or average composition. We may want to change that...
KOKKOS_FORCEINLINE_FUNCTION
Real BraginskiiConductivity(const Real &tele, const Real &ne, const Real &tion,
                            const Real &ni, const Real &mi, const Real &zbar,
                            const Real zbar_floor, const Real ion_number_density_floor) {
  using namespace CouplingModelConstants;

  // floor number densities and zbar
  const Real zbarf = std::max(zbar, zbar_floor);
  const Real nif = std::max(ni, ion_number_density_floor);
  const Real nef = std::max(ne, zbar_floor * ion_number_density_floor);

  const Real clog = CoulombLogarithmIonIon(tele, nef, tion, mi, mi, zbarf, zbarf);
  const Real zbar4 = zbarf * zbarf * zbarf * zbarf;

  const Real Ki = 3.9 * 3.0 * std::pow(kberg, 3.5) * std::pow(tion, 2.5) /
                  (4.0 * sqrtpi * std::sqrt(mi) * zbar4 * qe4 * clog);

  return Ki;
} // BraginskiiConductivity

//----------------------------------------------------------------------------------------
//! \fn  Real TFMf
//! \brief f(x) from More's Thomas Fermi model fit
//! TODO(JMM): This maybe needs to move at some point if we ever do
//! more complicated ioniztion state stuff.
KOKKOS_FORCEINLINE_FUNCTION
Real TFMf(const Real x) { return x / (1.0 + x + std::sqrt(1.0 + 2.0 * x)); }

//----------------------------------------------------------------------------------------
//! \fn  void ComputeIonizationState
//! \brief Compute the ionization state for a single material given the electron
//! temperature and the material's atomic mass and number. Maybe we can template
//! this function on the ionization model
KOKKOS_FORCEINLINE_FUNCTION
void ComputeIonizationState(const Real &anuc, const Real &znuc, const Real &rho,
                            const Real &tele, Real &zbar, const bool &fully_ionized) {
  using namespace IonizationModelConstants;
  using namespace ThomasFermiMore;
  constexpr Real fourthirds = 4.0 / 3.0;
  Real tev = tele * kbev;
  Real R = rho / (znuc * anuc) + 1e-15;
  Real T0 = tev / std::pow(znuc, fourthirds);
  Real TF = T0 / (1.0 + T0);
  Real A = a1 * std::pow(T0, a2) + a3 * std::pow(T0, a4);
  Real B = -std::exp(b0 + b1 * TF + b2 * std::pow(TF, 7));
  Real C = c1 * TF + c2;
  Real Q1 = A * std::pow(R, B);
  Real Q = std::pow(std::pow(R, C) + std::pow(Q1, C), 1.0 / C);
  Real x = alpha * std::pow(Q, beta);
  zbar = (rho > 0.) * TFMf(x) * znuc;
  zbar = fully_ionized * znuc + (1.0 - fully_ionized) * zbar;

} // ComputeIonizationState
} // namespace Ionization

#endif // IONIZATION_IONIZATION_BASE_HPP_
