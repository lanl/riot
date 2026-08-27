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
//========================================================================================
// This file ports routines from the Expander/polylogarithm repository
// (https://github.com/Expander/polylogarithm).
//
// Polylogarithm is licenced under the MIT License. See licenses/mit_polylogarithm.txt
// file for details
//========================================================================================
#ifndef RADIATION_TRANSPORT_TRANSPORT_UTILS_TRANSPORT_UTILS_HPP_
#define RADIATION_TRANSPORT_TRANSPORT_UTILS_TRANSPORT_UTILS_HPP_
// This file was made in part with generative AI.

// C++ includes
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// C includes
#include <string.h>

// Parthenon includes
#include <pack/sparse_pack/make_pack_descriptor.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Riot includes
#include "radiation_transport/angular_grids/angular_grid_utils.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

// Curvilinear geometry?
static constexpr bool do_angular_fluxes =
    parthenon::IsCoord<parthenon::UniformCylindrical>() ||
    parthenon::IsCoord<parthenon::UniformSpherical>();

// Initialization nulls
static const std::string snull = "UNINITIALIZED STRING";
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Null() {
  return std::numeric_limits<T>::quiet_NaN();
}
template <>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Null<int>() {
  return std::numeric_limits<int>::max();
}

#define DNAME(name) static constexpr char name[] = #name
namespace container_names {
// Explcit MeshData Registers
DNAME(r0);
DNAME(r1);
DNAME(a0);
DNAME(ro);
// Jacobi MeshData Registers
DNAME(ropac);
DNAME(rbase);
DNAME(riter);
DNAME(rout);
DNAME(ubase);
} // namespace container_names
#undef DNAME

//----------------------------------------------------------------------------------------
// ...Flux stencils
enum class FluxType { hll, upwind, null };
// ...Mean types
enum class StencilType { minimum, harmonic, arithmetic, null };

//----------------------------------------------------------------------------------------
//! \fn  int GI
//  \brief Returns group index given group-angle index n
KOKKOS_INLINE_FUNCTION
static int GI(const int nangles, const int n) { return n / nangles; }

//----------------------------------------------------------------------------------------
//! \fn  int AI
//  \brief Returns angle index given group-angle index n
KOKKOS_INLINE_FUNCTION
static int AI(const int nangles, const int n) { return n % nangles; }

//----------------------------------------------------------------------------------------
//! \fn int GAI
//! \brief Returns flattened index supplied a group and angle index
KOKKOS_INLINE_FUNCTION
static int GAI(const int nangles, const int gg, const int aa) {
  return gg * nangles + aa;
}

//----------------------------------------------------------------------------------------
//! \fn const std::shared_ptr<StateDescriptor> &GetRadPackage
//! \brief Return the active radiation transport package ("explicit" or "jacobi") from a
//! package collection, selecting based on the "riot" package's do_explicit flag.  Only
//! valid when radiation is enabled.
inline const std::shared_ptr<StateDescriptor> &
GetRadPackage(const parthenon::Packages_t &packages) {
  const bool do_explicit = packages.Get("riot")->Param<bool>("do_explicit");
  return packages.Get(do_explicit ? "explicit" : "jacobi");
}

//----------------------------------------------------------------------------------------
// Utility structs
struct FluxUtils {
  Real beta = 0.0;   // constant multiplying tauc for constructing Rusanov flux
  Real taumax = 0.0; // tau max for Rusanov flux stability
};

struct RootUtils {
  Real tol = 0.0; // coupling tolerance for advanced temperature
  int titer = 0;  // max #iter for advanced temp root find
};

struct UnitUtils {
  Real c = 0.0;         // speed of light
  Real planck = 0.0;    // planck's constant
  Real boltzmann = 0.0; // boltzmann's constant
  Real arad = 0.0;      // radiation constant
};

//----------------------------------------------------------------------------------------
//! \fn  Real OpacityStencil
//! \brief Combines two neighboring opacities into a face value (minimum, harmonic, or
//! arithmetic mean) for the Rusanov flux optical-depth weighting
template <StencilType ST = StencilType::minimum>
KOKKOS_FORCEINLINE_FUNCTION static Real OpacityStencil(const Real val1, const Real val2) {
  if constexpr (ST == StencilType::minimum) {
    return std::min(val1, val2);
  } else if constexpr (ST == StencilType::harmonic) {
    const Real sum = val1 + val2;
    return (sum == 0.0) ? 0.0 : 2.0 * val1 * val2 / sum;
  } else if constexpr (ST == StencilType::arithmetic) {
    return 0.5 * (val1 + val2);
  }
  return Null<Real>();
}

//----------------------------------------------------------------------------------------
//! Polylog functions are taken Expander/polylogarithm (licensed under MIT license:
//! https://github.com/Expander/polylogarithm/blob/master/LICENSE) and converted to
//! KOKKOS_INLINE_FUNCTIONs by @pdmullen on 05/16/25...

//----------------------------------------------------------------------------------------
//! \fn  Real Li2
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
static Real Li2(Real x) {
  const Real PI = 3.1415926535897932;
  const Real P[] = {0.9999999999999999502e+0, -2.6883926818565423430e+0,
                    2.6477222699473109692e+0, -1.1538559607887416355e+0,
                    2.0886077795020607837e-1, -1.0859777134152463084e-2};
  const Real Q[] = {1.0000000000000000000e+0, -2.9383926818565635485e+0,
                    3.2712093293018635389e+0, -1.7076702173954289421e+0,
                    4.1596017228400603836e-1, -3.9801343754084482956e-2,
                    8.2743668974466659035e-4};

  Real y = 0, r = 0, s = 1;

  // transform to [0, 1/2]
  if (x < -1) {
    const Real l = std::log(1 - x);
    y = 1 / (1 - x);
    r = -PI * PI / 6 + l * (0.5 * l - std::log(-x));
    s = 1;
  } else if (x == -1) {
    return -PI * PI / 12;
  } else if (x < 0) {
    const Real l = std::log1p(-x);
    y = x / (x - 1);
    r = -0.5 * l * l;
    s = -1;
  } else if (x == 0) {
    return x;
  } else if (x < 0.5) {
    y = x;
    r = 0;
    s = 1;
  } else if (x < 1) {
    y = 1 - x;
    r = PI * PI / 6 - std::log(x) * std::log1p(-x);
    s = -1;
  } else if (x == 1) {
    return PI * PI / 6;
  } else if (x < 2) {
    const Real l = std::log(x);
    y = 1 - 1 / x;
    r = PI * PI / 6 - l * (std::log(y) + 0.5 * l);
    s = 1;
  } else {
    const Real l = std::log(x);
    y = 1 / x;
    r = PI * PI / 3 - 0.5 * l * l;
    s = -1;
  }

  const Real y2 = y * y;
  const Real y4 = y2 * y2;
  const Real p = P[0] + y * P[1] + y2 * (P[2] + y * P[3]) + y4 * (P[4] + y * P[5]);
  const Real q =
      Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3]) + y4 * (Q[4] + y * Q[5] + y2 * Q[6]);

  return r + s * y * p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li3Neg
//! \brief Li_3(x) for x in [-1,0]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li3Neg(Real x) {
  const Real cp[] = {0.9999999999999999795e+0, -2.0281801754117129576e+0,
                     1.4364029887561718540e+0, -4.2240680435713030268e-1,
                     4.7296746450884096877e-2, -1.3453536579918419568e-3};
  const Real cq[] = {1.0000000000000000000e+0, -2.1531801754117049035e+0,
                     1.6685134736461140517e+0, -5.6684857464584544310e-1,
                     8.1999463370623961084e-2, -4.0756048502924149389e-3,
                     3.4316398489103212699e-5};

  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real p = cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5]);
  const Real q = cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) +
                 x4 * (cq[4] + x * cq[5] + x2 * cq[6]);

  return x * p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li3Pos
//! \brief Li_3(x) for x in [0,1/2]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li3Pos(Real x) {
  const Real cp[] = {0.9999999999999999893e+0, -2.5224717303769789628e+0,
                     2.3204919140887894133e+0, -9.3980973288965037869e-1,
                     1.5728950200990509052e-1, -7.5485193983677071129e-3};
  const Real cq[] = {1.0000000000000000000e+0, -2.6474717303769836244e+0,
                     2.6143888433492184741e+0, -1.1841788297857667038e+0,
                     2.4184938524793651120e-1, -1.8220900115898156346e-2,
                     2.4927971540017376759e-4};

  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real p = cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5]);
  const Real q = cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) +
                 x4 * (cq[4] + x * cq[5] + x2 * cq[6]);

  return x * p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li3
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
static Real Li3(Real x) {
  const Real zeta2 = 1.6449340668482264;
  const Real zeta3 = 1.2020569031595943;

  // transformation to [-1,0] and [0,1/2]
  if (x < -1) {
    const Real l = std::log(-x);
    return Li3Neg(1 / x) - l * (zeta2 + 1.0 / 6 * l * l);
  } else if (x == -1) {
    return -0.75 * zeta3;
  } else if (x < 0) {
    return Li3Neg(x);
  } else if (x == 0) {
    return x;
  } else if (x < 0.5) {
    return Li3Pos(x);
  } else if (x == 0.5) {
    return 0.53721319360804020;
  } else if (x < 1) {
    const Real l = std::log(x);
    return -Li3Neg(1 - 1 / x) - Li3Pos(1 - x) + zeta3 +
           l * (zeta2 + l * (-0.5 * std::log1p(-x) + 1.0 / 6 * l));
  } else if (x == 1) {
    return zeta3;
  } else if (x < 2) {
    const Real l = std::log(x);
    return -Li3Neg(1 - x) - Li3Pos(1 - 1 / x) + zeta3 +
           l * (zeta2 + l * (-0.5 * std::log(x - 1) + 1.0 / 6 * l));
  } else { // x >= 2.0
    const Real l = std::log(x);
    return Li3Pos(1 / x) + l * (2 * zeta2 - 1.0 / 6 * l * l);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li4Neg
//! \brief Li_4(x) for x in [-1,0]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li4Neg(Real x) {
  const Real cp[] = {0.9999999999999999952e+0, -1.8532099956062184217e+0,
                     1.1937642574034898249e+0, -3.1817912243893560382e-1,
                     3.2268284189261624841e-2, -8.3773570305913850724e-4};
  const Real cq[] = {1.0000000000000000000e+0, -1.9157099956062165688e+0,
                     1.3011504531166486419e+0, -3.7975653506939627186e-1,
                     4.5822723996558783670e-2, -1.8023912938765272341e-3,
                     1.0199621542882314929e-5};

  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real p = cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5]);
  const Real q = cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) +
                 x4 * (cq[4] + x * cq[5] + x2 * cq[6]);

  return x * p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li4Half
//! \brief Li_4(x) for x in [0,1/2]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li4Half(Real x) {
  const Real cp[] = {1.0000000000000000414e+0, -2.0588072418045364525e+0,
                     1.4713328756794826579e+0, -4.2608608613069811474e-1,
                     4.2975084278851543150e-2, -6.8314031819918920802e-4};
  const Real cq[] = {1.0000000000000000000e+0, -2.1213072418045207223e+0,
                     1.5915688992789175941e+0, -5.0327641401677265813e-1,
                     6.1467217495127095177e-2, -1.9061294280193280330e-3};

  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real p = cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5]);
  const Real q = cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5]);

  return x * p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li4Mid
//! \brief Li_4(x) for x in [1/2,8/10]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li4Mid(Real x) {
  const Real cp[] = {3.2009826406098890447e-9,  9.9999994634837574160e-1,
                     -2.9144851228299341318e+0, 3.1891031447462342009e+0,
                     -1.6009125158511117090e+0, 3.5397747039432351193e-1,
                     -2.5230024124741454735e-2};
  const Real cq[] = {1.0000000000000000000e+0, -2.9769855248411488460e+0,
                     3.3628208295110572579e+0, -1.7782471949702788393e+0,
                     4.3364007973198649921e-1, -3.9535592340362510549e-2,
                     5.7373431535336755591e-4};

  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  const Real p = cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) +
                 x4 * (cp[4] + x * cp[5] + x2 * cp[6]);
  const Real q = cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) +
                 x4 * (cq[4] + x * cq[5] + x2 * cq[6]);

  return p / q;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li4One
//! \brief Li_4(x) for x in [8/10,1]
KOKKOS_FORCEINLINE_FUNCTION
static Real Li4One(Real x) {
  const Real zeta2 = 1.6449340668482264;
  const Real zeta3 = 1.2020569031595943;
  const Real zeta4 = 1.0823232337111382;
  const Real l = std::log(x);
  const Real l2 = l * l;

  return zeta4 +
         l * (zeta3 + l * (0.5 * zeta2 +
                           l * (11.0 / 36 - 1.0 / 6 * std::log(-l) +
                                l * (-1.0 / 48 +
                                     l * (-1.0 / 1440 +
                                          l2 * (1.0 / 604800 - 1.0 / 91445760 * l2))))));
}

//----------------------------------------------------------------------------------------
//! \fn  Real Li4
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
static Real Li4(Real x) {
  const Real zeta2 = 1.6449340668482264;
  const Real zeta4 = 1.0823232337111382;

  Real app = 0, rest = 0, sgn = 1;

  // transform x to [-1,1]
  if (x < -1) {
    const Real l = std::log(-x);
    const Real l2 = l * l;
    x = 1 / x;
    rest = -7.0 / 4 * zeta4 + l2 * (-0.5 * zeta2 - 1.0 / 24 * l2);
    sgn = -1;
  } else if (x == -1) {
    return -7.0 / 8 * zeta4;
  } else if (x == 0) {
    return x;
  } else if (x < 1) {
    rest = 0;
    sgn = 1;
  } else if (x == 1) {
    return zeta4;
  } else { // x > 1
    const Real l = std::log(x);
    const Real l2 = l * l;
    x = 1 / x;
    rest = 2 * zeta4 + l2 * (zeta2 - 1.0 / 24 * l2);
    sgn = -1;
  }

  if (x < 0) {
    app = Li4Neg(x);
  } else if (x < 0.5) {
    app = Li4Half(x);
  } else if (x < 0.8) {
    app = Li4Mid(x);
  } else { // x <= 1
    app = Li4One(x);
  }

  return rest + sgn * app;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Emissivity
//! \brief Computes the group-integrated Planck emissivity (gray when ngroups == 1)
KOKKOS_FORCEINLINE_FUNCTION static Real Emissivity(const int gg, const Real temp,
                                                   const parthenon::ParArray1D<Real> fbnd,
                                                   const int ngroups,
                                                   const UnitUtils uu) {
  const Real gray = uu.arad * SQR(SQR(temp));
  if (ngroups == 1) return gray;

  // Perform integral over Planck function
  const Real hikt = uu.planck / (uu.boltzmann * temp);
  const Real fac = 15.0 / SQR(SQR(M_PI));
  const Real prefac = gray * fac;
  if (fbnd(gg) == 0.0) { // 0-->vf bin
    const Real lnp = hikt * fbnd(gg + 1);
    const Real lnpsq = SQR(lnp);
    const Real xp = std::exp(-lnp);
    const Real ee = 1.0 / fac + lnpsq * lnp * std::log(1.0 - xp) - 3.0 * lnpsq * Li2(xp) -
                    6.0 * lnp * Li3(xp) - 6.0 * Li4(xp);
    return std::max(prefac * ee, 0.0);
  } else if (fbnd(gg + 1) == std::numeric_limits<Real>::infinity()) { // vf-->infty bin
    const Real lnm = hikt * fbnd(gg);
    const Real lnmsq = SQR(lnm);
    const Real xm = std::exp(-lnm);
    const Real ee = -lnmsq * lnm * std::log(1.0 - xm) + 3.0 * lnmsq * Li2(xm) +
                    6.0 * lnm * Li3(xm) + 6.0 * Li4(xm);
    return std::max(prefac * ee, 0.0);
  } else { // vfm-->vfp bin
    const Real lnm = hikt * fbnd(gg);
    const Real lnp = hikt * fbnd(gg + 1);
    const Real lnmsq = SQR(lnm);
    const Real lnpsq = SQR(lnp);
    const Real xm = std::exp(-lnm);
    const Real xp = std::exp(-lnp);
    const Real ee = lnpsq * lnp * std::log(1.0 - xp) - lnmsq * lnm * std::log(1.0 - xm) -
                    3.0 * (lnpsq * Li2(xp) - lnmsq * Li2(xm)) -
                    6.0 * (lnp * Li3(xp) - lnm * Li3(xm)) - 6.0 * (Li4(xp) - Li4(xm));
    return std::max(prefac * ee, 0.0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real EmissivityDT
//! \brief Temperature derivative deps_g/dT of the group-INTEGRATED Planck emissivity
//! eps_g = Emissivity(gg, T) (NOT a monochromatic Planck function; sum_g eps_g = a_r T^4,
//! and the gray ngroups==1 result is 4 a_r T^3).  Writing eps_g = a_r T^4 * F_g with F_g
//! the normalized Planck fraction over the band (x = h nu / kT, dx/dT = -x/T), the
//! derivative of the band integral reduces to the integrand at the band edges -- no
//! polylogarithms.  The integral part reuses Emissivity(); the boundary part is
//! elementary:
//!   deps_g/dT = 4 eps_g / T - (a_r T^3)(15/pi^4)[ x_hi^4/(e^x_hi-1) - x_lo^4/(e^x_lo-1)
//!   ].
KOKKOS_FORCEINLINE_FUNCTION static Real
EmissivityDT(const int gg, const Real temp, const parthenon::ParArray1D<Real> fbnd,
             const int ngroups, const UnitUtils uu) {
  const Real arT3 = uu.arad * temp * temp * temp;
  if (ngroups == 1) return 4.0 * arT3; // gray: d(a_r T^4)/dT

  // Planck integrand x^4/(e^x - 1), guarded at the x -> 0 and x -> infinity band edges
  // (both vanish: ~x^3 as x -> 0, exponentially as x -> infinity).
  const Real fac = 15.0 / SQR(SQR(M_PI));
  const Real hikt = uu.planck / (uu.boltzmann * temp);
  auto edge = [](const Real x) -> Real {
    if (x <= 0.0) return 0.0;
    const Real ex = std::expm1(x); // e^x - 1
    return (ex > 0.0) ? SQR(SQR(x)) / ex : 0.0;
  };
  const Real xlo = (fbnd(gg) == 0.0) ? 0.0 : hikt * fbnd(gg);
  const Real xhi = (fbnd(gg + 1) == std::numeric_limits<Real>::infinity())
                       ? 0.0 // integrand already vanishes at the open upper edge
                       : hikt * fbnd(gg + 1);
  const Real bnd = fac * (edge(xhi) - edge(xlo));

  // deps_g/dT = 4 eps_g / T - a_r T^3 * bnd
  const Real emiss = Emissivity(gg, temp, fbnd, ngroups, uu);
  return 4.0 * emiss / temp - arT3 * bnd;
}

//----------------------------------------------------------------------------------------
//! \fn  Real InternalEnergy
//! \brief Computes bulk material energy from material densities and material sies
template <typename EOS, typename EOSMAP, typename V1>
KOKKOS_FORCEINLINE_FUNCTION static Real
InternalEnergy(const Real temp, const int nm1, const EOS eos, const EOSMAP map,
               const V1 vu, const int b, const int k, const int j, const int i) {
  namespace cm = cell_variables::material_averaged;
  namespace ccmat = cell_variables::cell_averaged::mat;

  Real ubulk = 0.0;
  for (int m = 0; m <= nm1; ++m) {
    const Real &rhom = vu(b, cm::rho(m), k, j, i);
    const Real &rhob = vu(b, ccmat::rho(m), k, j, i);
    auto &eosm = eos(map(vu(b, ccmat::rho(m)).sparse_id));
    ubulk += rhob * eosm.InternalEnergyFromDensityTemperature(rhom, temp);
  }
  return ubulk;
}

//----------------------------------------------------------------------------------------
//! \fn  Real PolyRoot
//! \brief Constructs polynomial for temperature root find
template <typename V1, typename EOS, typename EOSMAP, typename V2>
KOKKOS_FORCEINLINE_FUNCTION static Real
PolyRoot(const Real tp1, const Real ubulk, const UnitUtils uu, const Real cdt,
         // Radiation
         const int ngroups, const parthenon::ParArray1D<Real> fbnd, const V1 vr,
         // Fluid
         const int nm1, const EOS eos, const EOSMAP map, const V2 vu,
         // Indexing
         const int b, const int k, const int j, const int i) {
  namespace ccrad = cell_variables::cell_averaged::rad;

  Real term = 0.0;
  for (int gg = 0; gg < ngroups; ++gg) {
    const Real cdtsiga = cdt * vr(b, ccrad::aa(gg), k, j, i);
    const Real &aa1 = vr(b, ccrad::s1(gg), k, j, i);
    const Real &aa2 = vr(b, ccrad::s2(gg), k, j, i);
    const Real &aa3 = vr(b, ccrad::s3(gg), k, j, i);
    const Real &ee = Emissivity(gg, tp1, fbnd, ngroups, uu);
    term += ee * cdtsiga * (aa1 * aa3 - 1.0) + cdtsiga * aa2 * aa3;
  }
  const Real &aa = term;
  const Real &bb = ubulk;
  const Real &cc = InternalEnergy(tp1, nm1, eos, map, vu, b, k, j, i);
  const Real iwght = 1.0 / std::max(1.0e-12, std::abs(aa) + std::abs(bb) + std::abs(cc));
  return iwght * (aa + bb - cc);
}

//----------------------------------------------------------------------------------------
//! \fn  Real AdvancedTemperature
//! \brief Computes advanced temperature for radiation source term
template <bool jacobi = true, typename V1, typename EOS, typename EOSMAP, typename V2>
KOKKOS_FORCEINLINE_FUNCTION static bool
AdvancedTemperature(Real &temp, const Real &tbase, const bool &verbose, const Real &tflr,
                    // Utils and timestep
                    const RootUtils ru, const UnitUtils uu, const Real cdt,
                    // Radiation
                    const int ngroups, const parthenon::ParArray1D<Real> fbnd,
                    const V1 vr,
                    // Fluid
                    const int nm1, const EOS eos, const EOSMAP map, const V2 vu,
                    // Indexing
                    const int b, const int k, const int j, const int i) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  // Internal energy at T^m
  Real ubase = Null<Real>();

  [[maybe_unused]] auto &vu_ = vu;
  if constexpr (jacobi) {
    ubase = vu_(b, ccbulk::internal_energy(), k, j, i);
  } else {
    ubase = InternalEnergy(tbase, nm1, eos, map, vu, b, k, j, i);
  }

  // See if guess itself is within tolerance
  const Real yg =
      PolyRoot(temp, ubase, uu, cdt, ngroups, fbnd, vr, nm1, eos, map, vu, b, k, j, i);
  if (std::abs(yg) <= ru.tol) return true;

  // Attempt to bound the root
  constexpr int NBND = 6;
  constexpr Real tmag[NBND] = {1.1, 2.0, 1e1, 1e2, 1e4, 1e8};
  int mag = 0;
  Real ta = Null<Real>(), tb = Null<Real>();
  Real ya = std::numeric_limits<Real>::max(), yb = std::numeric_limits<Real>::max();
  while (ya * yb > 0.0) {
    if (mag == NBND) {
      if (verbose) PARTHENON_WARN("Cannot bound temperature root!");
      return false;
    }
    ta = temp / tmag[mag];
    tb = temp * tmag[mag];
    ya = PolyRoot(ta, ubase, uu, cdt, ngroups, fbnd, vr, nm1, eos, map, vu, b, k, j, i);
    yb = PolyRoot(tb, ubase, uu, cdt, ngroups, fbnd, vr, nm1, eos, map, vu, b, k, j, i);
    mag++;
  }

  // Regula falsi
  int iter_count = 0;
  int b1 = 0;
  int b2 = 0;
  const Real sign = (ya < 0.0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;
  // NOTE: ta, tb > 0 (temperatures), so the relative-tolerance test
  // |tb - ta| / (0.5 (ta + tb)) > tol is rearranged into a multiply.
  while ((std::abs(tb - ta) > ru.tol * 0.5 * (ta + tb)) &&   // \Delta T is below tol
         (std::abs(ya) > ru.tol || std::abs(yb) > ru.tol) && // ya and yb below tol
         (iter_count < ru.titer)) {
    // Evaluate tc
    Real tc = (ta * yb - tb * ya) / (yb - ya);

    // Guard against roundoff
    if (tc == ta) {
      tb = ta;
      iter_count++;
      continue;
    } else if (tc == tb) {
      ta = tb;
      iter_count++;
      continue;
    }

    // Compute internal energy at temp tc
    Real yc = sign;
    yc *= PolyRoot(tc, ubase, uu, cdt, ngroups, fbnd, vr, nm1, eos, map, vu, b, k, j, i);
    if (yc > 0.0) {
      tb = tc;
      yb = yc;
      b1++;
      ya *= (b1 > 1 ? 0.5 : 1.0);
      b2 = 0;
    } else if (yc < 0.0) {
      ta = tc;
      ya = yc;
      b2++;
      yb *= (b2 > 1 ? 0.5 : 1.0);
      b1 = 0;
    } else {
      ta = tc;
      tb = tc;
    }
    iter_count++;
  }

  // Return root or fail
  if (iter_count >= ru.titer) {
    if (verbose) PARTHENON_WARN("Radiation temperature root find cannot converge!");
    return false;
  } else {
    ta = 0.5 * (ta + tb);
    if (ta > tflr) {
      temp = ta;
      return true;
    } else {
      if (verbose) PARTHENON_WARN("Radiation temperature root unphysical!");
      return false;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn int GetBoundaryPackDescriptorMap
//! \brief Returns a map of pack descriptors to be used with boundary conditions.
template <class... var_ts>
using map_bc_pack_descriptor_t =
    std::unordered_map<bool, typename SparsePack<var_ts...>::Descriptor>;

template <class... var_ts>
map_bc_pack_descriptor_t<var_ts...>
GetBoundaryPackDescriptorMap(std::shared_ptr<MeshBlockData<Real>> &rc) {
  using namespace parthenon;
  map_bc_pack_descriptor_t<var_ts...> my_map;
  std::vector<parthenon::MetadataFlag> flags{parthenon::Metadata::FillGhost};
  std::set<PDOpt> opts{PDOpt::Coarse};
  my_map.emplace(
      std::make_pair(true, MakePackDescriptor<var_ts...>(rc.get(), flags, opts)));
  my_map.emplace(std::make_pair(false, MakePackDescriptor<var_ts...>(rc.get(), flags)));
  return my_map;
}

//----------------------------------------------------------------------------------------
//! \fn  void SetMomentsMesh
//! \brief
inline void SetMomentsMesh(MeshData<Real> *md) {
  namespace ccrad = cell_variables::cell_averaged::rad;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc =
      MakePackDescriptor<ccrad::intensity, ccrad::moments>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  if (v.GetMaxNumberOfVars() == 0) return;

  const auto rad_pkg = GetRadPackage(pm->packages);
  const auto agrid = GetAngularGridArrays(rad_pkg);
  const auto wt = agrid.weights;
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");

  // Compute moments (currently only the radiation energy density)
  auto idx_space = RiotFlatLoop::GetIndexSpace(IndexDomain::entire, v.GetNBlocks(), md);
  RiotFlatLoop::four_d(
      "SetMoments", idx_space,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int gg = 0; gg < ngroups; ++gg) {
          v(b, ccrad::moments(gg), k, j, i) = 0.0;
          for (int aa = 0; aa < nangles; ++aa) {
            const int n = GAI(nangles, gg, aa);
            v(b, ccrad::moments(gg), k, j, i) +=
                v(b, ccrad::intensity(n), k, j, i) * wt(aa);
          }
        }
      });
}

#endif // RADIATION_TRANSPORT_TRANSPORT_UTILS_TRANSPORT_UTILS_HPP_
