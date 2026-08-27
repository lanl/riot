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
// This file was made in part with generative AI.

#include <random>

#include "riot_pgen/pgen.hpp"

namespace rm {

using parthenon::ParArray2D;
using parthenon::ParArray3D;

class Interface {
 public:
  Interface(ParameterInput *pin)
      : nmodes(pin->GetOrAddInteger("rm", "nmodes", 6)),
        mode_min(pin->GetOrAddInteger("rm", "mode_min", 4)),
        x0(pin->GetOrAddReal("rm", "x0", 0.2)),
        nfine(pin->GetOrAddReal("rm", "nfine", 16)),
        alpha(ParArray2D<Real>("alpha", nmodes, nmodes)),
        beta(ParArray2D<Real>("beta", nmodes, nmodes)),
        gamma(ParArray2D<Real>("gamma", nmodes, nmodes)),
        delta(ParArray2D<Real>("delta", nmodes, nmodes)) {

    ndim = 1;
    if (pin->GetInteger("parthenon/mesh", "nx3") > 1)
      ndim = 3;
    else if (pin->GetInteger("parthenon/mesh", "nx2") > 1)
      ndim = 2;

    auto amp = pin->GetOrAddReal("rm", "amp", 3.e-3);
    auto seed = pin->GetOrAddInteger("rm", "seed", 123);

    // setup the rng
    std::mt19937 rng_engine(seed);
    std::uniform_real_distribution<Real> uniform(-1.0, 1.0);
    auto rng = [&]() { return uniform(rng_engine); };

    // we'll set them on host
    auto a_h = alpha.GetHostMirror();
    auto b_h = beta.GetHostMirror();
    auto g_h = gamma.GetHostMirror();
    auto d_h = delta.GetHostMirror();

    // now fill in with random numbers
    for (int n = 0; n < nmodes; n++) {
      for (int m = 0; m < nmodes; m++) {
        a_h(n, m) = amp * rng();
        b_h(n, m) = amp * rng();
        g_h(n, m) = amp * rng();
        d_h(n, m) = amp * rng();
      }
    }

    // and deep copy back to device
    alpha.DeepCopy(a_h);
    beta.DeepCopy(b_h);
    gamma.DeepCopy(g_h);
    delta.DeepCopy(d_h);
  }

  KOKKOS_INLINE_FUNCTION
  Real Position(const Real y, const Real z) const {
    Real x = x0;
    for (int n = 0; n < nmodes; n++) {
      const Real yl = 2.0 * M_PI * (n + mode_min) * y;
      const Real cyl = std::cos(yl);
      const Real syl = std::sin(yl);
      if (ndim == 2) {
        x += nmodes * (alpha(n, 0) * cyl + gamma(n, 0) * syl);
      } else {
        for (int m = 0; m < nmodes; m++) {
          const Real zl = 2.0 * M_PI * (m + mode_min) * z;
          const Real czl = std::cos(zl);
          const Real szl = std::sin(zl);
          x += alpha(n, m) * cyl * czl + beta(n, m) * cyl * szl +
               gamma(n, m) * syl * czl + delta(n, m) * syl * szl;
        }
      }
    }
    return x;
  }

  KOKKOS_INLINE_FUNCTION
  Real Vfrac(const Real xlo, const Real xhi, const Real ylo, const Real yhi,
             const Real zlo, const Real zhi) const {
    const Real dx = (xhi - xlo) / nfine;
    const Real dy = (yhi - ylo) / nfine;
    const Real dz = (zhi - zlo) / nfine;

    if (ndim == 1) {
      int inside = 0;
      for (int i = 0; i < nfine; i++) {
        const Real x = xlo + (i + 0.5) * dx;
        inside += (x < x0);
      }
      return (1.0 * inside) / nfine;
    } else if (ndim == 2) {
      int inside = 0;
      for (int j = 0; j < nfine; j++) {
        const Real y = ylo + (j + 0.5) * dy;
        Real xint = Position(y, 0.0);
        for (int i = 0; i < nfine; i++) {
          const Real x = xlo + (i + 0.5) * dx;
          inside += (x < xint);
        }
      }
      return (1.0 * inside) / (nfine * nfine);
    } else {

      int inside = 0;
      for (int j = 0; j < nfine; j++) {
        const Real y = ylo + (j + 0.5) * dy;
        for (int k = 0; k < nfine; k++) {
          const Real z = zlo + (k + 0.5) * dz;
          const Real xint = Position(y, z);
          for (int i = 0; i < nfine; i++) {
            const Real x = xlo + (i + 0.5) * dx;
            inside += (x < xint);
          }
        }
      }
      return (1.0 * inside) / (nfine * nfine * nfine);
    }
    return 0.0;
  }

 private:
  const int nmodes;
  const int mode_min;
  const Real x0;
  const int nfine;
  ParArray2D<Real> alpha, beta, gamma, delta;
  int ndim;
};

} // namespace rm
