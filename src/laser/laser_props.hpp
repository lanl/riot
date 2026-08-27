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
#ifndef LASER_LASER_PROPS_HPP_
#define LASER_LASER_PROPS_HPP_

#include <vector>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace Laser {

namespace sample {
enum LaserSample { x, y, z, nx, ny, nz, wgt, nvalues };
}

// class to store and provide laser energy between two time points.
// works by representing the cumulative energy released on a
// uniform grid in t, interpolating at start and stop points
// and then returning the difference, which is the integral of
// the power over the time interval
class LaserEnergy {
 public:
  LaserEnergy() : initialized(false) {}
  LaserEnergy(std::vector<Real> &t, std::vector<Real> &p) : initialized(true) {
    InterpToUniform(t, p);
  }

  Real GetEnergy(const Real tlo, const Real thi) const;
  bool initialized;

 private:
  Real t0, t1, dt;
  std::vector<Real> energy;

  Real interp(const Real tsamp, const int i, std::vector<Real> &t, std::vector<Real> &p) {
    const Real w = (tsamp - t[i]) / (t[i + 1] - t[i]);
    return (1.0 - w) * p[i] + w * p[i + 1];
  }

  Real integral_average(std::vector<Real> &t, std::vector<Real> &p, const Real tlo,
                        const Real thi);
  void InterpToUniform(std::vector<Real> &t, std::vector<Real> &p);
};

class LaserProfile {
 public:
  enum class ProfileType { flat, super_gaussian };
  LaserProfile(ParameterInput *pin, const std::string &name);

  inline Real NormalizedIntegral(const Real r0, const Real r1) {
    return Integrate(r0, r1) / total;
  }

  inline std::pair<Real, Real> Axes() const { return {rmax * smajor, rmax * sminor}; }

  inline Real Scale() const { return rmax; }

 private:
  ProfileType profile;
  Real smajor, sminor, rmax, total, gorder;

  inline Real super_gaussian(const Real x) { return std::exp(-std::pow(x * x, gorder)); }

  template <typename F>
  Real quad(F &f, const Real x0, const Real x1) {
    // Romberg integration
    constexpr int max_steps = 10;
    constexpr Real tol = 1.e-12;
    Real R1[max_steps], R2[max_steps];
    Real *Rp = &R1[0];
    Real *Rc = &R2[0];
    Real h = x1 - x0;
    Rp[0] = 0.5 * h * (f(x0) + f(x1));
    for (int i = 1; i < max_steps; i++) {
      h *= 0.5;
      Real c = 0.0;
      int n = 1 << (i - 1);
      for (int j = 1; j <= n; j++) {
        c += f(x0 + (2 * j - 1) * h);
      }
      Rc[0] = h * c + 0.5 * Rp[0];
      for (int j = 1; j <= i; j++) {
        int nk = 1 << (2 * j); // 4^j
        Rc[j] = (nk * Rc[j - 1] - Rp[j - 1]) / (nk - 1);
      }
      if (i > 1 && std::abs(Rp[i - 1] - Rc[i]) < tol) {
        return Rc[i];
      }

      Real *temp = Rp;
      Rp = Rc;
      Rc = temp;
    }
    return Rp[max_steps - 1];
  }

  Real Integrate(const Real r0, const Real r1);
  Real SampleWidth(const Real frac);
};

class LaserGrid {
 public:
  LaserGrid(ParameterInput *pin, const std::string &name);

  inline std::tuple<Real, Real, Real> LocationAndWeight(const int i) const {
    return {rgrid[i], thgrid[i], wgt[i]};
  }

  inline auto NumSamples() const { return rgrid.size(); }

  inline auto Axes() const { return lp.Axes(); }

 private:
  LaserProfile lp;
  std::vector<Real> rgrid, thgrid, wgt;
};

struct LaserInfo {
  std::array<std::array<std::vector<Real>, sample::nvalues>, 6> face_pts;
  std::array<std::vector<int>, 6> id;
  std::vector<LaserEnergy> energy;
  std::vector<Real> wavelength;
};

} // namespace Laser

#endif // LASER_LASER_PROPS_HPP_
