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

#include "laser_props.hpp"

namespace Laser {

//
// LaserEnergy members
//
Real LaserEnergy::GetEnergy(const Real tlo, const Real thi) const {
  Real elo, ehi;
  // get E at tlo
  if (tlo >= t0 && tlo < t1) {
    int i = (tlo - t0) / dt;
    Real wgt = (tlo - i * dt) / dt;
    elo = (1.0 - wgt) * energy[i] + wgt * energy[i + 1];
  } else if (tlo >= t1) {
    elo = energy.back();
  } else {
    elo = 0.0;
  }

  // get E at thi
  if (thi >= t0 && thi < t1) {
    int i = (thi - t0) / dt;
    Real wgt = (thi - i * dt) / dt;
    ehi = (1.0 - wgt) * energy[i] + wgt * energy[i + 1];
  } else if (thi < tlo) {
    ehi = 0.0;
  } else {
    ehi = energy.back();
  }
  return ehi - elo;
}

Real LaserEnergy::integral_average(std::vector<Real> &t, std::vector<Real> &p,
                                   const Real tlo, const Real thi) {
  int i = 0;
  while (t[i + 1] < tlo)
    i++;
  int j = i;
  while (t[j + 1] < thi)
    j++;

  if (i == j) {
    auto plo = interp(tlo, i, t, p);
    auto phi = interp(thi, i, t, p);
    return 0.5 * (plo + phi);
  }

  return (0.5 * (t[j] - tlo) * (p[j] + interp(tlo, i, t, p)) +
          0.5 * (thi - t[j]) * (interp(thi, j, t, p) + p[j])) /
         (thi - tlo);
}

void LaserEnergy::InterpToUniform(std::vector<Real> &t, std::vector<Real> &p) {
  // tabulate the integral of p at uniformly spaced t for easy calculations of energy from
  // t to t+dt to initialize the lasers for a time step
  t0 = t[0];
  t1 = t.back();
  // find min(dt) from input
  dt = 1.e300;
  for (int i = 1; i < t.size(); i++) {
    dt = std::min(dt, t[i] - t[i - 1]);
  }
  // make uniform grid for power/energy vs time
  int n = static_cast<int>(std::ceil((t1 - t0) / dt));
  dt = (t1 - t0) / n;
  std::vector<Real> pwr(n + 1);
  energy.resize(n + 1);

  // convert powers to erg/s from watts
  for (int i = 0; i < p.size(); i++)
    p[i] *= 1.0e7;

  // fill in power at uniform samples via integral averaging
  pwr[0] = integral_average(t, p, t[0], t[0] + 0.5 * dt);
  pwr.back() = integral_average(t, p, t.back() - 0.5 * dt, t.back() - 1.e-14);
  for (int i = 1; i < pwr.size() - 1; i++) {
    Real time = t0 + i * dt;
    pwr[i] = integral_average(t, p, time - 0.5 * dt, time + 0.5 * dt);
  }

  // check integrals
  Real int_orig = 0.0;
  for (int i = 0; i < p.size() - 1; i++) {
    int_orig += 0.5 * (t[i + 1] - t[i]) * (p[i] + p[i + 1]);
  }
  Real int_interp = 0.0;
  for (int i = 0; i < pwr.size() - 1; i++) {
    int_interp += 0.5 * dt * (pwr[i] + pwr[i + 1]);
  }

  // input times are in ns, convert to s
  t0 *= 1.e-9;
  t1 *= 1.e-9;
  dt *= 1.e-9;

  // now integrate to get energy vs time
  energy[0] = 0.0;
  for (int i = 1; i < energy.size(); i++) {
    energy[i] = energy[i - 1] + 0.5 * dt * (pwr[i - 1] + pwr[i]);
  }
}

//
// LaserProfile
//
LaserProfile::LaserProfile(ParameterInput *pin, const std::string &name) {
  auto dist = pin->GetString(name, "distribution");
  smajor = pin->GetReal(name, "power_semi_major_axis");
  sminor = pin->GetReal(name, "power_semi_minor_axis");
  if (dist == "flat") {
    profile = ProfileType::flat;
    total = M_PI * smajor * sminor;
    rmax = 1.0;
  } else if (dist == "super") {
    profile = ProfileType::super_gaussian;
    gorder = pin->GetReal(name, "power_super_exp");
    total = Integrate(0.0, 5.0);
    rmax = SampleWidth(
        pin->GetOrAddReal(name, "power_sample_frac", 0.999,
                          "Minimum fraction of power included in sampled beam profile"));
  } else {
    PARTHENON_FAIL("Invalid profile type for laser " + name);
  }
}

Real LaserProfile::Integrate(const Real r0, const Real r1) {
  Real integral = 0.0;
  if (profile == ProfileType::flat) {
    Real rmax = std::min(r1, 1.0);
    Real rmin = std::min(r0, 1.0);
    integral = M_PI * (r1 * r1 - r0 * r0) * smajor * sminor;
  } else if (profile == ProfileType::super_gaussian) {
    auto integrand = [&](const Real x) { return x * super_gaussian(x); };
    integral = 2 * M_PI * smajor * sminor * quad(integrand, r0, r1);
  }
  return integral;
}

Real LaserProfile::SampleWidth(const Real frac) {
  // return the half width necessary to sample "frac" fraction of total
  if (profile == ProfileType::flat) {
    return 1.0;
  } else if (profile == ProfileType::super_gaussian) {
    Real a = 0.1;
    Real b = 5.0;
    Real fa = NormalizedIntegral(0.0, a) - frac;
    Real fb = NormalizedIntegral(0.0, b) - frac;
    PARTHENON_REQUIRE_THROWS(fa * fb < 0, "SampleWidth calculation failed\n");
    while (b - a > 1.e-5) {
      Real c = 0.5 * (a + b);
      Real fc = NormalizedIntegral(0.0, c) - frac;
      if (fa * fc <= 0.0) {
        b = c;
        fb = fc;
      } else {
        a = c;
        fa = fc;
      }
    }
    return b;
  }
  return 0.0;
}

//
// LaserGrid
//
LaserGrid::LaserGrid(ParameterInput *pin, const std::string &name) : lp(pin, name) {
  auto [ga, gb] = lp.Axes();
  auto type = pin->GetString(name, "grid_type");
  if (type == "equal_area") {
    auto nr = pin->GetInteger(name, "nr");
    auto ntarget = pin->GetInteger(name, "ntarget");
    int nth = int((1.0 * ntarget) / nr);
    // total area of ellipse defined by {grid_a, grid_b} semi axes
    Real total_area = M_PI * ga * gb;
    // target area of each sample
    Real area0 = total_area / ntarget;
    Real rscale = lp.Scale();
    Real dr = rscale / nr;
    for (int ir = 0; ir < nr; ir++) {
      Real r = (ir + 0.5) * dr;
      Real r0 = ir * dr;
      Real r1 = r0 + dr;
      Real ring_area = M_PI * (r1 * r1 - r0 * r0) * ga * gb / (rscale * rscale);
      // int nth = ring_area / total_area * ntarget;
      Real dth = 2.0 * M_PI / nth;
      for (int it = 0; it < nth; it++) {
        rgrid.push_back(r);
        thgrid.push_back((it + 0.5) * dth);
        wgt.push_back(lp.NormalizedIntegral(r0, r1) / nth);
      }
    }
  } else {
    PARTHENON_FAIL("Invalid grid_type specified for laser " + name);
  }
  // make sure weights sum to one so all the power in the beam is captured
  Real wgt_sum = 0.0;
  for (auto &w : wgt)
    wgt_sum += w;
  for (auto &w : wgt)
    w /= wgt_sum;
}

} // namespace Laser
