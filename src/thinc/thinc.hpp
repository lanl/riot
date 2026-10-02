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
#ifndef THINC_THINC_HPP_
#define THINC_THINC_HPP_

#include "reconstruction/reconstruction.hpp"
#include "riot_limits.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <basic_types.hpp>
#include <cmath>
#include <coordinates/coordinates.hpp>
#include <limits>
#include <memory>
#include <parthenon/package.hpp>

namespace THINC {
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<parthenon::StateDescriptor> THINC::Initialize
//! \brief
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \struct  THINC::Point
//! \brief Components along the active coordinate directions; inactive ones stay zero.
struct Point {
  Real value[3] = {0, 0, 0};
  KOKKOS_INLINE_FUNCTION Real &operator[](int d) { return value[d]; }
  KOKKOS_INLINE_FUNCTION Real operator[](int d) const { return value[d]; }
};

//----------------------------------------------------------------------------------------
//! \fn  Real THINC::NeighborDelta
//! \brief A slope bound from the two directional neighbors, also respecting the
//!        material/complement enrichment bound. No new directional extrema are sharpened.
KOKKOS_INLINE_FUNCTION Real NeighborDelta(Real a, Real left, Real right, int side) {
  const Real dl = a - left, dr = right - a;
  if (dl * dr <= 0) return 0;
  const Real room = std::min(a, 1 - a);
  const Real jump = side < 0 ? dl : dr;
  return side * std::copysign(std::min(room, std::abs(jump)), dl);
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ProjectStencilFace
//! \brief A multidimensional interface may turn inside a coordinate-direction stencil.
//!         Project its faces onto surrounding-cell bounds without assuming a 1D slope.
//!         The common multiplier preserves the simplex, including absent materials.
KOKKOS_INLINE_FUNCTION void ProjectStencilFace(int count, const Real *alpha,
                                               const Real *minimum, const Real *maximum,
                                               Real *face, Real max_enrichment) {
  Real lo = 1, hi = -1;
  for (int m = 0; m < count; ++m) {
    const Real room = max_enrichment * std::min(alpha[m], 1 - alpha[m]);
    const Real lower = std::max(-room, minimum[m] - alpha[m]);
    const Real upper = std::min(room, maximum[m] - alpha[m]);
    lo = std::min(lo, face[m] - alpha[m] - upper);
    hi = std::max(hi, face[m] - alpha[m] - lower);
  }
  for (int it = 0; it < 56; ++it) {
    const Real mid = lo + .5 * (hi - lo);
    if (mid == lo || mid == hi) break;
    Real sum = 0;
    for (int m = 0; m < count; ++m) {
      const Real room = max_enrichment * std::min(alpha[m], 1 - alpha[m]);
      sum += std::max(
          std::max(-room, minimum[m] - alpha[m]),
          std::min(std::min(room, maximum[m] - alpha[m]), face[m] - alpha[m] - mid));
    }
    if (sum > 0)
      lo = mid;
    else
      hi = mid;
  }
  const Real lambda = lo + .5 * (hi - lo);
  for (int m = 0; m < count; ++m) {
    const Real room = max_enrichment * std::min(alpha[m], 1 - alpha[m]);
    face[m] = alpha[m] + std::max(std::max(-room, minimum[m] - alpha[m]),
                                  std::min(std::min(room, maximum[m] - alpha[m]),
                                           face[m] - alpha[m] - lambda));
  }
}

constexpr Real kMaterialSlopeLimit = 1.5;
constexpr Real kMaxGeometricEnrichment = 1;

//----------------------------------------------------------------------------------------
//! \struct  THINC::IntrinsicState
//! \brief
struct IntrinsicState {
  Real density, specific_energy, bulk_modulus;
};

//----------------------------------------------------------------------------------------
//! \struct  THINC::PartialState
//! \brief
struct PartialState {
  Real density, internal_energy, bulk_modulus;
};

//----------------------------------------------------------------------------------------
//! \struct  THINC::MaterialFaces
//! \brief
struct MaterialFaces {
  PartialState minus, plus;
};

//----------------------------------------------------------------------------------------
//! \fn  MaterialFaces THINC::ReconstructMaterial
//! \brief Pass the center state as both neighbors on unsupported material stencils.
//!        Specific energy, including any EOS reference offset, is limited directly.
//!        The energy and mass states therefore have the same material weighting:
//!        changing e by C changes each face's internal energy by C * partial_density.
KOKKOS_INLINE_FUNCTION MaterialFaces ReconstructMaterial(const IntrinsicState &left,
                                                         const IntrinsicState &center,
                                                         const IntrinsicState &right,
                                                         Real alpha_minus,
                                                         Real alpha_plus) {
  using RiotReconstruction::PiecewiseLinearSlope;
  const Real drho = PiecewiseLinearSlope(left.density, center.density, right.density,
                                         kMaterialSlopeLimit);
  const Real de = PiecewiseLinearSlope(left.specific_energy, center.specific_energy,
                                       right.specific_energy, kMaterialSlopeLimit);
  const Real dk = PiecewiseLinearSlope(left.bulk_modulus, center.bulk_modulus,
                                       right.bulk_modulus, kMaterialSlopeLimit);
  const Real rm = alpha_minus * (center.density - drho);
  const Real rp = alpha_plus * (center.density + drho);
  return {
      {rm, rm * (center.specific_energy - de), alpha_minus * (center.bulk_modulus - dk)},
      {rp, rp * (center.specific_energy + de), alpha_plus * (center.bulk_modulus + dk)}};
}

// Four-point Gauss-Legendre rule on the unit cell [-1/2, 1/2].
KOKKOS_INLINE_FUNCTION constexpr Real GaussPoint(int q) {
  constexpr Real x[4] = {-.4305681557970263, -.1699905217924281, .1699905217924281,
                         .4305681557970263};
  return x[q];
}
KOKKOS_INLINE_FUNCTION constexpr Real GaussWeight(int q) {
  constexpr Real w[4] = {.1739274225687269, .3260725774312731, .3260725774312731,
                         .1739274225687269};
  return w[q];
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::RadialWeights
//! \brief Volume weights at the x1 Gauss points of cell i, normalized to unit mean.
template <typename Coords>
KOKKOS_INLINE_FUNCTION void RadialWeights(const Coords &coords, int i, Real *radial) {
  for (int q = 0; q < 4; ++q)
    radial[q] = 1;
  if constexpr (!parthenon::IsCoord<parthenon::UniformCartesian>()) {
    constexpr int power = parthenon::IsCoord<parthenon::UniformSpherical>() ? 2 : 1;
    const Real dr = coords.template Dxf<parthenon::X1DIR>(i);
    const Real r0 = coords.template Xf<parthenon::X1DIR>(i) + .5 * dr;
    Real mean = 0;
    for (int q = 0; q < 4; ++q) {
      const Real r = std::abs(r0 + GaussPoint(q) * dr);
      radial[q] = power == 2 ? r * r : r;
      mean += GaussWeight(q) * radial[q];
    }
    for (int q = 0; q < 4; ++q)
      radial[q] /= mean;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real THINC::ProfileFraction
//! \brief Continuous THINC-type partition. Each conditional material profile is a
//!        hyperbolic tangent with one offset fixed by its cell average. The residual
//!        belongs to the remaining materials pointwise, preserving the simplex.
KOKKOS_INLINE_FUNCTION Real ProfileFraction(Real z) {
  const Real e = std::exp(-2 * std::abs(z));
  return z >= 0 ? 1 / (1 + e) : e / (1 + e);
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructProfile
//! \brief
template <int N, int D>
KOKKOS_INLINE_FUNCTION void
ReconstructProfile(int count, const Real *alpha, const Point *gradient, const int *ids,
                   const Real *radial, int axis, Real beta, Real *minus, Real *plus) {
  // Tensor Gauss rules with 4^D cell points and 4^(D-1) face points. Base-4 digit d
  // of a cell point index selects its coordinate along direction d; face points
  // enumerate the transverse directions in increasing order.
  constexpr int kCellPoints = 1 << (2 * D), kFacePoints = 1 << (2 * (D - 1));
  auto Dot = [](const Point &a, const Point &b) {
    Real sum = 0;
    for (int d = 0; d < D; ++d)
      sum += a[d] * b[d];
    return sum;
  };
  auto CellPoint = [&](const Point &n, int q, Real &weight) {
    Real projection = 0;
    weight = GaussWeight(q % 4) * radial[q % 4];
    for (int d = 0; d < D; ++d, q /= 4) {
      projection += n[d] * GaussPoint(q % 4);
      if (d > 0) weight *= GaussWeight(q % 4);
    }
    return projection;
  };
  auto FaceProjection = [&](const Point &n, int f) {
    Real projection = 0;
    for (int d = 0; d < D; ++d) {
      if (d == axis) continue;
      projection += n[d] * GaussPoint(f % 4);
      f /= 4;
    }
    return projection;
  };
  // x1 faces lie at fixed radius; other faces are averaged across radius.
  Real face_weight[kFacePoints];
  for (int f = 0; f < kFacePoints; ++f) {
    face_weight[f] = 1;
    for (int d = 0, r = f; d < D; ++d) {
      if (d == axis) continue;
      face_weight[f] *= GaussWeight(r % 4) * (d == 0 ? radial[r % 4] : 1);
      r /= 4;
    }
  }
  int order[N], active = 0;
  for (int m = 0; m < count; ++m) {
    minus[m] = plus[m] = 0;
    if (alpha[m] > 0) order[active++] = m;
  }
  // Deterministic material ordering and conditional normals for the nested profile.
  for (int i = 1; i < active; ++i) {
    const int m = order[i];
    const Real score = Dot(gradient[m], gradient[m]);
    int j = i;
    while (j > 0) {
      const int p = order[j - 1];
      const Real old = Dot(gradient[p], gradient[p]);
      if (old > score || (old == score && ids[p] < ids[m])) break;
      order[j] = p;
      --j;
    }
    order[j] = m;
  }
  Real remainder[kCellPoints], face_minus[kFacePoints], face_plus[kFacePoints];
  for (int q = 0; q < kCellPoints; ++q)
    remainder[q] = 1;
  for (int f = 0; f < kFacePoints; ++f)
    face_minus[f] = face_plus[f] = 1;
  Real remaining = 0;
  Point remaining_gradient;
  for (int m = 0; m < count; ++m) {
    remaining += alpha[m];
    for (int d = 0; d < D; ++d)
      remaining_gradient[d] += gradient[m][d];
  }
  for (int s = 0; s < active; ++s) {
    const int m = order[s];
    const bool last = s + 1 == active;
    Point n;
    for (int d = 0; d < D; ++d)
      n[d] = remaining * gradient[m][d] - alpha[m] * remaining_gradient[d];
    const Real norm = std::sqrt(Dot(n, n));
    Real width = 0; // |n|_1, the cell width along n
    if (norm > 0) {
      for (int d = 0; d < D; ++d) {
        n[d] /= norm;
        width += std::abs(n[d]);
      }
    }
    // Bracket analytically: all quadrature values lie below/above the
    // requested conditional mean at these two offsets. No tuned root bracket.
    Real offset = 0;
    if (!last) {
      Real residual_mass = 0;
      for (int t = s + 1; t < active; ++t)
        residual_mass += alpha[order[t]];
      offset = .5 * (std::log(alpha[m]) - std::log(residual_mass));
      const Real radius = .5 * beta * width;
      Real lo = offset - radius, hi = offset + radius;
      for (int it = 0; it < 48; ++it) {
        Real mean = 0, derivative = 0;
        for (int q = 0; q < kCellPoints; ++q) {
          Real weight;
          const Real h = ProfileFraction(beta * CellPoint(n, q, weight) + offset);
          weight *= remainder[q];
          mean += weight * h;
          derivative += 2 * weight * h * (1 - h);
        }
        if (std::abs(mean - alpha[m]) <=
            4 * std::numeric_limits<Real>::epsilon() * std::min(alpha[m], residual_mass))
          break;
        if (mean < alpha[m])
          lo = offset;
        else
          hi = offset;
        // Safeguarded Newton solves the same monotone volume constraint. The
        // analytic bracket remains authoritative if a Newton step leaves it.
        Real next =
            derivative > 0 ? offset + (alpha[m] - mean) / derivative : .5 * (lo + hi);
        if (!(next > lo && next < hi)) next = .5 * (lo + hi);
        if (next == offset) break;
        offset = next;
      }
    }
    for (int q = 0; q < kCellPoints; ++q) {
      Real weight;
      const Real z = beta * CellPoint(n, q, weight) + offset;
      remainder[q] *= last ? 0 : ProfileFraction(-z);
    }
    for (int f = 0; f < kFacePoints; ++f) {
      const Real normal = n[axis];
      const Real transverse = FaceProjection(n, f);
      const Real zm = beta * (-.5 * normal + transverse) + offset;
      const Real zp = beta * (.5 * normal + transverse) + offset;
      minus[m] += face_weight[f] * face_minus[f] * (last ? 1 : ProfileFraction(zm));
      plus[m] += face_weight[f] * face_plus[f] * (last ? 1 : ProfileFraction(zp));
      face_minus[f] *= last ? 0 : ProfileFraction(-zm);
      face_plus[f] *= last ? 0 : ProfileFraction(-zp);
    }
    remaining -= alpha[m];
    for (int d = 0; d < D; ++d)
      remaining_gradient[d] -= gradient[m][d];
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructFractions
//! \brief Replace mixed-cell face states BEFORE the bulk Riemann problem. Geometry and
//!        thermodynamics are deliberately inseparable here: changing alpha only in the
//!        material flux would make its sum inconsistent with momentum and energy fluxes.
//!        D is the mesh dimension; dy and dz are unused below it.
template <parthenon::CoordinateDirection DIR, int D, typename Pack, typename BaseRange,
          typename Range, typename Delta, typename MatScratch, typename Fallback>
KOKKOS_INLINE_FUNCTION void
ReconstructFractions(const Pack &v, const BaseRange &base, const Range &range, int b,
                     Delta dx, Delta dy, Delta dz, Real beta, MatScratch &mat_minus,
                     MatScratch &mat_plus, Fallback &fallback_minus,
                     Fallback &fallback_plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  static_assert(D >= 1 && D <= 3 && DIR - 1 < D, "THINC direction exceeds dimension.");
  constexpr int N = RiotLimits::MAX_MATERIALS;
  constexpr int axis = DIR - 1;
  const int count = v.GetSize(b, ccmat::volume_fraction());
  const auto &coords = v.GetCoordinates(b);
  const Delta axes[3] = {dx, dy, dz};
  // Transverse directions of this sweep (t2 only in 3D)
  const Delta dn = axes[axis], t1 = axes[(axis + 1) % D], t2 = axes[(axis + 2) % D];
  RiotLoop::inner(range, [&](auto idx) {
    Real alpha[N], minus[N], plus[N], left[N], right[N];
    Real minimum[N], maximum[N];
    Point gradient[N];
    int ids[N], active = 0;
    for (int m = 0; m < count; ++m) {
      fallback_minus(m, idx) = mat_minus(ccmat::volume_fraction(), m, idx);
      fallback_plus(m, idx) = mat_plus(ccmat::volume_fraction(), m, idx);
      auto pm = RiotLoop::make_sparse_pack_view(base, v, m);
      alpha[m] = pm(ccmat::volume_fraction(), idx);
      left[m] = pm(ccmat::volume_fraction(), idx - dn);
      right[m] = pm(ccmat::volume_fraction(), idx + dn);
      active += alpha[m] > 0;
      ids[m] = v(b, ccmat::volume_fraction(m)).sparse_id;
      auto A = [&](auto offset) { return pm(ccmat::volume_fraction(), idx + offset); };
      minimum[m] = std::min(alpha[m], std::min(left[m], right[m]));
      maximum[m] = std::max(alpha[m], std::max(left[m], right[m]));
      if constexpr (D == 1) {
        gradient[m][0] = 0.5 * (A(dx) - A(-dx));
      } else {
        for (int k = -(D > 2); k <= (D > 2); ++k)
          for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
              const Real value = A(i * dx + j * dy + k * dz);
              minimum[m] = std::min(minimum[m], value);
              maximum[m] = std::max(maximum[m], value);
            }
        // Transversely averaged (Youngs) gradients damp cell-scale orientation
        // noise without smoothing the transported material fractions.
        if constexpr (D == 2) {
          gradient[m][0] = 0.125 * (A(dx - dy) + 2 * A(dx) + A(dx + dy) - A(-dx - dy) -
                                    2 * A(-dx) - A(-dx + dy));
          gradient[m][1] = 0.125 * (A(dy - dx) + 2 * A(dy) + A(dy + dx) - A(-dy - dx) -
                                    2 * A(-dy) - A(-dy + dx));
        } else {
          for (int d = 0; d < 3; ++d) {
            const Delta a1 = axes[(d + 1) % 3], a2 = axes[(d + 2) % 3];
            Real sum = 0;
            for (int k = -1; k <= 1; ++k)
              for (int j = -1; j <= 1; ++j) {
                const auto t = j * a1 + k * a2;
                sum += (2 - std::abs(j)) * (2 - std::abs(k)) *
                       (A(axes[d] + t) - A(t - axes[d]));
              }
            gradient[m][d] = sum / 32;
          }
        }
        // Avoid sharpening a turning face-normal profile with a single plane.
        // Test the same transverse averages used by the Youngs normal, rather
        // than an individual row through a curved interface.
        auto Plane = [&](auto shift) {
          Real sum = 0;
          for (int k = -(D > 2); k <= (D > 2); ++k)
            for (int j = -1; j <= 1; ++j)
              sum += (2 - std::abs(j)) * (D > 2 ? 2 - std::abs(k) : 1) *
                     A(shift + j * t1 + k * t2);
          return sum / (D > 2 ? 16 : 4);
        };
        const Real center = Plane(0 * dn);
        const Real before = Plane(-dn);
        const Real after = Plane(dn);
        if ((center - before) * (after - center) <= 0) {
          minimum[m] = alpha[m];
          maximum[m] = alpha[m];
        }
      }
    }
    if (active <= 1) return; // retain ordinary hydro in pure cells
    const auto [k, j, i] = range.GetKJI(idx);
    Real radial[4];
    RadialWeights(coords, i, radial);
    // Project once onto the per-material bounds and simplex below. Applying
    // the common-factor enrichment limiter first lets a trace material flatten
    // all the other materials, even when their geometric faces are admissible.
    ReconstructProfile<N, D>(count, alpha, gradient, ids, radial, axis, beta, minus,
                             plus);
    if constexpr (D > 1) {
      ProjectStencilFace(count, alpha, minimum, maximum, minus, kMaxGeometricEnrichment);
      ProjectStencilFace(count, alpha, minimum, maximum, plus, kMaxGeometricEnrichment);
    } else {
      for (int side = -1; side <= 1; side += 2) {
        for (int m = 0; m < count; ++m) {
          const Real bound = NeighborDelta(alpha[m], left[m], right[m], side);
          minimum[m] = alpha[m] + std::min(0.0, bound);
          maximum[m] = alpha[m] + std::max(0.0, bound);
        }
        ProjectStencilFace(count, alpha, minimum, maximum, side < 0 ? minus : plus,
                           kMaxGeometricEnrichment);
      }
    }

    for (int m = 0; m < count; ++m) {
      mat_minus(ccmat::volume_fraction(), m, idx) = minus[m];
      mat_plus(ccmat::volume_fraction(), m, idx) = plus[m];
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::SelectFaceProfiles
//! \brief Face-local boundary variation selection. A single choice for the complete
//!        material vector preserves both donor simplices. Each face owns exactly one
//!        minus and one plus scratch entry, so adjacent faces do not race on writes.
//!        This minimizes the compositional jump presented to the Riemann solver
template <typename Range, typename Delta, typename Mat, typename Fallback>
KOKKOS_INLINE_FUNCTION void
SelectFaceProfiles(const Range &faces, Delta dn, int count, Mat &minus, Mat &plus,
                   const Fallback &fallback_minus, const Fallback &fallback_plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  RiotLoop::inner(faces, [&](auto idx) {
    Real ordinary_jump = 0, profile_jump = 0;
    for (int m = 0; m < count; ++m) {
      ordinary_jump += std::abs(fallback_plus(m, idx - dn) - fallback_minus(m, idx));
      profile_jump += std::abs(plus(ccmat::volume_fraction(), m, idx - dn) -
                               minus(ccmat::volume_fraction(), m, idx));
    }
    if (profile_jump < ordinary_jump) return;
    for (int m = 0; m < count; ++m) {
      plus(ccmat::volume_fraction(), m, idx - dn) = fallback_plus(m, idx - dn);
      minus(ccmat::volume_fraction(), m, idx) = fallback_minus(m, idx);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructMaterialStates
//! \brief
template <typename Pack, typename BaseRange, typename Range, typename Delta,
          typename MatScratch, typename SumScratch, typename SetScratch>
KOKKOS_INLINE_FUNCTION void
ReconstructMaterialStates(const Pack &v, const BaseRange &base, const Range &range, int b,
                          Delta dn, MatScratch &mat_minus, MatScratch &mat_plus,
                          SumScratch &sum_minus, SumScratch &sum_plus,
                          SetScratch &set_minus, SetScratch &set_plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  const int count = v.GetSize(b, ccmat::volume_fraction());
  RiotLoop::inner(range, [&](auto idx) {
    int active = 0;
    for (int m = 0; m < count; ++m) {
      auto pm = RiotLoop::make_sparse_pack_view(base, v, m);
      active += pm(ccmat::volume_fraction(), idx) > 0;
    }
    if (active <= 1) return;
    Real rho_m = 0, rho_p = 0, energy_m = 0, energy_p = 0, bmod_m = 0, bmod_p = 0;
    for (int m = 0; m < count; ++m) {
      auto pm = RiotLoop::make_sparse_pack_view(base, v, m);
      const Real alpha = pm(ccmat::volume_fraction(), idx);
      const Real left = pm(ccmat::volume_fraction(), idx - dn);
      const Real right = pm(ccmat::volume_fraction(), idx + dn);
      const Real rho = pm(cm::rho(), idx);
      const Real partial_rho = pm(ccmat::rho(), idx);
      const Real sie =
          partial_rho > 0 ? pm(ccmat::internal_energy(), idx) / partial_rho : 0;
      const IntrinsicState center{rho, sie, pm(cm::bulk_modulus(), idx)};
      IntrinsicState state_left = center, state_right = center;
      if (alpha > 0 && left > 0 && right > 0 && rho > 0 &&
          pm(ccmat::rho(), idx - dn) > 0 && pm(ccmat::rho(), idx + dn) > 0) {
        state_left = {pm(cm::rho(), idx - dn),
                      pm(ccmat::internal_energy(), idx - dn) / pm(ccmat::rho(), idx - dn),
                      pm(cm::bulk_modulus(), idx - dn)};
        state_right = {pm(cm::rho(), idx + dn),
                       pm(ccmat::internal_energy(), idx + dn) /
                           pm(ccmat::rho(), idx + dn),
                       pm(cm::bulk_modulus(), idx + dn)};
      }
      const auto faces = ReconstructMaterial(state_left, center, state_right,
                                             mat_minus(ccmat::volume_fraction(), m, idx),
                                             mat_plus(ccmat::volume_fraction(), m, idx));
      mat_minus(cm::rho(), m, idx) = faces.minus.density;
      mat_plus(cm::rho(), m, idx) = faces.plus.density;
      mat_minus(ccmat::internal_energy(), m, idx) = faces.minus.internal_energy;
      mat_plus(ccmat::internal_energy(), m, idx) = faces.plus.internal_energy;
      rho_m += faces.minus.density;
      rho_p += faces.plus.density;
      energy_m += faces.minus.internal_energy;
      energy_p += faces.plus.internal_energy;
      bmod_m += faces.minus.bulk_modulus;
      bmod_p += faces.plus.bulk_modulus;
    }
    sum_minus(ccbulk::rho(), idx) = rho_m;
    sum_plus(ccbulk::rho(), idx) = rho_p;
    sum_minus(ccbulk::internal_energy(), idx) = energy_m;
    sum_plus(ccbulk::internal_energy(), idx) = energy_p;
    set_minus(ccbulk::bulk_modulus(), idx) = bmod_m;
    set_plus(ccbulk::bulk_modulus(), idx) = bmod_p;
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructAdvected
//! \brief Complete the face state of a material-tied advected quantity in mixed cells.
//!       Like specific energy and stress, q is intrinsic to material m and is carried by
//!       that material's THINC mass flux (summed over its phases), so it is never
//!       extrapolated through a cell where the material is absent.
template <typename Pack, typename BaseRange, typename Range, typename Delta, typename Var,
          typename Scratch>
KOKKOS_INLINE_FUNCTION void
ReconstructAdvected(const Pack &v, const BaseRange &base, const Range &range, Delta dn,
                    int b, int m, const Var &q, Scratch &minus, Scratch &plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  const int count = v.GetSize(b, ccmat::volume_fraction());
  const int id = v(b, ccmat::volume_fraction(m)).sparse_id;
  RiotLoop::inner(range, [&](auto idx) {
    int active = 0;
    Real alpha = 0, left = 0, right = 0; // material fractions, summed over phases
    for (int n = 0; n < count; ++n) {
      auto other = RiotLoop::make_sparse_pack_view(base, v, n);
      active += other(ccmat::volume_fraction(), idx) > 0;
      if (v(b, ccmat::volume_fraction(n)).sparse_id != id) continue;
      alpha += other(ccmat::volume_fraction(), idx);
      left += other(ccmat::volume_fraction(), idx - dn);
      right += other(ccmat::volume_fraction(), idx + dn);
    }
    if (active <= 1) return; // preserve ordinary reconstruction in pure cells
    const bool supported = alpha > 0 && left > 0 && right > 0;
    const Real center = alpha > 0 ? q(idx) : 0;
    const Real dq = supported ? RiotReconstruction::PiecewiseLinearSlope(
                                    q(idx - dn), center, q(idx + dn), kMaterialSlopeLimit)
                              : 0;
    minus(idx) = center - dq;
    plus(idx) = center + dq;
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructElectronEnergy
//! \brief Reconstruct the bulk electron-energy density from its material contributions:
//!        u_e^face = sum_m (alpha rho)_m^face e_{e,m}^face. Electron entropy has no
//!        corresponding material decomposition and retains ordinary reconstruction.
template <typename Pack, typename BaseRange, typename Range, typename Delta, typename Mat,
          typename Scratch>
KOKKOS_INLINE_FUNCTION void
ReconstructElectronEnergy(const Pack &v, const BaseRange &base, const Range &range,
                          Delta dn, int b, const Mat &mat_minus, const Mat &mat_plus,
                          Scratch &minus, Scratch &plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  const int count = v.GetSize(b, ccmat::volume_fraction());
  RiotLoop::inner(range, [&](auto idx) {
    int active = 0;
    for (int m = 0; m < count; ++m) {
      auto pm = RiotLoop::make_sparse_pack_view(base, v, m);
      active += pm(ccmat::volume_fraction(), idx) > 0;
    }
    if (active <= 1) return; // preserve ordinary reconstruction in pure cells
    Real face_minus = 0, face_plus = 0;
    for (int m = 0; m < count; ++m) {
      auto pm = RiotLoop::make_sparse_pack_view(base, v, m);
      const bool present = pm(ccmat::volume_fraction(), idx) > 0;
      const bool supported = present && pm(ccmat::volume_fraction(), idx - dn) > 0 &&
                             pm(ccmat::volume_fraction(), idx + dn) > 0;
      const Real e = present ? pm(cm::electron_sie(), idx) : 0;
      const Real de = supported
                          ? RiotReconstruction::PiecewiseLinearSlope(
                                pm(cm::electron_sie(), idx - dn), e,
                                pm(cm::electron_sie(), idx + dn), kMaterialSlopeLimit)
                          : 0;
      face_minus += mat_minus(cm::rho(), m, idx) * (e - de);
      face_plus += mat_plus(cm::rho(), m, idx) * (e + de);
    }
    minus(idx) = face_minus;
    plus(idx) = face_plus;
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void THINC::ReconstructStrength
//! \brief
template <typename Pack, typename StrengthPack, typename BaseRange, typename Range,
          typename Delta, typename Mat, typename Stress, typename Shear>
KOKKOS_INLINE_FUNCTION void
ReconstructStrength(const Pack &v, const StrengthPack &vstr, const BaseRange &base,
                    const Range &range, Delta dn, int b, int m, int s, bool first,
                    const Mat &mat_minus, const Mat &mat_plus, Stress &stress_minus,
                    Stress &stress_plus, Shear &shear_minus, Shear &shear_plus) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  const int count = v.GetSize(b, ccmat::volume_fraction());
  auto material = RiotLoop::make_sparse_pack_view(base, v, m);
  auto strong = RiotLoop::make_sparse_pack_view(base, vstr, s);
  RiotLoop::inner(range, [&](auto idx) {
    int active = 0;
    for (int n = 0; n < count; ++n) {
      auto other = RiotLoop::make_sparse_pack_view(base, v, n);
      active += other(ccmat::volume_fraction(), idx) > 0;
    }
    if (active <= 1) return; // preserve ordinary reconstruction in pure cells
    if (first) {
      shear_minus(ccbulk::shear_modulus(), idx) = 0;
      shear_plus(ccbulk::shear_modulus(), idx) = 0;
    }
    const Real alpha = material(ccmat::volume_fraction(), idx);
    const bool supported = alpha > 0 &&
                           material(ccmat::volume_fraction(), idx - dn) > 0 &&
                           material(ccmat::volume_fraction(), idx + dn) > 0;
    const Real g = alpha > 0 ? strong(cm::shear_modulus(), idx) : 0;
    const Real dg = supported
                        ? RiotReconstruction::PiecewiseLinearSlope(
                              strong(cm::shear_modulus(), idx - dn), g,
                              strong(cm::shear_modulus(), idx + dn), kMaterialSlopeLimit)
                        : 0;
    shear_minus(ccbulk::shear_modulus(), idx) +=
        mat_minus(ccmat::volume_fraction(), m, idx) * (g - dg);
    shear_plus(ccbulk::shear_modulus(), idx) +=
        mat_plus(ccmat::volume_fraction(), m, idx) * (g + dg);
    for (int c = 0; c < 5; ++c) {
      const Real stress = alpha > 0 ? strong(cm::deviatoric_stress(c), idx) : 0;
      const Real ds =
          supported ? RiotReconstruction::PiecewiseLinearSlope(
                          strong(cm::deviatoric_stress(c), idx - dn), stress,
                          strong(cm::deviatoric_stress(c), idx + dn), kMaterialSlopeLimit)
                    : 0;
      stress_minus(cm::deviatoric_stress(c), s, idx) = stress - ds;
      stress_plus(cm::deviatoric_stress(c), s, idx) = stress + ds;
    }
  });
}
} // namespace THINC
#endif
