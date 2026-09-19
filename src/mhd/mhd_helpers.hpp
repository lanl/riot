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
#ifndef MHD_MHD_HELPERS_HPP_
#define MHD_MHD_HELPERS_HPP_
// This file was made in part with generative AI.
//
// Device-callable ideal-MHD primitives. Every magnetic quantity in RIOT is derived
// through exactly one of these helpers so that a unit factor or a face-to-cell
// convention cannot drift between the Riemann solver, the energy bookkeeping, the
// timestep vote, and the diagnostics.
//
// Adapted from the ideal-MHD implementation on the lanl/artemis branch dempsey/mhd
// (src/mhd/mhd.hpp @ 3e5aeb5). Cartesian-only: see ADR-003 in
// plan_histories/artemis_mhd_port/ for the scope decision and for the curvilinear
// terms deliberately not transcribed.

#include <parthenon/package.hpp>

#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace MHD {

using parthenon::Real;
using TE = parthenon::TopologicalElement;

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::MagneticEnergyDensity
//! \brief Magnetic energy density B^2 / (2 mu0).
//!
//! mu0 is the runtime parameter from ADR-001: 4*pi for Gaussian CGS (B in Gauss, so
//! this is B^2/8pi), or 1 to reproduce the donor's scale-free normalized field exactly
//! for validation. It is threaded by value into every kernel rather than being folded
//! into the field definition, so that the same code covers both.
KOKKOS_FORCEINLINE_FUNCTION Real MagneticEnergyDensity(const Real bx, const Real by,
                                                       const Real bz, const Real mu0) {
  return 0.5 * (SQR(bx) + SQR(by) + SQR(bz)) / mu0;
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::FastMagnetosonicSpeed
//! \brief Fast magnetosonic speed from the bulk modulus and the magnetic field.
//!
//!   cf^2 = 1/2 * [ (a^2 + vA^2) + sqrt( (a^2 + vA^2)^2 - 4 a^2 vAn^2 ) ]
//!
//! with a^2 = bmod/rho the acoustic derivative, vA^2 = b2/(mu0 rho) the total Alfven
//! speed, and vAn^2 = bn^2/(mu0 rho) its component along the interface normal.
//!
//! `bmod` is RIOT's bulk modulus K = rho (dp/drho)_s (ccbulk::bulk_modulus), NOT a
//! pressure and NOT gamma*p -- it is the general-EOS acoustic derivative that RIOT
//! already aggregates by volume fraction, so no constant-gamma assumption enters here.
//!
//! `bn` is the component NORMAL to the interface. The discriminant is exactly zero in
//! the degenerate cases (purely parallel or purely perpendicular propagation) and can
//! go slightly negative by roundoff there, so it is floored at zero. That floor is a
//! roundoff guard on an identity, not a fix for an invalid state: a genuinely invalid
//! thermodynamic state shows up as bmod < 0 or rho <= 0 and is detected separately.
KOKKOS_FORCEINLINE_FUNCTION Real FastMagnetosonicSpeed(const Real bmod, const Real rho,
                                                       const Real b2, const Real bn,
                                                       const Real mu0) {
  const Real wave_sum = (bmod + b2 / mu0) / rho;
  const Real discriminant =
      std::max(0.0, SQR(wave_sum) - 4.0 * bmod * SQR(bn) / (mu0 * SQR(rho)));
  return std::sqrt(0.5 * (wave_sum + std::sqrt(discriminant)));
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::FastMagnetosonicSpeed
//! \brief Convenience overload taking the three field components; `bn` is the first.
KOKKOS_FORCEINLINE_FUNCTION Real FastMagnetosonicSpeed(const Real bmod, const Real rho,
                                                       const Real bn, const Real bt1,
                                                       const Real bt2, const Real mu0) {
  return FastMagnetosonicSpeed(bmod, rho, SQR(bn) + SQR(bt1) + SQR(bt2), bn, mu0);
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::SignalSpeedBound
//! \brief Direction-independent upper bound on the fast speed, sqrt(a^2 + vA^2).
//!
//! Used by the timestep vote, matching the donor (artemis/src/gas/gas.cpp:543-643):
//! evaluating the true directional fast speed at every face is far more expensive than
//! this bound, and a bound is what CFL safety requires. Do NOT use this in a Riemann
//! solver, where the directional speed is needed for a sharp wave-speed estimate.
KOKKOS_FORCEINLINE_FUNCTION Real SignalSpeedBound(const Real bmod, const Real rho,
                                                  const Real b2, const Real mu0) {
  return std::sqrt((bmod + b2 / mu0) / rho);
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::FaceB
//! \brief Read the face-centered magnetic field at logical coordinates (k, j, i).
//!
//! WHY THIS TAKES THE RAW PACK AND LOGICAL COORDINATES, and never a loop-abstraction
//! pack_view: RIOT's face field is the first genuine (non-CellMemAligned) face field in
//! the code, so its storage is NODE-shaped -- Metadata::GetArrayDims
//! (external/parthenon/src/interface/metadata.cpp:381-387) adds one element per
//! non-degenerate direction when CellMemAligned is absent. The loop abstraction's
//! flat/memory pack views cache `data() + shift` and index with the index space's single
//! memory indexer, which is cell-shaped (IndexSpace only accepts memory_te == CC or NN,
//! index_space.hpp:203). Addressing node-shaped face storage with cell strides is
//! silently wrong -- wrong element, no out-of-bounds, no assertion. The loop-abstraction
//! contract states the rule directly: a kernel touching fields with different memory
//! layouts must use inner_tag::logical_coords
//! (LOOP_ABSTRACTION_CONTRACTS.md:96, i.e. RIOT's LoopConstraint::DifferentMemSpaces).
//!
//! Going through the pack with logical coordinates sidesteps that constraint entirely:
//! Parthenon's own accessor indexes each variable's array with that array's strides. It
//! costs one index computation per access and lets the surrounding loop keep the fast
//! cell-centered flat contract, which matters because two of the call sites are in the
//! hot hydro kernels (fill_shared_derived.cpp) that must not slow down when MHD is off.
template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real FaceB(const Pack_t &v, const int b, const TE te,
                                       const int k, const int j, const int i) {
  namespace fbulk = face_variables::bulk;
  return v(b, te, fbulk::magnetic_field(), k, j, i);
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::FaceToCellB
//! \brief Cell-centered magnetic field from the two bounding face values per direction.
//!
//! In Cartesian geometry the donor's centroid-weighted interpolation
//! (artemis/src/mhd/mhd.hpp:55-78) reduces exactly to the arithmetic mean, which is
//! what is used here.
//!
//! Degenerate directions are handled by taking the single available face value rather
//! than by zeroing the component: a transverse magnetic field is physically nonzero and
//! evolving even when its own direction is collapsed (B2 and B3 in 1D, B3 in 2D). The
//! caller passes multi_d/three_d as 0 in a collapsed direction, so the two reads
//! coincide and the mean is the value itself -- no branch is required.
template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION void
FaceToCellB(const Pack_t &v, const int b, const int k, const int j, const int i,
            const int multi_d, const int three_d, Real &bx, Real &by, Real &bz) {
  bx = 0.5 * (FaceB(v, b, TE::F1, k, j, i) + FaceB(v, b, TE::F1, k, j, i + 1));
  by = 0.5 * (FaceB(v, b, TE::F2, k, j, i) + FaceB(v, b, TE::F2, k, j + multi_d, i));
  bz = 0.5 * (FaceB(v, b, TE::F3, k, j, i) + FaceB(v, b, TE::F3, k + three_d, j, i));
}

//----------------------------------------------------------------------------------------
//! \fn  Real MHD::CellMagneticEnergyFromFaces
//! \brief Magnetic energy density at a cell center, straight from face state.
//!
//! The single helper used by BOTH energy touch points -- the thermal-energy recovery in
//! Multiphysics::FillInteriorDerived and the total-energy re-synthesis in
//! Multiphysics::PostCommsFillDerived (see ADR-002). Sharing one helper is what
//! guarantees the subtraction and the addition cannot disagree, which is the failure
//! mode that shows up as spurious heating in a static uniform field.
//!
//! Deliberately reads FACE state rather than the derived ccbulk::magnetic_energy field.
//! The derived field would be cheaper, but it introduces an ordering hazard -- a stale
//! or un-communicated value would make the subtraction and the addition disagree, which
//! is precisely the failure ADR-002 exists to rule out. Reading the authoritative state
//! at both sites makes agreement structural instead of scheduling-dependent.
template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real CellMagneticEnergyFromFaces(
    const Pack_t &v, const int b, const int k, const int j, const int i,
    const int multi_d, const int three_d, const Real mu0) {
  Real bx, by, bz;
  FaceToCellB(v, b, k, j, i, multi_d, three_d, bx, by, bz);
  return MagneticEnergyDensity(bx, by, bz, mu0);
}

} // namespace MHD

#endif // MHD_MHD_HELPERS_HPP_
