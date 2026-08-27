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

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <singularity-eos/eos/eos.hpp>

#include <globals.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>

#include "hydro.hpp"
#include "materials/materials.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/pte_closure.hpp"
#include "microphysics/strength_models.hpp"
#include "reconstruction/reconstruction.hpp"
#include "riemann.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "riot_utils/sparse_update.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexSplit;
using parthenon::ParArray1D;
using parthenon::team_mbr_t;

namespace Hydro {

using RiotLimits::MAX_ADV;
using RiotLimits::MAX_MATERIALS;
using RiotLimits::MAX_STRONG;

//----------------------------------------------------------------------------------------
//! \brief Transverse velocity differences (dvn, dvt) used by the low-Mach solvers
//!        (chllc/lhllc). Reads the (already-ghost-filled) state velocities at neighbor
//!        cells -- a wide input stencil on available data, not a producer halo -- and
//!        writes one dvn/dvt per face cell. Normal/transverse offsets+components come
//!        from the (cyclic) DirBasis, so this body is direction-agnostic.
template <parthenon::CoordinateDirection DIR, typename Pack_t, typename IdxRange,
          typename Basis, typename Scratch>
KOKKOS_INLINE_FUNCTION void GetVdiff(const Pack_t &v, const IdxRange &idx_range,
                                     const Basis &basis, Scratch &dvn, Scratch &dvt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  const auto dn = basis.normal();
  RiotLoop::inner(idx_range, [&](const auto kji) {
    auto V = [&](int c, auto idx) { return pv(ccbulk::velocity(c), idx); };
    // Min-limited transverse gradient of velocity component c along offset t (normal dn).
    auto trans = [&](int c, auto t) {
      return std::min(
          std::min(V(c, kji - dn) - V(c, kji - dn - t), V(c, kji) - V(c, kji - t)),
          std::min(V(c, kji + t - dn) - V(c, kji - dn), V(c, kji + t) - V(c, kji)));
    };
    dvn(kji) = V(Basis::nc, kji) - V(Basis::nc, kji - dn);

    constexpr Real BIG = std::numeric_limits<Real>::max();
    Real t = BIG;
    if (!RiotUtils::IsZeroOffset(basis.trans_a()))
      t = std::min(t, trans(Basis::tac, basis.trans_a()));
    if (!RiotUtils::IsZeroOffset(basis.trans_b()))
      t = std::min(t, trans(Basis::tbc, basis.trans_b()));
    dvt(kji) = (t == BIG) ? 0.0 : t;
  });
}

//----------------------------------------------------------------------------------------
//! \brief Bulk sound speed on a reconstructed face: max(sqrt(|bmod/rho|), tiny).
KOKKOS_FORCEINLINE_FUNCTION Real BulkSoundSpeed(const Real &bmod, const Real &rho) {
  constexpr Real tiny_cs = 1.e-12;
  return std::max(std::sqrt(std::abs(bmod / rho)), tiny_cs);
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::StoreFaceVelocity
//! \brief Store the Riemann face-velocity vector (v1,v2,v3) into the persistent
//!        ccbulk::face_velocity field for sweep direction DIR, when store_vf is set.
//!        face_velocity holds 9 components (3 per direction); direction DIR-1 (0-based)
//!        occupies components 3*(DIR-1) .. 3*(DIR-1)+2. This field is consumed downstream
//!        by the strength strain-rate source (CalculateStrengthSource) and partial
//!        ionization -- a real flux-kernel output, not scratch. Guarded at runtime so the
//!        store is skipped (and the field may be absent) when off.
template <parthenon::CoordinateDirection DIR, typename PackView, typename Index>
KOKKOS_FORCEINLINE_FUNCTION void StoreFaceVelocity(const PackView &pv, const Index &kji,
                                                   const bool store_vf, const Real v1face,
                                                   const Real v2face, const Real v3face) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  if (!store_vf) return;
  constexpr int base = 3 * (static_cast<int>(DIR) - 1);
  pv(ccbulk::face_velocity(base + 0), kji) = v1face;
  pv(ccbulk::face_velocity(base + 1), kji) = v2face;
  pv(ccbulk::face_velocity(base + 2), kji) = v3face;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::BulkRiemannFluxes
//! \brief Bulk Riemann flux loop for the HLLC-style solvers (hllc/hllcf/hll). Produces
//!        the bulk momentum/energy fluxes and the face_vel/riemann_vel scratch consumed
//!        by the (solver-independent) material-density flux loop, and stores the signal
//!        speed. Templated on DIR and the solver function FLUX_FN.
template <parthenon::CoordinateDirection DIR, auto FLUX_FN, typename Pack_t,
          typename IdxRange, typename Delta, typename SetBulk, typename SumBulk,
          typename Scratch>
KOKKOS_INLINE_FUNCTION void
BulkRiemannFluxes(const Pack_t &v, const IdxRange &idx_range, Delta delta,
                  const SetBulk &set_bulk_minus, const SetBulk &set_bulk_plus,
                  const SumBulk &sum_bulk_minus, const SumBulk &sum_bulk_plus,
                  Scratch &face_vel, Scratch &riemann_vel, const bool store_vf) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  constexpr auto te = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  auto fv = RiotLoop::make_flux_pack_view(idx_range, v, DIR);
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  RiotLoop::inner(idx_range, [&](const auto kji) {
    const auto kji_L = kji - delta;
    const auto kji_R = kji;
    const Real cs_L = BulkSoundSpeed(set_bulk_plus(ccbulk::bulk_modulus(), kji_L),
                                     sum_bulk_plus(ccbulk::rho(), kji_L));
    const Real cs_R = BulkSoundSpeed(set_bulk_minus(ccbulk::bulk_modulus(), kji_R),
                                     sum_bulk_minus(ccbulk::rho(), kji_R));

    Real f_v1, f_v2, f_v3, f_eng, v1face, v2face, v3face;
    const Real signal_speed =
        FLUX_FN(sum_bulk_plus(ccbulk::rho(), kji_L), sum_bulk_minus(ccbulk::rho(), kji_R),
                set_bulk_plus(ccbulk::velocity(0), kji_L),
                set_bulk_minus(ccbulk::velocity(0), kji_R),
                set_bulk_plus(ccbulk::velocity(1), kji_L),
                set_bulk_minus(ccbulk::velocity(1), kji_R),
                set_bulk_plus(ccbulk::velocity(2), kji_L),
                set_bulk_minus(ccbulk::velocity(2), kji_R),
                sum_bulk_plus(ccbulk::internal_energy(), kji_L),
                sum_bulk_minus(ccbulk::internal_energy(), kji_R),
                set_bulk_plus(ccbulk::pressure(), kji_L),
                set_bulk_minus(ccbulk::pressure(), kji_R), cs_L, cs_R, f_v1, f_v2, f_v3,
                f_eng, v1face, v2face, v3face, riemann_vel(kji));

    fv(ccbulk::momentum(0), kji) = f_v1;
    fv(ccbulk::momentum(1), kji) = f_v2;
    fv(ccbulk::momentum(2), kji) = f_v3;
    fv(ccbulk::total_material_energy(), kji) = f_eng;
    face_vel(kji) = (DIR == X1DIR) ? v1face : ((DIR == X2DIR) ? v2face : v3face);
    pv(te, ccbulk::face_signal(), kji) = signal_speed;
    StoreFaceVelocity<DIR>(pv, kji, store_vf, v1face, v2face, v3face);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::BulkRiemannFluxesLM
//! \brief Low-Mach variant of BulkRiemannFluxes for chllc/lhllc: identical, but also
//!        feeds the per-face transverse velocity differences dvn/dvt (from GetVdiff)
//!        into the solver's extended signature.
template <parthenon::CoordinateDirection DIR, auto FLUX_FN, typename Pack_t,
          typename IdxRange, typename Delta, typename SetBulk, typename SumBulk,
          typename Scratch>
KOKKOS_INLINE_FUNCTION void
BulkRiemannFluxesLM(const Pack_t &v, const IdxRange &idx_range, Delta delta,
                    const SetBulk &set_bulk_minus, const SetBulk &set_bulk_plus,
                    const SumBulk &sum_bulk_minus, const SumBulk &sum_bulk_plus,
                    Scratch &face_vel, Scratch &riemann_vel, Scratch &dvn, Scratch &dvt,
                    const bool store_vf) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  constexpr auto te = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  auto fv = RiotLoop::make_flux_pack_view(idx_range, v, DIR);
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  RiotLoop::inner(idx_range, [&](const auto kji) {
    const auto kji_L = kji - delta;
    const auto kji_R = kji;
    const Real cs_L = BulkSoundSpeed(set_bulk_plus(ccbulk::bulk_modulus(), kji_L),
                                     sum_bulk_plus(ccbulk::rho(), kji_L));
    const Real cs_R = BulkSoundSpeed(set_bulk_minus(ccbulk::bulk_modulus(), kji_R),
                                     sum_bulk_minus(ccbulk::rho(), kji_R));

    Real f_v1, f_v2, f_v3, f_eng, v1face, v2face, v3face;
    const Real signal_speed =
        FLUX_FN(sum_bulk_plus(ccbulk::rho(), kji_L), sum_bulk_minus(ccbulk::rho(), kji_R),
                set_bulk_plus(ccbulk::velocity(0), kji_L),
                set_bulk_minus(ccbulk::velocity(0), kji_R),
                set_bulk_plus(ccbulk::velocity(1), kji_L),
                set_bulk_minus(ccbulk::velocity(1), kji_R),
                set_bulk_plus(ccbulk::velocity(2), kji_L),
                set_bulk_minus(ccbulk::velocity(2), kji_R),
                sum_bulk_plus(ccbulk::internal_energy(), kji_L),
                sum_bulk_minus(ccbulk::internal_energy(), kji_R),
                set_bulk_plus(ccbulk::pressure(), kji_L),
                set_bulk_minus(ccbulk::pressure(), kji_R), cs_L, cs_R, f_v1, f_v2, f_v3,
                f_eng, v1face, v2face, v3face, riemann_vel(kji), dvn(kji), dvt(kji));

    fv(ccbulk::momentum(0), kji) = f_v1;
    fv(ccbulk::momentum(1), kji) = f_v2;
    fv(ccbulk::momentum(2), kji) = f_v3;
    fv(ccbulk::total_material_energy(), kji) = f_eng;
    face_vel(kji) = (DIR == X1DIR) ? v1face : ((DIR == X2DIR) ? v2face : v3face);
    pv(te, ccbulk::face_signal(), kji) = signal_speed;
    StoreFaceVelocity<DIR>(pv, kji, store_vf, v1face, v2face, v3face);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::StrengthFluxes
//! \brief The complete material-strength flux path (used by the "strong" solver). Runs
//!        reconstruction of the deviatoric stress + shear modulus, the strength Riemann
//!        solve (bulk momentum/energy fluxes, incl. the (4/3)gmod sound-speed term and
//!        stress-modified face velocities), and the per-strong-material stress fluxes.
//!        Self-contained: allocates its own strength scratch and builds its own
//!        strong-material map. Assumes standard reconstruction already ran: reads the
//!        set- bulk (velocity/pressure/bulk_modulus), sum-bulk (rho/internal_energy), and
//!        mat_{minus,plus} (rescaled face volume fractions in ccmat::volume_fraction;
//!        vfrac-weighted rho in cm::rho) scratch. The bulk deviatoric stress is summed
//!        (vfrac-weighted) from per-material stresses into a separate 5-component
//!        accumulator (the sum analog of the set/sum bulk split).
template <parthenon::CoordinateDirection DIR, int MAX_STRONG, typename Pack_t,
          typename StrPack_t, typename IdxRange, typename HaloRange, typename Delta,
          typename SetBulk, typename SumBulk, typename Mat, typename Scratch,
          typename StrengthArr>
KOKKOS_INLINE_FUNCTION void
StrengthFluxes(const Pack_t &v, const StrPack_t &vstr, const IdxRange &idx_range,
               const HaloRange &halo_range, Delta delta, const SetBulk &set_bulk_minus,
               const SetBulk &set_bulk_plus, const SumBulk &sum_bulk_minus,
               const SumBulk &sum_bulk_plus, const Mat &mat_minus, const Mat &mat_plus,
               Scratch &face_vel, Scratch &riemann_vel, const int b, const int nmat,
               const RiotReconstruction::Type recon_tag,
               const StrengthArr &mat_strength) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using TE = parthenon::TopologicalElement;
  constexpr auto te = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  constexpr int dir = static_cast<int>(DIR);

  // Build the strong-material map: compact strong index s -> material index m. The stress
  // fields live in their own pack (vstr) restricted to strength materials, so vstr's
  // sparse index IS the compact strong index; we recover the corresponding material index
  // m in v by matching sparse ids. (No assumption about relative pack densification
  // order: m is derived from the matched sparse id.)
  int strong_of[MAX_STRONG]; // compact strong index s -> material index m
  int nstrong = 0;
  for (int m = 0; m < nmat; ++m) {
    if (mat_strength(v(b, cm::rho(m)).sparse_id)) {
      const int s = nstrong++;
      PARTHENON_DEBUG_REQUIRE(
          vstr(b, cm::deviatoric_stress(5 * s)).sparse_id == v(b, cm::rho(m)).sparse_id,
          "Strength index mapping broke: vstr's s-th strong slot does not match the s-th "
          "strong material in v (pack densification order changed).");
      strong_of[s] = m;
    }
  }

  // Strength scratch (allocated only on this path). shear_modulus is SET (reconstructed
  // directly); the bulk stress is SUMMED (vfrac-weighted) from per-material stresses.
  auto sbulk_minus =
      GetTypeIndexedPerPointScratch<Real, set_strength_bulk_recon_types>(halo_range);
  auto sbulk_plus =
      GetTypeIndexedPerPointScratch<Real, set_strength_bulk_recon_types>(halo_range);
  auto smat_minus =
      GetTypeIndexedPerPointScratch<Real, strength_mat_recon_types, MAX_STRONG>(
          halo_range);
  auto smat_plus =
      GetTypeIndexedPerPointScratch<Real, strength_mat_recon_types, MAX_STRONG>(
          halo_range);
  auto bulk_sxx_minus = GetPerPointScratch<Real, 5>(halo_range); // summed bulk stress
  auto bulk_sxx_plus = GetPerPointScratch<Real, 5>(halo_range);

  // Reconstruct bulk shear modulus (like pressure).
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  ReconCells<ccbulk::shear_modulus>(pv, halo_range, delta, sbulk_minus, sbulk_plus,
                                    recon_tag);

  // Zero the bulk-stress accumulators (arena scratch is uninitialized).
  bulk_sxx_minus.Zero();
  bulk_sxx_plus.Zero();
  halo_range.TeamBarrier();

  // Per strong material: reconstruct the 5 stress components and sum vfrac-weighted into
  // the bulk stress (using the rescaled face volume fractions from mat_*).
  for (int s = 0; s < nstrong; ++s) {
    const int m = strong_of[s];
    // deviatoric_stress lives in vstr (restricted to strength materials), whose sparse
    // index is exactly the compact strong index s.
    auto spv = RiotLoop::make_sparse_pack_view(idx_range, vstr, s);
    ReconCells<cm::deviatoric_stress>(spv, halo_range, delta, smat_minus, smat_plus,
                                      recon_tag, s);
    RiotLoop::inner(halo_range, [&](auto kji) {
      const Real vfp = mat_plus(ccmat::volume_fraction(), m, kji);
      const Real vfm = mat_minus(ccmat::volume_fraction(), m, kji);
      for (int c = 0; c < 5; ++c) {
        bulk_sxx_plus(c, kji) += vfp * smat_plus(cm::deviatoric_stress(c), s, kji);
        bulk_sxx_minus(c, kji) += vfm * smat_minus(cm::deviatoric_stress(c), s, kji);
      }
    });
    halo_range.TeamBarrier();
  }

  // Bulk strength Riemann solve.
  auto fv = RiotLoop::make_flux_pack_view(idx_range, v, DIR);
  RiotLoop::inner(idx_range, [&](const auto kji) {
    const auto kji_L = kji - delta;
    const auto kji_R = kji;
    // Sound speed stiffened by shear modulus: bmod += (4/3) gmod.
    constexpr Real tiny_cs = 1.e-12;
    const Real csl = std::max(
        std::sqrt(std::abs((set_bulk_plus(ccbulk::bulk_modulus(), kji_L) +
                            (4.0 / 3.0) * sbulk_plus(ccbulk::shear_modulus(), kji_L)) /
                           sum_bulk_plus(ccbulk::rho(), kji_L))),
        tiny_cs);
    const Real csr = std::max(
        std::sqrt(std::abs((set_bulk_minus(ccbulk::bulk_modulus(), kji_R) +
                            (4.0 / 3.0) * sbulk_minus(ccbulk::shear_modulus(), kji_R)) /
                           sum_bulk_minus(ccbulk::rho(), kji_R))),
        tiny_cs);

    Real f_v1, f_v2, f_v3, f_eng, v1face, v2face, v3face;
    const Real signal_speed = lr_to_flux_strength<dir>(
        sum_bulk_plus(ccbulk::rho(), kji_L), sum_bulk_minus(ccbulk::rho(), kji_R),
        set_bulk_plus(ccbulk::velocity(0), kji_L),
        set_bulk_minus(ccbulk::velocity(0), kji_R),
        set_bulk_plus(ccbulk::velocity(1), kji_L),
        set_bulk_minus(ccbulk::velocity(1), kji_R),
        set_bulk_plus(ccbulk::velocity(2), kji_L),
        set_bulk_minus(ccbulk::velocity(2), kji_R),
        sum_bulk_plus(ccbulk::internal_energy(), kji_L),
        sum_bulk_minus(ccbulk::internal_energy(), kji_R),
        set_bulk_plus(ccbulk::pressure(), kji_L),
        set_bulk_minus(ccbulk::pressure(), kji_R), csl, csr,
        sbulk_plus(ccbulk::shear_modulus(), kji_L),
        sbulk_minus(ccbulk::shear_modulus(), kji_R), bulk_sxx_plus(0, kji_L),
        bulk_sxx_minus(0, kji_R), bulk_sxx_plus(1, kji_L), bulk_sxx_minus(1, kji_R),
        bulk_sxx_plus(2, kji_L), bulk_sxx_minus(2, kji_R), bulk_sxx_plus(3, kji_L),
        bulk_sxx_minus(3, kji_R), bulk_sxx_plus(4, kji_L), bulk_sxx_minus(4, kji_R), f_v1,
        f_v2, f_v3, f_eng, v1face, v2face, v3face, riemann_vel(kji));

    fv(ccbulk::momentum(0), kji) = f_v1;
    fv(ccbulk::momentum(1), kji) = f_v2;
    fv(ccbulk::momentum(2), kji) = f_v3;
    fv(ccbulk::total_material_energy(), kji) = f_eng;
    face_vel(kji) = (DIR == X1DIR) ? v1face : ((DIR == X2DIR) ? v2face : v3face);
    pv(te, ccbulk::face_signal(), kji) = signal_speed;
    // Strength implies store_vf, so always persist the face velocity for the strain-rate
    // source (CalculateStrengthSource).
    StoreFaceVelocity<DIR>(pv, kji, true, v1face, v2face, v3face);
  });

  // Per-strong-material stress fluxes: advect each of the 5 components with the
  // material's rho flux (frho), upwinded on face_vel. (frho is trivially recomputed here
  // to keep the strength path self-contained.)
  for (int s = 0; s < nstrong; ++s) {
    const int m = strong_of[s];
    // stress flux uses the compact strong index s into vstr; rho uses material m in v.
    auto fmat = RiotLoop::make_sparse_flux_pack_view(idx_range, vstr, DIR, s);
    RiotLoop::inner(idx_range, [&](auto kji) {
      const auto kji_L = kji - delta;
      const auto kji_R = kji;
      const Real frho =
          riemann_vel(kji) * ((face_vel(kji) >= 0.0) * mat_plus(cm::rho(), m, kji_L) +
                              (face_vel(kji) < 0.0) * mat_minus(cm::rho(), m, kji_R));
      for (int c = 0; c < 5; ++c) {
        fmat(ccmat::deviatoric_stress(c), kji) =
            frho *
            ((face_vel(kji) >= 0.0) * smat_plus(cm::deviatoric_stress(c), s, kji_L) +
             (face_vel(kji) < 0.0) * smat_minus(cm::deviatoric_stress(c), s, kji_R));
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AdvectionFluxes
//! \brief Anonymous/passive advection fluxes. Each variable flagged Metadata::Advected
//!        is a passive scalar carried in a separate pack (adv), paired with a controlling
//!        material's mass flux. For a material-associated scalar the advection velocity
//!        is that material's rho flux (summed over its phases); for a non-associated
//!        ("anonymous") scalar it is the bulk Riemann velocity. The upwind state is
//!        selected by the sign of that velocity (strictly > 0), matching the old kernel.
//!        Runs after the material-rho flux loop, which has already written
//!        v.flux(ccmat::rho(...)) that we read back as the velocity.
//!
//!        The advected set is discovered at runtime (Metadata::Advected), so adv is a
//!        typeless SparsePack indexed by an anonymous integer -- hence the single-var
//!        loop-abstraction views (make_var_view / make_flux_view) rather than the typed
//!        pack views used elsewhere.
template <parthenon::CoordinateDirection DIR, int MAX_ADV, typename Pack_t,
          typename AdvPack_t, typename IdxRange, typename HaloRange, typename Delta,
          typename Scratch, typename HaloScratch>
KOKKOS_INLINE_FUNCTION void
AdvectionFluxes(const Pack_t &v, const AdvPack_t &adv, const IdxRange &idx_range,
                const HaloRange &halo_range, Delta delta, Scratch &riemann_vel,
                HaloScratch &adv_minus, HaloScratch &adv_plus, const int b,
                const int nmat, const RiotReconstruction::Type recon_tag) {
  namespace ccmat = cell_variables::cell_averaged::mat;

  const int nadv = adv.Size(b);
  if (nadv <= 0) return;
  PARTHENON_DEBUG_REQUIRE(nadv <= MAX_ADV,
                          "More advected variables than MAX_ADV; raise the bound.");

  // Setup: map each advected var to its controlling material's rho pack index (or -1 for
  // an anonymous scalar), and record how many phase contributions to sum. The phase count
  // comes straight off the packed variable's tensor_shape[0] -- no external nphase array
  // needed (phases of one material share a sparse_id and pack contiguously).
  int adv_map[MAX_ADV]; // -1 => anonymous (use riemann_vel); else ccmat::rho pack index
  int nflux[MAX_ADV];   // number of phase contributions to sum
  for (int n = 0; n < nadv; ++n) {
    const int sid = adv.ConsSparseID(b, n);
    adv_map[n] = -1;
    nflux[n] = 1;
    if (sid != parthenon::InvalidSparseID) {
      for (int m = 0; m < nmat; ++m) {
        if (v(b, ccmat::rho(m)).sparse_id == sid) {
          adv_map[n] = v.GetIndex(b, ccmat::rho(m));
          nflux[n] = v(b, ccmat::rho(m)).tensor_shape[0];
          break;
        }
      }
    }
  }

  for (int n = 0; n < nadv; ++n) {
    // Reconstruct the passive scalar into halo scratch (plus = left face, minus = right
    // face), then flux over idx_range. The barrier separates the recon writes (over the
    // halo) from the flux reads.
    auto q = RiotLoop::make_var_view(idx_range, adv.Prims(), n);
    ReconVar(q, halo_range, delta, adv_minus, adv_plus, recon_tag);
    halo_range.TeamBarrier();

    auto fadv = RiotLoop::make_flux_view(idx_range, adv.Cons(), DIR, n);
    // Accumulate over phase contributions, one pass per phase (mask selects init on the
    // first, accumulate thereafter -- matches the old kernel). Each cell touches only its
    // own fadv slot, so no barrier is needed between passes. The velocity is the
    // material's mass flux for phase m (associated) or the bulk Riemann velocity
    // (anonymous, single contribution).
    for (int m = 0; m < nflux[n]; ++m) {
      const Real mask = 1.0 * (m > 0);
      if (adv_map[n] == -1) {
        RiotLoop::inner(idx_range, [&](auto kji) {
          const Real vel = riemann_vel(kji);
          const bool left = vel > 0.0;
          fadv(kji) =
              mask * fadv(kji) + (left ? adv_plus(kji - delta) : adv_minus(kji)) * vel;
        });
      } else {
        auto fvel = RiotLoop::make_flux_view(idx_range, v, DIR, adv_map[n] + m);
        RiotLoop::inner(idx_range, [&](auto kji) {
          const Real vel = fvel(kji);
          const bool left = vel > 0.0;
          fadv(kji) =
              mask * fadv(kji) + (left ? adv_plus(kji - delta) : adv_minus(kji)) * vel;
        });
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateFluxesImpl
//! \brief Templated single-direction flux calculation. Reconstructs bulk and per-material
//!        quantities to faces, rescales reconstructed volume fractions to sum to one,
//!        accumulates the reconstructed bulk state, then solves the bulk Riemann problem
//!        and computes the material density fluxes.
template <parthenon::CoordinateDirection DIR, typename Pack_t, typename StrPack_t,
          typename AdvPack_t, typename StrengthArr>
void CalculateFluxesImpl(MeshData<Real> *md, const Pack_t &v, const StrPack_t &vstr,
                         const AdvPack_t &adv, const RiotReconstruction::Type recon_tag,
                         const RiotReconstruction::Type vfrac_recon_tag,
                         const RiemannSolver rsolver_tag, const bool store_vf,
                         const StrengthArr &mat_strength, const bool do_viscosity) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  const int nblocks = v.GetNBlocks();

  // Create index space for faces
  using lt = RiotUtils::LoopType<>;
  constexpr parthenon::TopologicalElement face =
      (DIR == X1DIR) ? parthenon::TopologicalElement::F1
                     : ((DIR == X2DIR) ? parthenon::TopologicalElement::F2
                                       : parthenon::TopologicalElement::F3);

  const int nhalo = 2 * do_viscosity;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, nhalo, nblocks, md, face);

  // Select the appropriate halo range to reconstruct over

  using halo = std::tuple_element_t<
      DIR - 1, std::tuple<RiotLoop::halo::minus_i_t, RiotLoop::halo::minus_j_t,
                          RiotLoop::halo::minus_k_t>>;

  AddTypeIndexedPerPointScratch<Real, halo, sum_bulk_recon_types>(idx_space, 2);
  AddTypeIndexedPerPointScratch<Real, halo, set_bulk_recon_types>(idx_space, 2);
  AddTypeIndexedPerPointScratch<Real, halo, mat_recon_types, MAX_MATERIALS>(idx_space, 2);
  AddPerPointScratch<Real, halo>(idx_space, 4);
  AddPerPointScratch<Real>(idx_space, 4); // face_vel, riemann_vel, dvn, dvt
  // Strength path scratch: set shear modulus, per-strong-material stress, summed bulk
  // stress (5 components). Only touched on the strength ("strong") solver path.
  AddTypeIndexedPerPointScratch<Real, halo, set_strength_bulk_recon_types>(idx_space, 2);
  AddTypeIndexedPerPointScratch<Real, halo, strength_mat_recon_types, MAX_STRONG>(
      idx_space, 2);
  AddPerPointScratch<Real, halo, 5>(idx_space, 2); // summed bulk stress accumulators
  AddPerPointScratch<Real, halo>(idx_space, 2);    // advection recon (minus/plus)

  auto delta = idx_space.GetDelta(DIR);
  // Directional basis (normal + cyclic transverse offsets/components) for GetVdiff.
  auto basis = RiotUtils::MakeDirBasis<DIR>(idx_space);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::volume_fraction());

        // Index range over which quantities are reconstructed on plus and minus faces
        auto halo_range = idx_range.template AddHalo<halo>();

        // Per material recon and collect slopes for volume fraction rescaling
        auto mat_minus =
            GetTypeIndexedPerPointScratch<Real, mat_recon_types, MAX_MATERIALS>(
                halo_range);
        auto mat_plus =
            GetTypeIndexedPerPointScratch<Real, mat_recon_types, MAX_MATERIALS>(
                halo_range);
        auto scale_pos_plus = GetPerPointScratch<Real>(halo_range);
        auto scale_neg_plus = GetPerPointScratch<Real>(halo_range);
        auto scale_pos_minus = GetPerPointScratch<Real>(halo_range);
        auto scale_neg_minus = GetPerPointScratch<Real>(halo_range);
        scale_pos_plus.Zero();
        scale_neg_plus.Zero();
        scale_pos_minus.Zero();
        scale_neg_minus.Zero();
        halo_range.TeamBarrier();
        for (int m = 0; m < nmat; ++m) {
          auto spv = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          // Material density and internal energy use a tighter MC slope limit (1.5);
          // bulk and vfrac use the default THETA (1.99).
          ReconCells<cm::rho, ccmat::internal_energy>(spv, halo_range, delta, mat_minus,
                                                      mat_plus, recon_tag, m, 1.5);
          ReconCells<ccmat::volume_fraction>(spv, halo_range, delta, mat_minus, mat_plus,
                                             vfrac_recon_tag, m);
          RiotLoop::inner(halo_range, [&](auto kji) {
            const Real vf_p =
                std::min(1.0, std::max(mat_plus(ccmat::volume_fraction(), m, kji), 0.0));
            const Real vf_m =
                std::min(1.0, std::max(mat_minus(ccmat::volume_fraction(), m, kji), 0.0));
            const Real vf_c = spv(ccmat::volume_fraction(), kji);

            const Real slope_m = vf_c - vf_m;
            const Real slope_p = vf_p - vf_c;

            // Temporarily store slopes
            mat_plus(ccmat::volume_fraction(), m, kji) = slope_p;
            mat_minus(ccmat::volume_fraction(), m, kji) = slope_m;

            // Accumulate positive and negative "slopes" on either side of the cell
            // (require both because reconstruction may not be piecewise constant or
            // linear)
            scale_pos_plus(kji) += (slope_p > 0.0) * slope_p;
            scale_neg_plus(kji) += (slope_p < 0.0) * slope_p;

            scale_pos_minus(kji) += (slope_m > 0.0) * slope_m;
            scale_neg_minus(kji) += (slope_m < 0.0) * slope_m;
          });
          halo_range.TeamBarrier();
        }

        // Calculate the vfrac slope scale factors required for vfrac recons to sum to one
        RiotLoop::inner(halo_range, [&](auto kji) {
          const Real dv_pos_plus = scale_pos_plus(kji);
          const Real dv_neg_plus = scale_neg_plus(kji);
          scale_pos_plus(kji) = (dv_pos_plus + dv_neg_plus > 0.0)
                                    ? -dv_neg_plus / (dv_pos_plus + 1.e-16)
                                    : 1.0;
          scale_neg_plus(kji) = (dv_pos_plus + dv_neg_plus < 0.0)
                                    ? -dv_pos_plus / (dv_neg_plus - 1.e-16)
                                    : 1.0;

          const Real dv_pos_minus = scale_pos_minus(kji);
          const Real dv_neg_minus = scale_neg_minus(kji);
          scale_pos_minus(kji) = (dv_pos_minus + dv_neg_minus > 0.0)
                                     ? -dv_neg_minus / (dv_pos_minus + 1.e-16)
                                     : 1.0;
          scale_neg_minus(kji) = (dv_pos_minus + dv_neg_minus < 0.0)
                                     ? -dv_pos_minus / (dv_neg_minus - 1.e-16)
                                     : 1.0;
        });
        halo_range.TeamBarrier();

        // Reconstruct
        auto sum_bulk_minus =
            GetTypeIndexedPerPointScratch<Real, sum_bulk_recon_types>(halo_range);
        auto sum_bulk_plus =
            GetTypeIndexedPerPointScratch<Real, sum_bulk_recon_types>(halo_range);
        sum_bulk_plus.Zero();
        sum_bulk_minus.Zero();
        halo_range.TeamBarrier();
        for (int m = 0; m < nmat; ++m) {
          auto spv = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          RiotLoop::inner(halo_range, [&](auto kji) {
            mat_plus(cm::rho(), m, kji) = std::max(0.0, mat_plus(cm::rho(), m, kji));
            mat_minus(cm::rho(), m, kji) = std::max(0.0, mat_minus(cm::rho(), m, kji));

            const Real vf_c = spv(ccmat::volume_fraction(), kji);

            Real slope_plus = mat_plus(ccmat::volume_fraction(), m, kji);
            slope_plus *= (slope_plus > 0.0) * scale_pos_plus(kji) +
                          (slope_plus < 0.0) * scale_neg_plus(kji);
            mat_plus(ccmat::volume_fraction(), m, kji) = vf_c + slope_plus;

            Real slope_minus = mat_minus(ccmat::volume_fraction(), m, kji);
            slope_minus *= (slope_minus > 0.0) * scale_pos_minus(kji) +
                           (slope_minus < 0.0) * scale_neg_minus(kji);
            mat_minus(ccmat::volume_fraction(), m, kji) = vf_c - slope_minus;

            // Transform to ccmat::rho in the cm::rho scratch
            mat_plus(cm::rho(), m, kji) *= mat_plus(ccmat::volume_fraction(), m, kji);
            mat_minus(cm::rho(), m, kji) *= mat_minus(ccmat::volume_fraction(), m, kji);

            // Accumulate into reconstructed bulk quantities
            sum_bulk_plus(ccbulk::rho(), kji) += mat_plus(cm::rho(), m, kji);
            sum_bulk_minus(ccbulk::rho(), kji) += mat_minus(cm::rho(), m, kji);
            sum_bulk_plus(ccbulk::internal_energy(), kji) +=
                mat_plus(ccmat::internal_energy(), m, kji);
            sum_bulk_minus(ccbulk::internal_energy(), kji) +=
                mat_minus(ccmat::internal_energy(), m, kji);
          });
          halo_range.TeamBarrier();
        }

        // Reconstruct remaining bulk quantities directly (pressure, bmod, velocity)
        auto set_bulk_minus =
            GetTypeIndexedPerPointScratch<Real, set_bulk_recon_types>(halo_range);
        auto set_bulk_plus =
            GetTypeIndexedPerPointScratch<Real, set_bulk_recon_types>(halo_range);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        ReconCells<ccbulk::pressure, ccbulk::bulk_modulus, ccbulk::velocity>(
            pv, halo_range, delta, set_bulk_minus, set_bulk_plus, recon_tag);
        halo_range.TeamBarrier();

        // Bulk Riemann flux, dispatched on the solver. Each solver's bulk loop is a
        // BulkRiemannFluxes<DIR, FLUX_FN> instantiation. The low-Mach solvers
        // (chllc/lhllc) additionally need the transverse velocity differences dvn/dvt
        // from GetVdiff and run through BulkRiemannFluxesLM.
        auto face_vel = GetPerPointScratch<Real>(idx_range);
        auto riemann_vel = GetPerPointScratch<Real>(idx_range);
        auto dvn = GetPerPointScratch<Real>(idx_range);
        auto dvt = GetPerPointScratch<Real>(idx_range);
        constexpr int dir = static_cast<int>(DIR);
        switch (rsolver_tag) {
        case RiemannSolver::hllc:
          BulkRiemannFluxes<DIR, lr_to_flux_hllc<dir>>(
              v, idx_range, delta, set_bulk_minus, set_bulk_plus, sum_bulk_minus,
              sum_bulk_plus, face_vel, riemann_vel, store_vf);
          break;
        case RiemannSolver::hllcf:
          BulkRiemannFluxes<DIR, lr_to_flux_fleischmann<dir>>(
              v, idx_range, delta, set_bulk_minus, set_bulk_plus, sum_bulk_minus,
              sum_bulk_plus, face_vel, riemann_vel, store_vf);
          break;
        case RiemannSolver::hll:
          BulkRiemannFluxes<DIR, lr_to_flux_hll<dir>>(
              v, idx_range, delta, set_bulk_minus, set_bulk_plus, sum_bulk_minus,
              sum_bulk_plus, face_vel, riemann_vel, store_vf);
          break;
        case RiemannSolver::chllc:
          GetVdiff<DIR>(v, idx_range, basis, dvn, dvt);
          idx_range.TeamBarrier();
          BulkRiemannFluxesLM<DIR, lr_to_flux_chllc<dir>>(
              v, idx_range, delta, set_bulk_minus, set_bulk_plus, sum_bulk_minus,
              sum_bulk_plus, face_vel, riemann_vel, dvn, dvt, store_vf);
          break;
        case RiemannSolver::lhllc:
          GetVdiff<DIR>(v, idx_range, basis, dvn, dvt);
          idx_range.TeamBarrier();
          BulkRiemannFluxesLM<DIR, lr_to_flux_lhllc<dir>>(
              v, idx_range, delta, set_bulk_minus, set_bulk_plus, sum_bulk_minus,
              sum_bulk_plus, face_vel, riemann_vel, dvn, dvt, store_vf);
          break;
        case RiemannSolver::strong:
          StrengthFluxes<DIR, MAX_STRONG>(v, vstr, idx_range, halo_range, delta,
                                          set_bulk_minus, set_bulk_plus, sum_bulk_minus,
                                          sum_bulk_plus, mat_minus, mat_plus, face_vel,
                                          riemann_vel, b, nmat, recon_tag, mat_strength);
          break;
        }

        // Material density fluxes
        for (int m = 0; m < nmat; ++m) {
          auto fmat = RiotLoop::make_sparse_flux_pack_view(idx_range, v, DIR, m);
          RiotLoop::inner(idx_range, [&](auto kji) {
            const auto kji_L = kji - delta;
            const auto kji_R = kji;
            fmat(ccmat::rho(), kji) =
                riemann_vel(kji) *
                ((face_vel(kji) >= 0.0) * mat_plus(cm::rho(), m, kji_L) +
                 (face_vel(kji) < 0.0) * mat_minus(cm::rho(), m, kji_R));
          });
        }

        // Anonymous/passive advection: consumes the material rho fluxes just written and
        // the bulk riemann_vel. Reconstruction scratch is over the halo range.
        idx_range.TeamBarrier();
        auto adv_minus = GetPerPointScratch<Real>(halo_range);
        auto adv_plus = GetPerPointScratch<Real>(halo_range);
        AdvectionFluxes<DIR, MAX_ADV>(v, adv, idx_range, halo_range, delta, riemann_vel,
                                      adv_minus, adv_plus, b, nmat, recon_tag);
      });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::CalculateFluxes
//! \brief Reconstruct state and calculate hydrodynamic fluxes in all directions
TaskStatus CalculateFluxes(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  if (md->NumBlocks() == 0) return TaskStatus::complete;
  auto pm = md->GetParentPointer();
  const int ndim = pm->ndim;

  // Get parameters
  auto &options = pm->packages.Get("hydro");
  const auto recon_tag = options->Param<RiotReconstruction::Type>("recon");
  const auto vfrac_recon_tag = options->Param<RiotReconstruction::Type>("vfrac_recon");
  auto rsolver_tag = options->Param<RiemannSolver>("riemann_solver");
  const bool do_strength = pm->packages.Get("riot")->Param<bool>("do_strength");
  if (do_strength) rsolver_tag = RiemannSolver::strong;
  const bool store_vf = options->Param<bool>("store_vf");
  const auto &mat_strength =
      pm->packages.Get("materials")->Param<parthenon::ParArray1D<bool>>("d.strong");
  const auto &strength_mats =
      pm->packages.Get("materials")->Param<std::vector<int>>("strength_mats");

  // ionization parameters
  const bool do_ionization = pm->packages.Get("riot")->Param<bool>("do_ionization");
  bool do_plasma_viscosity = false;
  if (do_ionization) {
    do_plasma_viscosity = pm->packages.Get("ionization")->Param<bool>("plasma_viscosity");
  }

  // Create pack of reconstructed vars, fluxes, and auxiliary vars. The per-material
  // sparse fields here are allocated on *all* materials, so a single sparse index
  // addresses them uniformly.
  auto v = riot::MakePack<ccbulk::velocity, ccbulk::pressure, ccbulk::bulk_modulus,
                          ccbulk::shear_modulus, ccbulk::momentum,
                          ccbulk::total_material_energy, ccbulk::face_signal,
                          ccbulk::face_velocity, ccmat::volume_fraction,
                          ccmat::internal_energy, ccmat::rho, cm::rho>(
      md, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  // Deviatoric stress is allocated only on strength-supporting materials, so it lives in
  // its own pack restricted to strength_mats. Packed this way its sparse index is exactly
  // the compact "strong" index -- so it is never over-indexed by a material index (unlike
  // if it shared the mixed-sparsity pack v).
  auto vstr = riot::MakePack<cm::deviatoric_stress, ccmat::deviatoric_stress>(
      md, strength_mats, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});

  const bool do_viscosity = (do_ionization && do_plasma_viscosity);

  auto adv = MakeAdvectionPack(md);

  CalculateFluxesImpl<X1DIR>(md, v, vstr, adv, recon_tag, vfrac_recon_tag, rsolver_tag,
                             store_vf, mat_strength, do_viscosity);
  if (ndim > 1)
    CalculateFluxesImpl<X2DIR>(md, v, vstr, adv, recon_tag, vfrac_recon_tag, rsolver_tag,
                               store_vf, mat_strength, do_viscosity);
  if (ndim > 2)
    CalculateFluxesImpl<X3DIR>(md, v, vstr, adv, recon_tag, vfrac_recon_tag, rsolver_tag,
                               store_vf, mat_strength, do_viscosity);

  return TaskStatus::complete;
}

} // namespace Hydro
