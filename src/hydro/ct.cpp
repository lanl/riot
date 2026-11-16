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
// This file was made in part with generative AI.
//
// Constrained transport for ideal MHD. The face-centered magnetic field ccbulk::bfield
// is evolved with
//   dB/dt = -curl(E),  E = -v x B,
// where the EMF E lives on cell edges, in the edge flux of bfield (allocated by
// Metadata::WithFluxes, so Parthenon's edge flux correction handles AMR). The edge EMFs
// are built from the face EMFs returned by the HLLD solver with the upwind
// Gardiner & Stone (2005) corner integration, as in Athena++.
//
// bfield has node memory layout (required for FillGhost on face fields), so it is only
// accessed here with explicit (b, TE, var, k, j, i) indexing.

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>

#include "hydro.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

namespace {
using TE = parthenon::TopologicalElement;

KOKKOS_INLINE_FUNCTION constexpr TE FaceTE(const int d) { return (d == 0) ? TE::F1 : (d == 1 ? TE::F2 : TE::F3); }
KOKKOS_INLINE_FUNCTION constexpr TE EdgeTE(const int d) { return (d == 0) ? TE::E1 : (d == 1 ? TE::E2 : TE::E3); }

template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real InvDx(const Pack_t &v, const int b, const int d) {
  auto &coords = v.GetCoordinates(b);
  return 1.0 / ((d == 0)   ? coords.template Dx<X1DIR>()
                : (d == 1) ? coords.template Dx<X2DIR>()
                           : coords.template Dx<X3DIR>());
}

//----------------------------------------------------------------------------------------
//! \brief Gardiner & Stone (2005) upwind integration of the EMF component n to the cell
//!        edge e = (k, j, i). p = (n+1)%3 and q = (n+2)%3 are the two transverse
//!        directions; the edge sits at the lower p- and q-faces of cell (k, j, i). The four
//!        surrounding cells are c00 = e-dp-dq, c10 = e-dq, c01 = e-dp, c11 = e. The face
//!        EMFs come from the F_p faces at e-dq and e (slot0) and the F_q faces at e-dp and
//!        e (slot1). Each face's upwind weight is w = 0.5 + clamp(512 dt vrho/dx, +-0.5)
//!        (Athena++ GetWeightForCT), w = 1 meaning fully upwinded from the lower cell.
template <int n, typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real CornerEMF(const Pack_t &v, const int b, const int k,
                                           const int j, const int i, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int p = (n + 1) % 3;
  constexpr int q = (n + 2) % 3;
  constexpr TE tp = FaceTE(p);
  constexpr TE tq = FaceTE(q);
  // Index offsets of the lower p and q neighbors
  constexpr int pi = (p == 0), pj = (p == 1), pk = (p == 2);
  constexpr int qi = (q == 0), qj = (q == 1), qk = (q == 2);

  // Cell-centered reference EMF E_n = -(v_p B_q - v_q B_p)
  auto ecc = [&](const int kk, const int jj, const int ii) {
    return -(v(b, ccbulk::velocity(p), kk, jj, ii) * v(b, ccbulk::bcc(q), kk, jj, ii) -
             v(b, ccbulk::velocity(q), kk, jj, ii) * v(b, ccbulk::bcc(p), kk, jj, ii));
  };
  const Real c00 = ecc(k - pk - qk, j - pj - qj, i - pi - qi);
  const Real c10 = ecc(k - qk, j - qj, i - qi);
  const Real c01 = ecc(k - pk, j - pj, i - pi);
  const Real c11 = ecc(k, j, i);

  const Real fp0 = v(b, tp, ccbulk::face_emf(0), k - qk, j - qj, i - qi);
  const Real fp1 = v(b, tp, ccbulk::face_emf(0), k, j, i);
  const Real fq0 = v(b, tq, ccbulk::face_emf(1), k - pk, j - pj, i - pi);
  const Real fq1 = v(b, tq, ccbulk::face_emf(1), k, j, i);

  auto weight = [&](const TE te, const Real idx, const int kk, const int jj,
                    const int ii) {
    const Real v_over_c = 512.0 * dt * v(b, te, ccbulk::face_vrho(), kk, jj, ii) * idx;
    return 0.5 + std::max(-0.5, std::min(0.5, v_over_c));
  };
  const Real idxp = InvDx(v, b, p);
  const Real idxq = InvDx(v, b, q);
  const Real wp0 = weight(tp, idxp, k - qk, j - qj, i - qi);
  const Real wp1 = weight(tp, idxp, k, j, i);
  const Real wq0 = weight(tq, idxq, k - pk, j - pj, i - pi);
  const Real wq1 = weight(tq, idxq, k, j, i);

  // Upwinded q-derivatives along the F_p face line, and p-derivatives along the F_q line
  const Real dq_term = wp0 * (fq0 - c00) + (1.0 - wp0) * (fq1 - c10) -
                       wp1 * (c01 - fq0) - (1.0 - wp1) * (c11 - fq1);
  const Real dp_term = wq0 * (fp0 - c00) + (1.0 - wq0) * (fp1 - c01) -
                       wq1 * (c10 - fp0) - (1.0 - wq1) * (c11 - fp1);
  return 0.25 * (fp0 + fp1 + fq0 + fq1 + dq_term + dp_term);
}

template <typename Pack_t>
void SetEdgeEMF(MeshData<Real> *md, const Pack_t &v, const int n, const int ndim,
                const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const TE te = EdgeTE(n);
  const auto ib = md->GetBoundsI(IndexDomain::interior, te);
  const auto jb = md->GetBoundsJ(IndexDomain::interior, te);
  const auto kb = md->GetBoundsK(IndexDomain::interior, te);
  // Number of active dimensions transverse to this edge selects the integration
  //   2: full corner integration; 1: the edge coincides with the faces normal to the
  //   single active transverse direction; 0: no transverse variation, E_n is unused.
  const int p = (n + 1) % 3;
  const int q = (n + 2) % 3;
  const bool p_active = (p < ndim);
  const bool q_active = (q < ndim);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real e = 0.0;
        if (p_active && q_active) {
          e = (n == 0)   ? CornerEMF<0>(v, b, k, j, i, dt)
              : (n == 1) ? CornerEMF<1>(v, b, k, j, i, dt)
                         : CornerEMF<2>(v, b, k, j, i, dt);
        } else if (p_active) {
          // E_n is slot0 on the F_p faces
          e = v(b, FaceTE(p), ccbulk::face_emf(0), k, j, i);
        } else if (q_active) {
          // E_n is slot1 on the F_q faces
          e = v(b, FaceTE(q), ccbulk::face_emf(1), k, j, i);
        }
        v.flux(b, te, ccbulk::bfield(), k, j, i) = e;
      });
}

template <typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION Real CellBComponent(const Pack_t &v, const int b, const int d,
                                                const int k, const int j, const int i,
                                                const bool active) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const TE te = FaceTE(d);
  const Real blo = v(b, te, ccbulk::bfield(), k, j, i);
  if (!active) return blo;
  const Real bhi =
      v(b, te, ccbulk::bfield(), k + (d == 2), j + (d == 1), i + (d == 0));
  return 0.5 * (blo + bhi);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyFaceBToAligned
//! \brief Copy the normal face field of each active direction into the cell-aligned face
//!        scratch ccbulk::bface_aligned, so that the flux kernel can read it through the
//!        flat pack views. Covers every cell-memory face index (entire domain).
void CopyFaceBToAligned(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const int ndim = md->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bface_aligned>(md);
  if (v.GetNBlocks() == 0) return;
  const auto ib = md->GetBoundsI(IndexDomain::entire);
  const auto jb = md->GetBoundsJ(IndexDomain::entire);
  const auto kb = md->GetBoundsK(IndexDomain::entire);
  for (int d = 0; d < ndim; ++d) {
    const TE te = FaceTE(d);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          v(b, te, ccbulk::bface_aligned(), k, j, i) = v(b, te, ccbulk::bfield(), k, j, i);
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::CalculateCornerEMF
//! \brief Fill the edge fluxes of bfield with the edge EMFs E (not -E), computed from the
//!        face EMFs of the most recent CalculateFluxes call.
TaskStatus CalculateCornerEMF(MeshData<Real> *md, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::PDOpt;
  if (md->NumBlocks() == 0) return TaskStatus::complete;
  const int ndim = md->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bcc, ccbulk::velocity, ccbulk::face_emf,
                          ccbulk::face_vrho>(md, std::vector<int>{},
                                             std::set<PDOpt>{PDOpt::WithFluxes});
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  for (int n = 0; n < 3; ++n) {
    SetEdgeEMF(md, v, n, ndim, dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::UpdateFaceFields
//! \brief Low-storage RK stage update of the face field, matching UpdateToNextStage:
//!          u = gam0 u + gam1 u0 - beta_dt curl(E),
//!        with E read from the edge fluxes of u (after flux correction).
TaskStatus UpdateFaceFields(MeshData<Real> *umd, MeshData<Real> *u0md, const Real gam0,
                            const Real gam1, const Real beta_dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::PDOpt;
  if (umd->NumBlocks() == 0) return TaskStatus::complete;
  const int ndim = umd->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield>(umd, std::vector<int>{},
                                          std::set<PDOpt>{PDOpt::WithFluxes});
  auto v0 = riot::MakePack<ccbulk::bfield>(u0md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  for (int d = 0; d < 3; ++d) {
    const TE te = FaceTE(d);
    // (curl E)_d = d_p E_q - d_q E_p with p = (d+1)%3, q = (d+2)%3. The E_q edges are
    // offset from the F_d face along p, the E_p edges along q.
    const int p = (d + 1) % 3;
    const int q = (d + 2) % 3;
    const TE ep = EdgeTE(p);
    const TE eq = EdgeTE(q);
    const bool p_active = (p < ndim);
    const bool q_active = (q < ndim);
    const auto ib = umd->GetBoundsI(IndexDomain::interior, te);
    const auto jb = umd->GetBoundsJ(IndexDomain::interior, te);
    const auto kb = umd->GetBoundsK(IndexDomain::interior, te);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          Real curl = 0.0;
          if (p_active) {
            curl += (v.flux(b, eq, ccbulk::bfield(), k + (p == 2), j + (p == 1),
                            i + (p == 0)) -
                     v.flux(b, eq, ccbulk::bfield(), k, j, i)) *
                    InvDx(v, b, p);
          }
          if (q_active) {
            curl -= (v.flux(b, ep, ccbulk::bfield(), k + (q == 2), j + (q == 1),
                            i + (q == 0)) -
                     v.flux(b, ep, ccbulk::bfield(), k, j, i)) *
                    InvDx(v, b, q);
          }
          v(b, te, ccbulk::bfield(), k, j, i) =
              gam0 * v(b, te, ccbulk::bfield(), k, j, i) +
              gam1 * v0(b, te, ccbulk::bfield(), k, j, i) - beta_dt * curl;
        });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::FaceToCellB
//! \brief Average the face field to cell centers (ccbulk::bcc) over domain. Components in
//!        collapsed directions are uniform across the cell and copied.
void FaceToCellB(MeshData<Real> *md, IndexDomain domain) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const int ndim = md->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bcc>(md);
  if (v.GetNBlocks() == 0) return;
  const auto ib = md->GetBoundsI(domain);
  const auto jb = md->GetBoundsJ(domain);
  const auto kb = md->GetBoundsK(domain);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int d = 0; d < 3; ++d) {
          v(b, ccbulk::bcc(d), k, j, i) = CellBComponent(v, b, d, k, j, i, d < ndim);
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateDivB
//! \brief Discrete divergence of the face field in interior cells (ccbulk::divb).
void CalculateDivB(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const int ndim = md->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::divb>(md);
  if (v.GetNBlocks() == 0) return;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real div = 0.0;
        for (int d = 0; d < ndim; ++d) {
          const TE te = FaceTE(d);
          div += (v(b, te, ccbulk::bfield(), k + (d == 2), j + (d == 1), i + (d == 0)) -
                  v(b, te, ccbulk::bfield(), k, j, i)) *
                 InvDx(v, b, d);
        }
        v(b, ccbulk::divb(), k, j, i) = div;
      });
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::total_magnetic_energy
//! \brief History output: volume-integrated magnetic energy B^2/2 (cell-centered field).
Real total_magnetic_energy(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bcc>(md);
  if (v.GetNBlocks() == 0) return 0.0;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  Real eb = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &leb) {
        auto &coords = v.GetCoordinates(b);
        leb += 0.5 *
               (SQR(v(b, ccbulk::bcc(0), k, j, i)) + SQR(v(b, ccbulk::bcc(1), k, j, i)) +
                SQR(v(b, ccbulk::bcc(2), k, j, i))) *
               coords.CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(eb));
  return eb;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::max_divb
//! \brief History output: max |div B| dx_min over interior cells, recomputed from the
//!        face field (so it does not depend on when ccbulk::divb was last filled).
Real max_divb(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const int ndim = md->GetParentPointer()->ndim;
  auto v = riot::MakePack<ccbulk::bfield>(md);
  if (v.GetNBlocks() == 0) return 0.0;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  Real divb_max = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        Real div = 0.0;
        Real dxmin = std::numeric_limits<Real>::max();
        for (int d = 0; d < ndim; ++d) {
          const TE te = FaceTE(d);
          const Real idx = InvDx(v, b, d);
          div += (v(b, te, ccbulk::bfield(), k + (d == 2), j + (d == 1), i + (d == 0)) -
                  v(b, te, ccbulk::bfield(), k, j, i)) *
                 idx;
          dxmin = std::min(dxmin, 1.0 / idx);
        }
        lmax = std::max(lmax, std::abs(div) * dxmin);
      },
      Kokkos::Max<Real>(divb_max));
  return divb_max;
}

} // namespace Hydro
