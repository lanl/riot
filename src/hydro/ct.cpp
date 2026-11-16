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
// bfield and its edge fluxes have node memory layout (required for FillGhost on face
// fields). Kernels touching only those use flat views over an NN-layout index space;
// kernels that mix them with cell-layout fields use LoopConstraint::DifferentMemSpaces.
// All direction and dimensionality choices are template parameters, so the kernels carry
// no runtime branches on them; the public entry points dispatch on ndim once.

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>

#include "hydro.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

namespace {
using TE = parthenon::TopologicalElement;

KOKKOS_INLINE_FUNCTION constexpr TE FaceTE(const int d) { return (d == 0) ? TE::F1 : (d == 1 ? TE::F2 : TE::F3); }
KOKKOS_INLINE_FUNCTION constexpr TE EdgeTE(const int d) { return (d == 0) ? TE::E1 : (d == 1 ? TE::E2 : TE::E3); }
constexpr parthenon::CoordinateDirection Dir(const int d) {
  return static_cast<parthenon::CoordinateDirection>(d + 1);
}

//! Edge length, face area or cell volume of element el at index idx of idx_range
template <TE el, typename Coords_t, typename Range_t, typename Idx_t>
KOKKOS_FORCEINLINE_FUNCTION Real Measure(const Coords_t &coords, const Range_t &idx_range,
                                         const Idx_t idx) {
  const auto [k, j, i] = idx_range.GetKJI(idx);
  return coords.template Volume<el>(k, j, i);
}

//! Physical cell width along direction d at index idx of idx_range
template <int d, typename Coords_t, typename Range_t, typename Idx_t>
KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const Coords_t &coords, const Range_t &idx_range,
                                           const Idx_t idx) {
  const auto [k, j, i] = idx_range.GetKJI(idx);
  return coords.template CellWidth<d + 1>(k, j, i);
}

//----------------------------------------------------------------------------------------
//! \brief Gardiner & Stone (2005) upwind integration of the EMF component n to the cell
//!        edge e. p = (n+1)%3 and q = (n+2)%3 are the two transverse directions, and dp,
//!        dq the corresponding unit offsets; the edge sits at the lower p- and q-faces of
//!        cell e. The four surrounding cells are c00 = e-dp-dq, c10 = e-dq, c01 = e-dp,
//!        c11 = e. The face EMFs come from the F_p faces at e-dq and e (slot0) and the F_q
//!        faces at e-dp and e (slot1). Each face's upwind weight is
//!        w = 0.5 + clamp(512 dt vrho/dx, +-0.5) (Athena++ GetWeightForCT), w = 1 meaning
//!        fully upwinded from the lower cell; wfac_{p,q} = 512 dt / dx_{p,q}, with dx the
//!        physical cell width of cell e.
template <int n, typename View_t, typename Idx_t, typename Off_t>
KOKKOS_FORCEINLINE_FUNCTION Real CornerEMF(const View_t &pv, const Idx_t e, const Off_t dp,
                                           const Off_t dq, const Real wfac_p,
                                           const Real wfac_q) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int p = (n + 1) % 3;
  constexpr int q = (n + 2) % 3;
  constexpr TE tp = FaceTE(p);
  constexpr TE tq = FaceTE(q);

  // Cell-centered reference EMF E_n = -(v_p B_q - v_q B_p)
  auto ecc = [&](const auto c) {
    return -(pv(ccbulk::velocity(p), c) * pv(ccbulk::bcc(q), c) -
             pv(ccbulk::velocity(q), c) * pv(ccbulk::bcc(p), c));
  };
  const Real c00 = ecc(e - dp - dq);
  const Real c10 = ecc(e - dq);
  const Real c01 = ecc(e - dp);
  const Real c11 = ecc(e);

  const Real fp0 = pv(tp, ccbulk::face_emf(0), e - dq);
  const Real fp1 = pv(tp, ccbulk::face_emf(0), e);
  const Real fq0 = pv(tq, ccbulk::face_emf(1), e - dp);
  const Real fq1 = pv(tq, ccbulk::face_emf(1), e);

  auto weight = [&](const TE te, const Real wfac, const auto f) {
    const Real v_over_c = wfac * pv(te, ccbulk::face_vrho(), f);
    return 0.5 + std::max(-0.5, std::min(0.5, v_over_c));
  };
  const Real wp0 = weight(tp, wfac_p, e - dq);
  const Real wp1 = weight(tp, wfac_p, e);
  const Real wq0 = weight(tq, wfac_q, e - dp);
  const Real wq1 = weight(tq, wfac_q, e);

  // Upwinded q-derivatives along the F_p face line, and p-derivatives along the F_q line
  const Real dq_term = wp0 * (fq0 - c00) + (1.0 - wp0) * (fq1 - c10) -
                       wp1 * (c01 - fq0) - (1.0 - wp1) * (c11 - fq1);
  const Real dp_term = wq0 * (fp0 - c00) + (1.0 - wq0) * (fp1 - c01) -
                       wq1 * (c10 - fp0) - (1.0 - wq1) * (c11 - fp1);
  return 0.25 * (fp0 + fp1 + fq0 + fq1 + dq_term + dp_term);
}

//----------------------------------------------------------------------------------------
//! \brief Fill the E_n edge flux of bfield. The number of active dimensions transverse
//!        to the edge selects the integration -- 2: full corner integration; 1: the edge
//!        coincides with the faces normal to the single active transverse direction; 0:
//!        no transverse variation, E_n is unused and set to zero.
template <int n, int NDIM, typename Pack_t>
void SetEdgeEMF(MeshData<Real> *md, const Pack_t &v, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int p = (n + 1) % 3;
  constexpr int q = (n + 2) % 3;
  constexpr bool p_active = (p < NDIM);
  constexpr bool q_active = (q < NDIM);
  constexpr int edge_dir = static_cast<int>(EdgeTE(n)) % 3 + 1;

  // Cell-layout face_emf/face_vrho/velocity/bcc are mixed with the node-layout edge flux
  using lt = RiotUtils::LoopType<LoopConstraint::DifferentMemSpaces>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, EdgeTE(n));
  const auto dp = idx_space.GetDelta(Dir(p));
  const auto dq = idx_space.GetDelta(Dir(q));
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto fv = RiotLoop::make_flux_pack_view(idx_range, v, edge_dir);
        RiotLoop::inner(idx_range, [&](const auto e) {
          if constexpr (p_active && q_active) {
            const Real wfac_p = 512.0 * dt / CellWidth<p>(coords, idx_range, e);
            const Real wfac_q = 512.0 * dt / CellWidth<q>(coords, idx_range, e);
            fv(ccbulk::bfield(), e) = CornerEMF<n>(pv, e, dp, dq, wfac_p, wfac_q);
          } else if constexpr (p_active) {
            // E_n is slot0 on the F_p faces
            fv(ccbulk::bfield(), e) = pv(FaceTE(p), ccbulk::face_emf(0), e);
          } else if constexpr (q_active) {
            // E_n is slot1 on the F_q faces
            fv(ccbulk::bfield(), e) = pv(FaceTE(q), ccbulk::face_emf(1), e);
          } else {
            fv(ccbulk::bfield(), e) = 0.0;
          }
        });
      });
}

template <int NDIM>
void CalculateCornerEMFImpl(MeshData<Real> *md, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::PDOpt;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bcc, ccbulk::velocity, ccbulk::face_emf,
                          ccbulk::face_vrho>(md, std::vector<int>{},
                                             std::set<PDOpt>{PDOpt::WithFluxes});
  if (v.GetNBlocks() == 0) return;
  SetEdgeEMF<0, NDIM>(md, v, dt);
  SetEdgeEMF<1, NDIM>(md, v, dt);
  SetEdgeEMF<2, NDIM>(md, v, dt);
}

//----------------------------------------------------------------------------------------
//! \brief Stage update of the F_d face field, u = gam0 u + gam1 u0 - beta_dt curl(E).
//!        By Stokes, (curl E)_d is the circulation of E around the face divided by its
//!        area, [(L E_q)|_{+p} - (L E_q)|_{0} - (L E_p)|_{+q} + (L E_p)|_{0}] / A_d, with
//!        p = (d+1)%3, q = (d+2)%3 and L the edge lengths; the E_q edges are offset from
//!        the F_d face along p, the E_p edges along q. Every field touched is node layout,
//!        so this uses flat views over NN memory.
template <int d, int NDIM, typename Pack_t, typename Pack0_t>
void UpdateFaceField(MeshData<Real> *umd, const Pack_t &v, const Pack0_t &v0,
                     const Real gam0, const Real gam1, const Real beta_dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int p = (d + 1) % 3;
  constexpr int q = (d + 2) % 3;
  constexpr TE te = FaceTE(d);
  constexpr int ep_dir = static_cast<int>(EdgeTE(p)) % 3 + 1;
  constexpr int eq_dir = static_cast<int>(EdgeTE(q)) % 3 + 1;

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), umd, te, TE::NN);
  const auto dp = idx_space.GetDelta(Dir(p));
  const auto dq = idx_space.GetDelta(Dir(q));
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto pv0 = RiotLoop::make_pack_view(idx_range, v0);
        auto fp = RiotLoop::make_flux_pack_view(idx_range, v, ep_dir);
        auto fq = RiotLoop::make_flux_pack_view(idx_range, v, eq_dir);
        RiotLoop::inner(idx_range, [&](const auto f) {
          Real circ = 0.0;
          if constexpr (p < NDIM) {
            circ += Measure<EdgeTE(q)>(coords, idx_range, f + dp) *
                        fq(ccbulk::bfield(), f + dp) -
                    Measure<EdgeTE(q)>(coords, idx_range, f) * fq(ccbulk::bfield(), f);
          }
          if constexpr (q < NDIM) {
            circ -= Measure<EdgeTE(p)>(coords, idx_range, f + dq) *
                        fp(ccbulk::bfield(), f + dq) -
                    Measure<EdgeTE(p)>(coords, idx_range, f) * fp(ccbulk::bfield(), f);
          }
          // Radial faces on the axis have zero area (and circulation); leave them be
          const Real area = Measure<te>(coords, idx_range, f);
          const Real curl = (area > 0.0) ? circ / area : 0.0;
          pv(te, ccbulk::bfield(), f) = gam0 * pv(te, ccbulk::bfield(), f) +
                                        gam1 * pv0(te, ccbulk::bfield(), f) -
                                        beta_dt * curl;
        });
      });
}

template <int NDIM>
void UpdateFaceFieldsImpl(MeshData<Real> *umd, MeshData<Real> *u0md, const Real gam0,
                          const Real gam1, const Real beta_dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::PDOpt;
  auto v = riot::MakePack<ccbulk::bfield>(umd, std::vector<int>{},
                                          std::set<PDOpt>{PDOpt::WithFluxes});
  auto v0 = riot::MakePack<ccbulk::bfield>(u0md);
  if (v.GetNBlocks() == 0) return;
  UpdateFaceField<0, NDIM>(umd, v, v0, gam0, gam1, beta_dt);
  UpdateFaceField<1, NDIM>(umd, v, v0, gam0, gam1, beta_dt);
  UpdateFaceField<2, NDIM>(umd, v, v0, gam0, gam1, beta_dt);
}

//----------------------------------------------------------------------------------------
//! \brief Cell-centered value of face-field component d, interpolated linearly from the
//!        two faces to the cell centroid (the midpoint except for the radial direction of
//!        curvilinear coordinates). Components in collapsed directions are uniform across
//!        the cell and copied.
template <int d, int NDIM, typename Coords_t, typename Range_t, typename View_t,
          typename Idx_t, typename Off_t>
KOKKOS_FORCEINLINE_FUNCTION Real CellBComponent(const Coords_t &coords,
                                                const Range_t &idx_range, const View_t &pv,
                                                const Idx_t c, const Off_t dd) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr TE te = FaceTE(d);
  if constexpr (d < NDIM) {
    const Real blo = pv(te, ccbulk::bfield(), c);
    const Real bhi = pv(te, ccbulk::bfield(), c + dd);
    if constexpr (d == 0 && !parthenon::IsCoord<parthenon::UniformCartesian>()) {
      const auto [k, j, i] = idx_range.GetKJI(c);
      const Real xlo = coords.template Xf<X1DIR>(i);
      const Real xhi = coords.template Xf<X1DIR>(i + 1);
      const Real xc = coords.template Xc<X1DIR>(i);
      return ((xhi - xc) * blo + (xc - xlo) * bhi) / (xhi - xlo);
    } else {
      return 0.5 * (blo + bhi);
    }
  } else {
    return pv(te, ccbulk::bfield(), c);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Net magnetic flux out of cell c through its two F_d faces
template <int d, typename Coords_t, typename Range_t, typename View_t, typename Idx_t,
          typename Off_t>
KOKKOS_FORCEINLINE_FUNCTION Real NetFlux(const Coords_t &coords, const Range_t &idx_range,
                                         const View_t &pv, const Idx_t c, const Off_t dd) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr TE te = FaceTE(d);
  return Measure<te>(coords, idx_range, c + dd) * pv(te, ccbulk::bfield(), c + dd) -
         Measure<te>(coords, idx_range, c) * pv(te, ccbulk::bfield(), c);
}

//----------------------------------------------------------------------------------------
//! \brief Discrete divergence of the face field in cell c, the net flux through the cell
//!        faces divided by the cell volume; d1..d3 are the unit offsets.
template <int NDIM, typename Coords_t, typename Range_t, typename View_t, typename Idx_t,
          typename Off_t>
KOKKOS_FORCEINLINE_FUNCTION Real DivB(const Coords_t &coords, const Range_t &idx_range,
                                      const View_t &pv, const Idx_t c, const Off_t d1,
                                      const Off_t d2, const Off_t d3) {
  Real flux = NetFlux<0>(coords, idx_range, pv, c, d1);
  if constexpr (NDIM > 1) flux += NetFlux<1>(coords, idx_range, pv, c, d2);
  if constexpr (NDIM > 2) flux += NetFlux<2>(coords, idx_range, pv, c, d3);
  return flux / Measure<TE::CC>(coords, idx_range, c);
}

template <int NDIM>
void CopyFaceBToAlignedImpl(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bface_aligned>(md);
  if (v.GetNBlocks() == 0) return;
  using lt = RiotUtils::LoopType<LoopConstraint::DifferentMemSpaces>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md, TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto c) {
          pv(TE::F1, ccbulk::bface_aligned(), c) = pv(TE::F1, ccbulk::bfield(), c);
          if constexpr (NDIM > 1) {
            pv(TE::F2, ccbulk::bface_aligned(), c) = pv(TE::F2, ccbulk::bfield(), c);
          }
          if constexpr (NDIM > 2) {
            pv(TE::F3, ccbulk::bface_aligned(), c) = pv(TE::F3, ccbulk::bfield(), c);
          }
        });
      });
}

template <int NDIM>
void FaceToCellBImpl(MeshData<Real> *md, IndexDomain domain) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::bcc>(md);
  if (v.GetNBlocks() == 0) return;
  using lt = RiotUtils::LoopType<LoopConstraint::DifferentMemSpaces>;
  auto idx_space = lt::GetIndexSpace(domain, 0, v.GetNBlocks(), md, TE::CC);
  const auto d1 = idx_space.GetDelta(X1DIR);
  const auto d2 = idx_space.GetDelta(X2DIR);
  const auto d3 = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto c) {
          pv(ccbulk::bcc(0), c) = CellBComponent<0, NDIM>(coords, idx_range, pv, c, d1);
          pv(ccbulk::bcc(1), c) = CellBComponent<1, NDIM>(coords, idx_range, pv, c, d2);
          pv(ccbulk::bcc(2), c) = CellBComponent<2, NDIM>(coords, idx_range, pv, c, d3);
        });
      });
}

template <int NDIM>
void CalculateDivBImpl(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bfield, ccbulk::divb>(md);
  if (v.GetNBlocks() == 0) return;
  using lt = RiotUtils::LoopType<LoopConstraint::DifferentMemSpaces>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC);
  const auto d1 = idx_space.GetDelta(X1DIR);
  const auto d2 = idx_space.GetDelta(X2DIR);
  const auto d3 = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto c) {
          pv(ccbulk::divb(), c) = DivB<NDIM>(coords, idx_range, pv, c, d1, d2, d3);
        });
      });
}

template <int NDIM>
Real MaxDivBImpl(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bfield>(md);
  if (v.GetNBlocks() == 0) return 0.0;
  // Only node-layout bfield is read, so flat views over NN memory
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC, TE::NN);
  const auto d1 = idx_space.GetDelta(X1DIR);
  const auto d2 = idx_space.GetDelta(X2DIR);
  const auto d3 = idx_space.GetDelta(X3DIR);
  return RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner_reduce(idx_range, [&](const auto c, Real &lmax) {
          Real dxmin = CellWidth<0>(coords, idx_range, c);
          if constexpr (NDIM > 1) dxmin = std::min(dxmin, CellWidth<1>(coords, idx_range, c));
          if constexpr (NDIM > 2) dxmin = std::min(dxmin, CellWidth<2>(coords, idx_range, c));
          lmax = std::max(lmax,
                          std::abs(DivB<NDIM>(coords, idx_range, pv, c, d1, d2, d3)) * dxmin);
        });
      });
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyFaceBToAligned
//! \brief Copy the normal face field of each active direction into the cell-aligned face
//!        scratch ccbulk::bface_aligned, so that the flux kernel can read it through the
//!        flat pack views. Covers every cell-memory face index (entire domain).
void CopyFaceBToAligned(MeshData<Real> *md) {
  switch (md->GetParentPointer()->ndim) {
  case 1:
    return CopyFaceBToAlignedImpl<1>(md);
  case 2:
    return CopyFaceBToAlignedImpl<2>(md);
  default:
    return CopyFaceBToAlignedImpl<3>(md);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Hydro::CalculateCornerEMF
//! \brief Fill the edge fluxes of bfield with the edge EMFs E (not -E), computed from the
//!        face EMFs of the most recent CalculateFluxes call.
TaskStatus CalculateCornerEMF(MeshData<Real> *md, const Real dt) {
  if (md->NumBlocks() == 0) return TaskStatus::complete;
  switch (md->GetParentPointer()->ndim) {
  case 1:
    CalculateCornerEMFImpl<1>(md, dt);
    break;
  case 2:
    CalculateCornerEMFImpl<2>(md, dt);
    break;
  default:
    CalculateCornerEMFImpl<3>(md, dt);
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
  if (umd->NumBlocks() == 0) return TaskStatus::complete;
  switch (umd->GetParentPointer()->ndim) {
  case 1:
    UpdateFaceFieldsImpl<1>(umd, u0md, gam0, gam1, beta_dt);
    break;
  case 2:
    UpdateFaceFieldsImpl<2>(umd, u0md, gam0, gam1, beta_dt);
    break;
  default:
    UpdateFaceFieldsImpl<3>(umd, u0md, gam0, gam1, beta_dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::FaceToCellB
//! \brief Average the face field to cell centers (ccbulk::bcc) over domain. Components in
//!        collapsed directions are uniform across the cell and copied.
void FaceToCellB(MeshData<Real> *md, IndexDomain domain) {
  switch (md->GetParentPointer()->ndim) {
  case 1:
    return FaceToCellBImpl<1>(md, domain);
  case 2:
    return FaceToCellBImpl<2>(md, domain);
  default:
    return FaceToCellBImpl<3>(md, domain);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateDivB
//! \brief Discrete divergence of the face field in interior cells (ccbulk::divb).
void CalculateDivB(MeshData<Real> *md) {
  switch (md->GetParentPointer()->ndim) {
  case 1:
    return CalculateDivBImpl<1>(md);
  case 2:
    return CalculateDivBImpl<2>(md);
  default:
    return CalculateDivBImpl<3>(md);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::total_magnetic_energy
//! \brief History output: volume-integrated magnetic energy B^2/2 (cell-centered field).
Real total_magnetic_energy(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::bcc>(md);
  if (v.GetNBlocks() == 0) return 0.0;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space = rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC);
  return RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner_reduce(idx_range, [&](const auto c, Real &leb) {
          const auto [k, j, i] = idx_range.GetKJI(c);
          leb += 0.5 *
                 (SQR(pv(ccbulk::bcc(0), c)) + SQR(pv(ccbulk::bcc(1), c)) +
                  SQR(pv(ccbulk::bcc(2), c))) *
                 coords.CellVolume(k, j, i);
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::max_divb
//! \brief History output: max |div B| dx_min over interior cells (dx_min the smallest
//!        physical cell width of each cell), recomputed from the
//!        face field (so it does not depend on when ccbulk::divb was last filled).
Real max_divb(MeshData<Real> *md) {
  switch (md->GetParentPointer()->ndim) {
  case 1:
    return MaxDivBImpl<1>(md);
  case 2:
    return MaxDivBImpl<2>(md);
  default:
    return MaxDivBImpl<3>(md);
  }
}

} // namespace Hydro
