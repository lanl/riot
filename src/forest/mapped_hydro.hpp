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
#ifndef FOREST_MAPPED_HYDRO_HPP_
#define FOREST_MAPPED_HYDRO_HPP_
// This file was made in part with generative AI.
//
// Hydro on mapped (forest) meshes. The hooks here are called from the hydro kernels and
// are no-ops unless the coordinates are mapped. On mapped meshes the Riemann solvers
// work in the (normal, tangent) frame of each face:
//   - ReconVelocityToFaceFrame leaves the reconstructed face velocities (the meridional
//     ones in (R, z), all three in 3D) in the frame of their face,
//   - the solvers then run unchanged (slot DIR-1 is the face normal),
//   - FluxesToGlobalFrame rotates the momentum flux and stored face velocity back to the
//     global basis.
// GetVdiff is the mapped version of the low-Mach transverse velocity differences.

#include <algorithm>
#include <limits>
#include <type_traits>

#include <coordinates/coordinates.hpp>
#include <interface/mesh_data.hpp>
#include <mesh/domain.hpp>

#include "forest/mapped_geometry.hpp"
#include "hydro/hydro.hpp"
#include "reconstruction/reconstruction.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace RiotForest {

using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;

//----------------------------------------------------------------------------------------
//! \brief Host-side settings of the mapped flux kernels: whether velocities are
//!        reconstructed in the face frame (<hydro>/mapped_velocity_frame) and the
//!        interior bounds of the blocks (for GetVdiff). Empty unless the coordinates are
//!        mapped.
struct FluxContext {
  bool face_frame = false;
  IndexRange ib, jb, kb;
};

inline FluxContext MakeFluxContext(MeshData<Real> *md) {
  FluxContext c;
  if constexpr (parthenon::IsMappedCoord()) {
    auto pm = md->GetParentPointer();
    c.face_frame = pm->packages.Get("riot")->Param<bool>("mapped_face_frame");
    c.ib = md->GetBoundsI(IndexDomain::interior);
    c.jb = md->GetBoundsJ(IndexDomain::interior);
    c.kb = md->GetBoundsK(IndexDomain::interior);
  }
  return c;
}

//----------------------------------------------------------------------------------------
//! \brief Velocity reconstruction in the frame of the face being reconstructed to: the
//!        stencil's meridional velocities are rotated into that face's (normal, tangent)
//!        frame and reconstructed. The frame is the mesh's own, i.e. spherical
//!        (e_r, e_theta) on polar patches and cylindrical (R, z) on rectilinear ones. A
//!        uniform velocity is reconstructed exactly (one rotation per stencil), and the
//!        frame depends only on the face, so the two blocks sharing a face compute
//!        identical states. Fills all velocity components (velocity(2) is not rotated in
//!        (R, z)).
template <parthenon::CoordinateDirection DIR, typename PackView, typename IdxRange,
          typename delta_t, typename Coords, typename scratch_t>
KOKKOS_INLINE_FUNCTION void
ReconVelocityFaceFrame(const PackView &pv, const IdxRange &halo_range,
                       const delta_t delta, const Coords &coords, scratch_t &minus,
                       scratch_t &plus, const RiotReconstruction::Type t) {
  namespace RR = RiotReconstruction;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int di = (DIR == X1DIR), dj = (DIR == X2DIR), dk = (DIR == X3DIR);
  // op(q, l, r) reconstructs the stencil q[0..2W] (centered on q[W]) to its l/r states
  auto run = [&](auto width, auto op) {
    constexpr int W = decltype(width)::value;
    RiotLoop::inner(halo_range, [&](const auto kji) {
      const auto [k, j, i] = halo_range.GetKJI(kji);
      // the minus state lives on the lower face of the cell, the plus state on the upper
      const FaceFrame fm = MakeFaceFrame<DIR>(coords, k, j, i);
      const FaceFrame fp = MakeFaceFrame<DIR>(coords, k + dk, j + dj, i + di);
      // stencils of the velocity in the frames of the minus and plus faces
      Real qm[3][2 * W + 1], qp[3][2 * W + 1];
      for (int s = -W; s <= W; ++s) {
        Real vm[3], vp[3];
        for (int c = 0; c < 3; ++c)
          vm[c] = vp[c] = pv(ccbulk::velocity(c), kji + s * delta);
        fm.ToFace(vm);
        fp.ToFace(vp);
        for (int c = 0; c < 3; ++c) {
          qm[c][s + W] = vm[c];
          qp[c][s + W] = vp[c];
        }
      }
      Real unused;
      for (int c = 0; c < FrameDims(); ++c) {
        op(qp[c], plus(ccbulk::velocity(c), 0, kji), unused);
        op(qm[c], unused, minus(ccbulk::velocity(c), 0, kji));
      }
      // components that are not rotated (qm = qp)
      for (int c = FrameDims(); c < 3; ++c)
        op(qm[c], plus(ccbulk::velocity(c), 0, kji), minus(ccbulk::velocity(c), 0, kji));
    });
  };
  using W0 = std::integral_constant<int, 0>;
  using W1 = std::integral_constant<int, 1>;
  using W2 = std::integral_constant<int, 2>;
  switch (t) {
  case RR::Type::CONSTANT:
    run(W0(), [](const Real *q, Real &l, Real &r) { RR::PiecewiseConstant(q[0], l, r); });
    break;
  case RR::Type::PLM:
    run(W1(), [](const Real *q, Real &l, Real &r) {
      RR::PiecewiseLinear(q[0], q[1], q[2], l, r, Hydro::THETA);
    });
    break;
  case RR::Type::PPM4:
    run(W2(), [](const Real *q, Real &l, Real &r) {
      RR::PPM4(q[0], q[1], q[2], q[3], q[4], l, r);
    });
    break;
  case RR::Type::WENO5:
    run(W2(), [](const Real *q, Real &l, Real &r) {
      RR::WENO5(q[0], q[1], q[2], q[3], q[4], l, r);
    });
    break;
  case RR::Type::MP5:
    run(W2(), [](const Real *q, Real &l, Real &r) {
      RR::MP5(q[0], q[1], q[2], q[3], q[4], l, r);
    });
    break;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Hook after the (global-frame) velocity reconstruction. On mapped meshes it
//!        leaves the rotated face velocities in the frame of their face, either by
//!        reconstructing them in that frame (face_frame) or by rotating the
//!        global-frame states.
template <parthenon::CoordinateDirection DIR, typename PackView, typename IdxRange,
          typename delta_t, typename Coords, typename scratch_t>
KOKKOS_INLINE_FUNCTION void
ReconVelocityToFaceFrame(const PackView &pv, const IdxRange &halo_range,
                         const delta_t delta, const Coords &coords,
                         const FluxContext &ctx, scratch_t &minus, scratch_t &plus,
                         const RiotReconstruction::Type t) {
  if constexpr (IsMappedDir<DIR>()) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    halo_range.TeamBarrier();
    if (ctx.face_frame) {
      ReconVelocityFaceFrame<DIR>(pv, halo_range, delta, coords, minus, plus, t);
      return;
    }
    constexpr int di = (DIR == X1DIR), dj = (DIR == X2DIR), dk = (DIR == X3DIR);
    constexpr int nv = FrameDims();
    RiotLoop::inner(halo_range, [&](const auto kji) {
      const auto [k, j, i] = halo_range.GetKJI(kji);
      Real w[3];
      for (int c = 0; c < nv; ++c)
        w[c] = minus(ccbulk::velocity(c), 0, kji);
      MakeFaceFrame<DIR>(coords, k, j, i).ToFace(w);
      for (int c = 0; c < nv; ++c)
        minus(ccbulk::velocity(c), 0, kji) = w[c];
      for (int c = 0; c < nv; ++c)
        w[c] = plus(ccbulk::velocity(c), 0, kji);
      MakeFaceFrame<DIR>(coords, k + dk, j + dj, i + di).ToFace(w);
      for (int c = 0; c < nv; ++c)
        plus(ccbulk::velocity(c), 0, kji) = w[c];
    });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Hook after the bulk Riemann fluxes: rotates the momentum flux and the stored
//!        face velocity of each face from its frame back to the global basis.
template <parthenon::CoordinateDirection DIR, typename Pack_t, typename IdxRange,
          typename Coords>
KOKKOS_INLINE_FUNCTION void
FluxesToGlobalFrame(const Pack_t &v, const IdxRange &idx_range, const Coords &coords,
                    const bool store_vf) {
  if constexpr (IsMappedDir<DIR>()) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    constexpr int base = 3 * (static_cast<int>(DIR) - 1);
    constexpr int nv = FrameDims();
    idx_range.TeamBarrier();
    auto fv = RiotLoop::make_flux_pack_view(idx_range, v, DIR);
    auto pv = RiotLoop::make_pack_view(idx_range, v);
    RiotLoop::inner(idx_range, [&](const auto kji) {
      const auto [k, j, i] = idx_range.GetKJI(kji);
      const FaceFrame frame = MakeFaceFrame<DIR>(coords, k, j, i);
      Real w[3];
      for (int c = 0; c < nv; ++c)
        w[c] = fv(ccbulk::momentum(c), kji);
      frame.FromFace(w);
      for (int c = 0; c < nv; ++c)
        fv(ccbulk::momentum(c), kji) = w[c];
      if (store_vf) {
        for (int c = 0; c < nv; ++c)
          w[c] = pv(ccbulk::face_velocity(base + c), kji);
        frame.FromFace(w);
        for (int c = 0; c < nv; ++c)
          pv(ccbulk::face_velocity(base + c), kji) = w[c];
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Mapped-mesh version of Hydro::GetVdiff, the transverse velocity differences
//!        (dvn, dvt) of the low-Mach solvers. The velocities are taken in the frame of
//!        the face. The two blocks sharing a block boundary face must compute the same
//!        flux, but at the ends of the face row the transverse stencil reads a corner
//!        ghost that each block fills differently (or not at all where fewer than four
//!        trees meet at a vertex), so both drop the transverse differences on that side.
template <parthenon::CoordinateDirection DIR, typename Pack_t, typename IdxRange,
          typename Basis, typename Coords, typename Scratch>
KOKKOS_INLINE_FUNCTION void GetVdiff(const Pack_t &v, const IdxRange &idx_range,
                                     const Basis &basis, const Coords &coords,
                                     const FluxContext &ctx, Scratch &dvn, Scratch &dvt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  const auto dn = basis.normal();
  RiotLoop::inner(idx_range, [&](const auto kji) {
    const auto [k, j, i] = idx_range.GetKJI(kji);
    const FaceFrame frame = MakeFaceFrame<DIR>(coords, k, j, i);
    // velocity component c in the frame of this face
    auto V = [&](int c, auto idx) -> Real {
      if (c < FrameDims()) {
        Real w[3] = {pv(ccbulk::velocity(0), idx), pv(ccbulk::velocity(1), idx),
                     IsMapped3D() ? pv(ccbulk::velocity(2), idx) : 0.0};
        frame.ToFace(w);
        return w[c];
      }
      return pv(ccbulk::velocity(c), idx);
    };
    bool use_lo[3] = {true, true, true}, use_hi[3] = {true, true, true};
    const int n[3] = {i, j, k};
    const IndexRange bnd[3] = {ctx.ib, ctx.jb, ctx.kb};
    constexpr int nc = Basis::nc;
    if (n[nc] == bnd[nc].s || n[nc] == bnd[nc].e + 1) {
      for (const int c : {Basis::tac, Basis::tbc}) {
        use_lo[c] = n[c] > bnd[c].s;
        use_hi[c] = n[c] < bnd[c].e;
      }
    }
    constexpr Real BIG = std::numeric_limits<Real>::max();
    // Min-limited transverse gradient of velocity component c along offset t (normal dn).
    auto trans = [&](int c, auto t) {
      Real d = BIG;
      if (use_lo[c])
        d = std::min(
            d, std::min(V(c, kji - dn) - V(c, kji - dn - t), V(c, kji) - V(c, kji - t)));
      if (use_hi[c])
        d = std::min(
            d, std::min(V(c, kji + t - dn) - V(c, kji - dn), V(c, kji + t) - V(c, kji)));
      return d;
    };
    dvn(kji) = V(Basis::nc, kji) - V(Basis::nc, kji - dn);

    Real t = BIG;
    if (!RiotUtils::IsZeroOffset(basis.trans_a()))
      t = std::min(t, trans(Basis::tac, basis.trans_a()));
    if (!RiotUtils::IsZeroOffset(basis.trans_b()))
      t = std::min(t, trans(Basis::tbc, basis.trans_b()));
    dvt(kji) = (t == BIG) ? 0.0 : t;
  });
}

//----------------------------------------------------------------------------------------
//! \brief Hook in the hydro timestep: on mapped meshes, whose logical directions are not
//!        aligned with the global basis, the signal velocity in direction d is the
//!        velocity normal to the lower face in that direction.
template <typename Coords, typename PackView, typename Index>
KOKKOS_FORCEINLINE_FUNCTION void
FaceNormalVelocity(const Coords &coords, const PackView &pv, const Index &idx,
                   const int d, const int k, const int j, const int i, Real &vphys) {
  if constexpr (parthenon::IsMappedCoord()) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    if (d < FrameDims()) {
      Real n1, n2, n3;
      if (d == 0) {
        coords.template FaceNormal<X1DIR>(k, j, i, n1, n2, n3);
      } else if (d == 1) {
        coords.template FaceNormal<X2DIR>(k, j, i, n1, n2, n3);
      } else {
        coords.template FaceNormal<X3DIR>(k, j, i, n1, n2, n3);
      }
      vphys = n1 * pv(ccbulk::velocity(0), idx) + n2 * pv(ccbulk::velocity(1), idx);
      if constexpr (IsMapped3D()) vphys += n3 * pv(ccbulk::velocity(2), idx);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief The velocity normal to the lower DIR face of cell (k, j, i) from the stored
//!        face velocity at idx, for divergences by the divergence theorem: component
//!        DIR-1 of it, except on mapped meshes, where FluxesToGlobalFrame has rotated it
//!        to the global basis.
template <parthenon::CoordinateDirection DIR, typename Coords, typename PackView,
          typename Index>
KOKKOS_FORCEINLINE_FUNCTION Real StoredFaceNormalVelocity(const Coords &coords,
                                                          const PackView &pv,
                                                          const Index &idx, const int k,
                                                          const int j, const int i) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr int base = 3 * (static_cast<int>(DIR) - 1);
  if constexpr (IsMappedDir<DIR>()) {
    Real n[3];
    coords.template FaceNormal<DIR>(k, j, i, n[0], n[1], n[2]);
    Real vn = 0.0;
    for (int c = 0; c < FrameDims(); ++c)
      vn += n[c] * pv(ccbulk::face_velocity(base + c), idx);
    return vn;
  } else {
    return pv(ccbulk::face_velocity(base + static_cast<int>(DIR) - 1), idx);
  }
}

} // namespace RiotForest

#endif // FOREST_MAPPED_HYDRO_HPP_
