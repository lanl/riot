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
#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
// This file was made in part with generative AI.

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "hydro/advection.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/concepts_lite.hpp>

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;
using parthenon::ScratchPad1D;
using parthenon::ScratchPad2D;
using parthenon::ScratchPad3D;
using parthenon::ScratchPad4D;

#include "reconstruction/reconstruction.hpp"
#include "riemann.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Hydro {
//----------------------------------------------------------------------------------------
//! \brief  Bulk quantities reconstructed directly to faces (set each pass by ReconCells).
using set_bulk_recon_types =
    RiotLoop::IndexedVarTypeList<cell_variables::cell_averaged::bulk::pressure,
                                 cell_variables::cell_averaged::bulk::bulk_modulus,
                                 cell_variables::cell_averaged::bulk::velocity>;

//! \brief  Bulk quantities accumulated (+=) on faces by summing over materials; these are
//!         the scratch slots that must be zeroed before the material loop.
using sum_bulk_recon_types =
    RiotLoop::IndexedVarTypeList<cell_variables::cell_averaged::bulk::rho,
                                 cell_variables::cell_averaged::bulk::internal_energy>;

//! \brief  Per-material quantities reconstructed to faces.
using mat_recon_types =
    RiotLoop::IndexedVarTypeList<cell_variables::cell_averaged::mat::volume_fraction,
                                 cell_variables::material_averaged::rho,
                                 cell_variables::cell_averaged::mat::internal_energy>;

//! \brief  Strength bulk quantity reconstructed directly to faces (set, like pressure);
//!         used only on the strength ("strong") solver path. The bulk deviatoric stress
//!         is NOT here -- it is accumulated (+=) by summing vfrac-weighted per-material
//!         stresses, so it uses a separate 5-component scratch (see StrengthFluxes),
//!         mirroring the set/sum split of the pure-hydro bulk quantities.
using set_strength_bulk_recon_types =
    RiotLoop::IndexedVarTypeList<cell_variables::cell_averaged::bulk::shear_modulus>;

//! \brief  Per-(strong-)material quantity reconstructed to faces on the strength path.
using strength_mat_recon_types =
    RiotLoop::IndexedVarTypeList<cell_variables::material_averaged::deviatoric_stress>;

static constexpr Real THETA = 1.99;

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ReconCells
//! \brief Reconstruct variable(s) VARS on the minus and plus faces of cells in the
//!        direction defined by the offset delta in halo_range. The reconstruction
//!        *input* stencil width (1 cell for PLM, 2 for the 5-point methods) is
//!        independent of the loop bounds / halo, which are unchanged; the wider stencils
//!        simply read further into the pack's ghost zones (sized by stencil_width /
//!        nghost). plus/minus receive the l/r face states respectively (l = plus, r =
//!        minus), matching the (..., l, r) convention of the reconstruction routines.
template <typename... VARS, typename Pack_t, typename IdxRange_t, typename delta_t,
          typename scratch_t>
KOKKOS_INLINE_FUNCTION void
ReconCells(const Pack_t &v, const IdxRange_t &halo_range, delta_t delta, scratch_t &minus,
           scratch_t &plus, const RiotReconstruction::Type t, const int mat_idx = 0,
           const Real slope_limit = THETA) {
  namespace RR = RiotReconstruction;
  // Manual loop unswitching: pick the reconstruction method ONCE (this switch), then
  // run a branch-free inner loop for it. The per-cell stencil `op` is applied over
  // every VAR/component. Keeping the switch outside RiotLoop::inner is essential --
  // a switch inside the innermost loop would block vectorization on CPU and be a
  // per-cell branch on GPU.
  auto for_all = [&](auto op) {
    (
        [&] {                                      // Unroll over all variables in VARS...
          for (int c = 0; c < VARS::size(); c++) { // Loop over components of variable
            auto var = VARS(c);
            RiotLoop::inner(halo_range, [&](auto kji) {
              op(var, kji, plus(var, mat_idx, kji), minus(var, mat_idx, kji));
            });
          }
        }(),
        ...);
  };

  switch (t) {
  case RR::Type::CONSTANT:
    for_all([&](auto var, auto kji, Real &pl, Real &mn) {
      RR::PiecewiseConstant(v(var, kji), pl, mn);
    });
    break;
  case RR::Type::PLM:
    for_all([&](auto var, auto kji, Real &pl, Real &mn) {
      RR::PiecewiseLinear(v(var, kji - delta), v(var, kji), v(var, kji + delta), pl, mn,
                          slope_limit);
    });
    break;
  case RR::Type::PPM4:
    for_all([&](auto var, auto kji, Real &pl, Real &mn) {
      RR::PPM4(v(var, kji - 2 * delta), v(var, kji - delta), v(var, kji),
               v(var, kji + delta), v(var, kji + 2 * delta), pl, mn);
    });
    break;
  case RR::Type::WENO5:
    for_all([&](auto var, auto kji, Real &pl, Real &mn) {
      RR::WENO5(v(var, kji - 2 * delta), v(var, kji - delta), v(var, kji),
                v(var, kji + delta), v(var, kji + 2 * delta), pl, mn);
    });
    break;
  case RR::Type::MP5:
    for_all([&](auto var, auto kji, Real &pl, Real &mn) {
      RR::MP5(v(var, kji - 2 * delta), v(var, kji - delta), v(var, kji),
              v(var, kji + delta), v(var, kji + 2 * delta), pl, mn);
    });
    break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ReconVar
//! \brief Reconstruct a *single* anonymous variable (viewed through a var_view_t) onto
//!        the minus and plus faces, writing plain per-point scratch. The single-variable
//!        analog of ReconCells, for passive/advected scalars that are not typed pack
//!        members. Same reconstruction methods and (l=plus, r=minus) face convention; the
//!        manual unswitching (switch outside RiotLoop::inner) mirrors ReconCells so CPU
//!        vectorization is unaffected.
template <typename VarView_t, typename IdxRange_t, typename delta_t, typename scratch_t>
KOKKOS_INLINE_FUNCTION void ReconVar(const VarView_t &q, const IdxRange_t &halo_range,
                                     delta_t delta, scratch_t &minus, scratch_t &plus,
                                     const RiotReconstruction::Type t,
                                     const Real slope_limit = THETA) {
  namespace RR = RiotReconstruction;
  auto for_all = [&](auto op) {
    RiotLoop::inner(halo_range, [&](auto kji) { op(kji, plus(kji), minus(kji)); });
  };

  switch (t) {
  case RR::Type::CONSTANT:
    for_all([&](auto kji, Real &pl, Real &mn) { RR::PiecewiseConstant(q(kji), pl, mn); });
    break;
  case RR::Type::PLM:
    for_all([&](auto kji, Real &pl, Real &mn) {
      RR::PiecewiseLinear(q(kji - delta), q(kji), q(kji + delta), pl, mn, slope_limit);
    });
    break;
  case RR::Type::PPM4:
    for_all([&](auto kji, Real &pl, Real &mn) {
      RR::PPM4(q(kji - 2 * delta), q(kji - delta), q(kji), q(kji + delta),
               q(kji + 2 * delta), pl, mn);
    });
    break;
  case RR::Type::WENO5:
    for_all([&](auto kji, Real &pl, Real &mn) {
      RR::WENO5(q(kji - 2 * delta), q(kji - delta), q(kji), q(kji + delta),
                q(kji + 2 * delta), pl, mn);
    });
    break;
  case RR::Type::MP5:
    for_all([&](auto kji, Real &pl, Real &mn) {
      RR::MP5(q(kji - 2 * delta), q(kji - delta), q(kji), q(kji + delta),
              q(kji + 2 * delta), pl, mn);
    });
    break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real Hydro::CalculateStrainRate
//! \brief
template <typename PACK>
void CalculateStrainRate(MeshData<Real> *state, const PACK &vb) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  auto pm = state->GetParentPointer();

  const int ndim = pm->ndim;
  const int dj = (ndim > 1);
  const int dk = (ndim > 2);

  using lt = RiotUtils::LoopType<>;
  const int nhalo = 1;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, nhalo, vb.GetNBlocks(), state,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = vb.GetCoordinates(b);
        // assume these are constants
        const Real dx = coords.template Dx<X1DIR>();
        const Real dy = coords.template Dx<X2DIR>();
        const Real dz = coords.template Dx<X3DIR>();

        auto pvb = RiotLoop::make_pack_view(idx_range, vb);

        // first get the velocity gradient
        RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
          const Real ivol = 1.0 / coords.CellVolume(k, j, i);
          const Real area1 = coords.template FaceArea<X1DIR>(k, j, i);
          const Real area1p = coords.template FaceArea<X1DIR>(k, j, i + 1);
          const Real area2_lo = coords.template FaceArea<X2DIR>(k, j, i);
          const Real area2_hi = coords.template FaceArea<X2DIR>(k, j + dj, i);
          const Real area3_lo = coords.template FaceArea<X3DIR>(k, j, i);
          const Real area3_hi = coords.template FaceArea<X3DIR>(k + dk, j, i);
          const Real vdiv = (1.0 / 3.0) *
                            ((pvb(ccbulk::face_velocity(0), k, j, i + 1) * area1p -
                              pvb(ccbulk::face_velocity(0), k, j, i) * area1) +
                             (pvb(ccbulk::face_velocity(4), k, j + dj, i) * area2_hi -
                              pvb(ccbulk::face_velocity(4), k, j, i) * area2_lo) +
                             (pvb(ccbulk::face_velocity(8), k + dk, j, i) * area3_hi -
                              pvb(ccbulk::face_velocity(8), k, j, i) * area3_lo)) *
                            ivol;

          // Velocity gradient
          const Real dvxx = (pvb(ccbulk::face_velocity(0), k, j, i + 1) -
                             pvb(ccbulk::face_velocity(0), k, j, i)) /
                            dx;
          const Real dvyx = (pvb(ccbulk::face_velocity(1), k, j, i + 1) -
                             pvb(ccbulk::face_velocity(1), k, j, i)) /
                            dx;
          const Real dvzx = (pvb(ccbulk::face_velocity(2), k, j, i + 1) -
                             pvb(ccbulk::face_velocity(2), k, j, i)) /
                            dx;
          const Real dvxy = (pvb(ccbulk::face_velocity(3), k, j + dj, i) -
                             pvb(ccbulk::face_velocity(3), k, j, i)) /
                            dy;
          Real dvyy = (pvb(ccbulk::face_velocity(4), k, j + dj, i) -
                       pvb(ccbulk::face_velocity(4), k, j, i)) /
                      dy;
          const Real dvzy = (pvb(ccbulk::face_velocity(5), k, j + dj, i) -
                             pvb(ccbulk::face_velocity(5), k, j, i)) /
                            dy;
          const Real dvxz = (pvb(ccbulk::face_velocity(6), k + dk, j, i) -
                             pvb(ccbulk::face_velocity(6), k, j, i)) /
                            dz;
          const Real dvyz = (pvb(ccbulk::face_velocity(7), k + dk, j, i) -
                             pvb(ccbulk::face_velocity(7), k, j, i)) /
                            dz;
          Real dvzz = (pvb(ccbulk::face_velocity(8), k + dk, j, i) -
                       pvb(ccbulk::face_velocity(8), k, j, i)) /
                      dz;
          if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>() ||
                        parthenon::IsCoord<parthenon::UniformCylindrical>()) {

            const Real r0 = coords.template Xf<X1DIR>(i);
            const Real r1 = coords.template Xf<X1DIR>(i + 1);

            const Real avg_vr =
                (2.0 * r0 * r1 *
                     (pvb(ccbulk::face_velocity(0), k, j, i) +
                      pvb(ccbulk::face_velocity(0), k, j, i + 1)) +
                 SQR(r0) * (3.0 * pvb(ccbulk::face_velocity(0), k, j, i) +
                            pvb(ccbulk::face_velocity(0), k, j, i + 1)) +
                 SQR(r1) * (pvb(ccbulk::face_velocity(0), k, j, i) +
                            3.0 * pvb(ccbulk::face_velocity(0), k, j, i + 1))) /
                (4.0 * (SQR(r0) + r0 * r1 + SQR(r1)));

            const Real vr_over_r = avg_vr / coords.template Xc<X1DIR>(i);

            if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
              // theta-theta and phi-phi
              dvyy = vr_over_r;
              dvzz = vr_over_r;
            } else {
              // cylindrical phi-phi
              dvzz = vr_over_r;
            }
          }

          pvb(ccbulk::strain_rate(0), k, j, i) = dvxx - vdiv;
          pvb(ccbulk::strain_rate(1), k, j, i) = 0.5 * (dvxy + dvyx);
          pvb(ccbulk::strain_rate(2), k, j, i) = 0.5 * (dvxz + dvzx);
          pvb(ccbulk::strain_rate(3), k, j, i) = dvyy - vdiv;
          pvb(ccbulk::strain_rate(4), k, j, i) = 0.5 * (dvyz + dvzy);
          pvb(ccbulk::strain_rate(5), k, j, i) = dvzz - vdiv;
        });
      });
}

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *mat_pkg);
TaskStatus CalculateGeometricSource(MeshData<Real> *state, MeshData<Real> *src);
TaskStatus CalculateFluxes(MeshData<Real> *rc);
TaskStatus GuessCellVolumeFractions(MeshData<Real> *rc);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
TaskStatus CalculateMaxSignalSpeed(MeshData<Real> *state);
Real EstimateTimestepMesh(MeshData<Real> *rc);
PrimFluxPack MakeAdvectionPack(MeshData<Real> *md);
Real total_kinetic_energy(MeshData<Real> *md);
void CalculateStrainRate(MeshData<Real> *state);
} // namespace Hydro
#endif // HYDRO_HYDRO_HPP_
