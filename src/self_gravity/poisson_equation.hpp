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
#ifndef SELF_GRAVITY_POISSON_EQUATION_HPP_
#define SELF_GRAVITY_POISSON_EQUATION_HPP_
// This file was made in part with generative AI.
//! \file poisson_equation.hpp
//! \brief Discrete Poisson operator for self-gravity (following the poisson_gmg example
//! in Parthenon).

#include <memory>

#include <parthenon/package.hpp>

#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace SelfGravity {

// This class implements methods for calculating A.x = y and returning the diagonal of A,
// where A is the matrix representing the discretized Poisson operator, Laplacian(phi), on
// the grid.  Only uniform Cartesian coordinates are supported.
template <class var_t>
class PoissonEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  // Add tasks to calculate the result of the matrix A (which is implicitly defined by
  // this class) being applied to md_in and stored in md_out
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_in);
    if (!(md_mat->grid.type() == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
      auto send_flxcor =
          tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
      auto recv_flxcor =
          tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
  }

  // Calculate the diagonal of the matrix A and store it in var_t.  For a uniform grid or
  // when flux correction is ignored, this diagonal calculation is exact. Exactness is
  // (probably) not required since it is just used in Jacobi iterations.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    using TE = parthenon::TopologicalElement;
    const int ndim = md_mat->GetMeshPointer()->ndim;

    static auto desc = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack = desc.GetPack(md_diag.get());
    if (pack.GetNBlocks() == 0) return TaskStatus::complete;

    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(),
                                       md_diag.get(), TE::CC);
    // Combines per-direction face areas and center-to-center distances across a
    // neighbor stencil, so this uses (k, j, i).
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            Real diag_elem = coords.template FaceArea<X1DIR>(k, j, i) /
                                 coords.template Dxc<X1DIR>(k, j, i) +
                             coords.template FaceArea<X1DIR>(k, j, i + 1) /
                                 coords.template Dxc<X1DIR>(k, j, i + 1);
            if (ndim > 1) {
              diag_elem += coords.template FaceArea<X2DIR>(k, j, i) /
                               coords.template Dxc<X2DIR>(k, j, i) +
                           coords.template FaceArea<X2DIR>(k, j + 1, i) /
                               coords.template Dxc<X2DIR>(k, j + 1, i);
            }
            if (ndim > 2) {
              diag_elem += coords.template FaceArea<X3DIR>(k, j, i) /
                               coords.template Dxc<X3DIR>(k, j, i) +
                           coords.template FaceArea<X3DIR>(k + 1, j, i) /
                               coords.template Dxc<X3DIR>(k + 1, j, i);
            }
            pack(b, var_t(), k, j, i) = -diag_elem / coords.CellVolume(k, j, i);
          });
        });
    return TaskStatus::complete;
  }

  // Face fluxes F = -d(var_t)/dx, so that the flux divergence below yields +Laplacian.
  static parthenon::TaskStatus
  CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    using TE = parthenon::TopologicalElement;
    const int ndim = md->GetMeshPointer()->ndim;

    static auto desc =
        parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get());
    if (pack.GetNBlocks() == 0) return TaskStatus::complete;

    for (int dim = 0; dim < ndim; ++dim) {
      const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
      const int ioff = dim == 0;
      const int joff = dim == 1;
      const int koff = dim == 2;
      const int dir = dim + 1;
      using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
      auto idx_space =
          lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(), md.get(), te);
      RiotLoop::outer(
          idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
            const auto &coords = pack.GetCoordinates(b);
            RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
              pack.flux(b, dir, var_t(), k, j, i) =
                  (pack(b, var_t(), k - koff, j - joff, i - ioff) -
                   pack(b, var_t(), k, j, i)) /
                  coords.Dxc(dir, k, j, i);
            });
          });
    }
    return TaskStatus::complete;
  }

  // Calculate A in_t = out_t (in the region covered by md_in) for a given set of fluxes
  // calculated with in_t (which have possibly been corrected at coarse fine boundaries)
  static parthenon::TaskStatus
  FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    using TE = parthenon::TopologicalElement;
    const int ndim = md_in->GetMeshPointer()->ndim;

    static auto desc =
        parthenon::MakePackDescriptor<var_t>(md_in.get(), {}, {PDOpt::WithFluxes});
    static auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
    auto pack_in = desc.GetPack(md_in.get());
    auto pack_out = desc_out.GetPack(md_out.get());
    if (pack_in.GetNBlocks() == 0) return TaskStatus::complete;

    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_in.GetNBlocks(),
                                       md_in.get(), TE::CC);
    // Flux divergence combines face-centered fluxes with face areas and cell volume, and
    // reaches the high-side neighbor faces, so this uses (k, j, i).
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack_in.GetCoordinates(b);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            Real div = coords.template FaceArea<X1DIR>(k, j, i) *
                           pack_in.flux(b, X1DIR, var_t(), k, j, i) -
                       coords.template FaceArea<X1DIR>(k, j, i + 1) *
                           pack_in.flux(b, X1DIR, var_t(), k, j, i + 1);
            if (ndim > 1) {
              div += coords.template FaceArea<X2DIR>(k, j, i) *
                         pack_in.flux(b, X2DIR, var_t(), k, j, i) -
                     coords.template FaceArea<X2DIR>(k, j + 1, i) *
                         pack_in.flux(b, X2DIR, var_t(), k, j + 1, i);
            }
            if (ndim > 2) {
              div += coords.template FaceArea<X3DIR>(k, j, i) *
                         pack_in.flux(b, X3DIR, var_t(), k, j, i) -
                     coords.template FaceArea<X3DIR>(k + 1, j, i) *
                         pack_in.flux(b, X3DIR, var_t(), k + 1, j, i);
            }
            pack_out(b, var_t(), k, j, i) = div / coords.CellVolume(k, j, i);
          });
        });
    return TaskStatus::complete;
  }
};

} // namespace SelfGravity

#endif // SELF_GRAVITY_POISSON_EQUATION_HPP_
