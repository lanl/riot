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
#ifndef IONIZATION_CONDUCTION_EQUATION_HPP_
#define IONIZATION_CONDUCTION_EQUATION_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>
#include <singularity-eos/eos/eos.hpp>
#include <utils/constants.hpp>

#include "microphysics/eos_riot.hpp"
#include "microphysics/pte_closure.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace Ionization {
//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskStatus CalculateFluxes
//! \brief
template <class var_t, class D_t>
inline parthenon::TaskStatus
CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                std::shared_ptr<parthenon::MeshData<Real>> &md) {
  using namespace parthenon;
  using TE = parthenon::TopologicalElement;

  const int ndim = md->GetMeshPointer()->ndim;

  static auto desc =
      parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
  static auto desc_mat = parthenon::MakePackDescriptor<D_t>(md_mat.get());
  auto pack = desc.GetPack(md.get());
  auto pack_mat = desc_mat.GetPack(md_mat.get());

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
            const Real one_over_dx = 1.0 / coords.Dxc(dim + 1, k, j, i);
            pack.flux(b, dir, var_t(), k, j, i) =
                -pack_mat(b, te, D_t(), k, j, i) *
                (pack(b, var_t(), k, j, i) -
                 pack(b, var_t(), k - koff, j - joff, i - ioff)) *
                one_over_dx;
          });
        });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskStatus CalculateLocalLinear
//! \brief
template <class var_t, class diag_loc_t>
inline parthenon::TaskStatus
CalculateLocalLinear(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
  using namespace parthenon;
  using TE = parthenon::TopologicalElement;

  static auto desc_mat = parthenon::MakePackDescriptor<diag_loc_t>(md_mat.get());
  static auto desc = parthenon::MakePackDescriptor<var_t>(md_in.get());
  auto pack_mat = desc_mat.GetPack(md_mat.get());
  auto pack_in = desc.GetPack(md_in.get());
  auto pack_out = desc.GetPack(md_out.get());

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_in.GetNBlocks(),
                                     md_in.get(), TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pmat = RiotLoop::make_pack_view(idx_range, pack_mat);
        auto pin = RiotLoop::make_pack_view(idx_range, pack_in);
        auto pout = RiotLoop::make_pack_view(idx_range, pack_out);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pout(var_t(), kji) = pmat(diag_loc_t(), kji) * pin(var_t(), kji);
        });
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskStatus FluxMultiplyMatrix
//! \brief Calculate A in_t = out_t (in the region covered by md_in) for a given set of
//! fluxes calculated with in_t (which have possibly been corrected at coarse fine
//! boundaries)
template <class var_t>
inline parthenon::TaskStatus
FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                   std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
  using namespace parthenon;
  const int ndim = md_in->GetMeshPointer()->ndim;

  static auto desc =
      parthenon::MakePackDescriptor<var_t>(md_in.get(), {}, {PDOpt::WithFluxes});
  auto pack_in = desc.GetPack(md_in.get());
  auto pack_out = desc.GetPack(md_out.get());

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  using TE = parthenon::TopologicalElement;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_in.GetNBlocks(),
                                     md_in.get(), TE::CC);
  // Flux divergence combines face-centered fluxes with face areas and cell volume, and
  // reaches the high-side neighbor faces, so this uses (k, j, i).
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto &coords = pack_in.GetCoordinates(b);
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          const Real one_over_vol = 1.0 / coords.CellVolume(k, j, i);
          pack_out(b, var_t(), k, j, i) +=
              (coords.template FaceArea<X1DIR>(k, j, i + 1) *
                   pack_in.flux(b, X1DIR, var_t(), k, j, i + 1) -
               coords.template FaceArea<X1DIR>(k, j, i) *
                   pack_in.flux(b, X1DIR, var_t(), k, j, i)) *
              one_over_vol;
          pack_out(b, var_t(), k, j, i) +=
              (coords.template FaceArea<X2DIR>(k, j + (ndim > 1), i) *
                   pack_in.flux(b, X2DIR, var_t(), k, j + (ndim > 1), i) -
               coords.template FaceArea<X2DIR>(k, j, i) *
                   pack_in.flux(b, X2DIR, var_t(), k, j, i)) *
              one_over_vol;
          pack_out(b, var_t(), k, j, i) +=
              (coords.template FaceArea<X3DIR>(k + (ndim > 2), j, i) *
                   pack_in.flux(b, X3DIR, var_t(), k + (ndim > 2), j, i) -
               coords.template FaceArea<X3DIR>(k, j, i) *
                   pack_in.flux(b, X3DIR, var_t(), k, j, i)) *
              one_over_vol;
        });
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::TaskID AddFluxContribution
//! \brief
template <class var_t>
inline parthenon::TaskID
AddFluxContribution(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
  auto flux_res = depends_on;
  if (!(md_in->grid.type() == parthenon::GridType::two_level_composite)) {
    auto start_flxcor =
        tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
    auto send_flxcor = tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
    auto recv_flxcor = tl.AddTask(send_flxcor, parthenon::ReceiveFluxCorrections, md_in);
    flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
  }
  return tl.AddTask(flux_res, FluxMultiplyMatrix<var_t>, md_in, md_out);
}

template <class var_t>
class LinearizedDiffusionEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  // Add tasks to calculate the result of the matrix A (which is implicitly defined by
  // this class) being applied to md_in and stored in md_out
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto diag = tl.AddTask(depends_on, CalculateLocalLinear<var_t, diag_loc>, md_mat,
                           md_in, md_out);
    auto flux = tl.AddTask(depends_on, CalculateFluxes<var_t, D>, md_mat, md_in);
    return AddFluxContribution<var_t>(tl, diag | flux, md_in, md_out);
  }

  // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
  // For a uniform grid or when flux correction is ignored, this diagonal calculation
  // is exact. Exactness is (probably) not required since it is just used in Jacobi
  // iterations.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_mat->GetMeshPointer()->ndim;

    static auto desc_mat = parthenon::MakePackDescriptor<D, D, diag_loc>(md_mat.get());
    static auto desc = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack = desc.GetPack(md_diag.get());
    auto pack_mat = desc_mat.GetPack(md_mat.get());

    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    using TE = parthenon::TopologicalElement;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(),
                                       md_diag.get(), TE::CC);
    // Combines face-centered diffusion coefficients with per-direction face areas and
    // cell size across a neighbor stencil, so this uses (k, j, i).
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          auto pout = RiotLoop::make_pack_view(idx_range, pack);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            Real diag_elem = (pack_mat(b, TE::F1, D(), k, j, i) *
                                  coords.template FaceArea<X1DIR>(k, j, i) /
                                  coords.template Dxc<X1DIR>(k, j, i) +
                              pack_mat(b, TE::F1, D(), k, j, i + 1) *
                                  coords.template FaceArea<X1DIR>(k, j, i + 1) /
                                  coords.template Dxc<X1DIR>(k, j, i + 1)) /
                             coords.CellVolume(k, j, i);
            diag_elem += (ndim > 1) *
                         (pack_mat(b, TE::F2, D(), k, j, i) *
                              coords.template FaceArea<X2DIR>(k, j, i) /
                              coords.template Dxc<X2DIR>(k, j, i) +
                          pack_mat(b, TE::F2, D(), k, j + (ndim > 1), i) *
                              coords.template FaceArea<X2DIR>(k, j + (ndim > 1), i) /
                              coords.template Dxc<X2DIR>(k, j + (ndim > 1), i)) /
                         coords.CellVolume(k, j, i);
            diag_elem += (ndim > 2) *
                         (pack_mat(b, TE::F3, D(), k, j, i) *
                              coords.template FaceArea<X3DIR>(k, j, i) /
                              coords.template Dxc<X3DIR>(k, j, i) +
                          pack_mat(b, TE::F3, D(), k + (ndim > 2), j, i) *
                              coords.template FaceArea<X3DIR>(k + (ndim > 2), j, i) /
                              coords.template Dxc<X3DIR>(k + (ndim > 2), j, i)) /
                         coords.CellVolume(k, j, i);
            pout(var_t(), k, j, i) = pack_mat(b, diag_loc(), k, j, i) + diag_elem;
          });
        });

    return TaskStatus::complete;
  }
};

} // namespace Ionization

#endif // IONIZATION_CONDUCTION_EQUATION_HPP_
