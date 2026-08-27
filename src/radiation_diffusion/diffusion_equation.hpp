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
#ifndef RADIATION_DIFFUSION_DIFFUSION_EQUATION_HPP_
#define RADIATION_DIFFUSION_DIFFUSION_EQUATION_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>
#include <singularity-eos/eos/eos.hpp>
#include <utils/constants.hpp>

#include "microphysics/eos_riot.hpp"
#include "microphysics/pte_closure.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace RadiationDiffusion {

// The functions below can be generic for multi-group and energy integrated diffusion
template <class var_t, class Dlo_t, class Dup_t>
inline parthenon::TaskStatus
CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                std::shared_ptr<parthenon::MeshData<Real>> &md) {
  using namespace parthenon;
  using TE = parthenon::TopologicalElement;

  const int ndim = md->GetMeshPointer()->ndim;
  int nblocks = md->NumBlocks();

  static auto desc =
      parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
  using namespace RadiationDiffusion::MultiGroupVars;
  static auto desc_mat =
      parthenon::MakePackDescriptor<Dlo_t, Dup_t, DeltaX>(md_mat.get());
  auto pack = desc.GetPack(md.get());
  auto pack_mat = desc_mat.GetPack(md_mat.get());

  for (int dim = 0; dim < ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    // The diffusion coefficient arrays are cell mem aligned
    using lt = RiotUtils::LoopType<>;
    auto idx_space =
        lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(), md.get(), te);

    const auto dir = static_cast<parthenon::CoordinateDirection>(dim + 1);
    auto offset = idx_space.GetDelta(dir);
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const int ngroup = pack.GetSize(b, var_t());
          auto one_over_dx = make_var_view(idx_range, pack_mat, te, DeltaX());
          for (int n = 0; n < ngroup; ++n) {
            auto Dup = make_var_view(idx_range, pack_mat, te, Dup_t(n));
            auto Dlo = make_var_view(idx_range, pack_mat, te, Dlo_t(n));
            auto vv = make_var_view(idx_range, pack, var_t(n));
            auto fv = make_flux_view(idx_range, pack, dir, var_t(n));
            RiotLoop::inner(idx_range, [&](const auto kji) {
              fv(kji) =
                  -(Dup(kji) * vv(kji) - Dlo(kji) * vv(kji - offset)) * one_over_dx(kji);
            });
          }
        });
  }
  return TaskStatus::complete;
}

// Calculate A in_t = out_t (in the region covered by md_in) for a given set of fluxes
// calculated with in_t (which have possibly been corrected at coarse fine boundaries)
template <class var_t>
inline parthenon::TaskStatus
FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                   std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                   std::shared_ptr<parthenon::MeshData<Real>> &md_out, const Real fac) {
  using namespace parthenon;
  const int ndim = md_in->GetMeshPointer()->ndim;

  static auto desc =
      parthenon::MakePackDescriptor<var_t>(md_in.get(), {}, {PDOpt::WithFluxes});
  auto pack_in = desc.GetPack(md_in.get());
  auto pack_out = desc.GetPack(md_out.get());

  using namespace RadiationDiffusion::MultiGroupVars;
  static auto desc_mat =
      parthenon::MakePackDescriptor<face_area, DeltaX, volume>(md_mat.get());
  auto pack_mat = desc_mat.GetPack(md_mat.get());

  using TE = parthenon::TopologicalElement;
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_in.GetNBlocks(),
                                     md_in.get(), TE::CC);
  const auto di = idx_space.GetDelta(X1DIR);
  const auto dj = idx_space.GetDelta(X2DIR);
  const auto dk = idx_space.GetDelta(X3DIR);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int ngroup = pack_in.GetSize(b, var_t());
        auto AreaX1 = make_var_view(idx_range, pack_mat, TE::F1, face_area());
        auto AreaX2 = make_var_view(idx_range, pack_mat, TE::F2, face_area());
        auto AreaX3 = make_var_view(idx_range, pack_mat, TE::F3, face_area());
        auto iVol = make_var_view(idx_range, pack_mat, TE::CC, volume());
        for (int n = 0; n < ngroup; ++n) {
          auto fv1 = make_flux_view(idx_range, pack_in, X1DIR, var_t(n));
          auto fv2 = make_flux_view(idx_range, pack_in, X2DIR, var_t(n));
          auto fv3 = make_flux_view(idx_range, pack_in, X3DIR, var_t(n));
          auto vv = make_var_view(idx_range, pack_out, var_t(n));
          RiotLoop::inner(idx_range, [&](const auto kji) {
            Real div_flux = AreaX1(kji + di) * fv1(kji + di) - AreaX1(kji) * fv1(kji);
            div_flux += AreaX2(kji + dj) * fv2(kji + dj) - AreaX2(kji) * fv2(kji);
            div_flux += AreaX3(kji + dk) * fv3(kji + dk) - AreaX3(kji) * fv3(kji);
            vv(kji) += div_flux * iVol(kji) * fac;
          });
        }
      });
  return TaskStatus::complete;
}

template <class var_t, class diag_loc_t, bool group_couple = false>
inline parthenon::TaskStatus
CalculateLocalLinear(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
  using namespace parthenon;
  const int ndim = md_in->GetMeshPointer()->ndim;
  using TE = parthenon::TopologicalElement;

  static auto desc_mat =
      parthenon::MakePackDescriptor<diag_loc_t, MultiGroupVars::dSdT,
                                    MultiGroupVars::sigma>(md_mat.get());
  static auto desc = parthenon::MakePackDescriptor<var_t>(md_in.get());
  auto pack_mat = desc_mat.GetPack(md_mat.get());
  auto pack_in = desc.GetPack(md_in.get());
  auto pack_out = desc.GetPack(md_out.get());
  const int ngroup = pack_in.GetSizeHost(0, var_t());

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_mat.GetNBlocks(),
                                     md_mat.get(), TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        for (int g = 0; g < ngroup; ++g) {
          auto voutv = make_var_view(idx_range, pack_out, var_t(g));
          auto vinv = make_var_view(idx_range, pack_in, var_t(g));
          auto diagv = make_var_view(idx_range, pack_mat, diag_loc_t(g));
          RiotLoop::inner(idx_range,
                          [&](const auto kji) { voutv(kji) = diagv(kji) * vinv(kji); });
          if constexpr (group_couple) {
            auto dsdtv = make_var_view(idx_range, pack_mat, MultiGroupVars::dSdT(g));
            for (int gp = 0; gp < ngroup; ++gp) {
              if (gp != g) {
                auto sigmapv =
                    make_var_view(idx_range, pack_mat, MultiGroupVars::sigma(gp));
                auto vpv = make_var_view(idx_range, pack_in, var_t(gp));
                RiotLoop::inner(idx_range, [&](const auto kji) {
                  voutv(kji) -= sigmapv(kji) * dsdtv(kji) * vpv(kji);
                });
              }
            }
          }
        }
      });
  return TaskStatus::complete;
}

template <class var_t>
inline parthenon::TaskStatus
CorrectRefinementBoundaryFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md) {
  using namespace parthenon;
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  using TE = parthenon::TopologicalElement;

  int nblocks = md->NumBlocks();

  static const auto desc =
      parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
  auto pack = desc.GetPack(md.get());
  const std::size_t scratch_size_in_bytes = 0;
  const std::size_t scratch_level = 1;

  const parthenon::Indexer3D idxers[6]{
      parthenon::Indexer3D(kb, jb, {ib.s, ib.s}),
      parthenon::Indexer3D(kb, jb, {ib.e + 1, ib.e + 1}),
      parthenon::Indexer3D(kb, {jb.s, jb.s}, ib),
      parthenon::Indexer3D(kb, {jb.e + 1, jb.e + 1}, ib),
      parthenon::Indexer3D({kb.s, kb.s}, jb, ib),
      parthenon::Indexer3D({kb.e + 1, kb.e + 1}, jb, ib)};
  constexpr int x1off[6]{-1, 1, 0, 0, 0, 0};
  constexpr int x2off[6]{0, 0, -1, 1, 0, 0};
  constexpr int x3off[6]{0, 0, 0, 0, -1, 1};
  constexpr int dirs[6]{X1DIR, X1DIR, X2DIR, X2DIR, X3DIR, X3DIR};
  // NOTE(): This is deliberately left as par_for_outer/inner rather than a RiotLoop
  // abstraction. It is a block-face sweep, not a logical-interior sweep: it visits only
  // the 2*ndim boundary flux planes, selects each face at runtime by a per-block
  // fine-coarse GetLevel() check, and scales the face-centered flux there. That shape
  // maps onto neither RiotLoop::inner (whose contract is a full logical-interior
  // traversal) nor par_for_bndry (a per-MeshBlock helper with no pack-level GetLevel
  // selection), so it stays a raw boundary loop like the physical-BC loops.
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SetFluxBoundaries", DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const auto &coords = pack.GetCoordinates(b);
        const int level = pack.GetLevel(b, 0, 0, 0);
        for (int face = 0; face < ndim * 2; ++face) {
          const auto &idxer = idxers[face];
          const auto dir = dirs[face];
          // Correct for size of neighboring zone at fine-coarse boundary when using
          // constant prolongation
          if (pack.GetLevel(b, x3off[face], x2off[face], x1off[face]) == level - 1) {
            parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                     idxer.size() - 1, [&](const int idx) {
                                       const auto [k, j, i] = idxer(idx);
                                       pack.flux(b, dir, var_t(), k, j, i) /= 1.5;
                                     });
          }
        }
      });
  return TaskStatus::complete;
}

template <class var_t>
inline parthenon::TaskID
AddFluxContribution(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_out,
                    const Real fac = 1.0) {
  auto flux_res = depends_on;
  if (!(md_in->grid.type() == parthenon::GridType::two_level_composite)) {
    auto start_flxcor =
        tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
    auto send_flxcor = tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
    auto recv_flxcor = tl.AddTask(send_flxcor, parthenon::ReceiveFluxCorrections, md_in);
    flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
  }
  return tl.AddTask(flux_res, FluxMultiplyMatrix<var_t>, md_mat, md_in, md_out, fac);
}

template <class var_t>
inline parthenon::TaskStatus Scale(std::shared_ptr<parthenon::MeshData<Real>> &md_matrix,
                                   std::shared_ptr<parthenon::MeshData<Real>> &md) {
  using namespace parthenon;
  const int ndim = md->GetMeshPointer()->ndim;

  auto desc = parthenon::MakePackDescriptor<var_t>(md.get());
  auto pack = desc.GetPack(md.get());
  auto pack_matrix = desc.GetPack(md_matrix.get());
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ngroup = pack.GetSizeHost(0, var_t());
  parthenon::par_for(
      "DiffusionEquation::Scale", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int g = 0; g < ngroup; ++g) {
          pack(b, var_t(g), k, j, i) /= pack_matrix(b, var_t(g), k, j, i);
        }
      });
  return TaskStatus::complete;
}

template <class var_t, class dloc_t, class Dlo_t, class Dup_t, bool group_couple = false>
class LinearizedRadiationDiffusionEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  static parthenon::TaskStatus SetBoundary(std::shared_ptr<parthenon::MeshData<Real>> &md,
                                           bool coarse) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;

    std::set<PDOpt> opts{};
    if (coarse) opts.emplace(PDOpt::Coarse);
    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, opts);
    auto pack = desc.GetPack(md.get(), GetBlockSelector::OnPhysicalBoundary());
    if (pack.GetNBlocks() > 0) {
      CellLevel cl = coarse ? CellLevel::coarse : CellLevel::same;
      IndexRange ib = md->GetBoundsI(cl, IndexDomain::interior);
      IndexRange jb = md->GetBoundsJ(cl, IndexDomain::interior);
      IndexRange kb = md->GetBoundsK(cl, IndexDomain::interior);
      const int ngroup = pack.GetSizeHost(0, var_t());

      const int scratch_size_in_bytes = 0;
      const std::size_t scratch_level = 1;
      // NOTE(): This reflecting boundary is deliberately left as par_for_outer/inner
      // rather than a RiotLoop abstraction. It iterates the interior boundary layer but
      // *writes* only the adjacent ghost cell (q(ghost) = -q(interior)), gated per-block
      // by pack.IsPhysicalBoundary(). RiotLoop::inner's halo has union semantics (it
      // would also revisit the interior cells, which must not be overwritten), and
      // par_for_bndry is a per-MeshBlock helper without pack-level boundary selection --
      // so neither fits. It stays a raw boundary loop like the physical-BC loops.
      parthenon::par_for_outer(
          DEFAULT_OUTER_LOOP_PATTERN, "InitializeRadiationQuantities", DevExecSpace(),
          scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1, 0, ngroup - 1,
          -(ndim > 2), (ndim > 2), -(ndim > 1), (ndim > 1), -1, 1,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int g, int ok,
                        int oj, int oi) {
            const int tot_offset = std::abs(ok) + std::abs(oj) + std::abs(oi);

            auto get_lower = [](int offset, auto bound) {
              if (offset != 0) {
                return offset > 0 ? bound.e : bound.s;
              } else {
                return bound.s;
              }
            };
            auto get_upper = [](int offset, auto bound) {
              if (offset != 0) {
                return offset > 0 ? bound.e : bound.s;
              } else {
                return bound.e;
              }
            };

            if (tot_offset == 1 && pack.IsPhysicalBoundary(b, ok, oj, oi)) {
              parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member,
                                       get_lower(ok, kb), get_upper(ok, kb),
                                       get_lower(oj, jb), get_upper(oj, jb),
                                       get_lower(oi, ib), get_upper(oi, ib),
                                       [&](const int k, const int j, const int i) {
                                         pack(b, var_t(g), k + ok, j + oj, i + oi) =
                                             -pack(b, var_t(g), k, j, i);
                                       });
            }
          });
    }
    return TaskStatus::complete;
  }

  // Add tasks to calculate the result of the matrix A (which is implicitly defined by
  // this class) being applied to md_in and stored in md_out
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto diag = tl.AddTask(depends_on, CalculateLocalLinear<var_t, dloc_t, group_couple>,
                           md_mat, md_in, md_out);
    auto flux =
        tl.AddTask(depends_on, CalculateFluxes<var_t, Dlo_t, Dup_t>, md_mat, md_in);
    flux = tl.AddTask(flux, CorrectRefinementBoundaryFluxes<var_t>, md_in);
    return AddFluxContribution<var_t>(tl, diag | flux, md_mat, md_in, md_out);
  }

  // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
  // For a uniform grid or when flux correction is ignored, this diagonal calculation
  // is exact. Exactness is (probably) not required since it is just used in Jacobi
  // iterations.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_mat->GetMeshPointer()->ndim;
    int nblocks = md_mat->NumBlocks();

    static auto desc_mat =
        parthenon::MakePackDescriptor<Dlo_t, Dup_t, dloc_t>(md_mat.get());
    static auto desc = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack = desc.GetPack(md_diag.get());
    auto pack_mat = desc_mat.GetPack(md_mat.get());

    using TE = parthenon::TopologicalElement;
    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(),
                                       md_mat.get(), TE::CC);
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          const int ngroup = pack.GetSize(b, var_t());
          RiotLoop::inner(idx_range, [&](const int kk, const int jj, const int ii) {
            for (int n = 0; n < ngroup; ++n) {
              const Real DX1lo = pack_mat(b, TE::F1, Dup_t(n), kk, jj, ii);
              const Real DX1up = pack_mat(b, TE::F1, Dlo_t(n), kk, jj, ii + 1);

              const Real DX2lo = pack_mat(b, TE::F2, Dup_t(n), kk, jj, ii);
              const Real DX2up = pack_mat(b, TE::F2, Dlo_t(n), kk, jj + (ndim > 1), ii);

              const Real DX3lo = pack_mat(b, TE::F3, Dup_t(n), kk, jj, ii);
              const Real DX3up = pack_mat(b, TE::F3, Dlo_t(n), kk + (ndim > 2), jj, ii);

              const Real lam = pack_mat(b, dloc_t(n), kk, jj, ii);
              Real diag_elem = (DX1lo * coords.template FaceArea<X1DIR>(kk, jj, ii) /
                                    coords.template Dxc<X1DIR>(kk, jj, ii) +
                                DX1up * coords.template FaceArea<X1DIR>(kk, jj, ii + 1) /
                                    coords.template Dxc<X1DIR>(kk, jj, ii + 1)) /
                               coords.CellVolume(kk, jj, ii);
              diag_elem +=
                  (ndim > 1) *
                  (DX2lo * coords.template FaceArea<X2DIR>(kk, jj, ii) /
                       coords.template Dxc<X2DIR>(kk, jj, ii) +
                   DX2up * coords.template FaceArea<X2DIR>(kk, jj + (ndim > 1), ii) /
                       coords.template Dxc<X2DIR>(kk, jj + (ndim > 1), ii)) /
                  coords.CellVolume(kk, jj, ii);
              diag_elem +=
                  (ndim > 2) *
                  (DX3lo * coords.template FaceArea<X3DIR>(kk, jj, ii) /
                       coords.template Dxc<X3DIR>(kk, jj, ii) +
                   DX3up * coords.template FaceArea<X3DIR>(kk + (ndim > 2), jj, ii) /
                       coords.template Dxc<X3DIR>(kk + (ndim > 2), jj, ii)) /
                  coords.CellVolume(kk, jj, ii);
              pack(b, var_t(n), kk, jj, ii) = lam + diag_elem;
            }
          });
        });

    return TaskStatus::complete;
  }
};

} // namespace RadiationDiffusion

#endif // RADIATION_DIFFUSION_DIFFUSION_EQUATION_HPP_
