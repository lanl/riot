//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

// C/C++ headers
#include <cmath>

// Riot headers
#include "diagnostics.hpp"
#include "riot_utils/riot_loops.hpp"

namespace dsplanar {

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::ShellMass
//! \brief User-history function that tracks the shell mass
Real ShellMass(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccmat::rho>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rcmeasure = 0.01;
  constexpr int be_mat_id = 3; // TODO()
  constexpr int w_mat_id = 4;  // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real mass = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        // Hoist the material-existence / sparse-id checks out of the inner reduction:
        // one inner_reduce per shell material, all joining the same accumulator.
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          const int sid = vmesh(b, ccmat::rho(n)).sparse_id;
          if (sid != w_mat_id && sid != be_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lsum) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);
            if (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure)
              lsum += pv_n(ccmat::rho(), idx) * coords.CellVolume(k, j, i);
          });
        }
      });

  return mass;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::ShellMomentumX
//! \brief User-history function that tracks the shell momentum
Real ShellMomentumX(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccbulk::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rcmeasure = 0.01;
  constexpr int be_mat_id = 3; // TODO()
  constexpr int w_mat_id = 4;  // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real momentum = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, vmesh);
        // One inner_reduce per shell material; the material-existence check is hoisted.
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          const int sid = vmesh(b, ccmat::rho(n)).sparse_id;
          if (sid != w_mat_id && sid != be_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lsum) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);
            if (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure) {
              lsum += pv_n(ccmat::rho(), idx) * pv(ccbulk::velocity(0), idx) *
                      coords.CellVolume(k, j, i);
            }
          });
        }
      });

  return momentum;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::ShellTemperatureM
//! \brief User-history function that tracks the shell mass*Temperature (to later average)
Real ShellTemperatureM(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccbulk::temperature>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rcmeasure = 0.01;
  constexpr int be_mat_id = 3; // TODO()
  constexpr int w_mat_id = 4;  // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real temp = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, vmesh);
        // One inner_reduce per shell material; the material-existence check is hoisted.
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          const int sid = vmesh(b, ccmat::rho(n)).sparse_id;
          if (sid != w_mat_id && sid != be_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lsum) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);
            if (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure) {
              lsum += pv_n(ccmat::rho(), idx) * pv(ccbulk::temperature(), idx) *
                      coords.CellVolume(k, j, i);
            }
          });
        }
      });

  return temp;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::FoamPeakPressure
//! \brief User-history function that tracks peak foam pressure
Real FoamPeakPressure(MeshData<Real> *md) {
  namespace cm = cell_variables::material_averaged;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<cm::pressure>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmin = -std::numeric_limits<Real>::max();
  constexpr Real rcmeasure = 0.01;
  constexpr int foam_mat_id = 6; // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real peakp = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        // sparse_id is unique per material, so at most one material matches; hoist that
        // search out of the inner reduction.
        for (int n = 0; n < vmesh.GetSize(b, cm::pressure()); n++) {
          if (vmesh(b, cm::pressure(n)).sparse_id != foam_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmax) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);
            const Real fpres = (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure)
                                   ? pv_n(cm::pressure(), idx)
                                   : rmin;
            lmax = std::max(lmax, fpres);
          });
        }
      });

  return peakp;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::WPeakPressure
//! \brief User-history function that tracks peak W pressure
Real WPeakPressure(MeshData<Real> *md) {
  namespace cm = cell_variables::material_averaged;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<cm::pressure>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmin = -std::numeric_limits<Real>::max();
  constexpr Real rcmeasure = 0.01;
  constexpr int w_mat_id = 4; // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real peakp = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        // sparse_id is unique per material, so at most one material matches; hoist that
        // search out of the inner reduction.
        for (int n = 0; n < vmesh.GetSize(b, cm::pressure()); n++) {
          if (vmesh(b, cm::pressure(n)).sparse_id != w_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmax) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);
            const Real wpres = (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure)
                                   ? pv_n(cm::pressure(), idx)
                                   : rmin;
            lmax = std::max(lmax, wpres);
          });
        }
      });

  return peakp;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::ShockPosition
//! \brief User-history function that tracks shock front
Real ShockPosition(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccbulk::pressure>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmin = -std::numeric_limits<Real>::max();
  constexpr Real pthr = 1.0e12;
  constexpr Real rcmeasure = 0.01;

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real x1sh = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, vmesh);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmax) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          const Real x1fp = coords.Xf<X1DIR>(i + 1);
          const Real x2v = coords.Xc<X2DIR>(j);
          const Real x3v = coords.Xc<X3DIR>(k);

          const bool inside_planar = (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure);
          const bool is_shocked = (pv(ccbulk::pressure(), idx) > pthr);
          const Real spos = (inside_planar)*x1fp + (!inside_planar) * rmin;
          lmax = std::max(lmax, ((is_shocked)*spos + (!is_shocked) * rmin));
        });
      });

  return x1sh;
}

//----------------------------------------------------------------------------------------
// Shared mass fraction threshold between SpikeBubbleMaxLoc/SpikeBubbleMinLoc
constexpr Real mfrac_w_thresh = 0.25;

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::SpikeBubbleMaxLoc
//! \brief User-history function that tracks SpikeBubble spike/bubble max location
Real SpikeBubbleMaxLoc(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccbulk::rho, ccmat::rho>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmin = -std::numeric_limits<Real>::max();
  constexpr Real rcmeasure = 0.01;
  constexpr Real mfrac_thresh = mfrac_w_thresh;
  constexpr int be_mat_id = 3; // TODO()
  constexpr int w_mat_id = 4;  // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Max<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real locmax = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, vmesh);
        // The shell mass fraction is a per-cell sum over the w and be materials, so it
        // cannot split into one inner_reduce per material. But sparse_id is unique per
        // material, so we can hoist the search: find the two shell material indices once
        // (-1 if absent) and drop the per-cell sparse_id comparison.
        int n_w = -1, n_be = -1;
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          const int sid = vmesh(b, ccmat::rho(n)).sparse_id;
          if (sid == w_mat_id) n_w = n;
          if (sid == be_mat_id) n_be = n;
        }
        // Construct at a clamped index (safe even if the material is absent -- usage is
        // guarded by the n_* >= 0 flags below).
        auto pv_w = RiotLoop::make_sparse_pack_view(idx_range, vmesh, std::max(n_w, 0));
        auto pv_be = RiotLoop::make_sparse_pack_view(idx_range, vmesh, std::max(n_be, 0));
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmax) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          const Real x1fp = coords.Xf<X1DIR>(i + 1);
          const Real x2v = coords.Xc<X2DIR>(j);
          const Real x3v = coords.Xc<X3DIR>(k);

          Real wloc = rmin;
          if (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure) {
            Real mfrac = 0.0;
            if (n_w >= 0) mfrac += pv_w(ccmat::rho(), idx);
            if (n_be >= 0) mfrac += pv_be(ccmat::rho(), idx);
            mfrac /= pv(ccbulk::rho(), idx);
            if (mfrac > mfrac_thresh) {
              // NOTE(@chadmeyer): A better answer might be based on actual gradients
              // wloc = mfrac * x1fp + (1.0 - mfrac) * x1f;
              wloc = x1fp;
            }
          }
          lmax = std::max(lmax, wloc);
        });
      });

  return locmax;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::SpikeBubbleMinLoc
//! \brief User-history function that tracks SpikeBubble spike/bubble min location
//! NOTE(@chadmeyer): this is now the "inverse" of the max location above
Real SpikeBubbleMinLoc(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<ccmat::rho, ccbulk::rho>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  constexpr Real rmax = std::numeric_limits<Real>::max();
  constexpr Real rcmeasure = 0.01;
  constexpr Real mfrac_thresh = 1.0 - mfrac_w_thresh;
  constexpr int foam_mat_id = 6; // TODO()

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  const Real locmin = RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto &coords = vmesh.GetCoordinates(b);
        auto pv = RiotLoop::make_pack_view(idx_range, vmesh);
        // sparse_id is unique per material, so at most one material matches; hoist that
        // search out of the inner reduction.
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          if (vmesh(b, ccmat::rho(n)).sparse_id != foam_mat_id) continue;
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, vmesh, n);
          RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lmin) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            const Real x1f = coords.Xf<X1DIR>(i);
            const Real x2v = coords.Xc<X2DIR>(j);
            const Real x3v = coords.Xc<X3DIR>(k);

            Real floc = rmax;
            if (std::sqrt(SQR(x2v) + SQR(x3v)) < rcmeasure) {
              const Real mfrac = pv_n(ccmat::rho(), idx) / pv(ccbulk::rho(), idx);
              if (mfrac > mfrac_thresh) {
                floc = x1f;
              }
            }
            lmin = std::min(lmin, floc);
          });
        }
      });

  return locmin;
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::AddHistory
//! \brief User-history enrollment function
void AddHistory(parthenon::Params &params) {
  auto HstMax = parthenon::UserHistoryOperation::max;
  auto HstMin = parthenon::UserHistoryOperation::min;
  auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};

  // HstSum
  hst_vars.emplace_back(HstSum, ShellMass, "ShellMass");
  hst_vars.emplace_back(HstSum, ShellMomentumX, "ShellMomentumX");
  hst_vars.emplace_back(HstSum, ShellTemperatureM, "ShellTemperatureM");
  // HstMax
  hst_vars.emplace_back(HstMax, FoamPeakPressure, "FoamPeakPressure");
  hst_vars.emplace_back(HstMax, WPeakPressure, "WPeakPressure");
  hst_vars.emplace_back(HstMax, ShockPosition, "ShockPosition");
  hst_vars.emplace_back(HstMax, SpikeBubbleMaxLoc, "SpikeBubbleMaxLoc");
  // HstMin
  hst_vars.emplace_back(HstMin, SpikeBubbleMinLoc, "SpikeBubbleMinLoc");

  params.Add(parthenon::hist_param_key, hst_vars);
}

//----------------------------------------------------------------------------------------
//! \fn  Real dsplanar::ProblemPackage
//! \brief Creates dsplanar problem package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("dsplanar");
  dsplanar::AddHistory(pkg->AllParams());
  return pkg;
}

} // namespace dsplanar
