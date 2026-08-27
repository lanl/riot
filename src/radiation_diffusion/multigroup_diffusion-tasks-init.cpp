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

#include <chrono>
#include <cmath>

#include <parthenon/package.hpp>
#include <singularity-eos/eos/eos.hpp>
#include <utils/constants.hpp>

#include "diffusion_equation.hpp"
#include "material_helpers.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "microphysics/pte_closure.hpp"
#include "multigroup_diffusion.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;

namespace RadiationDiffusion {

// Set the matrix and the rhs
// Containers md_base and md_star are inputs
// Containers md_matrix and md_rhs are outputs
template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::InitializeRadiationQuantities(
    std::shared_ptr<parthenon::MeshData<Real>> md_base,
    std::shared_ptr<parthenon::MeshData<Real>> md_star,
    std::shared_ptr<parthenon::MeshData<Real>> md_matrix,
    std::shared_ptr<parthenon::MeshData<Real>> md_rhs, Real dt) {
  using namespace parthenon;

  using namespace MultiGroupVars;
  using TE = parthenon::TopologicalElement;

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pmesh = md_base->GetMeshPointer();
  const int ndim = pmesh->ndim;

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const bool flux_limit = pkg->Param<bool>("flux_limit");
  const Real a = pkg->Param<Real>("a_radiation");
  const Real c_light = pkg->Param<Real>("c_light");
  const int ngroup = pkg->Param<int>("ngroup");
  const bool update_temperature = pkg->Param<bool>("update_temperature");
  const auto group_energies_in_K = pkg->Param<ParArray1D<Real>>("group_energies_in_K");

  MaterialHelpers mat_helpers(pmesh, "multigroup_diffusion_package", temperature());
  SourceHelper source_helper(pmesh, dt);

  static const auto desc_matrix =
      parthenon::MakePackDescriptor<diag_loc, sigma, D, dSdT>(md_matrix.get());
  auto pack_matrix = desc_matrix.GetPack(md_matrix.get());

  static const auto desc_rhs = parthenon::MakePackDescriptor<Egroup>(md_rhs.get());
  auto pack_rhs = desc_rhs.GetPack(md_rhs.get());

  static const auto desc_base =
      parthenon::MakePackDescriptor<temperature, ccbulk::internal_energy, ccbulk::rho,
                                    ccmat::rho, cm::rho, ccmat::volume_fraction, Egroup,
                                    Fgroup, kappa_cell, kappa_face, dTc, dSdT>(
          md_base.get());
  auto pack_base = desc_base.GetPack(md_base.get());

  static const auto desc_star =
      parthenon::MakePackDescriptor<temperature0, Egroup, Fgroup>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  const Real dtcl = dt * c_light;

  // Interior plus one ghost layer (kappa_cell is consumed in the ghost layer by the
  // face-coefficient stencil below), matching the old RawMemoryIndexer halo of 1.
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 1, pack_base.GetNBlocks(),
                                     md_base.get(), TE::CC);
  idx_space.template AddPerPointScratch<Real, MAX_GROUPS>(5); // aa, daadT, atot, S, dSdt
  idx_space.template AddPerPointScratch<Real>(5); // denom, RT, cv, Egas, Egas0
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto aa = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
        auto daadT = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
        auto atot = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
        mat_helpers.CalculateMultiGroupOpacities<temperature>(
            idx_range, pack_base, pack_base, b, MaterialHelpers::AlphaAbsMGScratch(aa),
            MaterialHelpers::dAlphaAbsMGdTScratch(daadT));
        mat_helpers.CalculateMultiGroupOpacities<temperature0>(
            idx_range, pack_base, pack_star, b, MaterialHelpers::AlphaTotMGScratch(atot));

        auto S = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
        auto dSdt = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
        source_helper.CalculateSource<temperature>(idx_range, pack_base, pack_base, aa,
                                                   daadT, b, S, dSdt);

        auto denom = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto RT = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto cv = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto Egas = RiotLoop::GetPerPointScratch<Real>(idx_range);
        mat_helpers.CalculateEos<temperature>(idx_range, pack_base, pack_base, b,
                                              MaterialHelpers::CvScratch(cv),
                                              MaterialHelpers::EgasScratch(Egas));

        auto Egas0 = RiotLoop::GetPerPointScratch<Real>(idx_range);
        mat_helpers.CalculateEos<temperature0>(idx_range, pack_base, pack_star, b,
                                               MaterialHelpers::EgasScratch(Egas0));

        idx_range.TeamBarrier();
        RiotLoop::inner(idx_range, [&](const auto kji) {
          denom(kji) = cv(kji);
          RT(kji) = Egas(kji) - Egas0(kji);
        });

        for (int g = 0; g < ngroup; ++g) {
          idx_range.TeamBarrier();
          auto dsdtv = make_var_view(idx_range, pack_base, dSdT(g));
          RiotLoop::inner(idx_range, [&](const auto kji) {
            dsdtv(kji) = dSdt(g, kji);
            denom(kji) += dSdt(g, kji);
            RT(kji) += S(g, kji);
          });
        }

        idx_range.TeamBarrier();
        auto dtcv = make_var_view(idx_range, pack_base, dTc());
        RiotLoop::inner(idx_range,
                        [&](const auto kji) { dtcv(kji) = -RT(kji) / denom(kji); });

        // Caculate the RHS and the local part of the matrix diagonal
        idx_range.TeamBarrier();
        for (int g = 0; g < ngroup; ++g) {
          auto Eg0v = make_var_view(idx_range, pack_star, Egroup(g));
          auto Egv = make_var_view(idx_range, pack_base, Egroup(g));
          auto sigmav = make_var_view(idx_range, pack_matrix, sigma(g));
          auto kapv = make_var_view(idx_range, pack_base, kappa_cell(g));
          auto dsdtv = make_var_view(idx_range, pack_matrix, dSdT(g));
          auto diagv = make_var_view(idx_range, pack_matrix, diag_loc(g));
          auto rhsv = make_var_view(idx_range, pack_rhs, Egroup(g));
          if (update_temperature) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              rhsv(kji) = -(Egv(kji) - Eg0v(kji) +
                            dSdt(g, kji) / denom(kji) * (Egas(kji) - Egas0(kji)) -
                            cv(kji) / denom(kji) * S(g, kji));
              diagv(kji) =
                  1.0 + dtcl * aa(g, kji) * ((denom(kji) - dSdt(g, kji)) / denom(kji));
              sigmav(kji) = dtcl * aa(g, kji) / denom(kji);
              kapv(kji) = atot(g, kji);
              dsdtv(kji) = dSdt(g, kji);
            });
            for (int gp = 0; gp < ngroup; ++gp) {
              if (gp != g) {
                RiotLoop::inner(idx_range, [&](const auto kji) {
                  rhsv(kji) -= (dSdt(g, kji) * S(gp, kji) - dSdt(gp, kji) * S(g, kji)) /
                               denom(kji);
                });
              }
            }
          } else {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              rhsv(kji) = -(Egv(kji) - Eg0v(kji) - S(g, kji));
              diagv(kji) = 1.0 + dtcl * aa(g, kji);
              sigmav(kji) = dtcl * aa(g, kji) / denom(kji);
              kapv(kji) = atot(g, kji);
              dsdtv(kji) = dSdt(g, kji);
            });
          }
        }
      });

  // Set the diffusion coefficients on the cell faces
  for (int dim = 0; dim < ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    // The diffusion coefficient arrays are cell mem aligned
    using ltf = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space_f = ltf::GetIndexSpace(IndexDomain::interior, 0,
                                          pack_base.GetNBlocks(), md_base.get(), te);
    auto dir = static_cast<parthenon::CoordinateDirection>(dim + 1);
    const auto offset = idx_space_f.GetDelta(dir);
    RiotLoop::outer(
        idx_space_f, KOKKOS_LAMBDA(const ltf::idx_range_t &idx_range, const int b) {
          for (int g = 0; g < ngroup; ++g) {
            auto kappa_c = make_var_view(idx_range, pack_base, kappa_cell(g));
            auto kappa_f = make_var_view(idx_range, pack_base, te, kappa_face(g));
            auto Dv = make_var_view(idx_range, pack_matrix, te, D(g));
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real kf = 2.0 / (1.0 / kappa_c(kji - offset) + 1.0 / kappa_c(kji));
              kappa_f(kji) = kf;
              Dv(kji) = dtcl * dtcl / (3.0 * (1.0 + dtcl * kf));
            });
          }
        });
  }
  return TaskStatus::complete;
}

template struct MultiGroupTasks<cell_variables::cell_averaged::bulk::temperature>;
template struct MultiGroupTasks<
    cell_variables::cell_averaged::bulk::electron_temperature>;

} // namespace RadiationDiffusion
