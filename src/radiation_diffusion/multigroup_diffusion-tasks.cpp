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

template <class temperature>
parthenon::TaskStatus
MultiGroupTasks<temperature>::Completion(std::shared_ptr<AllReduce<Real>> step_norm,
                                         Mesh *pmesh, int partition) {
  using namespace parthenon;
  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  auto solver = pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver");
  const Real nr_tol = pkg->Param<Real>("nr_tolerance");
  const bool print_per_nr_step = pkg->Param<bool>("print_per_nr_step");

  const std::size_t ncells = pmesh->GetNumberOfMeshBlockCells() * pmesh->nbtotal;
  const Real relative_delta = std::pow(step_norm->val / ncells, 1.0 / 2.0);
  if (print_per_nr_step && Globals::my_rank == 0 && partition == 0)
    printf("L2(dT / T) = %e (%ld) initial residual = %e solver convergence = %e "
           "iterations = %i\n",
           relative_delta, ncells, solver->GetInitialResidual(),
           solver->GetFinalResidual(), solver->GetFinalIterations());
  if (partition == 0) {
    auto *solver_iters = pkg->MutableParam<int>("step_solver_iterations");
    *solver_iters += solver->GetFinalIterations();
    auto *nr_iters = pkg->MutableParam<int>("step_newt_iterations");
    *nr_iters += 1;
  }
  if (relative_delta < nr_tol) return TaskStatus::complete;
  return TaskStatus::iterate;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::SaveStarState(
    std::shared_ptr<parthenon::MeshData<Real>> md,
    std::shared_ptr<parthenon::MeshData<Real>> md_star) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  using TE = parthenon::TopologicalElement;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  static const auto desc =
      parthenon::MakePackDescriptor<temperature, Egroup, Fgroup>(md.get());
  auto pack = desc.GetPack(md.get());
  static const auto desc_star =
      parthenon::MakePackDescriptor<temperature0, Egroup, Fgroup>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  auto pkg = md->GetMeshPointer()->packages.Get("multigroup_diffusion_package");
  const int ngroup = pkg->Param<int>("ngroup");

  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::entire, 0, pack.GetNBlocks(), md.get(), TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          pack_star(b, temperature0(), k, j, i) = pack(b, temperature(), k, j, i);
          for (int g = 0; g < ngroup; ++g) {
            pack_star(b, Egroup(g), k, j, i) = pack(b, Egroup(g), k, j, i);
            pack_star(b, TE::F1, Fgroup(g), k, j, i) =
                pack(b, TE::F1, Fgroup(g), k, j, i);
            pack_star(b, TE::F2, Fgroup(g), k, j, i) =
                pack(b, TE::F2, Fgroup(g), k, j, i);
            pack_star(b, TE::F3, Fgroup(g), k, j, i) =
                pack(b, TE::F3, Fgroup(g), k, j, i);
          }
        });
      });
  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::ApplyUpdate(
    std::shared_ptr<parthenon::MeshData<Real>> md_star,
    std::shared_ptr<parthenon::MeshData<Real>> md_u,
    std::shared_ptr<parthenon::MeshData<Real>> md_base) {
  using namespace parthenon;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using namespace MultiGroupVars;

  static const auto desc_u = parthenon::MakePackDescriptor<Egroup>(md_u.get());
  auto pack_u = desc_u.GetPack(md_u.get());

  static const auto desc_base =
      parthenon::MakePackDescriptor<temperature, Egroup, dTc, sigma, dSdT,
                                    ccbulk::internal_energy>(md_base.get());
  auto pack_base = desc_base.GetPack(md_base.get());

  static const auto desc_star =
      parthenon::MakePackDescriptor<temperature0>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  auto pkg = md_u->GetMeshPointer()->packages.Get("multigroup_diffusion_package");
  const int ngroup = pkg->Param<int>("ngroup");
  const Real a = pkg->Param<Real>("a_radiation");

  using lt = RiotUtils::LoopType<>;
  using TE = parthenon::TopologicalElement;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_u.GetNBlocks(),
                                     md_u.get(), TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          for (int g = 0; g < ngroup; ++g) {
            pack_base(b, Egroup(g), k, j, i) += pack_u(b, Egroup(g), k, j, i);
          }
        });
      });

  if (pkg->Param<bool>("update_temperature")) {
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            Real Told = pack_base(b, temperature(), k, j, i);
            Real T0 = pack_star(b, temperature0(), k, j, i);
            Real &T = pack_base(b, temperature(), k, j, i);

            Real fac = pack_base(b, dTc(), k, j, i);
            for (int g = 0; g < ngroup; ++g) {
              fac += pack_u(b, Egroup(g), k, j, i) * pack_base(b, sigma(g), k, j, i);
            }

            // Treat this as a logarithmic temperature update if it would push the
            // temperature to be negative
            if (fac < -T * 0.9) {
              T *= exp(fac / T);
            } else {
              T += fac;
            }
            Real scale = std::max(std::abs(T - T0), 1.e-2 * T);
            scale = std::max(scale, 1.e2);
            pack_base(b, dTc(), k, j, i) = std::pow((T - Told) / scale, 1);
          });
        });
  }
  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::LocalSolve(
    std::shared_ptr<parthenon::MeshData<Real>> md_star,
    std::shared_ptr<parthenon::MeshData<Real>> md_base, Real dt, int niter) {

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

  static const auto desc_base =
      parthenon::MakePackDescriptor<temperature, ccbulk::internal_energy, ccbulk::rho,
                                    ccmat::rho, cm::rho, ccmat::volume_fraction, Egroup,
                                    Fgroup, kappa_cell, kappa_face, dTc, dSdT, face_area,
                                    volume>(md_base.get());
  auto pack_base = desc_base.GetPack(md_base.get());

  static const auto desc_star =
      parthenon::MakePackDescriptor<temperature0, Egroup>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  const Real dtcl = dt * c_light;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_base.GetNBlocks(),
                                     md_base.get(), TE::CC);
  idx_space.template AddPerPointScratch<Real, MAX_GROUPS>(5); // aa, daadT, atot, S, dSdt
  idx_space.template AddPerPointScratch<Real>(4);             // denom, Egas, Egas0, RT
  for (int iter = 0; iter < niter; ++iter) {
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack_base.GetCoordinates(b);

          auto aa = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
          auto daadT = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
          auto atot = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
          mat_helpers.CalculateMultiGroupOpacities<temperature>(
              idx_range, pack_base, pack_base, b, MaterialHelpers::AlphaAbsMGScratch(aa),
              MaterialHelpers::AlphaTotMGScratch(atot),
              MaterialHelpers::dAlphaAbsMGdTScratch(daadT));
          mat_helpers.CalculateMultiGroupOpacities<temperature0>(
              idx_range, pack_base, pack_star, b,
              MaterialHelpers::AlphaTotMGScratch(atot));

          auto S = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
          auto dSdt = RiotLoop::GetPerPointScratch<Real, MAX_GROUPS>(idx_range);
          source_helper.CalculateSource<temperature>(idx_range, pack_base, pack_base, aa,
                                                     daadT, b, S, dSdt);

          // denom is seeded with the total heat capacity (Cv).
          auto denom = RiotLoop::GetPerPointScratch<Real>(idx_range);
          auto Egas = RiotLoop::GetPerPointScratch<Real>(idx_range);
          mat_helpers.CalculateEos<temperature>(idx_range, pack_base, pack_base, b,
                                                MaterialHelpers::CvScratch(denom),
                                                MaterialHelpers::EgasScratch(Egas));

          auto Egas0 = RiotLoop::GetPerPointScratch<Real>(idx_range);
          mat_helpers.CalculateEos<temperature0>(idx_range, pack_base, pack_star, b,
                                                 MaterialHelpers::EgasScratch(Egas0));

          // RT, denom accumulation, the first Egroup correction, the temperature update,
          // and the final Egroup correction are all cell-local, so they fuse into one
          // inner pass over (k, j, i).
          idx_range.TeamBarrier();
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real AreaX1p = pack_base(b, TE::F1, face_area(), k, j, i + 1);
            const Real AreaX1m = pack_base(b, TE::F1, face_area(), k, j, i);
            const Real AreaX2p = pack_base(b, TE::F2, face_area(), k, j + (ndim > 1), i);
            const Real AreaX2m = pack_base(b, TE::F2, face_area(), k, j, i);
            const Real AreaX3p = pack_base(b, TE::F3, face_area(), k + (ndim > 2), j, i);
            const Real AreaX3m = pack_base(b, TE::F3, face_area(), k, j, i);
            const Real iVol = pack_base(b, volume(), k, j, i);

            Real RT = Egas(k, j, i) - Egas0(k, j, i);
            for (int g = 0; g < ngroup; ++g) {
              // Calculate the residual for this group
              Real REg = pack_base(b, Egroup(g), k, j, i) -
                         pack_star(b, Egroup(g), k, j, i) - S(g, k, j, i);
              Real flux_div{0.0};
              flux_div += dtcl *
                          (pack_base(b, TE::F1, Fgroup(g), k, j, i + 1) * AreaX1p -
                           pack_base(b, TE::F1, Fgroup(g), k, j, i) * AreaX1m) *
                          iVol;
              flux_div +=
                  dtcl *
                  (pack_base(b, TE::F2, Fgroup(g), k, j + (ndim > 1), i) * AreaX2p -
                   pack_base(b, TE::F2, Fgroup(g), k, j, i) * AreaX2m) *
                  iVol;
              flux_div +=
                  dtcl *
                  (pack_base(b, TE::F3, Fgroup(g), k + (ndim > 2), j, i) * AreaX3p -
                   pack_base(b, TE::F3, Fgroup(g), k, j, i) * AreaX3m) *
                  iVol;
              REg += flux_div;

              // Contribute to the temperature residual and denominator
              const Real tau = dtcl * aa(g, k, j, i);
              RT += S(g, k, j, i) + tau / (1.0 + tau) * REg;
              denom(k, j, i) += dSdt(g, k, j, i) / (1.0 + tau);

              // Add the first correction to Egroup
              pack_base(b, Egroup(g), k, j, i) -= REg / (1.0 + tau);
            }

            const Real dT = -RT / denom(k, j, i);
            Real &T = pack_base(b, temperature(), k, j, i);
            if (dT < -T * 0.9) {
              T *= exp(dT / T);
            } else {
              T += dT;
            }

            for (int g = 0; g < ngroup; ++g) {
              const Real tau = dtcl * aa(g, k, j, i);
              pack_base(b, Egroup(g), k, j, i) += dSdt(g, k, j, i) / (1.0 + tau) * dT;
            }
          });
        });
  }
  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::CorrectTotalEnergy(
    std::shared_ptr<parthenon::MeshData<Real>> md_base) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pmesh = md_base->GetMeshPointer();

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const int ngroup = pkg->Param<int>("ngroup");

  // Select which internal energy to update based on the type of the temperature, either
  // the electron internal energy or the ion internal energy
  using internal_energy =
      std::conditional_t<std::is_same_v<temperature, ccbulk::temperature>,
                         ccbulk::internal_energy, ccbulk::electron_internal_energy>;

  static const auto desc_base =
      parthenon::MakePackDescriptor<kappa_face, Fgroup, Egroup, temperature, ccbulk::rho,
                                    internal_energy, ccbulk::total_material_energy,
                                    ccmat::rho, cm::rho>(md_base.get());
  auto pack_base = desc_base.GetPack(md_base.get());

  MaterialHelpers mat_helpers(md_base->GetMeshPointer(), "multigroup_diffusion_package",
                              temperature());

  // Interior plus one ghost layer, matching the old RawMemoryIndexer halo of 1.
  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 1, pack_base.GetNBlocks(),
                                     md_base.get(), TE::CC);
  idx_space.template AddPerPointScratch<Real>(1); // eint_new
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto eint_new = RiotLoop::GetPerPointScratch<Real>(idx_range);
        mat_helpers.CalculateEos<temperature>(idx_range, pack_base, pack_base, b,
                                              MaterialHelpers::EgasScratch(eint_new));

        idx_range.TeamBarrier();
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          // Update the conserved energy density
          Real &eint = pack_base(b, internal_energy(), k, j, i);
          pack_base(b, ccbulk::total_material_energy(), k, j, i) +=
              eint_new(k, j, i) - eint;
          eint = eint_new(k, j, i);
        });
      });

  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::SetLaggedRadiationMomentumFlux(
    std::shared_ptr<parthenon::MeshData<Real>> md_star,
    std::shared_ptr<parthenon::MeshData<Real>> md_base, Real dt) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pmesh = md_base->GetMeshPointer();

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const Real c_light = pkg->Param<Real>("c_light");
  const Real dtcl = c_light * dt;
  const int ngroup = pkg->Param<int>("ngroup");

  static const auto desc_base = parthenon::MakePackDescriptor<kappa_face, Fgroup, Egroup>(
      md_base.get(), {}, {PDOpt::WithFluxes});
  auto pack_base = desc_base.GetPack(md_base.get());
  static const auto desc_star = parthenon::MakePackDescriptor<Fgroup>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  for (int dim = 0; dim < pmesh->ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    const int ioff = dim == 0;
    const int joff = dim == 1;
    const int koff = dim == 2;

    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_base.GetNBlocks(),
                                       md_base.get(), te);
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            for (int g = 0; g < ngroup; ++g) {
              pack_base.flux(b, dim + 1, Egroup(g), k, j, i) =
                  pack_star(b, te, Fgroup(g), k, j, i) * dtcl /
                  (1.0 + dtcl * pack_base(b, te, kappa_face(g), k, j, i));
            }
          });
        });
  }
  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::UpdateRadiationMomentum(
    std::shared_ptr<parthenon::MeshData<Real>> md_star,
    std::shared_ptr<parthenon::MeshData<Real>> md_base, Real dt) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pmesh = md_base->GetMeshPointer();

  auto pkg = pmesh->packages.Get("multigroup_diffusion_package");
  const Real c_light = pkg->Param<Real>("c_light");
  const Real dtcl = c_light * dt;
  const int ngroup = pkg->Param<int>("ngroup");

  static const auto desc_base = parthenon::MakePackDescriptor<kappa_face, Fgroup, Egroup>(
      md_base.get(), {}, {PDOpt::WithFluxes});
  auto pack_base = desc_base.GetPack(md_base.get());
  static const auto desc_star = parthenon::MakePackDescriptor<Fgroup>(md_star.get());
  auto pack_star = desc_star.GetPack(md_star.get());

  for (int dim = 0; dim < pmesh->ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    const int ioff = dim == 0;
    const int joff = dim == 1;
    const int koff = dim == 2;

    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, pack_base.GetNBlocks(),
                                       md_base.get(), te);
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack_base.GetCoordinates(b);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real dx = coords.Dxc(dim + 1, k, j, i);
            for (int g = 0; g < ngroup; ++g) {
              // Remember that the "diffusion coefficients" include an extra factor of c *
              // dt
              Real Fdiff = pack_base.flux(b, dim + 1, Egroup(g), k, j, i) / dtcl;
              // Real Fdiff = - dtcl * (pack_base(b, Egroup(g), k, j, i) - pack_base(b,
              // Egroup(g), k - koff, j - joff, i - ioff)) / (3.0 * dx);
              pack_base(b, te, Fgroup(g), k, j, i) = pack_star(b, te, Fgroup(g), k, j, i);
              pack_base(b, te, Fgroup(g), k, j, i) /=
                  1.0 + dtcl * pack_base(b, te, kappa_face(g), k, j, i);
              pack_base(b, te, Fgroup(g), k, j, i) += Fdiff;
            }
          });
        });
  }
  return TaskStatus::complete;
}

template <class temperature>
parthenon::TaskStatus MultiGroupTasks<temperature>::InitializeGeometry(
    std::shared_ptr<parthenon::MeshData<Real>> &md_mat) {
  using namespace parthenon;
  using namespace MultiGroupVars;
  const int ndim = md_mat->GetMeshPointer()->ndim;
  using TE = parthenon::TopologicalElement;
  using namespace RadiationDiffusion::MultiGroupVars;
  static auto desc =
      parthenon::MakePackDescriptor<face_area, DeltaX, volume>(md_mat.get());
  auto pack = desc.GetPack(md_mat.get());
  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::entire, 0, pack.GetNBlocks(), md_mat.get(), TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto &coords = pack.GetCoordinates(b);
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          pack(b, volume(), k, j, i) = 1.0 / coords.CellVolume(k, j, i);
          pack(b, TE::F1, face_area(), k, j, i) =
              coords.template FaceArea<X1DIR>(k, j, i);
          pack(b, TE::F2, face_area(), k, j, i) =
              coords.template FaceArea<X2DIR>(k, j, i);
          pack(b, TE::F3, face_area(), k, j, i) =
              coords.template FaceArea<X3DIR>(k, j, i);
          pack(b, TE::F1, DeltaX(), k, j, i) = 1.0 / coords.template Dxc<X1DIR>(k, j, i);
          pack(b, TE::F2, DeltaX(), k, j, i) = 1.0 / coords.template Dxc<X2DIR>(k, j, i);
          pack(b, TE::F3, DeltaX(), k, j, i) = 1.0 / coords.template Dxc<X3DIR>(k, j, i);
        });
      });
  return TaskStatus::complete;
}

template struct MultiGroupTasks<cell_variables::cell_averaged::bulk::temperature>;
template struct MultiGroupTasks<
    cell_variables::cell_averaged::bulk::electron_temperature>;

} // namespace RadiationDiffusion
