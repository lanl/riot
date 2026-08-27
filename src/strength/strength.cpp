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

#include <memory>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
using parthenon::ParArray1D;
using parthenon::ScratchPad1D;
using parthenon::team_mbr_t;

#include "hydro/hydro.hpp"
#include "microphysics/strength_models.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "strength/strength.hpp"
#include "variables.hpp"

namespace Strength {

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Strength::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            StateDescriptor *mat_pkg) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto pkg = std::make_shared<StateDescriptor>("strength");
  pkg->FillDerivedMesh = FillDerived;

  // Source term dU/dt variables: deviatoric stress on strength materials
  auto strength_mats = mat_pkg->Param<std::vector<int>>("strength_mats");
  pkg->RegisterMeshDataSubset(
      "dudt", RiotUtils::MakePackageDudtRequirements({ccmat::deviatoric_stress::name()},
                                                     {}, strength_mats));

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Strength::CalculateStrengthSource
//! \brief
TaskStatus CalculateStrengthSource(MeshData<Real> *state, MeshData<Real> *src) {

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = state->GetParentPointer();
  auto &materials = pm->packages.Get("materials");
  auto &strength_mats = materials->Param<std::vector<int>>("strength_mats");

  // Pack up sparse strength fields
  auto vm = riot::MakePack<ccmat::rho, cm::deviatoric_stress, cm::shear_modulus>(
      state, strength_mats);

  // Pack up bulk fields
  auto vb = riot::MakePack<ccbulk::face_velocity, ccbulk::strain_rate>(state);

  // Pack up deviatoric stresses for dU register
  auto dv = riot::MakePack<ccmat::deviatoric_stress>(src, strength_mats);

  // don't launch kernel if there are no blocks or strength materials
  const int nblocks = vb.GetNBlocks();
  if (nblocks == 0 || dv.GetMaxNumberOfVars() == 0) return TaskStatus::complete;

  // Calculate strain rate
  Hydro::CalculateStrainRate(state, vb);

  const int ndim = pm->ndim;
  const int dj = (ndim > 1);
  const int dk = (ndim > 2);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state,
                                     parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(3);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto wxy = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto wxz = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto wyz = RiotLoop::GetPerPointScratch<Real>(idx_range);

        auto &coords = vm.GetCoordinates(b);
        // assume these are constants
        const Real dx = coords.Dx<X1DIR>();
        const Real dy = coords.Dx<X2DIR>();
        const Real dz = coords.Dx<X3DIR>();

        auto pvb = RiotLoop::make_pack_view(idx_range, vb);
        auto pvm = RiotLoop::make_pack_view(idx_range, vm);
        auto pdv = RiotLoop::make_pack_view(idx_range, dv);
        const int nstr = vm.GetSize(b, ccmat::rho());

        // first get the rotation tensor
        RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
          const Real dvyx = (pvb(ccbulk::face_velocity(1), k, j, i + 1) -
                             pvb(ccbulk::face_velocity(1), k, j, i)) /
                            dx;
          const Real dvzx = (pvb(ccbulk::face_velocity(2), k, j, i + 1) -
                             pvb(ccbulk::face_velocity(2), k, j, i)) /
                            dx;
          const Real dvxy = (pvb(ccbulk::face_velocity(3), k, j + dj, i) -
                             pvb(ccbulk::face_velocity(3), k, j, i)) /
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

          wxy(k, j, i) = 0.5 * (dvxy - dvyx);
          wxz(k, j, i) = 0.5 * (dvxz - dvzx);
          wyz(k, j, i) = 0.5 * (dvyz - dvzy);
        });

        idx_range.TeamBarrier();
        for (int n = 0; n < nstr; n++) {
          auto pvm_n = RiotLoop::make_sparse_pack_view(idx_range, vm, n);
          auto pdv_n = RiotLoop::make_sparse_pack_view(idx_range, dv, n);
          RiotLoop::inner(idx_range, [&](const auto k, const auto j, const auto i) {
            const Real rho = pvm_n(ccmat::rho(), k, j, i);
            const Real mu = pvm_n(cm::shear_modulus(), k, j, i);
            const Real sxx = pvm_n(cm::deviatoric_stress(0), k, j, i);
            const Real sxy = pvm_n(cm::deviatoric_stress(1), k, j, i);
            const Real sxz = pvm_n(cm::deviatoric_stress(2), k, j, i);
            const Real syy = pvm_n(cm::deviatoric_stress(3), k, j, i);
            const Real syz = pvm_n(cm::deviatoric_stress(4), k, j, i);
            const Real exx = pvb(ccbulk::strain_rate(0), k, j, i);
            const Real exy = pvb(ccbulk::strain_rate(1), k, j, i);
            const Real exz = pvb(ccbulk::strain_rate(2), k, j, i);
            const Real eyy = pvb(ccbulk::strain_rate(3), k, j, i);
            const Real eyz = pvb(ccbulk::strain_rate(4), k, j, i);

            // diagonal components
            pdv_n(ccmat::deviatoric_stress(0), k, j, i) =
                2.0 * rho * (mu * exx + sxy * wxy(k, j, i) + sxz * wxz(k, j, i));
            pdv_n(ccmat::deviatoric_stress(3), k, j, i) =
                2.0 * rho * (mu * eyy + syz * wyz(k, j, i) - sxy * wxy(k, j, i));
            // off-diagonal components
            pdv_n(ccmat::deviatoric_stress(1), k, j, i) =
                rho * (2.0 * mu * exy + syz * wxz(k, j, i) + sxz * wyz(k, j, i) +
                       (syy - sxx) * wxy(k, j, i));
            pdv_n(ccmat::deviatoric_stress(2), k, j, i) =
                rho * (2.0 * mu * exz + syz * wxy(k, j, i) - sxy * wyz(k, j, i) -
                       (2.0 * sxx + syy) * wxz(k, j, i));
            pdv_n(ccmat::deviatoric_stress(4), k, j, i) =
                rho * (2.0 * mu * eyz - sxz * wxy(k, j, i) - sxy * wxz(k, j, i) -
                       (sxx + 2.0 * syy) * wyz(k, j, i));
          });
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Strength::RadialReturn
//! \brief Execute radial return for strength calculations
//! NOTE(@jonahm): This function is called immediately after update, and before any PTE
//! calls, hence, we end up working on lagged "primitive" variables like the density
//! and temperature (which may affect some strength models). This choice eliminates an
//! additional PTE call.
//! NOTE(@pdmullen): What are the consequences of the failure model not being informed by
//! materials "removed" in PTE calls or materials masked in PostCommsFillDerived?
TaskStatus RadialReturn(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();

  auto materials = pm->packages.Get("materials");
  const bool use_general_pte = materials->Param<bool>("use_general_pte");
  auto &strength_mats = materials->Param<std::vector<int>>("strength_mats");
  auto &strength_models =
      materials->Param<ParArray1D<Strength::StressModel>>("d.strength_models");
  auto &strength_model_ids =
      materials->Param<ParArray1D<Strength::stress_model>>("d.strength_model_ids");
  auto &strength_map = materials->Param<parthenon::ParArray1D<int>>("d.strength_map");

  auto vm = riot::MakePack<ccmat::rho, ccmat::volume_fraction, ccmat::deviatoric_stress,
                           ccmat::equivalent_plastic_strain, cm::strength_j2,
                           cm::pressure, cm::temperature>(md, strength_mats);
  auto vb = riot::MakePack<ccbulk::total_material_energy>(md);
  const int nblocks = vb.GetNBlocks();

  // don't launch kernel if there are no strength materials
  if (vm.GetMaxNumberOfVars() == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(1);

  // now we'll do radial return if there are strength materials
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nummat = vm.GetSize(b, ccmat::rho());

        // Per-cell energy change accumulated over materials; zero before the loop.
        auto du = RiotLoop::GetPerPointScratch<Real>(idx_range);
        du.Zero();
        idx_range.TeamBarrier();

        auto prim_cons = [](const Real fac, Real &sxx, Real &sxy, Real &sxz, Real &syy,
                            Real &syz, Real &eps) {
          sxx *= fac;
          sxy *= fac;
          sxz *= fac;
          syy *= fac;
          syz *= fac;
          eps *= fac;
        };
        auto to_prim = [](const Real rho) {
          return (rho > 1.0e-100) / (std::max(rho, 1.0e-100) + (rho <= 1.0e-100));
        };
        auto to_cons = [](const Real rho) { return std::max(rho, 0.0); };

        for (int m = 0; m < nummat; m++) {
          auto pvm_m = RiotLoop::make_sparse_pack_view(idx_range, vm, m);
          const int sid = vm(b, ccmat::rho(m)).sparse_id;
          const int str_idx = strength_map(sid);
          auto model_type = strength_model_ids(str_idx);
          auto &model = strength_models(str_idx);

          switch (model_type) {
          case Strength::stress_model::epp: {
            auto &epp = model.get<Strength::EPP>();
            RiotLoop::inner(idx_range, [&](const auto kji) {
              // Live references into grid memory: prim_cons / radial_return mutate the
              // deviatoric stress, plastic strain, and j2 in place (no copy-in/out).
              const Real ccmat_rho = pvm_m(ccmat::rho(), kji);
              Real &rsxx = pvm_m(ccmat::deviatoric_stress(0), kji);
              Real &rsxy = pvm_m(ccmat::deviatoric_stress(1), kji);
              Real &rsxz = pvm_m(ccmat::deviatoric_stress(2), kji);
              Real &rsyy = pvm_m(ccmat::deviatoric_stress(3), kji);
              Real &rsyz = pvm_m(ccmat::deviatoric_stress(4), kji);
              Real &reps = pvm_m(ccmat::equivalent_plastic_strain(), kji);
              Real &j2 = pvm_m(cm::strength_j2(), kji);

              prim_cons(to_prim(ccmat_rho), rsxx, rsxy, rsxz, rsyy, rsyz, reps);
              du(kji) += Strength::radial_return(rsxx, rsxy, rsxz, rsyy, rsyz, reps, epp,
                                                 j2, ccmat_rho);
              const Real fail = model.failure(ccmat_rho);
              prim_cons(fail * to_cons(ccmat_rho), rsxx, rsxy, rsxz, rsyy, rsyz, reps);
            });
          } break;
          // case Strength::stress_model::sg: {
          //   auto sg = model.get<Strength::SG>();
          //   RiotLoop::inner(idx_range, [&](const auto kji) {
          //     const Real ccmat_rho = pvm_m(ccmat::rho(), kji);
          //     const Real vfrac = pvm_m(ccmat::volume_fraction(), kji);
          //     const Real press = pvm_m(cm::pressure(), kji);
          //     const Real temp = pvm_m(cm::temperature(), kji);
          //     Real &rsxx = pvm_m(ccmat::deviatoric_stress(0), kji);
          //     Real &rsxy = pvm_m(ccmat::deviatoric_stress(1), kji);
          //     Real &rsxz = pvm_m(ccmat::deviatoric_stress(2), kji);
          //     Real &rsyy = pvm_m(ccmat::deviatoric_stress(3), kji);
          //     Real &rsyz = pvm_m(ccmat::deviatoric_stress(4), kji);
          //     Real &reps = pvm_m(ccmat::equivalent_plastic_strain(), kji);
          //     Real &j2 = pvm_m(cm::strength_j2(), kji);
          //     const Real gmod = pvm_m(cm::shear_modulus(), kji);
          //     prim_cons(to_prim(ccmat_rho), rsxx, rsxy, rsxz, rsyy, rsyz, reps);
          //     const Real rho_phys = ccmat_rho / (vfrac + (vfrac == 0.0));
          //     du(kji) += Strength::radial_return(rsxx, rsxy, rsxz, rsyy, rsyz, reps,
          //     sg,
          //                                        gmod, j2, ccmat_rho, rho_phys, press,
          //                                        temp, reps);
          //     prim_cons(to_cons(ccmat_rho), rsxx, rsxy, rsxz, rsyy, rsyz, reps);
          //   });
          // } break;
          default:
            PARTHENON_FAIL("Strength model not implemented.");
          }
          idx_range.TeamBarrier();
        }

        // Accumulate du into total energy
        auto pvb = RiotLoop::make_pack_view(idx_range, vb);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pvb(ccbulk::total_material_energy(), kji) += du(kji);
        });
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Strength::FillDerived
//! \brief
void FillDerived(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  Mesh *pm = md->GetMeshPointer();
  auto &materials = pm->packages.Get("materials");
  auto &strength_models =
      materials->Param<parthenon::ParArray1D<Strength::StressModel>>("d.strength_models");
  auto &strength_model_ids =
      materials->Param<parthenon::ParArray1D<Strength::stress_model>>(
          "d.strength_model_ids");
  auto &strength_map = materials->Param<parthenon::ParArray1D<int>>("d.strength_map");

  auto v = riot::MakePack<ccmat::rho, ccbulk::shear_modulus, ccmat::volume_fraction,
                          ccmat::deviatoric_stress, ccmat::equivalent_plastic_strain,
                          cm::deviatoric_stress, cm::equivalent_plastic_strain,
                          cm::shear_modulus>(md);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        const int nmat = v.GetSize(b, ccmat::rho());

        // Set bulk shear modulus to zero. Will accumulate per material below.
        RiotLoop::inner(idx_range,
                        [&](const auto kji) { pv(ccbulk::shear_modulus(), kji) = 0.0; });

        int nstr = 0;
        for (int n = 0; n < nmat; n++) {
          const int sid = v(b, ccmat::rho(n)).sparse_id;
          if (strength_map(sid) >= 0) {
            const int str_idx = strength_map(sid);
            auto model_type = strength_model_ids(str_idx);
            auto &model = strength_models(str_idx);

            auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
            auto pv_nstr = RiotLoop::make_sparse_pack_view(idx_range, v, nstr);

            // Convert between ccmat and cm stress/strain
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ccmat_rho = pv_n(ccmat::rho(), kji);
              const Real u2p = ((ccmat_rho > 0.0) / (ccmat_rho + (ccmat_rho <= 0.0)));

              // ccmat -> cm (divide by rho)
              pv_nstr(cm::deviatoric_stress(0), kji) =
                  pv_nstr(ccmat::deviatoric_stress(0), kji) * u2p;
              pv_nstr(cm::deviatoric_stress(1), kji) =
                  pv_nstr(ccmat::deviatoric_stress(1), kji) * u2p;
              pv_nstr(cm::deviatoric_stress(2), kji) =
                  pv_nstr(ccmat::deviatoric_stress(2), kji) * u2p;
              pv_nstr(cm::deviatoric_stress(3), kji) =
                  pv_nstr(ccmat::deviatoric_stress(3), kji) * u2p;
              pv_nstr(cm::deviatoric_stress(4), kji) =
                  pv_nstr(ccmat::deviatoric_stress(4), kji) * u2p;
              pv_nstr(cm::equivalent_plastic_strain(), kji) =
                  pv_nstr(ccmat::equivalent_plastic_strain(), kji) * u2p;

              // cm -> ccmat (multiply by rho)
              pv_nstr(ccmat::deviatoric_stress(0), kji) =
                  pv_nstr(cm::deviatoric_stress(0), kji) * ccmat_rho;
              pv_nstr(ccmat::deviatoric_stress(1), kji) =
                  pv_nstr(cm::deviatoric_stress(1), kji) * ccmat_rho;
              pv_nstr(ccmat::deviatoric_stress(2), kji) =
                  pv_nstr(cm::deviatoric_stress(2), kji) * ccmat_rho;
              pv_nstr(ccmat::deviatoric_stress(3), kji) =
                  pv_nstr(cm::deviatoric_stress(3), kji) * ccmat_rho;
              pv_nstr(ccmat::deviatoric_stress(4), kji) =
                  pv_nstr(cm::deviatoric_stress(4), kji) * ccmat_rho;
              pv_nstr(ccmat::equivalent_plastic_strain(), kji) =
                  pv_nstr(cm::equivalent_plastic_strain(), kji) * ccmat_rho;
            });

            // Compute shear modulus
            switch (model_type) {
            case Strength::stress_model::epp: {
              auto &epp = model.get<Strength::EPP>();
              RiotLoop::inner(idx_range, [&](const auto kji) {
                const Real ccmat_rho = pv_n(ccmat::rho(), kji);
                pv_nstr(cm::shear_modulus(), kji) = epp.shear_modulus(ccmat_rho);
              });
            } break;
            // case Strength::stress_model::sg: {
            //   auto sg = model.get<Strength::SG>();
            //   RiotLoop::inner(idx_range, [&](const auto kji) {
            //     const Real ccmat_rho = pv_n(ccmat::rho(), kji);
            //     const Real press = pv_n(cm::pressure(), kji);
            //     const Real temp = pv_n(cm::temperature(), kji);
            //     pv_nstr(cm::shear_modulus(), kji) =
            //         sg.shear_modulus(ccmat_rho, press, temp);
            //   });
            // } break;
            default:
              PARTHENON_FAIL("Strength model not implemented.");
            }

            // Accumulate into bulk shear modulus
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real vfrac = pv_n(ccmat::volume_fraction(), kji);
              const Real gmod = pv_nstr(cm::shear_modulus(), kji);
              pv(ccbulk::shear_modulus(), kji) += vfrac * gmod;
            });

            nstr++;
          }
        }
      });

  return;
}

} // namespace Strength
