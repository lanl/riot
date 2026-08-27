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

#include "fill_shared_derived.hpp"

#include <string>

#include <parthenon/package.hpp>
#include <utils/index_split.hpp>
using namespace parthenon::package::prelude;
using parthenon::IndexSplit;

#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/eos/eos.hpp>

// RIOT includes
#include "hydro/hydro.hpp"
#include "ionization/ionization.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/pte_closure.hpp"
#include "microphysics/strength_models.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Multiphysics {

//----------------------------------------------------------------------------------------
//! \fn  void Multiphysics::FillInteriorDerived
//! \brief FillInteriorDerived is the first step in a two part saga of setting all derived
//! variables following either (a) an integration stage or (b) a remeshing event.
//! FillInteriorDerived's goal is twofold: (1) to *set* fields over IndexDomain::interior
//! that *need to be communicated* *using* Metadata::Independent fields and/or
//! Metadata::ForceRemeshComm fields and (2) set the PTE pressure, vfracs, and temperature
//! over IndexDomain::interior.  (Aside: recall that general PTE requires volume fraction
//! guesses set in Hydro::GuessVolumeFractions and material temperatures from the previous
//! RK cycle...). The only Metadata::FillGhost fields are the material densities, volume
//! fractions, bulk velocity, bulk temperature, and bulk pressure. In the following, we
//! first compute the bulk density (needed for PTE *and* velocity computation). Then we
//! compute the bulk velocity needed for communication. Then we compute the bulk internal
//! energy needed for PTE. Then we execute PTE, thereby setting temperature, pressure, and
//! vfracs (all needed for communication) over IndexDomain::interior.
void FillInteriorDerived(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto riot = pm->packages.Get("riot");
  auto materials = pm->packages.Get("materials");
  const bool do_ionization = riot->Param<bool>("do_ionization");
  const bool use_general_pte = materials->Param<bool>("use_general_pte");

  auto v = riot::MakePack<ccmat::rho, ccbulk::rho, ccbulk::momentum, ccbulk::velocity,
                          ccbulk::total_material_energy, ccbulk::internal_energy,
                          ccbulk::electron_internal_energy, ccbulk::pressure,
                          ccbulk::electron_pressure, ccmat::ionization_zbar,
                          cm::ionization_zbar>(md);

  if (v.GetNBlocks() == 0) return;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        const int nummat = v.GetSize(b, ccmat::rho());

        // Compute bulk density (flooring material densities)
        for (int m = 0; m < nummat; ++m) {
          auto pv_sp = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv(ccbulk::rho(), kji) =
                (m == 0)
                    ? std::max(pv_sp(ccmat::rho(), kji), 0.0)
                    : pv(ccbulk::rho(), kji) + std::max(pv_sp(ccmat::rho(), kji), 0.0);
          });
          idx_range.TeamBarrier();
        }

        // Calculate bulk velocity from conserved bulk momentum and bulk internal energy
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const Real irho = 1.0 / (pv(ccbulk::rho(), kji) + 1.e-100);
          pv(ccbulk::velocity(0), kji) = pv(ccbulk::momentum(0), kji) * irho;
          pv(ccbulk::velocity(1), kji) = pv(ccbulk::momentum(1), kji) * irho;
          pv(ccbulk::velocity(2), kji) = pv(ccbulk::momentum(2), kji) * irho;
          pv(ccbulk::internal_energy(), kji) =
              pv(ccbulk::total_material_energy(), kji) -
              0.5 * pv(ccbulk::rho(), kji) *
                  (SQR(pv(ccbulk::velocity(0), kji)) + SQR(pv(ccbulk::velocity(1), kji)) +
                   SQR(pv(ccbulk::velocity(2), kji)));
        });

        if (do_ionization) {
          idx_range.TeamBarrier();
          // Need to use ion energy and pressure for PTE
          Ionization::ConvertEnergyPressureBulkIon<Ionization::ToIon>(idx_range, v, b);
          for (int m = 0; m < nummat; m++) {
            auto pv_sp = RiotLoop::make_sparse_pack_view(idx_range, v, m);
            RiotLoop::inner(idx_range, [&](const auto kji) {
              pv_sp(cm::ionization_zbar(), kji) = (pv_sp(ccmat::rho(), kji) > 0.0) *
                                                  pv_sp(ccmat::ionization_zbar(), kji) /
                                                  (pv_sp(ccmat::rho(), kji) + 1.e-100);
            });
          }
        }
      });

  // Now that material densities and internal energies are set, pass system through PTE
  if (use_general_pte) {
    Closure::ApplyMixedCellClosure(md, IndexDomain::interior);
  } else {
    Closure::ApplyIdealGasClosure(md, IndexDomain::interior);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Multiphysics::PostCommsFillDerived
//!  \brief PostCommsFillDerived is the second step in a two part saga of setting all
//!  derived variables following either (a) an integration stage or (b) a remeshing event.
//!  PostCommsFillDerived's goal is twofold: (1) ensure that all derived fields are set
//!  over IndexDomain::entire and (2) ensure that fields are synced^*** (e.g., material
//!  densities sum to bulk density, momentum is product of synced bulk density and
//!  (potentially communicated velocity), etc.).  Following FillInteriorDerived and the
//!  communications step, we should now have material densities, volume fractions,
//!  bulk velocity, bulk temperature, and bulk pressure over IndexDomain::entire (recall
//!  that these fields have Metadata::FillGhost and could have potentially been
//!  prolongated/restricted at fine/coarse boundaries).  We here set all remaining derived
//!  fields over IndexDomain::entire.  In this operation, we mask fields according to mass
//!  fraction and volume fraction thresholds.  The masking operation has the opportunity
//!  to get Metadata::Independent (or bulk) fields out of sync, therefore, we conclude by
//!  passing over IndexDomain::entire, to ensure all fields are in sync.
//!
//!  ***Albeit we take multiple steps to ensure that fields are synced, one notable
//!  exception is that we have not ensure that prolongated/restricted cells at fine/coarse
//!  boundaries are still valid PTE solutions, i.e., the prolongated/restricted
//!  ccmat::rho, ccmat::volume_fraction, ccbulk::temperature, and ccbulk::pressure may not
//!  correspond to equilibrium solutions.  Since these values are only used in
//!  reconstruction stencils, it is our hope that this does not spoil the solution --- we
//!  do ensure that volume fractions sum to 1 following the FillGhost operation.
void PostCommsFillDerived(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using RiotEOS::EOS_Array_t;

  auto pm = md->GetParentPointer();
  auto &riot = pm->packages.Get("riot");

  auto &materials = pm->packages.Get("materials");
  const auto &eos = materials->Param<EOS_Array_t>("d.d.EOS");
  const auto &eos_from_matid =
      materials->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  const bool do_ionization = riot->Param<bool>("do_ionization");
  EOS_Array_t electron_eos;
  Real Te_root_tol;
  if (do_ionization) {
    const auto &ion_pkg = pm->packages.Get("ionization");
    Te_root_tol = ion_pkg->Param<Real>("Te_root_tol");
    electron_eos = materials->Param<EOS_Array_t>("d.d.electron_EOS");
  }

  auto &hydro = pm->packages.Get("hydro");
  const Real mass_frac_thresh = hydro->Param<Real>("mass_frac_thresh");
  const Real vol_frac_thresh = hydro->Param<Real>("vol_frac_thresh");

  auto v =
      riot::MakePack<ccmat::rho, ccmat::volume_fraction, ccmat::internal_energy,
                     ccmat::electron_internal_energy, cm::ionization_zbar, cm::rho,
                     cm::sie, cm::temperature, cm::pressure, cm::bulk_modulus,
                     cm::specific_heat, cm::electron_sie, cm::lT_cache, cm::lr_cache,
                     ccbulk::rho, ccbulk::momentum, ccbulk::total_material_energy,
                     ccbulk::velocity, ccbulk::internal_energy, ccbulk::pressure,
                     ccbulk::bulk_modulus, ccbulk::temperature,
                     ccbulk::electron_internal_energy, ccbulk::electron_temperature,
                     ccbulk::electron_pressure, ccbulk::electron_number_density,
                     ccbulk::electron_bulk_modulus, ccbulk::electron_gruneisen_parameter>(
          md);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(1);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto vfrac_sum = RiotLoop::GetPerPointScratch<Real>(idx_range);
        const int nmat = v.GetSize(b, ccmat::rho());

        // Renormalize vfrac and recompute bulk density over IndexDomain::entire
        for (int n = 0; n < nmat; n++) {
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real vf = pv_n(ccmat::volume_fraction(), kji);
            pv_n(ccmat::volume_fraction(), kji) = (vf > vol_frac_thresh) ? vf : 0.0;
            vfrac_sum(kji) = (n == 0)
                                 ? pv_n(ccmat::volume_fraction(), kji)
                                 : (vfrac_sum(kji) + pv_n(ccmat::volume_fraction(), kji));
          });
          idx_range.TeamBarrier();
        }
        for (int n = 0; n < nmat; n++) {
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv(ccbulk::rho(), kji) =
                (n == 0)
                    ? std::max(pv_n(ccmat::rho(), kji), 0.0)
                    : (pv(ccbulk::rho(), kji) + std::max(pv_n(ccmat::rho(), kji), 0.0));
            pv_n(ccmat::volume_fraction(), kji) /= std::max(vfrac_sum(kji), 1.0e-16);
          });
          idx_range.TeamBarrier();
        }

        // Mask fields of materials below mass fraction and volume fraction thresholds
        // Compute remaining bulk fields
        for (int n = 0; n < nmat; n++) {
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
          const int mat_id = v(b, ccmat::rho(n)).sparse_id;
          const int phase_id = v(b, ccmat::rho(n)).v;
          auto &eosm = eos(eos_from_matid(mat_id) + phase_id);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            // Set rho/mass_frac/vol_frac threshold mask
            const Real ccmat_rho_val = pv_n(ccmat::rho(), kji);
            const Real vfrac_val = pv_n(ccmat::volume_fraction(), kji);
            const Real mask =
                1.0 * (ccmat_rho_val > mass_frac_thresh * pv(ccbulk::rho(), kji) &&
                       vfrac_val > vol_frac_thresh);
            // Mask materials if necessary
            pv_n(ccmat::rho(), kji) = ccmat_rho_val * mask;
            pv_n(ccmat::volume_fraction(), kji) = vfrac_val * mask;
            pv_n(cm::rho(), kji) = mask * ccmat_rho_val / (vfrac_val + 1.e-16);
            pv_n(cm::temperature(), kji) = mask * pv(ccbulk::temperature(), kji);
            pv_n(cm::pressure(), kji) = mask * pv(ccbulk::pressure(), kji);
          });
          idx_range.TeamBarrier();

          auto eosmloop = [&](const auto &eosm_c) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              // TODO(JMM): Eventually may eventually want this with ionization off if we
              // have other physics that needs lambdas
              RiotEOS::LambdaIndexerSingle lambda(pv_n, kji);
              const Real rho_val = pv_n(cm::rho(), kji);
              const Real T = pv(ccbulk::temperature(), kji);
              pv_n(cm::bulk_modulus(), kji) =
                  (rho_val > 0.0)
                      ? eosm_c.BulkModulusFromDensityTemperature(rho_val, T, lambda)
                      : 0.0;
              pv_n(cm::specific_heat(), kji) =
                  (rho_val > 0.0)
                      ? eosm_c.SpecificHeatFromDensityTemperature(rho_val, T, lambda)
                      : 0.0;
              pv_n(cm::sie(), kji) =
                  (rho_val > 0.0)
                      ? eosm_c.InternalEnergyFromDensityTemperature(rho_val, T, lambda)
                      : 0.0;
              // NOTE(@pdmullen): ccmat::rho already contains mask
              pv_n(ccmat::internal_energy(), kji) =
                  pv_n(ccmat::rho(), kji) * pv_n(cm::sie(), kji);
            });
          };
          eosm.EvaluateDevice(eosmloop);
        }
        idx_range.TeamBarrier();

        if (do_ionization) {
          // NOTE(JMM): Order matters
          Ionization::ComputeElectronTemperature(idx_range, v, electron_eos,
                                                 eos_from_matid, Te_root_tol, b, nmat);
          idx_range.TeamBarrier();

          // Updates per-mat electron energy and pressure, total pressure, and total bmod
          Ionization::PerMaterialEnergyPressureBmod(idx_range, v, eos, electron_eos,
                                                    eos_from_matid, b, nmat);
          idx_range.TeamBarrier();
        }

        // Recompute bulk quantities
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv(ccbulk::rho(), kji) = 0.0;
          pv(ccbulk::internal_energy(), kji) = 0.0;
          pv(ccbulk::bulk_modulus(), kji) = 0.0;
        });
        idx_range.TeamBarrier();
        for (int n = 0; n < nmat; n++) {
          auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv(ccbulk::rho(), kji) += pv_n(ccmat::rho(), kji);
            pv(ccbulk::internal_energy(), kji) += pv_n(ccmat::internal_energy(), kji);
            pv(ccbulk::bulk_modulus(), kji) +=
                pv_n(ccmat::volume_fraction(), kji) * pv_n(cm::bulk_modulus(), kji);
          });
          idx_range.TeamBarrier();
        }

        if (do_ionization) {
          // Resets bulk pressure and energy to totals, rather than ion-only
          Ionization::ConvertEnergyPressureBulkIon<Ionization::ToTotal>(idx_range, v, b);

          // Set per-material internal energy by volume to be total
          for (int n = 0; n < nmat; ++n) {
            auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
            RiotLoop::inner(idx_range, [&](const auto kji) {
              pv_n(ccmat::internal_energy(), kji) +=
                  pv_n(ccmat::electron_internal_energy(), kji);
              pv_n(cm::sie(), kji) += pv_n(cm::electron_sie(), kji);
            });
          }
          idx_range.TeamBarrier();
        }

        // Ensure consistency b/w masked fields and other relevant fields
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const Real rho_val = pv(ccbulk::rho(), kji);
          pv(ccbulk::momentum(0), kji) = rho_val * pv(ccbulk::velocity(0), kji);
          pv(ccbulk::momentum(1), kji) = rho_val * pv(ccbulk::velocity(1), kji);
          pv(ccbulk::momentum(2), kji) = rho_val * pv(ccbulk::velocity(2), kji);
          const Real vsq = SQR(pv(ccbulk::velocity(0), kji)) +
                           SQR(pv(ccbulk::velocity(1), kji)) +
                           SQR(pv(ccbulk::velocity(2), kji));
          pv(ccbulk::total_material_energy(), kji) =
              pv(ccbulk::internal_energy(), kji) + 0.5 * rho_val * vsq;
        });
      });

  // Recompute electron pseudo-entropy from electron internal energy or pressure
  Ionization::ConvertElectronEnergyEntropyWork(
      md, Ionization::EntropyDirection::ToEntropy, IndexDomain::entire);
}

//----------------------------------------------------------------------------------------
//! \fn  void Multiphysics::FillInteriorBlockDerived
//! \brief
void FillInteriorBlockDerived(std::shared_ptr<MeshBlockData<Real>> pmbd) {
  MeshData<Real> md;
  parthenon::BlockList_t block_list{pmbd->GetBlockSharedPointer()};
  md.Initialize(block_list, pmbd->GetMeshPointer());
  md.GetBlockData(0) = pmbd;
  FillInteriorDerived(&md);
}

} // namespace Multiphysics
