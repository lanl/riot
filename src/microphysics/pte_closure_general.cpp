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

// Singularity includes
#include <ports-of-call/portability.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/eos/eos.hpp>

// Riot includes
#include <array>
#include <cmath>

#include "eos_riot.hpp"
#include "ionization/ionization.hpp"
#include "materials/materials.hpp"
#include "pte_closure.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"
#include "void/void.hpp"
#include "void/void_closure.hpp"

using namespace parthenon::package::prelude;
using namespace RiotEOS;
using RiotLimits::MAX_MATERIALS;

//----------------------------------------------------------------------------------------
//! \fn  Closure::ApplyMixedCellClosure
//! \brief
void Closure::ApplyMixedCellClosure(MeshData<Real> *md, IndexDomain domain) {
  using namespace closure_impl;
  using parthenon::ParArray1D;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pm = md->GetParentPointer();
  auto &riot = pm->packages.Get("riot");

  // Verbosity
  const bool verbose = riot->Param<bool>("verbose");

  // Materials params
  auto materials = pm->packages.Get("materials");
  const auto &matlist = materials->Param<std::vector<int>>("h.pte_matlist");
  auto &eos = materials->Param<ParArray1D<RiotEOS::EOS>>("d.d.EOS");
  auto &eos_from_matid = materials->Param<ParArray1D<int>>("d.EOS_from_matid");
  auto &nphase = materials->Param<ParArray1D<int>>("d.nphase");
  auto &max_size = materials->Param<int>("max_array_size");
  PARTHENON_REQUIRE(max_size <= MAX_MATERIALS,
                    "Number of materials exceeds MAX_MATERIALS compile-time limit");

  // ionization physics
  const bool do_ionization = riot->Param<bool>("do_ionization");

  // PTE params
  auto pte_params = materials->Param<singularity::MixParams>("pte_params");
  const bool track_pte_statistics = materials->Param<bool>("track_pte_statistics");
  const bool pte_stats_avg_fields = materials->Param<bool>("pte_stats_avg_fields");
  const bool pte_stats_reset_fields = materials->Param<bool>("pte_stats_reset_fields");
  pte_params.verbose = verbose;

  // Hydro params
  auto &hydro = pm->packages.Get("hydro");
  const Real &mfrac_thr = hydro->Param<Real>("mass_frac_thresh");
  const Real &vfrac_thr = hydro->Param<Real>("vol_frac_thresh");
  const Real &tfloor = hydro->Param<Real>("temp_floor");
  const Real tceil = hydro->Param<Real>("temperature_ceiling");
  const Real tceil_rho = hydro->Param<Real>("temperature_ceiling_rho");

  // Void params (matid -1 if there is no void)
  const auto vp = Void::GetParams(pm);

  // Packing and Indexing
  auto v =
      riot::MakePack<ccmat::rho, ccmat::volume_fraction, cm::ionization_zbar,
                     cm::temperature, cm::lr_cache, cm::lT_cache, ccbulk::rho,
                     ccbulk::pressure, ccbulk::temperature, ccbulk::internal_energy,
                     ccbulk::total_material_energy, diag::pte_niter, diag::pte_nfails,
                     diag::pte_nbackups, diag::pte_ncalls, diag::pte_avg_niter,
                     diag::pte_avg_nbackups, diag::pte_fail_fraction>(md, matlist);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return;

  // Per-cell PTE scratch lives on the stack in fixed-size arrays sized by the
  // compile-time material bound; the solver's own workspace is sized to match.
  using LambdaIndexer_t = RiotEOS::LambdaIndexerMultiCoord<decltype(v)>;

  using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
  auto idx_space =
      lt::GetIndexSpace(domain, 0, nblocks, md, parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());

        std::array<int, MAX_MATERIALS> eos_map;
        FillEosMap<ccmat::rho>(v, b, nmat, eos_from_matid, nphase, eos_map);
        const int mv = Void::SliceIndex<ccmat::rho>(v, b, nmat, vp.matid);

        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto [k, j, i] = idx_range.GetKJI(kji);

          // Scratch arrays
          std::array<int, MAX_MATERIALS> pte2slice;
          std::array<bool, MAX_MATERIALS> active_mat = {false};
          std::array<Real, MAX_MATERIALS> ccmat_vfrac_s, cm_rho_s;
          std::array<Real, MAX_MATERIALS> cm_temp_s, cm_press_s, cm_sie_s;
          std::array<Real, singularity::PTESolverRhoTRequiredScratch(MAX_MATERIALS)>
              pte_scratch;

          // Reset fields if necessary for PTE diagnostics
          if (pte_stats_reset_fields) {
            pv(diag::pte_niter(), kji) = 0;
            pv(diag::pte_nfails(), kji) = 0;
            pv(diag::pte_nbackups(), kji) = 0;
            pv(diag::pte_ncalls(), kji) = 0;
          }

          // Set active materials in this cell and vfrac normalization
          int nmat_cell = 0;
          Real vfrac_sum = 0.0;
          const Real ccbulk_rho = pv(ccbulk::rho(), kji);
          for (int m = 0; m < nmat; m++) {
            const Real ccmat_rho = v(b, ccmat::rho(m), k, j, i);
            if (m != mv && ccmat_rho > mfrac_thr * ccbulk_rho) {
              const Real ccmat_vfrac = v(b, ccmat::volume_fraction(m), k, j, i);
              PARTHENON_REQUIRE(ccmat_vfrac > 0.0, "Negative vfrac in PTE setup");

              active_mat[m] = true;
              pte2slice[nmat_cell] = m;
              vfrac_sum += ccmat_vfrac;

              nmat_cell++;
            }
          }

          // Stash the temperature from previous soltution as a guess
          const Real Tguess = std::max(pv(ccbulk::temperature(), kji), tfloor);

          // Void: it never enters PTE. It is available wherever it is allocated, and
          // then open (alpha_V > 0, p = p_void) or closed (alpha_V = 0, p >= p_void).
          // Its (tracer) mass plays no role.
          const bool void_ok = mv >= 0;
          const Real rhobar_void = void_ok ? v(b, ccmat::rho(mv), k, j, i) : 0.0;
          bool void_open = false;

          // The partial densities of the n materials in pte2slice, and their densities
          // (guesses) from the volume fractions, for the open-void solves
          auto gather_void = [&](const int n, Real *rhobar_s, Real *rho_s) {
            for (int m = 0; m < n; m++) {
              const int mid = pte2slice[m];
              rhobar_s[m] = v(b, ccmat::rho(mid), k, j, i);
              rho_s[m] = rhobar_s[m] / v(b, ccmat::volume_fraction(mid), k, j, i);
            }
          };
          // ... and their volume fractions from the solved densities
          auto scatter_void = [&](const int n, const Real *rhobar_s, const Real *rho_s) {
            for (int m = 0; m < n; m++) {
              v(b, ccmat::volume_fraction(pte2slice[m]), k, j, i) =
                  rhobar_s[m] / rho_s[m];
            }
          };
          LocalEosIndexer<RiotEOS::EOS> eos_v(pte2slice.data(), eos_map.data(), eos);
          LambdaIndexer_t lambda_v(v, pte2slice.data(), b, k, j, i);

          // Set an open-void state of the n materials in pte2slice, at their isobar
          // densities rho_s and temperature T (returns false if the void is closed)
          auto try_void = [&](const int n, Real *rho_s, Real &T) {
            std::array<Real, MAX_MATERIALS> rhobar_s;
            gather_void(n, rhobar_s.data(), rho_s);
            int niter = 0, neos = 0;
            const bool open =
                Void::SolveOpen(n, eos_v, lambda_v, rhobar_s.data(), rho_s,
                                pv(ccbulk::internal_energy(), kji), vp.pressure, tfloor,
                                T, vp.tol, vp.max_iter, niter, neos);
            if (track_pte_statistics) pv(diag::pte_niter(), kji) += niter;
            if (!open) return false;
            scatter_void(n, rhobar_s.data(), rho_s);
            pv(ccbulk::temperature(), kji) = T;
            pv(ccbulk::pressure(), kji) = vp.pressure;
            return true;
          };

          if (nmat_cell == 0 && void_ok) {
            // Pure void, at p_void (its temperature is meaningless)
            void_open = true;
            pv(ccbulk::pressure(), kji) = vp.pressure;
          } else if (nmat_cell > 1) {
            // More than one material in cell, PTE callchain
            if (track_pte_statistics) pv(diag::pte_ncalls(), kji) += 1;

            // Now go through active materials and initialize scratch/guesses
            LocalEosIndexer<RiotEOS::EOS> eos_pte(pte2slice.data(), eos_map.data(), eos);
            LambdaIndexer_t lambda_pte(v, pte2slice.data(), b, k, j, i);
            Real rhoavg_pte = 0.0;
            auto init_pte = [&](const Real T0) {
              rhoavg_pte = 0.0;
              for (int m = 0; m < nmat_cell; m++) {
                const int mid = pte2slice[m];

                ccmat_vfrac_s[m] = v(b, ccmat::volume_fraction(mid), k, j, i) / vfrac_sum;
                cm_rho_s[m] = v(b, ccmat::rho(mid), k, j, i) / ccmat_vfrac_s[m];
                cm_temp_s[m] = T0;
                cm_press_s[m] = eos_pte[m].PressureFromDensityTemperature(
                    cm_rho_s[m], cm_temp_s[m], lambda_pte[m]);
                cm_sie_s[m] = eos_pte[m].InternalEnergyFromDensityTemperature(
                    cm_rho_s[m], cm_temp_s[m], lambda_pte[m]);

                rhoavg_pte += cm_rho_s[m] * ccmat_vfrac_s[m];
              }
            };
            init_pte(Tguess);

            // Void: open it first if the guess state is below p_void
            auto mean_pressure = [&]() {
              Real p = 0.0;
              for (int m = 0; m < nmat_cell; m++) {
                p += ccmat_vfrac_s[m] * cm_press_s[m];
              }
              return p;
            };
            // The matter's densities and temperature if the void were open (p = p_void)
            std::array<Real, MAX_MATERIALS> rho_open;
            Real T_open = Tguess;
            const bool void_tried = void_ok && mean_pressure() < vp.pressure;
            if (void_tried) void_open = try_void(nmat_cell, rho_open.data(), T_open);

            // Call PTE and record success/statistics
            constexpr Real vfrac_tot = 1.0; // we renormalized to enforce this
            const Real sie_tot = pv(ccbulk::internal_energy(), kji) / rhoavg_pte;
            std::array<Real *, 6> scr_pte{cm_rho_s.data(),   ccmat_vfrac_s.data(),
                                          cm_sie_s.data(),   cm_temp_s.data(),
                                          cm_press_s.data(), pte_scratch.data()};
            bool success = false, converged_cold = false;
            auto solve_pte = [&](const Real T0) {
              singularity::PTESolverRhoT<LocalEosIndexer<RiotEOS::EOS>, Real *,
                                         LambdaIndexer_t>
                  method(nmat_cell, eos_pte, vfrac_tot, sie_tot, scr_pte[0], scr_pte[1],
                         scr_pte[2], scr_pte[3], scr_pte[4], lambda_pte, scr_pte[5], T0,
                         pte_params);
              auto status = singularity::PTESolver(method);
              success = status.converged && (cm_temp_s[0] > tfloor);
              converged_cold = status.converged && cm_temp_s[0] <= tfloor;
              if (track_pte_statistics) pv(diag::pte_niter(), kji) += status.max_niter;
            };
            if (!void_open) {
              solve_pte(Tguess);

              // Void: open it if PTE failed or is below p_void (before the backups,
              // which remove materials)
              if (void_ok && !void_tried &&
                  !(success && mean_pressure() >= vp.pressure)) {
                void_open = try_void(nmat_cell, rho_open.data(), T_open);
              }

              // Void closed: retry PTE from T_open, which (unlike the previous
              // temperature) matches this step's energy
              if (!success && !void_open && T_open != Tguess) {
                init_pte(T_open);
                solve_pte(T_open);
              }
            }

            // Handle PTE return statuses
            if (void_open) {
              // Open-void state already set
            } else if (success) {
              // Set PTE solution
              Real mean_p = 0.0;
              for (int m = 0; m < nmat_cell; m++) {
                const int mid = pte2slice[m];
                v(b, ccmat::volume_fraction(mid), k, j, i) = ccmat_vfrac_s[m];
                mean_p += ccmat_vfrac_s[m] * cm_press_s[m];
              }

              pv(ccbulk::temperature(), kji) = cm_temp_s[0];
              pv(ccbulk::pressure(), kji) = mean_p;
            } else { // PTE failed
              // A converged solution below the floor is clamped to it (by a FixedT
              // solve at the floor); otherwise fall back to T_open (the previous
              // temperature unless the void was tried)
              const Real Tfix = converged_cold ? tfloor : T_open;

              // Record failure
              if (track_pte_statistics) pv(diag::pte_nfails(), kji) += 1;

              // Report failure (if verbose)
              if (verbose) {
                printf("PTE failing at b=%d i=%d j=%d k=%d x=%g y=%g z=%g\n", b, i, j, k,
                       v.GetCoordinates(b).Xc<parthenon::X1DIR>(i),
                       v.GetCoordinates(b).Xc<parthenon::X2DIR>(j),
                       v.GetCoordinates(b).Xc<parthenon::X3DIR>(k));
                printf("Trying backup with T = %g\n", Tfix);
              }

              // Iterate on PTE FixedT solves
              // NOTE(@pdmullen): Upon failed PTE Fixed T solves, we sequentialy remove
              // mats with the least ccmat_rho, until PTE (or single mat) solution
              // acquired...
              for (int iter = 0; iter < nmat_cell; iter++) {
                // Record backup attempt
                if (track_pte_statistics) pv(diag::pte_nbackups(), kji) += 1;

                // Check number of materials in cell this iteration
                int nmat_backup = 0;
                for (int m = 0; m < nmat; m++) {
                  if (active_mat[m]) {
                    pte2slice[nmat_backup] = m;
                    nmat_backup++;
                  }
                }

                // PTE FixedT solve
                if (nmat_backup > 1) {
                  // Go through (remaining) active mats and initialize scratch/guesses
                  LocalEosIndexer<RiotEOS::EOS> eos_fixt(pte2slice.data(), eos_map.data(),
                                                         eos);
                  LambdaIndexer_t lambda_fixt(v, pte2slice.data(), b, k, j, i);
                  for (int m = 0; m < nmat_backup; m++) {
                    const int mid = pte2slice[m];
                    ccmat_vfrac_s[m] =
                        v(b, ccmat::volume_fraction(mid), k, j, i) / vfrac_sum;
                    cm_rho_s[m] = v(b, ccmat::rho(mid), k, j, i) / ccmat_vfrac_s[m];
                    cm_temp_s[m] = Tfix;
                    cm_press_s[m] = eos_fixt[m].PressureFromDensityTemperature(
                        cm_rho_s[m], cm_temp_s[m], lambda_fixt[m]);
                    cm_sie_s[m] = eos_fixt[m].InternalEnergyFromDensityTemperature(
                        cm_rho_s[m], cm_temp_s[m], lambda_fixt[m]);
                  }

                  // Call FixedT PTE and record success/statistics
                  std::array<Real *, 6> scr_fixt{cm_rho_s.data(),   ccmat_vfrac_s.data(),
                                                 cm_sie_s.data(),   cm_temp_s.data(),
                                                 cm_press_s.data(), pte_scratch.data()};
                  singularity::PTESolverFixedT<LocalEosIndexer<EOS>, Real *,
                                               LambdaIndexer_t>
                      backup(nmat_backup, eos_fixt, vfrac_tot, Tfix, scr_fixt[0],
                             scr_fixt[1], scr_fixt[2], scr_fixt[3], scr_fixt[4],
                             lambda_fixt, scr_fixt[5], pte_params);
                  auto status = singularity::PTESolver(backup);
                  success = status.converged;
                  if (track_pte_statistics) {
                    pv(diag::pte_niter(), kji) += status.max_niter;
                  }

                  // Handle FixedT PTE return statuses
                  if (success) {
                    // Set FixedT PTE solution
                    Real mean_p = 0.0;
                    for (int m = 0; m < nmat_backup; m++) {
                      const int mid = pte2slice[m];
                      v(b, ccmat::volume_fraction(mid), k, j, i) = ccmat_vfrac_s[m];
                      mean_p += ccmat_vfrac_s[m] * cm_press_s[m];
                    }
                    pv(ccbulk::temperature(), kji) = Tfix;
                    pv(ccbulk::pressure(), kji) = mean_p;

                    // PTE FixedT success; break out of iterative loop
                    break;
                  }
                } else { // nmat_backup == 1
                  // NOTE(@pdmullen): If reaching this branch, then only one active
                  // material remains in the cell.  Therefore, we call the EOS directly
                  const int mid = pte2slice[0];
                  pv(ccbulk::temperature(), kji) = Tfix;
                  v(b, ccmat::volume_fraction(mid), k, j, i) = 1.0;

                  // Now ccmat::rho is equivalent to cm::rho
                  const Real rhom = v(b, ccmat::rho(mid), k, j, i);

                  // Call EOS for pressure
                  RiotEOS::LambdaIndexerSingleCoord<decltype(v)> ll(v, b, mid, k, j, i);
                  pv(ccbulk::pressure(), kji) =
                      eos[eos_map[mid]].PressureFromDensityTemperature(rhom, Tfix, ll);

                  // Break out of iterative loop
                  break;
                }

                // If reaching here, the PTE FixedT solve (i.e., with nmat_backup > 1) has
                // failed.  Determine the index of the material with smallest ccmat::rho
                int idx_min = pte2slice[0];
                Real rhobar_min = v(b, ccmat::rho(idx_min), k, j, i);
                for (int m = 1; m < nmat_backup; m++) {
                  const int mid = pte2slice[m];
                  const Real rhobar = v(b, ccmat::rho(mid), k, j, i);
                  if (rhobar < rhobar_min) {
                    idx_min = mid;
                    rhobar_min = rhobar;
                  }
                }

                // Remove identified material
                if (verbose) printf("Failed FixedT Solve, removing minimum ccmat_rho\n");
                active_mat[idx_min] = false;
                vfrac_sum -= v(b, ccmat::volume_fraction(idx_min), k, j, i);
                pv(ccbulk::rho(), kji) -= v(b, ccmat::rho(idx_min), k, j, i);
                v(b, ccmat::rho(idx_min), k, j, i) = 0.0;
                v(b, ccmat::volume_fraction(idx_min), k, j, i) = 0.0;
              }
            }
          } else if (nmat_cell == 1) {
            // Handle single material case
            const int mid = pte2slice[0];
            const Real alpha_guess = v(b, ccmat::volume_fraction(mid), k, j, i);
            v(b, ccmat::volume_fraction(mid), k, j, i) = 1.0;

            // Obtain temperature directly from updated internal energy and bulk density
            // (less the void, which has no energy)
            const Real rho1 = ccbulk_rho - rhobar_void;
            const Real sie = pv(ccbulk::internal_energy(), kji) / rho1;
            RiotEOS::LambdaIndexerSingleCoord<decltype(v)> ll(v, b, mid, k, j, i);
            Real tmat =
                eos[eos_map[mid]].TemperatureFromDensityInternalEnergy(rho1, sie, ll);
            // Clamp to the floor (falling back to the previous temperature if NaN)
            tmat = std::isnan(tmat) ? Tguess : std::max(tmat, tfloor);
            pv(ccbulk::temperature(), kji) = tmat;

            // Obtain pressure directly from EOS call
            pv(ccbulk::pressure(), kji) =
                eos[eos_map[mid]].PressureFromDensityTemperature(rho1, tmat, ll);

            // Void: open it if the material alone is below p_void
            if (void_ok && pv(ccbulk::pressure(), kji) < vp.pressure) {
              v(b, ccmat::volume_fraction(mid), k, j, i) = alpha_guess;
              Real rho_s[1], T_open = tmat;
              void_open = try_void(1, rho_s, T_open);
              if (!void_open) v(b, ccmat::volume_fraction(mid), k, j, i) = 1.0;
            }

            // Single material
          } else { // nmat_cell < 1
            // Empty cell encountered
            if (verbose) {
              printf("Empty cell!  %g  %d %d %d   %g %g %g\n", ccbulk_rho, b, j, i,
                     v.GetCoordinates(b).Xc<parthenon::X1DIR>(i),
                     v.GetCoordinates(b).Xc<parthenon::X2DIR>(j),
                     v.GetCoordinates(b).Xc<parthenon::X3DIR>(k));
              for (int m = 0; m < nmat; m++) {
                printf("%d %g %g\n", m, v(b, ccmat::rho(m), k, j, i),
                       v(b, ccmat::volume_fraction(m), k, j, i));
              }
            }

            // For now we just fail.  In future, we may consider "floors"...
            PARTHENON_FAIL("Aborting due to empty cell!");
          }

          // Temperature ceiling below a density: clamp the temperature and reset the
          // pressure to the EOS at the ceiling, at the closure's volume fractions
          // (PostCommsFillDerived rebuilds the energies from T)
          // An open void stays open at the ceiling (the isobar densities grow as T drops)
          const bool ceil = tceil > 0.0 && pv(ccbulk::rho(), kji) < tceil_rho &&
                            pv(ccbulk::temperature(), kji) > tceil;
          bool void_ceil = ceil && void_open;
          if (void_ceil && nmat_cell > 0) {
            std::array<Real, MAX_MATERIALS> rhobar_s, rho_s;
            gather_void(nmat_cell, rhobar_s.data(), rho_s.data());
            int neos = 0;
            void_ceil = Void::SolveOpenFixedT(nmat_cell, eos_v, lambda_v, rhobar_s.data(),
                                              rho_s.data(), vp.pressure, tceil, vp.tol,
                                              vp.max_iter, neos);
            if (void_ceil) scatter_void(nmat_cell, rhobar_s.data(), rho_s.data());
          }
          if (void_ceil) {
            // Open void at the ceiling (p = p_void)
            pv(ccbulk::temperature(), kji) = tceil;
          } else if (ceil) {
            Real p = 0.0;
            for (int m = 0; m < nmat; m++) {
              const Real alpha = v(b, ccmat::volume_fraction(m), k, j, i);
              const Real rhom = v(b, ccmat::rho(m), k, j, i);
              if (!active_mat[m] || !(alpha > 0.0 && rhom > 0.0)) continue;
              RiotEOS::LambdaIndexerSingleCoord<decltype(v)> ll(v, b, m, k, j, i);
              p += alpha * eos[eos_map[m]].PressureFromDensityTemperature(rhom / alpha,
                                                                          tceil, ll);
            }
            pv(ccbulk::temperature(), kji) = tceil;
            pv(ccbulk::pressure(), kji) = p;
          }

          // Void volume fraction: the rest of the cell if open, else none (its mass is
          // then removed by the volume fraction mask below)
          if (void_ok) {
            Real alpha_void = 0.0;
            if (void_open) {
              alpha_void = 1.0;
              for (int m = 0; m < nmat; m++) {
                if (active_mat[m]) alpha_void -= v(b, ccmat::volume_fraction(m), k, j, i);
              }
            }
            v(b, ccmat::volume_fraction(mv), k, j, i) = std::max(alpha_void, 0.0);
          }

          // Handle PTE diagnostic averaging (if enabled)
          if (track_pte_statistics && pte_stats_avg_fields) {
            const Real ncalls = pv(diag::pte_ncalls(), kji);
            pv(diag::pte_avg_niter(), kji) =
                pv(diag::pte_niter(), kji) / std::max(1.0, ncalls);
            pv(diag::pte_avg_nbackups(), kji) =
                pv(diag::pte_nbackups(), kji) / std::max(1.0, ncalls);
            pv(diag::pte_fail_fraction(), kji) =
                pv(diag::pte_nfails(), kji) / std::max(1.0, ncalls);
          }
        });
        idx_range.TeamBarrier();

        // Apply masks to material quantities marked Metadata::FillGhost *prior* to comms
        for (int m = 0; m < nmat; m++) {
          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          RiotLoop::inner(idx_range, [&](const auto kji) {
            // (the void is exempt from the mass fraction threshold)
            const Real mthr_mask =
                (m == mv || pv_m(ccmat::rho(), kji) > mfrac_thr * pv(ccbulk::rho(), kji));
            const Real vthr_mask = (pv_m(ccmat::volume_fraction(), kji) > vfrac_thr);
            const Real mask = mthr_mask * vthr_mask;
            pv_m(ccmat::rho(), kji) *= mask;
            pv_m(ccmat::volume_fraction(), kji) *= mask;
          });
        }
      });
}
