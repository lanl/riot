# Capability matrix — RIOT ideal MHD

Status vocabulary, applied strictly:

- **validated** — implemented, and a named test at a frozen threshold passes.
- **implemented-unverified** — code exists, no passing gate. Not a support claim.
- **unsupported/rejected** — refused at startup with an actionable message.

Combinations are recorded, not just toggles: "AMR validated" and "2T validated"
would not imply "2T + AMR validated."

**Current overall status (2026-09-19, head `62903d9`): gates G0–G4 pass, Stage 5
is complete except for the `tst/scripts/mhd/` harness and the eight-suite
hydro-with-MHD-off sweep. End-to-end MHD simulations have been run and compared
against an independent (Athena++) reference and against analytic solutions.**
Test IDs refer to [`TEST_LEDGER.md`](TEST_LEDGER.md); doubts about work already
done are in [`OPEN_CONCERNS.md`](OPEN_CONCERNS.md), which must be read before
any capability here is quoted elsewhere.

The published, user-facing version of this matrix is the MHD chapter of the docs
(`doc/sphinx/src/packages/mhd.rst`), and the same boundary is printed at startup
by `MHD::Initialize`.

## Numerical primitives

| Capability | Status | Evidence |
| --- | --- | --- |
| Magnetic energy `B²/(2μ₀)`, both unit conventions | **validated** | U01 |
| Unit-convention invariance (`μ₀ = 1` vs `4π`) | **validated** | U01b |
| Thermal-energy round trip through the conserved total | **validated** | U01 (incl. low-beta) |
| Fast magnetosonic speed, oblique | **validated** | U03 (quartic residual ≤ 1e-11) |
| Fast magnetosonic speed, degenerate orientations and `b→0` | **validated** | U03 |
| Face-to-cell B conventions, incl. collapsed directions | **validated** | 3 tests |
| Discrete curl of a vector potential is divergence free | **validated** | U04 (η ≤ 1e-13) |
| Cell-centered B / magnetic energy / div B from face state | **validated** | G4.1–G4.4 (div B at roundoff on 4 problems) |
| Face-aware stage register copy (`u1 ← u0`) | **implemented-unverified** | U05 not run; covered indirectly by every multi-stage run being correct (C11) |

## Solvers

| Capability | Status | Evidence |
| --- | --- | --- |
| MHD HLLE | **validated** | U02 (36 consistency combinations), G3/G4.1 Brio–Wu vs Athena++ |
| MHD HLLD, incl. all six degeneracy guards | **validated** | G5.1 unit tests, G5.2 vs Athena++, G5.3 CT undisturbed |
| MHD LLF (Rusanov) | **validated** | G5.8 (consistency, exact `−½a·Δρ` smearing, distinct from HLLE/HLLD) |
| Solver diffusivity ordering HLLD < HLLE < LLF | **validated** | G5.8 — strictly monotone on all five Brio–Wu fields |
| Normal induction flux exactly `0.0` | **validated** | U02, G5.8 (all three solvers, all directions) |
| Shared-normal-B substitution in reconstruction | **validated** | G5.5 (all five modes at a shock, `max\|Bx − 0.75\| = 0` exactly) |
| Fast speed in `BulkSoundSpeed` and the CFL vote | **validated** | U03, and every stable production run |

## Constrained transport and infrastructure

| Capability | Status | Evidence |
| --- | --- | --- |
| Gardiner–Stone upwind EMF, 1D/2D/3D incl. collapsed dimensions | **validated** | G4.1–G4.5, P01.3 (collapsed-EMF branch) |
| CT face update preserves div B | **validated** | `max\|divB\|` **exactly 0.0** (Brio–Wu), ≤ 3.3e-13 (Orszag–Tang), η ≤ 5.3e-15 (field loop) |
| No spurious out-of-plane / axial field | **validated** | G4.2 (`max\|B3\|` exactly 0.0), G4.3 all three permutations |
| Second-order accuracy on a smooth solution | **validated** | G4.5 CPAW, observed order 1.55 → 1.83 → 2 |
| MPI rank invariance | **validated** | P01 — ranks 1/2/3/4/5/8, bitwise identical; incl. shocks (P01.4) |
| Genuine face-centered field (no `CellMemAligned`) | **validated** | shared block faces single-valued; div B identical across block boundaries |
| Edge (EMF) flux register via automatic Face→Edge promotion | **validated** | exercised by every CT run |
| Hydro-only behavior preserved with MHD off | **validated** | G1.2, G5.4 (hydro suite 7/7), H01 re-run — bitwise vs pre-port `193b3fa` |
| `mhd` physics toggle, default off | **validated** | N01 positive control |
| Startup rejection of unsupported combinations | **validated** | N01 — 23/23, found and fixed two dead guards |
| Restart of nonzero face state | **validated** | G5.6 — bitwise for face B, conserved and magnetic derived state |
| Restart on a changed rank count | **validated** | G5.6 case 5 — 1-rank checkpoint on 4 ranks, ≤ 5.6e-15, bit-identical to the same-rank case |
| Rejection of restart across an MHD/hydro change | **validated** | G5.6 cases 6 and 7 (both directions; each was a real defect) |
| `mhd/monitor_divb` and solver-health counters | **validated** | G5.9 — rank-invariant, and the fallback counter immediately corrected a G5.8 overclaim |
| Restart bitwise for derived *primitives* | **known limitation** | C13 — `velocity`/`pressure` differ at 1–2 ulp. Proven **pre-existing** RIOT behavior by an A/B against `193b3fa` (G5.7), not introduced by this port |

## Reconstruction

All five modes are **certified for MHD** (G5.5); none is marked unsupported.
Requirement `nghost >= stencil_width + 1` is enforced and verified to abort with
an actionable message below that.

| Mode | Status | Measured order (CPAW, temporal error suppressed) |
| --- | --- | --- |
| `constant` | **validated** | 0.65 → 0.89 |
| `plm` | **validated** | 1.57 → 1.79 |
| `ppm4` | **validated** | 1.78 → 2.10 |
| `weno5` | **validated**, with a caveat | 4.21 / 3.93 / 4.16 — **not 5** |
| `mp5` | **validated**, with a caveat | 3.93 / 3.87 / 4.15 — **not 5**; additionally holds `\|B\|` uniform to roundoff |

Two recorded limitations, measured rather than argued: the ~4th-order cap is
attributed to the second-order EMF corner averaging in the CT update, and with
`rk2` at a fixed CFL the temporal error dominates so `weno5`/`mp5` show only
first-order *apparent* convergence and buy nothing over `ppm4` asymptotically.
Both are stated in the docs.

## Problem generators

| Generator | Status | Evidence |
| --- | --- | --- |
| `mhd_shock_tube` (Brio & Wu) | **validated** | G3/G4.1 vs Athena++, normalized L1 ≤ 3.8e-3 |
| `mhd_field_loop` (2D planar, 3D tilted) | **validated** | G4.2, G4.3, P01.1–P01.3 |
| `mhd_orszag_tang` | **validated** | G4.4, 1 vs 4 ranks bitwise |
| `mhd_cpaw` | **validated** | G4.5 order of accuracy, all direction permutations |

## Not yet done in this scope

| Item | Status |
| --- | --- |
| `tst/scripts/mhd/` regression harness under `tst/run_tests.py` | not implemented — the tests exist as scripts in `claude_sessions/mhd_runs/`, not yet as suite members |
| The eight non-hydro regression suites with MHD off (C1) | **NOT RUN** — advection, ionization, levelsets, mix, radiation_diffusion, radiation_transport, strength, tn |
| U05 as a live-mesh test | not implemented (C11) |
| Absolute (as opposed to comparative) Brio–Wu threshold | not defined (C3) |
| Oblique CPAW propagation | not covered (C9) |

## Unsupported / rejected at startup

Enforced in `src/riot.cpp`; rationale in
[`adr/004-support-matrix.md`](adr/004-support-matrix.md). Every row below is
exercised by N01.

| Configuration | Status | Why |
| --- | --- | --- |
| More than one material | **rejected** | Common-field/common-velocity mixture model needs scientific review |
| Non-Cartesian coordinates | **rejected** | Curvilinear face/edge metrics and magnetic geometry sources not implemented |
| Mesh refinement (SMR/AMR) | **rejected** | Divergence-preserving operators are registered but untested; the donor's MHD tests never exercise AMR, so no oracle exists |
| General PTE closure | **rejected** | Mixed-cell closure + general-EOS acoustic derivative need separate validation. Checked on the **resolved** flag, because a non-ideal `eos_type` turns it on regardless of the input |
| Fixed/frozen fluid | **rejected** | Induction transports the field with the fluid velocity |
| Material strength | **rejected** | Combined stresses and signal speeds need review |
| BHR mix model | **rejected** | Energy/stress audit not done |
| Thermonuclear burn | **rejected** | Energy audit not done |
| Ionization / two-temperature | **rejected** | Electron energy/entropy coupling needs separate validation |
| Level sets | **rejected** | Advection consistency not audited |
| Multigroup diffusion | **rejected** | Matter–energy exchange audit not done |
| Radiation transport | **rejected** | Matter+radiation budget audit not done |
| Lasers | **rejected** | Deposition-reservoir audit not done. Unreachable in practice: "Lasers requires ionization" fires first |
| Prescribed sources | **rejected** | Energy handoff audit not done |
| Gravity | **rejected** | Gravitational work term needs an energy audit |
| Passive scalars | **rejected** | Advection must use the MHD transport flow |
| Tracer particles | **rejected** | Advection consistency not audited |
| Hydro-only Riemann solvers with MHD | **rejected** | `hllc`/`hllcf`/`chllc`/`lhllc`/`hll` have no magnetic terms |
| An MHD Riemann solver with MHD off | **rejected** | Both directions of the mismatch are refused |
| `mhd_hlld` with a non-ideal EOS | **rejected** | Donor restricts HLLD to an ideal gas; use `mhd_hlle` |
| Restart across a change in `physics/mhd` | **rejected** | Reinterprets magnetic energy as heat, or starts from a zero field with no magnetic part in the total |
| Restart from an identically-zero field | **rejected** unless `mhd/allow_zero_field_restart` | That is what restarting from a hydro checkpoint by mistake looks like |

## Not available from this donor at all

Hall MHD, resistive induction, electron inertia, Biermann battery, ambipolar
diffusion, Nernst transport, anisotropic/Braginskii magnetized transport. The
donor is **ideal** MHD. Adding any of these is a new physics project, not a port.

This work must not be described as extended MHD, nor as porting the Yehan Toh
paper. RIOT's electron thermodynamics coexisting with ideal induction would not
make the result extended MHD.

## Environment coverage

| Axis | Status |
| --- | --- |
| Double precision, Kokkos Serial host backend | **validated** — unit tests and every simulation gate |
| MPI | **validated** for MHD — ranks 1/2/3/4/5/8, bitwise (P01), incl. restart on a changed rank count |
| OpenMP | **NOT RUN** |
| CUDA / GPU | **NOT RUN** — unavailable on this machine. Kernels follow Kokkos portability rules and are reviewed, but unvalidated is not validated |
| Single node | only — no multi-node run (C9) |
| 1D / 2D / 3D | **validated** in all three, including collapsed-dimension EMF branches |
| Uniform grid | only — refinement rejected |
