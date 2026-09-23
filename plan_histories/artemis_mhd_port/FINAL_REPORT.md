# Final report — Artemis → RIOT ideal-MHD port

Branch `taitano/mhd-porting`, 30 commits on integration base
`193b3fa2a61557cb4fc761bb87de6687ca781edf`. Work performed 2026-09-18 → 2026-09-23,
Agentic-AI-assisted (Claude), recorded per RIOT's contribution guide.

Donor: `lanl/artemis` branch `dempsey/mhd` at `3e5aeb5`
([PR #123](https://github.com/lanl/artemis/pull/123)), comparison base `e8a4e0f5`.
Full revision identity in [`REVISION_MANIFEST.md`](REVISION_MANIFEST.md).

---

## 1. What was reached — the one-sentence claim

**Donor ideal-MHD parity on a certified matrix: Cartesian, uniform grid, single
material, ideal gas, double precision, Kokkos Serial host backend, with MPI.**

ADR-004 named three distinct accomplishments that must not be conflated. This is
the second of them:

| # | Accomplishment | Reached? |
| --- | --- | --- |
| 1 | Working prototype — compiles, plausible output | yes, and superseded |
| 2 | **Donor ideal-MHD parity on the certified matrix** | **yes — this is the claim** |
| 3 | RIOT coupled-physics qualification | **no** — needs Stages 6–9 and human scientific review |

What this is **not**:

- Not AMR-capable, not curvilinear, not general-EOS, not multi-material, not
  two-temperature. Each is refused at startup, not silently downgraded.
- Not coupled to any RIOT source package. Every one of the sixteen physics
  couplings is rejected at startup pending its own energy/stress audit.
- Not extended MHD, and not a port of the Yehan Toh extended-MHD paper. The donor
  is **ideal** MHD; Hall, resistivity, Biermann, ambipolar, Nernst and Braginskii
  transport are absent from the donor entirely.
- Not GPU-validated. Kernels follow Kokkos portability rules and were reviewed by
  the `kokkos-portability-reviewer` subagent, but no CUDA device was available.
  Reviewed is not validated.
- Not OpenMP-validated, and not multi-node.

## 2. What was built

| Area | Files | Lines added |
| --- | --- | --- |
| MHD package | `src/mhd/{mhd.cpp,mhd.hpp,mhd_helpers.hpp,riemann_mhd.hpp,emf.hpp}` | 2196 |
| Problem generators | `src/riot_pgen/mhd_problems.cpp` | 712 |
| Host-code integration | `hydro/`, `multiphysics/`, `riot.cpp`, `riot_driver.cpp`, `riot_utils/`, `variables.hpp` | 756 (−31) |
| Unit tests | `tst/unit/test_mhd.cpp` | 1216 |
| Regression suite | `tst/scripts/mhd/*.py`, `tst/scripts/utils/mhd_analysis.py` | 1015 |
| Docs | `doc/sphinx/src/packages/mhd.rst` + toctree/hydro cross-refs | 522 (−1) |
| Input decks | `inputs/mhd/*.py` (deck generators) | 608 |
| Work record | `plan_histories/artemis_mhd_port/` | 5110 |

Capabilities, in the vocabulary of [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md)
(*validated* = a named test at a frozen threshold passes):

- **Three MHD Riemann solvers** — HLLE, HLLD (all six donor degeneracy guards,
  ideal gas only), LLF/Rusanov. Diffusivity ordering HLLD < HLLE < LLF is strictly
  monotone on all five Brio–Wu fields.
- **Gardiner–Stone upwind constrained transport** on a genuine face-centered field,
  1D/2D/3D including collapsed-dimension EMF branches.
- **Conserved-energy contract** (ADR-002): `total_material_energy` includes
  `B²/(2μ₀)` when and only when MHD is on, at two choke points in
  `fill_shared_derived.cpp`. One conservative update.
- **One unit parameter** `mhd/mu0`, default `4π` (Gaussian CGS, matching RIOT's
  EOS/opacity). Tests use `mu0 = 1`, reproducing donor normalization exactly so
  comparison needs no reconciliation step.
- **Four problem generators** — Brio–Wu shock tube, field loop (2D planar, 3D
  tilted, rotatable onto each axis pair), Orszag–Tang, circularly polarized
  Alfvén wave.
- **All five reconstruction modes certified** for MHD: `constant`, `plm`, `ppm4`,
  `weno5`, `mp5`.
- **Instrumentation** — `mhd/monitor_divb`, plus `hlld_fallback` and
  `density_floor` health counters, MPI-summed and normalized per cell per step.
- **Startup rejection of everything outside the matrix**, 23 configurations,
  no silent fallback.

Two findings that shaped the work, both contradicting the pre-audit plan:

1. **No dependency change was needed.** RIOT's pinned Parthenon (`928544a6d`) is
   byte-identical to the donor's in every CT-relevant file and additionally carries
   a flux-correction race fix the donor's older pin lacks. Two donor files
   (`prolongation.hpp`, `restriction.hpp`) were therefore **not** ported.
2. **RIOT had no genuine face field before this.** All six pre-existing "face"
   fields carry `Metadata::CellMemAligned`. Two shared routines assumed cell
   topology and would have silently corrupted a CT field — one truncating the stage
   register copy, one applying a cell-centered flux divergence to it. Both fixed in
   `src/riot_utils/sparse_update.hpp` (ADR-003).

## 3. Verification — evidence, re-confirmed at HEAD

Three independent oracle classes, because donor agreement alone cannot detect a
defect both implementations share:

1. Analytic identities and conservation/divergence invariants.
2. The pinned donor at matched units, EOS, mesh, CFL, solver and final time.
3. `artemis/tst/scripts/mhd/athena_bw.std` — an Athena++ Brio–Wu solution, the
   only genuinely third-party reference in the port.

Thresholds were frozen before evaluation and are the donor's own demonstrated
values. No skipped test is recorded as a pass, and gold data generated by this
implementation is never presented as validation of it.

### Re-run at HEAD (`1da9040`), 2026-09-23

| Suite | Result |
| --- | --- |
| `ctest --test-dir build` | **49/49 pass** |
| `python run_tests.py mhd` | **4/4 pass** — `brio_wu` 6.4 s, `cpaw` 11.4 s, `field_loop` 100 s, `orszag_tang` 2.9 s |
| `python run_tests.py hydro` | **7/7 pass** — `adiabatic_compression` 73 s, `carbuncle` 184 s, `gacc` 4.9 s, `linwave` 43 s, `linwave_mm` 54 s, `rt_amr` 162 s, `rt_unigrid` 304 s |

The eight-suite non-hydro sweep (§ below) was **not** re-run at HEAD; it was run at
`a95f001`. The last commit to touch `src/` is `50bde9c` (the D01 fix), which
*precedes* that sweep, and everything after `a95f001` is `plan_histories/` markdown
plus eleven lines of `doc/sphinx/src/packages/mhd.rst`. **No compiled code
separates that sweep from HEAD**, which is why it was not repeated — the `ctest`,
`mhd` and `hydro` runs above were, because they are cheap enough to not need the
argument.

### Headline numbers

| Property | Measured |
| --- | --- |
| `max\|div B\|`, Brio–Wu | **exactly 0.0** |
| `max\|div B\|`, Orszag–Tang | ≤ 3.3e-13 |
| dimensionless η, field loop | ≤ 5.3e-15 |
| Spurious axial field, 3D Gardiner–Stone | `max\|B3\|` **exactly 0.0**, all three permutations |
| Brio–Wu vs Athena++ | normalized L1 ≤ 3.8e-3; HLLD beats HLLE on every field (ratios 0.65–0.80) |
| CPAW order of accuracy | 1.55 → 1.83 → 2, error identical to every printed digit across all three axes (exact rotational isotropy) |
| MPI rank invariance | **bitwise** at ranks 1/2/3/4/5/8, 2D and 3D, with shocks |
| Mesh decomposition invariance | 1 vs 8 blocks as 2×2×2 **bitwise**; 2/3/4/6 blocks per axis at roundoff |
| Hydro unchanged with MHD off | **bitwise** vs pre-port `193b3fa` (H01, and a second problem via G5.7) |
| Face-B restart | **bitwise**; evolved restart ≤ 1e-14, incl. a changed rank count |

Per-test detail, commands and artifacts: [`TEST_LEDGER.md`](TEST_LEDGER.md) (2090
lines, every run recorded including the failures).

### Regression sweep over the rest of RIOT

Stage 1 changed **shared** machinery (`sparse_update::UpdateToNextStage`, the
`DeepCopyIndependentData` filters) that every physics package uses, so the eight
never-before-run suites were swept with MHD off (concern C1):

`advection`, `ionization`, `levelsets`, `mix`, `radiation_diffusion`,
`radiation_transport`, `strength`, `tn` → **12 of 13 pass.**

The single failure, `radiation_transport.marshak` (self-similarity spread 3.504e-02
against its own 3.5e-2 threshold, 0.11 % over), is **pre-existing upstream
behavior, not a regression.** Proven rather than asserted: the pre-port `193b3fa`
binary, built with the cmake arguments read out of `tst/build/CMakeCache.txt` and
with submodules symlinked so dependency SHAs are provably identical, reproduces the
same metric to all four digits **and** gives **bitwise-identical**
`c.c.bulk.temperature` and `c.c.rad.moments` across all nine output dumps. The port
has no effect whatsoever on that test. **Its threshold must not be relaxed here.**

Plus the hydro suite 7/7 with output bitwise identical to pre-port.

## 4. Defects found, and what finding them cost

Ten real defects were found by verification work rather than by inspection. That
count is the argument for the verification stance, so it is reported rather than
buried.

| # | Defect | Found by |
| --- | --- | --- |
| 1 | Stage register copy truncated a face field (cell-topology assumption) | ADR-003 audit |
| 2 | Cell-centered flux divergence applied to a face field | ADR-003 audit |
| 3 | Face field declared `CellMemAligned` — never ghost-exchanged | G1 |
| 4 | HLLD ideal-gas guard read the wrong input block → **dead code, could never fire** | N01 |
| 5 | Both general-PTE guards read the input flag, which cannot see a non-ideal `eos_type` → a real EOS bypassed them entirely | N01 |
| 6 | Derived magnetic state never rebuilt on restart → 3.7e-06 **permanent** trajectory error | G5.6 |
| 7 | Restart across an MHD→hydro change ran silently → `B²/2μ₀` reinterpreted as heat, 80 % pressure error at exit code 0 | G5.6 |
| 8 | Restart across hydro→MHD ran silently (other direction) | G5.6 |
| 9 | **D01** — constrained transport lost div B with 4+ blocks per periodic axis | `tst/scripts/mhd/field_loop`, first run |
| 10 | `mhd/monitor_divb` was a dead parameter | C7 |

Two claims were also **retracted and re-established** rather than left standing: an
early "pre-existing" attribution for C13 that rested on a control not using the
pre-port binary, and G5.8's field-loop corroboration of HLLD-over-HLLE, which the
new fallback counter showed to be weak evidence (HLLD falls back to HLLE on ~7
faces per cell per step on a planar field). Brio–Wu, with zero fallbacks, is the
real evidence.

### D01 is the finding worth carrying forward

**Root cause:** `physics/sparse_physics` (default **true**) deallocates
`ccbulk::cell_delta` on blocks where nothing is changing, and `riot::GetPack` then
drops those blocks from *every* pack built through it. `AssembleEdgeEMF` and
`ApplyFaceUpdate` take their block loop bound from the pack, so a deactivated block
was silently never updated — its face field froze while neighbours kept updating
the faces they *shared* with it, permanently breaking that block's divergence
budget. Symptom: `max|div B|` 6.8e-17 at two blocks per axis, 6.5e-17 at three,
**6.2e-08 at four**.

**Why the donor could not have warned about it: Artemis has no block-deactivation
mechanism at all.** A CT scheme was moved into a code that skips blocks. This is
the class of defect that only integration testing in the *recipient* code can find,
and it is the strongest argument in this report for building the regression harness
instead of trusting the ad-hoc runs that had already "validated" CT.

**Fix:** force `sparse_physics = false` when MHD is on, with a warning
(`src/riot.cpp`), following the existing global-solver precedent. Patching only the
MHD packs would **not** have worked — the EMF is assembled from hydro flux
registers and `Hydro::CalculateFluxes` packs through the same helper, so a
deactivated block has no fluxes to build an EMF from either. Accepted cost: MHD
runs lose that optimization (concern C15, targeted fix known but unimplemented).

**Two diagnostic traps cost most of the hunt and produced confident wrong
"no mismatch" answers:** `md->GetBlockData(b)` is **not** aligned with a pack's
block index `b`, and `pmb->coords.Xf<X1DIR>(is)` also collides across blocks. Take
block identity from `pack.GetCoordinates(b)` inside the same device kernel that
reads the data, and print `pack.GetNBlocks()` next to `md->NumBlocks()` — that one
line exposed the cause immediately.

## 5. Honest limitations

### Measured, not argued

- **WENO5 and MP5 reach order ~4, not 5** (4.21/3.93/4.16 and 3.93/3.87/4.15 with
  the temporal error suppressed). Attributed to the second-order EMF corner
  averaging in the CT update — **that attribution is a hypothesis, not a
  measurement** (concern C4).
- **With `rk2` at fixed CFL the temporal error dominates**, so `weno5`/`mp5` show
  only first-order *apparent* convergence and buy nothing over `ppm4`
  asymptotically. Quantified: an 8× CFL cut gives 63× error reduction, and HLLD
  reproduces HLLE's numbers in every digit, so the floor is the integrator.
- **Restarts are not bitwise for derived primitives** — `velocity` 2.2e-16,
  `pressure` 1.1e-15, even on a zero-step reload. Proven **pre-existing** RIOT-wide
  behavior by an A/B against `193b3fa` (G5.7). Worth raising upstream; concern C13
  stays open for the behavior itself.
- **Sparse physics is off for all MHD runs.** A correctness-preserving performance
  loss that scales with the fraction of blocks that would have been deactivated.
  Unmeasured and problem-dependent (C15).

### Gaps in the verification itself

These are the places a defect could still be hiding behind a passing test. Full
register in [`OPEN_CONCERNS.md`](OPEN_CONCERNS.md); **no HIGH-severity entry
remains open.**

| Concern | Severity | Gap |
| --- | --- | --- |
| C3 | MEDIUM | Brio–Wu has **no absolute frozen threshold** — only "HLLD beats HLLE". Accuracy could regress 2× with both solvers degrading together and every test would still pass. Monitored, not gated. |
| C4 | MEDIUM | The order-4 cap on WENO5/MP5 is a guess. A genuine defect in the high-order path would present identically. |
| C8 | MEDIUM | The only third-party oracle (`athena_bw.std`) lives outside the repo, and two baseline directories are **untracked**. `tst/scripts/mhd/brio_wu` therefore asserts oracle-independent invariants only, and the comparison would *silently skip* in CI. |
| C9 | LOW | Single-node, uniform-grid, axis-aligned only. No interconnect, no coarse/fine flux or EMF correction, no oblique wave (an oblique wave couples all three EMF components at once and is strictly stronger). |
| C10 | LOW | HLLD consistency is conditioning-limited for β ~ 1e-12 states; their real value is as NaN-regression cases. |
| C11 | LOW | The face-aware stage copy (U05) is exercised by every RK2 MHD run and was measured on a live mesh, but has no *unit* test with deliberately distinct stage values. |
| C12 | LOW | HLLD's transverse face velocities are unverified in use. Risk is nil today because scalars and tracers are rejected under MHD — and will not be obvious the moment either is enabled. |
| C15 | LOW | Sparse physics disabled globally rather than made CT-aware. |

## 6. What remains — Stages 6–9

Recorded in full in [`DEFERRED_STAGES.md`](DEFERRED_STAGES.md), with gates and
ordering constraints. **Nothing there is a commitment**; each stage is a separate
project needing scope approval, and several need scientific review before any code.

| Stage | Scope | Blocker |
| --- | --- | --- |
| 6 | Static/dynamic AMR, load balancing | **No donor oracle** — Artemis's own MHD tests never exercise AMR; its linear-wave pgen hard-rejects multilevel and every MHD input uses `refinement = none`. Gates would be analytic-only. |
| 7 | Curvilinear geometry, axes, poles | Face/edge metrics, Maxwell-stress geometry sources, axis parity. The donor's curvilinear EMF carries an **O(Δ) metric inconsistency that must be re-derived, not transcribed** (ADR-003). Interacts with Stage 6 rather than following it. |
| 8A | General EOS | Needs the correct frozen/equilibrated acoustic derivative. `γp/ρ` with an invented effective γ is not acceptable; HLLD contains constant-γ assumptions. |
| 8B | Two-temperature / electron energy | Electron/ion shock-heating partition is not determined by ideal induction. The donor's dual-energy variable is **not** an electron-energy equation. |
| 8C | Mixed materials, sparse allocation | The common-field/common-velocity model over a volume-additive mixture is a **physics modeling assumption requiring scientific review**, not an implementation detail. |
| 9 | Per-package coupling, production qualification | One package, one energy/stress audit, one gate, one matrix row. Independent row by row. |

**The one thing that looks built and is not: AMR.** The face field carries
`ProlongateInternalTothAndRoe` and `RestrictAverage`, and Parthenon supplies
coarse/fine edge correction, so the metadata is complete and correct. **Registered
refinement operators are not evidence that refinement works.**

Sharpened 2026-09-23, and it cuts the other way from how this report first read:
the donor **does** have face/edge AMR code (commits `8529742`, `78dbc13`) — it is
merely untested there, since every donor MHD input is `refinement = none`. More
importantly, the operator RIOT registered is Parthenon's `ProlongateInternalTothAndRoe`,
which shares a **name but not an algorithm** with the donor's `ProlongateTothAndRoe`:
~83 lines of direct formula versus ~450 lines of face-area-weighted
divergence-constraint solve that `78dbc13` introduced expressly to stop div B
growing during prolongation. `DONOR_DELTA.md` had recorded the donor file as a
redundancy to skip; that entry is corrected, and Stage 6 now opens with the
measurement that decides whether the framework operator suffices in Cartesian. This
is a latent porting error found before it could ship, and it is unreachable today
only because refinement is rejected at startup.

Also outstanding and cheap: MHD eigenmodes in `linear_modes.cpp`; a GPU build;
OpenMP; a multi-node run.

**Permanently out of scope:** Hall MHD, resistive induction, electron inertia,
Biermann battery, ambipolar diffusion, Nernst, anisotropic/Braginskii transport.
The donor is ideal MHD. Each is a new physics project.

## 7. Where the work record lives

All of it is **in the repository** under `plan_histories/artemis_mhd_port/`, not in
an untracked scratch directory. Read order in [`README.md`](README.md).

| File | Contents |
| --- | --- |
| [`REVISION_MANIFEST.md`](REVISION_MANIFEST.md) | Pinned SHAs, submodules, toolchain. Immutable — append, never overwrite. |
| [`SOURCE_MAP.md`](SOURCE_MAP.md) | Donor responsibility → verified RIOT owner, with `file:line` evidence |
| [`DONOR_DELTA.md`](DONOR_DELTA.md) | All 67 changed donor files, each port/adapt/reimplement/skip/defer with rationale |
| [`adr/001-units.md`](adr/001-units.md) | One `mhd/mu0`; `E_mag = B²/(2μ₀)` |
| [`adr/002-energy.md`](adr/002-energy.md) | Conserved-energy ownership. **Read before touching energy code.** |
| [`adr/003-field-topology-and-stages.md`](adr/003-field-topology-and-stages.md) | Face/edge topology, metadata, stage ownership, task graph, the two integration hazards |
| [`adr/004-support-matrix.md`](adr/004-support-matrix.md) | What is claimed, what is rejected, and the honesty constraints on this report |
| [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md) | Per-capability validated / implemented-unverified / rejected, with test IDs |
| [`TEST_LEDGER.md`](TEST_LEDGER.md) | Every run: command, threshold, result, artifact |
| [`OPEN_CONCERNS.md`](OPEN_CONCERNS.md) | Doubts about work already done. **Read before quoting any capability.** |
| [`DEFERRED_STAGES.md`](DEFERRED_STAGES.md) | Stages 6–9: planned and NOT built |
| [`CHECKPOINT.md`](CHECKPOINT.md) | State and re-run recipes for every gate |

Pre-audit planning documents are preserved in
`claude_sessions/artemis_riot_mhd_plan/` (untracked, session-local): where local
evidence contradicted them, the ADRs record what changed and why.

The user-facing version of the capability boundary is
`doc/sphinx/src/packages/mhd.rst`, and the same boundary is printed
unconditionally at startup by `MHD::Initialize` — the rejections only speak to a
user who already asked for something unsupported.

## 8. Rules for whoever reads this next

1. **Quote the matrix, not this report.** [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md)
   is the authority, and [`OPEN_CONCERNS.md`](OPEN_CONCERNS.md) must be read
   alongside it.
2. **Donor agreement is necessary but insufficient.** A defect shared by both
   implementations would agree perfectly. That is why analytic identities and the
   Athena++ reference exist alongside donor comparison.
3. **Never present gold data generated by this implementation as validation of
   it.**
4. **A Cartesian uniform-grid result is a milestone, not completion of the donor
   capability.** Passing 2D does not certify 3D. Unvalidated is not validated.
5. **Don't relax `radiation_transport.marshak`'s threshold.** Its failure is
   `main`'s, proven bitwise.
6. **Don't delete from `OPEN_CONCERNS.md`** — move entries to its Resolved section
   with the evidence, so a later reader can re-judge that evidence.
7. **Read a fallback-counter rate before trusting any HLLD-vs-HLLE comparison on a
   new problem.**
8. **A removed rejection requires its gate genuinely passed *and* a
   `CAPABILITY_MATRIX.md` row naming the test IDs that justify the new claim.**
9. **A user requirement for an unsupported combination is an escalation point, not
   something to quietly enable.**
