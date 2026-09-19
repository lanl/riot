# Test ledger

Status vocabulary: **PASS** · **FAIL** · **NOT RUN** · **BLOCKED** · **N/A**.
A skipped test is never recorded as a pass. Thresholds are frozen before
evaluation.

Revision identity for every entry below unless stated otherwise:
RIOT base `193b3fa2a61557cb4fc761bb87de6687ca781edf` (branch
`taitano/mhd-porting`), donor `3e5aeb5`, Parthenon `928544a6d`. Toolchain:
Homebrew GCC 16.2.0, CMake 4.4.3, Open MPI 5.0.10, double precision, Kokkos
Serial host backend, `UniformCartesian`. Full detail in
[`REVISION_MANIFEST.md`](REVISION_MANIFEST.md).

---

## Gate G0 — baselines

### G0.1 RIOT unit tests (pre-change baseline)

| Field | Value |
| --- | --- |
| Command | `ctest --test-dir build --output-on-failure` |
| Build | `/Users/taitano/Documents/git/riot/build`, Release, `RIOT_ENABLE_UNIT_TESTS=ON` |
| Result | **PASS** — `100% tests passed out of 18` |
| Test count | 18 (nonzero, so the discovery gate is satisfied) |
| Wall time | 0.85 s |
| Artifact | `claude_sessions/mhd_runs/riot_hydro_baseline/unit_tests_baseline.log` |

### G0.2 RIOT hydro regression suite (pre-change baseline)

| Field | Value |
| --- | --- |
| Command | `cd tst && python run_tests.py hydro --reuse_build --save_build` |
| Build | `/Users/taitano/Documents/git/riot/tst/build`, Release, `RIOT_ENABLE_REGRESSION_TESTS=ON` |
| Gold data | version **20260831**, SHA512 verified by the build (checksum verification NOT disabled) |
| Result | **PASS** — `7 out of 7 tests passed` |
| Per test | `adiabatic_compression` 72.1 s · `carbuncle` 147 s · `gacc` 4.25 s · `linwave` 29.8 s · `linwave_mm` 38.0 s · `rt_amr` 128 s · `rt_unigrid` 264 s |
| Artifact | `claude_sessions/mhd_runs/riot_hydro_baseline/hydro_suite_baseline.log` |

Numeric outputs archived with checksums, to serve as the **bitwise reference**
for the "hydro unchanged with MHD off" requirement of every later gate:

| File | MD5 |
| --- | --- |
| `linwave-errs.dat` | `803b9e8e703eec7c0395bf7ba697413b` |
| `linwave_mm-errs.dat` | `7c64197ddce8fb316f3321c21f345826` |
| `compression.out0.hst` | `ad2b639e222329dc379342ace6537125` |

No pre-existing failures. Nothing triaged out.

### G0.3 Artemis donor MHD baseline

| Field | Value |
| --- | --- |
| Command | `cd tst && python run_tests.py mhd/brio_wu mhd/field_loop mhd/linwave mhd/orszag_tang --exe <artemis>/build/src/artemis --output_dir <riot>/claude_sessions/mhd_runs/artemis_baseline` |
| Executable | pre-existing `/Users/taitano/Documents/git/artemis/build/src/artemis` (donor's own runner options used, not RIOT CMake switches) |
| Result | **PASS** — `4 out of 4 tests passed` |
| Per test | `brio_wu` 2.43 s · `field_loop` 63.2 s · `linwave` 48.5 s · `orszag_tang` 1.87 s |
| Artifacts | 102 files / 30 MB under `claude_sessions/mhd_runs/artemis_baseline/testing/{data,figs,logs}` |

This is the reference data the port is validated against. Note `brio_wu`
exercises **all three** MHD solvers (llf, hlle, hlld). No donor defects to work
around: every donor test passes at its own thresholds, so those thresholds are
adopted verbatim rather than invented (see
[`DONOR_DELTA.md`](DONOR_DELTA.md)).

Independent (non-donor) oracle available for Brio–Wu:
`artemis/tst/scripts/mhd/athena_bw.std`, an Athena++ primitive dump at
t = 0.08 (2049 rows). Used because donor agreement alone cannot detect a defect
shared by both implementations.

**G0 verdict: PASS.** Both baselines reproduce with nonzero test counts; revision
identity and conserved-energy meaning are unambiguous (ADR-002).

---

## Gate G1 — infrastructure fixes + disabled-by-default package shell

Requirement: hydro-only behavior unchanged (bitwise where the output is
numeric), no unsupported configuration can run, face-aware stage copy proven.

| ID | Test | Threshold | Status |
| --- | --- | --- | --- |
| G1.1 | Unit tests after the `sparse_update` filter change + package shell | 18/18, no change | **PASS** — `100% tests passed out of 18` |
| G1.2 | Hydro regression suite, MHD off | 7/7 pass; numeric outputs MD5-identical to G0.2 | **PASS** — see below |
| U05 | Face-aware stage register copy over the full face extent, distinct stage values | exact | **NOT RUN** — needs a live Mesh with a registered face field; scheduled with the CT update (Stage 4) |
| N01 | Every rejected configuration fails at startup with an actionable message | must throw | **PASS** — 23/23, 2026-09-18; see the N01 section at the end. Found and fixed two dead general-PTE guards. |

### G1.2 detail — and a harness artifact worth recording

First re-run reported `5 out of 7`, with `hydro.linwave` and `hydro.linwave_mm`
failing in `analyze()`. **This was not a regression.**
`src/riot_pgen/linear_modes.cpp:388-399` *appends* to `<problem_id>-errs.dat`
when the file already exists, and `adiabatic_compression`'s `.hst` behaves the
same way. Re-running with `--reuse_build` therefore doubled both files
(25 → 49 lines; 1484 → 2968 lines) and `analyze()`'s fixed-shape `reshape` threw.

The doubling turned out to be a *stronger* check than a plain re-run, since the
appended half was produced by the modified binary while the first half came from
the baseline:

| Comparison | Result |
| --- | --- |
| `linwave-errs.dat`, appended 24 rows vs baseline 24 rows | **bitwise identical** |
| `linwave_mm-errs.dat`, appended 24 rows vs baseline 24 rows | **bitwise identical** |
| `compression.out0.hst`, first 1484 lines vs last 1484 lines | **bitwise identical** |

Confirmed with a clean run after deleting the stale accumulator files:

```
$ rm -f tst/build/src/{linwave-errs.dat,linwave_mm-errs.dat,compression.out0.hst}
$ python run_tests.py hydro/linwave hydro/linwave_mm --reuse_build --save_build
    hydro.linwave: passed; time elapsed: 29.6 s
    hydro.linwave_mm: passed; time elapsed: 38.3 s
Summary: 2 out of 2 tests passed
```

| File | MD5 after change | MD5 at baseline | Match |
| --- | --- | --- | --- |
| `linwave-errs.dat` | `803b9e8e703eec7c0395bf7ba697413b` | `803b9e8e703eec7c0395bf7ba697413b` | yes |
| `linwave_mm-errs.dat` | `7c64197ddce8fb316f3321c21f345826` | `7c64197ddce8fb316f3321c21f345826` | yes |

**G1.2 verdict: PASS.** The `sparse_update` audit was complete — no field lost its
update. Operational note for later gates: **delete the accumulator files before
any `--reuse_build` re-run**, or the failure will be misread as a regression.

### G1.3 New MHD unit tests

| Field | Value |
| --- | --- |
| Command | `ctest --test-dir tst/build --output-on-failure` |
| Result | **PASS** — `100% tests passed out of 28` (18 pre-existing + 10 new) |
| Wall time | 1.08 s |

Covers U01, U01b, U03, U04 and the face-to-cell conventions — see Gate G2 below.

### Why G1.2 is a meaningful check, not a formality

The two `sparse_update` changes were designed to be no-ops for existing physics:

- `UpdateToNextStage` gained required flags `Metadata::Cell` and
  `Metadata::Independent`. An audit of all 19 `Metadata::WithFluxes`
  registrations in `src/` confirmed every one carries both. **If the audit missed
  a field, that field stops being updated and the hydro suite changes.**
- `DeepCopyIndependentData` gained required flag `Metadata::Cell`. Today every
  non-`OperatorSplit` independent field is cell-centered, so this is also a
  no-op.

A bitwise-identical result is therefore evidence the audit was complete. Anything
else is a real regression, not tolerance drift.

Two corrections found while implementing, both recorded in ADR-003:

1. The originally planned filter `{WithFluxes, Conserved}` would have **broken**
   advected scalars (`src/scalars/scalars.cpp:62,81`) and level sets
   (`src/levelsets/levelsets.cpp:41`), which are updated here but carry no
   `Metadata::Conserved`.
2. `MakePackDescriptor<any>` accepts only a required-flag vector with AND
   semantics, so `FlagCollection` subtraction cannot be used there at all.

---

---

## Gate G2 — units, field initialization, energy conversion

Unit tests in `tst/unit/test_mhd.cpp`. All thresholds frozen before evaluation.

| ID | Test | Threshold | Result | Status |
| --- | --- | --- | --- | --- |
| U01 | Thermal energy round trip `u → E → u` over 5 states spanning zero field, aligned field, fully oblique field+flow, low beta, and supersonic flow | scaled error ≤ `1e-12` | passed at every state | **PASS** |
| U01 | `MagneticEnergyDensity` pinned against hand-computed values in both conventions (`mu0=1` → `B²/2`; `mu0=4π` → `B²/8π`) | ≤ `1e-12` | exact | **PASS** |
| U01b | Unit-convention invariance: identical physical state at `mu0=1` with field `b` and at `mu0=4π` with `B=b√(4π)` gives identical magnetic energy, fast speed, and CFL bound | rel ≤ `1e-12` | all three agree | **PASS** |
| U03 | Fast speed satisfies its defining quartic `c⁴ − (a²+vA²)c² + a²vAn² = 0` over 5 oblique states at `mu0 = 1.7` | scaled residual ≤ `1e-11` | passed | **PASS** |
| U03 | Fast speed bounded below by both `a` and `vAn`, above by the CFL bound `√(a²+vA²)` | ≤ `1e-12` slack | passed | **PASS** |
| U03 | Degenerate orientations: parallel (`cf² = max(a²,vA²)`), perpendicular (`cf² = a²+vA²`), `b→0` (`cf² = a²`), and the exactly-degenerate `a² = vA²` case | rel ≤ `1e-11`, finite | passed, no NaN | **PASS** |
| — | Face-to-cell B reproduces a uniform field and its energy exactly | ≤ `1e-12` | exact | **PASS** |
| — | Face-to-cell B averages a linear face profile (pins mean, not one-sided pickup) | ≤ `1e-12` | exact | **PASS** |
| — | Face-to-cell B retains transverse components in a collapsed direction (1D/2D physics) | ≤ `1e-12` | retained | **PASS** |
| U04 | Discrete curl of a non-separable analytic vector potential is discretely divergence free | `η = max\|D\|·Δx/max\|B\| ≤ 1e-13` | passed | **PASS** |
| U04 | Uniform field is exactly divergence free, and the curl reproduces the intended uniform field | ≤ `1e-13` / `1e-12` | passed | **PASS** |

Notes on test design, since a weak version of any of these would pass trivially:

- U03 checks the **residual of the defining quartic** rather than comparing against
  a second copy of the same closed form, so a sign or density-factor error cannot
  cancel out. `mu0 = 1.7` is used deliberately — neither 1 nor 4π — so that a
  dropped `mu0` cannot hide.
- U04 uses a deliberately non-separable, non-symmetric potential. `div(curl A) = 0`
  is an identity of the discrete operators, so it must hold for *arbitrary* A; a
  symmetric choice could mask an index error through accidental cancellation.
- U01's low-beta state (`u = 1e-3` with magnetic energy ~7) makes the round trip a
  real cancellation test rather than a trivially satisfied absolute comparison.

**Still outstanding for G2**: field initialization on a live mesh, restart
persistence of nonzero face state, and the "uniform force-free field changes no
pressure" end-to-end check. Those need the problem generators (Stage 4).

**G2 verdict: PARTIAL.** The unit-level contracts (units, energy round trip, wave
speeds, discrete curl/divergence identity, face-to-cell conventions) all pass. The
mesh-level items are not yet run and are not claimed.

---

---

## Gate G3 — MHD fluxes and wave speeds

Unit-level portion complete. `ctest` **34/34** (18 pre-existing + 16 MHD).

| ID | Test | Threshold | Result | Status |
| --- | --- | --- | --- | --- |
| U02 | HLLE consistency: equal L/R states reproduce the exact analytic MHD flux — 6 states × 3 sweep directions × both `mu0` conventions (36 combinations) | scaled ≤ `1e-12` | passed everywhere | **PASS** |
| U02 | Normal induction flux is **exactly** `0.0`, for *unequal* L/R states, in all three directions | exact `== 0.0` | exact | **PASS** |
| U02 | With `bn = 0`, transverse induction reduces to pure advection `v_n b_t` (no tension term) | ≤ `1e-12` | passed | **PASS** |
| U02 | Zero field reduces to the exact Euler flux, with pressure in the normal momentum component only | ≤ `1e-12` | passed | **PASS** |
| U02 | Invariance under cyclic relabelling of the axes: momentum and induction components permute, energy and `smax` are invariant | ≤ `1e-12` | passed | **PASS** |
| U02 | Returned signal speed bounds both `\|v_n\|` and the fast speed | ≤ `1e-12` slack | passed | **PASS** |
| G3.1 | Hydro regression suite unchanged after the MHD solver and the CFL change | 7/7, numeric outputs MD5-identical | *in progress* |

Test-design notes:

- The exact-flux reference in `ExactMHDFlux` is written **independently** of the solver,
  from the governing equations, so agreement is evidence rather than tautology. It pins
  the Maxwell stress `−b_n b_i/μ₀`, the total-pressure term, the Poynting flux
  `−b_n(v·B)/μ₀`, and the induction signs simultaneously.
- Consistency is checked at **both** `μ₀ = 1` and `μ₀ = 4π`, so a `μ₀` factor applied in
  one term but not another cannot pass.
- The low-beta state (`u = 1e-3`, `|B|² ≈ 15`) makes the energy-flux comparison a real
  cancellation test.
- Cyclic-relabelling invariance is a pure index-bookkeeping test. It is cheap and is the
  most reliable way to catch a mis-permuted component, which would otherwise be a
  multidimensional-only bug surfacing much later.

**Outstanding for G3** (needs the flux-path wiring, not yet written): 1D Brio–Wu against
the donor, and the end-to-end zero-field hydro limit H02.

**G3 verdict: PARTIAL.** The solver itself is verified at unit level. It is not yet
called by anything.

---

## Gates G4–G5

**NOT RUN.** Definitions and frozen thresholds are in the approved plan.

| Gate | Scope | Status |
| --- | --- | --- |
| G4 | Multi-D CT: div B at roundoff, field loop, Orszag–Tang, 3D Alfvén all permutations, MPI (P01) | **PASS** — div B, 2D/3D field loop, Orszag–Tang (G4.4), CPAW order of accuracy (G4.5) and MPI (P01) all PASS. Oblique CPAW propagation not covered; see G4.5. |
| G5 | HLLD/LLF coverage, degeneracies, reconstruction certification, restart (R01) | NOT RUN |

## Permanently out of scope for this port (ADR-004)

Recorded so the absence is never mistaken for an untriaged failure.

| Item | Status |
| --- | --- |
| AMR / static refinement (A01–A03) | **N/A** — rejected at startup; no donor oracle exists, since Artemis's MHD tests never exercise AMR |
| Curvilinear geometry (C01, C02) | **N/A** — rejected at startup |
| Non-ideal EOS (E01) | **N/A** — rejected at startup |
| Two-temperature (T01, T02) | **N/A** — rejected at startup |
| Multi-material (S01, S02) | **N/A** — rejected at startup; needs scientific review |
| Source-package coupling (X01, X02) | **N/A** — every package rejected with MHD |
| GPU / CUDA comparison | **NOT RUN** — not available on this machine; kernels are written to Kokkos portability rules and reviewed, but unvalidated is not validated |

---

## Stage 3 + Stage 4 results (2026-09-18, commit `dced139`)

### G3/G4.1 — Brio & Wu shock tube vs the Athena++ reference

Independent oracle (not the donor): `artemis/tst/scripts/mhd/athena_bw.std`, an Athena++
`prim` dump at t = 0.08 on 2048 zones. RIOT run at 512 zones, `mhd_hlle`, PLM, CFL 0.4,
`mhd/mu0 = 1`, gamma = 2, on 2 mesh blocks.

```
build/src/riot -i inputs/mhd/brio_wu.rin
```

| Metric | Frozen threshold | Measured | Verdict |
| --- | --- | --- | --- |
| normalized L1, rho | 3.8e-3 (donor's own) | 4.17e-3 | marginal |
| normalized L1, P | 3.8e-3 | 3.86e-3 | marginal |
| normalized L1, Bcc2 | 3.8e-3 | 3.82e-3 | **PASS** |
| normalized L1, vel1 | — | 2.96e-2 | recorded |
| normalized L1, vel2 | — | 2.42e-2 | recorded |
| `max abs(divB)` | 1e-10 | **0.0 exactly** | **PASS** |
| `max abs(Bx - 0.75)` | 1e-12 | **0.0 exactly** | **PASS** |
| out-of-plane `max abs(Bz)` | 1e-12 | **0.0 exactly** | **PASS** |
| out-of-plane `max abs(vz)` | 1e-12 | **0.0 exactly** | **PASS** |
| rho > 0, P > 0 | required | min rho 0.1163, min P 0.0865 | **PASS** |

The two "marginal" entries sit within 10% of a threshold that was set for the donor at its
own resolution, solver, and reconstruction — this comparison is 512 vs 2048 zones against a
different code, so agreement at the few times 1e-3 level is the expected outcome and the
threshold is not yet a like-for-like criterion. **It must be re-frozen against a matched
Artemis run before it can be treated as a pass/fail gate.** Not lowered — matched. The
velocity L1s are larger because the normalization is the mean of a field that is zero over
most of the domain; they need a different normalization to be meaningful.

The four exact-zero invariants are the load-bearing results here: they are the ones that
would break under an indexing, orientation, or normal-B-substitution error.

### G4.2 — 2D advected field loop, multi-block

Gardiner & Stone (2005) field loop: 128 x 64 on a 2 x 1 periodic box in **4 mesh blocks**
(64 x 32 each), v = (2, 1), one full diagonal crossing to t = 1, `amp = 1e-3`,
`r_loop = 0.3`, gamma = 5/3. Field initialized as the discrete curl of A3.

```
build/src/riot -i inputs/mhd/field_loop.rin
```

| Metric | Frozen threshold | t = 0 | t = 1 | Verdict |
| --- | --- | --- | --- | --- |
| normalized divergence `eta` | 1e-10 | 2.28e-15 | 5.27e-15 | **PASS** |
| `max abs(divB)` | — | 1.46e-16 | 3.69e-16 | recorded |
| magnetic energy retained | >= 0.5 | 1.0 | **0.871** | **PASS** |
| axial field `max abs(B3)` | 1e-10 | 0.0 | **0.0 exactly** | **PASS** |
| total mass | 1e-10 rel | 8.1920000000e+03 | 8.1920000000e+03 | **PASS** (exact) |
| total energy | 1e-10 rel | 3.2768000567e+04 | 3.2768000567e+04 | **PASS** (11 digits) |

This is the strongest evidence in the port so far. `eta` at 5e-15 across four mesh blocks
means the shared-face exchange is single-valued and the edge EMFs cancel algebraically at
block boundaries — the failure mode that a single-block run cannot detect. Energy retention
of 0.871 rules out an EMF averaging error, which shows up as diffusive decay.

### Coverage this does NOT establish

- **3D.** The `E1` edge EMF only takes its full upwind form when `three_d`; in 2D it uses
  the collapsed branch. So `UpwindEMF<X1DIR>` is currently unexercised. A 3D circularly
  polarized Alfven wave in all direction permutations is required before 3D can be claimed.
- **MPI.** Everything above is serial. Test P01 (1/2/4 ranks, varied decompositions) has not
  been run, so a missing task dependency could still be hidden by serial execution order.
- **Shocks with a strong field in more than 1D.** Orszag-Tang is not yet ported.
- **Restart** of face state.
- **N01**, the startup-rejection matrix, has still not been run *as of this G4.2 entry*. It
  ran later on 2026-09-18 and passed 22/22 — see the N01 section at the end of this file.

### H01 (re-run) — hydro bitwise unchanged: resolved by direct A/B, not by stored MD5

A `hydro/linwave` run against the MD5s recorded for G0.2 initially appeared to differ:
`linwave-errs.dat` `0ca31bde...` vs the recorded `803b9e8e...`, with `max abs diff = 1e-15`
and 226/240 entries byte-identical. Two candidate explanations: my changes perturbed hydro
at roundoff, or the stored baseline came from a differently configured build.

Moving the MHD reconstruction scratch off the hydro path (commit `9ece176`) did **not**
change the MD5, ruling out the scratch-footprint/chunking hypothesis.

Settled by a controlled A/B with the build configuration held fixed: the pre-port commit
`193b3fa` was checked out into a git worktree (submodules symlinked from the main tree, so
the dependency SHAs are provably identical) and configured with the *same* cmake arguments
as the current `tst/build`, then both binaries were run on the same input.

```
git worktree add /tmp/riot_base 193b3fa
# symlink external/* from the main tree, then configure with identical args
riot -i inputs/linear_modes/linear_modes.rin parthenon/time/nlim=1000 \
     problem/amp=1.0e-6 problem/nperiod=1 problem/wave_flag=0 problem/vflow=0.0 \
     parthenon/output1/dt=-1.0
```

| Comparison | Result |
| --- | --- |
| `linear_modes-errs.dat` | **byte-identical**, md5 `d4098ae81dfc5224a234489b4b93957d` both |
| `c.c.bulk.rho` | **exact**, max abs diff 0 |
| `c.c.bulk.momentum` | **exact**, max abs diff 0 |
| `c.c.bulk.velocity` | **exact**, max abs diff 0 |
| `c.c.bulk.pressure` | **exact**, max abs diff 0 |
| `c.c.bulk.total_material_energy` | **exact**, max abs diff 0 |
| `c.c.mat.rho_0` | **exact**, max abs diff 0 |

**H01 verdict: PASS.** Every hydro field is bitwise identical to the pre-port commit. The
`.phdf` file MD5s differ, but that is HDF5 container metadata (timestamps, version strings,
the embedded input deck), not numerics — which is why the comparison is done on the array
contents.

**Operational correction, and the reason the earlier reading was misleading:** the MD5s
recorded under G0.2 are **specific to the build configuration that produced them** and are
not a valid cross-session invariant. `tst/build` had to be reconfigured by hand this session
(`run_tests.py` deletes it without `--save_build`, and reconfigures with AppleClang without
`--reuse_build`, which cannot compile RIOT), and a differently configured build changes the
last digits of a quantity with ~10 orders of cancellation. **For "hydro unchanged", run the
A/B above rather than comparing against a stored MD5.** A stored hash looks like the
stronger check and is actually the weaker one, because it silently conflates a build
difference with a code regression.

---

## G4.3 — 3D constrained transport: the axial-field test

This closes the coverage gap flagged after G4.2. In 2D, `MHD::UpwindEMF<X1DIR>` and
`<X2DIR>` are never reached — `AssembleEdgeEMF` takes the collapsed branch for any edge
whose basis has a collapsed E3 direction — so the full upwind reconstruction had only ever
run for the `X3DIR` edge. A 3D run reaches all three.

**Why this particular test rather than a generic 3D run.** Gardiner & Stone (2005) section
5.4: advect a 2D field loop in 3D with a velocity component *along* the loop axis. With the
axis on x3, B = (B1, B2, 0) and v = (v1, v2, v3), so

```
E1 = v3 B2      E2 = -v3 B1      E3 = v2 B1 - v1 B2
d_t B3 = -(d_1 E2 - d_2 E1) = v3 (d_1 B1 + d_2 B2) = v3 * div B = 0
```

All three EMFs are nonzero, and the axial component must stay exactly zero — but only
because E1 and E2 cancel against each other. Any inconsistency between how the two are
assembled appears immediately as a spurious axial field, with no other term to absorb it.
So `max abs(B_axial)` is a direct, quantitative probe of exactly the code paths 2D could not
reach. A generic 3D run would exercise the branches without testing them against anything.

`mhd_field_loop/loop_axis` rotates the vector potential and the two nonzero face components
cyclically, so each pair of edge directions is checked in turn.

64 x 32 x 32 on a 2 x 1 x 1 periodic box in **8 mesh blocks**, v = (2, 1, 1), one full
crossing in every direction (tlim = 1), `amp = 1e-3`, `r_loop = 0.3`, gamma = 5/3.

```
build/src/riot -i inputs/mhd/field_loop_3d.rin mhd_field_loop/loop_axis={1,2,3}
```

| Run | `max abs(B_axial)/max abs(B)` | `eta` | Emag retained | `dE/E` |
| --- | --- | --- | --- | --- |
| 8 blocks, axis = 1 | 5.80e-15 | 5.02e-15 | 0.7638 | 2.0e-16 |
| 8 blocks, axis = 2 | 1.39e-15 | 3.92e-15 | 0.7261 | 7.9e-16 |
| 8 blocks, axis = 3 | 1.44e-15 | 4.06e-15 | 0.7261 | 5.9e-16 |
| 1 block,  axis = 3 | 1.44e-15 | 4.06e-15 | 0.7261 | — |

Frozen thresholds: axial field <= 1e-10 normalized, `eta` <= 1e-10, Emag retained >= 0.5.
**All three permutations PASS**, with the axial field at roundoff rather than merely small.

Axis 2 and 3 give identical energy retention while axis 1 retains more. That is expected,
not an anomaly: the in-plane advection speeds are {1,2} for axes 2 and 3 but {1,1} for axis
1, so axis 1 sees less numerical diffusion. All three perpendicular planes have the same
grid spacing (1/32), which is why axes 2 and 3 agree exactly.

### P01 (partial) — decomposition invariance in 3D

The same case at 1 block versus 8 blocks, compared after reassembling the 8 blocks onto the
global grid via `LogicalLocations`:

| Field | Result |
| --- | --- |
| `c.c.bulk.magnetic_field` | **exact**, max abs diff 0 |
| `c.c.bulk.rho` | **exact** |
| `c.c.bulk.velocity` | **exact** |
| `c.c.bulk.total_material_energy` | **exact** |
| `c.c.bulk.magnetic_energy` | **exact** |
| `c.c.bulk.div_magnetic_field` | **exact** |

**Bitwise identical.** Block decomposition does not affect the result at all, which covers
shared-face single-valuedness and edge-EMF agreement across block boundaries in 3D.

**Still not covered:** this is all single-rank. P01 proper needs 2 and 4 MPI ranks — with
one rank, a missing task dependency can still be masked by serial execution order, and the
communication path itself is only exercised for on-rank neighbours.

## P01 (complete) — MPI rank invariance, 2026-09-18, commit `5e6c08b`

Binary `build/src/riot` (Release, gcc-16, MPI on open-mpi 5.0.10), scratch dir `/tmp/mhdmpi`,
analysis by `claude_sessions/mhd_runs/analyze_field_loop.py`. Thresholds were frozen in
CHECKPOINT before the runs: the per-cell invariants must hold at roundoff at every rank
count, and cross-rank agreement was *predicted* to be bitwise but not guaranteed.

### P01.1 — 3D field loop (`inputs/mhd/field_loop_3d.rin`, 8 mesh blocks), rank sweep

```
for N in 1 2 3 4 5 8; do
  mpiexec -n $N build/src/riot -i inputs/mhd/field_loop_3d.rin \
    parthenon/job/problem_id=fl3d_r$N
done
```

Every rank count reached `tlim=1.0` at **cycle=630** with `dt` agreeing to all printed
digits, so the CFL global reduction did not reorder — which is why bitwise agreement was
attainable here rather than merely hoped for.

| ranks | axial | eta | emag_retained | dE/E | dM/M | vs 1 rank |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | — |
| 2 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | **bitwise identical** |
| 3 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | **bitwise identical** |
| 4 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | **bitwise identical** |
| 5 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | **bitwise identical** |
| 8 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | 5.92e-16 | 3.33e-16 | **bitwise identical** |

All 7 compared fields exact (max abs diff 0), `div_magnetic_field` included. The numbers also
reproduce the single-rank G4.3 row exactly, so the analysis path is consistent between the
two sessions.

`-n 3` and `-n 5` matter independently of 2/4/8: 8 blocks over 3 or 5 ranks gives **uneven
block ownership**, so the neighbour pattern is no longer the symmetric one that an even split
produces, and a rank-boundary bug that happens to cancel in a symmetric decomposition cannot
hide.

### P01.2 — all three loop axes under MPI

With `loop_axis=3` the field is `(B1, B2, 0)`, so the **F3 face exchange carries only zeros**
and a defect in it would be invisible. Sweeping the axis puts nonzero data on each face
direction in turn. 1 rank vs 8 ranks, same input, `mhd_field_loop/loop_axis` overridden:

| loop_axis | axial | eta | emag_retained | 1 vs 8 ranks |
| --- | --- | --- | --- | --- |
| 1 | 5.800127e-15 | 5.020559e-15 | 0.7637599 | **bitwise identical** |
| 2 | 1.394086e-15 | 3.919490e-15 | 0.7260742 | **bitwise identical** |
| 3 | 1.441305e-15 | 4.055658e-15 | 0.7260742 | **bitwise identical** |

Axial field at roundoff in every permutation confirms `MHD::UpwindEMF<X1DIR>` and `<X2DIR>`
are correct *across rank boundaries*, not only within a rank.

### P01.3 — 2D field loop (`inputs/mhd/field_loop.rin`), the collapsed-EMF branch

2D is not a subset of 3D here: with x3 collapsed, `MHD::AssembleEdgeEMF` dispatches to
`CollapsedEdgeEMF` rather than `UpwindEMF`, so this is a separate code path. 128x64 on 4
blocks, 1 rank vs 4 ranks, cycle=894 both:

```
axial 0.000000e+00   eta 5.267631e-15   emag_retained 8.711372e-01
dE/E 2.22e-16        dM/M 2.22e-16      ->  BITWISE IDENTICAL
```

`eta` and `emag_retained` match the serial G4.2 values (5.268e-15, 0.8711).

### P01.4 — Brio & Wu under MPI, with shocks

Smooth advection can miss a defect that only appears where a limiter activates at a rank
boundary. 512 zones forced onto 4 mesh blocks (`parthenon/meshblock/nx1=128`), 1 vs 4 ranks,
cycle=399 both. All 7 fields exact, and the 1D structural invariants hold **exactly**, not
approximately:

| quantity | 1 rank | 4 ranks |
| --- | --- | --- |
| `max abs(Bx - 0.75)` | 0.000e+00 | 0.000e+00 |
| `max abs(Bz)` | 0.000e+00 | 0.000e+00 |
| `max abs(vz)` | 0.000e+00 | 0.000e+00 |
| `max abs(divB)` | 0.000e+00 | 0.000e+00 |
| `min rho` | 0.116264 | 0.116264 |
| `min P` | 0.086516 | 0.086516 |

Positivity holds with no floor activation, and out-of-plane quantities stay identically zero.

### Verdict

**P01 PASS.** Result is independent of rank count (1/2/3/4/5/8), of whether blocks divide
evenly across ranks, of loop axis, of dimensionality (2D collapsed branch and 3D upwind
branch), and of whether the flow contains shocks — bitwise, in all cases. This exercises the
face `FillGhost` exchange, edge-EMF agreement across rank boundaries, and the task-dependency
graph under genuinely concurrent execution.

**Limits of this result.** Uniform grid, no AMR (rejected in scope), so no flux/EMF
correction at coarse/fine boundaries is tested. Single node, shared memory — no interconnect.
And bitwise agreement here depended on `dt` being reduction-order-independent in these cases;
a problem where the CFL reduction does reorder would legitimately differ at roundoff without
being a bug.

## G4.4 — Orszag & Tang vortex: shocks plus multi-dimensional CT

Added `mhd_orszag_tang` (`src/riot_pgen/mhd_problems.cpp`) and
`inputs/mhd/orszag_tang.rin`. 64x64 unit square on 4 mesh blocks, gamma = 5/3, mu0 = 1,
PLM + `mhd_hlle`, CFL 0.4, to t = 0.5 in 370 cycles. Analysis:
`claude_sessions/mhd_runs/analyze_orszag_tang.py`.

**Why this test was worth adding even after G4.3 and P01 passed.** Every other MHD case in
the port is either one-dimensional (Brio-Wu) or smooth for all time (the field loop, whose
exact solution is pure advection). This is the only one where constrained transport has to
hold div B = 0 *while limiters are firing and the EMF stencil straddles discontinuities*. A
CT defect masked by smoothness, or a shock-capturing defect masked by one-dimensionality,
has nowhere left to hide.

**Deviation from the donor, stated up front:** `artemis/inputs/orszag_tang/orszag_tang.in`
runs HLLD, which this port does not have yet (Stage 5), so this deck runs `mhd_hlle`. That is
why the acceptance criteria are conserved global quantities and the local energy split rather
than a pointwise reference. None of the criteria is solver-specific — they are conservation
and consistency statements — but a *failure* here would have to be checked against the solver
difference before being attributed to the port.

Thresholds are the donor's own (`artemis/tst/scripts/mhd/orszag_tang.py`), adopted verbatim
and frozen before evaluation.

### Results

| Quantity | t = 0 | t = 0.5, 1 rank | t = 0.5, 4 ranks | Threshold |
| --- | --- | --- | --- | --- |
| all finite | yes | yes | yes | no NaN/Inf |
| `min rho` | 2.21048532e-01 | 1.01171526e-01 | 1.01171526e-01 | > 0 |
| `min P` | 1.32629119e-01 | 3.65275291e-02 | 3.65275291e-02 | > 0 |
| `max abs(divB)` | **0.0** | 1.98951966e-13 | 1.98951966e-13 | <= 1e-10 |
| `mean rho` | 2.21048532e-01 | 2.21048532e-01 | 2.21048532e-01 | rtol 1e-12 of 25/(36 pi) |
| `max abs(mean momentum)` | 1.39e-17 | 6.94e-18 | 6.94e-18 | <= 1e-10 |
| `mean E` | 3.49256681e-01 | 3.49256681e-01 | 3.49256681e-01 | rtol 1e-8 of 0.34925668067288668 |
| `max abs(E - u - KE - Emag)` | 8.33e-17 | 1.01e-16 | 1.01e-16 | <= 1e-10 |
| `mean u` | 1.98943679e-01 | 2.50419244e-01 | 2.50419244e-01 | >= U0 + 1e-3 E0 at t>0 |

**PASS at both rank counts, and 1 vs 4 ranks is bitwise identical** across all seven fields.

Positivity holds with margin (`min P` is 28% of its initial value, not clinging to a floor),
so no clipping was needed — which matters because a floored solve can pass conservation
checks while hiding a defect.

### Reading the divergence number

`max abs(divB) = 1.99e-13` is larger than the field loop's `5e-15`, and that is a units
artifact rather than a degradation: this field is O(b0) = O(0.28) whereas the loop field is
O(1e-3), and `div_magnetic_field` carries units of B/length. Normalized the same way as
`eta` elsewhere in this ledger, `1.99e-13 * dx / max|B| = 1.99e-13 / 64 / 0.28 ~ 1.1e-14` —
i.e. roundoff, consistent with every other CT result here.

### The t = 0 column is a pgen check, not a formality

Every criterion except shock heating is meaningful at t = 0, and is a *stronger* statement
there: the initial mean of `sin^2` over a cell-centered grid spanning whole periods is exactly
1/2, so the analytic box means `rho0`, `E0 = P0/(gamma-1) + rho0 v0^2/2 + b0^2/2` are exact for
the discrete initial state, not accurate to O(dx^2). They are matched, and `max abs(divB)` is
**identically zero**. That confirms the field really is divergence free by construction and
that `AddMagneticEnergyToTotal` produced exactly the ADR-002 total. The shock-heating
criterion is the one check that cannot apply at t = 0 (no shocks yet); the analysis script
takes `expect_heating=False` for that snapshot rather than reporting a spurious failure.

### Why no vector potential here

Rule 3 of the pgen contract asks for a discrete curl "where the field is not trivially
divergence free by construction". This field is: `B1 = -b0 sin(2 pi y)` depends only on x2 and
`B2 = b0 sin(4 pi x)` only on x1, so on the staggered mesh the two x1-faces of any cell carry
identical B1 and the two x2-faces identical B2. Each term of the discrete divergence vanishes
separately and exactly at any resolution — which the t = 0 row confirms empirically. Writing B
pointwise also matches the donor exactly, whereas a node-differenced potential would give the
O(dx^2) finite-difference sine instead.

## G4.5 — Circularly polarized Alfven wave: order of accuracy

Added `mhd_cpaw` (`src/riot_pgen/mhd_problems.cpp`), `inputs/mhd/cpaw.py`, and
`claude_sessions/mhd_runs/analyze_cpaw.py`. rho = 1, P = 0.1, `b_par` = 1, amplitude 0.1,
mu0 = 1, PLM + `mhd_hlle`, CFL 0.4. The wave direction spans [0,1] with the two transverse
directions kept short (4 cells over 0.125) but **nondegenerate**, so the run is genuinely 3D
and every transverse face exchange and EMF component is live while costing almost nothing.

**Why this test rather than the donor's linwave.** The donor has no CPAW; its `linwave` uses 7
linear eigenmodes at amplitude 1e-6, which probes only the linearized system. The CPAW is an
**exact nonlinear** solution — the perpendicular field rotates at constant magnitude, so |B|
is uniform, the magnetic pressure gradient vanishes identically, and the wave translates at
v_A with no steepening at amplitude 0.1. It is therefore an *analytic* oracle (class 1 in the
verification plan) rather than a donor comparison, and it tests order of accuracy at finite
amplitude, which a linear mode cannot.

**Why the port needed it at all.** Every earlier MHD test checks either an identity that holds
at roundoff (div B, axial field, conservation) or agreement with a reference at ONE
resolution. None of them would notice a scheme that is stable, conservative, divergence free
and merely *first* order — the signature of a subtly wrong EMF average or reconstruction.

`tlim` is one full period (v_A = 1, wavelength 1), so the exact solution at the final time is
the initial condition and L1 is measured directly against the t=0 snapshot: no analytic
evaluator at t>0, no interpolation. L1 is the mean absolute error of the two perpendicular B
components and the two perpendicular velocities, normalized by the initial amplitude.

Frozen threshold, adopted from the donor's linwave criterion: successive-resolution ratio
<= 0.35 (equivalently observed order >= 1.51).

### Convergence, wave along x1

| N | L1 | ratio | observed order | eta(divB) | \|B\| spread |
| --- | --- | --- | --- | --- | --- |
| 16 | 5.690184e-02 | — | — | **0.0** | 1.344e-03 |
| 32 | 1.938062e-02 | 0.3406 | 1.554 | **0.0** | 7.566e-04 |
| 64 | 5.822398e-03 | 0.3004 | 1.735 | **0.0** | 2.936e-04 |
| 128 | 1.637264e-03 | 0.2812 | 1.830 | **0.0** | 1.307e-04 |

**PASS.** Observed order rises monotonically 1.55 → 1.73 → 1.83 toward 2, which is the
expected behaviour for PLM on a smooth profile with extrema: the limiter clips at the sine
peaks and costs local order most severely at coarse resolution. The important reading is the
*trend*: an asymptotically second-order scheme approaches 2 from below, whereas a scheme that
is genuinely first order somewhere would plateau. The 16→32 ratio of 0.3406 sits close to the
0.35 threshold and would be worth investigating on its own; the trend is what settles it.

`eta` (normalized max|div B|) is **identically zero** at every resolution, and the parallel
field component drifts by exactly zero — both are algebraic identities of this setup, and
neither is merely small.

### Direction permutations and reversed propagation

`wave_dir` = 1/2/3 rotates the wave onto each axis; `travel_sign` = -1 reverses propagation,
which is what would catch an induction-term sign error that a standing pattern hides.

| Case | L1(32) | L1(64) | L1(128) | ratios | Verdict |
| --- | --- | --- | --- | --- | --- |
| wave_dir 1 | 1.938062e-02 | 5.822398e-03 | 1.637264e-03 | 0.3004, 0.2812 | PASS |
| wave_dir 2 | 1.938062e-02 | 5.822398e-03 | 1.637264e-03 | 0.3004, 0.2812 | PASS |
| wave_dir 3 | 1.938062e-02 | 5.822398e-03 | 1.637264e-03 | 0.3004, 0.2812 | PASS |
| wave_dir 1, reversed | 1.938062e-02 | 5.822398e-03 | 1.637264e-03 | 0.3004, 0.2812 | PASS |

The error is **identical to every printed digit across all three axes**. That is exact
rotational isotropy of the discretization, and it is a strong statement: any asymmetry in how
the three EMF components or face directions are assembled would show up here as a
direction-dependent error. Verified that this is not an artifact of comparing the same data —
the assembled arrays have genuinely different shapes per orientation, `(3,4,4,64)` vs
`(3,4,64,4)` vs `(3,64,4,4)`; the perpendicular amplitude decays identically 0.1 → 0.098856 in
each; `b_par` is uniform to exactly 0 in each; and the reversed run differs from the forward
one by `max|dV| = 1.973e-01`, i.e. 2 v_perp, as a sign flip requires.

### MPI

64 cells on 4 blocks, 1 rank vs 4 ranks: **bitwise identical** across all seven fields.

### Limitation, stated rather than glossed

This is an **axis-aligned** wave. A wave propagating along a box diagonal additionally couples
all three EMF components simultaneously and is a strictly stronger test of the CT
discretization; it is NOT covered here. The axis sweep above establishes that each direction
is individually correct and mutually consistent, not that oblique propagation is.

---

## Gate G5 (in progress)

### G5.1 — HLLD solver

`MHD::lr_to_flux_mhd_hlld` added to `src/mhd/riemann_mhd.hpp`, ported from
`artemis/src/utils/fluxes/riemann/hlld.hpp:40-430` (single-material branch; the donor's
species >= 1 HLLC fallback is not carried over). Selected with `hydro/riemann = mhd_hlld`.
Every robustness guard from DONOR_KERNELS.md section 13 is transcribed with its exact
constants.

Two deliberate structural notes, both recorded because they are the parts a future reader is
most likely to "fix" wrongly:

- **HLLD works in the rotated (normal, t_a, t_b) frame**, unlike the HLLE above which is
  written componentwise in global coordinates. This is forced: the rotational-discontinuity
  jump mixes the two transverse components through `sgn(b_n)`, so there is no componentwise
  form. The permutation is the cyclic right-handed one, matching `RiotUtils::DirBasis`.
- **Its wave-speed estimate is `min/max(v_n -+ c_f)`, not the Roe average** the HLLE uses.
  That is the donor's choice, kept because HLLD's star states and their degeneracy
  tolerances were derived and tuned against that bound.

Restricted to an ideal gas, as the donor is (`hydro.cpp` rejects `mhd_hlld` with
`use_general_pte`). HLLE carries no such restriction — it needs only the bulk modulus.

#### Unit tests: `ctest` 41/41 (34 before, 7 new)

| Test | What it pins |
| --- | --- |
| HLLD consistency | equal states give the EXACT MHD flux, all 6 `kStates`, all 3 directions, mu0 = 1 and 4pi |
| normal induction flux | exactly `0.0`, all 3 directions, asymmetric states |
| cyclic relabelling | flux components permute with the axes — the sharpest check on the rotated frame |
| degeneracy consistency | 10 states that trip each guard, all 3 directions, both mu0 |
| finiteness | all 100 ordered pairs of degenerate states: no NaN/Inf, `smax > 0`, normal induction exactly 0 |
| stationary contact | zero mass flux and zero transverse induction flux |
| HLLD != HLLE | the two are genuinely different fluxes on a real jump |

That last one exists because no consistency test can detect an over-eager degeneracy guard
that silently routes every call to the HLLE fallback — which would leave HLLD "correct" and
useless.

**Two test-authoring errors found and fixed while writing these, both worth recording because
each looked like a solver bug:**

1. *Tolerance model.* The degeneracy consistency test first failed with induction error
   3.8e-12 against a 1e-12 tolerance, then energy error 1.7e2. Neither is a solver defect:
   HLLD forms star states as differences of conserved-state-sized quantities multiplied by the
   wave speeds, so its roundoff floor is `eps * |s| * |U|`, not `eps * |flux|`. Near-vacuum
   (rho = 1e-8) gives `s ~ 1.7e4` and `eps*s*|B| ~ 3.8e-12`; the 1e6 transverse field gives
   `E ~ 1e12`, `s ~ 1.4e6` and `eps*s*E ~ 3e2`. Both measured values sit within a factor of
   two of those predictions, i.e. the arithmetic is as accurate as double precision allows.
   The tolerance now states that model explicitly rather than being loosened to fit. Honest
   limit: for the two 1e6-field states (beta ~ 1e-12, far outside the certified regime) the
   resulting absolute tolerance is large and consistency there is conditioning-limited, not a
   sharp check — their value is as NaN regression cases, which the finiteness test asserts
   strictly.
2. *A wrong premise.* A test asserted HLLD and HLLE must differ on a stationary contact. They
   agree there **exactly**, and that is correct: with `v = 0` and `P`, `b_n`, `|b_t|`
   continuous, `ptl == ptr` so `sm = 0`, and HLLD's `sm >= 0` branch reduces to
   `fl_mx + ptl` — the same value HLLE averages from two identical inputs. The
   solvers-differ check moved to a state with a real velocity jump.

#### G5.2 — Brio & Wu, HLLE vs HLLD against the independent Athena++ reference

Analysis: `claude_sessions/mhd_runs/analyze_brio_wu.py`. 512 zones on 2 blocks, t = 0.08,
gamma = 2, mu0 = 1, PLM.

**A coordinate reconciliation was required and is the reason to keep this in a script.** The
Athena++ reference `athena_bw.std` is on `[0,1]` with the discontinuity at x = 0.5 (2048
zones, first cell center 1/4096); `inputs/mhd/brio_wu.py` uses `[-0.5, 0.5]` with the jump at
x = 0. Comparing without the 0.5 shift gives a normalized L1 of ~0.45 on every field —
large enough to read as a catastrophic solver bug when it is purely a frame mismatch. The
verification plan lists this reconciliation as a required pre-comparison step; this is the
case it means.

Structural invariants, both solvers, **all exactly zero**:
`max|Bx - 0.75| = 0`, `max|Bz| = 0`, `max|vz| = 0`, `max|divB| = 0`; `min rho` and `min P`
positive with margin.

Normalized L1 against Athena++ (mean absolute difference / reference dynamic range):

| field | HLLE | HLLD | ratio HLLD/HLLE |
| --- | --- | --- | --- |
| rho | 2.659044e-03 | 1.723601e-03 | 0.648 |
| press | 2.280626e-03 | 1.523292e-03 | 0.668 |
| vx | 5.799217e-03 | 4.633461e-03 | 0.799 |
| vy | 4.006543e-03 | 2.749096e-03 | 0.686 |
| By | 1.689883e-03 | 1.186258e-03 | 0.702 |

**HLLD is more accurate on every field.** This relative statement is what is asserted, and it
is immune to the calibration problem below: HLLD resolves five waves including the contact and
the two rotational discontinuities, and the Brio-Wu compound structure (a slow shock attached
to a rotational discontinuity) is exactly where that shows.

**The donor's `_profile_tolerance = 3.8e-3` is NOT used as a threshold here**, and should not
be. It is calibrated against `brio_wu.std`, which is Artemis's *own* gold file, at the donor's
resolution and solver. Against a third-party reference it is the wrong yardstick; it is
printed for orientation only. An absolute threshold for RIOT still needs to be frozen against
a matched run — unchanged from the earlier note in this ledger.

#### G5.3 — HLLD does not disturb constrained transport

| Case | HLLE | HLLD | Reading |
| --- | --- | --- | --- |
| 2D field loop `eta` | 5.267631e-15 | 5.930142e-15 | both roundoff |
| 2D field loop axial field | 0.0 | **0.0** | exact |
| 2D field loop `emag_retained` | 0.8711372 | **0.8751974** | HLLD less diffusive |
| 2D field loop `dE/E` | 2.22e-16 | **0.0** | — |
| Orszag-Tang | PASS | **PASS** | `max|divB|` 3.27e-13, all criteria met |
| Orszag-Tang mean `u` | 0.2504192 | 0.2469889 | HLLD dissipates less into heat |
| CPAW order (N = 16..128) | 1.55/1.73/1.83 | 1.56/1.73/1.84 | indistinguishable |

The field-loop energy retention and the Orszag-Tang heating both move in the direction lower
numerical diffusion predicts, which is independent corroboration that the new solver is doing
what it claims rather than merely running.

CPAW barely distinguishes the two solvers, and that is expected rather than disappointing:
for a smooth wave with no contact or shock the error is dominated by PLM reconstruction, not
by the flux function. It is why Brio-Wu is the solver test and CPAW is the order test.

#### G5.4 — hydro unchanged with MHD off, after the HLLD addition

`src/hydro/{riemann.hpp,hydro.cpp,calculate_fluxes.cpp}` were all touched, so this is
re-checked rather than assumed. The change is additive — one new enum value, one new `switch`
case, one new templated free function, one new startup guard — so no hydro code path is
altered.

```
cd tst && python run_tests.py hydro --reuse_build --save_build
```

**PASS — 7 out of 7.** `adiabatic_compression` 75.4 s · `carbuncle` 159 s · `gacc` 4.92 s ·
`linwave` 41.7 s · `linwave_mm` 51.5 s · `rt_amr` 145 s · `rt_unigrid` 304 s.

Unit tests `ctest` **41/41**.

### G5.5 — Reconstruction certification with MHD

Every MHD result before this entry used `recon=plm` and nothing else, leaving four of the five
`RiotReconstruction::Type` modes entirely unexercised with a magnetic field. All five are now
certified. Stencil widths are `CONSTANT 0, PLM 1, PPM4 2, WENO5 2, MP5 2`
(`src/reconstruction/reconstruction.hpp:32-36`) and the requirement is
`nghost > stencil_width` (`src/hydro/hydro.cpp:86`), so the minimum is 1/2/3/3/3.

#### Ghost budget: sufficient, and proven rather than argued

`MHDFluxIndexSpace` widens the flux bounds by one layer in the two directions TRANSVERSE to
the sweep. The comment there argues this keeps the requirement at `nghost > stencil_width`
rather than raising it — but the fit is exact, with zero margin, so it was tested instead of
trusted. Each mode was run at its minimum `nghost` and again with one more layer, on the
4-block 2D field loop so block boundaries are crossed:

| mode | min nghost | min vs min+1 |
| --- | --- | --- |
| CONSTANT | 1 | **bitwise identical** |
| PLM | 2 | **bitwise identical** |
| PPM4 | 3 | **bitwise identical** |
| WENO5 | 3 | **bitwise identical** |
| MP5 | 3 | **bitwise identical** |

An extra halo layer changes nothing, so the minimum really is sufficient. Had it not been, the
extra transverse flux layer would have been built from stale ghost data and `div B` would have
broken at the block boundaries — which is why a multi-block case was used.

**Negative test.** Every mode run one ghost BELOW its minimum aborts with
`Must have enough ghosts for reconstruction stencil <NAME>` — an actionable message naming the
mode. Checked for all five; none silently produced a wrong answer.

#### CT invariants, 2D field loop (4 blocks, one full crossing)

| mode | axial field | eta (normalized div B) | emag retained | dE/E |
| --- | --- | --- | --- | --- |
| CONSTANT | **0.0** | 1.192112e-14 | 0.1461848 | 2.22e-16 |
| PLM | **0.0** | 5.267631e-15 | 0.8711372 | 2.22e-16 |
| PPM4 | **0.0** | 5.261217e-15 | 0.9102367 | 0.0 |
| WENO5 | **0.0** | 5.662492e-15 | 0.9261151 | 2.22e-16 |
| MP5 | **0.0** | 5.287496e-15 | 0.9337241 | 0.0 |

Divergence at roundoff and the axial field identically zero for every mode. Magnetic-energy
retention increases monotonically with reconstruction order, which is the independent evidence
that each mode is genuinely active and behaving as its order implies rather than silently
falling back to a lower one.

#### Order of accuracy, and a finding about the time integrator

Measured on the CPAW. At the default CFL 0.4 with `rk2`, WENO5 and MP5 do NOT show their
formal order — they plateau at apparent order ~1 and, tellingly, converge to the *same* L1
(2.049459e-04 at N=64, 9.302529e-05 at N=128, agreeing to five digits). A shared floor points
away from the reconstruction, and two experiments identify it:

- **Not the solver.** WENO5 with `mhd_hlld` instead of `mhd_hlle` reproduces those numbers in
  every digit.
- **It is `rk2`.** Cutting CFL from 0.4 to 0.05 at fixed N = 64 drops L1 from 2.049459e-04 to
  3.255717e-06 — a factor of **63**, against the 8^2 = 64 expected from a second-order-in-time
  term. So the total error is `A*dx^p + B*dt^2`, and with `dt ~ dx` at fixed CFL the temporal
  term dominates once the spatial term is small enough.

Re-running with `dt ~ dx^2.5` (`cfl = 0.4*(16/N)^1.5`), which pushes the temporal term to
O(dx^5), exposes the spatial order:

| mode | N=16 | N=32 | N=64 | N=128 | observed order |
| --- | --- | --- | --- | --- | --- |
| CONSTANT | 4.630736e-01 | 2.953412e-01 | 1.698002e-01 | 9.143864e-02 | 0.65 → 0.89 (first order) |
| PLM | 5.690184e-02 | 1.910746e-02 | 5.537642e-03 | — | 1.57 → 1.79 |
| PPM4 | 5.109414e-02 | 1.488293e-02 | 3.638068e-03 | 8.461690e-04 | 1.78 → 2.03 → 2.10 |
| WENO5 | 9.180752e-04 | 4.961728e-05 | 3.255717e-06 | 1.825677e-07 | 4.21 / 3.93 / 4.16 |
| MP5 | 7.221003e-04 | 4.729316e-05 | 3.237291e-06 | 1.822929e-07 | 3.93 / 3.87 / 4.15 |

**The PLM control run under the identical dt scaling is unchanged** (1.57 → 1.79, versus
1.55 → 1.74 at fixed CFL), which is what shows the scaling exposes order rather than
manufacturing it: PLM's spatial error still dominates, so suppressing the temporal term does
nothing for it.

WENO5 and MP5 reach ~4, not their formal 5. The cap is not the reconstruction — plausibly the
second-order EMF corner averaging in the CT update — and is recorded as measured rather than
explained away. MP5 additionally holds `|B|` uniform to **roundoff at every resolution**
(spread <= 2.2e-16), i.e. it preserves the circular polarization exactly, which WENO5 does not
(spread 1.2e-05 → 2.6e-10).

**Practical consequence, and the reason this belongs in the docs:** with `rk2` at CFL 0.4,
WENO5 and MP5 buy nothing over PPM4 asymptotically. They are still dramatically better at
coarse resolution (62-79x lower L1 at N = 16), which is where it usually matters, but a user
expecting fifth-order convergence will not get it without a higher-order time integrator or a
CFL that shrinks faster than dx.

#### Shock robustness, Brio & Wu, all five modes

All five: `max|Bx - 0.75|`, `max|Bz|`, `max|vz|`, `max|divB|` **all exactly 0.0**; `min rho`
and `min P` positive with margin (no floor activation); no non-finite values.

Normalized L1 vs the Athena++ reference:

| mode | rho | press | vx | vy | By |
| --- | --- | --- | --- | --- | --- |
| CONSTANT | 1.506631e-02 | 1.590445e-02 | 2.833858e-02 | 2.105584e-02 | 9.087799e-03 |
| PLM | 2.659044e-03 | 2.280626e-03 | 5.799217e-03 | 4.006543e-03 | 1.689883e-03 |
| PPM4 | 2.025147e-03 | 2.492375e-03 | 6.391433e-03 | 4.020663e-03 | 1.832337e-03 |
| WENO5 | 1.938905e-03 | 1.989748e-03 | 5.785285e-03 | 3.111283e-03 | 1.394793e-03 |
| MP5 | 1.882237e-03 | 1.862492e-03 | 5.192272e-03 | 2.688953e-03 | 1.287064e-03 |

CONSTANT is roughly 6x worse than the rest, as its first-order accuracy requires. Among the
others the ordering is not uniform — PPM4 is better than PLM on density but slightly worse on
pressure, vx and By — which is expected at a discontinuity, where limiter behaviour rather
than formal order decides the result. No mode is unstable or non-positive at a shock.

#### Verdict

**All five reconstruction modes are certified for use with MHD.** None needs to be marked
unsupported. Requirements: `nghost >= stencil_width + 1`, enforced already and verified to
abort with an actionable message below that. Recorded limitation: WENO5 and MP5 cannot reach
their formal order with `rk2` at a fixed CFL.

---

## N01 — startup rejection matrix, 2026-09-18

This closes concern C2, which was the highest-consequence gap in the port: every accuracy
result above is stated only for the certified matrix and is silent about everything outside
it, and that silence is only safe if the outside configurations cannot **run**. Until this
test ran, "rejected at startup" was an unverified claim about untested code.

| Field | Value |
| --- | --- |
| Harness | `claude_sessions/mhd_runs/n01_startup_rejections.py` |
| Command | `cd /tmp/n01 && python3 <repo>/claude_sessions/mhd_runs/n01_startup_rejections.py` |
| Base deck | `inputs/mhd/brio_wu.rin` (regenerated from `brio_wu.py`), with `parthenon/time/nlim=0` |
| Criterion per case | nonzero exit **and** an output message containing a fragment unique to the rejection being tested |
| Result | **PASS** — 23/23, plus a positive and a negative control |

### Why the criterion has two halves

A run that aborts for an unrelated reason — a mistyped override, an unreadable deck — also
exits nonzero, and a harness that checked only the exit status would record it as a
successful rejection. Each case therefore also requires a message fragment that appears in
exactly one `PARTHENON_REQUIRE` in `src/`. Generic fragments such as "not supported" were
deliberately not used: several messages share that phrase, so one rejection could stand in
for another.

The `base` case (unmodified deck, expect exit 0) is a positive control and is not optional.
Without it, a harness in which *every* invocation fails — wrong executable path, missing
deck — reports a clean sweep of passes.

### Cases

| Case | Override | Rejected by |
| --- | --- | --- |
| `hydro_solver_with_mhd` | `hydro/riemann=hllc` | `hydro.cpp:148` |
| `mhd_solver_without_mhd` | `physics/mhd=false` | `hydro.cpp:148` |
| `two_materials` | `material1/eos_type=IdealGas` | `riot.cpp` |
| `refinement_adaptive` | `parthenon/mesh/refinement=adaptive` | `riot.cpp` |
| `refinement_static` | `parthenon/mesh/refinement=static` | `riot.cpp` |
| `general_pte_flag` | `materials/use_general_pte=true` | `riot.cpp` (deferred check) |
| `non_ideal_eos` | `material0/eos_type=Gruneisen` + its 10 required parameters | `riot.cpp` (deferred check) |
| `fixed_fluid` | `physics/fixed_fluid=true` | `riot.cpp` |
| `strength`, `mix`, `tn`, `ionization`, `levelsets`, `multigroup_diffusion`, `radiation_transport`, `prescribed_sources`, `gravity`, `scalars`, `tracers` | `physics/<name>=true` | `riot.cpp` |
| `lasers_needs_ionization` | `physics/lasers=true` | `riot.cpp:125` (see note) |
| `mu0_zero`, `mu0_negative` | `mhd/mu0=0.0`, `mhd/mu0=-1.0` | `mhd.cpp:55` |

Two cases need their expected message explained rather than assumed:

- **`lasers_needs_ionization`** asserts `"Lasers requires ionization"`, not the MHD laser
  message. Lasers require ionization independently of MHD, and that check runs first, so
  asserting the MHD message here would have failed for a reason unrelated to MHD. The
  MHD-specific laser line is consequently unreachable from a valid input: reaching it needs
  ionization on, which is itself rejected earlier in the gate. It is covered by inspection,
  not by execution, and that is recorded rather than papered over.
- **`non_ideal_eos`** asserts the broad MHD general-PTE message, not HLLD's
  `"requires an ideal gas"`. The MHD gate runs before `Hydro::Initialize`, so it shadows the
  HLLD-specific message by design. The HLLD line is defense in depth for if the broader gate
  is ever relaxed.

### Two real defects found by running this

Both were in the general-PTE rejections, and both are the reason this test was worth running
rather than assuming.

1. **`hydro.cpp` read the wrong input block.** It used
   `pin->GetOrAddBoolean("multiphysics", "use_general_pte", false)`, but the option lives in
   `<materials>` (`materials.cpp:373`). The read therefore always returned the default
   `false` — and, being a `GetOrAdd`, silently created a spurious
   `multiphysics/use_general_pte` entry in the recorded input. **The HLLD ideal-gas guard
   was dead code and could never have fired.**

2. **Both checks read the input flag, which cannot see a non-ideal EOS.**
   `materials.cpp:450` sets `use_general_pte = true` whenever any material's `eos_type` is
   not ideal, *regardless of the input flag*. So the most likely way for a user to reach the
   general closure — selecting a real EOS, never touching the flag — was exactly the way
   neither check could detect. Verified directly: with `eos_type=Gruneisen` the original code
   ran past both guards.

Fixed by reading the **resolved** package parameter instead of the input file:
`hydro.cpp` now uses `mat_pkg->Param<bool>("use_general_pte")`, and the MHD gate's
general-PTE clause moved out of the block at `riot.cpp:140` to immediately after
`Materials::Initialize`, since the resolved value does not exist before that call. A comment
at the original site records why the check is not there.

Regression evidence for the fix: `ctest` **41/41**, and `inputs/mhd/brio_wu.rin` reproduces
its G5.5 PLM row exactly (`rho 2.659044e-03`, `press 2.280626e-03`, `vx 5.799217e-03`,
`vy 4.006543e-03`, `By 1.689883e-03`), with all four structural invariants exactly 0.0. Both
changes are startup-only and cannot affect a run that starts.

One cosmetic consequence, recorded so it is not mistaken for a regression later: HLLD runs
no longer carry the spurious `multiphysics/use_general_pte` line in the input deck embedded
in their `.phdf` attributes. Array data is unaffected.

### The non-Cartesian rejection needs a second binary

`COORDINATE_TYPE` is a **compile-time** setting (`external/parthenon/src/config.hpp.in:25`,
driven by `-DPARTHENON_COORDINATES`, default `UniformCartesian`). In a Cartesian binary
`IsCoord<UniformCartesian>()` is constexpr-true, so no input deck can make that rejection
fire. It was therefore tested against a separate build:

```
cmake -S . -B /tmp/riot_cyl_build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-16 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DRIOT_ENABLE_MPI=ON -DRIOT_ENABLE_HDF5=ON \
  -DRIOT_BUILD_PYTHON=ON -DRIOT_BUILD_CATCH2=ON -DRIOT_ENABLE_UNIT_TESTS=OFF \
  -DPARTHENON_COORDINATES=UniformCylindrical
cmake --build /tmp/riot_cyl_build --parallel 8
python3 claude_sessions/mhd_runs/n01_startup_rejections.py --cyl-exe /tmp/riot_cyl_build/src/riot
```

| Case | Command | Result |
| --- | --- | --- |
| `cyl_control` | cylindrical binary on `inputs/sedov_rz.rin`, `nlim=2` | **exit 0**, 38 mesh blocks, ran to completion |
| `non_cartesian` | cylindrical binary on `inputs/mhd/brio_wu.rin` | **exit 134**, `"MHD is only supported in Cartesian coordinates."` |

The control is the point of the pair. "A cylindrical build aborts on an MHD deck" is not by
itself evidence about the MHD gate — a cylindrical build that could not run anything would
give the same result. Running hydro RZ successfully in the *same* binary makes `physics/mhd`
the only difference between completing and aborting.

Two incidental facts worth recording, since both were open questions before the build:
RIOT **does** compile against a curvilinear Parthenon (the build succeeded with no source
changes), and the MHD coordinate rejection fires before Parthenon objects to the Brio–Wu
deck's negative radial range — so it is genuinely the first thing to stop the run, not a
lucky ordering.

---

## G5.6 — Restart equivalence for face-centered state, 2026-09-18

Closes concern C5. Harness: `claude_sessions/mhd_runs/analyze_restart.py`, which compares two
snapshots by explicit PATH and **aborts if their `Time`/`NCycle` differ** — a restart
comparison whose two sides are at different cycles measures the solution advancing, not
restart fidelity, and it fails in the direction of looking like a solver bug.

### Result summary

| # | Test | Criterion | Result |
| --- | --- | --- | --- |
| 1 | Zero-step reload, MHD: restart at cycle 20, integrate nothing, compare to the uninterrupted run's cycle-20 snapshot | bitwise | **PASS for face B, conserved and all magnetic derived fields.** `velocity`/`pressure` differ at 1–2 ulp — see below |
| 2 | Same, hydro-only control (`inputs/noh.rin`, MHD never enabled) | — | Identical signature: conserved bitwise, `velocity` 2.2e-16, `pressure` 3.6e-17 |
| 3 | Reproducibility control: the same run twice, no restart | bitwise | **PASS — bitwise identical** |
| 4 | Evolved, same rank count: restart at cycle 20, integrate to cycle 40 | rel ≤ 1e-14 | **PASS** — worst 5.55e-15 |
| 5 | Evolved, changed rank count: 1-rank checkpoint restarted on **4 ranks** to cycle 40 | rel ≤ 1e-14 | **PASS** — worst 5.55e-15, *bit-for-bit the same differences as #4* |
| 6 | Hydro checkpoint loaded into an MHD run | must be refused | **PASS after fix** (was: ran silently) |
| 7 | MHD checkpoint loaded into a hydro run | must be refused | **PASS after fix** (was: ran silently, 80% pressure error) |

Test #3 is what makes the rest interpretable. Without it, the roundoff-level differences in
#4/#5 could be the code being non-deterministic rather than the restart introducing anything,
and the two have completely different implications.

Test #5 is stronger than it looks: the differences are *identical* to the same-rank case, so
changing the rank count across a restart contributes nothing at all. That is consistent with
P01's bitwise MPI invariance.

### Defect 1 — derived magnetic state was never rebuilt on restart

`MHD::PostInitialization` is installed as `PostInitializationMesh`, and Parthenon calls
`Mesh::Initialize(!is_restart, ...)`. **Everything** inside that function's `if (init_problem)`
block — the problem generator, every PostInitialization hook, and the
PreCommFillDerived/communicate/FillDerived cycle that follows them (`mesh.cpp:872-953`) — is
skipped on a restart. So the three derived magnetic fields stayed at zero.

This was not merely a cosmetic first-output problem. `ccbulk::magnetic_field` is reconstructed
to supply the **transverse** field in the Riemann solve, and the per-stage `MHD::SetDerived`
task runs *after* the update, not before the fluxes. The first post-restart stage therefore
computed its fluxes from a zero transverse field. The run continued, stayed divergence-free,
and was permanently wrong: **3.7e-06** deviation from the uninterrupted trajectory at cycle 40
— roughly nine orders of magnitude above roundoff, and invisible to every existing test.

Fixed with `MHD::RestoreDerivedOnRestart`, installed as `UserWorkBeforeLoopMesh` (one of the
few hooks that runs regardless of restart, and before both the first output and the first
step). Restart-only, so the verified fresh-start path is untouched.

**`entire` vs `interior` was measured, not assumed.** Nothing communicates the cell-centered
ghosts between that hook and the first stage's reconstruction, so they must be derived
locally. With `interior` the restarted trajectory diverges from the uninterrupted one by
1.4e-10 *in time* by cycle 40; with `entire` the time stays bitwise. One argument, four orders
of magnitude.

### Defect 2 — restarting across an MHD/hydro change ran silently, in both directions

Neither direction was refused. The second is the dangerous one:

- **hydro checkpoint → MHD run.** No face field in the file, so it is zero-filled and the run
  proceeds as MHD with B = 0. Self-consistent, but the user asked to continue one calculation
  and silently got a different one.
- **MHD checkpoint → hydro run.** `ccbulk::total_material_energy` includes B²/(2μ₀) by the
  ADR-002 contract, and with MHD off nothing subtracts it, so the entire magnetic energy is
  reinterpreted as heat. Measured on Brio & Wu at cycle 20: max pressure **1.781574** instead
  of **1.000185**. The error is **7.991440e-01**, which equals `max(B²/2μ₀)` to every printed
  digit — i.e. exactly (γ−1)·E_mag at γ = 2. An **80% pressure error, silent, exit code 0.**

Two complementary checks now reject both, and the pair is deliberate because either alone has
a hole:

1. **Deck-based** (`riot.cpp`): `mhd/mu0` is present in the input deck Parthenon embeds in
   every checkpoint an MHD run writes, and absent otherwise. On restart that deck is reloaded
   into `pin` before command-line overrides, so it reports the *checkpoint's* answer. Must be
   tested before `MHD::Initialize`, which would otherwise create the parameter and make the
   test vacuous. **Hole:** a user who passes `mhd/mu0=...` on the command line creates the
   parameter themselves and masks it — and passing mu0 is the natural thing to do when
   enabling MHD. Verified: the check fires without the override and is defeated by it.
2. **Data-based** (`MHD::RestoreDerivedOnRestart`): a global max-reduction over
   `ccbulk::magnetic_energy`. Zero everywhere is the signature of a zero-filled face field.
   Cannot be masked by any input override. The reduction **must** be global — a localized
   field such as the field loop leaves whole ranks with zero magnetic energy, so a rank-local
   test would abort a good restart. Verified against a 4-rank restart of the field loop: no
   false positive.

`mhd/allow_zero_field_restart` is the opt-in for a genuinely unmagnetized MHD restart, which
is how the MHD-reduces-to-hydro check (H02) is posed. Registered in `MHD::Initialize` rather
than on the restart path so it appears in the parameter table of every MHD run — otherwise a
user hitting the rejection could not discover the escape hatch its message names.

### The residual: restarts are not bitwise for derived primitives

Even the zero-step reload leaves `velocity` at 2.2e-16 and `pressure` at 1.1e-15. Those are
derived from the checkpointed conserved state by a different arithmetic path than the one that
produced them before the dump. The hydro-only control (test #2) shows the identical signature
with MHD never enabled.

**That control alone was not enough, and the gap was closed with a pre-port A/B — see G5.7.**
Run with the current binary it excludes MHD-specific code but not the port's Stage 1 edits to
shared `sparse_update` machinery, which also run in hydro-only mode. G5.7 builds `193b3fa` and
settles it: the pre-port binary produces the *same numbers to every digit*, so the port did not
introduce the residual. Tracked as concern **C13**.

This is why tests #4/#5 are stated at rel ≤ 1e-14 instead of bitwise: the threshold comes from
the measured 1.1e-15 seed plus modest amplification over 20 steps, and it still sits eleven
orders of magnitude below the 3.7e-06 defect that motivated the work.

### Reproducing

```
mkdir -p /tmp/mhdrst && cd /tmp/mhdrst
R=/Users/taitano/Documents/git/riot
V=c.c.bulk.rho,c.c.bulk.pressure,c.c.bulk.velocity,c.c.bulk.magnetic_field,c.c.bulk.div_magnetic_field,c.c.bulk.magnetic_energy,c.c.bulk.total_material_energy,f.bulk.magnetic_field
# uninterrupted, with a checkpoint at cycle 20 and per-cycle snapshots
$R/build/src/riot -i $R/inputs/mhd/field_loop_3d.rin parthenon/job/problem_id=a \
  parthenon/time/nlim=40 parthenon/output1/dn=1 parthenon/output1/dt=-1 \
  parthenon/output1/variables=$V parthenon/output2/file_type=rst parthenon/output2/dn=20
# zero-step reload, and the evolved restart
$R/build/src/riot -r a.out2.00001.rhdf parthenon/job/problem_id=b0 parthenon/time/nlim=20 \
  parthenon/output1/variables=$V
$R/build/src/riot -r a.out2.00001.rhdf parthenon/job/problem_id=b1 parthenon/time/nlim=40 \
  parthenon/output1/variables=$V
. $R/riot_venv/bin/activate
python3 $R/claude_sessions/mhd_runs/analyze_restart.py a.out1.00020.phdf b0.out1.final.phdf
python3 $R/claude_sessions/mhd_runs/analyze_restart.py --rtol 1e-14 a.out1.final.phdf b1.out1.final.phdf
```

`f.bulk.magnetic_field` must be added to `parthenon/output1/variables` explicitly — the
tracked decks do not output it, and without it the one field this test exists for is silently
skipped (the harness says `ABSENT` rather than passing quietly). The restart cadence must be
`parthenon/output2/dn`, not `dt`: with `dt` the checkpoint lands at a wall-clock-independent
but time-based cadence that a short `nlim` run may never reach, which is how an earlier
attempt ended up with no cycle-20 checkpoint at all.

Regression evidence for both fixes: `ctest` **41/41**; N01 **23/23**; a fresh
`field_loop_3d` run is **bitwise identical** to one from before the change; Brio & Wu
reproduces its G5.5 PLM row to every printed digit. Both fixes are inert on a fresh start.


---

## G5.7 — Is the non-bitwise restart the port's fault? Pre-port A/B, 2026-09-18

Settles the attribution half of concern C13. G5.6 recorded that `velocity` and `pressure` are
not bitwise across a restart and attributed it to pre-existing RIOT behaviour on the strength of
a hydro-only control. **That attribution was not supported by the evidence given**, and this
entry exists because the claim was challenged rather than because the test was planned.

The hydro-only control (`inputs/noh.rin`, `physics/mhd` never enabled) was run with the
**current, MHD-modified binary**. That excludes:

- everything in `src/mhd/`,
- the restart of face state itself,
- the magnetic-energy add/subtract in `fill_shared_derived.cpp`,

because all of those are gated on `do_mhd` and none executes with MHD off. It does **not**
exclude this port's Stage 1 edits to *shared* machinery, which do run in hydro-only mode:
`sparse_update::UpdateToNextStage` gained required flags `{Cell, Independent}` and
`DeepCopyIndependentData` gained `Cell`. A change to which fields land in a stage register is a
plausible mechanism for a derived primitive being recomputed by a different path after a
restart, which is precisely the observed symptom. So the supported statement was "not caused by
MHD-specific code", not "pre-existing".

### Procedure

The H01 method, with the build configuration held fixed so a build difference cannot be
mistaken for a code difference:

```
git worktree add /tmp/riot_base 193b3fa
cd /tmp/riot_base
# submodule SHAs provably identical rather than merely equal
for n in Catch2 kokkos-kernels parthenon singularity-eos singularity-opac; do
  rm -rf external/$n && ln -s /Users/taitano/Documents/git/riot/external/$n external/$n
done
cmake -S /tmp/riot_base -B /tmp/riot_base_build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-16 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DRIOT_ENABLE_MPI=ON -DRIOT_ENABLE_HDF5=ON \
  -DRIOT_BUILD_PYTHON=ON -DRIOT_BUILD_CATCH2=ON -DRIOT_ENABLE_UNIT_TESTS=OFF
cmake --build /tmp/riot_base_build --parallel 8
```

Then the G5.6 zero-step reload on `inputs/noh.rin` with each binary in turn. `noh` needs
`PYTHONPATH` to cover site-packages *and* `script/inputs` *and* the built `singularity_eos`,
because its pgen is a Python script that imports `riot`; without all three it aborts at 134 in a
way that looks like a physics crash.

### Result

| Binary | `velocity` max\|diff\| | `pressure` max\|diff\| | `rho`, `total_material_energy` |
| --- | --- | --- | --- |
| pre-port `193b3fa` | 2.220446e-16 | 2.086005e-16 | bitwise |
| current port | 2.220446e-16 | 2.086005e-16 | bitwise |

**Identical to every printed digit.** Not merely the same order of magnitude — the same doubles.
The port did not introduce the non-bitwise restart of derived primitives. C13's attribution is
closed; the behaviour remains open as a RIOT-wide property worth raising upstream.

### Two free side results

| Comparison | Result |
| --- | --- |
| Uninterrupted `noh` at cycle 20, pre-port vs port | **bitwise identical** |
| The two zero-step reloads, pre-port vs port | **bitwise identical** |

The first is worth more than it looks: **it extends H01's "hydro unchanged" evidence to a second
problem and a different pgen path.** H01 used `linear_modes`; `noh` is a different problem
generator, goes through the embedded-Python pgen route, and is multi-material-capable. Getting
bitwise agreement there is independent corroboration that the Stage 1 `sparse_update` filter
change did not drop a field — which is exactly the risk concern C1 exists to test. It does not
replace C1's eight suites, but it is a second data point rather than a restatement of the first.

---

## G5.8 — LLF (Rusanov) MHD solver, 2026-09-18

Third and last MHD solver. Ported from `artemis/src/utils/fluxes/riemann/llf.hpp`,
transcribed in DONOR_KERNELS.md section 14. Selected with `hydro/riemann = mhd_llf`.

`F = ½(F_L + F_R) − ½a(U_R − U_L)`, `a = max(|v_n| + c_f)` (Toro Eq. 10.43). **No degeneracy
guards, because there are no intermediate states to degenerate**, and **no ideal-gas
restriction**, because like HLLE it consumes only the bulk modulus. HLLD's
`use_general_pte` requirement is deliberately *not* copied.

### Why keep the least accurate solver

It is the fallback with no failure mode. When a problem misbehaves under HLLD or HLLE,
re-running it under LLF separates "the solver's intermediate states are breaking down" from
"the reconstruction, CT, or initial condition is wrong" — because LLF has no intermediate
states to break down. That diagnostic is why the donor ships all three, and it is the reason
this is worth the code rather than being dropped as redundant.

### Unit tests (6 new, `ctest` 41/41 → **47/47**)

| Test | Criterion | Result |
| --- | --- | --- |
| Consistency: equal states give the exact MHD flux, 6 states × 3 directions | rel ≤ 1e-12 | **PASS** |
| Normal induction flux | `== 0.0` exactly | **PASS** |
| Equal-state signal speed equals \|v_n\| + c_f, and equals HLLE's | rel ≤ 1e-12 | **PASS** |
| Finite on all 100 degenerate-state pairs × 3 directions | all finite, `pairs == 100` | **PASS** |
| Genuinely different flux from both HLLE and HLLD | \|Δf_eng\| > 1e-6 | **PASS** |
| Smears a stationary contact by exactly −½a·Δρ | rel ≤ 1e-12 | **PASS** |

Two of these are worth their space:

- **The signal-speed identity.** For equal states, Toro's `|v_n| + c_f` and HLLE's
  `max(|s_l|, |s_r|)` are provably the same number. Pinning it catches the classic porting
  error for this solver — using the acoustic speed where the fast speed belongs — which
  would leave every consistency test passing while making the scheme unstable in any
  field-dominated region.
- **The contact-smearing test pins the dissipation COEFFICIENT, not just its sign.** Across
  a stationary contact the exact mass flux is zero; LLF must produce exactly `−½a(ρ_R − ρ_L)`.
  Asserting that against an independently computed `a` is much stronger than observing that
  the answer is nonzero, and it is checked against HLLD being exactly zero on the same state.

### Run-level verification: the diffusivity ordering

Brio & Wu, 512 zones, normalized L1 against the Athena++ reference. **This is an
oracle-independent check**: the ordering HLLD < HLLE < LLF is required by the wave structure
(five waves, two waves, none), so it holds regardless of how good the reference is.

| field | HLLD | HLLE | LLF | monotone? |
| --- | --- | --- | --- | --- |
| rho | 1.723601e-03 | 2.659044e-03 | 3.183125e-03 | yes |
| press | 1.523292e-03 | 2.280626e-03 | 2.853562e-03 | yes |
| vx | 4.633461e-03 | 5.799217e-03 | 6.636279e-03 | yes |
| vy | 2.749096e-03 | 4.006543e-03 | 4.774920e-03 | yes |
| By | 1.186258e-03 | 1.689883e-03 | 2.034144e-03 | yes |

**Strictly monotone on all five fields with no exceptions** — ten independent inequalities,
all in the required direction. HLLE/LLF ratios are 0.80–0.87 (cf. HLLD/HLLE 0.65–0.80).

Structural invariants under LLF are all **exactly 0.0** (`max|Bx − 0.75|`, `max|Bz|`,
`max|vz|`, `max|divB|`), with `min rho` 1.161995e-01 and `min P` 8.642078e-02 — positive with
margin, so no floor fired.

### Independent corroboration, and CT under LLF

3D field loop (`inputs/mhd/field_loop_3d.rin`), which is a different problem and a different
diagnostic:

| solver | axial field | eta | E_mag retained | dE/E | dM/M |
| --- | --- | --- | --- | --- | --- |
| LLF | 1.470461e-15 | 4.070497e-15 | **0.6553** | 1.97e-16 | 0.0 |
| HLLE | 1.441305e-15 | 4.055658e-15 | **0.7261** | 5.92e-16 | 3.33e-16 |
| HLLD | 1.401965e-15 | 3.521546e-15 | **0.7381** | 7.89e-16 | 4.44e-16 |

Retention reproduces the same ordering from an unrelated measurement. Constrained transport is
unaffected by the solver swap: the axial field stays at roundoff and mass and energy are
conserved to roundoff.

> **AMENDED after G5.9 — the HLLD row of this table is weak corroboration, not independent
> evidence.** Solver-health counters added later show that on this problem HLLD falls back to
> HLLE on ~7 faces per cell per step, i.e. on the large majority of faces, because the field
> lies in the (x1,x2) plane with `b3 = 0` and vanishes outside the loop. So the HLLD and HLLE
> runs here mostly execute the *same* solver, and their 1.6% retention difference comes from
> the minority of faces with a non-negligible normal field. The **Brio & Wu** table above is
> the real evidence for HLLD < HLLE: it has **zero** fallbacks. The LLF vs HLLE comparison is
> unaffected, since neither solver has a fallback path. See G5.9.

**MPI**: 1 vs 4 ranks on the 3D field loop is **bitwise identical** in every field, so the new
flux path introduces no rank dependence.

### Not covered

No absolute frozen threshold for LLF, for the same reason as concern C3: the donor's tolerance
is calibrated against its own gold file. The ordering claim above is relative and therefore
immune to that, but LLF's absolute accuracy is monitored rather than gated.

---

## G5.9 — Solver-health counters and the div B monitor, 2026-09-19

Closes concerns C6 (no solver-failure/floor counters) and C7 (`mhd/monitor_divb` was a dead
parameter). Both are instrumentation, and both immediately reported something that was
previously invisible — including a correction to this ledger.

### C7 — `mhd/monitor_divb` now does something

Was registered at `mhd.cpp` and read nowhere, so a user who set it (as
`inputs/mhd/brio_wu.py` and `field_loop.py` both do) got silence. Now installed via
`PostStepDiagnosticsMesh` and reports once per step, after the update:

```
divB: cycle=1 t=1.587708e-03  max=3.122502e-17  mean=8.092140e-19  eta=9.788523e-16  |B|max=1.009930e-03
```

Three numbers rather than one, because `max|divB|` alone is **not interpretable**: it carries
units and scales like 1/dx, so the same quality of solution reports a larger number on a
finer mesh. `eta = |divB| * l_cell / max(|B|_cell, b_ref)` is the dimensionless one and is
the number to judge; "at roundoff" means ~1e-16 regardless of mesh, units or field strength.
The volume-weighted mean is there because the max-to-mean ratio distinguishes a few bad cells
(indexing bug) from a uniformly poor solution (wrong stencil).

It reuses the already-computed `ccbulk::div_magnetic_field` rather than re-deriving the face
stencil, so the runtime monitor and the analysis scripts cannot disagree about what
"divergence free" means.

Verified: 1 vs 4 ranks reports **identical values to every printed digit**, including the
volume-weighted sums. Off by default and silent when off (0 report lines).

**A bug found in the first version of this code, worth recording because it is a pattern.**
I first set `b_ref` to the global max |B| and wrote the denominator as
`max(|B|_cell, b_ref)`. Since the global max is ≥ every local value, that expression is
*always* `b_ref` — the per-cell term was **dead code** and eta had silently collapsed to a
global normalization, which is the weaker measure. Fixed by making `b_ref` a weak-field floor
(1e-6 of peak |B|) so the local term is live. Confirmed by the numbers changing: cycle-2 eta
went from 9.58e-16 to 4.88e-15 once the local normalization took effect, i.e. a weaker-field
cell now sets the value, which is exactly the sensitivity the local form is for.

Also corrected: the parameter's registered description said "before and after each step",
but only a post-step hook is installed. The description now matches the code.

### C6 — solver-health counters

Two cumulative counters, incremented from inside the flux kernels via
`Kokkos::atomic_add` on a raw device pointer (a pointer, not a `View`, because the three
solvers are handed to `MHDFluxes` as plain function pointers and a View in their signature
would force them to become templates):

| Counter | Meaning |
| --- | --- |
| `hlld_fallback` | HLLD hit one of its six degeneracy guards and used its HLLE fallback for that face |
| `density_floor` | a solver clamped a non-positive reconstructed density (an *invalid state repaired*, which the verification plan classes as a failed test, not a success — so it also prints an explicit WARNING) |

Reported whenever a counter **increases**, not every step (which would bury the signal) and
not only at the end of the run (which would lose which step it happened on). Counts are
MPI-summed, because a fallback confined to one rank's blocks would otherwise be invisible
depending on which rank printed.

Counts are also reported **per cell per step**. A raw count is the same uninterpretable-number
mistake as raw `max|divB|`; per cell per step has a known ceiling (≈ ndim × nstages face
solves, so ~6 for rk2 in 3D), which turns the number into an answer to "is the solver I
selected the solver I am getting?".

Why this matters: `hlld_fallback` is the one instrument that can detect HLLD being silently
degraded to HLLE. The existing unit test "HLLD and HLLE are genuinely different fluxes"
catches the *always*-falling-back case; nothing caught the *usually*-falling-back case.

### The counters immediately corrected a claim in G5.8

| run | solver | fallbacks per cell per step |
| --- | --- | --- |
| Brio & Wu, 512 zones | HLLD | **0** over 20 cycles |
| 3D field loop | HLLD | **7.17, 6.74, 6.46** (cycles 1–3) |

Brio & Wu has a uniform `bn = 0.75`, so no guard ever fires and HLLD is genuinely HLLD
throughout — the G5.8 Brio-Wu ordering stands unchanged.

The 3D field loop is the opposite. Its field lies in the (x1,x2) plane with `b3 = 0`, and it
is zero outside the loop, so the normal field vanishes on a large majority of faces. At
6.5–7.2 fallbacks per cell per step against a ceiling of roughly that same magnitude, **most
faces are being solved by HLLE even though the user asked for HLLD.**

**Consequence for G5.8, stated as a correction rather than left standing.** G5.8 cited
field-loop magnetic-energy retention (HLLD 0.7381 vs HLLE 0.7261) as *independent
corroboration* of the HLLD < HLLE ordering. That was overstated: on this problem the two
runs mostly execute the same solver, so the 1.6% difference comes from the minority of faces
with a non-negligible normal field rather than from a genuine HLLD-vs-HLLE comparison. The
Brio-Wu table in G5.8, with zero fallbacks, is the real evidence for that ordering; the
field-loop number is weak corroboration and G5.8 has been amended to say so. The LLF vs HLLE
half of the ordering is unaffected — neither solver has a fallback path.

This is not a defect in HLLD: falling back where the normal field vanishes is exactly what
the guards are for, and the donor does the same. What was wrong was my confidence in a
corroborating measurement I had not instrumented.

### Verification

- `ctest` 47/47 → **49/49**. The two new tests check the counters in **both** directions: a
  well-conditioned oblique state must report zero fallbacks (a counter that always fires is
  as useless as one that never does), a zero-`bn` state must report exactly one, HLLE and LLF
  must never touch the HLLD counter, and the density floor must not fire on healthy states.
- **MHD output is bitwise identical** to before this change, on the 3D field loop at 40
  cycles. This was the risk worth checking: threading an extra argument through `MHDFluxes`
  could have changed the scratch footprint or register pressure, and the file's own comment
  records that such a change once perturbed linear-wave L1 errors in the 7th digit.
- Silent when nothing is wrong: Brio & Wu with HLLD emits no diagnostic lines at all.

---

## G5.10 — Documentation and published limitations, 2026-09-19

Gate G5 requires the certified matrix to be **published in the docs and printed at startup**,
not merely enforced. Both halves are now done. This entry exists because a documentation
deliverable can be "finished" without being correct, and the checks below are what make it
verifiable rather than a claim.

| Field | Value |
| --- | --- |
| Revision | `62903d9` + this change |
| Artifacts | `doc/sphinx/src/packages/mhd.rst` (new), `doc/sphinx/index.rst`, `doc/sphinx/src/introduction.rst`, `doc/sphinx/src/packages/hydro.rst`, `src/mhd/mhd.cpp` (startup banner) |
| Result | **PASS** — docs build clean under CI's own strict flags; banner verified in a live run |

### What is documented

- Governing equations, in RIOT's own notation rather than the donor's.
- **The energy convention, as an admonition rather than a sentence in a paragraph.**
  `ccbulk::total_material_energy` includes `B^2/(2 mu0)` when and only when MHD is on. This is
  the single most likely way for a user to get a wrong answer from a correct code, so it is
  stated in the MHD chapter *and* cross-referenced from the hydro chapter's field table, which
  is where someone looking up `total_material_energy` will actually land. The docs say
  explicitly that `magnetic_energy` exists to DECOMPOSE the total, not to be added to it.
- The `mu0` normalization table, both conventions, and why `mu0 = 1` is what the test problems
  use.
- Shared-normal-B substitution in reconstruction, and the CT/EMF construction including why
  the divergence is preserved algebraically rather than approximately.
- The fast magnetosonic speed and the unchanged directional-sum CFL form.
- **The two measured reconstruction caveats from G5.5**, stated as measured limitations: the
  ~4th-order cap on `weno5`/`mp5` attributed to the second-order EMF corner averaging, and the
  fact that `rk2` at fixed CFL floors them so they buy nothing over `ppm4` asymptotically.
  Written to be useful rather than exculpatory: it also says they are 60-80x better at coarse
  resolution, which is where a user actually operates.
- **The HLLD fallback caveat**, with the instruction to read the counter before interpreting
  any HLLD-vs-HLLE comparison. This is the G5.9 lesson made permanent in user-facing form.
- The `density_floor` counter described as a FAILED result rather than a rescued one.
- Restart behaviour, honestly: bitwise for face B, the conserved variables and magnetic derived
  state; 1-2 ulp on `velocity`/`pressure` with the reason (recomputed, not read); 5.6e-15 on an
  evolved restart, rank-count-independent. The two rejected restart configurations.
- The full rejection table with reasons, and the "not available at all" list (resistive, Hall,
  Biermann, ambipolar, Nernst, Braginskii) so the capability cannot be mistaken for extended MHD.
- That the package has **not been run on a GPU**.
- How to write an MHD pgen: face B from a discrete curl of a vector potential (and why assigning
  an analytic B to faces is not equivalent -- CT preserves the initial error forever), and the
  mandatory `MHD::AddMagneticEnergyToTotal` call last.

### Startup banner

`MHD::Initialize` prints five lines on rank 0 when MHD is enabled. The rejections in
`riot.cpp` only fire for a configuration the user *asked for*; they say nothing to a user who
is inside the supported matrix but does not know where its edge is. The banner states the
validated boundary unconditionally.

Verified live:

```
$ build/src/riot -i inputs/mhd/brio_wu.rin parthenon/time/nlim=2
MHD: ideal MHD with face-centered constrained transport is enabled.
MHD: validated for Cartesian, uniform-grid, single-material, ideal-gas,
MHD:   single-temperature problems only; see the MHD chapter of the docs.
MHD: no resistivity, Hall term, Biermann battery, or anisotropic transport.
MHD: not run on GPU. total_material_energy INCLUDES B^2/(2*mu0).
```

### Checks that make this more than a claim

| Check | Why it matters | Result |
| --- | --- | --- |
| `sphinx-build -b html -W --keep-going -n` | `.github/workflows/docs.yml` builds with exactly these flags, so any warning is a CI failure. `-n` (nitpicky) catches a broken `:ref:` that would otherwise render as literal text | **PASS**, `build succeeded`, zero warnings |
| Output variable names taken from a real deck | The docs first used `ccbulk::magnetic_field`, which is the C++ variable name and NOT what an output block accepts. Corrected to `c.c.bulk.magnetic_field` against `inputs/mhd/orszag_tang.py` | fixed before commit |
| Restart claim checked against G5.6 rather than paraphrased | The first draft said restart is "verified bitwise ... including on a different rank count". G5.6 says bitwise for the zero-step reload and 5.6e-15 for the evolved case. Overclaiming in the docs is worse than in a ledger, because a user cannot check it | corrected |
| Parameter table checked against `MHD::Initialize` | `mu0`, `monitor_divb`, `allow_zero_field_restart` -- names, types and defaults read from the registration calls, not from the plan | matches |
| `ctest` and a live MHD run after the banner | The banner is host-side and cannot change numerics, but "cannot" is not "did not" | `ctest` **49/49**; Brio-Wu runs unchanged |
| `CAPABILITY_MATRIX.md` refreshed | It still described the port as "mid-implementation, no end-to-end MHD simulation run", which was true at Stage 2 and badly wrong at Stage 5. A stale matrix is worse than none: it is the file everything else points at for what may be claimed | rewritten against the ledger |

### Deliberately not claimed in the docs

No performance numbers, no statement about GPU behaviour beyond "not run", no absolute
accuracy claim for Brio & Wu (there is no absolute threshold -- concern C3), and no mention of
AMR working. The reconstruction order table gives *measured* orders, not formal ones.

---

## D01 — DEFECT: constrained transport loses div B with four or more blocks per periodic axis

**Found 2026-09-19 by the new `tst/scripts/mhd/field_loop` regression test, on its first run.**
This is the defect the regression harness was built to catch, and it invalidates part of what
was recorded above; read this section before quoting any earlier CT or MPI result.

| Field | Value |
| --- | --- |
| Revision | `3eaae22` (present since the CT implementation, `dced139`) |
| Reproducer | `claude_sessions/mhd_runs/repro_divb_blocks.py --exe tst/build/src/riot` |
| Status | **OPEN** — characterized and attributed, root cause not yet found. Concern C14. |

### The measurement

Mesh resolution held FIXED at 48 x 48 on a square periodic box; only the meshblock size
varies, so the block count along x1 is the only thing that moves:

| blocks along x1 | `max abs(divB)` | verdict |
| --- | --- | --- |
| 2 | 6.796e-17 | ok |
| 3 | 6.510e-17 | ok |
| **4** | **6.160e-08** | **BROKEN** |
| **6** | **6.386e-11** | **BROKEN** |

The control matters and was nearly botched. A first pass swept `nx1` and the block size
together and concluded "4 blocks in x1 breaks, 4 blocks in x2 does not" -- both halves were
partly resolution artifacts, and the x1/x2 asymmetry disappeared entirely once the box and
resolution were made square. Nothing below rests on a column where more than one thing moved.

### Scope, established by controlled variation

| Axis varied | Result |
| --- | --- |
| Direction | Fails in x1 AND x2 (square box, 64 x 64: 1.412e-09 and 1.228e-10) |
| Dimensionality | Fails in 2D and in 3D (3D field loop, 4 blocks along x1: `eta` 1.8e-06 vs 1.7e-15 at 2 blocks) |
| Reconstruction | Fails with CONSTANT, PLM and PPM4 -- so it is NOT a reconstruction stencil or ghost-width problem |
| `nghost` | 2, 3, 4 all fail, slightly WORSE with more ghosts -- rules out the ghost budget |
| Riemann solver | Fails with `mhd_hlle`, `mhd_hlld`, `mhd_llf` |
| Rank count | Serial and 4 ranks give the SAME value (1.412e-09) -- not a communication-order or MPI bug |
| Boundary condition | 4 blocks with `outflow` on that axis is CLEAN (9.267e-17). **Periodicity is required.** |

### Mechanism, as far as it is pinned down

Dumping `f.bulk.magnetic_field` and comparing a 4-block run against a 1-block run of the same
problem:

- The shared faces **are** single-valued: block b's face at its high edge equals block b+1's
  face at its low edge, `max abs(diff) = 0.0` exactly, at every block boundary. So the ghost
  exchange is not leaving two different values behind.
- But the shared face between blocks 0 and 1 differs from the single-block solution by
  **4.123e-11**, while other shared faces differ by only ~5e-14.
- `4.123e-11 / dx = 1.32e-09`, which is the observed `max abs(divB)` to three digits.

So the two adjacent blocks compute DIFFERENT constrained-transport updates for the same
shared face; single-valuing then keeps one of the two, and the block whose value was discarded
no longer has a balanced divergence budget. That is exactly the failure mode CT cannot
recover from: the constraint is preserved algebraically thereafter, so the error is frozen in
for the rest of the run rather than decaying.

Confirmed frozen: with the `mhd/monitor_divb` output, `max abs(divB)` is at roundoff through
cycle 66, jumps to 7.043e-10 at cycle 67 and 1.411662e-09 at cycle 68, and then holds
1.411662e-09 unchanged to the end of the run. A single injection event, faithfully preserved.

What is NOT yet explained: why four blocks along an axis differs from three. All the inputs to
the boundary edge EMF (the extended transverse flux layer, the ghost cell-centered field and
velocity, the mass flux used by the upwind selector) are, by inspection, available and are
bitwise copies of the neighbour's interior data in every layout. The mass flux is ~2.0 and
uniform, so the `UpwindEMFGradient` branch cannot be flipping. Three blocks and four blocks
differ in that with three every block is a neighbour of every other, which points at
neighbour-set or block-ownership handling rather than at the EMF arithmetic -- but that is a
hypothesis, not a finding.

### Attribution: this is the port's defect, NOT inherited from the donor

Required control, run rather than assumed (see the "do not repeat" entry on attribution):

```
mkdir -p /tmp/art_fl && cd /tmp/art_fl
/Users/taitano/Documents/git/artemis/build/src/artemis \
  -i /Users/taitano/Documents/git/artemis/inputs/field_loop/field_loop.in \
  parthenon/job/problem_id=art_4blk parthenon/time/nlim=200
```

The donor's own DEFAULT field-loop deck is `nx1 = 64, nx2 = 32` on `16 x 16` meshblocks --
that is **four blocks along x1**, precisely the failing layout. It gives relative div B of
**3.244e-15** and its regression test passes at a 1e-10 threshold. RIOT in the same layout
gives 4.4e-08. Same Parthenon pin, same `nghost = 2`.

#### The single-point control was not enough; here is the matched sweep

A single donor data point leaves two holes: the donor's `divB` might be a different diagnostic,
and one layout is not a trend. Both are now closed.

**The diagnostic is the same quantity.** `artemis/src/derived/fill_derived.cpp:266` computes
`divB` as the signed sum of face area times face B over the cell volume, with the same
`multid`/`threed` collapse as RIOT's `SetDerivedMagneticFields`. Formula for formula identical,
so the two numbers are comparable. (Had the donor differenced the CELL-centered field instead,
its "clean" value would have been meaningless -- that form does not vanish for a discretely
divergence-free field.)

**The sweep.** Both codes, `nx1 = 64`, `nx2 = 32`, one block in x2, run to a full crossing, only
the meshblock size varying:

| blocks along x1 | Artemis `max abs(divB)` | RIOT `max abs(divB)` |
| --- | --- | --- |
| 2 | 1.123e-16 | 1.300e-16 |
| **4** | **1.123e-16** | **1.412e-09** |
| **8** | **1.123e-16** | **4.139e-10** |

The donor is not merely below threshold -- it is **identical to the last digit** across all three
decompositions, i.e. bitwise decomposition-invariant. RIOT is flat only at two blocks and then
loses seven orders. That is as clean an attribution as this comparison can produce: the defect
was introduced by the port.

The donor run used its deck's own `tlim` (2.24, one crossing, 702 cycles) rather than the
`nlim = 200` of the first control, so it is not a case of the donor stopping before the
triggering event.

### Why every earlier test missed it

Every CT test recorded above used at most TWO blocks along any axis:

- G4.2 2D field loop: 4 blocks as 2 x 2.
- G4.3 3D axial-field test: 8 blocks as 2 x 2 x 2.
- P01 MPI rank invariance: ranks 1/2/3/4/5/8 -- but always over that same 2 x 2 x 2 layout.
  Sweeping the RANK count over a fixed block layout does not vary the block layout, and it is
  the layout that matters here. P01's conclusion (results are rank-invariant) is still true and
  still passes; it simply never covered this.
- G4.4 Orszag-Tang: 4 blocks as 2 x 2.
- G5.5 reconstruction certification, G5.6 restart, G5.8 LLF, G5.9 instrumentation: all reused
  those decks.
- Brio-Wu is 1D, where `divB = d_1 B1` with B1 uniform, so it is identically zero whatever the
  decomposition and cannot detect this.

The gap was systematic, not accidental: the two-block layout is what the four MHD input decks
ship with, and every test inherited it.

### Consequence for claims already made

`CAPABILITY_MATRIX.md` has been corrected: "CT face update preserves div B" and the MPI row
are now qualified as validated only for at most two blocks per axis. No result recorded above
is retracted -- each was measured on the layout it states -- but the general claim that
constrained transport preserves the constraint is now known to be false outside that envelope.

---

## G5.11 — `tst/scripts/mhd/` regression harness, 2026-09-19

The MHD tests are now suite members driven by `tst/run_tests.py mhd`, not ad-hoc scripts.

| Field | Value |
| --- | --- |
| Command | `cd tst && python run_tests.py mhd --reuse_build --save_build` |
| Artifacts | `tst/scripts/mhd/{__init__,brio_wu,field_loop,orszag_tang,cpaw}.py`, `tst/scripts/utils/mhd_analysis.py` |
| Result | **3 of 4 pass. `field_loop` FAILS, and the failure is real — see D01.** |

| Test | Time | Result |
| --- | --- | --- |
| `mhd/brio_wu` | 4.2 s | **PASS** |
| `mhd/cpaw` | 11.3 s | **PASS** |
| `mhd/orszag_tang` | 2.7 s | **PASS** |
| `mhd/field_loop` | 97.5 s | **FAIL — defect D01, not a threshold problem** |

Total under two minutes, which matters: a suite slower than the hydro suite (carbuncle alone is
159 s) would get skipped.

### Where the helper module lives, and why

`scripts/utils/mhd_analysis.py`, NOT `scripts/mhd/`. `run_tests.py` treats every module it finds
in a suite directory as a test and calls `run()`/`analyze()` on it, so a helper placed next to the
tests is collected and fails. `scripts/utils` is excluded from collection.

### Design decision: no vendored reference data (concern C8)

`brio_wu` asserts no absolute L1 against the Athena++ reference. That reference lives outside this
repository, and vendoring it into `tst/scripts/gold/files` would change the gold release assets,
which is a user decision. Instead the test asserts only oracle-independent criteria, which are the
stronger evidence anyway:

- Structural invariants that are EXACT statements rather than tolerances — `Bx` stays 0.75 (the
  field is uniform along x, so `div B = d_1 B1 = 0` identically), and `vz`/`Bz` stay identically
  zero because the problem is coplanar and nothing can generate them. All measured at exactly 0.0.
- **The solver diffusivity ordering HLLD < HLLE < LLF**, which follows from wave structure (five
  resolved waves vs two vs none) and needs no reference solution. This is the criterion that
  catches a silently degraded solver — an HLLD whose guards fire everywhere would leave every
  other check in the file passing.

#### How the ordering is measured without a reference, and without stacking the deck

Each solver's L1 is taken against a 4x-resolution run RESTRICTED onto the coarse grid by exact
volume averaging (`restrict1d` — on a uniform mesh the cell average over a group of fine cells IS
the coarse cell average, so this adds no error of its own; interpolating instead would add an
O(dx^2) term of unknown sign to every comparison).

Measuring against a fine run of solver X flatters X, because X's truncation error is self-similar
across resolutions and partly cancels. So each inequality is asserted against the reference that
biases AGAINST it:

| Inequality | Reference used | Which solver that favours |
| --- | --- | --- |
| HLLD < HLLE | fine HLLE | HLLE — i.e. against the claim |
| HLLE < LLF | fine LLF | LLF — i.e. against the claim |

Neither inequality is ever evaluated against a reference built with its own winner. Measured
ratios, all five fields, both comparisons:

| field | HLLD/HLLE (ref: fine HLLE) | HLLE/LLF (ref: fine LLF) |
| --- | --- | --- |
| rho | 0.659 | 0.829 |
| press | 0.708 | 0.798 |
| vx | 0.821 | 0.872 |
| vy | 0.723 | 0.843 |
| By | 0.731 | 0.832 |

Strictly monotone everywhere with 13-34% margin, and the HLLD/HLLE column reproduces the 0.65-0.80
range measured against the genuinely independent Athena++ reference in G5.2 — which is worth more
than either number alone, since the two use different oracles and agree.

### Thresholds

`field_loop` and `orszag_tang` take the DONOR's thresholds verbatim from
`artemis/tst/scripts/mhd/{field_loop,orszag_tang}.py`, together with its metric DEFINITIONS — a
threshold is meaningless without the normalization it was calibrated against. `cpaw` uses the
donor's linwave convergence criterion (ratio <= 0.35).

**One threshold is tight and is recorded as such rather than quietly adjusted:** cpaw's N=32/16
ratio is 0.3406 against the 0.35 bound, a 2.7% margin. That is the donor's frozen number and the
measured value, so it stays; but it means this criterion will flag on a small regression, and the
first response should be to read the observed-order column (1.554 there, rising to 1.735 at
N=64/32) rather than to loosen the bound. The N=16 point is simply too coarse for an asymptotic
ratio.

### Coverage differences from the donor, stated rather than glossed

- The donor's second field-loop case is a loop tilted onto a box DIAGONAL. RIOT's generator offers
  `loop_axis` (1/2/3), which rotates the loop onto each axis, so the 3D case sweeps all three axes
  — covering every pair of EMF components — but nothing here propagates a loop obliquely to the
  grid (concern C9).
- The 2D field-loop case runs with `v3 = 1.0`, overriding the deck's 0.0. With `v3 = 0` the
  axial-field criterion is VACUOUS: the check is that `d_t B3 = v3 (d_1 B1 + d_2 B2) = v3 div B`
  cancels between `E1 = -v3 B2` and `E2 = +v3 B1`, and with `v3 = 0` there is nothing to cancel.
  Turning it on is what made the suite non-trivial — though the failure it exposed (D01) turned out
  to be independent of `v3`.
- `cpaw` pins `plm` rather than sweeping reconstruction. With `rk2` at fixed CFL, `weno5`/`mp5`
  would fail a 0.35 ratio while being far MORE accurate, because the O(dt^2) term floors them
  (G5.5). Reconstruction coverage is G5.5's job, not this test's.
- `orszag_tang` checks the t=0 snapshot as well as the final one. Every criterion except shock
  heating is meaningful at t=0 and is a STRONGER statement there, because the analytic box means
  are exact for the discrete initial state — so that snapshot validates the problem generator
  rather than only the time integration.

---

## D01 root-cause hunt, round 1 — what has been ELIMINATED, 2026-09-19

Root cause **still open**. This section exists so the next pass does not repeat any of it. All
of these were run against the reproducer (48x48 square periodic box, meshblock varied at fixed
resolution) unless noted.

### Ruled out

| Hypothesis | How it was ruled out |
| --- | --- |
| Reconstruction stencil width | Fails with **CONSTANT** reconstruction, which has no stencil at all. Also fails with PLM and PPM4. |
| Ghost budget | `nghost` 2, 3 and 4 all fail, and *slightly worse* with more ghosts. |
| Riemann solver | Fails identically with `mhd_hlle`, `mhd_hlld`, `mhd_llf`. |
| MPI / communication order | Serial and 4 ranks give the **same value to every digit** (1.412e-09). |
| Flux corrections clobbering the edge register | Skipping `SetFluxCorrections` entirely (env-var patch, safe with `refinement=none`) gives **the identical 1.412e-09**. This was a genuine suspicion: flux correction touches the edge EMF register and runs AFTER the EMF assembly but BEFORE the face update, so an EMF-agreement check taken at assembly time cannot see it. It is not the cause. |
| **The face stage register copy** (`sparse_update::DeepCopyFaceData`) | Instrumented the copy to verify `u1 == u0` over `IndexDomain::entire` on all three face elements after every copy: **zero mismatch across 70 cycles**. This is the check U05 was supposed to be and never was (concern C11) — so U05's substance now has a positive result, on a live mesh, even though the unit test still does not exist. The copy is complete and is NOT the defect. |
| Shared **edge EMF** disagreement between adjacent blocks | Instrumented `AssembleEdgeEMF` to compare block b's E3 at its high-i boundary against block b+1's at its low-i boundary: **exactly 0.0 at every call through cycle 70**, including the cycles where div B jumps. Caveat: this diagnostic's cross-block bookkeeping was unreliable (see below), so it deserves one clean re-run before being treated as final. |
| A structural difference from the donor | Line-by-line comparison found **no difference** in: face-field metadata (identical flag set), EMF assembly loop bounds, `ApplyFaceUpdate` loop bounds and RK combination, the Cartesian curl expressions term by term, or the task graph (EMF after the flux task, flux-correction send depending on the EMF, face update depending on `emf | set_flx`). RIOT's `divB` diagnostic is also formula-identical to the donor's. |

### Positively established

- The dumped face field is **genuinely divergent**: recomputing `div B` by hand from the dumped
  face values reproduces the code's own diagnostic **exactly (difference 0.0)**, so this is not
  a diagnostic artifact.
- The divergence is confined to **one block** (the one at `lx1 = 0`, which owns the periodic
  wrap boundary): 1.412e-09 there versus 7e-17, 7e-17 and 3e-17 in the other three.
- The error is **proportional to the field amplitude** — `max|divB| / max|B|` is flat at
  1.4e-06 to 3.0e-06 across `amp` = 1e-3, 1e-2, 1e-1, 1.0. So it is a relative inconsistency,
  which rules out uninitialized memory or data imported from an unrelated location.
- It is **specific to the field loop among the MHD tests**: CPAW gives `max|divB|` **exactly
  0.0** at 2, 4 and 8 blocks, and Orszag-Tang is **identical** (1.990e-13) at 2x2 and 4x4. The
  field loop is the only one of the three with a large region where B is exactly zero, and the
  only one whose fluid state is trivial (uniform rho, P, v for all time).
- It needs **periodic** boundaries on the split axis: 4 blocks with `outflow` is clean.

### A methodological trap worth recording

Two conclusions in this hunt were nearly wrong because of uncontrolled variables:

1. **"It only fails in x1, not x2."** False. That came from a run where the box was 2:1 and the
   resolution anisotropic, so splitting x1 and splitting x2 were not comparable. On a square box
   at equal resolution both directions fail (1.412e-09 and 1.228e-10).
2. **"3 blocks is fine, 4 is broken."** True, but the first version of the sweep varied `nx1`
   together with the meshblock size, so part of the effect was resolution. Only the fixed-`nx1`
   sweep establishes it.

Both are the same mistake as the C13 attribution error: a comparison with more than one thing
moving. The reproducer script now holds the resolution fixed for exactly this reason.

Also: the in-code cross-block diagnostics were unreliable because RIOT splits the mesh into
several `MeshData` partitions per stage, so a single task invocation does not see every block,
and accumulating across invocations gave irregular results (19 reports for 140 invocations).
Anything comparing blocks must either dump to a file and post-process, or be written to tolerate
partial block sets. The `u1 == u0` check above avoided the problem entirely by being block-local,
which is why it is the most trustworthy measurement in this section.

### Narrowed search space for the next pass

What remains, given the eliminations: the defect produces a face field that is genuinely
divergent in the block owning the periodic wrap, with correct EMFs, a correct stage register, and
no structural difference from a donor that is provably clean. The two candidates that survive:

1. **The shared-face/ghost exchange for a `Face` + `FillGhost` field at a periodic boundary when
   the wrap partner is not also the interior neighbour.** Consistent with: needs periodicity,
   needs >= 4 blocks (with 2 or 3 blocks on a periodic axis every block neighbours every other),
   and localizes to the block at `lx1 = 0`. The EMF measurement above says the inputs agree, so
   an exchange that overwrites a correctly-computed face with a differently-computed one is the
   remaining way to get a divergent field.
2. A re-run of the shared-edge EMF check with reliable (file-dump) bookkeeping, to confirm
   agreement rather than infer it from a diagnostic that was demonstrably flaky.

Start with (2), because it is cheap and it decides whether (1) is the only survivor.

---

## D01 — **RESOLVED 2026-09-19.** Root cause: sparse physics deactivating blocks

**Root cause: `physics/sparse_physics` (default TRUE) is incompatible with constrained
transport.** Fixed by forcing it off when MHD is enabled, with a warning.

### The mechanism

`ccbulk::cell_delta` is a sparse field whose thresholds are set from `sparse_physics`
(`riot.cpp:286-290`). With sparse physics on, Parthenon deallocates it on blocks where nothing
is changing. `riot::GetPack` (`variables.hpp`, the `include_blocks` branch) then **drops those
blocks from every pack built through it**:

```cpp
if (has_cell_active_field) {
  auto include_blocks = data->AllocationStatus(block_active_flag);
  return desc.GetPack(data, include_blocks);   // deactivated blocks are NOT in the pack
}
```

`MHD::AssembleEdgeEMF` and `MHD::ApplyFaceUpdate` take their block loop bound from
`pack.GetNBlocks()`, so a deactivated block is silently skipped. For cell-centered physics that
is precisely the intended saving -- an unchanging block needs no update. **For CT it is fatal:**
the skipped block's face field is frozen while its neighbours go on updating the faces they
SHARE with it, so that block's discrete divergence budget stops balancing. And because CT
preserves whatever divergence exists, the error is then frozen in for the rest of the run.

Caught red-handed by a diagnostic printing both counts:

```
H 13 nblocks=3 NumBlocks=4      <-- MeshData has 4 blocks; the PACK has 3
call 13 distinct (b,xmin) pairs: [(0,-0.5), (1,0.0), (2,0.5)]   <-- the block at -1.0 is gone
```

The missing block is the one at `xmin = -1.0`, which is exactly the block the divergence was
localized to.

### Why every symptom follows

| Observation | Explanation |
| --- | --- |
| Needs >= 4 blocks along an axis | A block must lie ENTIRELY outside the field loop to stop changing and be deactivated. With 2 or 3 blocks of that size every block contains part of the loop. |
| Only the field loop | It is the only MHD test with large regions where nothing changes. CPAW has a wave everywhere (`max\|divB\|` exactly 0.0 at 2/4/8 blocks) and Orszag-Tang has everything changing (identical at 2x2 and 4x4). |
| Error proportional to field amplitude | It is the difference between a frozen face and its updated neighbour, which scales with the field. |
| Appears suddenly at one cycle, then frozen to the last digit | The cycle the block is first deactivated; CT preserves the injected error thereafter. |
| Periodic needed, outflow clean | Different activity pattern near the boundary, so the outflow case did not deactivate that block within the run. |
| Independent of solver, reconstruction, nghost, rank count | None of them affects which blocks are active. |
| **The donor is clean** | Artemis has no block-deactivation mechanism at all. This is precisely how the port introduced the defect: a CT scheme was moved into a code that skips blocks, and nothing in the donor could have warned about it. |

### The fix, and why disable rather than reject

`riot.cpp` now forces `sparse_physics = false` when `do_mhd`, with a `PARTHENON_WARN`. This
follows the precedent immediately above it in the same function, which does the same for global
solvers (`do_multigroup_diffusion || do_radiation_transport || do_ionization`). The result is
correct physics and no existing MHD deck has to change.

**Making only the MHD packs ignore the block-active filter would NOT have been sufficient.** The
EMF is assembled from the hydro flux registers, and `Hydro::CalculateFluxes` packs through the
same helper -- so on a deactivated block those fluxes are never computed either. The fix has to
keep every block active, not just include them in the CT loops.

### Verification

| Check | Before | After |
| --- | --- | --- |
| Reproducer sweep, 48x48, blocks along x1 = 2 / 3 / 4 / 6 | 6.8e-17 / 6.5e-17 / **6.2e-08** / **6.4e-11** | 6.8e-17 / 6.5e-17 / **7.2e-17** / **6.4e-17** — all clean |
| 2D field loop, 64x32, 4 blocks along x1 | 1.412e-09 | **1.300e-16**, i.e. identical to the 2-block value |
| 3D field loop, 4 blocks along x1 | 2.879e-08 | **7.067e-17** |
| `run_tests.py mhd` | 3 of 4 (`field_loop` FAILED) | **4 of 4 PASS** |
| `ctest` | 49/49 | 49/49 |
| Hydro suite, MHD off | 7/7 | 7/7 (the gate is `do_mhd && sparse_physics`, so hydro-only runs are untouched) |

The regression test that found this now guards it: `tst/scripts/mhd/field_loop`'s planar case
uses four blocks along x1, so a reintroduction fails the suite.

### What this says about the earlier eliminations

Every elimination in round 1 stands -- none of those mechanisms was the cause. But two of the
round-1 conclusions were reported with more confidence than the measurement supported, because
the cross-block bookkeeping was broken in two successive ways:

1. `md->GetBlockData(b)` is **not** aligned with the pack's block index `b`. Two iterations
   aliased onto one block, which is what made "only 3 distinct blocks" look like a partitioning
   quirk rather than the actual finding.
2. `pmb->coords.Xf<X1DIR>(is)` also collided across blocks.

The identity has to come from the PACK: `pack.GetCoordinates(b)`, read on device inside the same
kernel that reads the data. Once that was done the diagnostic immediately printed
`nblocks=3 NumBlocks=4` and the cause was obvious. The lesson is that a diagnostic comparing
blocks needs its block identity verified before its conclusions are trusted -- an aliased
identity produces confident, wrong "no mismatch" results.
