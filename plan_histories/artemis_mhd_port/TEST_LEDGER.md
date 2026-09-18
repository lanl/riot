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
| N01 | Every rejected configuration fails at startup with an actionable message | must throw | **NOT RUN** |

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
| G4 | Multi-D CT: div B at roundoff, field loop, Orszag–Tang, 3D Alfvén all permutations, MPI (P01) | NOT RUN |
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
