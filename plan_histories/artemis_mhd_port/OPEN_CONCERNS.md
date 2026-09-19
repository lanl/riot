# Open concerns

Things that are unresolved, unverified, or that felt wrong while doing the work — recorded so
they survive a context compaction or a new session. This is deliberately separate from
`CHECKPOINT.md`'s "Not implemented" list: that one tracks **features not yet written**, this one
tracks **doubts about what has already been written**, including places where a defect could
currently be hiding behind a passing test.

Rules for this file:

- Nothing gets deleted from here because it looks resolved. It gets moved to **Resolved** with
  the evidence that closed it, so a later reader can re-judge that evidence.
- Severity is about **what a defect here would cost**, not about how likely it is.
- If a concern is a guess rather than a measurement, it says so.

Last updated: 2026-09-19, after the MHD regression harness landed. **C14 is new and is the most
serious entry in this file: constrained transport does not preserve div B when a periodic axis is
split into four or more mesh blocks.** It was found by the new `tst/scripts/mhd/field_loop` test on
its first run, which is the clearest possible argument for building the harness rather than relying
on the ad-hoc runs that had already "validated" CT.

Earlier: after G5.9 (counters + div B monitor). **C2, C5, C6 and C7 are resolved.**
Between them they found six real defects and forced one correction to an earlier recorded claim
(G5.8's field-loop corroboration), which is the argument for keeping this file honest rather
than optimistic. C13 is new: RIOT restarts are not bitwise for derived primitives. An initial claim that this was
"pre-existing" rested on a control that did not use the pre-port binary; the proper A/B against
`193b3fa` has since been run and confirms it, with identical numbers.

---

## C14 — CT loses div B with four or more blocks per periodic axis [HIGH — this is a live defect, not a doubt]

Unlike every other entry in this file, this is not a place where a defect *could* be hiding. It is
a measured, reproducible defect with an attribution control already run.

**Symptom.** With the mesh resolution held fixed and only the meshblock size varied, `max|div B|`
on the 2D field loop is 6.8e-17 at two blocks along x1, 6.5e-17 at three, and **6.2e-08 at four**,
6.4e-11 at six. Fails in x1 and x2, in 2D and 3D, with every reconstruction (including CONSTANT)
and every solver, identically in serial and on 4 ranks, and at nghost 2, 3 and 4. Requires
periodic boundaries on that axis: the same layout with `outflow` is clean at 9.3e-17.

**Mechanism, pinned down but not explained.** The shared faces are single-valued (adjacent blocks
agree exactly), but the shared face value differs from a single-block run of the same problem by
4.1e-11, and 4.1e-11/dx reproduces the observed `max|div B|` to three digits. So adjacent blocks
compute different CT updates for the same shared face; single-valuing discards one, and that
block's divergence budget no longer balances. The `mhd/monitor_divb` trace shows a single injection
event at cycle 67-68 and then a value frozen to the last digit for the rest of the run — CT
faithfully preserving an error it cannot undo.

**Attribution: the port's, not the donor's — established by a matched sweep, not one data point.**
Both codes at `nx1 = 64, nx2 = 32`, one block in x2, full crossing, only the meshblock size
varying:

| blocks along x1 | Artemis | RIOT |
| --- | --- | --- |
| 2 | 1.123e-16 | 1.300e-16 |
| 4 | 1.123e-16 | **1.412e-09** |
| 8 | 1.123e-16 | **4.139e-10** |

The donor is identical to the last digit across all three, i.e. bitwise decomposition-invariant.
Its `divB` diagnostic was also checked to be the same quantity (face-area-weighted flux balance
over cell volume, `artemis/src/derived/fill_derived.cpp:266`) — had it differenced the
cell-centered field instead, its clean value would have proved nothing. Commands in TEST_LEDGER D01.

**Why it survived everything.** Every CT test in the ledger used at most two blocks per axis,
because that is what the four MHD decks ship with. P01 swept the RANK count (1/2/3/4/5/8) but
always over a fixed 2x2x2 block layout — and it is the layout, not the rank count, that matters
here. Brio-Wu cannot detect it at all: in 1D `div B = d_1 B1` with B1 uniform.

**Severity.** A divergence-constraint violation is not a tolerance to be relaxed, and four or more
blocks along an axis is an ordinary production decomposition — it is the donor's own default. Until
this is fixed, any claim that RIOT's constrained transport preserves the constraint must be
qualified to at most two blocks per axis, which is how `CAPABILITY_MATRIX.md` now reads.

**Round 1 of the root-cause hunt is done and the cause is still open.** The full elimination
table is in TEST_LEDGER under "D01 root-cause hunt, round 1". Ruled out: reconstruction stencil
(CONSTANT fails too), ghost budget (2/3/4 all fail), Riemann solver, MPI, flux corrections
clobbering the edge register (skipping `SetFluxCorrections` changes nothing), **the face stage
register copy** (instrumented: `u1 == u0` over `entire` with zero mismatch across 70 cycles —
which is the substance of U05, finally measured), and any structural difference from the donor
(metadata, loop bounds, curl terms, task graph all match).

Positively established: the dumped face field is genuinely divergent (hand-recomputed `div B`
matches the code exactly), the divergence is confined to the single block owning the periodic wrap,
the error is proportional to the field amplitude, and among the MHD tests only the field loop shows
it — CPAW is exactly 0.0 at 2/4/8 blocks and Orszag-Tang is identical at 2x2 and 4x4.

**Surviving hypothesis:** the shared-face/ghost exchange for a `Face` + `FillGhost` field at a
periodic boundary when the wrap partner is not also the interior neighbour. That fits needing
periodicity, needing >= 4 blocks (with 2 or 3 blocks every block neighbours every other), and
localizing to the block at `lx1 = 0`. Before pursuing it, re-run the shared-edge EMF agreement
check with file-dump bookkeeping: the in-code version reported agreement but was demonstrably
flaky, because RIOT splits the mesh into several `MeshData` partitions per stage and a single task
invocation does not see every block.

Reproducer: `claude_sessions/mhd_runs/repro_divb_blocks.py --exe tst/build/src/riot`.

---

## C1 — Eight regression suites not yet run [HIGH severity, but SCHEDULED — do this LAST]

`tst/scripts/` contains `advection`, `hydro`, `ionization`, `levelsets`, `mix`,
`radiation_diffusion`, `radiation_transport`, `strength`, `tn`. **Only `hydro` has ever been
run in this port**, at any stage.

Why this matters more than it looks: Stage 1 changed
`sparse_update::UpdateToNextStage` to require `{WithFluxes, Cell, Independent}` and
`DeepCopyIndependentData` to require `Cell`. That is **shared machinery used by every physics
package**, not something MHD-local. The justification was an audit of all 19
`Metadata::WithFluxes` registrations in `src/`, plus a bitwise-clean hydro suite. But the audit
is exactly the kind of thing that is wrong by one entry, and the eight unrun suites are the
test of it. A field that quietly stopped being updated would show up there and nowhere else.

Note ADR-003 already records two corrections found while implementing that filter — the
originally planned `{WithFluxes, Conserved}` would have broken advected scalars and level sets.
That is direct evidence this filter is easy to get wrong, which raises rather than lowers the
value of running `levelsets` and the rest.

**This is not an oversight — it is planned Stage 5 work.** The plan schedules it there
("Confirm hydro-only regressions unchanged, including unrelated source packages"), and Gate G1
only ever required the *hydro* baseline. The severity above is about the risk still being
open, not about a step having been skipped.

**And it should stay last.** The sweep validates whatever the final state of the code is, so
running it before the remaining Stage 5 code changes (LLF, failure counters, `monitor_divb`)
would only mean running it again afterwards. Sequence it after the last code change lands.

**To close:** `cd tst && python run_tests.py <suite> --reuse_build --save_build` for each of
advection, ionization, levelsets, mix, radiation_diffusion, radiation_transport, strength, tn,
with MHD off, and record counts in the ledger. Expect several hours. Delete the stale
accumulator files first (`tst/build/src/{linwave-errs.dat,linwave_mm-errs.dat,compression.out0.hst}`)
or `analyze()` throws on doubled output and it reads as a regression that is not one. Both flags
are mandatory: without `--save_build` the runner deletes `tst/build`, and without `--reuse_build`
it reconfigures with AppleClang, which cannot compile RIOT.

## C2 — Startup rejections (N01) have never been executed → **RESOLVED**, see Resolved section

## C3 — Brio–Wu has no absolute frozen threshold [MEDIUM]

Only a **relative** claim is asserted: HLLD must beat HLLE at the same resolution. That is
sound and oracle-independent, but it means the absolute accuracy is ungated. Brio–Wu could
regress by a factor of two and every test would still pass, because both solvers would degrade
together.

The donor's `_profile_tolerance = 3.8e-3` is *not* usable: it is calibrated against
`brio_wu.std`, which is Artemis's own gold file, at the donor's resolution and solver. This was
already flagged earlier in the ledger and is still open.

**To close:** run Artemis at RIOT's exact resolution, solver and final time, and freeze a
threshold from that matched comparison. Until then, treat Brio–Wu accuracy as monitored, not
gated.

## C4 — WENO5/MP5 reach order ~4, not their formal 5, and the cap is a GUESS [MEDIUM]

Measured 4.21/3.93/4.16 (WENO5) and 3.93/3.87/4.15 (MP5) with the temporal error suppressed
(G5.5). I wrote that the cap is "plausibly the second-order EMF corner averaging in the CT
update". **That is a hypothesis I did not test.**

Why it deserves to stay open: a genuine defect in the high-order MHD reconstruction path — say
the normal-B substitution interacting badly with a wide stencil — would present exactly like
this, as a clean but wrong convergence rate. Order 4 instead of 5 is comfortable enough to
accept and precisely for that reason worth confirming.

**To close:** either identify the limiting term (a hydro-only CPAW-equivalent with the same
reconstruction would separate CT from reconstruction — if hydro also caps at 4, CT is
exonerated), or accept order 4 as the scheme's property with evidence rather than assertion.

## C5 — Restart of face-centered state is untested → **RESOLVED**, see Resolved section

## C13 — RIOT restarts are not bitwise for derived primitives [LOW — attribution now PROVEN pre-existing]

Found while closing C5. Even a zero-step reload — restart, integrate nothing, dump — leaves
`c.c.bulk.velocity` differing by 2.2e-16 and `c.c.bulk.pressure` by 1.1e-15 from the
uninterrupted run's snapshot at the same cycle. The conserved fields and the face field are
bitwise; only the derived primitives differ, because they are recomputed from the checkpointed
conserved state by a different arithmetic path than the one that produced them before the dump.

**ATTRIBUTION SETTLED by direct A/B against the pre-port commit** (2026-09-18). The first
version of this entry called the behaviour "pre-existing" on the strength of a hydro-only
control run with the CURRENT binary. That was an overstatement: it excluded MHD-specific code
(all gated on `do_mhd`) but not this port's Stage 1 edits to *shared* machinery —
`sparse_update.hpp` changed `UpdateToNextStage` to require `{Cell, Independent}` and
`DeepCopyIndependentData` to require `Cell`, and both run in hydro-only mode. A change to which
fields land in a stage register is a plausible route to exactly the observed symptom.

So the test was run properly: `193b3fa` built in a worktree with `external/*` symlinked from the
main tree (so the submodule SHAs are provably identical) and configured with identical cmake
arguments, then the G5.6 zero-step reload on `inputs/noh.rin` with both binaries.

| Binary | `velocity` max\|diff\| | `pressure` max\|diff\| | conserved |
| --- | --- | --- | --- |
| pre-port `193b3fa` | 2.220446e-16 | 2.086005e-16 | bitwise |
| current port | 2.220446e-16 | 2.086005e-16 | bitwise |

**Identical to every printed digit, not merely the same order of magnitude.** The port did not
introduce this. Severity stays LOW and the attribution question is closed; the behaviour itself
remains open, because RIOT restarts still are not bitwise.

Two side results from the same A/B, recorded because they are free evidence: the uninterrupted
`noh` run at cycle 20 is **bitwise identical** between the two binaries, which extends H01's
"hydro unchanged" coverage to a second problem and a different pgen path (H01 used
`linear_modes`); and the two zero-step reloads are bitwise identical to each other.

It is recorded here anyway for two reasons:

1. It means **no RIOT restart can be claimed bitwise**, so any future test that asserts
   bitwise restart equivalence — for any physics package, not just MHD — will fail, and the
   cause will not be in the code under test. Someone will otherwise spend a day on it.
2. It sets the floor for what a restart test can assert. G5.6's evolved comparisons use
   rel ≤ 1e-14 for exactly this reason, and that threshold is derived from the measured seed
   rather than from what happened to pass.

MHD's `pressure` residual (1.1e-15) is about an order of magnitude above hydro's (3.6e-17),
which is consistent with the extra magnetic-energy subtraction in the recovery path adding a
rounding step. That is an explanation, not a measurement — **it has not been confirmed.**

**To close:** either identify the specific operation whose ordering differs and make the
recovery path reproducible, or state in the docs that RIOT restarts are equivalent to roundoff
rather than bitwise. Now demonstrably a RIOT-wide property rather than the port's, so this is
worth raising with the RIOT team — with the A/B numbers above as the evidence, since they
predate the MHD work entirely.

## C6 — No solver-failure, floor, or invalid-state counters → **RESOLVED**, see Resolved section

## C7 — `mhd/monitor_divb` is a dead parameter → **RESOLVED**, see Resolved section

## C8 — Reference and baseline data live outside the repo [MEDIUM]

Three artifacts that verification depends on are not in the repo and are not reproducible from
it:

- `athena_bw.std` — the only third-party oracle in the whole port. Lives in the donor tree at
  `/Users/taitano/Documents/git/artemis/tst/scripts/mhd/`. If `tst/scripts/mhd/` is ever meant
  to run in CI, the comparison **silently cannot run** — `analyze_brio_wu.py` exits 2 rather
  than reporting a failure, which is the right behaviour locally and the wrong signal in CI.
- `claude_sessions/mhd_runs/artemis_baseline/` — 102 files, ~30 MB, **untracked**. Lost if that
  directory is cleaned. Regenerable, but only while the donor tree exists.
- `claude_sessions/mhd_runs/riot_hydro_baseline/` — same, untracked.

**To close:** decide whether `athena_bw.std` gets vendored into `tst/scripts/mhd/` (it is
third-party data, so provenance and licensing need a look) and make the harness fail loudly
rather than skip when it is absent.

## C9 — Verification is single-node and uniform-grid only [LOW — by design, but easy to misread]

Everything is one node, shared memory, so no interconnect is exercised; and uniform grid, so no
coarse/fine flux or EMF correction is exercised. AMR is out of scope and rejected at startup —
**but that rejection is itself untested, see C2**, which is the part that turns this from a
scope boundary into a risk.

Also: no test propagates a wave obliquely to the grid. The CPAW is axis-aligned in all three
permutations, which establishes that each direction is individually correct and mutually
consistent, not that oblique propagation is. An oblique wave couples all three EMF components
simultaneously and is strictly stronger.

## C10 — HLLD consistency is conditioning-limited for extreme states [LOW]

In the G5.1 degeneracy test, the two 1e6-field states (beta ~ 1e-12) get a tolerance scaled by
`smax * |U|`, which is large in absolute terms. The tolerance model is justified and measured —
errors sit within 2x of `eps * |s| * |U|` — but the consequence is that **consistency is not
sharply checked for those two states**. Their real value is as NaN-regression cases, which the
separate finiteness test does assert strictly. Recorded so nobody later reads that test as
tighter than it is.

## C11 — U05 is exercised but not pinned [LOW]

The face-aware stage-register copy is used by every two-stage RK MHD run, so it is certainly
exercised. It has never been asserted directly with deliberately distinct stage values, which
is what would catch a partial copy that happens to be harmless for the current integrator and
harmful for a future one.

## C12 — Transverse face velocities from HLLD are unverified in use [LOW]

`lr_to_flux_mhd_hlld` writes `v1face`/`v2face`/`v3face` using the same upwind convention as the
HLLE, inherited from RIOT's hydro solvers. The donor sets only the normal (mass-flux) velocity.
The transverse ones are consumed by advected scalars and tracers, both of which are rejected
under MHD, so the risk is currently nil — but it becomes live the moment any of those couplings
is enabled, and it will not be obvious then that this was never checked.

---

## Resolved

Items move here with the evidence that closed them, and are not deleted.

### C6 — No solver-failure, floor, or invalid-state counters [was MEDIUM] — closed 2026-09-19

**The original concern.** `grep -c counter src/mhd/` returned 0. HLLD's six degeneracy guards
all fall back to HLLE, so if one were tripping on most cells HLLD would be silently degraded
to HLLE and every existing test would still pass — the unit test asserting the two solvers
differ catches the *always*-falling-back case, not the *usually*-falling-back one.

**Evidence that closed it.** TEST_LEDGER G5.9. Two cumulative counters (`hlld_fallback`,
`density_floor`) incremented by atomics inside the flux kernels, MPI-summed, reported on
increase and normalized per cell per step so the number has a known ceiling. Two new unit
tests check both directions — a healthy state must report zero, a degenerate one exactly one,
and HLLE/LLF must never touch the HLLD counter. MHD output verified **bitwise unchanged**
afterwards, which was the real risk of threading an argument through `MHDFluxes`.

**The concern was justified: the instrument immediately corrected a recorded claim.** On the
3D field loop, HLLD falls back on ~7 faces per cell per step — the large majority — because
the field is planar and vanishes outside the loop. G5.8 had cited that problem's energy
retention as *independent corroboration* of the HLLD < HLLE ordering; it is not, because both
runs mostly execute the same solver. G5.8 is amended and Brio & Wu (zero fallbacks) is now
the sole strong evidence for that ordering.

**Residual limitations:**

- Only two counters. There is no counter for non-finite intermediate states (checking would
  cost seven `isfinite` calls per face) nor for energy-repair events outside the solver. The
  finiteness property is covered by unit tests instead.
- The per-cell-per-step denominator is the mesh **cell** count, not a true face-solve count,
  because counting faces would need an atomic on every face. So the ceiling is "roughly
  ndim × nstages", not an exact 1.0-is-everything scale. Fine for judging orders of
  magnitude, not a precise fraction.
- The atomics are unconditional. On a problem with a high fallback rate on a GPU this could
  serialize; not measured, since all verification here is CPU. **Untested performance
  concern**, flagged rather than dismissed.

### C7 — `mhd/monitor_divb` was a dead parameter [was LOW but user-visible] — closed 2026-09-19

**The original concern.** Registered at `src/mhd/mhd.cpp` and read nowhere in `src/`. A user
who set it — as two tracked input decks do — got silence, which is worse than a missing
feature because it looks present.

**Evidence that closed it.** TEST_LEDGER G5.9. Now installed via `PostStepDiagnosticsMesh`,
reporting max, volume-weighted mean, and the dimensionless eta once per step; verified
identical to every digit at 1 vs 4 ranks, and silent when off. Reuses the already-computed
`ccbulk::div_magnetic_field` so the monitor and the analysis scripts cannot disagree.

**A dead-code bug in my own first version, kept here because it is a recurring pattern.**
With `b_ref` set to the global max |B|, the expression `max(|B|_cell, b_ref)` is *always*
`b_ref`, so the per-cell term did nothing and eta had silently collapsed to the weaker global
normalization. Fixed by making `b_ref` a weak-field floor; confirmed by the reported values
changing (cycle-2 eta 9.58e-16 → 4.88e-15). Same shape as the two N01 defects: an expression
that reads correctly and evaluates to a constant.

### C5 — Restart of face-centered state was untested [was MEDIUM] — closed 2026-09-18

**The original concern.** `fbulk::magnetic_field` is `Independent`, so it should be written to
and read from a checkpoint by the normal machinery, but this had never been exercised. Three
sub-cases were listed as untested: same-rank restart, changed-rank restart, and loading a
hydro-only checkpoint into an MHD run.

**Evidence that closed it.** TEST_LEDGER G5.6, seven tests including a reproducibility control
(the same run twice, bitwise) and a hydro-only control. Face B restarts **bitwise**; the
evolved restart agrees to rel ≤ 1e-14 at both the same and a changed rank count, with the
4-rank differences bit-for-bit identical to the 1-rank ones.

**The concern was justified: it found two real defects.**

1. **Derived magnetic state was never rebuilt on restart.** `PostInitializationMesh` — hence
   `MHD::SetDerivedMagneticFields` — is inside Parthenon's `if (init_problem)` block, which is
   skipped entirely on a restart. Because `ccbulk::magnetic_field` supplies the *transverse*
   field to the Riemann solve, and the per-stage `SetDerived` task runs *after* the update, the
   first post-restart stage computed fluxes from a zero transverse field. The run continued,
   stayed divergence-free, and was permanently wrong by 3.7e-06 — nine orders above roundoff
   and invisible to every existing test. Fixed via `UserWorkBeforeLoopMesh`.
2. **Restarting across an MHD/hydro change ran silently in both directions.** MHD checkpoint
   into a hydro run reinterprets B²/(2μ₀) as heat: measured **80% pressure error on Brio & Wu,
   exit code 0**. Both directions are now rejected by a deck-based check plus a data-based
   global reduction, because each alone has a hole (see G5.6).

**Residual limitations, stated rather than glossed:**

- Derived primitives are not bitwise across a restart. I described this as pre-existing RIOT
  behaviour on the strength of a hydro-only control, but that control used the CURRENT binary,
  not `193b3fa`, so it does not actually exclude this port's Stage 1 changes to shared
  machinery. Split out as **C13**, where the gap and the test that settles it are written down.
- The zero-field escape hatch `mhd/allow_zero_field_restart` disables the data-based half of
  the checkpoint-mismatch check. A user who sets it *and* passes an `mhd/*` override on the
  command line has defeated both halves. Judged acceptable — it takes two deliberate opt-outs
  — but it is a hole, not an absence of one.
- All of this is uniform-grid. Restart with refinement is untested and AMR is rejected anyway
  (C9).

### C2 — Startup rejections (N01) had never been executed [was HIGH] — closed 2026-09-18

**The original concern.** The ledger had said `NOT RUN` since Stage 1. The port claimed to
reject, at startup: `nmat > 1`, non-Cartesian coordinates, `refinement != none`,
`use_general_pte`, `fixed_fluid`, and every one of strength / mix / tn / ionization /
levelsets / multigroup diffusion / radiation transport / lasers / prescribed sources /
scalars / tracers. None had been observed to fire. It was rated the highest-consequence gap
because every other verification result is conditional on the unsupported configurations
being unreachable.

**Evidence that closed it.** `claude_sessions/mhd_runs/n01_startup_rejections.py`, 23/23
cases passing with both a positive control (unmodified deck must exit 0) and, for the
compile-time coordinate case, a negative control (the cylindrical binary must run hydro RZ
successfully). Each case asserts a nonzero exit **and** a message fragment unique to the
specific rejection, so an abort for an unrelated reason cannot be recorded as a rejection.
Full detail in `TEST_LEDGER.md` under "N01 — startup rejection matrix".

**The concern was justified: it found two real defects, both in the general-PTE rejections.**

1. `hydro.cpp` read `pin->GetOrAddBoolean("multiphysics", "use_general_pte", false)`, but the
   option lives in `<materials>`. The read always returned the default, so the HLLD
   ideal-gas guard was **dead code that could never fire** — and being a `GetOrAdd`, it also
   silently injected a spurious `multiphysics/use_general_pte` entry into the recorded input.
2. Both the `hydro.cpp` and `riot.cpp` checks read the *input flag*, which cannot see
   `materials.cpp:450` forcing `use_general_pte = true` whenever any `eos_type` is
   non-ideal. So the most natural route to the general closure — pick a real EOS, never touch
   the flag — was the one route neither guard could detect. Confirmed by direct run:
   `eos_type=Gruneisen` sailed past both.

Both now read the **resolved** package parameter. `riot.cpp`'s general-PTE clause had to move
out of the main gate to just after `Materials::Initialize`, because the resolved value does
not exist earlier; a comment at the original site records why it is not there.

**Two residual limitations, stated rather than glossed:**

- The MHD-specific *laser* rejection (`"MHD with lasers"`) is unreachable from any valid
  input: lasers require ionization, which is itself rejected earlier in the gate. So it is
  covered by inspection, not execution. Harmless today, but it means that one line has never
  run — relevant if the ionization rejection is ever lifted.
- The non-Cartesian case needs a purpose-built binary and so is **not** part of the routine
  sweep. It will silently drop out of any CI that does not build a curvilinear RIOT. If this
  test is promoted to `tst/scripts/mhd/`, decide there whether to skip it loudly or omit it.
