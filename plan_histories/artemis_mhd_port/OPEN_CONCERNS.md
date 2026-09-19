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

Last updated: 2026-09-18, after G5.6 (restart equivalence). **C2 and C5 are both resolved** — and
between them they found four real defects, which is the argument for keeping this file honest
rather than optimistic. C13 is new: restarts are not bitwise for derived primitives, and whether that is the port's
fault is **UNPROVEN** — the control that looked conclusive did not use the pre-port binary.

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

## C13 — Restarts are not bitwise for derived primitives [LOW **IF** pre-existing; HIGH if not — UNPROVEN]

Found while closing C5. Even a zero-step reload — restart, integrate nothing, dump — leaves
`c.c.bulk.velocity` differing by 2.2e-16 and `c.c.bulk.pressure` by 1.1e-15 from the
uninterrupted run's snapshot at the same cycle. The conserved fields and the face field are
bitwise; only the derived primitives differ, because they are recomputed from the checkpointed
conserved state by a different arithmetic path than the one that produced them before the dump.

**"PRE-EXISTING" IS NOT YET PROVEN — read this before repeating the claim.** The evidence is a
hydro-only control: `inputs/noh.rin` with `physics/mhd` never enabled, which shows the identical
signature (conserved bitwise, `velocity` 2.2e-16, `pressure` 3.6e-17). But that control was run
with the **current, MHD-modified binary**, not with the pre-port commit `193b3fa`.

So the control establishes something narrower than "pre-existing":

- **Ruled out**: anything in `src/mhd/`, the restart of face state itself, and the
  magnetic-energy add/subtract in `fill_shared_derived.cpp`. All of those are gated on `do_mhd`
  and none of them executes in the control.
- **NOT ruled out**: this port's Stage 1 edits to *shared* machinery, which DO run in hydro-only
  mode. Specifically `sparse_update.hpp` (`UpdateToNextStage` gaining required flags
  `{Cell, Independent}`, `DeepCopyIndependentData` gaining `Cell`) and any ungated edit in
  `fill_shared_derived.cpp`. If one of those changed which fields get copied into a stage
  register, a derived primitive could plausibly be recomputed by a different path after a
  restart — which is exactly the observed symptom.

**To settle it, run the H01 A/B**: `git worktree add /tmp/riot_base 193b3fa`, symlink `external/*`
from the main tree so the submodule SHAs are provably identical, configure with the *same* cmake
arguments, and run the zero-step reload test from G5.6 on `inputs/noh.rin` with both binaries. If
`193b3fa` shows the same 2.2e-16 / 3.6e-17, the concern is genuinely RIOT's. If it restarts
bitwise, **this is a defect introduced by this port** and the severity of this entry goes from
LOW to HIGH, because it would mean a Stage 1 change to shared machinery perturbed every physics
package's restart.

Until that runs, the honest statement is "not caused by MHD-specific code", not "pre-existing".

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
rather than bitwise. Worth raising with the RIOT team, since it is theirs rather than the
port's.

## C6 — No solver-failure, floor, or invalid-state counters [MEDIUM]

The plan is explicit that routine floor use is a failed test, not a success, and that solver
failures must not be hidden behind clipping. There is currently **no instrumentation at all** —
`grep -c counter src/mhd/` returns 0 — so the port cannot answer "did a floor fire?" except by
inspecting output positivity after the fact.

Concretely: HLLD has six degeneracy guards, all of which fall back to HLLE. If one of them were
tripping on every cell of a production run, HLLD would be silently degraded to HLLE and every
current test would still pass. The G5.1 unit test "HLLD and HLLE are genuinely different
fluxes" catches the *always*-falling-back case but not a *usually*-falling-back one.

## C7 — `mhd/monitor_divb` is a dead parameter [LOW, but user-visible]

Registered at `src/mhd/mhd.cpp:58-61` and read nowhere else in `src/`. A user who sets
`mhd/monitor_divb = true` — as `inputs/mhd/brio_wu.py` and `field_loop.py` both do — gets
silence, not monitoring. It is worse than a missing feature because it looks present.

**To close:** implement it (mirroring `artemis/src/utils/artemis_utils.cpp:28-180`) or remove
the parameter so the input deck errors on an unknown option.

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
