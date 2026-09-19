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

Last updated: 2026-09-18, after commit `bd51645` (reconstruction certification).

---

## C1 — Eight regression suites have never been run [HIGH]

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

**To close:** `cd tst && python run_tests.py --reuse_build --save_build` for each suite with MHD
off, and record counts in the ledger. Expect several hours; `rt_unigrid` alone is ~5 min.

## C2 — Startup rejections (N01) have never been executed [HIGH]

The ledger has said `NOT RUN` since Stage 1. The port claims to reject, at startup: `nmat > 1`,
non-Cartesian coordinates, `refinement != none`, `use_general_pte`, `fixed_fluid`, and every one
of strength / mix / tn / ionization / levelsets / multigroup diffusion / radiation transport /
lasers / prescribed sources / scalars / tracers.

**None of those rejections has been observed to fire.** Untested rejection code is code that
might not reject. This is the highest-consequence gap in the port, because every other
verification result is conditional on the unsupported configurations being unreachable — if
`nmat > 1` with MHD silently runs instead of aborting, it produces a physically meaningless
answer with no warning, and the entire certified-matrix claim is void.

**To close:** one run per rejected combination, asserting a nonzero exit and a message that
names the offending option. Cheap — each aborts during initialization.

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

## C5 — Restart of face-centered state is untested [MEDIUM]

`fbulk::magnetic_field` is `Independent`, so it should be written to and read from a checkpoint
by the normal machinery, but this has never been exercised. A restart that silently loses or
mis-shapes face B would produce a run that continues plausibly and diverges from the
uninterrupted trajectory.

Two sub-cases, neither tested: restart at the same rank count (should be bitwise), and restart
at a *different* rank count. Also untested: loading a hydro-only checkpoint into an MHD run,
which should be refused rather than silently zero-filling the field.

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

Nothing yet. Items move here with the evidence that closed them, and are not deleted.
