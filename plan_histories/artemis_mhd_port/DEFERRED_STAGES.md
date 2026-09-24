# Deferred stages 6–9 — what was planned but NOT built

This port executed **Stages 0–5** of a nine-stage plan: a certified Cartesian,
uniform-grid, single-material, ideal-gas ideal-MHD core. Stages 6–9 were
deliberately deferred, and every capability they would add is **rejected at
startup today** (see [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md) and
[`adr/004-support-matrix.md`](adr/004-support-matrix.md)).

This file exists because the originating plan lived in
`claude_sessions/artemis_riot_mhd_plan/`, which is **not tracked by git**.
ADR-004 referred to "Stage 9 of the supplied plan" for the per-package
coupling contracts — a dangling reference to a file a future reader would not
have. The substance is therefore recorded here, in the repo.

**Nothing below is a commitment.** Each stage is a separate project needing its
own scope approval, and several need scientific review before any code.

## What "unfinished" does and does not mean

The deferred capabilities are not half-built. In every case the port either
rejects the configuration at startup with an actionable message, or (for sparse
physics) forces it off with a warning. There is no partially-wired path that
could be reached by flipping a flag. The one thing that *looks* built and is
not is AMR: the face field carries `ProlongateInternalTothAndRoe` and
`RestrictAverage`, and Parthenon supplies coarse/fine edge correction, so the
metadata is complete and correct. **Registered refinement operators are not
evidence that refinement works.** Artemis's own MHD tests never exercise AMR,
so there is no donor oracle for it either.

## Stage 6 — Static refinement, dynamic AMR, load balancing

### Read this first: the donor HAS face/edge AMR code, and RIOT did not port it

Corrected 2026-09-23. Two donor commits on `dempsey/mhd` are explicitly this work:

| Commit | Subject | Touches |
| --- | --- | --- |
| `8529742` | *amr support for face and edge fields* | `mhd.cpp`, `artemis_utils.cpp` +154, `prolongation.hpp` +150, `restriction.hpp` +27 |
| `78dbc13` | *Add div(B) monitoring. **Fix issues with div(B) growth during prolongation. This comes at increased complexity.*** | `artemis.cpp`, `artemis_utils.cpp` +200, `prolongation.hpp` +451 |

The donor registers three ops on its face B field via
`EnrollArtemisFaceRefinementOps` (`artemis/src/mhd/mhd.cpp:52`): `ProlongateShared`,
`RestrictAverage`, and its own **`ProlongateTothAndRoe`** — a ~450-line
face-area-weighted least-squares divergence-constraint solve with pivoted Gaussian
elimination, templated over all six coordinate systems, and registered **for
Cartesian too**. It also prints pre-remesh and post-remesh `max|divB|`
(`artemis/src/artemis.cpp:217`).

RIOT registers `ProlongateSharedMinMod, RestrictAverage, ProlongateInternalTothAndRoe`
(`src/mhd/mhd.cpp:128`), taking the last from Parthenon. **That shares the donor
operator's name but not its algorithm** — Parthenon's is ~83 lines and a direct
formula. `DONOR_DELTA.md` originally recorded the donor file as a redundancy to
skip; that entry is now corrected with the full comparison. Read it before writing
any Stage 6 code.

`78dbc13`'s message is direct evidence that a simpler prolongation grew div B
across remesh and was deliberately replaced. Neither operator has a test in either
tree.

### First measurement, before any code

The Cartesian question is decidable in a day and each outcome picks a different
path, so it comes before design:

> Take the 2D field loop, refine a static patch, remesh, and measure `max|div B|`
> across the coarse/fine boundary **with the operator RIOT already registers**,
> using the ported `mhd/monitor_divb`.

1. **div B at roundoff** → Parthenon's direct formula is divergence-preserving in
   Cartesian, the donor's weighted solve is curvilinear generality Stage 6 does not
   need, and Stage 6 reduces to gating plus edge flux correction.
2. **div B grows** → the donor's ~450-line weighted solve must be ported, and the
   original `DONOR_DELTA.md` entry was a porting error caught before it shipped.
3. **In between** → the div B monitor is the instrument that says which.

**RUN 2026-09-23 — outcome 1, three times.** `TEST_LEDGER.md` "S6.0" (static), "S6.1"
(2D adaptive) and "S6.2" (3D adaptive, plus MPI).

| Probe | mesh activity | peak η | total E drift |
| --- | --- | --- | --- |
| uniform 2D control | none | 5.40e-15 | −2.2e-16 |
| S6.0 static, 2D | built refined, never remeshed | 9.09e-15 | — |
| S6.1 adaptive, 2D | **36 created, 24 destroyed** | 9.38e-15 | **+1.8e-15** |
| uniform 3D control | none | 4.06e-15 | −3.9e-16 |
| S6.2a adaptive, **3D** | **168 created, 112 destroyed** | 7.20e-15 | **0.0e+00** |

Adaptive is indistinguishable from static and 3D from 2D; all at roundoff. The
post-regrid energy audit **passes** in both dimensionalities: total energy and mass
conserved to roundoff across ~60 (2D) and 280 (3D) remesh events, exactly in 3D. The
magnetic-energy loss is not a remeshing defect in either — the uniform control at the
same base resolution loses **more** (12.9 % vs 6.3 % in 2D; 27.4 % vs 10.3 % in 3D), so
it is ordinary numerical diffusion and the refined run loses less by being locally finer.

S6.2 also settles the two checks 2D structurally could not make:

- **The 3D axial-field cancellation survives refinement.** `d_t B3 = v3 * div B` holds only
  through `E1 = v3 B2` cancelling `E2 = -v3 B1`, and prolongation interpolates `B1` and
  `B2` independently. Measured 2.27e-15 against 1.44e-15 uniform. This is the first time
  `MHD::UpwindEMF<X1DIR>`/`<X2DIR>` ran under refinement at all.
- **MPI with refinement is bitwise.** 1/2/3/4/5 ranks in 2D and 1/3/4 in 3D are **bitwise
  identical to serial** on all seven fields, with the identical mesh history at every rank
  count. Stronger than P01, which swept ranks on a *fixed* layout; here blocks are created,
  destroyed and re-owned mid-run. **C14's blind spot is closed.**

**The donor's weighted prolongation is not needed for Cartesian AMR in 2D or 3D.**

Two things these probes surfaced that change the shape of this stage:

1. **RIOT has no MHD refinement criterion, and its only criterion is vacuous for a
   single material.** `Hydro::CheckRefinement` votes on `volume_fraction` jumps,
   which are identically zero with one material, so it always votes derefine. An
   adaptive MHD run would never refine and would pass vacuously. The probes added a
   magnetic-energy criterion in their own build; **the tree still has none**, and
   choosing one (magnetic energy? current density? `|div B|`?) is a design question
   needing review, not a port.
2. Still untested after all three probes: **restart on a refined mesh** (the highest-value
   remaining item, given that G5.6 found derived magnetic state was never rebuilt on
   restart even on a uniform mesh), **shocks crossing a level boundary** (every probe used
   the β ~ 1e6 field loop), **more than two levels** (`numlevel = 2` throughout), and
   curvilinear, which needs the donor's version regardless.

This gate is analytic-only but **sharp, not weak**: div B at roundoff across a
refinement boundary is an exact criterion, the 3D axial field is a second exact one, and
bitwise rank-independence is a third. The absence of a donor oracle costs less here than
the "analytic-only" framing suggests.

Independent of the outcome: the donor's face-area weighting exists *for*
curvilinear geometry, so **Stage 7 needs the donor's version either way.**

### The rest of the stage

- Divergence-preserving prolongation, conservative face restriction, shared-face
  synchronization.
- Couple fluid flux correction with magnetic edge correction at coarse/fine
  interfaces. **Verify the measure**: area-integrated flux versus edge line
  integral, plus stage/time weighting. Getting this wrong injects divergence
  only at refinement boundaries.
- Sequence: one static refinement boundary → repeated refine/derefine with no
  physical evolution → advected magnetic structure crossing a level boundary →
  load balancing and MPI ownership change → restart on a refined mesh.
- **The distinctive audit**: compare conserved `E` against reconstructed
  magnetic energy after regrid. If interpolation changes `B²/2μ₀` while `E` is
  held fixed, the difference silently becomes heat. Quantify it; do not
  "repair" `E` without accounting and review.
- Confirm refinement criteria and diagnostics read valid face/derived state
  after a mesh change.

**Gate G6**: no divergence injection attributable to regrid, no coarse/fine
conservation defect, stable pressure recovery at refinement boundaries,
uniform-grid and decomposition comparisons still pass. Any energy-repair policy
needs its own ledger entry and approved scope.

Related constraint already recorded: `physics/sparse_physics` is forced off
under MHD because constrained transport cannot tolerate a skipped block. AMR
work should revisit that — the durable fix is to make a block carrying a CT
field ineligible for deactivation, rather than disabling the optimization
globally. See the D01 section of [`TEST_LEDGER.md`](TEST_LEDGER.md).

## Stage 7 — Curvilinear geometry and physical boundaries

Treat each geometry/boundary combination as a separate feature.

- Derive face/edge metrics and the curl/divergence identities per geometry.
- Derive and implement the magnetic contribution to the momentum geometry
  sources in RIOT's component conventions.
- Handle inactive coordinates without dropping physically nonzero transverse
  fields or metric terms.
- Implement physical-boundary parity and the EMF constraint **together**. Start
  with regular domains, excluding axes and poles; add coordinate singularities
  as a separate patch.
- Add cylindrical/spherical analytic equilibria and waves; test a uniform
  physical Cartesian field expressed in curvilinear components.
- Validate AMR **per geometry**. Cartesian AMR does not certify cylindrical or
  spherical AMR.

**Do not transcribe the donor here.** ADR-003 records an O(Δ) scale-factor
inconsistency in Artemis's curvilinear EMF (`gb_lo_left` / `gb_*_right`
pairing). It must be re-derived. RIOT also caps spherical at 1D and cylindrical
at 2D (`src/riot_driver.cpp:58-75`), which bounds what can be claimed at all.

**Gate G7**: per claimed geometry — free-stream/equilibrium, divergence,
conservation with the correct measure, boundary, and convergence evidence.

## Stage 8 — Thermodynamics extensions (three independent subprojects)

Do not enable these simultaneously; each has its own gate.

### 8A — Single-material general EOS

Preserve EOS-specific energy/pressure references, recover thermal state without
magnetic contamination, obtain an appropriate acoustic derivative. **Start with
HLLE**; certify `mhd_hlld` only after auditing its thermodynamic assumptions
(the donor restricts HLLD to an ideal gas, and RIOT rejects the combination
today). Test a smooth manufactured case with a non-gamma-law EOS plus a
reference shock problem; check table-domain handling and floor diagnostics.

**Gate G8A**: ideal-gas regressions unchanged; nonideal inversion, wave
propagation and energy balance pass. The matrix must name certified EOS/solver
*pairs* — one passing test does not certify "all EOS".

*This is the smallest self-contained next step of the four stages.*

### 8B — Single-material two-temperature plasma

Use RIOT's existing electron variables. Start fully ionized with exchange,
conduction and radiation off; add electron–ion exchange as a separate check.
Audit pressure composition and characteristic speeds. **Do not import the
donor's dual-energy variable as electron energy** — it is not the same thing.

Required evidence: correct electron adiabatic work under smooth compression; no
artificial electron heating from a static uniform field; correct conserved total
with a separate electron partition; correct exchange at fixed magnetic and
kinetic energy. Test energy and entropy transport separately where both exist,
and document the numerical shock-heating partition.

**Gate G8B**: explicit energy ledger and thermal-closure consistency in 1D and
multi-D, plus restart/AMR for any claimed combination.

### 8C — Mixed materials and sparse allocation

**Requires human approval of the common-field / common-velocity mixture model
before any code is written.** This is a physics modeling assumption, not an
implementation detail, and is the reason `nmat > 1` is rejected today.

Keep material partial densities as partial densities, not independent
conducting fluids. Reuse RIOT's closure with **thermal** energy. Preserve
volume-fraction and material-flux consistency; define mixture sound-speed
semantics with the closure owner.

Tests: differing material IDs and orderings; an initially absent material
crossing a block boundary; allocation/deallocation; passive composition
advection under uniform `b`; a mixed-cell equilibrium; a magnetized material
interface; regridding through mixed cells.

**Gate G8C**: bulk results invariant to arbitrary material numbering (relabeling
identical-EOS materials must not change the solution beyond roundoff); no
special magnetic behavior for material 0; partial-density sums and closure
residuals valid. Combine with 8B only after both isolated paths pass.

## Stage 9 — Source-package compatibility and production qualification

This is the stage that would license the phrase "RIOT multiphysics with MHD".
Until it is done, the supportable claim remains **donor ideal-MHD parity on the
certified matrix**.

One row per currently-rejected package. Each needs the contract *and* the test:

| Candidate coupling | Required contract / test |
| --- | --- |
| Gravity | Momentum and gravitational-work update preserve the intended total-energy balance and the magnetic state |
| Prescribed heating / lasers | Deposited heat reaches the intended thermal/electron reservoir exactly once; uniform `b` unchanged at zero flow |
| Electron–ion exchange | Equal and opposite transfer; no net change to closed-system `E` |
| Radiation transport / P1 | Matter-plus-radiation budget closes; source rebuilds retain magnetic energy |
| Existing conduction | Thermal diffusion does not diffuse `b`; closed-boundary heat budget closes; isotropic transport distinguished from magnetized anisotropic transport |
| Burn / ionization | Composition and binding/ionization energy consistent with EOS conventions |
| Tracers / passive scalars / level sets | Advection uses the MHD transport flow and stays consistent with material transport |
| Strength / BHR / plasma viscosity | Separate scientific and algorithmic review of combined stresses, signal speeds and source energy; defer unless required |

Also in scope: full public and available internal test suites; CPU/GPU and MPI
performance; **measured** MHD-off and MHD-on overhead (report the measurement,
do not invent a throughput target); memory growth per cell/face/edge/pack;
license and notice audit for copied algorithms and reference data.

Final integration: fetch upstream `main`, record the new SHA, review the delta
from the pinned base `193b3fa`, merge per the team's policy without rewriting a
shared branch, re-run the affected gates and the full hydro suite, and state the
exact upstream SHA the port is tested against.

**Gate G9**: release checklist complete, capability matrix truthful, no
untriaged failures, human numerical and physics review done. Publishing
branches, opening PRs, changing gold assets and triggering protected CI all
require explicit authorization.

## Ordering constraints (not free choice)

- 8C combines with 8B only after both isolated paths pass.
- Stage 7 requires AMR validated per geometry, so it interacts with Stage 6
  rather than following it cleanly.
- Stage 9's per-package work is independent row by row, so it can be done
  piecemeal — one package, one audit, one gate, one matrix row.
- GPU remains unvalidated from Stage 5 (no device available on the development
  machine). Kernels follow Kokkos portability rules and were reviewed, but
  reviewed is not validated.

## Out of scope permanently

Hall MHD, resistive induction, electron inertia, Biermann battery, ambipolar
diffusion, Nernst transport, anisotropic/Braginskii transport. The donor is
**ideal** MHD; each of these is a new physics project, not a continuation of
this port. This work must not be described as extended MHD.
