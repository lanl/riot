# Artemis → RIOT ideal-MHD port

Agentic-AI-assisted work record, retained per RIOT's contribution guide.
Started 2026-09-18.

## What this is

Porting the ideal-MHD capability of the `lanl/artemis` branch `dempsey/mhd`
(`3e5aeb5`, [PR #123](https://github.com/lanl/artemis/pull/123)) into RIOT:
face-centered constrained transport, Gardiner–Stone upwind EMFs, and MHD
HLLE/HLLD/LLF Riemann solvers.

Target: a **certified Cartesian, uniform-grid, single-material, ideal-gas
ideal-MHD capability**, verified against Artemis and against analytic identities.
Hydro-only behavior must be unchanged. Everything outside that matrix is rejected
at startup rather than run untested.

The work is not "add an induction equation." It is making magnetic flux, bulk
conserved energy, thermodynamic recovery, RK stage scheduling, and mesh
communication agree on one state.

## Read order

| File | Contents |
| --- | --- |
| [`REVISION_MANIFEST.md`](REVISION_MANIFEST.md) | Pinned SHAs, submodules, toolchain, build configurations. Immutable. |
| [`SOURCE_MAP.md`](SOURCE_MAP.md) | Donor responsibility → verified RIOT owner, with file:line evidence and corrections to the original planning assumptions |
| [`DONOR_DELTA.md`](DONOR_DELTA.md) | All 67 changed donor files, classified and given a port/adapt/reimplement/skip/defer status with rationale |
| [`adr/001-units.md`](adr/001-units.md) | Magnetic units: one `mhd/mu0` parameter, `E_mag = B²/(2μ₀)` |
| [`adr/002-energy.md`](adr/002-energy.md) | Conserved-energy ownership. **Read before touching any energy code.** |
| [`adr/003-field-topology-and-stages.md`](adr/003-field-topology-and-stages.md) | Face/edge topology, metadata, stage ownership, task graph, and the two RIOT integration hazards |
| [`adr/004-support-matrix.md`](adr/004-support-matrix.md) | What is claimed, what is rejected, and the honesty constraints on the final report |
| [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md) | Per-capability validated / implemented-unverified / rejected status |
| [`TEST_LEDGER.md`](TEST_LEDGER.md) | Every test run: command, threshold, result, artifact |
| [`CHECKPOINT.md`](CHECKPOINT.md) | Current state and the next safe action |
| [`DEFERRED_STAGES.md`](DEFERRED_STAGES.md) | **Stages 6–9: what was planned and NOT built** — AMR, curvilinear geometry, general EOS, two-temperature, multi-material, per-source-package coupling. Their gates, ordering constraints, and what needs scientific review before any code. Read before proposing to extend the port. |
| [`OPEN_CONCERNS.md`](OPEN_CONCERNS.md) | **Doubts about work already done** — where a defect could still be hiding behind a passing test. Distinct from CHECKPOINT's "not implemented" list. Read before claiming any capability. |

The original planning documents produced before any local source audit are in
`claude_sessions/artemis_riot_mhd_plan/`. They are preserved deliberately: where
local evidence contradicted them, the ADRs record what changed and why.

## The three decisions that shape everything

1. **Energy** — `ccbulk::total_material_energy` includes `B²/(2μ₀)` when and only
   when MHD is enabled. One conservative energy update; machine-precision total
   energy conservation. ADR-002.
2. **Units** — one `mhd/mu0` parameter. Default `4π` (Gaussian CGS, matching
   RIOT's CGS EOS/opacity). Regression tests use `mu0 = 1`, which reproduces the
   donor's normalization exactly and is what makes direct comparison possible
   with no reconciliation step. ADR-001.
3. **Scope** — single material, Cartesian, uniform grid, ideal gas. Everything
   else rejected at startup. ADR-004.

## Two findings worth knowing before reading the code

**No dependency change is needed.** RIOT's pinned Parthenon (`928544a6d`) is
byte-identical to the donor's in every CT-relevant file, and additionally carries
a flux-correction race fix the donor's older pin lacks. It supplies
`ProlongateInternalTothAndRoe`, area-weighted face/edge restriction,
single-valued shared faces, and edge flux correction. Two donor files
(`prolongation.hpp`, `restriction.hpp`) were therefore **not** ported — the
framework already provides them.

**RIOT had no genuine face field before this.** All six pre-existing "face" fields
carry `Metadata::CellMemAligned` (cell-aligned layout, never ghost-exchanged).
Two existing routines assumed cell topology and would have silently corrupted a
CT field — one truncating the stage register copy, one applying a cell-centered
flux divergence to it. Both are fixed in `src/riot_utils/sparse_update.hpp`; see
ADR-003 for the audit that determined the correct fix, including two assumptions
from the original plan that the audit invalidated.

## Verification stance

Three independent oracle classes, because donor agreement alone cannot detect a
defect that both implementations share:

1. Analytic identities and conservation/divergence invariants.
2. The pinned donor at matched units, EOS, mesh, CFL, solver, and final time.
3. `artemis/tst/scripts/mhd/athena_bw.std` — an Athena++ Brio–Wu solution, a
   genuinely third-party reference.

Thresholds are frozen before evaluation and are the donor's own demonstrated
values, not invented ones. A skipped test is never recorded as a pass. Gold data
generated by this implementation is never presented as validation of it.

## Current status

See [`CHECKPOINT.md`](CHECKPOINT.md). In short: infrastructure and the numerical
primitives are in and unit-tested; the solver, EMF, and CT update are not yet
written, so **no MHD simulation has been run**.
