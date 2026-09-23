# ADR-004 — Supported configuration matrix and rejection policy

Status: **accepted** (scope user-approved 2026-09-18)
Affects: `src/riot.cpp` startup validation, `src/hydro/hydro.cpp` solver
enumeration, documentation, every support claim made about this port.

## Question

Which configurations does the MHD port claim to support, how are unsupported
combinations prevented from running, and what is deliberately deferred?

## Principle

**Unsupported combinations must fail clearly at startup, not be accepted with
untested behavior.** Token availability, a clean compile, and a plausible-looking
plot are not substitutes for a passing numerical gate. Anything not in the
certified matrix is rejected, and the rejection is itself a tested behavior
(test ID N01).

A corollary that governs the whole port: *validated toggles do not compose*.
"AMR validated" and "2T validated" would not imply "2T + AMR validated." The
matrix therefore records combinations, not just individual switches.

## Decision — the certified target for this scope

Approved scope is Stages 0–5: a **Cartesian, uniform-grid, single-material,
ideal-gas ideal-MHD** capability. Concretely, MHD may run only with:

| Axis | Certified |
| --- | --- |
| Dimensions | 1D, 2D, 3D |
| Coordinates | `UniformCartesian` only |
| Mesh | uniform grid, `refinement = none`, multi-block, MPI 1/2/4 ranks |
| Materials | exactly one |
| Closure | `ApplyIdealGasClosure` (ideal gas); general PTE rejected |
| Thermodynamics | one temperature |
| Reconstruction | certified per-mode in Stage 5; uncertified modes rejected |
| Riemann solvers | `mhd_hlle`, `mhd_hlld` (ideal gas only), `mhd_llf` |
| Integrators | `rk1`, `rk2`, `rk3` (whichever pass the gates) |
| Boundaries | periodic; outflow in the narrow Brio–Wu configuration |
| Precision / backend | double, Kokkos host (Serial) |
| Restart | yes, including changed-rank where supported |

## Decision — rejections at startup

Implemented with RIOT's existing idiom at `src/riot.cpp:102-122`, whose
`"Radiation transport with ionization is not yet supported."` line is the exact
precedent for a not-yet-implemented combination. With `mhd = true`, reject:

**Physics packages** — strength, mix, tn (burn), ionization, levelsets,
multigroup diffusion, radiation transport, lasers, prescribed sources, scalars,
tracers. Each is rejected because it writes or reads energy, adds stresses, or
adds signal speeds that have not been audited against the magnetic terms — not
because coupling is impossible. The per-package contract each would need is
enumerated in [`../DEFERRED_STAGES.md`](../DEFERRED_STAGES.md) under Stage 9.

**Configuration** — `nmat > 1`; non-Cartesian coordinates; `refinement != none`;
`use_general_pte`; `fixed_fluid`; non-ideal EOS; the hydro-only Riemann solvers
(`hllc`, `hllcf`, `chllc`, `lhllc`, `hll`); insufficient `nghost`.

**Also required** — `mhd = true` requires `hydro = true`.

Two mechanisms are used, matching existing practice:

1. `PARTHENON_REQUIRE(...)` for cross-package combinations (`riot.cpp:102-122`).
2. The enumerated-allowed-values argument to `pin->GetOrAddString` for the
   solver list (`src/hydro/hydro.cpp:109-112`), which makes Parthenon itself
   reject an out-of-list solver name.

**No silent fallback.** If the user requests a solver or physics combination that
is unsupported, the run must stop — never substitute a default and proceed. A
silently downgraded solver is the failure mode that makes a "passing" test
meaningless.

## Explicitly deferred, with reasons

| Deferred | Status | Why |
| --- | --- | --- |
| Static/dynamic AMR, load balancing | rejected | Parthenon supplies `ProlongateInternalTothAndRoe` and edge flux correction, and the metadata is registered correctly, but **Artemis's own MHD tests never exercise AMR** — its linear-wave pgen hard-rejects multilevel (`artemis/src/pgen/linear_wave.hpp:310-311`) and every MHD input uses `refinement = none`. There is no donor oracle, so gates would be analytic-only. Refinement metadata being present is not evidence that refinement works. **Corrected 2026-09-23:** the donor does have face/edge AMR *code* (`8529742`, `78dbc13`) — untested, not absent — and the operator RIOT registered shares only a *name* with the donor's, which is a ~450-line weighted divergence-constraint solve. See [`../DONOR_DELTA.md`](../DONOR_DELTA.md) and [`../DEFERRED_STAGES.md`](../DEFERRED_STAGES.md) Stage 6. |
| Cylindrical / spherical geometry, axes and poles | rejected | Requires deriving face/edge metrics, curvilinear Maxwell-stress momentum sources, and axis parity. RIOT also caps spherical at 1D and cylindrical at 2D (`src/riot_driver.cpp:58-75`). Note ADR-003 records an O(Δ) metric inconsistency in the donor's curvilinear EMF that must be re-derived, not transcribed. |
| General (non-ideal-gas) EOS | rejected | Needs the correct frozen/equilibrated acoustic derivative for the hyperbolic step. `γp/ρ` with an invented effective γ is not acceptable, and HLLD contains constant-γ assumptions. |
| Multiple materials | rejected | The common-field / common-velocity model over a volume-additive mixture is a **physics modeling assumption requiring scientific review**, not an implementation detail. It is not a per-material magnetic equilibrium model, does not treat differing permeabilities, and does not treat separate ion/neutral velocities. |
| Two-temperature / electron energy or entropy | rejected | Electron/ion shock-heating partition is not determined by ideal induction. RIOT's existing discretization must be preserved, and the donor's dual-energy variable is **not** an electron-energy equation. |
| Hall MHD, resistivity, Biermann battery, ambipolar diffusion, Nernst, anisotropic/Braginskii transport | **not in the donor at all** | This donor is ideal MHD. Adding these is a new physics project, not a port. |
| GPU (CUDA) build | not validated | Not exercised on this machine. Kernels are written to Kokkos portability rules and reviewed by the `kokkos-portability-reviewer` subagent, but **unvalidated is not validated** and the matrix will say so. |

## Honesty constraints on the final report

Three distinct accomplishments must not be conflated:

1. **Working prototype** — compiles and produces plausible output.
2. **Donor ideal-MHD parity on the certified matrix** — this scope's target.
3. **RIOT coupled-physics qualification** — requires Stages 6–9 and human
   scientific review.

The final report states which one was reached and what remains. Specifically:

- A Cartesian uniform-grid prototype is a **milestone, not completion of the
  donor-capability port**.
- Passing 2D does not certify 3D.
- Donor agreement is necessary evidence of faithful transfer but **insufficient
  evidence of correctness** — a shared defect would agree perfectly. This is why
  the validation plan carries analytic identities and the independent Athena++
  Brio–Wu reference (`artemis/tst/scripts/mhd/athena_bw.std`) alongside donor
  comparison.
- Gold data generated by this implementation must never be presented as
  validation of this implementation.
- This work must **not** be described as porting the Yehan Toh extended-MHD
  paper, nor as extended MHD of any kind. RIOT's existing electron
  thermodynamics can coexist with ideal induction after integration; that does
  not make the result extended MHD.

## What would invalidate this decision

- A user requirement for an unsupported combination in production — which is an
  explicit escalation point, not something to be quietly enabled.
- Completion of a deferred stage with its gates genuinely passed, at which point
  the corresponding rejection is removed **and** `CAPABILITY_MATRIX.md` is
  updated with the test IDs that justify the new claim.
