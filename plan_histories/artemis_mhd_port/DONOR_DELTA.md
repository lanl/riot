# Donor delta inventory — every changed Artemis file, classified

Donor: `lanl/artemis` `dempsey/mhd` @ `3e5aeb5`
Comparison base: `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee` (sole merge base with
`origin/develop`)
Delta: **67 files, +7596 / −308, across 71 commits**

Classification: **algorithm** (numerics to transfer) · **integration**
(lifecycle/registration to re-express natively) · **test** · **doc/build** ·
**prerequisite** (donor-side fix needed by MHD) · **unrelated** (read for
lessons, do not import).

Status: **port** · **adapt** · **reimplement** (RIOT-native equivalent) ·
**skip** · **defer** (out of Stages 0–5 scope).

## Core MHD algorithm — new files

| Donor path | Class | Status | Target / note |
| --- | --- | --- | --- |
| `src/mhd/mhd.hpp` (146 L) | algorithm | **port** | `MagneticEnergyDensity`, `FastMagnetosonicSpeed`, `FaceToCellCenteredB`, `ScaleMHDFlux`, `SetCellCenteredMagneticFields` → `src/mhd/mhd_helpers.hpp` + `src/mhd/mhd.hpp`. `ScaleMHDFlux` is a Cartesian no-op — omitted, with the reason recorded. |
| `src/mhd/mhd.cpp` (111 L) | integration | **adapt** | Package registration + `AssembleEdgeEMF` geometry dispatch → RIOT `StateDescriptor` conventions. |
| `src/mhd/emf.hpp` (231 L) | algorithm | **adapt** | `CellEMF`, `UpwindEMFGradient`, `UpwindEMF`, edge assembly. Cartesian specialization written directly — see ADR-003 for the O(Δ) curvilinear metric inconsistency deliberately not transcribed. |
| `src/utils/fluxes/riemann/hlld.hpp` (550 L) | algorithm | **port** (Stage 5) | → `src/mhd/riemann_mhd.hpp`. Degeneracy guards (`eps = 1e-12`, `deg_tol = 1e-4·ptst`, non-finite bail-outs) preserved verbatim. Species-`n≥1` HLLC branch skipped — RIOT is single-material here. |

## Modified solvers, reconstruction, fluxes

| Donor path | Class | Status | Note |
| --- | --- | --- | --- |
| `src/utils/fluxes/riemann/hlle.hpp` | algorithm | **port** | The Stage-3 baseline solver. Magnetic pressure to a separate face register; normal-B flux slot written exactly `0.0`. |
| `src/utils/fluxes/riemann/llf.hpp` | algorithm | **port** (Stage 5) | Diagnostic fallback. |
| `src/utils/fluxes/riemann/riemann.hpp` | integration | **reimplement** | Donor added `mu0`/`do_mhd` to a unified functor signature. RIOT uses free functions passed as `auto FLUX_FN`; extended signature follows the `StrengthFluxes` precedent instead. |
| `src/utils/fluxes/riemann/hllc.hpp` | integration | **reimplement** | Donor only added an MHD *rejection*. RIOT expresses this in `riot.cpp`/`hydro.cpp` validation (ADR-004). |
| `src/utils/fluxes/fluid_fluxes.hpp` | algorithm + integration | **adapt** | Two essentials: total-pressure gradient assembled from gas + magnetic face registers (`:451-491`), and `ExtendMHDFluxBounds` (`:75-94`) for the extra transverse flux layer. Curvilinear curvature source deferred. |
| `reconstruction/{reconstruction,pcm,plm,ppm,wenoz,wenomz}.hpp` | integration | **reimplement** | Donor threaded a `skip_index` through all five methods to skip normal B. RIOT instead reconstructs all three components then overwrites the normal one from the shared face value — **numerically identical**, and avoids touching five RIOT reconstruction methods. Recorded as a deliberate deviation. |

## Lifecycle, driver, derived state

| Donor path | Class | Status | Note |
| --- | --- | --- | --- |
| `src/artemis.cpp` / `.hpp` | integration | **reimplement** | `<physics> mhd` toggle, package registration, compatibility gates, `divB` monitor hooks → `src/riot.cpp`. |
| `src/artemis_driver.cpp` / `.hpp` | integration | **reimplement** | Task-graph edges → `src/riot_driver.cpp`. **RIOT's driver is never replaced**; five edges are added (ADR-003). |
| `src/utils/integrators/artemis_integrator.hpp` | algorithm + integration | **adapt** | `ApplyFaceUpdate` CT kernel (`:157-259`) is the authoritative index/orientation pattern → `MHD::ApplyFaceUpdate`. `DeepCopyConservedData`'s face branch → `sparse_update::DeepCopyFaceData` (fixes RIOT Hazard B). |
| `src/derived/fill_derived.cpp` | algorithm | **adapt** | `divB`, cell-centered B, magnetic energy, and `E = u + KE + E_mag` → RIOT's `fill_shared_derived.cpp` touch points and `MHD::SetDerivedMagneticFields` (ADR-002). |
| `src/utils/artemis_utils.{cpp,hpp}` | integration | **adapt** | Face refinement-op enrollment and the `divB` remesh diagnostics. `DualEnergySIE` has **no RIOT analog** — RIOT has no dual-energy scheme, and the donor's constant-γ fallback must not be transplanted into RIOT's arbitrary-EOS path (ADR-002). |

## Units

| Donor path | Class | Status | Note |
| --- | --- | --- | --- |
| `src/utils/units.{cpp,hpp}` | prerequisite | **reimplement** | Donor added an `artemis/current` base unit and a `mu0_code` derivation. RIOT is CGS with no unit layer, so this becomes a single `mhd/mu0` parameter — see ADR-001. The donor's incidental `GetEnergyDensityCodeToPhysical` bugfix has no RIOT counterpart. |

## Geometry and AMR — deferred (out of Stages 0–5)

| Donor path | Class | Status |
| --- | --- | --- |
| `src/geometry/{geometry.hpp,geometry.cpp,cylindrical.hpp,spherical.hpp,axisymmetric.hpp}` | algorithm | **defer** — edge scale factors `hx1e1/hx2e2/hx3e3` for curvilinear CT (Stage 7) |
| `src/utils/refinement/prolongation.hpp` | algorithm | **deferred, NOT equivalent** — see the correction below. Originally recorded as "skip, Parthenon already supplies it." That was wrong: same name, **different algorithm**. |
| `src/utils/refinement/restriction.hpp` | algorithm | **skip** — verified equivalent. Parthenon's `RestrictAverage` handles `TE::E1/E2/E3` in its dimension guards (`pr_ops.hpp:121-125`), computes the weight generically via `coords.Volume<el>` so edges get edge lengths, and carries the same `tvol > 0.0` guard the donor added (`pr_ops.hpp:160`). |

### Correction (2026-09-23): the two prolongations share a name, not an algorithm

The original entry claimed RIOT's pinned Parthenon "already supplies"
`ProlongateInternalTothAndRoe` (`external/parthenon/src/prolong_restrict/pr_ops.hpp:391`),
so the donor file could be skipped and the framework operator registered instead
(`src/mhd/mhd.cpp:128`). **The names match; the algorithms do not.**

| | Parthenon `ProlongateInternalTothAndRoe` | Donor `ProlongateTothAndRoe` |
| --- | --- | --- |
| Length | ~83 lines (`pr_ops.hpp:391-473`) | ~450 lines (`prolongation.hpp:298`ff) |
| Method | direct formula, cyclic permutation of the x-component | assembles a divergence-constraint matrix `D`, forms the residual and the weighted normal equations `M = D·W·Dᵀ`, and solves with **partial-pivoted Gaussian elimination** on the device |
| Face weighting | none | face-area weighted via `ArtemisUtils::GetFaceAverageWeight` |
| Geometry | Cartesian-shaped | templated over all six donor coordinate systems |

Decisive evidence that this difference is deliberate and was forced by a real
defect: donor commit **`78dbc13`** — *"Add div(B) monitoring. **Fix issues with
div(B) growth during prolongation. This comes at increased complexity.**"* —
rewrote `prolongation.hpp` (+451 lines) and simultaneously wired pre/post-remesh
`max|divB|` prints into `artemis.cpp:217`. A simpler prolongation exhibited div B
growth across remesh and was replaced by the expensive weighted solve. The donor
registers the least-squares version **for Cartesian as well**
(`artemis_utils.cpp:378-387`), not only for curvilinear geometry.

Neither operator has a test. `grep TothAndRoe external/parthenon/tst/` returns
nothing; Parthenon's is exercised only by `example/fine_advection`.

**Why this was harmless until now, and what makes it live:** refinement is
rejected at startup, so the registered operator is never invoked. It becomes
load-bearing the moment AMR is enabled. **Do not treat the name match as
evidence of equivalence when opening Stage 6** — the first Stage 6 measurement
must be whether the operator RIOT currently registers holds div B across a
coarse/fine boundary in Cartesian. See [`DEFERRED_STAGES.md`](DEFERRED_STAGES.md)
Stage 6.

Note also that the donor's face-area weighting exists *for* curvilinear geometry,
so Stage 7 needs the donor's version regardless of how the Cartesian measurement
turns out.

One of the two files the donor had to write is genuinely unnecessary because of
RIOT's newer Parthenon pin (`restriction.hpp`, verified above). The other is a
real deferral, not a redundancy.

## Problem generators

| Donor path | Class | Status | Note |
| --- | --- | --- | --- |
| `src/pgen/orszag_tang.hpp` (new) | test | **reimplement** | → `src/riot_pgen/mhd_orszag_tang.cpp` |
| `src/pgen/field_loop.hpp` (new) | test | **reimplement** | → `src/riot_pgen/mhd_field_loop.cpp`, vector-potential curl |
| `src/pgen/shock.hpp` | test | **reimplement** | Brio–Wu + its fixed-B outflow BC → `src/riot_pgen/mhd_shock_tube.cpp` |
| `src/pgen/linear_wave.hpp` | test | **reimplement** | 7-mode MHD eigensystem → extend existing `src/riot_pgen/linear_modes.cpp`, reusing its `ProblemPackage` L1-error hook |
| `src/pgen/blast.hpp` | test | **defer** | Strongly magnetized blast is a Stage-5 low-β robustness case |
| `src/pgen/{pgen.hpp,params.yaml}` | integration | **reimplement** | RIOT uses the `FOREACH_PROBLEM` macro list |
| `src/pgen/disk.hpp` | unrelated | **skip** | Artemis disk physics. Read for its `etot = max(etot, eint + ke + emag)` floor lesson only. |

## Tests

| Donor path | Class | Status |
| --- | --- | --- |
| `tst/scripts/mhd/{brio_wu,field_loop,linwave,orszag_tang}.py` | test | **adapt** → `tst/scripts/mhd/`, thresholds frozen as-is (they are demonstrated donor performance, not invented) |
| `tst/scripts/mhd/*_mpi.py` | test | **adapt** — 4-rank variants |
| `tst/scripts/mhd/brio_wu.std` | test | **reuse as reference** — Artemis 4096-zone run restricted to 512 |
| `tst/scripts/mhd/athena_bw.std` | test | **reuse as independent oracle** — Athena++ t=0.08; the only genuinely third-party reference available. Provenance and license preserved. |
| `tst/suites/{serial,parallel,parallel_slow}.suite` | build | **reimplement** — RIOT auto-discovers `tst/scripts/<dir>/<name>.py` |
| `inputs/{shock/brio_wu,orszag_tang/orszag_tang,field_loop/field_loop,linwave/linear_wave}.in` | test | **reimplement** — RIOT decks are Python scripts emitting `.rin` |

## Unrelated / read-only

| Donor path | Class | Status | Lesson retained |
| --- | --- | --- | --- |
| `src/drag/drag.hpp` | unrelated | **skip** | Every consumer of the dual-energy helper had to subtract `E_mag` — the generalized lesson is ADR-002's writer audit |
| `src/gas/cooling/beta_cooling.cpp` | unrelated | **skip** | same |
| `src/gas/gas.cpp`, `src/gas/params.yaml` | integration | **adapt** | Solver/MHD compatibility gates and the MHD CFL. RIOT's CFL is already the same directional-sum convention, so only `c_s² → c_f²` changes. |
| `inputs/{diffusion/gaussian_bump,drag/simple_drag}.in` | unrelated | **skip** | incidental input churn |
| `doc/src/physics.rst` | doc | **reimplement** | → `doc/sphinx`, including the energy convention and capability matrix |
| `src/CMakeLists.txt`, `src/artemis_params.yaml` | build | **reimplement** | RIOT `SRC_LIST` + `<physics>` docs |

## Commits whose *final* implementation was audited

Commit subjects are not proof that an issue is solved in RIOT. Each of these was
read in its final form and its behavior re-derived:

| Commit | Subject | Where it lands |
| --- | --- | --- |
| `d78099c` | Gardiner–Stone EMF upwinding; split out `emf.hpp` | ADR-003; the Cartesian EMF kernel |
| `3864a82` | `ExtendMHDFluxBounds` loop-extension helper | Stage 1 ghost budget |
| `e7bd93e` | Skip normal B in reconstruction | Stage 3, via overwrite instead of skip |
| `2047b20` | Face→cell B helper | `FaceToCellB` |
| `205b77c` | Divide B² by ρ in the coordinate source | curvilinear only — deferred |
| `337495b` | Dual-energy fix with MHD | ADR-002 (no RIOT dual energy; subtraction only) |
| `46b99e5` | Solver/MHD compatibility checks | ADR-004 rejections |
| `3e5aeb5` | Add missing dependency (task ordering) | ADR-003 task graph |

## Intentional omissions, summarized

1. **Donor face prolongation/restriction** — superseded by RIOT's Parthenon pin.
2. **Curvilinear geometry and its EMF/momentum metric terms** — Stage 7, and the
   donor's version contains a metric inconsistency to be re-derived, not copied.
3. **Dual-energy / `DualEnergySIE`** — no RIOT analog; constant-γ fallback unsafe
   for arbitrary EOS.
4. **`ScaleMHDFlux`** — Cartesian no-op.
5. **Multi-species (`n ≥ 1`) HLLC branch in HLLD** — single material in scope.
6. **All Artemis-specific physics** (drag, cooling, disk, dust, nbody) — not this
   port.
