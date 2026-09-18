# ADR-003 — Field topology, metadata, and stage ownership

Status: **accepted**
Affects: variable registration, `sparse_update`, the driver task graph, EMF
construction, communication, refinement metadata.

## Question

How is the magnetic state represented (face / cell / edge), what Parthenon
metadata does each piece carry, who owns it at each RK stage, and does RIOT's
pinned Parthenon support the required primitives without a dependency change?

## Finding 1 — the pinned Parthenon is sufficient. No dependency change.

RIOT pins `external/parthenon` at `928544a6d` (v25.12-639). Artemis pins
`287313a77` (v25.12-193). **Every CT-relevant file is byte-identical between the
two:**

```
$ git diff --stat 287313a77 928544a6d -- src/prolong_restrict/
(no output — identical)
$ git diff --stat 287313a77 928544a6d -- src/mesh/forest/ src/coordinates/
(no output — identical)
```

`src/bvals/comms/bnd_info.cpp` (containing `GetFluxCorrectionElements` and
`CalcIndices`) is likewise unchanged. Available primitives:

| Primitive | Location |
| --- | --- |
| `TopologicalElement {CC, F1..F3, E1..E3, NN}`, `TopologicalType` | `basic_types.hpp:220-230` — the staggering convention matches Athena++/Artemis exactly |
| `IsSubmanifold` (drives Stokes-theorem curls) | `basic_types.hpp:308-329` |
| Face storage: 7-D, `n+1` extent per non-degenerate dim | `interface/metadata.cpp:352-387` |
| **Automatic flux-topology promotion** Cell→Face, **Face→Edge**, Edge→Node | `interface/metadata.cpp:189-197` |
| Area-weighted face restriction / length-weighted edge restriction | `RestrictAverage`, `prolong_restrict/pr_ops.hpp:105-166` |
| Divergence-free internal face prolongation (Tóth & Roe 2002) | `ProlongateInternalTothAndRoe`, `pr_ops.hpp:391-472` |
| Shared-then-internal prolongation ordering | `bvals/comms/boundary_communication.cpp:454-459` |
| Single-valued shared faces via block ownership | `bnd_info.cpp:259-283`, `mesh/forest/block_ownership.cpp:89-140` |
| Component sign flips across reoriented block interfaces | `logical_coordinate_transformation.hpp:55-78` |
| **Edge (EMF) flux correction at coarse/fine**, incl. the shared-edge case | `GetFluxCorrectionElements`, `bnd_info.cpp:86-99` |
| Divergence-preserving AMR remesh (shared-from-neighbor, then internal) | `mesh/mesh-amr_loadbalance.cpp:953-988` |
| Face/edge + cell co-packing in one `SparsePack` | `sparse_pack.hpp:333-340`, `sparse_pack_base.cpp:250-300` |
| Working CT+AMR reference (`numlevel = 3`) | `example/fine_advection/` |

RIOT's version is additionally **strictly better** for CT: it carries
`90dbea5a4 Fix race condition in flux corr comm with sparse=off (#1405)`, a
flux-correction bugfix absent from the donor's pin.

Two caveats recorded:

- `example/fine_advection/stokes.hpp:174-183` carries an upstream
  `TODO(LFR): This is untested` on precisely the face-variable (curl) sign
  branch. **The port does not use it.** The CT update is written directly from
  the donor's kernel and its signs are independently re-derived and unit-tested
  (U04).
- `Metadata::RegisterRefinementOps` propagates ops to `flux_metadata`
  (`metadata.hpp:567-582`), so registering `ProlongateInternalTothAndRoe` on the
  face field also registers it on the edge flux. Harmless — that operator's
  `OperationRequired` returns false for non-Face elements — but noted so a future
  custom operator includes the same guard.

## Finding 2 — two RIOT integration hazards, verified by reading the code

RIOT has **no genuine face field today.** All six existing "face" fields
(`ccbulk::face_signal`, `mat::diffusive_fluxes`, `RadiationDiffusion::{Fgroup,
D, kappa_face, face_area, DeltaX}`, `Ionization::D`) carry
`Metadata::CellMemAligned`, i.e. cell-aligned layout, never ghost-exchanged. The
correctness rationale is documented at `src/mix/mix.cpp:148-155`, and
`src/riot_driver.cpp:319-322` explicitly acknowledges the gap:

```cpp
// NOTE(@pdmullen): This task goes away if we permit Metadata::FillGhost for
// Metadata::CellMemAligned face fields
```

The CT field is RIOT's first true face field and must **not** carry
`CellMemAligned`. Two existing routines assume cell topology and would silently
corrupt it:

**Hazard A — the RK update sweeps face fields into a cell-centered kernel.**
`sparse_update::UpdateToNextStage`, `src/riot_utils/sparse_update.hpp:145,154`:

```cpp
std::vector<MetadataFlag> flags({Metadata::WithFluxes});
auto desc = parthenon::MakePackDescriptor<any>(umd, flags, {PDOpt::WithFluxes});
...
auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, umd, TE::CC);
```

then per variable applies `FaceArea`/`CellVolume` (`:192-203`). A face
`Independent + WithFluxes` field matches this filter and would be updated with
the wrong index space, the wrong geometric measure, and none of the curl sign
structure.

**Fix — corrected after a full audit of every `Metadata::WithFluxes`
registration in `src/`.** The audit invalidated two assumptions made while
planning:

1. **Not every updated field is `Conserved`.** `src/scalars/scalars.cpp:62,81`
   (`{Cell, Independent, Intensive, FillGhost, Advected, WithFluxes}`) and
   `src/levelsets/levelsets.cpp:41` carry no `Metadata::Conserved`. Requiring
   `Conserved` would have silently stopped updating advected scalars and level
   sets — a far worse regression than the bug being fixed.
2. **`MakePackDescriptor<any>` takes only a required-flag vector** with AND
   semantics (`external/parthenon/src/pack/sparse_pack/make_pack_descriptor.hpp:69-75`).
   There is no `FlagCollection` overload for the regex/`any` form, so
   set subtraction cannot be expressed there at all.

The audit did establish two properties that hold across all 19 `WithFluxes`
registrations: every one carries **`Metadata::Cell`**, and every one carries
**`Metadata::Independent`**. That yields a minimal fix expressible in the
existing API — add both to the required flags:

```cpp
std::vector<MetadataFlag> flags({Metadata::WithFluxes, Metadata::Cell,
                                 Metadata::Independent});
```

- `Metadata::Cell` excludes the face-centered CT field (which is `Face`, not
  `Cell`) — the topology-based exclusion this hazard requires.
- `Metadata::Independent` excludes the `Derived + WithFluxes` cell fields that
  carry the MHD scratch flux registers (`ccbulk::magnetic_field`,
  `ccbulk::magnetic_energy`), which must not receive a flux divergence.

One pre-existing `Derived + WithFluxes` field exists —
`Ionization::delta` (`src/ionization/ionization.cpp:190`) — but it is also
`OperatorSplit`, and `RiotUtils::GetUnsplitVarNames`
(`src/riot_utils/riot_utils.cpp:90-100`) excludes `OperatorSplit` from the
`u0`/`u1` registers, so it never reaches this kernel. Adding `Independent`
therefore changes nothing today.

Because both added flags are already universal among the affected fields, this
fix is a no-op for hydro-only runs. **That is exactly what makes it verifiable:
the hydro baseline must come back bitwise identical**, and any difference means
the audit missed a field.

**Hazard B — the stage register copy truncates face storage.**
`sparse_update::DeepCopyData`, `src/riot_utils/sparse_update.hpp:245`:

```cpp
auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, from, TE::CC);
```

with `make_var_view` defaulting to `TE::CC`
(`external/parthenon/src/loop_abstraction/pack_view.hpp:334-336`). Face storage
has one extra element in its normal direction, so `u1 ← u0` would silently drop
the last face plane. **Fix:** add `sparse_update::DeepCopyFaceData` looping
`TE::F1/F2/F3` index spaces with the 4-arg
`make_var_view(range, pack, te, var)` overload (`pack_view.hpp:308-312`).
Covered by unit test U05, which writes distinct values into each stage register
and asserts the full face extent round-trips.

## Decision — representation

| Quantity | Storage | Metadata | Lifetime rule |
| --- | --- | --- | --- |
| Normal `B` on faces | `face_variables::bulk::magnetic_field` — **scalar** per face variable; the three components live in the `F1/F2/F3` slots | `{Face, Independent, Conserved, WithFluxes, FillGhost}` + `RegisterRefinementOps<ProlongateSharedMinMod, RestrictAverage, ProlongateInternalTothAndRoe>()`. **No `CellMemAligned`.** | The authoritative magnetic state. Stage-aware, restartable (via `Independent`), communicated. CT is the sole update authority. |
| Edge EMF | **No separate variable** — it *is* the flux register of the face field, at `TE::E1/E2/E3` | automatic (`metadata.cpp:189-197`) | Stage scratch. Built from current-stage Riemann output; consumed by CT; no authority across restarts. Gets coarse/fine correction for free. |
| Cell-centered `B` | `ccbulk::magnetic_field` (3 components) | `{Cell, Derived, Intensive, OneCopy, FillGhost, WithFluxes, Vector}` | Derived from faces by one shared helper. Its flux slots hold the transverse induction fluxes from the Riemann solve. Never independently advanced. |
| Face magnetic pressure | flux register of `ccbulk::magnetic_energy` | `{Cell, Derived, Intensive, OneCopy, FillGhost, WithFluxes}` | Stage scratch; applied as a gradient alongside gas pressure. |
| `B²/(2μ₀)` | `ccbulk::magnetic_energy` | as above | Derived. Not a second conserved energy register. |
| Discrete `∇·B` | `ccbulk::div_magnetic_field` | `{Cell, Derived, OneCopy}` | Diagnostic only. Computed from **face flux balance**, never from centered differences of cell-centered `B`. |

Following the donor, the face field is a scalar-per-element rather than a
3-vector; RIOT's existing `VARIABLE_FACE` macro (`src/variables.hpp:68`) already
produces exactly this shape, so no new macro is needed. No `VARIABLE_EDGE` macro
is needed either, since the EMF is a flux register.

**Degenerate dimensions are handled explicitly, not by dropping components.** In
1D the transverse components still evolve; in 2D the out-of-plane component still
evolves. `nx2 == 1` or `nx3 == 1` never justifies omitting a physically nonzero
field component. The face-to-cell helper degenerates to copying the single face
value in a collapsed direction, matching `artemis/src/mhd/mhd.hpp:55-78`.

## Decision — stage ownership and the task graph

Dependency invariants, realized in `src/riot_driver.cpp`:

| Task | Depends on | Rationale | Site |
| --- | --- | --- | --- |
| `MHD::DeepCopyFaceState` (`u1 ← u0`) | none | face-aware register copy (Hazard B) | `:186`, beside `DeepCopyIndependentData` |
| `MHD::AssembleEdgeEMF` | `hydro_flx` | needs the transverse induction fluxes **and** the mass fluxes, unmodified while read | after `:217` |
| `LoadAndSendFluxCorrections` | **+= `mhd_emf`** | EMFs must exist before flux correction is sent, or the edge registers ship stale | `:240` |
| `MHD::ApplyFaceUpdate` (CT) | `mhd_emf \| set_flx` | one consistent edge line integral at shared interfaces | `:313`, beside `update` |
| `PreCommFillDerived` | **+= `update_mhd`** | no stale magnetic-energy subtraction in thermal recovery | `:361` |

Face `B` is `FillGhost`, so it rides the existing
`AddBoundaryExchangeTasks` at `:366` — **no new communication task.** The cell
and magnetic updates run concurrently; they are joined before thermal recovery.
No serial chain is imposed that would create false dependencies.

The CT update applies the integrator's own coefficients, identically to
`UpdateToNextStage`:

```
Φ_f ← gam0·Φ_f + gam1·Φ_f¹ − β∆t · Σ_{e∈∂f} s_fe L_e ε_e ,   Φ_f = A_f B_{n,f}
```

with signed face–edge orientation `s_fe`. Shared-edge contributions cancel
algebraically, which is what makes `∇·B` preserved to roundoff rather than merely
small.

### The one deliberate deviation from the donor

Artemis's `UpwindEMF` (`artemis/src/mhd/emf.hpp:49-143`) is written for general
curvilinear coordinates. Because this scope is Cartesian, all edge and face scale
factors are unity and the kernel collapses to the clean Gardiner & Stone (2005)
Eq. 51 form. The port writes that form directly.

This is not merely a simplification: the donor's curvilinear version pairs
`ha_qm` with `mm`-based coordinate differences in `gb_lo_left`, and mixes `ha_pp`
with `pm`/`pp` differences in `gb_lo_right`/`gb_hi_right`
(`artemis/src/mhd/emf.hpp:126-129`). In Cartesian all `h = 1` so it is exact, but
in curvilinear it is an O(Δ) inconsistency. Writing the Cartesian form directly
avoids importing a latent defect that would only surface in Stage 7. **Flagged
here so that any future curvilinear extension re-derives the metric factors
rather than transcribing them.**

The EMF sign convention is re-derived independently and unit-tested rather than
assumed:

```
E = −(v×B),   ∂ₜB = −∇×E
Fx(By) = −Ez    Fy(Bx) = +Ez
Fx(Bz) = +Ey    Fz(Bx) = −Ey
Fy(Bz) = −Ex    Fz(By) = +Ex
```

### Ghost-cell budget

The Gardiner–Stone stencil reaches one cell back in **both** transverse
directions, so face fluxes need one extra transverse layer. Artemis handles this
with a directional extension (`ExtendMHDFluxBounds`,
`artemis/src/utils/fluxes/fluid_fluxes.hpp:75-94`) which keeps `nghost = 2`
viable. The port prefers the same directional approach via
`RiotUtils::halo::pm_{i,j,k}_t`; if the loop abstraction forces an isotropic
halo, `nghost >= stencil_width + 2` is required and documented instead.

## Reuse note

`RiotUtils::DirBasis` / `MakeDirBasis<DIR>`
(`src/riot_utils/riot_loops.hpp:64-95`) already provides the cyclic
right-handed `(normal, transverse-a, transverse-b)` permutation, and its own
comment states it exists for "curl/EMF signs for CT-MHD." It is used rather than
re-deriving index permutations.

## What would invalidate this decision

- A Parthenon bump that changes face/edge storage layout, ownership semantics, or
  the flux-topology promotion rule.
- Enabling AMR (out of scope here): the metadata is already correct, but
  `ProlongateInternalTothAndRoe` and the edge correction must then be *validated*,
  not merely registered. Refinement metadata being present is not evidence that
  refinement works.
- Any future field that needs a multi-component face variable, which would
  require a new `VARIABLE_FACE_VECTOR` macro threading `NCOMP` through
  `base_w_tt_t`.
