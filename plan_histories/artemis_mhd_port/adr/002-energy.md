# ADR-002 — Conserved-energy ownership and magnetic-energy accounting

Status: **accepted** (user-approved 2026-09-18)
Affects: `ccbulk::total_material_energy` semantics, thermal recovery, every
package that writes energy, output/restart interpretation, analysis scripts.

This is the highest-risk decision in the port. Get it wrong and the symptom is a
slow energy drift or spurious heating that no divergence test detects.

## Question

RIOT's evolved bulk energy currently excludes magnetic energy. Artemis's
includes it. Which does the port adopt, and where exactly does the conversion
live?

## Source evidence

**RIOT's `ccbulk::total_material_energy` is total = internal + kinetic, with no
magnetic term.** Established by reading both directions of the conversion, not
inferred from documentation:

Thermal recovery — `src/multiphysics/fill_shared_derived.cpp:99-104`
(`FillInteriorDerived`, `IndexDomain::interior`):

```cpp
pv(ccbulk::internal_energy(), kji) =
    pv(ccbulk::total_material_energy(), kji) -
    0.5 * pv(ccbulk::rho(), kji) *
        (SQR(pv(ccbulk::velocity(0), kji)) + SQR(pv(ccbulk::velocity(1), kji)) +
         SQR(pv(ccbulk::velocity(2), kji)));
```

Re-synthesis — `src/multiphysics/fill_shared_derived.cpp:325-335`
(`PostCommsFillDerived`, `IndexDomain::entire`):

```cpp
const Real vsq = SQR(pv(ccbulk::velocity(0), kji)) + ... ;
pv(ccbulk::total_material_energy(), kji) =
    pv(ccbulk::internal_energy(), kji) + 0.5 * rho_val * vsq;
```

Corroborated by `doc/sphinx/src/packages/hydro.rst:285-287`
(`E = u + ½ρ|v|²`) and by its sole registration at `src/hydro/hydro.cpp:157`
(`{Cell, Independent, Intensive, Conserved, WithFluxes}`, associated with
`ccbulk::internal_energy`).

**Artemis's total energy includes magnetic energy**
(`artemis/src/derived/fill_derived.cpp:429-437`), and its dual-energy recovery
subtracts it before forming thermal energy
(`artemis/src/utils/artemis_utils.hpp:60-80`, commit `337495b`).

**Writer audit of `ccbulk::total_material_energy`** — the decisive input. Three
distinct classes:

| Class | Behavior under a redefinition | Sites |
| --- | --- | --- |
| **Flux writers** (`=` into the flux register) | Safe — the MHD flux is supplied by the MHD solver | `hydro/calculate_fluxes.cpp:156, 208, 351` |
| **Incremental writers** (`+=` / `-=` a *delta*) | **Safe by construction** — adding a thermal delta to a total that also contains magnetic energy is still correct | `mix/mix.cpp:538, 571, 685, 702`; `ionization/ionization.cpp:1112, 1148, 1186, 1492, 1542, 1593`; `gravity/gravity.cpp:87`; `tnburn/{shared_sources.cpp:68,91, tnburn.cpp:392,458}`; `laser/laser.cpp:462, 474, 522, 707, 711, 755, 781`; `strength/strength.cpp:198, 295`; `radiation_diffusion/multigroup_diffusion-tasks.cpp:331, 353`; `prescribed_sources/prescribed_sources.cpp:186, 212` |
| **Absolute reconstructors** (rebuild `E` from scratch) | **Must be updated** | `fill_shared_derived.cpp:325-335`; ~16 pgens in `src/riot_pgen/`; `diagnostics/energies.cpp:71, 85, 105` (read-only) |

The critical finding: **the large majority of writers are incremental and need
no change at all.** Only one absolute reconstructor exists in the physics path
(`fill_shared_derived.cpp:325-335`); the rest are problem generators, and every
one of them runs only in hydro-only configurations because MHD requires its own
pgens. `microphysics/pte_closure_general.cpp:81` packs the field but never
dereferences it — a dead pack entry, so the closure is unaffected.

This inverts the supplied plan's concern that "too many external/plugin
contracts depend on the existing field's material-only meaning." They do not.

## Alternatives considered

1. **Separate magnetic-energy register**, leaving `total_material_energy`
   material-only. Less invasive to readers, but the fluid energy equation becomes
   non-conservative — magnetic work reappears as a source term
   `∂ₜE_mat + ∇·[(E_mat+p)v] = −v·(J×B)`. Total energy would then be conserved
   only to truncation error. The donor's own acceptance gates measure exactly
   this: Orszag–Tang requires `⟨E⟩` stable to `rtol = 1e-8` and an energy
   decomposition residual `|E − u − KE − E_mag| ≤ 1e-10`. Those gates would be
   unreachable. **Rejected.**
2. **A new authoritative conserved-MHD-energy field**, deriving material-only
   energy from it. Avoids reinterpreting an existing field, but doubles the
   energy bookkeeping surface and requires mediating every source exchange
   through a new indirection — for no benefit, given the writer audit above.
   **Rejected.**
3. **Chosen: extend the meaning of `ccbulk::total_material_energy` to include
   magnetic energy when and only when MHD is enabled.**

## Decision

```
MHD OFF:  E = u + ½ρv²                    (unchanged, bitwise)
MHD ON:   E = u + ½ρv² + B²/(2μ₀)
```

Exactly one conservative update transports `E`. The MHD energy flux is the
donor's, transcribed unchanged:

```
F_E = (E + p + B²/2μ₀) v_n − B_n (v·B)/μ₀
```

### The three cell-centered touch points

1. **`fill_shared_derived.cpp:99-104`** (`FillInteriorDerived`, `interior`) —
   subtract `B²/(2μ₀)` when recovering `ccbulk::internal_energy`. Interior face
   values are valid here because the CT face update precedes
   `PreCommFillDerived` in the task graph (see ADR-003).
2. **`fill_shared_derived.cpp:325-335`** (`PostCommsFillDerived`, `entire`) —
   add `B²/(2μ₀)` back when re-synthesizing `E`. This runs *after*
   `AddBoundaryExchangeTasks`, so face ghost values are valid on `entire`.
3. **`MHD::SetDerivedMagneticFields`** — the single owner of
   `ccbulk::magnetic_field`, `magnetic_energy`, and `div_magnetic_field`,
   computed from face state via one shared face-to-cell helper.

### Invariants

- `E = thermal + kinetic + magnetic` must hold after initialization, after every
  RK stage, after every operator-split source, after restart. This does **not**
  license overwriting a conservatively-transported `E` with an independently
  computed sum. Rebuilding happens only at touch point 2, which is where RIOT
  already rebuilds it.
- **Magnetic energy must never reach an EOS call as heat.** This holds by
  construction: both closures consume `ccbulk::internal_energy`
  (`pte_closure_general.cpp:166`, `pte_closure_ideal.cpp:93-105`), which is the
  post-subtraction thermal quantity. No closure reads
  `total_material_energy` — the one pack entry that names it is never
  dereferenced.
- **No source routine may erase magnetic energy when assembling a new total.**
  Enforced by the writer audit: every source is incremental. Any future absolute
  reconstructor is a defect.
- No cell-centered Lorentz-force or magnetic-work source is added on top of the
  fluxes. Magnetic pressure and tension live in the momentum flux; Poynting
  transport lives in the energy flux.

### Low-beta thermal recovery

Deferred, deliberately. RIOT has no dual-energy or entropy fallback, and
Artemis's is a constant-γ construct that must not be transplanted into RIOT's
arbitrary-EOS path (the supplied plan is explicit on this, and correct).
Behavior for this scope: recover thermal energy by subtraction, and **count and
report** every floor activation. Routine floor use is a failed test, not a
numerical-success criterion. If low-beta cases prove to need a repair policy,
that is a separate reviewed decision with its own ledger — not a silent fix.

## Consequences

- The output/restart meaning of `total_material_energy` changes when MHD is on.
  Mitigations: documented in `doc/sphinx`, stated in the capability matrix,
  printed at startup, and `ccbulk::magnetic_energy` is written separately so
  analysis scripts can decompose rather than guess. Scripts must **not** add
  `B²/2μ₀` to a total that already contains it.
- `Metadata::Associate(ccbulk::internal_energy::name())` on the energy field
  (`hydro.cpp:156`) remains correct — the associated variable is still thermal.
- Verification that this decision was implemented correctly is the Orszag–Tang
  energy-decomposition residual (`≤ 1e-10`) plus the U01 prim→cons→prim round
  trip (`≤ 1e-12`) over a range of velocity and field orientations. A doubled or
  dropped `B²/2μ₀` fails both immediately.

## What would invalidate this decision

- Discovery of an absolute energy reconstructor in a package that can co-run
  with MHD (none exists in the certified matrix, since all of strength, mix, tn,
  ionization, radiation, lasers, and prescribed sources are rejected with MHD in
  this scope). **Each must be re-audited before it is enabled**, which is
  Stage 9 work and out of scope here.
- A plugin (`src/plugins.hpp`) that rebuilds `E`. Plugins can only add `dudt`
  sources, which are incremental, so this is currently impossible by
  construction — but the constraint should be documented for plugin authors.
