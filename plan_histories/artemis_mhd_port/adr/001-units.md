# ADR-001 — Magnetic field units and normalization

Status: **accepted** (user-approved 2026-09-18)
Supersedes: nothing. Affects: field registration, all Riemann/CT kernels, I/O,
restart, every regression comparison against the donor.

## Question

What units does the evolved magnetic field carry inside RIOT, what does the user
supply in the input deck, and how is the donor's normalization reconciled for
validation?

## Source evidence

**Artemis uses the SI-style form `E_mag = B²/(2 μ₀)`** with `μ₀` a code-unit
scalar — not Gaussian `B²/8π` and not Heaviside–Lorentz `B²/2`:

```cpp
// artemis/src/mhd/mhd.hpp:21-24
KOKKOS_FORCEINLINE_FUNCTION Real MagneticEnergyDensity(const Real bx, const Real by,
                                                       const Real bz, const Real mu0) {
  return 0.5 * (SQR(bx) + SQR(by) + SQR(bz)) / mu0;
}
```

`mu0_code` is fetched from the package (`artemis/src/mhd/mhd.cpp:41-42`) and
threaded into every kernel; it defaults to `1.0` when MHD is off. Artemis adds a
base unit `artemis/current` and defines
`mu0_code = mu0_physical * current² * time² / (mass * length)`
(`artemis/src/utils/units.cpp:145`), with `mu0_physical = 1.0` in `scalefree`
mode.

**Every Artemis MHD regression test runs scale-free, so `mu0_code == 1`.**
Confirmed independently from the input decks: Orszag–Tang uses
`b0 = 0.28209479177387814 = 1/√(4π)` with `rho0 = 25/(36π)`, `p0 = 5/(12π)`
(`artemis/inputs/orszag_tang/orszag_tang.in`), and the test asserts an initial
total energy containing `½ b0²` — i.e. `μ₀ = 1`
(`artemis/tst/scripts/mhd/orszag_tang.py`). So the donor's field variable is
effectively a normalized `b` with magnetic energy `b²/2`.

**RIOT is CGS throughout.** `doc/sphinx/src/programmer_guide.rst:510`:

> RIOT configures `singularity-opac` in CGS throughout and restricts itself to
> the coefficient calls above to avoid any unit-system mismatch with the
> diffusion module.

RIOT has no `units.cpp` and no unit-conversion layer; EOS and opacity calls are
CGS by construction. There is therefore no existing unit abstraction to extend
and no risk of silently changing RIOT input units.

## Alternatives considered

1. **Normalized `b` internally with explicit physical I/O conversion** (the
   supplied plan's recommendation). Reduces factors in kernels, but requires a
   conversion layer at input/output/restart boundaries that RIOT does not have,
   and adds a units-version field to checkpoints. Rejected as unnecessary
   machinery for a CGS-only code.
2. **Hard-code Gaussian `B²/8π`.** Simplest, but loses the ability to reproduce
   the donor's normalization exactly, which would force every donor comparison
   to go through a conversion — precisely the class of error the validation plan
   warns about.
3. **Chosen: carry one scalar `mhd/mu0` and use Artemis's exact formula
   `E_mag = B²/(2 μ₀)`.**

## Decision

Register a single package parameter `mhd/mu0`, threaded by value into every
device kernel exactly as Artemis threads `mu0_code`.

| Configuration | `mhd/mu0` | Meaning |
| --- | --- | --- |
| **Default** (production, CGS) | `4π` | `B` in Gauss, `E_mag = B²/8π`, `v_A² = B²/(4πρ)` — standard Gaussian CGS, consistent with RIOT's EOS/opacity units |
| **Donor-parity / scale-free** | `1` | Reproduces Artemis's normalized field bit-for-bit; `E_mag = B²/2`, `v_A² = B²/ρ` |

No conversion occurs at input, output, or restart: what the user writes is what
is stored and what is printed. The stored quantity's meaning is fixed entirely by
`mhd/mu0`, which is persisted in the parameter dump.

**Every regression test in `tst/scripts/mhd/` sets `mhd/mu0 = 1`.** This is the
lever that makes direct numerical comparison with the donor possible without any
reconciliation step, and it is why the donor's own thresholds can be adopted
verbatim as frozen acceptance criteria.

## Naming

The supplied plan's warning is honored: **no quantity is named `B`.** RIOT
already uses `bulk_modulus` / `bmod` / `IBL` for the thermodynamic bulk modulus,
and a bare `B` would be ambiguous in exactly the kernels where both appear
(the fast magnetosonic speed takes *both*). Chosen names:

- `face_variables::bulk::magnetic_field` — the evolved normal component on faces
- `ccbulk::magnetic_field` — derived cell-centered vector
- `ccbulk::magnetic_energy` — derived `B²/(2μ₀)`
- `ccbulk::div_magnetic_field` — divergence diagnostic

## Consequences

- Formulas are transcribed from the donor unchanged, including the `1/μ₀` on
  every Maxwell stress term and the `B²/(μ₀ρ)` in the Alfvén speed. This
  minimizes transcription risk, which is the dominant risk in this port.
- Because `μ₀` is a runtime scalar rather than a compile-time `1`, the
  dimensional tests required by the plan (magnetic pressure, Alfvén speed,
  energy) are meaningful: running the same physical state at `μ₀ = 1` and
  `μ₀ = 4π` with correspondingly scaled `B` must give identical dynamics. This
  is a cheap and strong check on missing or doubled `μ₀` factors, and is added
  as unit test U01b.
- A hydro-only run is unaffected: `mhd/mu0` is not read when `mhd = false`.

## What would invalidate this decision

- RIOT growing a general unit-conversion layer, at which point B should join it.
- A requirement to accept SI input (`μ₀ = 4π×10⁻⁷`, B in Tesla) — supported
  mechanically by setting `mhd/mu0`, but the EOS would also need SI, so this is
  a whole-code question, not an MHD question.
- Evidence that any RIOT package infers units from a field name rather than from
  the CGS convention.
