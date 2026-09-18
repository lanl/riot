# Checkpoint

Updated UTC: 2026-09-18
Riot original base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf`
Riot current integration base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf` (no drift yet)
Donor head / comparison base: `3e5aeb5` / `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee`
Port HEAD: uncommitted working tree on branch `taitano/mhd-porting`
Current stage: **Stage 2 complete. Last passed gate: G1. G2 partial (unit level).**

Working tree (formatted with clang-format 20.1.8, copyright headers inserted):

```
 M src/CMakeLists.txt                        (+4)
 M src/multiphysics/fill_shared_derived.cpp   (+46)
 M src/riot.cpp                               (+81)
 M src/riot_utils/sparse_update.hpp          (+112)
 M src/variables.hpp                          (+33)
 M tst/unit/CMakeLists.txt                     (+1)
?? src/mhd/{mhd.cpp,mhd.hpp,mhd_helpers.hpp}
?? tst/unit/test_mhd.cpp
?? plan_histories/artemis_mhd_port/
```
268 insertions, 9 deletions across tracked files.

## Implemented and verified

- **Both RIOT integration hazards fixed** (`src/riot_utils/sparse_update.hpp`).
  `UpdateToNextStage` now requires `{WithFluxes, Cell, Independent}`, so a
  face-centered evolved field cannot be fed to the cell-centered flux-divergence
  kernel and a `Derived` scratch-flux carrier cannot receive a divergence. Added
  `DeepCopyFaceData` / `DeepCopyIndependentFaceData`, which copy `TE::F1/F2/F3`
  with per-element bounds instead of truncating face storage.
  *Evidence*: hydro suite 7/7, and `linwave-errs.dat`, `linwave_mm-errs.dat`,
  `compression.out0.hst` all **bitwise identical** to the pre-change baseline
  (MD5s in `TEST_LEDGER.md` G0.2/G1.2). Since both added flags are universal
  among affected fields, bitwise identity is the evidence that the 19-registration
  audit was complete.
- **MHD package shell** (`src/mhd/`), disabled by default. Face field registered
  with genuine face topology (no `CellMemAligned`), `Independent + Conserved +
  WithFluxes + FillGhost`, plus divergence-preserving refinement operators. Its
  edge flux register is the EMF, obtained automatically via Parthenon's
  Face→Edge flux promotion.
- **Startup rejection of every unsupported combination** (`src/riot.cpp`):
  multi-material, non-Cartesian, refinement, general PTE, fixed fluid, and all 13
  other physics packages.
  *Not yet verified* — N01 has not been run.
- **Device helpers** (`src/mhd/mhd_helpers.hpp`): `MagneticEnergyDensity`,
  `FastMagnetosonicSpeed` (+ overload), `SignalSpeedBound`, `FaceToCellB`,
  `CellMagneticEnergyFromFaces`.
  *Evidence*: 10 new unit tests, ctest 28/28.
- **Energy conversion wired** (ADR-002), at exactly the two choke points:
  `fill_shared_derived.cpp` subtracts `B²/2μ₀` in `FillInteriorDerived` and adds
  it back in `PostCommsFillDerived`, through the *same* helper so the two cannot
  disagree.
  *Evidence*: hydro remains bitwise identical with MHD off (`emag` is exactly
  `0.0` on that path); U01 round trip passes at ≤1e-12 including a low-beta state.
- **Gate G2 unit-level contracts** all pass: units in both conventions, unit
  invariance, fast-speed quartic residual, degenerate orientations, face-to-cell
  conventions incl. collapsed directions, and the discrete curl/divergence
  identity at η ≤ 1e-13.

## Implemented but not verified

- `MHD::SetDerivedMagneticFields` and `MHD::PostInitialization` — compile and are
  wired to `PostInitializationMesh`, but nothing exercises them: no MHD problem
  generator exists yet, so no run has ever allocated a face field.
- The startup rejections (N01).
- Face-aware stage register copy on a live mesh (U05).

## Current invariants / architecture decisions

- `E = u + ½ρv² + B²/(2μ₀)` when MHD is on; unchanged when off. ADR-002.
- One `mhd/mu0` parameter; default `4π` (Gaussian CGS), tests use `1` for exact
  donor parity. ADR-001.
- Face B is authoritative; cell B, magnetic energy, and div B are derived by one
  owner. EMF is the face field's edge flux register. ADR-003.
- Certified scope: Cartesian, uniform grid, single material, ideal gas, 1T.
  ADR-004.

## Failing / blocked tests

None. Zero known failures.

One **operational trap** recorded so it is not misdiagnosed as a regression:
`src/riot_pgen/linear_modes.cpp:388-399` appends to `<problem_id>-errs.dat`, and
`adiabatic_compression`'s `.hst` behaves the same way. **Delete
`tst/build/src/{linwave-errs.dat,linwave_mm-errs.dat,compression.out0.hst}` before
any `--reuse_build` re-run**, or `analyze()` throws on a doubled file.

## Decisions needed

None outstanding. The three scope decisions were approved by the user on
2026-09-18 (energy convention, Stages 0–5 scope, single-material restriction).

Future escalation points, per ADR-004: any change to the physical closure, the
energy-repair policy, or a numerical acceptance threshold; any Parthenon pin
change (not currently expected); and any claim of multi-material, 2T, AMR, or
curvilinear support.

## Next actions

1. **Stage 3 — MHD Riemann solvers.** Create `src/mhd/riemann_mhd.hpp` with MHD
   HLLE in RIOT's `template <int DIR> KOKKOS_FORCEINLINE_FUNCTION` free-function
   style, following the `StrengthFluxes` precedent
   (`src/hydro/calculate_fluxes.cpp:228-380`) for an extended-signature
   optional-physics path. Preserve the donor's structure: magnetic pressure to a
   separate face register (not folded into the normal-momentum flux), normal-B
   flux slot written exactly `0.0`, and retain the mass flux for the EMF
   upwinding. Add U02 (equal-state flux consistency, all component permutations)
   to `tst/unit/test_mhd.cpp`.
2. Insert the fast speed at `BulkSoundSpeed` (`calculate_fluxes.cpp:85`) and
   `EstimateTimestepMesh` (`hydro.cpp:717`). RIOT's CFL is already the
   directional-sum convention, so only `c_s² → c_f²` changes.
3. Add `set_bulk_mhd_recon_types` and the shared normal-B substitution.
4. Then Stage 4: `src/mhd/emf.hpp`, `ApplyFaceUpdate`, the five task-graph edges,
   and the four MHD problem generators (which is what finally makes G2's
   mesh-level items and G3/G4 runnable).

Expected result after step 1: ctest count rises by the U02 cases; hydro stays
bitwise identical.

## Do not repeat

- **Do not filter the update pack on `Metadata::Conserved`.** Audited: advected
  scalars (`src/scalars/scalars.cpp:62,81`) and level sets
  (`src/levelsets/levelsets.cpp:41`) are updated by `UpdateToNextStage` but carry
  no `Conserved` flag. That filter would have silently stopped updating them.
- **Do not try `FlagCollection` subtraction in the update descriptor.**
  `MakePackDescriptor<any>` accepts only a required-flag vector with AND
  semantics (`make_pack_descriptor.hpp:69-75`); there is no exclusion overload
  for the regex form. Topology-based exclusion via required `Metadata::Cell` is
  the working approach.
- **Do not port `artemis/src/utils/refinement/{prolongation,restriction}.hpp`.**
  RIOT's newer Parthenon pin already provides `ProlongateInternalTothAndRoe` and
  edge-aware `RestrictAverage`. Register the framework operators instead.
- **Do not use `external/parthenon/example/fine_advection/stokes.hpp`** for the CT
  curl. Its face-variable sign branch carries an upstream
  `TODO(LFR): This is untested`.
- **Do not transcribe the donor's curvilinear EMF metric factors.** ADR-003
  records an O(Δ) inconsistency in its `gb_*` scale-factor pairing
  (`artemis/src/mhd/emf.hpp:126-129`); harmless in Cartesian, wrong in
  curvilinear. Re-derive if geometry is ever added.
- `RIOT_BUILD_CATCH2=OFF` fails on this machine (no system Catch2); keep it `ON`.
- `clang-format-20` is not installed system-wide. It is now in `riot_venv`
  (`clang-format==20.1.8`); invoke as
  `CFM=$(pwd)/riot_venv/bin/clang-format VERBOSE=1 ./script/format.sh`.
