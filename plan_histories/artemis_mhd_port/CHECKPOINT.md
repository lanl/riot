# Checkpoint

Updated UTC: 2026-09-18
Riot original base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf`
Riot current integration base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf` (no drift yet)
Donor head / comparison base: `3e5aeb5` / `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee`
Branch: `taitano/mhd-porting`, working tree clean
Commits: `725a19a` (Stages 0–2), `f0af069` (Stage 3 solver)
Current stage: **Stage 3 partially complete.** Gates passed: G0, G1. G2 and G3 partial
(unit level only).

**No MHD simulation has ever been run.** No run has allocated a face field. Nothing
calls the MHD solver yet. Everything below marked "verified" is verified at unit level or
by the hydro-unchanged invariant, never by an MHD simulation.

## Implemented and verified

- **Both RIOT integration hazards fixed** (`src/riot_utils/sparse_update.hpp`).
  `UpdateToNextStage` requires `{WithFluxes, Cell, Independent}`; added
  `DeepCopyFaceData` / `DeepCopyIndependentFaceData` for `TE::F1/F2/F3` with per-element
  bounds. *Evidence*: hydro 7/7 and all three numeric outputs bitwise identical to the
  pre-port baseline. Because both added flags are already universal among affected
  fields, bitwise identity is the evidence that the 19-registration audit was complete.
- **MHD package** (`src/mhd/`), off by default. Face field with genuine face topology
  (deliberately no `CellMemAligned`), `Independent + Conserved + WithFluxes + FillGhost`,
  divergence-preserving refinement ops registered. Its edge flux register is the EMF via
  Parthenon's automatic Face→Edge promotion.
- **Energy contract** (ADR-002) wired at the two choke points in
  `fill_shared_derived.cpp`, through one shared helper so subtraction and re-synthesis
  cannot disagree.
- **Device helpers** (`src/mhd/mhd_helpers.hpp`): `MagneticEnergyDensity`,
  `FastMagnetosonicSpeed` (+overload), `SignalSpeedBound`, `FaceToCellB`,
  `CellMagneticEnergyFromFaces`.
- **MHD HLLE solver** (`src/mhd/riemann_mhd.hpp`), `lr_to_flux_mhd_hlle<DIR>`.
- **Magnetic term in the CFL vote** (`hydro.cpp` `EstimateTimestepMesh`): `B^2/mu0` added
  to the effective bulk modulus. Bitwise the hydro expression when `b2 == 0`.
- **Tests**: `ctest` **34/34** (18 pre-existing + 16 MHD). U01, U01b, U02, U03, U04 and
  the face-to-cell conventions all pass. Details and frozen thresholds in
  `TEST_LEDGER.md`.

## Implemented but NOT verified

- `MHD::SetDerivedMagneticFields`, `MHD::PostInitialization` — compile, wired to
  `PostInitializationMesh`, never executed (no MHD pgen exists).
- The startup rejections (test N01 not run).
- Face-aware stage register copy on a live mesh (U05 — needs a Mesh with a registered
  face field).

## Not implemented

Reconstruction wiring, flux-path wiring, EMF construction, CT face update, task-graph
edges, MHD problem generators, MHD regression tests, restart of face state, HLLD, LLF.

## Current invariants / architecture decisions

- `E = u + ½ρv² + B²/(2μ₀)` when MHD on; unchanged when off. ADR-002.
- One `mhd/mu0` parameter; default `4π` (Gaussian CGS); tests use `1` for exact donor
  parity. ADR-001.
- Face B authoritative; cell B / magnetic energy / div B derived by one owner. EMF is the
  face field's edge flux register. ADR-003.
- Scope: Cartesian, uniform grid, single material, ideal gas, 1T. ADR-004.

## Failing / blocked tests

None. Zero known failures.

## Decisions needed

None outstanding. Scope approved by the user 2026-09-18.

## Next actions — wire the solver into the flux path

This is the largest remaining piece and it is what unlocks every end-to-end test. Work in
`src/hydro/calculate_fluxes.cpp`, mirroring the **`StrengthFluxes` precedent at
`:228-380`**, which is the established pattern for an optional-physics path that allocates
its own scratch, does its own reconstruction, and writes its own flux registers.

1. **Add the solver enum + input option.** `RiemannSolver` enum
   (`src/hydro/riemann.hpp:25`) gains `mhd_hlle` (later `mhd_hlld`, `mhd_llf`). Add these
   to the enumerated list in `hydro.cpp:109-112` so Parthenon rejects out-of-list names.
   Require an MHD solver when `mhd = true` and reject the hydro-only solvers — the
   rejection lines already exist in `riot.cpp`, extend them.
2. **Add a reconstruction type list** for `ccbulk::magnetic_field` alongside
   `set_bulk_recon_types` (`src/hydro/hydro.hpp:44-71`), and reconstruct it via
   `Hydro::ReconCells`.
3. **Substitute the shared normal B.** After reconstruction, overwrite the normal
   component of BOTH L and R states with the face-centered value
   `pv(te, fbulk::magnetic_field(), kji)`. Do NOT reconstruct it twice. This is
   numerically identical to the donor's skip-then-overwrite (DONOR_KERNELS.md §11) and
   avoids threading a `skip_index` through all five RIOT reconstruction methods.
4. **Add an `MHDFluxes<DIR, FLUX_FN>` kernel** modelled on `BulkRiemannFluxes`
   (`:117-161`). Write momentum/energy into the existing `ccbulk` flux registers and the
   induction fluxes into `ccbulk::magnetic_field`'s flux register (which is why that
   field carries `WithFluxes`).
5. **Extend the transverse flux bounds by one layer** when MHD is on. The GS EMF stencil
   reaches one cell back in both transverse directions. Prefer the donor's *directional*
   extension (`ExtendMHDFluxBounds`, DONOR_KERNELS.md §12) via
   `RiotUtils::halo::pm_{i,j,k}_t` so `nghost = 2` stays viable; if the loop abstraction
   forces an isotropic halo, require `nghost >= stencil_width + 2` and document it.
6. **Dispatch** in the `switch (rsolver_tag)` at `:658-694`, keeping the switch outside
   all inner loops as RIOT does.

**Already-resolved design question, do not re-derive:** the Gardiner–Stone EMF upwinding
needs only the *sign* of the face mass flux. Two facts make this free:
`riemann_vel = frho / rho_upwind` with `rho > 0`, so `sign(riemann_vel) == sign(mass
flux)`; and for a single material the `ccmat::rho` flux register **is** the bulk mass
flux and is persistent across tasks (unlike the `riemann_vel` scratch, which is
block-local to `CalculateFluxesImpl`). So Stage 4's EMF task can read
`ccmat::rho`'s flux register directly — **no new field and no extra storage is needed.**

After the wiring, Stage 4: `src/mhd/emf.hpp` (use the Cartesian reduction in
DONOR_KERNELS.md §4b, not the donor's curvilinear form), `MHD::ApplyFaceUpdate`
(§6, signs re-derived in §6a), the five task-graph edges (ADR-003), and the four MHD
problem generators.

Expected result after each step: `ctest` count rises, hydro stays bitwise identical.

## Do not repeat

- **Do not filter the update pack on `Metadata::Conserved`.** Audited: advected scalars
  (`src/scalars/scalars.cpp:62,81`) and level sets (`src/levelsets/levelsets.cpp:41`) are
  updated by `UpdateToNextStage` but carry no `Conserved` flag. That filter would have
  silently stopped updating them.
- **Do not try `FlagCollection` subtraction in the update descriptor.**
  `MakePackDescriptor<any>` takes only a required-flag vector with AND semantics
  (`make_pack_descriptor.hpp:69-75`); there is no exclusion overload for the regex form.
  Topology-based exclusion via required `Metadata::Cell` is the working approach.
- **Do not change `BulkSoundSpeed`.** The MHD solver computes the fast speed internally
  from the acoustic speed it is passed (`bmod = rho*c^2`), so the only fast-speed site
  outside the solver is the CFL vote, which is done. The plan's "three sites" was wrong.
- **Do not port `artemis/src/utils/refinement/{prolongation,restriction}.hpp`.** RIOT's
  newer Parthenon pin already supplies `ProlongateInternalTothAndRoe` and edge-aware
  `RestrictAverage`.
- **Do not use `external/parthenon/example/fine_advection/stokes.hpp`** for the CT curl.
  Its face-variable sign branch carries an upstream `TODO(LFR): This is untested`.
- **Do not transcribe the donor's curvilinear EMF metric factors.** Verified from source:
  `emf.hpp:126-129` pairs `ha_qm`/`ha_pp` scale factors with `mm`/`pm` coordinate
  differences, while `ga_lo_left` correctly pairs `hb_mm` with `mm`. Exact in Cartesian
  (all `h = 1`), O(Δ) wrong otherwise. Re-derive from the Stokes line integral if
  geometry is ever added.
- **Delete the accumulator files before any `--reuse_build` re-run**:
  `tst/build/src/{linwave-errs.dat,linwave_mm-errs.dat,compression.out0.hst}`.
  `linear_modes.cpp:388-399` appends when the file exists, which doubles it and makes
  `analyze()` throw — this looks exactly like a regression and is not one.
- `RIOT_BUILD_CATCH2=OFF` fails on this machine (no system Catch2); keep it `ON`.
- `clang-format-20` lives in the venv:
  `. riot_venv/bin/activate && CFM=$(pwd)/riot_venv/bin/clang-format VERBOSE=1 ./script/format.sh`
- Do not stage `external/parthenon`. It is dirty only from the nested Kokkos
  scratch-limit patch applied by the build from the tracked `riot_kokkos.patch`.
