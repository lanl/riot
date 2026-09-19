# Checkpoint

Updated UTC: 2026-09-18
Riot original base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf`
Donor head / comparison base: `3e5aeb5` / `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee`
Branch: `taitano/mhd-porting`
Commits: `725a19a` (Stages 0-2), `f0af069` (Stage 3 solver), `4d1a8e0` (checkpoint),
`1350c20` (face memory-layout fix), `dced139` (flux wiring + CT + pgens)

Current stage: **Stage 4 complete, Stage 5 in progress (HLLD, reconstruction, N01 done).
Gates G0, G1, G2, G3 and G4 all pass.** Brio-Wu agrees
with the Athena++ reference; the 2D and 3D field loops hold div B at roundoff; Orszag-Tang
holds it through a shock network; the CPAW converges at second order with error identical
across all three axes; and every result is bitwise independent of rank count. Remaining known
gaps are listed under "Not implemented" — none of them blocks G4.

## Implemented and verified

- **Both RIOT integration hazards fixed** (`src/riot_utils/sparse_update.hpp`):
  `UpdateToNextStage` requires `{WithFluxes, Cell, Independent}`; `DeepCopyFaceData` /
  `DeepCopyIndependentFaceData` handle the face stage register.
- **MHD package** (`src/mhd/`), off by default. Genuine face topology (deliberately no
  `CellMemAligned`); its edge flux register is the EMF via Parthenon's Face->Edge flux
  promotion.
- **Energy contract** (ADR-002) at the two choke points in `fill_shared_derived.cpp`, plus
  `MHD::AddMagneticEnergyToTotal` called from every MHD pgen (see the trap below).
- **MHD HLLE solver** (`src/mhd/riemann_mhd.hpp`) wired into `calculate_fluxes.cpp` via
  `MHDFluxes<DIR, FLUX_FN>` and `MHDFluxIndexSpace` (transverse-only bound extension).
- **Upwind CT** (`src/mhd/emf.hpp`, `MHD::AssembleEdgeEMF`, `MHD::ApplyFaceUpdate`) with
  the five task-graph edges in `riot_driver.cpp`.
- **Three problem generators**: `mhd_shock_tube` (Brio-Wu), `mhd_field_loop` (2D and 3D, with
  a `loop_axis` option that rotates the setup onto each pair of edge directions), and
  `mhd_orszag_tang`. Inputs in `inputs/mhd/`.
- **Orszag-Tang verified** (TEST_LEDGER G4.4): the only case in the port combining shocks
  with multi-D CT. div B at roundoff, mean rho/E conserved, the ADR-002 energy split exact to
  1e-16, positivity with margin (no floor), and 1 vs 4 ranks bitwise identical. Runs
  `mhd_hlle` rather than the donor's HLLD, which is why its criteria are conservation
  identities rather than a pointwise reference.
- **Order of accuracy verified** (TEST_LEDGER G4.5): `mhd_cpaw`, an exact NONLINEAR solution
  and so an analytic oracle rather than a donor one. Observed L1 order rises 1.55 -> 1.73 ->
  1.83 toward 2 over N = 16..128, with div B and parallel-field drift identically zero. The
  error is identical to every printed digit across all three wave axes, i.e. exact rotational
  isotropy. Axis-aligned only; oblique propagation is not covered.
- **3D CT verified** (TEST_LEDGER G4.3): the Gardiner-Stone axial-field test passes in all
  three permutations with the axial field at roundoff, and 1-block vs 8-block results are
  bitwise identical.
- **MPI verified** (TEST_LEDGER P01, complete): results are **bitwise identical** to serial at
  1/2/3/4/5/8 ranks, for all three loop axes, in 2D (collapsed EMF branch) and 3D (upwind
  branch), and with shocks (Brio-Wu on 4 blocks at 4 ranks). `-n 3`/`-n 5` were included
  deliberately so block ownership does not divide evenly. `dt` agreed to all printed digits at
  every rank count, which is why bitwise agreement was attainable here.
- **Tests**: `ctest` **34/34**; hydro regression 7/7; hydro output **bitwise identical to
  the pre-port commit**, established by a build-config-held-fixed A/B against `193b3fa`
  rather than by a stored MD5 (see TEST_LEDGER H01 re-run for why the stored hash is not a
  valid cross-session invariant). Numeric results in `TEST_LEDGER.md`.

- **HLLD** (`MHD::lr_to_flux_mhd_hlld`, TEST_LEDGER G5.1-G5.4): all donor degeneracy guards
  verbatim, ideal gas only. Beats HLLE on every Brio-Wu field against the Athena++ reference
  (ratios 0.65-0.80), corroborated by higher field-loop energy retention and less Orszag-Tang
  shock heating. `ctest` 41/41.
- **All five reconstruction modes certified with MHD** (TEST_LEDGER G5.5): CONSTANT, PLM,
  PPM4, WENO5, MP5. div B at roundoff and the axial field identically zero for every mode;
  `nghost >= stencil_width + 1` verified sufficient by a bitwise min-vs-min+1 comparison and
  verified to abort with an actionable message below that. Measured spatial order (with the
  temporal term suppressed) 0.9 / 1.8 / 2.1 / 4.2 / 4.0. **Recorded limitation: WENO5 and MP5
  cannot reach their formal order with `rk2` at fixed CFL — the O(dt^2) temporal term floors
  them.** Quantified: 8x CFL cut gives 63x error reduction, and HLLD reproduces HLLE's numbers
  in every digit, so the floor is the integrator, not the solver or the reconstruction.

- **Restart verified** (TEST_LEDGER G5.6): face B restarts **bitwise**; the evolved restart
  agrees to rel <= 1e-14 at both the same and a changed rank count, with the 4-rank differences
  bit-for-bit identical to the 1-rank ones. Backed by a reproducibility control (same run twice
  = bitwise) and a hydro-only control. Found and fixed two real defects: derived magnetic state
  was never rebuilt on restart (3.7e-06 permanent trajectory error, because the transverse field
  fed to the first stage's Riemann solve was zero), and restarting across an MHD/hydro change ran
  silently in both directions (MHD checkpoint into a hydro run reinterprets B^2/2mu0 as heat: an
  80% pressure error on Brio-Wu at exit code 0). Both directions are now rejected.
  **Not** bitwise for `velocity`/`pressure`. A pre-port A/B (TEST_LEDGER G5.7) settles that the
  port did not cause it: `193b3fa` produces the same two numbers to every digit. Concern C13
  stays open for the behaviour itself, which is RIOT-wide and worth raising upstream. That A/B
  also showed the uninterrupted `noh` run bitwise identical between the two binaries, which
  extends H01's hydro-unchanged evidence to a second problem and pgen path.
- **LLF ported and verified** (TEST_LEDGER G5.8): the third and last MHD solver,
  `hydro/riemann = mhd_llf`. No degeneracy guards (nothing to degenerate) and no ideal-gas
  restriction (bulk modulus only, like HLLE) -- do not copy HLLD's `use_general_pte` guard
  here. 6 new unit tests, `ctest` **47/47**. The diffusivity ordering HLLD < HLLE < LLF is
  strictly monotone on all five Brio-Wu fields with no exceptions, and is corroborated by
  field-loop magnetic-energy retention 0.655 / 0.726 / 0.738 from an unrelated measurement.
  This is oracle-INDEPENDENT: the ordering follows from the wave structure. CT unaffected
  (axial field at roundoff); 1 vs 4 ranks bitwise.
- **Instrumentation landed** (TEST_LEDGER G5.9): `mhd/monitor_divb` now works (max,
  volume-weighted mean, and the dimensionless eta per step; identical at 1 vs 4 ranks), and
  two solver-health counters (`hlld_fallback`, `density_floor`) report on increase, MPI-summed
  and normalized per cell per step. `ctest` **49/49**; MHD output bitwise unchanged.
  **The fallback counter immediately corrected G5.8**: on the 3D field loop HLLD falls back to
  HLLE on ~7 faces per cell per step (planar field, `b3 = 0`, zero outside the loop), so that
  problem's HLLD-vs-HLLE retention comparison is weak corroboration, not independent evidence.
  Brio-Wu (zero fallbacks) is the real evidence. **Read a fallback rate before trusting any
  HLLD-vs-HLLE comparison on a new problem.**
- **Startup rejections verified** (TEST_LEDGER N01): 23/23 with a positive and a negative
  control. Found two dead general-PTE guards (see the "do not repeat" entry on resolved-vs-input
  parameters).

## Open concerns — READ THIS BEFORE CLAIMING ANYTHING

**[`OPEN_CONCERNS.md`](OPEN_CONCERNS.md) is the register of doubts about work already done**, as
opposed to the "Not implemented" list below, which is features not yet written. It records where
a defect could currently be hiding behind a passing test. Two items there are HIGH and both
undercut results recorded in this file if they turn out badly:

- **C1**: eight of the nine `tst/scripts/` suites are not yet run, despite Stage 1 changing
  shared `sparse_update` machinery that every physics package uses. This is *scheduled* Stage 5
  work rather than a skipped step, and it should be sequenced **after the last code change** so
  it validates the final state — see C1 for why running it earlier just means running it twice.
- ~~**C2**: the startup rejections (N01) have never been executed~~ — **RESOLVED 2026-09-18**,
  23/23 with a positive and a negative control (TEST_LEDGER "N01"). It found two real defects:
  the HLLD ideal-gas guard read the wrong input block and was dead code, and both general-PTE
  guards read the input flag rather than the resolved one, so a non-ideal `eos_type` bypassed
  them entirely. Two residual gaps are recorded in the C2 Resolved entry.

Do not delete entries from that file; move them to its Resolved section with evidence.

## Not implemented

MHD eigenmodes in `linear_modes.cpp`; the `tst/scripts/mhd/`
regression harness;
U05 (face stage copy
on a live mesh) not run as a unit test, though it is now exercised implicitly by every
two-stage RK MHD run. All verification is single-node (shared memory, no interconnect) and
uniform-grid, so no coarse/fine flux or EMF correction is exercised. The CPAW is axis-aligned
only, so no test propagates a wave obliquely to the grid.

## Next actions

1. ~~**Orszag-Tang pgen**~~ — **DONE, PASS** (TEST_LEDGER G4.4). To re-run:
   ```
   mkdir -p /tmp/mhdot && cd /tmp/mhdot
   R=/Users/taitano/Documents/git/riot
   for N in 1 4; do
     mpiexec -n $N $R/build/src/riot -i $R/inputs/mhd/orszag_tang.rin \
       parthenon/job/problem_id=ot_r$N
   done
   . $R/riot_venv/bin/activate
   PYTHONPATH=$R/claude_sessions/mhd_runs \
     python3 $R/claude_sessions/mhd_runs/analyze_orszag_tang.py ot_r1 ot_r4
   ```
   Pass `expect_heating=False` when checking the t=0 snapshot — the shock-heating criterion
   is the one check that is meaningless before shocks form. Everything else IS meaningful at
   t=0 and is a stronger statement there, because the analytic box means are exact for the
   discrete initial state; that snapshot is what validates the pgen itself.
2. ~~**MPI (P01)**~~ — **DONE, PASS** (TEST_LEDGER "P01 (complete)"). Bitwise identical to
   serial at 1/2/3/4/5/8 ranks across all three loop axes, 2D and 3D, and with shocks. To
   re-run it:
   ```
   mkdir -p /tmp/mhdmpi && cd /tmp/mhdmpi
   R=/Users/taitano/Documents/git/riot
   for N in 1 2 3 4 5 8; do
     mpiexec -n $N $R/build/src/riot -i $R/inputs/mhd/field_loop_3d.rin \
       parthenon/job/problem_id=fl3d_r$N
   done
   . $R/riot_venv/bin/activate
   S=$R/claude_sessions/mhd_runs/analyze_field_loop.py
   python3 $S stats fl3d_r4 --axis 3
   python3 $S compare fl3d_r1 fl3d_r4
   ```
   `mpiexec` is open-mpi from `/opt/homebrew/opt/open-mpi` (the build is configured against
   it, not mpich). Sweep `mhd_field_loop/loop_axis` — with `loop_axis=3` the F3 exchange
   carries only zeros, so one axis alone does not test all three face directions. Include a
   rank count that does not divide the 8 blocks evenly.
3. ~~**Circularly polarized Alfven wave**~~ — **DONE, PASS** (TEST_LEDGER G4.5). To re-run:
   ```
   mkdir -p /tmp/mhdcpaw && cd /tmp/mhdcpaw
   R=/Users/taitano/Documents/git/riot
   for N in 16 32 64 128; do
     $R/build/src/riot -i $R/inputs/mhd/cpaw.rin parthenon/job/problem_id=d1_n$N \
       parthenon/mesh/nx1=$N parthenon/meshblock/nx1=$((N/2))
   done
   . $R/riot_venv/bin/activate
   PYTHONPATH=$R/claude_sessions/mhd_runs \
     python3 $R/claude_sessions/mhd_runs/analyze_cpaw.py d1_n16 d1_n32 d1_n64 d1_n128
   ```
   To sweep `wave_dir`, the LONG axis must move with it: set that direction to N over [0,1]
   and the other two to 4 cells over 0.125, adjusting `parthenon/meshblock` to match or
   Parthenon aborts with "Block size is not evenly divisible into the base mesh size". Note
   the shell is zsh, where an unquoted `$VAR` holding several arguments is **not**
   word-split — build override lists as an array and expand `"${arr[@]}"`, or the whole list
   arrives as one argument and produces exactly that divisibility abort.
4. ~~**Startup rejections (N01)**~~ — **DONE, PASS**, 23/23. To re-run:
   ```
   mkdir -p /tmp/n01 && cd /tmp/n01
   R=/Users/taitano/Documents/git/riot
   . $R/riot_venv/bin/activate
   python3 $R/claude_sessions/mhd_runs/n01_startup_rejections.py
   ```
   Add `--cyl-exe <path>` to include the non-Cartesian case, which needs a build configured
   with `-DPARTHENON_COORDINATES=UniformCylindrical` (the coordinate type is compile-time, so
   no input can reach that rejection in the normal binary). The `base` positive control and
   the `cyl_control` negative control are load-bearing, not decoration — without them the
   sweep can pass vacuously. Requires `inputs/mhd/brio_wu.rin` and, for the curvilinear case,
   `inputs/sedov_rz.rin`; both are generated from their `.py` sources.
5. ~~**Restart equivalence (C5)**~~ — **DONE, PASS** (TEST_LEDGER G5.6). Found two real
   defects: derived magnetic state was never rebuilt on restart (3.7e-06 permanent error),
   and restarting across an MHD/hydro change ran silently in both directions (80% pressure
   error one way). Re-run instructions are in the G5.6 entry. Note that
   `f.bulk.magnetic_field` must be added to `parthenon/output1/variables` explicitly or the
   one field the test exists for is skipped, and the checkpoint cadence must be
   `parthenon/output2/dn`, not `dt`.
6. ~~**LLF**~~ — **DONE, PASS** (TEST_LEDGER G5.8). Re-run the three-solver comparison with:
   ```
   mkdir -p /tmp/llf && cd /tmp/llf
   R=/Users/taitano/Documents/git/riot
   for s in mhd_llf mhd_hlle mhd_hlld; do
     $R/build/src/riot -i $R/inputs/mhd/brio_wu.rin \
       parthenon/job/problem_id=bw_$s hydro/riemann=$s
   done
   . $R/riot_venv/bin/activate
   python3 $R/claude_sessions/mhd_runs/analyze_brio_wu.py bw_mhd_llf bw_mhd_hlle bw_mhd_hlld
   ```
7. ~~**Counters (C6) and `mhd/monitor_divb` (C7)**~~ — **DONE** (TEST_LEDGER G5.9). To see the
   fallback counter do something, run the field loop under HLLD -- Brio-Wu reports nothing
   because it never degenerates:
   ```
   $R/build/src/riot -i $R/inputs/mhd/field_loop_3d.rin hydro/riemann=mhd_hlld \
     parthenon/time/nlim=3 mhd/monitor_divb=true
   ```
8. Remaining Stage 5: MHD docs in `doc/sphinx` (currently zero MHD content), the
   `tst/scripts/mhd/` harness, and the eight-suite regression sweep LAST (C1). The sweep is
   hours of runtime but nearly free in context, so it is the one item worth backgrounding.

## Do not repeat

- **Face fields must never be read through a flat loop-abstraction view.** Parthenon
  allocates a non-`CellMemAligned` face field with NODE-shaped extents
  (`metadata.cpp:381-387`), while the flat pack/flux views index through the index space's
  single memory indexer, which is cell- or node-shaped only (`index_space.hpp:203`).
  Cell strides on node-shaped storage read the wrong element with no fault and no
  assertion. The contract states this outright
  (`LOOP_ABSTRACTION_CONTRACTS.md:96`). Use `MHD::FaceB` / logical coordinates, or
  `LoopConstraint::DifferentMemSpaces` for a face-only kernel. Fixed in `1350c20`; three
  Stage 1-2 sites had it wrong.
- **`MHD::AddMagneticEnergyToTotal` must be called from the pgen, not
  `PostInitialization`.** `Mesh::Initialize` runs a full
  PreCommFillDerived/communicate/FillDerived cycle before any PostInitialization hook, and
  `FillInteriorDerived` there subtracts the magnetic energy from a total that does not yet
  contain it. That pair is self-inverse only while the intermediate thermal energy stays
  positive; where it is clipped the information is gone. Observed as Brio-Wu's right state
  at P = 0.78125 (exactly `B^2/2`) instead of 0.1, **while the left state came out exactly
  right** — a bug that only corrupts states where a floor triggers.
- **Do not call a difference "pre-existing" without building the pre-port commit.** Running a
  hydro-only case with the CURRENT binary excludes MHD-specific code (it is all gated on
  `do_mhd`) but NOT the Stage 1 edits to shared `sparse_update` machinery, which run in
  hydro-only mode too. That distinction was collapsed once for concern C13 and had to be
  retracted. The A/B is cheap and the recipe is in TEST_LEDGER G5.7; `/tmp/riot_base` +
  `/tmp/riot_base_build` may still exist.
- **Do not assume any initialization hook runs on a restart.** Parthenon calls
  `Mesh::Initialize(!is_restart, ...)`, and the ENTIRE `if (init_problem)` block
  (`mesh.cpp:872-953`) is skipped when resuming — the problem generator, every
  `PostInitialization`/`PostInitializationMesh` hook, AND the
  PreCommFillDerived/communicate/FillDerived cycle that follows them. Anything a package sets up
  in those hooks is simply absent after a restart. `UserWorkBeforeLoopMesh` is the hook that
  does run either way. This cost a 3.7e-06 permanently wrong restarted trajectory that stayed
  divergence-free and passed every other test (G5.6). Note `levelsets.cpp:97` uses
  `Globals::is_restart` to *skip* its hook on restart — the same flag, the opposite need, so
  seeing that pattern is not evidence that restart is handled.
- **Do not derive a FillGhost field's ghosts as an afterthought on restart.** Nothing
  communicates between `UserWorkBeforeLoopMesh` and the first stage's reconstruction, so
  `IndexDomain::interior` is not enough there even though it is correct inside the step loop.
  Measured: `interior` costs 1.4e-10 in time by cycle 40 where `entire` is bitwise.
- **Do not validate a configuration by reading the input file when a package has already
  resolved it.** `pin->GetOrAddBoolean(...)` returns what the user typed, or a default —
  which is *not* the value the code will use if a package derives it. Concretely,
  `materials.cpp:450` sets `use_general_pte = true` for any non-ideal `eos_type` regardless
  of the flag, so both MHD general-PTE guards read `false` on exactly the configurations they
  existed to reject. It is also easy to name the wrong block and get a silent default
  forever: the `hydro.cpp` guard read `<multiphysics>` for an option that lives in
  `<materials>` and was dead code from the day it was written, with `GetOrAdd` quietly
  creating the bogus entry. Read `pkg->Param<T>(...)` and order the check after that
  package's `Initialize`. Found by N01; both fixed.
- **Do not filter the update pack on `Metadata::Conserved`.** Advected scalars
  (`scalars.cpp:62,81`) and level sets (`levelsets.cpp:41`) are updated by
  `UpdateToNextStage` and carry no `Conserved` flag.
- **Do not try `FlagCollection` subtraction in the update descriptor.**
  `MakePackDescriptor<any>` takes only a required-flag vector with AND semantics
  (`make_pack_descriptor.hpp:69-75`).
- **Do not change `BulkSoundSpeed`.** The MHD solver computes the fast speed internally
  from the acoustic speed it is passed. The only other fast-speed site is the CFL vote,
  which is done.
- **Do not port `artemis/src/utils/refinement/{prolongation,restriction}.hpp`.** RIOT's
  Parthenon pin already supplies `ProlongateInternalTothAndRoe` and edge-aware
  `RestrictAverage`.
- **Do not use `external/parthenon/example/fine_advection/stokes.hpp`** for the CT curl;
  its face-variable sign branch carries an upstream `TODO(LFR): This is untested`.
- **Do not transcribe the donor's curvilinear EMF metric factors** (DONOR_KERNELS.md
  section 4c). Exact in Cartesian, O(delta) wrong otherwise.

## Operational traps

- **Delete the accumulator files before any `--reuse_build` re-run**:
  `tst/build/src/{linwave-errs.dat,linwave_mm-errs.dat,compression.out0.hst}`.
  `linear_modes.cpp:388-399` appends when the file exists, which doubles it and makes
  `analyze()` throw. This looks exactly like a regression and is not one.
- **`run_tests.py` deletes `tst/build` unless `--save_build` is passed**, taking the
  numeric outputs with it — so an MD5 comparison needs `--save_build`. And a run WITHOUT
  `--reuse_build` reconfigures from scratch with default compilers, which on macOS is
  AppleClang and fails. Configure `tst/build` by hand (see `build_macos_gcc` memory), then
  always `--reuse_build --save_build`.
- **The MD5s recorded in TEST_LEDGER for the hydro baseline are build-configuration
  specific.** Do not use them to judge "hydro unchanged" across sessions; run the A/B in
  TEST_LEDGER H01 (worktree at `193b3fa`, submodules symlinked, identical cmake args)
  instead. A stored hash conflates a build difference with a code regression.
- **Analysis helper**: `claude_sessions/mhd_runs/analyze_field_loop.py` computes the CT
  invariants and does exact cross-run comparison, reassembling blocks onto the global grid
  via `LogicalLocations`. Use it rather than rewriting the reassembly: two runs of the same
  problem with different decompositions have different array shapes and block orderings, so
  they can only be compared after that mapping, and comparing `.phdf` hashes is meaningless
  (HDF5 containers carry timestamps and the embedded input deck). Verified against the
  recorded G4.3 numbers.
- **`carbuncle` needs numpy visible to the EMBEDDED interpreter.** Its pgen is
  `inputs/noh.py`. Run with
  `PYTHONPATH=$(pwd)/riot_venv/lib/python3.12/site-packages`, or it aborts with
  `loader.exec_module failed: No module named 'numpy'` (exit 134), which reads like a
  physics crash and is not one.
- `RIOT_BUILD_CATCH2=OFF` fails on this machine (no system Catch2); keep it `ON`.
- `clang-format-20` lives in the venv:
  `. riot_venv/bin/activate && CFM=$(pwd)/riot_venv/bin/clang-format VERBOSE=1 ./script/format.sh`
- Do not stage `external/parthenon`. It is dirty only from the nested Kokkos scratch-limit
  patch applied by the build from the tracked `riot_kokkos.patch`.
- **Input decks are `.py` generators, not `.rin` files.** `.gitignore:59` ignores `*.rin`,
  because in RIOT a `.rin` is a generated artifact: the tracked source is a Python script
  under `inputs/` that calls `riot.input(...)` and ends with `riot.input.generate_input()`
  (see any file under `inputs/advection/`). Hand-writing a `.rin` appears to work and then
  silently fails to be committed — the four MHD decks were written that way at first and were
  absent from every commit that claimed to add them, discovered only when `git add` refused
  `inputs/mhd/orszag_tang.rin`. To regenerate:
  ```
  . riot_venv/bin/activate
  cd inputs/mhd
  PYTHONPATH=$PWD/../../script/inputs:$PWD/../../build/singularity-eos/python \
    python3 orszag_tang.py     # -> orszag_tang.rin
  ```
  `script/inputs/riot.py` imports the built `singularity_eos` module, which is why the build
  directory has to be on `PYTHONPATH` as well.
