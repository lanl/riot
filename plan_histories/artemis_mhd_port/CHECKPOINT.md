# Checkpoint

Updated UTC: 2026-09-18
Riot original base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf`
Donor head / comparison base: `3e5aeb5` / `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee`
Branch: `taitano/mhd-porting`
Commits: `725a19a` (Stages 0-2), `f0af069` (Stage 3 solver), `4d1a8e0` (checkpoint),
`1350c20` (face memory-layout fix), `dced139` (flux wiring + CT + pgens)

Current stage: **Stage 4 complete. Gates G0, G1, G2, G3 and G4 all pass.** Brio-Wu agrees
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

## Not implemented

MHD eigenmodes in `linear_modes.cpp`; the `tst/scripts/mhd/`
regression harness; HLLD and LLF; `mhd/monitor_divb` is registered but not yet consumed;
restart of face state untested; test N01 (startup rejections) not run; U05 (face stage copy
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
4. **`tst/scripts/mhd/`** harness, adopting the donor's own thresholds.
5. Stage 5: HLLD (keep its degeneracy guards verbatim, DONOR_KERNELS.md section 13), LLF,
   reconstruction certification, restart equivalence.

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
