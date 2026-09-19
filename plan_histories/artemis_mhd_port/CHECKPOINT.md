# Checkpoint

Updated UTC: 2026-09-18
Riot original base SHA: `193b3fa2a61557cb4fc761bb87de6687ca781edf`
Donor head / comparison base: `3e5aeb5` / `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee`
Branch: `taitano/mhd-porting`
Commits: `725a19a` (Stages 0-2), `f0af069` (Stage 3 solver), `4d1a8e0` (checkpoint),
`1350c20` (face memory-layout fix), `dced139` (flux wiring + CT + pgens)

Current stage: **Stage 4 substantially complete.** Gates passed: G0, G1, G2, G3, and G4 in
part. **MHD now runs end to end**: Brio-Wu agrees with the Athena++ reference and the 2D
multi-block field loop holds div B at roundoff.

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
- **Two problem generators**: `mhd_shock_tube` (Brio-Wu), `mhd_field_loop` (2D and 3D, with
  a `loop_axis` option that rotates the setup onto each pair of edge directions). Inputs in
  `inputs/mhd/`.
- **3D CT verified** (TEST_LEDGER G4.3): the Gardiner-Stone axial-field test passes in all
  three permutations with the axial field at roundoff, and 1-block vs 8-block results are
  bitwise identical.
- **Tests**: `ctest` **34/34**; hydro regression 7/7; hydro output **bitwise identical to
  the pre-port commit**, established by a build-config-held-fixed A/B against `193b3fa`
  rather than by a stored MD5 (see TEST_LEDGER H01 re-run for why the stored hash is not a
  valid cross-session invariant). Numeric results in `TEST_LEDGER.md`.

## Not implemented

`mhd_orszag_tang` and MHD eigenmodes in `linear_modes.cpp`; the `tst/scripts/mhd/`
regression harness; HLLD and LLF; `mhd/monitor_divb` is registered but not yet consumed;
restart of face state untested; test N01 (startup rejections) not run; U05 (face stage copy
on a live mesh) not run as a unit test, though it is now exercised implicitly by every
two-stage RK MHD run. **Everything verified so far is single-rank.**

## Next actions

1. **Orszag-Tang pgen** — the remaining G4 item, and the one that exercises shocks with a
   magnetic field. Checks: `max|divB| <= 1e-10`, mean rho to rtol 1e-12, mean E to 1e-8,
   and the decomposition residual `|E - u - KE - E_mag| <= 1e-10`.
2. **MPI (P01)** — 2 and 4 ranks. This is the next thing to do and the largest untested
   surface. Single-rank decomposition invariance in 3D is already established and is bitwise
   (TEST_LEDGER G4.3 / P01 partial), so what remains is genuinely the communication path and
   the task-dependency graph, which serial execution order can mask.

   Ready to run — the inputs and the analysis already exist:
   ```
   cd /tmp/mhdrun   # or any scratch dir
   R=/Users/taitano/Documents/git/riot
   for N in 1 2 4 8; do
     mpiexec -n $N $R/build/src/riot -i $R/inputs/mhd/field_loop_3d.rin \
       parthenon/job/problem_id=fl3d_r$N
   done
   . $R/riot_venv/bin/activate
   S=$R/claude_sessions/mhd_runs/analyze_field_loop.py
   python3 $S stats fl3d_r4 --axis 3          # invariants must hold at every rank count
   python3 $S compare fl3d_r1 fl3d_r4         # cross-rank comparison
   ```
   The 3D input is 8 mesh blocks, so 1/2/4/8 ranks all divide it evenly. Also worth a run
   where blocks do NOT divide evenly across ranks (e.g. `-n 3`), since that exercises a
   different ownership pattern.

   **Predicted results, so a benign difference is not misread as a bug.** The invariants
   (`axial`, `eta`) must hold at roundoff at every rank count — those are per-cell algebraic
   identities and MPI cannot excuse a violation. Cross-rank *bitwise* identity is the
   expectation but NOT guaranteed: the CFL timestep is a global reduction, so changing the
   rank count changes its summation order and `dt` can differ in its last bits, after which
   trajectories separate at roundoff. So: if `compare` reports a difference, first check
   whether the cycle counts and `dt` histories match. A difference that is roundoff-level
   AND accompanied by a differing dt history is benign; one that grows with time, or is
   localized at block boundaries, or appears while dt histories agree exactly, is a real
   bug — most likely a missing task-graph dependency or a face/edge exchange defect.

   Note `mpiexec` is open-mpi from `/opt/homebrew/opt/open-mpi` (the build is configured
   against it, not mpich).
3. **3D circularly polarized Alfven wave**, for order-of-accuracy rather than for branch
   coverage. The 3D EMF branches are now exercised AND checked (G4.3), so this is no longer
   the coverage gap it was; it is now about convergence rate.
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
