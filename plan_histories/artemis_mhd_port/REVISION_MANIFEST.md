# Revision manifest — Artemis → RIOT ideal-MHD port

This file records the immutable revision identity of the port. **Never overwrite
an entry when `main` advances.** Append a new integration-base entry and retain
the previous one.

Fetch/inspection time (UTC): **2026-09-18T19:23:13Z**

## Target (RIOT)

| Item | Value |
| --- | --- |
| Repository | `https://github.com/lanl/riot.git` (remote `origin`, verified) |
| Integration base SHA | `193b3fa2a61557cb4fc761bb87de6687ca781edf` |
| Base ref | `origin/main` — local `HEAD` equals `origin/main` exactly |
| Base commit subject | `Merge pull request #17 from lanl/chadmeyer/fix-typo` |
| Working branch | `taitano/mhd-porting` |
| Worktree status | Clean except `external/parthenon` (see below) and untracked session dirs (`.claude/`, `.codegraph/`, `.mcp.json`, `CLAUDE.md`, `claude_sessions/`, `riot_venv/`) |

### RIOT submodules (recursive, at the recorded superproject commit)

| Path | SHA | Describe |
| --- | --- | --- |
| `external/Catch2` | `fa43b77429ba76c462b1898d6cd2f2d7a9416b14` | v3.7.1 |
| `external/kokkos-kernels` | `2c7c3c301e810c7ebde0f154db900850a108ddb0` | 5.1.0 |
| **`external/parthenon`** | **`928544a6d903f4440611771ac4e12b3f40df1d59`** | **v25.12-639-g928544a6d** |
| `external/parthenon/external/Catch2` | `216713a4066b79d9803d374f261ccb30c0fb451f` | v2.13.8 |
| `external/parthenon/external/Kokkos` | `267ebc25fc5c8b96bb321f34d78f97b1c30f8830` | 5.1.1 |
| `external/singularity-eos` | `a91489d69e52ed982c3e474e292468d3d33da256` | release-1.5.0-3798-ga91489d6 |
| `external/singularity-eos/utils/eigen` | `2859db0220cd8644c1a75a3bf04f62f551f73f22` | before-git-migration-591-g2859db022 |
| `external/singularity-eos/utils/kokkos` | `3ec81abe1816109f6f62ac48cef41921f91a4d00` | 5.1.0 |
| `external/singularity-eos/utils/kokkos-kernels` | `2c7c3c301e810c7ebde0f154db900850a108ddb0` | 5.1.0 |
| `external/singularity-eos/utils/ports-of-call` | `2c687c13c6adb34b01ba56c34d15a4bed505c17f` | v1.3.0-305-g2c687c1 |
| `external/singularity-eos/utils/pybind11` | `d03662f0984f652b60e7ddce53d3868002275197` | v3.0.4 |
| `external/singularity-eos/utils/spiner` | `ecbc699917a20a4fc8a248d2be635277ddec4214` | 1.6.4-36-gecbc699 |
| `external/singularity-opac` | `f0540035bb463090716d89afd6839bd6eb5f1f05` | gold-files-20260901~2 |
| `external/singularity-opac/utils/herumi-fmath` | `168ede1bc4a822b9ee2d584a6e20d2ba409d435a` | 168ede1 |
| `external/singularity-opac/utils/kokkos` | `08ceff92bcf3a828844480bc1e6137eb74028517` | post-formating-9968-g08ceff92b |
| `external/singularity-opac/utils/ports-of-call` | `a284b6b2d42e70afeb99babb23522c869200d6ea` | v1.3.0-256-ga284b6b |
| `external/singularity-opac/utils/spiner` | `e40c75df722aee786a6bf18ea88650630b0aed35` | v1.7.0-3-ge40c75d |

### Explanation of the dirty `external/parthenon` submodule

`git status` reports ` m external/parthenon`. **No Parthenon source file is
modified.** The commit pointer matches the recorded SHA; the dirtiness is
confined to the *nested* Kokkos submodule working tree:

```
 M core/src/Cuda/Kokkos_Cuda_Parallel_Team.hpp
 M core/src/HIP/Kokkos_HIP_TeamPolicyInternal.hpp
 M core/src/Kokkos_TypeInfo.hpp
 M core/src/OpenMP/Kokkos_OpenMP_Team.hpp
 M core/src/Serial/Kokkos_Serial_Parallel_Team.hpp
?? .riot_patch_applied
```

These are exactly the five files patched by RIOT's own build via
`riot_kokkos.patch` (65 lines, sentinel file `.riot_patch_applied`), which raises
the Kokkos level-1 scratch-memory ceiling from 20 MiB to 1 GiB. It is fully
reproducible from the tracked patch file, is intentional, and is not at risk of
being lost. It is helpful but incidental to MHD — CT kernels are scratch-heavy.

**No dependency change is required for this port.** See ADR-003 for the evidence
that the pinned Parthenon already supplies every CT primitive needed.

## Donor (Artemis)

| Item | Value |
| --- | --- |
| Repository | `https://github.com/lanl/artemis.git` (remote `origin`, verified) |
| Donor head SHA | `3e5aeb5` — branch `dempsey/mhd`, subject `Add missing dependency` |
| Donor comparison base SHA | `e8a4e0f5ad965a8ddb0171f6dad81d6a653870ee` |
| Base selection rationale | `git merge-base --all origin/develop dempsey/mhd` returned **exactly one** commit. The donor branch is *not* merged into `develop`, so the feature diff is non-empty and no PR-metadata reconstruction was needed. |
| Feature delta | 67 files changed, +7596 / −308, across 71 commits |
| Worktree status | Clean (no modifications, no untracked files of consequence) |
| Public PR | [lanl/artemis#123](https://github.com/lanl/artemis/pull/123) — ideal MHD, `dempsey/mhd` → `develop` |

The historical inspection anchors named in the supplied plan
(`3e5aeb5` "Add missing dependency" following `d78099c` "Add GS upwinding for
emfs") **are** the current donor tip. No later commits exist on the branch, so
the chosen donor equals the final coherent donor state including its late fixes.
The donor's Parthenon pin is `287313a77` (v25.12-193) — older than RIOT's.

## Toolchain and machine

| Item | Value |
| --- | --- |
| Platform | macOS (Darwin 23.6.0), arm64 |
| C compiler | `/opt/homebrew/bin/gcc-16` |
| C++ compiler | `/opt/homebrew/bin/g++-16` — Homebrew GCC 16.2.0 |
| CMake | 4.4.3 (requires `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`) |
| MPI | Open MPI 5.0.10 (`mpiexec`) |
| Python | 3.12.5 |
| Precision | double (`Real` = `double`, default) |
| Backend | Kokkos Serial/host — `RIOT_ENABLE_CUDA=OFF`, `RIOT_ENABLE_OPENMP=OFF` |
| Geometry | `UniformCartesian` |

AppleClang cannot compile RIOT; Homebrew GCC is mandatory on this machine.

### Build configurations used

**Unit-test build** — `/Users/taitano/Documents/git/riot/build`
(pre-existing, reused):
```
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16
-DRIOT_ENABLE_UNIT_TESTS=ON  -DRIOT_ENABLE_REGRESSION_TESTS=OFF
-DRIOT_ENABLE_MPI=ON  -DRIOT_ENABLE_HDF5=ON
-DRIOT_ENABLE_CUDA=OFF  -DRIOT_ENABLE_OPENMP=OFF  -DRIOT_ENABLE_SANITIZE=OFF
```

**Regression build** — `/Users/taitano/Documents/git/riot/tst/build`
(created fresh for this baseline; `tst/run_tests.py` hardcodes this path):
```
cmake -S .. -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-16 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DRIOT_ENABLE_UNIT_TESTS=ON -DRIOT_ENABLE_REGRESSION_TESTS=ON \
  -DRIOT_BUILD_CATCH2=ON -DRIOT_ENABLE_MPI=ON -DRIOT_ENABLE_HDF5=ON
```
Note `RIOT_BUILD_CATCH2` must stay `ON` on this machine — no system Catch2 is
installed, and `OFF` makes `find_package(Catch2 REQUIRED)` fail at
`CMakeLists.txt:428`.

**Regression gold data**: `RIOT_REGRESSION_GOLD_SYNC=ON` succeeded over the
network; version **20260831**, SHA512 verified by the build. Checksum
verification was *not* disabled.

**Donor build** — `/Users/taitano/Documents/git/artemis/build/src/artemis`
(pre-existing, reused). Donor tests were driven with the runner's own
`--exe` / `--output_dir` options rather than RIOT CMake switches, per the plan's
instruction to discover the donor's commands independently.

## Baseline results

Recorded in [`TEST_LEDGER.md`](TEST_LEDGER.md). Artifacts under
`claude_sessions/mhd_runs/`.

## Final upstream drift check

To be performed before handoff: re-fetch `origin/main`, record the new SHA,
review the delta from `193b3fa`, and rerun the affected gates. **Not yet done.**
