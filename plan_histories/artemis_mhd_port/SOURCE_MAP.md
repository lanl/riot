# Source map — donor responsibility → verified RIOT owner

Every RIOT path/symbol below was read in the pinned tree at
`193b3fa2a61557cb4fc761bb87de6687ca781edf`, not inferred from documentation.
Where the supplied plan's documented clues proved wrong, the correction is
flagged **[CORRECTION]**.

## 1. Package registration and physics configuration

| Responsibility | RIOT owner |
| --- | --- |
| Physics toggles, compatibility gates, package construction order | `RiotDriver::ProcessPackages`, `src/riot.cpp:56-272`. Toggles read at `:72-100`, gates at `:102-122`, `AddParam("do_xxx")` at `:125-140`, package `Initialize` calls at `:175-241`. |
| Rejection idiom for an unimplemented combination | `src/riot.cpp:112-113` — `PARTHENON_REQUIRE(!(do_ionization && do_radiation_transport), "Radiation transport with ionization is not yet supported.")` |
| Enumerated allowed parameter values (framework-enforced) | `pin->GetOrAddString(..., std::vector<std::string>{...}, "desc")`, e.g. `src/hydro/hydro.cpp:109-112` |
| Geometry / required-input validation | `RiotDriver::RiotDriver`, `src/riot_driver.cpp:58-75` (`IsCoord<>` checks, `CheckRequired`, `CheckDesired`) |
| Out-of-tree physics | `src/plugins.hpp:22-30`; hooks at `riot.cpp:243`, `riot_driver.cpp:108,267`. **Plugins can only add `dudt` sources** — they cannot add flux or update tasks, so MHD must be in-tree. |
| New source file registration | `src/CMakeLists.txt` `SRC_LIST` (`:17-160`, alphabetical, headers listed too) |
| `<physics>` block documentation | `doc/sphinx/src/introduction.rst:72-122` |

## 2. Variables, metadata, packs

| Responsibility | RIOT owner |
| --- | --- |
| Variable declaration macros | `src/variables.hpp:40` `VARIABLE_SCALAR`, `:49` `VARIABLE_VECTOR`, `:58` `VARIABLE_TENSOR`, **`:68` `VARIABLE_FACE`** (already exists; wraps `parthenon::variable_names::base_w_tt_t<false, TopologicalType::Face>`) |
| Namespace aliases | `ccbulk = cell_variables::cell_averaged::bulk` (string prefix `c.c.bulk.`), `ccmat`, `cm = cell_variables::material_averaged` |
| **The only two `Independent + WithFluxes` bulk fields** | `ccbulk::total_material_energy` (`src/hydro/hydro.cpp:157`), `ccbulk::momentum` (`:164`). Both `{Cell, Independent, Intensive, Conserved, WithFluxes}`. |
| Derived bulk state | `ccbulk::rho` `:149`, `velocity` `:168`, `temperature` `:174`, `pressure` `:179`, `internal_energy`/`bulk_modulus` `:184-187`, `max_signal` `:196` |
| Pack construction | `riot::MakePack<Vars...>(md[, matids][, PDOpts])`, `src/variables.hpp:391`; descriptor memoized in a function-local static |
| Existing face fields (**all `CellMemAligned`**) | `ccbulk::face_signal` (`hydro.cpp:190`), `mat::diffusive_fluxes` (`mix/mix.cpp:157`), `RadiationDiffusion::{Fgroup,D,kappa_face,face_area,DeltaX}`, `Ionization::D` (`ionization.cpp:202`). Rationale: `src/mix/mix.cpp:148-155`. |
| No edge variable exists | confirmed — and none is needed; the EMF is the face field's flux register |
| Cyclic right-handed basis **written for CT-MHD** | `RiotUtils::DirBasis`, `MakeDirBasis<DIR>`, `src/riot_utils/riot_loops.hpp:64-95` |

**[CORRECTION]** The plan's documented clues named `ccbulk::momentum`,
`ccbulk::total_material_energy`, `ccmat::rho` — all three confirmed correct.

## 3. Energy: which quantity is transported, and every writer

**`ccbulk::total_material_energy` = internal + kinetic. No magnetic term.**
Proven in both directions:

| Direction | Site |
| --- | --- |
| Thermal recovery `u = E − ½ρv²` | `src/multiphysics/fill_shared_derived.cpp:99-104` (`interior`) |
| Re-synthesis `E = u + ½ρv²` | `src/multiphysics/fill_shared_derived.cpp:325-335` (`entire`) |
| Documentation | `doc/sphinx/src/packages/hydro.rst:285-287` |

Complete writer audit and its classification into flux / incremental / absolute
writers is in [`adr/002-energy.md`](adr/002-energy.md). Key results:

- **Absolute reconstructors** (must be MHD-aware): `fill_shared_derived.cpp:325-335`, plus ~16 pgens in `src/riot_pgen/` (hydro-only, unaffected).
- **Incremental writers** (safe unchanged): `mix.cpp:538,571,685,702`; `ionization.cpp:1112,1148,1186,1492,1542,1593`; `gravity.cpp:87`; `tnburn/shared_sources.cpp:68,91`, `tnburn.cpp:392,458`; `laser.cpp:462,474,522,707,711,755,781`; `strength.cpp:198,295`; `multigroup_diffusion-tasks.cpp:331,353`; `prescribed_sources.cpp:186,212`.
- **Read-only**: `diagnostics/energies.cpp:71,85,105`.
- **[CORRECTION]** `microphysics/pte_closure_general.cpp:81` packs
  `total_material_energy` but **never dereferences it** — a dead pack entry. The
  closure consumes `ccbulk::internal_energy` (`:166`, `:316`), so no closure
  change is needed.

## 4. Reconstruction, Riemann, fluxes, timestep

| Responsibility | RIOT owner |
| --- | --- |
| Flux driver | `Hydro::CalculateFluxes`, `src/hydro/calculate_fluxes.cpp:722-784`; per-direction `CalculateFluxesImpl<DIR>` `:475-717` |
| Reconstruction dispatch (switch hoisted out of inner loops) | `Hydro::ReconCells<VARS...>`, `src/hydro/hydro.hpp:84-140`; `ReconVar` `:150-188` |
| Reconstruction type lists | `src/hydro/hydro.hpp:44-71` — `set_bulk_recon_types`, `sum_bulk_recon_types`, `mat_recon_types`, `set_strength_bulk_recon_types` |
| Reconstruction methods | `src/reconstruction/reconstruction.hpp:28-42` — `{CONSTANT, PLM, PPM4, WENO5, MP5}`, stencil widths `{0,1,2,2,2}` |
| Riemann solvers (free functions, `template <int DIR>`) | `src/hydro/riemann.hpp` — `lr_to_flux_fleischmann` `:31`, `hllc` `:117`, `lhllc/chllc` `:179,283,297`, `hll` `:313`, `strength` `:379`. Fixed 22-arg signature, **no B slot**; return value is the max signal speed. |
| Solver dispatch | `switch (rsolver_tag)` at `calculate_fluxes.cpp:658-694`, instantiating `BulkRiemannFluxes<DIR, FLUX_FN>` with the solver as a template non-type arg |
| **Precedent for an optional-physics extended-signature solver** | `StrengthFluxes`, `calculate_fluxes.cpp:228-380` — allocates its own scratch, does its own recon, writes its own flux registers |
| Sound speed at faces | `BulkSoundSpeed(bmod, rho)`, `calculate_fluxes.cpp:85-88`; strength-stiffened `bmod += (4/3)gmod` at `:315-326` ← the exact model for adding a magnetic term |
| Acoustic derivative source | `cm::bulk_modulus` from `eosm_c.BulkModulusFromDensityTemperature` (`fill_shared_derived.cpp:257-260`), volume-fraction-summed into `ccbulk::bulk_modulus` (`:302-303`). **Neither closure returns a sound speed.** |
| CFL | `Hydro::EstimateTimestepMesh`, `src/hydro/hydro.cpp:683-733`. **Directional sum**: `dt = cfl / Σ_d (max_sig_d / dx_d)` — the same convention as Artemis. |
| Face signal-speed pathway | Riemann return → `ccbulk::face_signal` → `Hydro::CalculateMaxSignalSpeed` (`hydro.cpp:635-678`) → `ccbulk::max_signal(d)`. **`:660-673` is the canonical example of reading a face field from a cell kernel.** |

## 5. Integrator stages, update, register copies

| Responsibility | RIOT owner |
| --- | --- |
| Integrator | `parthenon::LowStorageIntegrator`, constructed `src/riot_driver.cpp:56`. Names include `rk1, rk2, vl2, rk3, rk34, rk4`. |
| Stage coefficients | read at `src/riot_driver.cpp:196-201` (`beta`, `gam0`, `gam1`, `c`); applied **only** inside `UpdateToNextStage`. `delta[]` is unused — so `vl2` may not behave as intended. |
| Conserved update | `sparse_update::UpdateToNextStage`, `src/riot_utils/sparse_update.hpp:139-228`. **HAZARD A** — packs `<any>` on `{WithFluxes}` over `TE::CC` with `FaceArea`/`CellVolume`. |
| Register deep copy | `sparse_update::DeepCopyData` `:234-259`, `DeepCopyIndependentData` `:264-268`. **HAZARD B** — `TE::CC` + `entire`, truncates face storage. |
| Registers | `RiotStepInit`, `src/riot_driver.cpp:117-138` — `u0 = AddShallow(base, names)`, `u1 = Add(u0)`; names from `RiotUtils::GetUnsplitVarNames` (`riot_utils.cpp:90-100`, everything not `OperatorSplit`) |
| `dudt` source aggregation | `riot_driver.cpp:307-312`; register via `RegisterMeshDataSubset("dudt", MakePackageDudtRequirements(...))` — e.g. `gravity.cpp:49-51`, `mix.cpp:168-171` |
| Full task list | `RiotDriver::RiotStepTasks`, `src/riot_driver.cpp:169-375`. Insertion points in [`adr/003`](adr/003-field-topology-and-stages.md). |
| Derived-state hooks | `src/riot.cpp:253-256` — `PreCommFillDerivedMesh = Multiphysics::FillInteriorDerived`, `PreFillDerivedMesh = Multiphysics::PostCommsFillDerived`. **[CORRECTION]** `fill_shared_derived.cpp` is never called directly by the driver; Parthenon invokes it at `riot_driver.cpp:363,370`. |
| Operator-split ordering | `RiotDriver::Step`, `src/riot_driver.cpp:143-164` — laser → all RK stages → split packages → post-step |

## 6. Closure and EOS

**[CORRECTION]** The plan referenced `src/microphysics/pte_closure.cpp`. **No such
file exists.** The split is:

| File | Contents |
| --- | --- |
| `src/microphysics/pte_closure.hpp:31-55` | declarations, `LocalEosIndexer` |
| `src/microphysics/pte_closure_general.cpp:37-371` | `ApplyMixedCellClosure` — `PTESolverRhoT` at `:170-175`, energy constraint `sie_tot = internal_energy/rho_pte` at `:166` |
| `src/microphysics/pte_closure_ideal.cpp:36-153` | `ApplyIdealGasClosure` — analytic shortcut, `T = u/Σ(ρ cv)` at `:93-105` |
| `src/microphysics/eos_riot.hpp` | `RiotEOS::EOS` singularity `Variant` (`:75`); device dispatch idiom `eosm.EvaluateDevice(...)` |

Both closures consume **thermal** energy and return **no** sound speed.

## 7. Loop abstractions

| Responsibility | RIOT owner |
| --- | --- |
| Aliases / tags | `src/riot_utils/riot_loops.hpp` — `RiotLoop = parthenon::loop_abstraction` `:26`, `LoopType<Cs...>` `:156-168`, `ReductionType` `:178-208` |
| Index space | `lt::GetIndexSpace(domain, nhalo, nblocks, md, TopologicalElement)` — **already TE-parameterized** |
| Views | `make_pack_view` (supports `pv(TE, var, kji)`), `make_sparse_pack_view`, `make_flux_pack_view(range, pack, DIR)`, `make_var_view`, `make_flux_view` |
| TE-aware var view (needed for face state) | 4-arg `make_var_view(range, pack, te, var)`, `external/parthenon/src/loop_abstraction/pack_view.hpp:308-312` |
| Face/edge flux indexing | `SparsePack::flux(b, TE, ...)` maps `TE → dir = (int(te)%3)+1`, `sparse_pack.hpp:390-395` — **works for edges unchanged** |
| Halo tags | `RiotLoop::halo::minus_{i,j,k}_t`; symmetric `RiotUtils::halo::pm_{i,j,k}_t` `:43-62` |
| Representative kernel to match | `BulkRiemannFluxes`, `src/hydro/calculate_fluxes.cpp:117-161` |

## 8. Problem generators and tests

| Responsibility | RIOT owner |
| --- | --- |
| pgen registration | `FOREACH_PROBLEM` macro list, `src/riot_pgen/pgen.hpp:40-54`; `RegisterProblem` overloads `:65-68`; explicit registrations with modifier/package `:96-116` |
| Canonical pgen body | `src/riot_pgen/shock_tube.cpp:26-84` — `pmb->par_for` over `IndexDomain::entire`, sets conserved state |
| Per-problem package + L1 error accumulation | `linear_modes::ProblemPackage`, `src/riot_pgen/linear_modes.cpp:309-353` → `<problem_id>-errs.dat` |
| **No precedent for writing a face field in a pgen** | confirmed — new ground |
| Unit tests | `tst/unit/` with custom Catch2 main (`catch2_define.cpp:16-30`); add one line to `tst/unit/CMakeLists.txt:18-33`. On-device pattern documented at `tst/unit/test_riemann.cpp:15-35`. |
| Regression harness | `tst/run_tests.py` — auto-discovers `scripts/<dir>/<name>.py` exposing `run(**kwargs)` and `analyze()`; helper `tst/scripts/utils/riot.py` (`make`, `generate`, `run`, `mpirun`) |
| **Runner path constraint** | `riot.py` hardcodes `build/src/riot`, `../inputs/`, and `build/singularity-eos/python` **relative to `tst/`** — so the regression build must live at `tst/build`, not the repo-root `build`. |
| Gold data | `tst/CMakeLists.txt:15-62` — `RIOT_REGRESSION_GOLD_{SYNC,VER,HASH,LOCAL}`, SHA512-verified, from GitHub releases. New gold requires a release + version bump. |
| Input decks | Python scripts under `inputs/` emitting `.rin` |
