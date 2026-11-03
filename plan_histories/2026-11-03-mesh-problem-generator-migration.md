# Migrate RIOT problem generators to Parthenon's mesh-level generator

## Goal

Switch RIOT from Parthenon's per-`MeshBlock` problem generator to the mesh-level
`MeshProblemGenerator` (`MeshData`), and process the regions generator
(`region_pgen`) across many blocks at once. Keep the Python regions interface
(`mask(pos)`, `f(pos, out)`) and every existing Python region file unchanged.

This evaluation comes from reading the code. Nothing was built or run.

## Current state

### Problem generator dispatch (`src/riot_pgen/pgen.{hpp,cpp}`)

- Each problem registers a generator plus an optional modifier and package.
  The problem is looked up from `<riot>/problem`.
- Parthenon sees exactly two hook points: `pman.app_input->ProblemGenerator`
  (`src/template_main.hpp:55`) and `riot::ProblemGenerator` (`pgen.cpp:99`).
  That makes the switch cheap at this layer.
- `ClipMaterials` and `ValidateTemperature` are already written as packs with
  `b = 0`, so they port to `MeshData` almost unchanged.
- **Bug:** `ValidateVfracs` (`pgen.cpp:226-229`) computes
  `sqrt(Σ(1-vfsum)²) / N`, but an RMS check would be `sqrt(Σ/N)`. As written,
  the volume fractions in every cell of a 16³ block can be off by about 6.4 and
  the check still passes, so it almost never fires.
- **Cleanup:** the "function pointers must be passed by value" comment
  (`pgen.hpp:63`) has a simple cause: the `RiotProblem` constructors take
  non-const `pgen_t&`. If they took the function by value and moved it, the four
  `RegisterProblem` overloads could become one with default arguments.

### Regions code (`src/riot_pgen/regions.{hpp,cpp}`, `region_python.*`, `region_primitives.*`)

What works well:

- The design is solid: masks, precedence by region id, adaptive sampling of
  mixed ("dirty") cells, and init functions that fall back from region to
  defaults to constants.
- The Python interface is already batched. `mask(pos)` and `f(pos, out)` take
  an `(N,3)` array. None of the Python region inputs (`inputs/rose.py`,
  `logo.py`, `doubleshell/`, `dsplanar/`, `ionization/rt_plasma_viscosity.py`,
  `advection/slotted_cylinder.py`, `radiation_transport/equilibrium_box.py`)
  assume that `N` is one block or that points come in block order. The two
  `reshape` calls were checked and are harmless.
- The built-in masks can already run on the GPU (`KOKKOS_INLINE_FUNCTION`, plus
  the `eval_mask`/`mask_types` tuple used by `diagnostics/riot_viz.hpp`).

Likely bugs:

1. **Mixed cells use pressure as if it were density for P-first inits**
   (`regions.hpp:1240-1241`). `rho_added = vol·pf·local_state[0]·local_state[1]`.
   For `PT`, `PTiTe` and `PTiEquil`, `local_state[1]` is the pressure. So
   isotopes, strength and BHR quantities in dirty cells are weighted by
   `vfrac·P`, and the `mats_to_remove` test checks the wrong number. The clean
   path correctly uses `rho(p,k,j,i)`.
2. **Overlapping regions can lose material data** (`regions.hpp:1263-1265`).
   For `p == 0`, the material-specific state is assigned with `=` rather than
   added. If the same material appears in two regions that overlap a dirty
   cell, the second region overwrites the first one's contribution.
3. **An erase that does nothing** (`regions.cpp:276-277`). It removes
   `ccmat::electron_internal_energy` from `independent_init_vars`, but the set
   holds `ccbulk::electron_internal_energy` (`regions.hpp:112`). It also changes
   a header-`static` global from inside `InitBlock`.
4. **Latent bug in the dirty `bulk_state` loop** (`regions.hpp:1147-1151`). It
   writes only component 0 and reuses `vs` at whatever shape the previous
   velocity call left. Nothing reaches it today because no `bulk_state` key is
   in `var_mirror_map`.
5. Velocity init is set up twice (`regions.cpp:94` and `:180`). Harmless.

Structural issues that matter for the migration:

- **`BlockInitData` is one shared, mutable object.** It is stored as a package
  parameter (`src/riot.cpp:264`), and `InitBlock` rewrites `region_id`,
  `vol_frac_sum`, `coords` and `ndim` on every call. Blocks can't be processed
  in parallel, and the setup config is mixed with per-block scratch space.
- **Python is called with very small batches.** Masks and clean-cell fields get
  one call per block per region. In the dirty path, field functions are called
  with `N = 1` per dirty cell, per region, per field (`regions.hpp:1119-1209`).
  With Python regions and `nlev_max > 0`, this is probably the main setup cost.
- **The Python wrappers capture raw `this`** (`region_python.hpp:78, 93`).
  `default_python_region` is held by value inside `Regions`, so copying or
  moving `BlockInitData` during a refactor would leave dangling pointers.

## Parthenon constraints

- **Parthenon throws if both a mesh and a block generator are defined**
  (`external/parthenon/src/mesh/mesh.cpp:786-789`). RIOT has to switch over all
  at once at the dispatcher. It can't mix the two.
- **Allocating sparse fields mid-generator is safe.** Sparse packs rebuild when
  allocation status changes (`pack/sparse_pack/sparse_pack_cache.cpp:32-37`).
  The regions code can keep calling `AllocateSparse` on blocks it gets from
  `md->GetBlockData(b)->GetBlockPointer()`.
- During AMR initialization, Parthenon calls the generator again on every pass,
  so any per-call state has to be reset on each call.

## Effort estimate

| Piece | Effort | Risk |
|---|---|---|
| Dispatcher plus the 3 post-processing utilities | ~1 day | Low |
| Regions as a mesh generator that loops over blocks | 1–2 days | Low |
| Regions batched across a whole `MeshData` partition | 1–2 weeks | Medium (mixed-cell volume fractions) |
| 16 analytic generators | ~0.5 day each, mechanical | Low; optional |
| Built-in masks on the GPU | Later; Python regions stay on the CPU | Medium |

## Migration plan

### Phase 0 – Mesh dispatcher with a fallback for block generators (no behaviour change)

```cpp
using mesh_pgen_t = std::function<void(Mesh*, ParameterInput*, MeshData<Real>*)>;
// RiotProblem holds either pgen_t or mesh_pgen_t
void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto &prob = internal::GetProb(pin);
  if (prob.MeshPgen) prob.MeshPgen(pm, pin, md);
  else for (int b = 0; b < md->NumBlocks(); ++b)
    prob.ProblemGenerator(md->GetBlockData(b)->GetBlockPointer(), pin);
  ClipMaterials(md, pin); ValidateTemperature(md); ValidateVfracs(md);
}
```

- Set `app_input->MeshProblemGenerator` and remove the block hook.
- Add a `RIOT_MESH_PROBLEM(name)` macro next to `RIOT_PROBLEM`.
- Port the three utilities to `MeshData` packs. Fix the `ValidateVfracs`
  formula at the same time.

### Phase 1 – Split the regions state, then add a mesh entry point

- Separate `BlockInitData` into:
  - an immutable `RegionSet`: `Regions`, Python objects, init functions and
    flags. It is built once and kept in the package parameter. Make it
    non-copyable and non-movable to protect the captured `this` pointers.
  - a per-call workspace for scratch data.
- **Design the workspace in the multi-block layout from the start** (one per
  `MeshData`, with block indices; see the next section), not one per block.
  Phase 1 can still fill it one block at a time. Phase 2 then only changes how
  the passes are ordered, without another rewrite of the data structures.
- Make the init routine const with respect to `RegionSet`. Move the
  `independent_init_vars` pruning into the constructor (fixes bug 3).
- `region_pgen` becomes a mesh generator that loops over blocks. Initial dumps
  should be bitwise identical to the current code.
- This phase alone gives no speedup: blocks are still handled one at a time,
  with their own Python calls. What it does is remove the shared mutable state,
  so blocks no longer depend on each other.

### Phase 2 – Batch across the partition (the actual payoff; same Python API)

Rework `set_clean_mask`, `set_clean_state` and `set_dirty_state` into
gather / evaluate / scatter:

1. **Gather** sample points from every block in `md` into one
   `sample_positions_t`, with a back-index `(b, k, j, i)`, or
   `(b, cell, region)` for dirty-cell centroids.
2. **Evaluate** each region's `mask` once per refinement level. Evaluate each
   field function once per region on all clean points, and once on all dirty
   centroids. This removes the `N = 1` calls.
3. **Decide allocation** for each block from the mask results, then call
   `AllocateSparse` and build the pack.
4. **Scatter** into the pack or host copies, keeping the
   allocate/zero/deallocate logic per block.

Keep the region-order loops when summing within a cell. Then floating-point
results stay identical, and the existing `inputs/**/*.rin` region problems
become a regression test against Phase 1 output. For Python users, `N` gets
larger and points from different blocks are interleaved. No current input file
depends on either.

Fix bugs 1 and 2 as separate commits, before or after this phase, so the
regression comparison stays clean.

### Phase 3 – Optional

- When every region uses a built-in mask, evaluate masks on the GPU with
  `eval_mask`. Any Python region sends the setup down the CPU path.
- Port the analytic generators one at a time. The Phase 0 fallback means
  nothing forces this.

## Multi-block layout for regions

Every per-point array gets a block index, using offset arrays because each
block contributes a different number of points:

```
positions      : (Npts_total, 3)     all blocks in md, concatenated
pt_block/k/j/i : (Npts_total)        back-index into the pack
blk_offset     : (nblocks+1)         where each block's points start
region_of_pt   : (Npts_total)        result of the mask pass
```

With that layout, each stage becomes one loop over the whole partition:

| Stage | Loop shape | Where it runs |
|---|---|---|
| Gather cell centres | `par_for(b, k, j, i)` over a `MeshData` pack | CPU or GPU |
| Mask evaluation | one call per region over all `Npts_total` points | CPU for Python; built-in masks could use `eval_mask` on the GPU |
| Dirty-cell subcell refinement | per dirty cell, flattened across blocks | CPU (the tree is irregular) |
| Field functions | one call per region per field over all points | CPU for Python |
| Sparse allocation decision | per block, from `region_of_pt` | CPU (`AllocateSparse` is per block) |
| Scatter + EOS conversion (`set_cell_conserved*`) | `par_for` over points, using `(b, k, j, i)` | CPU now; GPU possible later |

Constraints:

- **Allocation splits the work into two passes.** A block's sparse material
  fields can't be written until they are allocated, and which materials a block
  needs is only known after the mask pass. The order is: mask everything →
  decide allocations per block → `AllocateSparse` → build the pack (it rebuilds
  itself when allocation status changes) → scatter. Today the code allocates as
  it goes, interleaved with the writes, so these steps have to be pulled apart.
- **Python can't run inside a device loop, and threads won't help.** Python
  stays a CPU batch step between gather and scatter. Splitting the Python calls
  across CPU threads per block wouldn't help either, because of the GIL.
  Batching is the only real gain for Python regions, and a large one,
  especially in the dirty path.
- **Batch size equals partition size.** With `pack_size = -1`, one batch covers
  every block on the rank. Clean cells are cheap, but with a high `nlev_max`
  the dirty-cell subcell corners can add up. Make the gather chunkable by block
  range; the offset layout supports that without changes.
- **The GPU scatter is not verified.** Running `set_cell_conserved_3t` on the
  GPU assumes the EOS calls (`RiotEOS::rho_from_P_T` and friends) and the
  ionization calls can run on the device. The current code uses the CPU-side
  EOS copies (`h.h.EOS`). Check device-callability before planning on it.
