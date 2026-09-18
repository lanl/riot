# Capability matrix — RIOT ideal MHD

Status vocabulary, applied strictly:

- **validated** — implemented, and a named test at a frozen threshold passes.
- **implemented-unverified** — code exists, no passing gate. Not a support claim.
- **unsupported/rejected** — refused at startup with an actionable message.

Combinations are recorded, not just toggles: "AMR validated" and "2T validated"
would not imply "2T + AMR validated."

**Current overall status: the port is mid-implementation. No end-to-end MHD
simulation has been run. Nothing below is marked validated at the simulation
level.** Test IDs refer to [`TEST_LEDGER.md`](TEST_LEDGER.md).

## Numerical primitives

| Capability | Status | Evidence |
| --- | --- | --- |
| Magnetic energy `B²/(2μ₀)`, both unit conventions | **validated** | U01 |
| Unit-convention invariance (`μ₀ = 1` vs `4π`) | **validated** | U01b |
| Thermal-energy round trip through the conserved total | **validated** | U01 (incl. low-beta) |
| Fast magnetosonic speed, oblique | **validated** | U03 (quartic residual ≤ 1e-11) |
| Fast magnetosonic speed, degenerate orientations and `b→0` | **validated** | U03 |
| Face-to-cell B conventions, incl. collapsed directions | **validated** | 3 tests |
| Discrete curl of a vector potential is divergence free | **validated** | U04 (η ≤ 1e-13) |
| Face-aware stage register copy (`u1 ← u0`) | **implemented-unverified** | U05 not run — needs a live mesh |
| Cell-centered B / magnetic energy / div B from face state | **implemented-unverified** | no mesh-level test yet |

## Infrastructure

| Capability | Status | Evidence |
| --- | --- | --- |
| Hydro-only behavior preserved with MHD off | **validated** | G1.2 — hydro suite 7/7, numeric outputs MD5-identical to baseline |
| Genuine face-centered field registration (no `CellMemAligned`) | **implemented-unverified** | compiles and registers; not exercised |
| Edge (EMF) flux register via automatic Face→Edge promotion | **implemented-unverified** | relies on Parthenon `metadata.cpp:189-197` |
| `mhd` physics toggle, default off | **implemented-unverified** | N01 not run |
| Startup rejection of unsupported combinations | **implemented-unverified** | N01 not run |

## Not yet implemented (this scope, Stages 3–5)

| Capability | Status |
| --- | --- |
| MHD Riemann solvers (HLLE, HLLD, LLF) | not implemented |
| Magnetic reconstruction + shared normal-B substitution | not implemented |
| Fast-speed insertion into `BulkSoundSpeed` and the CFL vote | not implemented |
| Gardiner–Stone upwind EMF construction | not implemented |
| Constrained-transport face update | not implemented |
| Task-graph wiring (5 edges) | not implemented |
| MHD problem generators | not implemented |
| MHD regression tests | not implemented |
| Restart of nonzero face state | not implemented |

## Unsupported / rejected at startup

Enforced in `src/riot.cpp`; rationale in
[`adr/004-support-matrix.md`](adr/004-support-matrix.md).

| Configuration | Status | Why |
| --- | --- | --- |
| More than one material | **rejected** | Common-field/common-velocity mixture model needs scientific review |
| Non-Cartesian coordinates | **rejected** | Curvilinear face/edge metrics and magnetic geometry sources not implemented |
| Mesh refinement (SMR/AMR) | **rejected** | Divergence-preserving operators are registered but untested; the donor's MHD tests never exercise AMR, so no oracle exists |
| General PTE closure | **rejected** | Mixed-cell closure + general-EOS acoustic derivative need separate validation |
| Fixed/frozen fluid | **rejected** | Induction transports the field with the fluid velocity |
| Material strength | **rejected** | Combined stresses and signal speeds need review |
| BHR mix model | **rejected** | Energy/stress audit not done |
| Thermonuclear burn | **rejected** | Energy audit not done |
| Ionization / two-temperature | **rejected** | Electron energy/entropy coupling needs separate validation |
| Level sets | **rejected** | Advection consistency not audited |
| Multigroup diffusion | **rejected** | Matter–energy exchange audit not done |
| Radiation transport | **rejected** | Matter+radiation budget audit not done |
| Lasers | **rejected** | Deposition-reservoir audit not done |
| Prescribed sources | **rejected** | Energy handoff audit not done |
| Gravity | **rejected** | Gravitational work term needs an energy audit |
| Passive scalars | **rejected** | Advection must use the MHD transport flow |
| Tracer particles | **rejected** | Advection consistency not audited |
| Hydro-only Riemann solvers with MHD | **rejected** | `hllc`/`hllcf`/`chllc`/`lhllc`/`hll` have no magnetic terms |

## Not available from this donor at all

Hall MHD, resistive induction, electron inertia, Biermann battery, ambipolar
diffusion, Nernst transport, anisotropic/Braginskii magnetized transport. The
donor is **ideal** MHD. Adding any of these is a new physics project, not a port.

This work must not be described as extended MHD, nor as porting the Yehan Toh
paper. RIOT's electron thermodynamics coexisting with ideal induction would not
make the result extended MHD.

## Environment coverage

| Axis | Status |
| --- | --- |
| Double precision, Kokkos Serial host backend | exercised (unit tests) |
| MPI | exercised for hydro only; MHD not run |
| CUDA / GPU | **NOT RUN** — unavailable on this machine. Kernels follow Kokkos portability rules and are reviewed, but unvalidated is not validated |
| 1D / 2D / 3D | conventions unit-tested (incl. collapsed directions); no MHD simulation run |
