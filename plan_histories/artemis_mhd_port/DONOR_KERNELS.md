# Donor kernels — verbatim source and Cartesian reductions

Quoted directly from `/Users/taitano/Documents/git/artemis` at `dempsey/mhd` `3e5aeb5`,
read from the files (not from a summary). This exists so the port's numerics can be
reviewed, and re-derived, without needing the Artemis tree checked out.

Attribution: these kernels are the work of the Artemis authors (Adam Dempsey et al.,
Triad National Security, LLC), reproduced here under the same LANL/Triad terms that
cover both codebases. Where the RIOT port deviates, the deviation is stated and
justified.

Each section gives (a) the donor code verbatim, (b) the Cartesian reduction that RIOT
actually implements, since this port is Cartesian-only (ADR-004).

---

## 1. Sign conventions

```
E = -(v x B)        =>   d_t B = -curl(E)

Fx(By) = -Ez        Fy(Bx) = +Ez
Fx(Bz) = +Ey        Fz(Bx) = -Ey
Fy(Bz) = -Ex        Fz(By) = +Ex
```

Transverse induction flux out of the Riemann solver:
`F_dir(B_t) = v_dir B_t - v_t B_dir`. The **normal**-component flux slot is written
as exactly `0.0` so it cannot contaminate EMF assembly.

Magnetic energy density is `B^2/(2 mu0)` — SI-style, not Gaussian `B^2/8pi` and not
Heaviside–Lorentz `B^2/2`. All donor MHD tests run scale-free with `mu0_code == 1`.
See ADR-001.

---

## 2. Cell-centered EMF — `src/mhd/emf.hpp:22-34`

```cpp
template <int EDGE, typename PACK>
KOKKOS_INLINE_FUNCTION Real CellEMF(const PACK &v, const int b, const int k, const int j,
                                    const int i) {
  const Real v1 = v(b, gas::prim::velocity(0), k, j, i);
  const Real v2 = v(b, gas::prim::velocity(1), k, j, i);
  const Real v3 = v(b, gas::prim::velocity(2), k, j, i);
  const Real b1 = v(b, field::cell::B(0), k, j, i);
  const Real b2 = v(b, field::cell::B(1), k, j, i);
  const Real b3 = v(b, field::cell::B(2), k, j, i);
  if constexpr (EDGE == X1DIR) return v3 * b2 - v2 * b3;
  if constexpr (EDGE == X2DIR) return v1 * b3 - v3 * b1;
  return v2 * b1 - v1 * b2;
}
```

Note this returns `-(v x B)` component-wise: `Ex = v3 b2 - v2 b3 = -(v x B)_x`.

## 3. Gardiner–Stone donor-cell gradient selector — `src/mhd/emf.hpp:38-42`

```cpp
KOKKOS_INLINE_FUNCTION Real UpwindEMFGradient(const Real mass_flux, const Real left,
                                              const Real right) {
  const int sign = (mass_flux > 0.0) - (mass_flux < 0.0);
  return 0.5 * ((1.0 + sign) * left + (1.0 - sign) * right);
}
```

`mass_flux > 0 -> left`; `< 0 -> right`; `== 0 -> arithmetic mean`. Branch-free, which
matters because this is called four times per edge.

## 4. Upwind edge EMF — `src/mhd/emf.hpp:49-143` (verbatim)

```cpp
template <int E1, Coordinates G, typename PACK, typename GEO>
KOKKOS_INLINE_FUNCTION Real UpwindEMF(const PACK &v, const GEO &vg,
                                      const geometry::CoordParams &cpars, const int block,
                                      const int k, const int j, const int i) {
  constexpr int E2 = E1 == X1DIR ? X2DIR : X1DIR;
  constexpr int E3 = E1 == X3DIR ? X2DIR : X3DIR;
  constexpr int e2_bcomp = E3 - 1;
  constexpr int e3_bcomp = E2 - 1;
  constexpr int e2_sign = 1 - 2 * (E1 % 2);
  constexpr int e3_sign = -e2_sign;
  const std::array<int, 3> pp{k, j, i};
  auto pm = pp;
  auto qm = pp;
  auto mm = pp;
  pm[3 - E2] -= 1;
  qm[3 - E3] -= 1;
  mm[3 - E2] -= 1;
  mm[3 - E3] -= 1;

  // ... geometry::Coords objects cpp/cpm/cqm/cmm, cell centers xpp/xpm/xqm/xmm,
  //     bounds xlo_pp / xlo_pm / xhi_pm / xlo_qm / xhi_qm / xhi_mm,
  //     face scale factors ha_* (direction E2) and hb_* (direction E3)  [lines 68-93]

  const Real ea_lo = e2_sign *
                     v.flux(block, E2, field::cell::B(e2_bcomp), qm[0], qm[1], qm[2]) /
                     ha_qm[e2_bcomp];
  const Real ea_hi = e2_sign *
                     v.flux(block, E2, field::cell::B(e2_bcomp), pp[0], pp[1], pp[2]) /
                     ha_pp[e2_bcomp];
  const Real eb_lo = e3_sign *
                     v.flux(block, E3, field::cell::B(e3_bcomp), pm[0], pm[1], pm[2]) /
                     hb_pm[e3_bcomp];
  const Real eb_hi = e3_sign *
                     v.flux(block, E3, field::cell::B(e3_bcomp), pp[0], pp[1], pp[2]) /
                     hb_pp[e3_bcomp];

  const Real emm = CellEMF<E1>(v, block, mm[0], mm[1], mm[2]);
  const Real epm = CellEMF<E1>(v, block, pm[0], pm[1], pm[2]);
  const Real eqm = CellEMF<E1>(v, block, qm[0], qm[1], qm[2]);
  const Real epp = CellEMF<E1>(v, block, pp[0], pp[1], pp[2]);

  constexpr int ia = E2 - 1;
  constexpr int ib = E3 - 1;
  const Real ga_lo_left = (eb_lo - emm) / (hb_mm[ib] * (xhi_mm[ib] - xmm[ib]));
  const Real ga_lo_right = (eb_hi - eqm) / (hb_qm[ib] * (xhi_qm[ib] - xqm[ib]));
  const Real ga_hi_left = (epm - eb_lo) / (hb_pm[ib] * (xpm[ib] - xlo_pm[ib]));
  const Real ga_hi_right = (epp - eb_hi) / (hb_pp[ib] * (xpp[ib] - xlo_pp[ib]));
  const Real ga_lo =
      UpwindEMFGradient(v.flux(block, E2, gas::cons::density(), qm[0], qm[1], qm[2]),
                        ga_lo_left, ga_lo_right);
  const Real ga_hi =
      UpwindEMFGradient(v.flux(block, E2, gas::cons::density(), pp[0], pp[1], pp[2]),
                        ga_hi_left, ga_hi_right);

  const Real gb_lo_left = (ea_lo - emm) / (ha_qm[ia] * (xhi_mm[ia] - xmm[ia]));
  const Real gb_lo_right = (ea_hi - epm) / (ha_pp[ia] * (xhi_pm[ia] - xpm[ia]));
  const Real gb_hi_left = (eqm - ea_lo) / (ha_qm[ia] * (xqm[ia] - xlo_qm[ia]));
  const Real gb_hi_right = (epp - ea_hi) / (ha_pp[ia] * (xpp[ia] - xlo_pp[ia]));
  const Real gb_lo =
      UpwindEMFGradient(v.flux(block, E3, gas::cons::density(), pm[0], pm[1], pm[2]),
                        gb_lo_left, gb_lo_right);
  const Real gb_hi =
      UpwindEMFGradient(v.flux(block, E3, gas::cons::density(), pp[0], pp[1], pp[2]),
                        gb_hi_left, gb_hi_right);

  const Real da_lo = ha_qm[ib] * (xhi_qm[ib] - xqm[ib]);
  const Real da_hi = ha_pp[ib] * (xpp[ib] - xlo_pp[ib]);
  const Real db_lo = hb_pm[ia] * (xhi_pm[ia] - xpm[ia]);
  const Real db_hi = hb_pp[ia] * (xpp[ia] - xlo_pp[ia]);
  return 0.25 * ((ea_lo + da_lo * ga_lo) + (ea_hi - da_hi * ga_hi) +
                 (eb_lo + db_lo * gb_lo) + (eb_hi - db_hi * gb_hi));
}
```

### 4a. Index decoding

Worked for `E1 = X3DIR` (the 2D case), which is the one to reason about first:

| Symbol | Value |
| --- | --- |
| `E2`, `E3` | `X1DIR`, `X2DIR` |
| `e2_bcomp`, `e3_bcomp` | `1` (B2), `0` (B1) |
| `e2_sign`, `e3_sign` | `-1`, `+1` |
| `pp` / `pm` / `qm` / `mm` | `(k,j,i)` / `(k,j,i-1)` / `(k,j-1,i)` / `(k,j-1,i-1)` |
| `ia`, `ib` | `0` (x1), `1` (x2) |

so the four face EMFs surrounding the edge at the `(i-1/2, j-1/2)` corner are

```
ea_lo = -F1(B2)|(k, j-1, i)      ea_hi = -F1(B2)|(k, j, i)      [x1 faces, offset in x2]
eb_lo = +F2(B1)|(k, j, i-1)      eb_hi = +F2(B1)|(k, j, i)      [x2 faces, offset in x1]
```

and the four cell EMFs are `emm = Ez(k,j-1,i-1)`, `epm = Ez(k,j,i-1)`,
`eqm = Ez(k,j-1,i)`, `epp = Ez(k,j,i)`.

`ga_*` are transverse derivatives along `x2`, used to shift the **x1-face** EMFs to the
corner, upwind-selected by the **x1 mass flux at that same x1 face**. `gb_*` are
derivatives along `x1` shifting the **x2-face** EMFs, selected by the x2 mass flux.
`da_*` / `db_*` are the half-cell distances from face centroid to edge.

### 4b. Cartesian reduction (what RIOT implements)

In Cartesian every scale factor is 1 and every half-width equals `da_*`/`db_*`, so the
`Δ` in each gradient **cancels exactly against the `da`/`db` prefactor**. The kernel
collapses to differences of EMFs with no geometry at all:

```
E_edge = 0.25 * [  ea_lo + ea_hi + eb_lo + eb_hi
                 + upwind( F_a(rho)|qm ;  eb_lo - emm ,  eb_hi - eqm )
                 - upwind( F_a(rho)|pp ;  epm  - eb_lo,  epp  - eb_hi )
                 + upwind( F_b(rho)|pm ;  ea_lo - emm ,  ea_hi - epm )
                 - upwind( F_b(rho)|pp ;  eqm  - ea_lo,  epp  - ea_hi ) ]
```

where `F_a` is the mass flux in direction `E2` and `F_b` in direction `E3`, and
`upwind(f; L, R)` is `UpwindEMFGradient`. This is Gardiner & Stone (2005) Eq. 51: each
of the four face EMFs is extrapolated a half cell to the edge with a donor-cell
transverse derivative, then the four are averaged. The `0.25 * half-difference` is the
`1/8` of the standard GS form.

**No cell widths, no scale factors, no coordinate objects appear.** This is a large
simplification over the donor and removes the entire `geometry::Coords` dependency from
the EMF kernel.

### 4c. A curvilinear inconsistency deliberately NOT carried over

Verified by reading `emf.hpp:126-129`. The `gb_*` denominators pair scale factors and
coordinate differences from **different** cells:

- `gb_lo_left` uses `ha_qm[ia]` with `mm`-based differences `(xhi_mm[ia] - xmm[ia])`
- `gb_lo_right` uses `ha_pp[ia]` with `pm`-based differences `(xhi_pm[ia] - xpm[ia])`
- `gb_hi_right` uses `ha_pp[ia]` with `pp`-based differences

Compare `ga_lo_left`, which correctly pairs `hb_mm` with `mm` differences. In Cartesian
all `h = 1`, so this is exact and harmless; in curvilinear geometry it is an O(Δ)
inconsistency. Writing the Cartesian form directly avoids importing a latent defect
that would only surface if geometry support were added later.

**If curvilinear support is ever added, re-derive these metric factors from the Stokes
line integral. Do not transcribe them.**

---

## 5. Edge EMF assembly and loop extents — `src/mhd/emf.hpp:147-227`

The stored quantity is `h_edge * E_physical`, so multiplying by `Δq` in the CT update
yields `E · L_edge`. In Cartesian `h_edge = 1` and the stored value is the EMF itself.

Extents (all on `IndexDomain::interior`), and the reduced-dimension shortcuts:

| Case | Element | Extents | Expression |
| --- | --- | --- | --- |
| multi_d | E3 | `kb.s..kb.e`, `jb.s..jb.e+1`, `ib.s..ib.e+1` | `h3e * UpwindEMF<X3DIR>` |
| 1D | E3 | `kb.s..kb.e`, `jb.s..jb.e`, `ib.s..ib.e+1` | `-h3e * F1(B2) / hx1[1]` |
| three_d | E2 | `kb.s..kb.e+1`, `jb.s..jb.e`, `ib.s..ib.e+1` | `h2e * UpwindEMF<X2DIR>` |
| 2D | E2 | `kb.s..kb.e`, `jb.s..jb.e`, `ib.s..ib.e+1` | `+h2e * F1(B3) / hx1[2]` |
| three_d & multi_d | E1 | `kb.s..kb.e+1`, `jb.s..jb.e+1`, `ib.s..ib.e` | `h1e * UpwindEMF<X1DIR>` |
| 2D & multi_d | E1 | `kb.s..kb.e`, `jb.s..jb.e+1`, `ib.s..ib.e` | `-h1e * F2(B3) / hx2[2]` |

The reduced-dimension branches matter: **in 1D the transverse components still evolve,
and in 2D the out-of-plane component still evolves.** In those cases there is no
transverse direction to upwind along, so the EMF is just the (signed, rescaled) face
induction flux. Dropping these branches would silently freeze physically evolving field
components.

---

## 6. Constrained-transport face update — `artemis_integrator.hpp:157-259`

Stage coefficients are applied exactly as in the cell-centered update:
`g0 = gam0[stage-1]`, `g1 = gam1[stage-1]`, `beta_dt = beta[stage-1] * dt`.

X1 faces, loop `kb.s..kb.e, jb.s..jb.e, ib.s..ib.e+1` (verbatim):

```cpp
Real &v0n = v0(b, TE::F1, field::face::B(), k, j, i);
Real &v1n = v1(b, TE::F1, field::face::B(), k, j, i);
v0n = g0 * v0n + g1 * v1n;

const Real bdt = beta_dt / coords.GetFaceAreaX1()[0];
const Real dq2m = bnds.x2[1]    - bnds.x2[0];      // coords    (k, j, i)
const Real dq2p = bnds_kp.x2[1] - bnds_kp.x2[0];   // coords_kp (k+three_d, j, i)
const Real dq3m = bnds.x3[1]    - bnds.x3[0];
const Real dq3p = bnds_jp.x3[1] - bnds_jp.x3[0];   // coords_jp (k, j+multi_d, i)

v0n -= bdt * ((dq3p * v0.flux(b, TE::E3, field::face::B(), k, j + multi_d, i) -
               dq3m * v0.flux(b, TE::E3, field::face::B(), k, j, i)) +
              (dq2m * v0.flux(b, TE::E2, field::face::B(), k, j, i) -
               dq2p * v0.flux(b, TE::E2, field::face::B(), k + three_d, j, i)));
```

X2 faces, loop `kb.s..kb.e, jb.s..jb.e+multi_d, ib.s..ib.e`:

```cpp
const Real bdt = beta_dt / coords.GetFaceAreaX2()[0];
v0n -= bdt * ((dq3m * v0.flux(b, TE::E3, field::face::B(), k, j, i) -
               dq3p * v0.flux(b, TE::E3, field::face::B(), k, j, i + 1)) +
              (dq1p * v0.flux(b, TE::E1, field::face::B(), k + three_d, j, i) -
               dq1m * v0.flux(b, TE::E1, field::face::B(), k, j, i)));
```

X3 faces, loop `kb.s..kb.e+three_d, jb.s..jb.e, ib.s..ib.e`:

```cpp
const Real bdt = beta_dt / coords.GetFaceAreaX3()[0];
v0n -= bdt * ((dq2p * v0.flux(b, TE::E2, field::face::B(), k, j, i + 1) -
               dq2m * v0.flux(b, TE::E2, field::face::B(), k, j, i)) +
              (dq1m * v0.flux(b, TE::E1, field::face::B(), k, j, i) -
               dq1p * v0.flux(b, TE::E1, field::face::B(), k, j + multi_d, i)));
```

### 6a. Independent re-derivation of the signs

These implement `d_t (B_n A_f) = -\oint E · dl` with right-handed orientation:

```
d_t(B1 A1) = -[ E3 L3 |_{j+1} - E3 L3 |_j ] + [ E2 L2 |_{k+1} - E2 L2 |_k ]
d_t(B2 A2) = -[ E1 L1 |_{k+1} - E1 L1 |_k ] + [ E3 L3 |_{i+1} - E3 L3 |_i ]
d_t(B3 A3) = -[ E2 L2 |_{i+1} - E2 L2 |_i ] + [ E1 L1 |_{j+1} - E1 L1 |_j ]
```

Checked against all three code blocks above (note the X2 and X3 blocks carry the
`-(...)` through `v0n -=`, which flips the apparent sign of the second bracket). This
is the cyclic pattern `d_t B_a = -(d_b E_c - d_c E_b)`.

### 6b. Cartesian reduction

`A_f` and all `Δq` are uniform constants, so `bdt = beta_dt / (dx_b dx_c)` and each
`Δq` is the corresponding constant edge length. The X1 update becomes

```
B1 <- g0 B1 + g1 B1_stage
      - beta_dt * [ (E3|_{j+1} - E3|_j) / dx2  -  (E2|_{k+1} - E2|_k) / dx3 ]
```

and cyclically. **Why the flux-balance form is still the one to implement:** the
shared-edge contributions must cancel *algebraically*, not just to roundoff. Writing it
as differences of the same stored edge value guarantees that two neighbouring faces
subtract exactly the same number, which is what makes `div B` preserved at roundoff
rather than merely small. Verified as an operator identity by unit test U04.

### 6c. Register-copy requirement

`v0n = g0*v0n + g1*v1n` reads the `u1` stage register on **every** face, including the
last plane in the normal direction. RIOT's `DeepCopyData` walks a `TE::CC` index space
and would truncate that plane — hence `sparse_update::DeepCopyFaceData` (ADR-003,
Hazard B). A truncated copy is invisible in a first RK stage and corrupts the second.

---

## 7. Riemann solver interface

Unified donor signature (`riemann/riemann.hpp:25-36`), with `mu0` and `do_mhd` added
by this branch:

```cpp
operator()(const EOS &eos, const Real c, const Real chat, const Real mu0,
           const bool do_mhd, parthenon::team_mbr_t const &member, const int b,
           const int k, const int j, const int il, const int iu, const int dir,
           const parthenon::ScratchPad2D<Real> &wl,
           const parthenon::ScratchPad2D<Real> &wr, const V1 &p, const V2 &q,
           const V3 &vf) const;
```

RIOT does not adopt this shape: its solvers are free functions taking scalars and
returning the max signal speed (`src/hydro/riemann.hpp:117`). The port extends that
signature instead, following the `StrengthFluxes` precedent
(`src/hydro/calculate_fluxes.cpp:228-380`).

### Primitive scratch index layout (drives all donor Riemann code)

Prim pack order is `density, velocity(3), pressure, sie, bmod, field::cell::B(3),
field::cell::energy`, so with `n` species:

```
IDN  = n_index                      IPR  = 4n + n_index      ISE  = 5n + n_index
ivx/ivy/ivz = n + 3*n_index + cyclic(dir)                    IBL  = 6n + n_index
IBX  = 7n + (dir-1)                 IBY = 7n + ((dir-1)+1)%3  IBZ = 7n + ((dir-1)+2)%3
IBM  = 7n + 3   (== field::cell::energy slot: the FACE MAGNETIC PRESSURE)
IBXG = dir-1, IBYG = dir%3, IBZG = (dir+1)%3   (global B indices for flux writes)
```

### Two structural choices the CT layer depends on

1. **Magnetic pressure is not folded into the normal-momentum flux.** It is written to a
   separate face register (`IBM`) and applied as a gradient alongside gas pressure in
   `fluid_fluxes.hpp:451-491`:
   ```cpp
   Real Pl = vp_.flux(b, d1, IPR, k, j, i)
           + (mhd ? vp_.flux(b, d1, field::cell::energy(), k, j, i) : 0.0);
   Real Pr = vp_.flux(b, d1, IPR, k, j, i + 1)
           + (mhd ? vp_.flux(b, d1, field::cell::energy(), k, j, i + 1) : 0.0);
   vc_(b, IMX, k, j, i) += dtdx[0] * (Pl - Pr);
   ```
   The dual-energy `PdV` term uses **only** `IPR`, never the magnetic part.
2. **The normal-B flux slot is written as exactly `0.0`**, so EMF assembly cannot pick
   up a spurious normal contribution.

Additionally, **the mass flux must be retained at every face** — it is the only
auxiliary the GS upwinding needs (no stored face velocities, no upwind flags).

## 8. Fast magnetosonic speed — `src/mhd/mhd.hpp:26-41`

```cpp
KOKKOS_FORCEINLINE_FUNCTION Real FastMagnetosonicSpeed(const Real bulk,
                                                       const Real density, const Real b2,
                                                       const Real bx, const Real mu0) {
  const Real wave_sum = (bulk + b2 / mu0) / density;
  const Real discriminant =
      std::max(0.0, SQR(wave_sum) - 4.0 * bulk * SQR(bx) / (mu0 * SQR(density)));
  return std::sqrt(0.5 * (wave_sum + std::sqrt(discriminant)));
}
```

`bulk` is the **bulk modulus** `K = rho (dp/drho)_s` (`gas::prim::bmod`), i.e. the
general-EOS acoustic derivative — *not* `gamma*p`. So no constant-gamma assumption
enters, which is why this transfers to RIOT unchanged (`ccbulk::bulk_modulus`).
Ported verbatim to `src/mhd/mhd_helpers.hpp`; verified by unit test U03 against the
residual of the defining quartic.

## 9. Timestep — `src/gas/gas.cpp:543-643`

```cpp
const Real ss = std::sqrt((bulk + b2 / mu0_code) / dens);
Real denom = 0.0;
for (int d = 0; d < ndim; d++)
  denom += (std::abs(vmesh(b, gas::prim::velocity(VI(n, d)), k, j, i)) + ss) / dx[d];
ldt = std::min(ldt, 1.0 / denom);
```

Uses the direction-independent **bound** `sqrt(a^2 + vA^2)` from cell-centered B, not
the true directional fast speed — deliberately, per the donor's own comment, because a
bound is what CFL safety needs and is far cheaper. RIOT's CFL is already this same
directional-**sum** convention (`src/hydro/hydro.cpp:719-727`), so only
`c_s^2 -> c_f^2` changes. Exposed as `MHD::SignalSpeedBound`.

## 10. Face-to-cell B — `src/mhd/mhd.hpp:55-78`

Centroid-weighted linear interpolation, which **reduces exactly to the arithmetic mean
in Cartesian**; degenerate directions take the single available face value rather than
zeroing the component. Ported as `MHD::FaceToCellB`; three unit tests pin the
conventions, including the collapsed-direction case.

## 11. Normal-B substitution in reconstruction — `reconstruction/reconstruction.hpp:111-121`

```cpp
if (do_mhd) {
  const int n = q.GetIndex(b, field::cell::B(dir - 1));
  TE fd = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);
  parthenon::par_for_inner(
      DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
        const int ipl = i + (dir == 1);
        ql(n, ipl) = qc(b, fd, field::face::B(), k + (dir == 3), j + (dir == 2), ipl);
        qr(n, i)   = qc(b, fd, field::face::B(), k, j, i);
      });
}
```

So `wl(IBX,i) == wr(IBX,i) == B_face(i)` exactly — **one** normal-B value shared by both
states, never reconstructed twice. The donor achieves this by threading a `skip_index`
through all five reconstruction methods to skip the normal component, then overwriting
here.

**RIOT deviation**: reconstruct all three components, then overwrite the normal one from
the shared face value. Numerically identical (the skipped values were overwritten
anyway) and avoids touching five RIOT reconstruction methods.

## 12. Flux-bound extension — `utils/fluxes/fluid_fluxes.hpp:75-94`

```cpp
template <int DIR>
void ExtendMHDFluxBounds(const bool multi_d, const bool three_d, int &il, int &iu,
                         int &jl, int &ju, int &kl, int &ku) {
  if constexpr (DIR == X1DIR) {
    jl -= multi_d;  ju += multi_d;  kl -= three_d;  ku += three_d;
  } else if constexpr (DIR == X2DIR) {
    il -= 1;  iu += 1;  kl -= three_d;  ku += three_d;
  } else if constexpr (DIR == X3DIR) {
    il -= 1;  iu += 1;  jl -= 1;  ju += 1;
  }
}
```

One extra **transverse** layer of face fluxes, which is exactly what the GS stencil
needs (it reaches `pm`/`qm`/`mm`, one cell back in both transverse directions). Note it
is directional, never extending along the sweep direction — that is what keeps
`nghost = 2` viable.

## 13. HLLD robustness constants — `riemann/hlld.hpp`

Preserve these verbatim when porting in Stage 5; they are the difference between a
solver that survives low-beta states and one that produces NaNs:

| Guard | Value / condition | Line |
| --- | --- | --- |
| Vanishing normal field -> full HLLE fallback | `abs(bxi) <= 1e-12` | 203 |
| Degenerate denominator | `abs(denom) <= 1e-20` | 203 |
| Non-finite wave speeds | `!isfinite(sl) \|\| !isfinite(sr)` | 204 |
| Coincident speeds | `abs(sl-sm) <= small \|\| abs(sr-sm) <= small` | 226 |
| Invalid star state | `dlst <= 0 \|\| drst <= 0 \|\| ptst <= 0` | 232-233 |
| Rotational degeneracy tolerance | `deg_tol = 1e-4 * ptst` | 238-268 |
| Double-star branch cutoff | `0.5 * bxsq_n < deg_tol` | 289 |
| Non-finite intermediate energy | bail to HLLE | 321-323 |

HLLD uses an internal `B/sqrt(mu0)` normalization and **requires an ideal gas**
(`PARTHENON_REQUIRE(do_mhd, ...)` at line 54); it is restricted accordingly in RIOT.
Its species-`n >= 1` HLLC branch (lines 427-545) is not ported — single material.

---

## Provenance summary

| Donor location | RIOT destination | Status |
| --- | --- | --- |
| `mhd/mhd.hpp:21-24, 26-41, 55-78` | `src/mhd/mhd_helpers.hpp` | ported, unit-tested |
| `mhd/emf.hpp:22-42` | `src/mhd/emf.hpp` | pending (Stage 4) |
| `mhd/emf.hpp:49-143` | `src/mhd/emf.hpp` | pending — Cartesian form §4b, not a transcription |
| `mhd/emf.hpp:147-227` | `MHD::AssembleEdgeEMF` | pending (Stage 4) |
| `artemis_integrator.hpp:157-259` | `MHD::ApplyFaceUpdate` | pending (Stage 4) |
| `artemis_integrator.hpp:30-90` (face branch) | `sparse_update::DeepCopyFaceData` | ported |
| `riemann/hlle.hpp` | `src/mhd/riemann_mhd.hpp` | pending (Stage 3) |
| `riemann/hlld.hpp`, `llf.hpp` | `src/mhd/riemann_mhd.hpp` | pending (Stage 5) |
| `reconstruction/reconstruction.hpp:111-121` | `src/hydro/` recon path | pending (Stage 3), deviation §11 |
| `fluid_fluxes.hpp:75-94, 451-491` | `src/hydro/calculate_fluxes.cpp` | pending (Stage 3) |
| `gas/gas.cpp:543-643` | `Hydro::EstimateTimestepMesh` | pending (Stage 3) |
| `refinement/{prolongation,restriction}.hpp` | — | **skipped**: RIOT's Parthenon already provides them |
| `geometry/*` edge scale factors | — | **deferred**: curvilinear out of scope; see §4c |
