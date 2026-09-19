# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# This file was made in part with generative AI.

# Circularly polarized Alfven wave (Toth 2000; Gardiner & Stone 2005 section 5.2): the ORDER OF
# ACCURACY test, and the only one in this suite that would notice a scheme which is stable,
# conservative, divergence-free and merely FIRST order -- which is what a subtly wrong EMF
# average or reconstruction produces. Every other MHD test here checks either an identity that
# holds at roundoff or agreement at a single resolution.
#
# The oracle is analytic and exact. This is an exact NONLINEAR solution, so at finite amplitude
# (0.1 here) there is no linearization error to contaminate it -- unlike a linear eigenmode test,
# where a scheme can be second order in the linearized system and lose it at finite amplitude.
# tlim is one full period (v_A = b_par/sqrt(rho) = 1, wavelength = box length), so the exact
# solution at the final time IS the initial condition, and the error is |final - initial| with no
# analytic evaluator and no interpolation.
#
# WHY THE DIRECTION SWEEP. mhd_cpaw/wave_dir rotates the wave onto each axis so every face
# direction carries the nonuniform components in turn, and travel_sign reverses propagation,
# which catches an induction sign error that a standing pattern would hide. The three axes were
# measured to give L1 identical in every printed digit (TEST_LEDGER G4.5) -- exact rotational
# isotropy -- so the sweep is asserted as an equality between directions, which is a far sharper
# criterion than each direction independently passing a convergence threshold.
#
# LIMITATION, stated rather than glossed: this is an AXIS-ALIGNED wave. A wave along a box
# diagonal additionally couples all three EMF components and is a strictly stronger test; it is
# not covered here (concern C9).

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from scripts.utils.mhd_analysis import Checks, assemble, field_snapshot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

input_id = "mhd/cpaw"

# Resolutions along the wave direction. The transverse extents stay at the deck's 4 cells over
# 0.125: the solution is uniform across them, so resolving them buys nothing and only costs wall
# time, but they are kept NONDEGENERATE so the run is genuinely 3D and all the transverse face
# exchanges and EMF components are live.
resolutions = (16, 32, 64)

# Frozen: the donor's own linwave convergence criterion, ratio <= 0.35, corresponding to observed
# order >= 1.51. Measured here: 1.55 -> 1.73 -> 1.83, rising toward 2 (TEST_LEDGER G4.5).
#
# This tolerance is only meaningful for second-order-ish reconstruction, and that is a property of
# the TIME integrator, not a caveat about the test: with rk2 at fixed CFL the error is
# A*dx^p + B*dt^2, so once the spatial term is small the temporal one floors it. weno5 and mp5
# would "fail" this at the default CFL for exactly that reason while being far MORE accurate --
# which is why this test pins plm rather than sweeping reconstruction. Reconstruction coverage is
# G5.5's job.
ratio_tol = 0.35
tol_eta = 1.0e-12
# |B| is uniform in the exact solution, so a growing spread is direct evidence the scheme is not
# preserving the circular polarization -- a failure the L1 norm alone averages away. plm at these
# resolutions holds it to ~1e-3 of |B|; the bound is deliberately loose because this criterion
# exists to catch a qualitative loss of polarization, not to measure it.
tol_bmag_spread = 1.0e-2
tol_bpar_drift = 1.0e-12
# The three axes agreed to every printed digit, so this is an equality assertion with only enough
# slack to survive a different summation order in the L1 reduction.
tol_isotropy = 1.0e-12

wave_dirs = (1, 2, 3)


def _tag(n, wave_dir):
    return "cpaw_d{:d}_n{:d}".format(wave_dir, n)


def _mesh_args(n, wave_dir):
    """Resolution n along `wave_dir`, 4 cells across the other two.

    The LONG axis must move with wave_dir: setting nx1 while the wave travels along x2 would
    change the aspect ratio instead of rotating the test. The meshblock is half the long axis so
    the wave crosses a block boundary, and matches the transverse extents exactly -- Parthenon
    aborts with "Block size is not evenly divisible into the base mesh size" otherwise.
    """
    args = []
    for d in (1, 2, 3):
        if d == wave_dir:
            args += [
                "parthenon/mesh/nx{:d}={:d}".format(d, n),
                "parthenon/mesh/x{:d}min=0.0".format(d),
                "parthenon/mesh/x{:d}max=1.0".format(d),
                "parthenon/meshblock/nx{:d}={:d}".format(d, n // 2),
            ]
        else:
            args += [
                "parthenon/mesh/nx{:d}=4".format(d),
                "parthenon/mesh/x{:d}min=0.0".format(d),
                "parthenon/mesh/x{:d}max=0.125".format(d),
                "parthenon/meshblock/nx{:d}=4".format(d),
            ]
    return args


def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    for wave_dir in wave_dirs:
        for n in resolutions:
            riot.mpirun(
                1,
                input_id + ".rin",
                [
                    "parthenon/job/problem_id=" + _tag(n, wave_dir),
                    "mhd_cpaw/wave_dir={:d}".format(wave_dir),
                    "parthenon/time/ncycle_out=100000",
                ]
                + _mesh_args(n, wave_dir),
            )


def _perp_indices(wave_dir):
    """Perpendicular component indices in cyclic order, matching the generator: wave along 1 ->
    (2,3), 2 -> (3,1), 3 -> (1,2), returned zero-based."""
    return {1: (1, 2), 2: (2, 0), 3: (0, 1)}[wave_dir]


def _measure(tag, wave_dir):
    f0, f1 = field_snapshot(tag, "00000"), field_snapshot(tag, "final")
    b0 = assemble(f0, "c.c.bulk.magnetic_field", 3)
    b1 = assemble(f1, "c.c.bulk.magnetic_field", 3)
    v0 = assemble(f0, "c.c.bulk.velocity", 3)
    v1 = assemble(f1, "c.c.bulk.velocity", 3)
    divb = assemble(f1, "c.c.bulk.div_magnetic_field")

    pa, pb = _perp_indices(wave_dir)
    # Resolution along the wave direction, read off the assembled global shape (comp, nk, nj, ni)
    # so it needs no argument and cannot disagree with what was actually run.
    n_along = {1: b1.shape[3], 2: b1.shape[2], 3: b1.shape[1]}[wave_dir]

    # L1 over the four wave components, normalized by the amplitude so the number is
    # dimensionless and comparable across resolutions and directions.
    amp = np.abs(b0[pa]).max()
    err = (
        np.mean(np.abs(b1[pa] - b0[pa]))
        + np.mean(np.abs(b1[pb] - b0[pb]))
        + np.mean(np.abs(v1[pa] - v0[pa]))
        + np.mean(np.abs(v1[pb] - v0[pb]))
    )
    bmag0 = np.sqrt((b0**2).sum(axis=0))
    bmag1 = np.sqrt((b1**2).sum(axis=0))
    return {
        "n": n_along,
        "l1": err / (4.0 * amp),
        "eta": np.abs(divb).max() * (1.0 / n_along) / bmag0.mean(),
        "bmag_spread": (bmag1.max() - bmag1.min()) / bmag0.mean(),
        # The component along the wave direction is uniform and constant in the exact solution.
        "bpar_drift": np.abs(b1[wave_dir - 1] - b0[wave_dir - 1]).max() / bmag0.mean(),
    }


def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    per_dir = {}

    for wave_dir in wave_dirs:
        rows = [_measure(_tag(n, wave_dir), wave_dir) for n in resolutions]
        per_dir[wave_dir] = rows
        c = Checks(logger, "wave_dir {:d}".format(wave_dir))
        for idx, r in enumerate(rows):
            c.le("N={:d} eta(divB)".format(r["n"]), r["eta"], tol_eta)
            c.le("N={:d} |B| spread".format(r["n"]), r["bmag_spread"], tol_bmag_spread)
            c.le("N={:d} b_par drift".format(r["n"]), r["bpar_drift"], tol_bpar_drift)
            if idx > 0:
                ratio = r["l1"] / rows[idx - 1]["l1"]
                order = np.log2(rows[idx - 1]["l1"] / r["l1"])
                # The fitted order is logged alongside the ratio deliberately: a ratio that
                # passes at 0.34 is a scheme barely above first order and is worth looking at
                # even though it passed.
                c.record(
                    "L1 ratio N={:d}/{:d}".format(r["n"], rows[idx - 1]["n"]),
                    ratio,
                    ratio <= ratio_tol,
                    "<= {:.2f}  (L1 {:.6e}, observed order {:.3f})".format(
                        ratio_tol, r["l1"], order
                    ),
                )
        analyze_status &= c.ok

    # Rotational isotropy: the error must not depend on which axis the wave travels along.
    c = Checks(logger, "rotational isotropy across wave_dir")
    reference = per_dir[wave_dirs[0]]
    for wave_dir in wave_dirs[1:]:
        for idx, r in enumerate(per_dir[wave_dir]):
            ref = reference[idx]["l1"]
            c.le(
                "N={:d} |L1(dir {:d}) - L1(dir {:d})| / L1".format(
                    r["n"], wave_dir, wave_dirs[0]
                ),
                abs(r["l1"] - ref) / ref,
                tol_isotropy,
            )
    analyze_status &= c.ok

    return bool(analyze_status)
