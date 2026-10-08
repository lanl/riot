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

# Orszag & Tang (1979) vortex: the only test in this suite where constrained transport has to
# hold div B = 0 WHILE limiters are firing and the EMF stencil straddles discontinuities. The
# field loop is smooth for all time and Brio-Wu is one-dimensional; neither can catch a CT
# defect that only appears at a shock.
#
# WHY THERE IS NO POINTWISE REFERENCE. The shock network is chaotic: by t = 0.5 two correct
# codes disagree pointwise, so a gold file would freeze this build rather than the physics.
# Every criterion below is instead an exact statement that any correct ideal-MHD scheme must
# satisfy on a periodic box, evaluable with no oracle at all:
#
#   - mean rho is conserved exactly (the density update is a flux divergence that telescopes
#     on a periodic mesh)
#   - mean momentum stays zero (it starts zero, and the initial field and velocity are both
#     odd about the box centre)
#   - mean total energy is conserved -- which is only true if the ADR-002 energy convention is
#     applied consistently, since total_material_energy carries the magnetic part
#   - E - u - KE - E_mag vanishes cell by cell, i.e. the ADR-002 identity itself
#   - max|div B| at roundoff, the CT property under test
#   - positivity, i.e. no floor had to rescue the solve
#
# The shock-heating check exists so a run that somehow stayed smooth cannot pass by being
# trivially well behaved: the mean internal energy must have risen measurably above its initial
# value, which happens only if the shocks actually formed and dissipated.
#
# Thresholds are the DONOR's own, from artemis/tst/scripts/mhd/orszag_tang.py, adopted verbatim.

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from scripts.utils.mhd_analysis import Checks, assemble, field_snapshot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

input_id = "mhd/orszag_tang"

# Frozen setup constants, matching inputs/mhd/orszag_tang.py. These are the standard closed
# forms rho0 = 25/(36 pi), P0 = 5/(12 pi), b0 = 1/sqrt(4 pi).
gamma = 1.66666666667
rho0 = 0.22104853207207686
p0 = 0.13262911924324611
v0 = 1.0
b0 = 0.28209479177387814

# Analytic box means at t = 0. The sin^2 averages are exactly 1/2 on a cell-centred uniform
# grid spanning a whole number of periods, so these are exact for the DISCRETE initial state
# too, not merely to O(dx^2) -- which is what lets them be compared at rtol 1e-12 rather than
# at truncation error.
u0 = p0 / (gamma - 1.0)
e0 = u0 + 0.5 * rho0 * v0**2 + 0.5 * b0**2

# Frozen thresholds, quoted from the donor script.
tol_divb = 1.0e-10
tol_rho_rtol = 1.0e-12
tol_momentum = 1.0e-10
tol_energy_rtol = 1.0e-8
tol_residual = 1.0e-10

# Rank counts. 1 vs 4 was measured BITWISE identical (TEST_LEDGER G4.4); the assertion is at
# 1e-12 because the CFL step is a global reduction whose order can legitimately differ on
# another machine, and whether it was bitwise is logged either way.
nranks = (1, 4)
tol_rank = 1.0e-12


def _tag(n):
    return "ot_r{:d}".format(n)


def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    for n in nranks:
        riot.mpirun(
            n,
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + _tag(n),
                "mhd/monitor_divb=false",
                "parthenon/time/ncycle_out=100000",
            ],
        )


def _fields(tag, snap):
    path = field_snapshot(tag, snap)
    return {
        "rho": assemble(path, "c.c.bulk.rho"),
        "P": assemble(path, "c.c.bulk.pressure"),
        "u": assemble(path, "c.c.bulk.internal_energy"),
        "E": assemble(path, "c.c.bulk.total_material_energy"),
        "emag": assemble(path, "c.c.bulk.magnetic_energy"),
        "divB": assemble(path, "c.c.bulk.div_magnetic_field"),
        "vel": assemble(path, "c.c.bulk.velocity", 3),
        "B": assemble(path, "c.c.bulk.magnetic_field", 3),
    }


def _check(tag, snap, expect_heating):
    """`expect_heating` must be False at t = 0: no shocks have formed, so that one criterion is
    meaningless there. Every OTHER check IS meaningful at t = 0 and is in fact a stronger
    statement then, because the analytic box means are exact for the discrete initial state --
    which makes the t = 0 snapshot a direct validation of the problem generator rather than
    only of the time integration. That is why both snapshots are checked."""
    f = _fields(tag, snap)
    c = Checks(logger, "{} ({})".format(tag, snap))

    c.finite("all finite", *f.values())
    c.gt("min rho", f["rho"].min(), 0.0)
    c.gt("min P", f["P"].min(), 0.0)
    c.le("max|divB|", np.abs(f["divB"]).max(), tol_divb)
    c.close("mean rho", f["rho"].mean(), rho0, tol_rho_rtol)

    # rho is (1, k, j, i) and vel is (3, k, j, i), so the product broadcasts over components.
    mom = (f["rho"] * f["vel"]).mean(axis=(1, 2, 3))
    c.le("max|mean momentum|", np.abs(mom).max(), tol_momentum)

    c.close("mean total energy", f["E"].mean(), e0, tol_energy_rtol)

    ke = 0.5 * f["rho"] * (f["vel"] ** 2).sum(axis=0)
    c.le(
        "max|E - u - KE - Emag|",
        np.abs(f["E"] - f["u"] - ke - f["emag"]).max(),
        tol_residual,
    )

    if expect_heating:
        c.ge("mean internal energy", f["u"].mean(), u0 + 1.0e-3 * e0)
    else:
        logger.debug(
            "  mean internal energy {:.8e} (initial {:.17g}); heating criterion N/A at t=0".format(
                f["u"].mean(), u0
            )
        )
    return c.ok


def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    for n in nranks:
        analyze_status &= _check(_tag(n), "00000", expect_heating=False)
        analyze_status &= _check(_tag(n), "final", expect_heating=True)

    # Rank invariance across the shock network.
    a = _fields(_tag(nranks[0]), "final")
    b = _fields(_tag(nranks[1]), "final")
    c = Checks(logger, "rank invariance, {} vs {} ranks".format(*nranks))
    bitwise = True
    for name in ("rho", "P", "E", "emag", "B", "vel", "divB"):
        scale = max(np.abs(a[name]).max(), 1.0e-300)
        diff = np.abs(a[name] - b[name])
        bitwise &= bool((diff == 0.0).all())
        c.le("rel diff {}".format(name), diff.max() / scale, tol_rank)
    logger.debug(
        "  {} vs {} ranks was {}bitwise identical".format(
            nranks[0], nranks[1], "" if bitwise else "NOT "
        )
    )
    analyze_status &= c.ok

    return bool(analyze_status)
