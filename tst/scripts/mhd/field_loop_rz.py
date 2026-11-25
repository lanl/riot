# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

#
# Regression test based on the advection of a weak poloidal field loop in cylindrical
# (r, z) coordinates with adaptive mesh refinement on the field. Checks that CT keeps
# div B at round-off with the cylindrical edge lengths and face areas across refinement
# boundaries, that AMR is active, that the loop is advected to the right place, and that
# the numerical dissipation of the magnetic energy is bounded. Only meaningful in a
# PARTHENON_COORDINATES=UniformCylindrical build; skipped otherwise.

# Modules
import logging
import os
import numpy as np
import scripts.utils.riot as riot
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
problem_id = "field_loop_rz"
TLIM = 0.5
# Measured ME(t=0.5) / ME(0) is 0.941
ME_RATIO_MIN = 0.9


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("mhd/field_loop_rz.py", ["--tlim", str(TLIM), "--amr"])
    # Parthenon appends to an existing history file, so remove one left by a previous
    # run in a reused build directory
    hst = "build/src/" + problem_id + ".out0.hst"
    if os.path.exists(hst):
        os.remove(hst)
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + problem_id,
        "parthenon/output1/dt=" + str(TLIM),
    ]
    riot.mpirun(nranks, "mhd/field_loop_rz.rin", arguments)


# B_r^2 + B_z^2 weighted centroid of the loop in (r, z)
def loop_centroid(data):
    b = data.Get("c.c.bulk.bfield", flatten=False)
    w = b[:, 0] ** 2 + b[:, 1] ** 2
    rc = np.array([0.5 * (xf[1:] + xf[:-1]) for xf in data.xf])
    zc = np.array([0.5 * (yf[1:] + yf[:-1]) for yf in data.yf])
    vol = np.array(
        [
            np.outer(yf[1:] - yf[:-1], 0.5 * (xf[1:] ** 2 - xf[:-1] ** 2))
            for xf, yf in zip(data.xf, data.yf)
        ]
    )[:, None]
    w = w * vol
    r = (w * rc[:, None, None, :]).sum() / w.sum()
    z = (w * zc[:, None, :, None]).sum() / w.sum()
    return r, z


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    data0 = phdf("build/src/" + problem_id + ".out1.00000.phdf")
    if data0.Coordinates != "UniformCylindrical":
        logger.warning("Skipping " + __name__ + " in a " + data0.Coordinates + " build")
        return True
    analyze_status = True
    # time, dt, cycle, nbtotal, ME, max_divB_dx
    hst = np.loadtxt("build/src/" + problem_id + ".out0.hst", ndmin=2)
    if np.isnan(hst).any():
        logger.warning("NaN encountered")
        return False
    time, nblocks, me, divb = hst[:, 0], hst[:, 3], hst[:, 4], hst[:, 5]
    if abs(time[-1] - TLIM) > 1.0e-2:
        logger.warning("Final history time {0:g} != {1:g}".format(time[-1], TLIM))
        analyze_status = False
    if divb.max() > 1.0e-12:
        logger.warning("max div B dx too large: {0:g}".format(divb.max()))
        analyze_status = False
    if nblocks.min() == nblocks.max():
        logger.warning("Number of blocks never changed; AMR not exercised")
        analyze_status = False
    me_ratio = me[-1] / me[0]
    if not (ME_RATIO_MIN < me_ratio <= 1.0):
        logger.warning(
            "Magnetic energy ratio {0:g} outside ({1:g}, 1]".format(
                me_ratio, ME_RATIO_MIN
            )
        )
        analyze_status = False
    # v_z = 1, so the loop moves from z = 0 to z = TLIM at fixed r
    r0, z0 = loop_centroid(data0)
    r1, z1 = loop_centroid(phdf("build/src/" + problem_id + ".out1.final.phdf"))
    if abs(r1 - r0) > 0.01 or abs(z1 - z0 - TLIM) > 0.01:
        logger.warning(
            "Loop centroid moved from ({0:g}, {1:g}) to ({2:g}, {3:g})".format(
                r0, z0, r1, z1
            )
        )
        analyze_status = False
    return analyze_status
