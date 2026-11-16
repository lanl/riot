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

# Regression test based on the advection of a weak magnetic field loop (Gardiner & Stone
# 2005) with adaptive mesh refinement on the field. Checks that CT keeps div B at
# round-off across refinement boundaries and through refinement / derefinement, that
# AMR is active, and that the numerical dissipation of the magnetic energy is bounded.

# Modules
import logging
import os
import numpy as np
import scripts.utils.riot as riot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
problem_id = "field_loop_aniso"
# Measured ME(t=1) / ME(0) is 0.874 at this resolution
ME_RATIO_MIN = 0.8


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(
        "mhd/field_loop.py", ["--nx", "64", "--ny", "16", "--tlim", "1.0", "--amr"]
    )
    # Parthenon appends to an existing history file, so remove one left by a previous
    # run in a reused build directory
    hst = "build/src/" + problem_id + ".out0.hst"
    if os.path.exists(hst):
        os.remove(hst)
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + problem_id,
        "parthenon/output1/dt=-1.0",
    ]
    riot.mpirun(nranks, "mhd/field_loop.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # time, dt, cycle, nbtotal, ME, max_divB_dx
    hst = np.loadtxt("build/src/" + problem_id + ".out0.hst", ndmin=2)
    if np.isnan(hst).any():
        logger.warning("NaN encountered")
        return False
    time, nblocks, me, divb = hst[:, 0], hst[:, 3], hst[:, 4], hst[:, 5]
    if abs(time[-1] - 1.0) > 1.0e-2:
        logger.warning("Final history time {0:g} != 1".format(time[-1]))
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
    return analyze_status
