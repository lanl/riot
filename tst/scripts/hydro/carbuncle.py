# ========================================================================================
#  (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

# Regression test checking for checking absence of carbuncle instability

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

dtout = 0.35
nranks = 8
problem_id = "noh"
input_id = "noh.rin"
riemann = ["chllc", "lhllc"]

# This appears to be related to the maximum thread team size on
# V100s...
nx1_cpu = 512
nx1_gpu = 256

# HLLC has a value of about 1000.
CARBUNCLE_THRESH = 200


def find_block(f):
    xmin = 0
    xmax = 0.15
    ymin = 0
    ymax = 0.15
    upper_right_quadrant = np.logical_and(f.x[:, 0] >= xmin, f.y[:, 0] >= ymin)
    too_far = np.logical_or(f.x[:, 0] > xmax, f.y[:, 0] > ymax)
    mask = np.logical_and(upper_right_quadrant, np.logical_not(too_far))
    b = np.where(mask)[0][0]
    return b


def get_carbuncle_detector(f, field, b):
    cmin = 0.02
    cmax = 0.085

    dx = f.xf[b, 1] - f.xf[b, 0]
    dy = f.yf[b, 1] - f.yf[b, 0]

    dfdx = np.zeros_like(field)
    dfdx[b, ..., :-1] = (field[b, ..., 1:] - field[b, ..., :-1]) / dx

    dfdy = np.zeros_like(field)
    dfdy[b, ..., :-1, :] = (field[b, ..., 1:, :] - field[b, ..., :-1, :]) / dy

    Z, Y, X = f.GetVolumeLocations(False)

    detector = np.abs(dfdx) * np.logical_and(
        np.logical_and(Y >= cmin, Y <= cmax), X <= cmin
    ) + np.abs(dfdy) * np.logical_and(np.logical_and(X >= cmin, X <= cmax), Y <= cmin)

    return detector


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("noh.py")
    logger.debug("Running test " + __name__)
    nx1 = nx1_gpu if kwargs["gpu"] else nx1_cpu
    my_nranks = 1 if kwargs["gpu"] else nranks
    for solver in riemann:
        job_id = f"{problem_id}_{solver}"
        arguments = [
            "parthenon/job/problem_id=" + job_id,
            "parthenon/time/tlim=" + str(dtout),
            "parthenon/time/ncycle_out=250",
            "parthenon/meshblock/nx1=" + str(nx1),
            "parthenon/output1/dt=" + str(dtout),
            "hydro/riemann=" + solver,
        ]
        riot.mpirun(my_nranks, input_id, arguments)


def analyze():
    logger.debug("Analyzing test " + __name__)
    var = "c.c.bulk.rho"

    analyze_status = True
    for solver in riemann:
        f = phdf(f"build/src/{problem_id}_{solver}.out1.final.phdf")
        q = f.Get(var, False)
        b = find_block(f)
        detector = get_carbuncle_detector(f, q, b)
        if detector.max() > CARBUNCLE_THRESH:
            logger.warning(
                "Carbuncle instability detected for Riemann solver "
                + solver
                + " with max instability gradient {:g}".format(detector.max())
            )
            analyze_status = False
    return analyze_status
