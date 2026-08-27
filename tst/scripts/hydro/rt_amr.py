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

# Regression test checking for symmetry of Rayleigh-Taylor instability

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
import h5py
from compute_asymmetry import compute_asymmetry

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 8
problem_id = "rt_symmetry_amr"
input_id = "rt/rt_amr"


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    job_id = problem_id
    arguments = [
        "parthenon/job/problem_id=" + job_id,
        "parthenon/time/ncycle_out=500",
    ]
    # 4^2 meshblocks, which are the CPU run are pathologically small
    # this runs a larger test on larger blocks which paradoxically
    # runs much faster on GPU
    if kwargs["gpu"]:
        arguments += [
            "parthenon/mesh/nx1=48",
            "parthenon/mesh/nx2=24",
            "parthenon/meshblock/nx1=16",
            "parthenon/meshblock/nx2=12",
        ]
    # Multiple ranks per GPU actually slow this problem down
    my_nranks = 1 if kwargs["gpu"] else nranks
    riot.mpirun(my_nranks, input_id + ".rin", arguments)


def analyze():
    logger.debug("Analyzing test " + __name__)
    var = "c.c.mat.rho_0"

    analyze_status = True
    with h5py.File(f"build/src/{problem_id}.out1.final.phdf", "r") as f:
        var_diff = compute_asymmetry(f, var)
        maxdiff = np.max(np.abs(var_diff))
        if maxdiff > 1e-10:
            logger.warning(
                "Rayleigh-Taylor instability not y-symmetric "
                + "with maximum error {:g}".format(maxdiff)
            )
            analyze_status = False
    return analyze_status
