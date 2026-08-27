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
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
nx1 = 300
nx2 = 100
gravity_dim = 0
x1min = -0.75
x1max = 0.75
x2min = -0.25
x2max = 0.25
problem_id = "rt_symmetry_unigrid"
input_id = "rt/rt_singlemode"
riemann = ["hllc", "chllc"]


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    for solver in riemann:
        job_id = f"{problem_id}_{solver}"
        arguments = [
            "parthenon/job/problem_id=" + job_id,
            "parthenon/mesh/nx1=" + repr(nx1),
            "parthenon/mesh/nx2=" + repr(nx2),
            "parthenon/mesh/x1min=" + repr(x1min),
            "parthenon/mesh/x1max=" + repr(x1max),
            "parthenon/mesh/ix1_bc=reflecting",
            "parthenon/mesh/ox1_bc=reflecting",
            "parthenon/mesh/ix2_bc=periodic",
            "parthenon/mesh/ox2_bc=periodic",
            "parthenon/mesh/x2min=" + repr(x2min),
            "parthenon/mesh/x2max=" + repr(x2max),
            "parthenon/meshblock/nx1=" + repr(nx1),
            "parthenon/meshblock/nx2=" + repr(nx2),
            "gravity/gravity_dim=" + repr(gravity_dim),
            "hydro/riemann=" + solver,
            "parthenon/time/ncycle_out=500",
        ]
        riot.mpirun(nranks, input_id + ".rin", arguments)


def analyze():
    logger.debug("Analyzing test " + __name__)
    var = "c.c.mat.rho_0"

    analyze_status = True
    for solver in riemann:
        f = phdf(f"build/src/{problem_id}_{solver}.out1.final.phdf")
        q = f.Get(var, False)[0, 0, 0]
        ny = q.shape[0]
        diff = q[: ny // 2] - np.flip(q[ny // 2 :], axis=0)
        maxdiff = np.max(np.abs(diff))
        if maxdiff > 1e-9:
            logger.warning(
                "Rayleigh-Taylor instability not y-symmetric for Riemann solver "
                + solver
                + " with maximum error {:g}".format(maxdiff)
            )
            analyze_status = False
    return analyze_status
