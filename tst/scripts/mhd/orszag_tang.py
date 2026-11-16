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

# Regression test based on the Orszag-Tang vortex. The initial conditions are invariant
# under a 180 degree rotation about the domain center, which the CT scheme preserves to
# round-off. Also checks that div B stays at round-off.

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
nx = 64
problem_id = "orszag_tang"


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("mhd/orszag_tang.py", ["--nx", str(nx), "--tlim", "0.5"])
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + problem_id,
        "parthenon/meshblock/nx1=" + str(nx),
        "parthenon/meshblock/nx2=" + str(nx),
        "parthenon/output1/dt=0.5",
    ]
    riot.mpirun(nranks, "mhd/orszag_tang.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    data = phdf("build/src/" + problem_id + ".out1.final.phdf")
    if abs(data.Time - 0.5) > 1.0e-10:
        logger.warning("Final time {0:g} != 0.5".format(data.Time))
        return False
    # (comp,) j, i of the single block and k = 0
    fields = {
        "rho": (data.Get("c.c.bulk.rho", False)[0, 0], 1.0),
        "pressure": (data.Get("c.c.bulk.pressure", False)[0, 0], 1.0),
    }
    v = data.Get("c.c.bulk.velocity", False)[0, :, 0]
    b = data.Get("c.c.bulk.bcc", False)[0, :, 0]
    for c in range(2):
        fields["v" + str(c + 1)] = (v[c], -1.0)
        fields["b" + str(c + 1)] = (b[c], -1.0)
    for name, (a, sign) in fields.items():
        if np.isnan(a).any():
            logger.warning("NaN encountered in " + name)
            return False
        # Rotation by pi about the center maps cell (j, i) to (n-1-j, n-1-i)
        err = np.abs(a - sign * a[::-1, ::-1]).max() / np.abs(a).max()
        if err > 1.0e-10:
            logger.warning("{0} breaks rotational symmetry by {1:g}".format(name, err))
            analyze_status = False
    divb = np.abs(data.Get("c.c.bulk.divb", False)).max() / nx
    if divb > 1.0e-12:
        logger.warning("div B dx too large: {0:g}".format(divb))
        analyze_status = False
    rho = fields["rho"][0]
    if rho.min() <= 0.0 or fields["pressure"][0].min() <= 0.0:
        logger.warning("Non-positive density or pressure")
        analyze_status = False
    return analyze_status
