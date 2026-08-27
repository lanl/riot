# ========================================================================================
#  (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

# Regression test based on Newtonian linear advection of a sharp material cube

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("advection/advection_sharp.py")
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/mesh/nghost=2",
        "parthenon/mesh/nx1=32",
        "parthenon/mesh/nx2=32",
        "parthenon/mesh/nx3=32",
        "parthenon/mesh/x3min=0.0",
        "parthenon/mesh/x3max=1.0",
        "parthenon/meshblock/nx1=16",
        "parthenon/meshblock/nx2=16",
        "parthenon/meshblock/nx3=16",
        "parthenon/time/tlim=1.0",
        "parthenon/output1/dt=2.0",
        "parthenon/time/nlim=300",
        "parthenon/time/ncycle_out=50",
        "region1/z0=0.25",
        "region1/z1=0.75",
        "region0/vz=1",
        "region1/vz=1",
    ]
    riot.mpirun(nranks, "advection/advection_sharp.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    tol = 0.06
    data0 = phdf("build/src/sharp_advection.out1.00000.phdf")
    data1 = phdf("build/src/sharp_advection.out1.final.phdf")
    errors = []
    for var in [
        "mat_tied_0",
        "mat_tied_1",
        "bulk_tied",
    ]:
        d0 = data0.Get(var, False, False)[:, :, :, :]
        d1 = data1.Get(var, False, False)[:, :, :, :]
        errors.append(np.sum(np.abs(d1 - d0)) / 32**3)
    maxerr = np.max(np.array(errors))
    if maxerr > tol:
        logger.warning(
            "3d advection maximum error too large, "
            "error: {0:g} threshold: {1:g}".format(maxerr, tol)
        )
        analyze_status = False
    return analyze_status
