# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

# Regression test based on Newtonian linear advection of sharp material square
# with a level set convergence problem

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

integrators = ["rk2"]
recon = ["plm"]
modcyc = ["1", "999999"]
nranks = 1
file_id = "levelset_advection"

output_vars = [
    "c.c.bulk.rho",
    "c.c.bulk.velocity",
    "c.c.mat.volume_fraction",
    "c.c.bulk.pressure",
    "levelset.levelset",
]


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("levelsets/" + file_id + ".py")
    logger.debug("Runnning test " + __name__)
    for iv in integrators:
        for rv in recon:
            for mc in modcyc:
                for res in (16, 64):
                    pname = "levelset_advection_{:s}_{:s}_{:s}_{:d}".format(
                        iv, rv, mc, res
                    )
                    arguments = [
                        "parthenon/job/problem_id=" + pname,
                        "parthenon/time/integrator=" + iv,
                        "parthenon/mesh/nx1=" + repr(res),
                        "parthenon/mesh/nx2=" + repr(res),
                        "hydro/recon=" + rv,
                        "levelsets/reinit_modcyc=" + mc,
                    ]
                    riot.mpirun(nranks, "levelsets/" + file_id + ".rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    run_dir = "build/src"
    analyze_status = True
    # tolerance when levelset is advected and reinitialized
    reinit_tol = 1.0e-2
    # tolerance when levelset is advected and reinitialized
    advect_tol = 5.0e-2

    errs = []
    # calculate errors
    for iv in integrators:
        for rv in recon:
            for mc in modcyc:
                for res in (16, 64):
                    pname = "levelset_advection_{:s}_{:s}_{:s}_{:d}".format(
                        iv, rv, mc, res
                    )
                    data0 = phdf("{:s}/{:s}.out1.00000.phdf".format(run_dir, pname))
                    data1 = phdf("{:s}/{:s}.out1.final.phdf".format(run_dir, pname))
                    lset0 = data0.Get("levelset.levelset", False, False)[:, 0, :, :]
                    lset1 = data1.Get("levelset.levelset", False, False)[:, 0, :, :]
                    if int(mc) >= 1000:
                        error = np.sum(np.abs(lset1 - lset0)) / res**2 / advect_tol
                    else:
                        error = np.sum(np.abs(lset1 - lset0)) / res**2 / reinit_tol

                    errs.append(error)

    if np.max(errs) > 1.0:
        logger.warning(
            f"Levelset advection and reinitialization error too large: "
            f"err: {maxerr:5.2e} tolerance: {tol:5.2e}"
        )
        analyze_status = False

    return analyze_status
