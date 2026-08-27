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

# Regression test based on the 1D Hohlraum problem

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import subprocess
import sys

sys.path.insert(
    0,
    "../external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools",
)
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

res_survey = [1, 3]
c = 1.0
tf = 1.0 / c
err_tol_imp = 7.0e-3  # should change if res_survey modified
conv_tol_imp = 0.4  # should change if res_survey modified


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("radiation_transport/hohlraum.py")
    logger.debug("Runnning test " + __name__)
    args = [
        "parthenon/time/tlim=" + str(tf),
        "parthenon/time/nlim=1000",
        "parthenon/time/integrator=rk1",
        "parthenon/mesh/nx1=128",
        "parthenon/mesh/nx2=1",
        "parthenon/mesh/nx3=1",
        "parthenon/meshblock/nx1=128",
        "parthenon/meshblock/nx2=1",
        "parthenon/meshblock/nx3=1",
        "parthenon/mesh/ix2_bc=periodic",
        "parthenon/mesh/ox2_bc=periodic",
        "parthenon/mesh/x1min=0.0",
        "parthenon/mesh/x1max=1.0",
        "parthenon/mesh/x2min=-0.5",
        "parthenon/mesh/x2max=0.5",
        "parthenon/mesh/refinement=none",
        "parthenon/output1/dt=1.0",
        "radiation_transport/cfl=0.5",
        "radiation_transport/coupling=false",
        "radiation_transport/drive/ix1_bc=drive",
        "radiation_transport/drive/ix2_bc=default",
        "physics/hydro=true",
        "physics/fixed_fluid=true",
    ]
    for res in res_survey:
        args_imp = args + [
            "parthenon/job/problem_id=hohlraum_imp_nlevel" + str(res),
            "radiation_transport/nlevel=" + str(res),
            "radiation_transport/do_jacobi=True",
            "radiation_transport/do_explicit=False",
            "radiation_transport/jacobi/dt_ratio_hyperbolic=5.0",
            "radiation_transport/jacobi/niter_limit=200",
        ]
        riot.mpirun(1, "radiation_transport/hohlraum.rin", args_imp)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # Simulation solutions
    urad_imp = []
    for res in res_survey:
        data_imp = phdf("build/src/hohlraum_imp_nlevel" + str(res) + ".out1.final.phdf")
        urad_imp.append(data_imp.Get("c.c.rad.moments", False, False)[0, 0, 0, 0])

    # Analytic solution
    xc = 0.5 * (data_imp.xng[0, 1:] + data_imp.xng[0, :-1])
    rtt = 0.5 * (1.0 - xc / (c * tf))
    rtt[xc > c * tf] = 0.0
    # L1 errors
    l1_imp = []
    dx = 1.0 / 128
    for i in range(len(res_survey)):
        l1_imp.append(np.sum(np.abs(urad_imp[i] - rtt) * dx))

    # Check failure modes
    if l1_imp[1] > err_tol_imp:
        logger.warning(
            "Hohlraum 1D error too large with implicit algorithm, "
            "error: {0:g} threshold: {1:g}".format(l1_imp[1], err_tol_imp)
        )
        analyze_status = False
    if l1_imp[1] / l1_imp[0] > conv_tol_imp:
        logger.warning(
            "Hohlraum 1D not converging with implicit algorithm, "
            "conv: {0:g} threshold: {1:g}".format(l1_imp[1] / l1_imp[0], conv_tol_imp)
        )
        analyze_status = False
    return analyze_status
