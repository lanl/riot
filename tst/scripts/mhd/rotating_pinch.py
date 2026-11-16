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
# Regression test based on a rotating screw-pinch equilibrium in cylindrical (r, z)
# coordinates. The equilibrium is held by the centrifugal and magnetic hoop stress
# geometric source terms, so this checks that they balance the pressure gradient. Fast
# waves launched by the outflow radial boundaries do not reach 0.8 < r < 1.2 by t = 0.1.
# Only meaningful in a PARTHENON_COORDINATES=UniformCylindrical build; skipped otherwise.

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
problem_id = "rotating_pinch"
OMEGA = 1.0
B0 = 0.5
# Measured errors in 0.8 < r < 1.2 at nr = 64 are 2.9e-4 (v_r), 2.5e-5 (v_phi) and
# 8.5e-5 (B_phi); all converge with resolution
TOL_VR = 1.0e-3
TOL_VPHI = 1.0e-4
TOL_BPHI = 3.0e-4


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("mhd/rotating_pinch.py", ["--nr", "64", "--tlim", "0.1"])
    logger.debug("Runnning test " + __name__)
    arguments = ["parthenon/job/problem_id=" + problem_id]
    riot.mpirun(nranks, "mhd/rotating_pinch.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    data = phdf("build/src/" + problem_id + ".out1.final.phdf")
    if data.Coordinates != "UniformCylindrical":
        logger.warning(
            "Skipping " + __name__ + " in a " + data.Coordinates + " build"
        )
        return True
    analyze_status = True
    v = data.Get("c.c.bulk.velocity", flatten=False)
    b = data.Get("c.c.bulk.bcc", flatten=False)
    divb = data.Get("c.c.bulk.divb", flatten=False)
    if np.isnan(v).any() or np.isnan(b).any():
        logger.warning("NaN encountered")
        return False
    r = np.array([0.5 * (xf[1:] + xf[:-1]) for xf in data.xf])
    r = r[:, None, None, :] * np.ones_like(divb)
    inner = (r > 0.8) & (r < 1.2)
    # native components are (r, z, phi)
    errors = {
        "v_r": (np.abs(v[:, 0][inner]).max(), TOL_VR),
        "v_z": (np.abs(v[:, 1]).max(), 1.0e-12),
        "v_phi": (np.abs(v[:, 2] - OMEGA * r)[inner].max(), TOL_VPHI),
        "B_phi": (np.abs(b[:, 2] - B0 * r)[inner].max(), TOL_BPHI),
        "div B": (np.abs(divb).max(), 1.0e-12),
    }
    for name, (err, tol) in errors.items():
        if err > tol:
            logger.warning("{0} error {1:g} > {2:g}".format(name, err, tol))
            analyze_status = False
    return analyze_status
