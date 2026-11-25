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

# Regression test based on the Brio & Wu MHD shock tube. The tube is run along each of
# X1, X2 and X3 on equivalent 3D grids; the three runs are cyclic permutations of each
# other and must agree to round-off (checks the HLLD rotations and the CT EMF signs).

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
nx = 128
directions = [1, 2, 3]


def problem_id(d):
    return "brio_wu_x" + str(d)


# Run riot
def run(**kwargs):
    for d in directions:
        logger.debug("Generating input " + __name__ + " dir " + str(d))
        riot.generate("mhd/brio_wu.py", ["--dir", str(d), "--nx", str(nx), "--full3d"])
        logger.debug("Runnning test " + __name__ + " dir " + str(d))
        arguments = [
            "parthenon/job/problem_id=" + problem_id(d),
            "parthenon/output0/dt=-1.0",
        ]
        riot.mpirun(nranks, "mhd/brio_wu.rin", arguments)


# Profiles along the tube: rho, v_n, v_t, B_n, B_t, with t the cyclic successor of n
def load(d):
    data = phdf("build/src/" + problem_id(d) + ".out1.final.phdf")
    n = d - 1
    t = (n + 1) % 3
    # Arrays are (block, [comp,] k, j, i); profiles are (k, j, i) with the tube axis last
    axis = 2 - n

    def profile(a):
        return np.moveaxis(a, axis, -1).reshape(-1, nx)

    rho = profile(data.Get("c.c.bulk.rho", False)[0])
    v = data.Get("c.c.bulk.velocity", False)[0]
    b = data.Get("c.c.bulk.bfield", False)[0]
    divb = data.Get("c.c.bulk.divb", False)
    return (
        rho,
        profile(v[n]),
        profile(v[t]),
        profile(b[n]),
        profile(b[t]),
        np.abs(divb).max(),
    )


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    names = ["rho", "v_n", "v_t", "B_n", "B_t"]
    ref = load(directions[0])
    rho, vn, vt, bn, bt, divb = ref
    for name, a in zip(names, ref[:5]):
        if np.isnan(a).any():
            logger.warning("NaN encountered in " + name)
            return False
        # All transverse cells must be identical
        if np.abs(a - a[0]).max() > 1.0e-12:
            logger.warning("Transverse variation in " + name)
            analyze_status = False
    if divb > 1.0e-10:
        logger.warning("div B too large: {0:g}".format(divb))
        analyze_status = False
    if np.abs(bn - 0.75).max() > 1.0e-12:
        logger.warning("Normal field not preserved")
        analyze_status = False
    # Coarse check of the solution structure (Brio & Wu 1988, fig. 2): densities stay
    # within the initial states, the compound wave drives v_t to about -1.6, and the
    # tangential field reverses between the left and right states.
    if rho.min() < 0.1 or rho.max() > 1.0 + 1.0e-6:
        logger.warning(
            "Density out of range [{0:g}, {1:g}]".format(rho.min(), rho.max())
        )
        analyze_status = False
    if not (-1.8 < vt.min() < -1.3):
        logger.warning("Unexpected minimum v_t: {0:g}".format(vt.min()))
        analyze_status = False
    if abs(bt[0, 0] - 1.0) > 1.0e-6 or abs(bt[0, -1] + 1.0) > 1.0e-6:
        logger.warning("Tangential field boundary states changed")
        analyze_status = False
    for d in directions[1:]:
        other = load(d)
        for name, a, b in zip(names, ref[:5], other[:5]):
            err = np.abs(a - b).max()
            if err > 1.0e-10:
                logger.warning("X{0} and X1 {1} differ by {2:g}".format(d, name, err))
                analyze_status = False
    return analyze_status
