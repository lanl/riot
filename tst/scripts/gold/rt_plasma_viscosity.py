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

# Single mode 2d planar Rayleigh-Taylor instability with plasma viscosity

# Modules
import logging
import scripts.utils.riot as riot
import subprocess
import phdf
import numpy as np

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

file_id = "rt_plasma_viscosity"
gold_id = "rt_plasma_viscosity"
diff_tol = 1.0e-8

# velocity and momentum diff on GPU using gold files from CPU for some reason.
# This is only in the electron conduction pre-heat region. For now, exclude these vars from the error calculation
# exclude_vars = ["c.c.bulk.velocity", "c.c.bulk.momentum"]
exclude_vars = []


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("ionization/rt_plasma_viscosity.py")
    logger.debug("Runnning test " + __name__)

    if kwargs["gpu"]:
        # on GPU, run on rank
        nranks = 1
        arguments = ["parthenon/job/problem_id=rt_plasma_viscosity"]
        riot.mpirun(nranks, "ionization/rt_plasma_viscosity.rin", arguments)
    else:
        # on CPU run with different numbers of ranks
        nranks = 6
        arguments = ["parthenon/job/problem_id=rt_plasma_viscosity"]
        riot.mpirun(nranks, "ionization/rt_plasma_viscosity.rin", arguments)


import numpy as np


def mixed_L1_error(u, uhat, atol=1e-12, rtol=1e-6):
    """Compute a stable L1 error norm using mixed absolute/relative scaling.

    Divides by a block- and component-local magnitude to help avoid
    "phase" errors where the local error is a slight phase shift in a
    zero-crossing.

    Parameters
    ----------
    u : array_like
        “True” or reference solution.
    uhat : array_like
        Approximate or numerical solution to compare.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.

    Returns
    -------
    float
        Normalized L1 error (≈1 means errors are at tolerance).
    """
    u = np.asarray(u)
    uhat = np.asarray(uhat)

    # pointwise errors
    e = np.abs(u - uhat)

    u_max_local = np.abs(u).max(axis=-1)

    # scaling using max(absTol, relTol * |u|)
    scale = np.maximum(atol, rtol * u_max_local)

    # normalized per-point errors
    err = e / scale[..., np.newaxis]

    # L1 norm (mean)
    return np.mean(err)


# Analyze outputs
def analyze():
    analyze_status = True

    logger.debug("Analyzing test " + __name__)
    test_dump = "build/src/rt_plasma_viscosity.out1.final.phdf"
    gold_dump = "scripts/gold/files/rt_plasma_viscosity.out1.final.phdf"

    # h5diff flags up roundoff errors in quantities with large absolute values,
    # but using arelative diff causes divide by zero errors for things that
    # are legitimately zero; so calculate explicitly here instead
    run1 = phdf.phdf(test_dump)
    run2 = phdf.phdf(gold_dump)
    max_err_mixed = 0.0
    which_vars = [var for var in run1.Variables if "c.c." in var]

    for var in which_vars:
        if var not in exclude_vars:
            vtest = run1.Get(var, flatten=True)
            vgold = run2.Get(var, flatten=True)
            abs_err = np.abs(vtest - vgold)
            mask = np.abs(vgold) > diff_tol
            if np.count_nonzero(mask) > 0:
                rel_err = np.max(np.abs(abs_err[mask] / vgold[mask]))
                mixed_err = mixed_L1_error(vtest, vgold, diff_tol, diff_tol)
                if mixed_err > 1.0:
                    print(
                        f"error rel/abs/mixed: {var:40s} : {rel_err:23.15e} {np.max(abs_err):23.15e} {mixed_err:23.15e}"
                    )
                max_err_mixed = max(max_err_mixed, mixed_err)

    if max_err_mixed > 1.0:
        logger.warning(f"rt_plasma_viscosity: max mixed error = {max_err_mixed:23.15e}")
        analyze_status = False

    return analyze_status
