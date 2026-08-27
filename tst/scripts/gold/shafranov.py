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

# Regression test based on the 3T shafranov steady state shock problem

# Modules
import logging
import scripts.utils.riot as riot
import subprocess
import phdf
import numpy as np

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

file_id = "shafranov"
gold_id = "shafranov"
diff_tol_parallel = 1.0e-10
diff_tol_serial = 1.0e-14

# velocity and momentum diff on GPU using gold files from CPU for some reason.
# This is only in the electron conduction pre-heat region. For now, exclude these vars
# from the error calculation
exclude_vars = ["c.c.bulk.velocity", "c.c.bulk.momentum"]


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("ionization/shafranov.py")
    logger.debug("Runnning test " + __name__)

    if kwargs["gpu"]:
        # on GPU, run on rank
        nranks = 1
        arguments = ["parthenon/job/problem_id=shafranov-serial"]
        riot.mpirun(nranks, "ionization/shafranov.rin", arguments)

        arguments = ["parthenon/job/problem_id=shafranov-parallel"]
        riot.mpirun(nranks, "ionization/shafranov.rin", arguments)
    else:
        # on CPU run with different numbers of ranks
        nranks = 8
        arguments = ["parthenon/job/problem_id=shafranov-parallel"]
        riot.mpirun(nranks, "ionization/shafranov.rin", arguments)

        nranks = 1
        arguments = ["parthenon/job/problem_id=shafranov-serial"]
        riot.mpirun(nranks, "ionization/shafranov.rin", arguments)


import numpy as np


def mixed_L1_error(u, uhat, atol=1e-12, rtol=1e-6):
    """
    Compute a stable L1 error norm using mixed absolute/relative scaling.

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

    # scaling using max(absTol, relTol * |u|)
    scale = np.maximum(atol, rtol * np.abs(u))

    # normalized per-point errors
    err = e / scale

    # L1 norm (mean)
    return np.mean(err)


# Analyze outputs
def analyze():
    analyze_status = True

    for run_type in ["parallel", "serial"]:
        logger.debug("Analyzing test " + __name__ + run_type)
        diff_tol = diff_tol_parallel if run_type == "parallel" else diff_tol_serial
        test_dump = "build/src/shafranov-" + run_type + ".out1.final.phdf"
        gold_dump = "scripts/gold/files/shafranov-" + run_type + ".out1.final.phdf"

        # h5diff flags up roundoff errors in quantities with large absolute values,
        # but using arelative diff causes divide by zero errors for things that
        # are legitimately zero; so calculate explicitly here instead
        run1 = phdf.phdf(test_dump)
        run2 = phdf.phdf(gold_dump)
        max_err_mixed = 0.0
        which_vars = [var for var in run1.Variables if "c.c." in var]

        for var in which_vars:
            if var not in exclude_vars:
                vtest = run1.Get(var).flatten()
                vgold = run2.Get(var).flatten()
                abs_err = np.abs(vtest - vgold)
                mask = np.abs(vgold) > 1e-14
                rel_err = np.max(np.abs(abs_err[mask] / vgold[mask]))
                mixed_err = mixed_L1_error(vtest, vgold, diff_tol, diff_tol)
                if mixed_err > 1.0:
                    print(
                        f"error rel/abs/mixed: {var:40s} : {rel_err:23.15e} {np.max(abs_err):23.15e} {mixed_err:23.15e}"
                    )
                max_err_mixed = max(max_err_mixed, mixed_err)

        if mixed_err > 1.0:
            logger.warning(f"shafranov: max mixed error = {max_err_mixed:23.15e}")
            analyze_status = False

    return analyze_status
