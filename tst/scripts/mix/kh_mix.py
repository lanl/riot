# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

# Smoke/regression test that the BHR mix model runs.
#
# There is not yet a verification test for mix/BHR.  Until one exists, this test
# guards against mix silently breaking: it runs the Kelvin-Helmholtz problem with
# mix enabled for a handful of cycles and checks that
#   1. the run completes,
#   2. the BHR mix variables are present in the output,
#   3. those variables stay finite, and
#   4. the source-driven mix quantities (rho_bhr_a, rho_bhr_ST, rho_bhr_SD),
#      which start at zero, have become nonzero -- i.e. the mix source terms
#      are actually being integrated.

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
problem_id = "kh_mix"
input_id = "kh.rin"
nlim = 20

# BHR mix variables that must be present in the output when mix is on.
mix_vars = [
    "c.c.bulk.rho_bhr_a",
    "c.c.bulk.rho_bhr_b",
    "c.c.bulk.rho_bhr_ST",
    "c.c.bulk.rho_bhr_SD",
    "c.c.bulk.rho_reynolds_stress",
]

# Subset that starts at zero and is driven purely by the mix source terms, so a
# nonzero final value is direct evidence that mix was integrated.
source_driven_vars = [
    "c.c.bulk.rho_bhr_a",
    "c.c.bulk.rho_bhr_ST",
    "c.c.bulk.rho_bhr_SD",
]


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("kh.py")
    logger.debug("Running test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + problem_id,
        # Enable mix regardless of how kh.py was generated.
        "physics/mix=True",
        "parthenon/time/nlim=" + repr(nlim),
        # Dump only the final state; we just need to inspect the mix fields.
        "parthenon/output1/dt=0.0",
    ]
    riot.mpirun(nranks, input_id, arguments)


def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    f = phdf(f"build/src/{problem_id}.out1.final.phdf")

    # 1/2. Mix variables must be present.
    for var in mix_vars:
        if var not in f.Variables:
            logger.warning("Mix variable {0} missing from output".format(var))
            analyze_status = False
    if not analyze_status:
        logger.warning(
            "BHR mix variables not found -- mix does not appear to be enabled"
        )
        return False

    # 3. All mix variables must remain finite.
    for var in mix_vars:
        q = f.Get(var, False)
        if not np.all(np.isfinite(q)):
            logger.warning("Mix variable {0} contains non-finite values".format(var))
            analyze_status = False

    # 4. Source-driven quantities start at zero; a nonzero value means the mix
    #    source terms were actually integrated.
    for var in source_driven_vars:
        q = f.Get(var, False)
        maxabs = np.nanmax(np.abs(q))
        if not maxabs > 0.0:
            logger.warning(
                "Mix source-driven variable {0} is identically zero; "
                "mix source terms do not appear to be active".format(var)
            )
            analyze_status = False

    return analyze_status
