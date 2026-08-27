# ========================================================================================
#  (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import sys
import os

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1
input_id = "advection/advection_trivial"


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    arguments = []
    riot.mpirun(nranks, input_id + ".rin", arguments)


def analyze():
    import h5py

    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    with h5py.File("build/src/trivial_advection.out1.final.phdf", "r") as f:
        data = f["bulk_tied"][:]
        if not np.all(data == 1.0):
            logger.warning("Trivial advection test failed. Data not 1 everywhere.")
            analyze_status = False

    return analyze_status
