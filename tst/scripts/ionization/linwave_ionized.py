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

# Regression test based on Newtonian hydro linear wave convergence problem

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import scripts.hydro.linwave as linwave

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

linwave.problem_id = "linwave_ionized"
linwave.input_id = "mm_linear_modes_ionized"
linwave.multiple_materials = True
linwave.do_high_order = False
linwave.integrators = ["rk4"]
linwave.recon = ["plm"]


# Run riot
def run(**kwargs):
    return linwave.run(**kwargs)


# Analyze outputs
def analyze():
    return linwave.analyze()
