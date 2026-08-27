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

# Regression test based on the multiple material triple point problem

# Modules
import importlib
import logging
import scripts.utils.riot as riot
import scripts.gold.triple as triple

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

importlib.reload(triple)
triple.file_id = "triple_amr_sparse_parallel"
triple.gold_id = "triple_amr"
triple.amr = True
triple.sparse = True
triple.nranks = 8


# Run riot
def run(**kwargs):
    return triple.run(**kwargs)


# Analyze outputs
def analyze():
    return triple.analyze()
