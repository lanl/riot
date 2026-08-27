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
import logging
import scripts.utils.riot as riot
import subprocess

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

file_id = "triple"
gold_id = "triple"
amr = False
sparse = False
nranks = 1
# JMM: M2 Max seems to disagree with all other architectures at 1.15e-12
# PDM: Move to reconstructing material internal energy increments tol to 2.0e-11
diff_tol = 2.0e-11


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("triple/triple.py")
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/time/tlim=5.0",
        "parthenon/time/nlim=2000",
        "parthenon/time/integrator=rk2",
        "parthenon/mesh/nx1=112" "parthenon/mesh/nx2=48",
        "parthenon/mesh/nx3=1",
        "parthenon/meshblock/nx1=16",
        "parthenon/meshblock/nx2=16",
        "parthenon/meshblock/nx3=1",
        "hydro/recon=plm",
        # "hydro/riemann=hllc",
        "parthenon/output1/dt=6.0",
        "parthenon/output2/dt=-1.0",
    ]
    args = arguments + [
        "parthenon/job/problem_id=" + file_id,
        "parthenon/mesh/refinement=" + ("adaptive" if amr else "none"),
        "parthenon/mesh/sparse_init=" + str(sparse).lower(),
        "materials/sparse_init=" + str(sparse).lower(),
        "materials/sparse_dealloc=" + str(sparse).lower(),
    ]
    riot.mpirun(nranks, "triple/triple.rin", args)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    h5diff_result = subprocess.getoutput(
        "h5diff -d " + str(diff_tol) + " -c "
        "--exclude-attribute /Info "
        "--exclude-attribute /Input "
        "--exclude-path /Blocks "
        "--exclude-path /Params "
        "build/src/" + file_id + ".out1.final.phdf "
        "scripts/gold/files/" + gold_id + ".out1.final.phdf"
    )
    if len(h5diff_result) > 0:
        logger.warning(h5diff_result)
        analyze_status = False
    return analyze_status
