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

# Regression test: analytic Fourier mode diffusion for electron thermal conduction

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot

file_id = "ionization/conduction_analytic"
logger = logging.getLogger("riot" + __name__[7:])  # set logger name

# fourier mode numbers for problem
ndim = 3
nx = 2
ny = 2 * (ndim > 1)
nz = 2 * (ndim > 2)

nblock = 4
err_tol = 1e-4
tlim = 1.0e-4
dt = 1e-6
max_ranks = 8

# 3d
if ndim > 2:
    # resolutions = np.array([32, 48, 64, 96, 128, 192, 256], dtype=np.int32)
    resolutions = np.array([32], dtype=np.int32)
elif ndim > 1:
    # resolutions = np.array([32, 48, 64, 96, 128, 192, 256], dtype=np.int32)
    resolutions = np.array([64], dtype=np.int32)
else:
    # resolutions = np.array([32, 48, 64, 96, 128, 192, 256], dtype=np.int32)
    # resolutions = np.array([128], dtype=np.int32)
    resolutions = np.array([64], dtype=np.int32)

block_resolutions = resolutions // nblock


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(file_id + ".py")
    for i in range(len(resolutions)):
        logger.debug(f"Runnning test {__name__}; resolution = {resolutions[i]}")

        resx = resolutions[i]
        resy = resolutions[i] if ndim > 1 else 1
        resz = resolutions[i] if ndim > 2 else 1

        blockresx = block_resolutions[i]
        blockresy = block_resolutions[i] if ndim > 1 else 1
        blockresz = block_resolutions[i] if ndim > 2 else 1

        ylim = 5 if ndim > 1 else 0.5
        zlim = 5 if ndim > 2 else 0.5

        dt_use = dt

        arguments = [
            f"parthenon/mesh/nx1={resx}",
            f"parthenon/mesh/nx2={resy}",
            f"parthenon/mesh/nx3={resz}",
            f"parthenon/mesh/x2min={-ylim}",
            f"parthenon/mesh/x2max={ylim}",
            f"parthenon/mesh/x3min={-zlim}",
            f"parthenon/mesh/x3max={zlim}",
            f"parthenon/meshblock/nx1={blockresx}",
            f"parthenon/meshblock/nx2={blockresy}",
            f"parthenon/meshblock/nx3={blockresz}",
            f"parthenon/time/tlim={tlim}",
            f"parthenon/time/dt_force={dt_use}",
            "parthenon/time/ncycle_out=100",
            f"parthenon/output1/dt=-1",
            f"conduction_analytic/nx={nx}",
            f"conduction_analytic/ny={ny}",
            f"conduction_analytic/nz={nz}",
        ]

        nranks = min(max_ranks, int(nblock**ndim))
        riot.mpirun(nranks, file_id + ".rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # Load data
    nx, err = np.loadtxt(
        "build/src/conduction_analytic-errs.dat", dtype=np.float64, usecols=[0, 4]
    ).T
    if isinstance(err, np.ndarray):
        err = err[-1]
    if err > err_tol:
        logger.warning(
            f"error tolerance exceeded in conduction_analytic 3d verification; err = {err:23.15e}"
        )
        analyze_status = False
    return analyze_status
