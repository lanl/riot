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

# Regression test for gravitational acceleration

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

resx = 270
resy = 270

bsx = 90
bsy = 90

nranks = 8
problem_id = "compression"
input_id = "adiabatic_compression"


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    job_id = f"{problem_id}"
    arguments = [
        "parthenon/job/problem_id=" + job_id,
        "parthenon/mesh/nx1=" + repr(resx),
        "parthenon/mesh/nx2=" + repr(resy),
        "parthenon/meshblock/nx1=" + repr(bsx),
        "parthenon/meshblock/nx2=" + repr(bsy),
    ]
    riot.mpirun(nranks, input_id + ".rin", arguments)


def get_vol(dataset):
    alpha = dataset.Get("c.c.mat.volume_fraction_1", False, True)[:, 0, 0]
    dy = (dataset.yf[:, 1:] - dataset.yf[:, :-1])[:, 0]
    dx = (dataset.xf[:, 1:] - dataset.xf[:, :-1])[:, 0]
    dA = (dy * dx)[:, np.newaxis, np.newaxis]
    vol = (alpha * dA).sum()
    return vol


def cyl_vol(r):
    return np.pi * (r**2)


def compute_de(r_i, r_f):
    L = 2
    Gamma1 = 4.0 / 3.0
    Gamma2 = 4.0 / 3.0

    P_start = 1

    vol2_start = cyl_vol(r_i)
    vol1_start = L**2 - vol2_start

    vol2_end = cyl_vol(r_f)
    vol1_end = L**2 - vol2_end

    rho1_start = 1
    rho2_start = 1 * rho1_start

    K_2 = P_start / rho2_start**Gamma2

    M1 = rho1_start * vol1_start
    M2 = rho2_start * vol2_start

    rho1_end = M1 / vol1_end
    rho2_end = M2 / vol2_end

    e1_start = P_start / ((Gamma1 - 1) * rho1_start)

    P_end = K_2 * rho2_end**Gamma2

    e1_end = P_end / ((Gamma1 - 1) * rho1_end)
    de1 = (e1_end - e1_start) * M1
    return de1


def analyze():
    analyze_status = True
    logger.debug("Analyzing test " + __name__)
    var = "c.c.mat.rho_1"

    init = phdf(f"build/src/{problem_id}.out1.00000.phdf")
    final = phdf(f"build/src/{problem_id}.out1.final.phdf")
    vol_i = get_vol(init)
    vol_f = get_vol(final)
    r_i = np.sqrt(np.abs(vol_i / np.pi))
    r_f = np.sqrt(np.abs(vol_f / np.pi))

    r_ana_i = 0.5
    r_ana_f = 0.25
    dE = compute_de(r_ana_i, r_ana_f)

    # vol_ana_i = cyl_vol(r_i)
    # vol_ana_f = cyl_vol(r_f)

    hst = np.loadtxt(f"build/src/{problem_id}.out0.hst")
    dE_thresh = 1e-4  # set by precision in history output
    dE_measured = hst[-1, 4] - hst[0, 4]
    if np.abs(dE_measured - dE) > dE_thresh:
        logger.warning(
            "Amount of energy added does not agree! "
            f"Expected = {dE}, Measured = {dE_measured}"
        )
        analyze_status = False

    dr_i = np.abs(r_i - r_ana_i)
    if dr_i > 1e-2 * r_ana_i:
        logger.warning(
            "Initial radius does not agree! "
            f"Expected = {r_ana_i}, Measured = {r_i}, Difference = {dr_i}"
        )
        analyze_status = False

    dr_f = np.abs(r_f - r_ana_f)
    if dr_f > 1e-1 * r_ana_f:  # very sensitive to truncation error
        logger.warning(
            "Final radius does not agree! "
            f"Expected = {r_ana_f}, Measured = {r_f}, Difference = {dr_f}"
        )
        analyze_status = False

    hst = np.loadtxt(f"build/src/{problem_id}.out0.hst")
    etot = hst[:, 4]
    detot = etot[-1] - etot[0]

    return analyze_status
