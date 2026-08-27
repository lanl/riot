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

# Regression test based on the Be shell problem generator

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import subprocess
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

mout_thr = 0.16
min_thr = 0.84
mtot_thr = 0.97


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("strength/be_shell.py")
    logger.debug("Runnning test " + __name__)
    arguments = [
        "riot/verbose=False",
        "parthenon/time/tlim=1.3e-4",
        "parthenon/time/nlim=2000",
        "parthenon/time/integrator=rk2",
        "parthenon/time/ncycle_out=50",
        "parthenon/mesh/nx1=40",
        "parthenon/mesh/nx2=40",
        "parthenon/mesh/nx3=1",
        "parthenon/meshblock/nx1=8",
        "parthenon/meshblock/nx2=8",
        "parthenon/meshblock/nx3=1",
        "hydro/recon=ppm4",
        "parthenon/output1/dt=1.0",
    ]
    riot.mpirun(1, "strength/be_shell.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # Analytic solution
    rho0 = 1.845
    rin = 5.0
    rout = 7.81
    mass_analytic = rho0 * np.pi * (rout * rout - rin * rin)
    # Sum mass inside analytic shell
    be_data = phdf("build/src/be_shell.out1.final.phdf")
    be_rho = be_data.Get("c.c.bulk.rho", False, False)[:, 0, :, :]
    inside = 0.0
    outside = 0.0
    for i in range(be_rho.shape[0]):
        # Cell Area
        dx = be_data.xng[i, 1] - be_data.xng[i, 0]
        dy = be_data.yng[i, 1] - be_data.yng[i, 0]
        area = dx * dy
        # Cylindrical radius
        xc = 0.5 * (be_data.xng[i, 1:] + be_data.xng[i, :-1])
        yc = 0.5 * (be_data.yng[i, 1:] + be_data.yng[i, :-1])
        rc = np.sqrt(
            np.array([xc for ii in range(len(xc))]) ** 2.0
            + np.array([yc for jj in range(len(yc))]).transpose() ** 2.0
        )
        # Mass in shell
        inside_shell = (rc <= rout) & (rc >= rin)
        inside += np.sum(be_rho[i, :, :][inside_shell] * area)
        outside += np.sum(be_rho[i, :, :][~inside_shell] * area)
    # Check failure modes
    total_mass = outside + inside
    frac_mass_outside = outside / total_mass
    frac_mass_inside = inside / total_mass
    if frac_mass_outside > mout_thr:
        logger.warning(
            "Fractional mass outside shell too large "
            "frac_mass: {0:g} threshold: {1:g}".format(frac_mass_outside, mout_thr)
        )
        analyze_status = False
    if frac_mass_inside < min_thr:
        logger.warning(
            "Fractional mass inside shell too small "
            "frac_mass: {0:g} threshold: {1:g}".format(frac_mass_inside, min_thr)
        )
        analyze_status = False
    if total_mass / mass_analytic < mtot_thr:
        logger.warning(
            "Total mass inconsistent with initial condition "
            "mtot/mtot_analytic: {0:g} threshold: {1:g}".format(
                total_mass / mass_analytic, mtot_thr
            )
        )
        analyze_status = False
    return analyze_status
