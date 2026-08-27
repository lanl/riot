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

# Regression test for gravitational acceleration

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import os
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

integrators = ["rk2"]  # TODO(): Consider also testing RK4
recon = ["weno5"]
errfac = [3.9]  # JMM: fudge factor
resx = [10, 20]
resy = [100, 200]

nranks = 1
problem_id = "gacc"
input_id = "gacc"
g = -10


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)
    for integrator, rec in zip(integrators, recon):
        for rx, ry in zip(resx, resy):
            job_id = f"{problem_id}_{ry}"
            arguments = [
                "parthenon/job/problem_id=" + job_id,
                "parthenon/time/integrator=" + integrator,
                "parthenon/mesh/nghost=4",
                "parthenon/mesh/nx1=" + repr(rx),
                "parthenon/mesh/nx2=" + repr(ry),
                "parthenon/meshblock/nx1=" + repr(rx / 2),
                "parthenon/meshblock/nx2=" + repr(ry / 2),
                "hydro/recon=" + rec,
                "gravity/gravity_g=" + repr(g),
            ]

            riot.mpirun(nranks, input_id + ".rin", arguments)


def rho_shape(y):
    return 1.0 + 0.1 * np.sin(2.0 * y * np.pi)


def rho_ana(t, y):
    y_shift = 0.5 * g * t * t
    return rho_shape((y + y_shift) % 1)


def analyze():
    logger.debug("Analyzing test " + __name__)
    var = "c.c.bulk.rho"

    analyze_status = True
    for fac in errfac:
        soln_coarse = phdf(f"build/src/{problem_id}_{resy[0]}.out1.final.phdf")
        soln_fine = phdf(f"build/src/{problem_id}_{resy[1]}.out1.final.phdf")
        ana_coarse = rho_ana(soln_coarse.Time, soln_coarse.y[..., np.newaxis])
        ana_fine = rho_ana(soln_coarse.Time, soln_fine.y[..., np.newaxis])
        dcoarse = np.max(np.abs(soln_coarse.Get(var, False, True)[:, 0] - ana_coarse))
        dfine = np.max(np.abs(soln_fine.Get(var, False, True)[:, 0] - ana_fine))
        if dfine / dcoarse > 1.0 / fac:
            logger.warning(
                "2D gravitational acceleration error too large, "
                "coarse, fine, ratio, factor = "
                "{:g}, {:g}, {:g}, {:g}".format(
                    dcoarse, dfine, dfine / dcoarse, 1.0 / fac
                )
            )
            analyze_status = False
    return analyze_status
