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

# Regression test based on Newtonian linear advection of a sharp material cube

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import sys
import os

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

nranks = 1

gamma = 1.4
gm1 = gamma - 1
Cv = 1
rho0 = 1
T0 = 1
ds_ratio_thresh = 0.15


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("ionization/ionized_shocktube.py")
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/mesh/nghost=2",
        "parthenon/mesh/nx1=1024",
        "parthenon/mesh/nx2=1",
        "parthenon/mesh/nx3=1",
        "parthenon/meshblock/nx1=1024",
        "parthenon/meshblock/nx2=1",
        "parthenon/meshblock/nx3=1",
        "parthenon/time/tlim=0.2",
        "parthenon/time/ncycle_out=100",
        "parthenon/output1/dt=0.2",
        "parthenon/time/nlim=300",
        "parthenon/time/ncycle_out=50",
        f"air/Gamma={gamma}",
        f"air/Cv={Cv}",
    ]
    riot.mpirun(nranks, "ionization/ionized_shocktube.rin", arguments)


# Analyze outputs
def get_entropy_rhoT(rho, T):
    return Cv * np.log(T / T0) + gm1 * Cv * np.log(rho0 / rho)


def get_entropy_rhoE(rho, sie):
    return get_entropy_rhoT(rho, sie / Cv)


def get_entropy(state, electrons=False):
    rho = state["c.c.bulk.rho"][0, 0, 0]
    if electrons:
        sie = state["c.c.bulk.electron_internal_energy"][0, 0, 0] / rho
    else:
        sie = state["c.c.mat.internal_energy_0"][0, 0, 0, 0] / rho
    return get_entropy_rhoE(rho, sie)


def analyze():
    from scipy import integrate
    import h5py

    try:
        simpson = integrate.simpson
    except:
        simpson = integrate.simps

    logger.debug("Analyzing test " + __name__)

    analyze_status = True

    data0 = h5py.File("build/src/ionized_sod.out1.00000.phdf", "r")
    data1 = h5py.File("build/src/ionized_sod.out1.final.phdf", "r")

    s_ions_i = get_entropy(data0, False)
    s_electrons_i = get_entropy(data0, True)
    s_ions_f = get_entropy(data1, False)
    s_electrons_f = get_entropy(data1, True)

    xf = data0["Locations/x"][0]
    x = 0.5 * (xf[1:] + xf[:-1])

    rho_i = data0["c.c.bulk.rho"][0, 0, 0]
    rho_f = data1["c.c.bulk.rho"][0, 0, 0]

    stot_ions_i = simpson(rho_i * s_ions_i, x=x)
    stot_ions_f = simpson(rho_f * s_ions_f, x=x)
    dstot_ions = stot_ions_f - stot_ions_i

    stot_electrons_i = simpson(rho_i * s_electrons_i, x=x)
    stot_electrons_f = simpson(rho_f * s_electrons_f, x=x)
    dstot_electrons = stot_electrons_f - stot_electrons_i

    if dstot_electrons < 0:
        logger.warning(
            "ionized sod negative change "
            + f"in electron entropy: ds = {dstot_electrons}"
        )
        analyze_status = False
    if dstot_ions < 0:
        logger.warning(
            "ionized sod negative change " + f"in ion entropy: ds = {dstot_ions}"
        )
        analyze_status = False
    dsratio = np.abs(dstot_electrons) / np.abs(dstot_ions)
    if dsratio > ds_ratio_thresh:
        logger.warning(
            "ionized sod change in electron entropy "
            + "too large. "
            + f"ds_e, ds_i, ratio = {dstot_electrons}, {dstot_ions}, {dsratio}"
        )
        analyze_status = False

    return analyze_status
