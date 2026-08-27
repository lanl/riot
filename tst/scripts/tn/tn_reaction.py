# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

# Regression test based on artificial TN reactions

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
import subprocess
from phdf import phdf

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

drho_tol = 1.0e-14
iso_tol = np.array([2.0e-6, 2.0e-7, 5.0e-5, 5.0e-5])


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("tn/tn_test.py")
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/time/tlim=1.0e-8",
        "parthenon/time/nlim=100",
        "parthenon/time/integrator=rk2",
        "parthenon/mesh/nx1=4",
        "parthenon/mesh/nx2=1",
        "parthenon/mesh/nx3=1",
        "parthenon/meshblock/nx1=4",
        "parthenon/meshblock/nx2=1",
        "parthenon/meshblock/nx3=1",
        "hydro/recon=plm",
        "parthenon/output1/dt=1.0e-9",
        "tnburn/deposit_locally_1=true",
        "isotope_data/filename=../../../inputs/tn/dummy_tn_data.hdf5",
    ]
    riot.mpirun(1, "tn/tn_test.rin", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # Read in data
    times = []
    rho_riot = []
    iso_0_0_riot = []
    iso_0_1_riot = []
    iso_0_2_riot = []
    iso_0_3_riot = []
    for i in range(0, 11):
        tag = "%05d" % i if i <= 9 else "final"
        data_riot = phdf("build/src/tn_test.out1." + tag + ".phdf")
        times.append(data_riot.Time)
        rho_riot.append(data_riot.Get("c.c.mat.rho_0", False, False)[0, 0, 0, 0])
        iso_riot = data_riot.Get("c.c.mat.iso_0", False, False)
        iso_0_0_riot.append(iso_riot[0, 0, 0, 0])
        iso_0_1_riot.append(iso_riot[0, 1, 0, 0])
        iso_0_2_riot.append(iso_riot[0, 2, 0, 0])
        iso_0_3_riot.append(iso_riot[0, 3, 0, 0])
    # Extract only one cell in 4 zone mesh
    rho = [tt[0] for tt in rho_riot]
    iso_0_0 = np.array([tt[0] for tt in iso_0_0_riot])
    iso_0_1 = np.array([tt[0] for tt in iso_0_1_riot])
    iso_0_2 = np.array([tt[0] for tt in iso_0_2_riot])
    iso_0_3 = np.array([tt[0] for tt in iso_0_3_riot])
    # Analytic solution
    masses_amu = np.array([1.00866, 2.0141, 3.0155, 4.0026])
    masses_cgs = masses_amu * 1.660538921e-24
    rate = masses_amu[1] * masses_amu[2] * 5e26 * np.array([1, -1, -1, 1])
    mass_init = np.array([0, masses_amu[1], masses_amu[2], 0])
    mass_true = np.array([mass_init + rate * masses_cgs * time for time in times]).T
    # Check failure modes
    iso_sum = iso_0_0 + iso_0_1 + iso_0_2 + iso_0_3
    drho = np.max(np.abs(iso_sum - rho) / rho)
    if drho > drho_tol:
        logger.warning(
            "Isotope masses do not appropriately sum to bulk "
            "delta: {0:g} tol: {1:g}".format(drho, drho_tol)
        )
        analyze_status = False
    tiny = np.finfo(float).eps
    iso_errs = np.array(
        [
            np.max(np.abs(iso_0_0 - mass_true[1]) / (mass_true[1] + tiny)),
            np.max(np.abs(iso_0_1 - mass_true[2]) / (mass_true[2] + tiny)),
            np.max(np.abs(iso_0_2 - mass_true[3]) / (mass_true[3] + tiny)),
            np.max(np.abs(iso_0_3 - mass_true[0]) / (mass_true[0] + tiny)),
        ]
    )
    iso_idxs = np.array(
        [
            np.where(
                (np.abs(iso_0_0 - mass_true[1]) / (mass_true[1] + tiny)) >= iso_errs[0]
            )[0][0],
            np.where(
                (np.abs(iso_0_1 - mass_true[2]) / (mass_true[2] + tiny)) >= iso_errs[1]
            )[0][0],
            np.where(
                (np.abs(iso_0_2 - mass_true[3]) / (mass_true[3] + tiny)) >= iso_errs[2]
            )[0][0],
            np.where(
                (np.abs(iso_0_3 - mass_true[0]) / (mass_true[0] + tiny)) >= iso_errs[3]
            )[0][0],
        ]
    )

    if np.any(iso_errs > iso_tol):
        logger.warning(
            "Isotope masses do not match exact solution: \n"
            "iso_0_0 :: idx: {} value: {:g} true: {:g} err: {:g} tol: {:g} \n"
            "iso_0_1 :: idx: {} value: {:g} true: {:g} err: {:g} tol: {:g} \n"
            "iso_0_2 :: idx: {} value: {:g} true: {:g} err: {:g} tol: {:g} \n"
            "iso_0_3 :: idx: {} value: {:g} true: {:g} err: {:g} tol: {:g} ".format(
                iso_idxs[0],
                iso_0_0[iso_idxs[0]],
                mass_true[1][iso_idxs[0]],
                iso_errs[0],
                iso_tol[0],
                iso_idxs[1],
                iso_0_1[iso_idxs[1]],
                mass_true[2][iso_idxs[1]],
                iso_errs[1],
                iso_tol[1],
                iso_idxs[2],
                iso_0_2[iso_idxs[2]],
                mass_true[3][iso_idxs[2]],
                iso_errs[2],
                iso_tol[2],
                iso_idxs[3],
                iso_0_3[iso_idxs[3]],
                mass_true[0][iso_idxs[3]],
                iso_errs[3],
                iso_tol[3],
            )
        )
        analyze_status = False
    return analyze_status
