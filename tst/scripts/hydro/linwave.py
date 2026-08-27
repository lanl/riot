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

# Regression test based on Newtonian hydro linear wave convergence problem

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

integrators = ["rk2", "rk4"]
recon = ["plm", "weno5"]
flux = ["hllc"]
wave = ["L-sound", "R-sound", "entropy"]
nranks = 1
problem_id = "linwave"
input_id = "linear_modes"
multiple_materials = False


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("linear_modes/" + input_id + ".py")
    logger.debug("Runnning test " + __name__)
    # L-going sound wave
    for iv in integrators:
        for rv in recon:
            for fv in flux:
                for res in (16, 32):
                    arguments = [
                        "parthenon/job/problem_id=" + problem_id,
                        "parthenon/time/nlim=1000",
                        "parthenon/time/integrator=" + iv,
                        "parthenon/mesh/sparse_init=false",
                        "parthenon/mesh/refinement=none",
                        "parthenon/mesh/nghost=4",
                        "parthenon/mesh/x1min=0.0",
                        "parthenon/mesh/x1max=3.0",
                        "parthenon/mesh/x2min=0.0",
                        "parthenon/mesh/x2max=1.5",
                        "parthenon/mesh/x3min=0.0",
                        "parthenon/mesh/x3max=1.5",
                        "parthenon/mesh/nx1=" + repr(res),
                        "parthenon/mesh/nx2=" + repr(res / 2),
                        "parthenon/mesh/nx3=" + repr(res / 2),
                        "parthenon/meshblock/nx1=" + repr(res / 4),
                        "parthenon/meshblock/nx2=" + repr(res / 4),
                        "parthenon/meshblock/nx3=" + repr(res / 4),
                        "hydro/recon=" + rv,
                        # "hydro/riemann=" + fv,
                        "problem/amp=1.0e-6",
                        "problem/iprob=" + str(4 if multiple_materials else 1),
                        "problem/nperiod=1",
                        "parthenon/output1/dt=-1.0",
                    ]
                    # L-going sound wave
                    args_l = arguments + ["problem/wave_flag=0", "problem/vflow=0.0"]
                    # riot.run('linear_modes/' + input_id + '.rin', args_l)
                    riot.mpirun(nranks, "linear_modes/" + input_id + ".rin", args_l)
                    # R-going sound wave
                    args_r = arguments + ["problem/wave_flag=4", "problem/vflow=0.0"]
                    # riot.run('linear_modes/' + input_id + '.rin', args_r)
                    riot.mpirun(nranks, "linear_modes/" + input_id + ".rin", args_r)
                    # entropy wave
                    args_entr = arguments + ["problem/wave_flag=3", "problem/vflow=1.0"]
                    # riot.run('linear_modes/' + input_id + '.rin', args_entr)
                    riot.mpirun(nranks, "linear_modes/" + input_id + ".rin", args_entr)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    # Load data
    data = np.loadtxt(
        "build/src/" + problem_id + "-errs.dat", dtype=np.float64, ndmin=2
    )
    if np.isnan(data).any():
        logger.warning("NaN encountered")
        analyze_status = False
        raise FloatingPointError("NaN encountered")
    data = data.reshape(
        [len(integrators), len(recon), len(flux), 2, len(wave), data.shape[-1]]
    )
    for ii, iv in enumerate(integrators):
        for ri, rv in enumerate(recon):
            error_threshold = [0.0] * len(wave)
            conv_threshold = [0.0] * len(wave)
            if rv == "weno5":
                if iv == "rk2":
                    error_threshold[0] = error_threshold[1] = 5e-8  # sound
                    error_threshold[2] = 1.3e-8  # entropy
                    # The mixed truncation errors of the 2nd-order
                    # integrator and 5th-order recon produces slightly
                    # worse convergence factors at these resolutions,
                    # despite the total error being much lower
                    conv_threshold[0] = conv_threshold[1] = conv_threshold[2] = 0.3
                else:  # if iv == rk4
                    error_threshold[0] = error_threshold[1] = error_threshold[2] = (
                        1.6e-9
                    )
                    conv_threshold[0] = conv_threshold[1] = conv_threshold[2] = 0.04
            else:  # plm recon
                # spatial discretization error dominates,
                # don't distinguish between RK2 and RK4
                error_threshold[0] = error_threshold[1] = 1.5e-7  # sound
                error_threshold[2] = 1.25e-7  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.25  # sound
                conv_threshold[2] = 0.28  # entropy
            for fi, fv in enumerate(flux):
                for wi, wv in enumerate(wave):
                    l1_rms_n16 = data[ii][ri][fi][0][wi][4]
                    l1_rms_n32 = data[ii][ri][fi][1][wi][4]
                    if l1_rms_n32 > error_threshold[wi]:
                        logger.warning(
                            "{0} wave error too large for {1}+"
                            "{2}+{3} configuration, "
                            "error: {4:g} threshold: {5:g}".format(
                                wv, iv, rv, fv, l1_rms_n32, error_threshold[wi]
                            )
                        )
                        analyze_status = False
                    if l1_rms_n32 / l1_rms_n16 > conv_threshold[wi]:
                        logger.warning(
                            "{0} wave not converging for {1}+"
                            "{2}+{3} configuration, "
                            "conv: {4:g} threshold: {5:g}".format(
                                wv,
                                iv,
                                rv,
                                fv,
                                l1_rms_n32 / l1_rms_n16,
                                conv_threshold[wi],
                            )
                        )
                        analyze_status = False
                l1_rms_l = data[ii][ri][fi][1][wave.index("L-sound")][4]
                l1_rms_r = data[ii][ri][fi][1][wave.index("R-sound")][4]
                if 2 * np.abs(l1_rms_l - l1_rms_r) / (l1_rms_l + l1_rms_r) > 1e-5:
                    logger.warning(
                        "Errors in L/R-going sound waves not "
                        "equal for {0}+{1}+{2} configuration, "
                        "{3:g} {4:g}".format(iv, rv, fv, l1_rms_l, l1_rms_r)
                    )
                    analyze_status = False
    return analyze_status
