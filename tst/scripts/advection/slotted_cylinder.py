# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

import logging
import os

import numpy as np
from phdf import phdf

import scripts.utils.riot as riot

logger = logging.getLogger("riot" + __name__[7:])
input_id = "advection/slotted_cylinder"
advection_modes = ("solid_body", "diagonal")
problem_ids = {
    (mode, enabled): "slotted_cylinder_{}_{}".format(
        mode, "thinc" if enabled else "plm"
    )
    for mode in advection_modes
    for enabled in (False, True)
}
fraction_variable = "c.c.mat.volume_fraction_1"
material_mass_variables = ("c.c.mat.rho_0", "c.c.mat.rho_1")
pressure_variable = "c.c.bulk.pressure"


def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    for (mode, enabled), problem_id in problem_ids.items():
        logger.debug("Running %s", problem_id)
        riot.run(
            input_id + ".rin",
            [
                "parthenon/job/problem_id=" + problem_id,
                "physics/thinc=" + str(enabled).lower(),
                "slotted_cylinder/params/advection=" + mode,
                "parthenon/mesh/ix1_bc=periodic",
                "parthenon/mesh/ox1_bc=periodic",
                "parthenon/mesh/ix2_bc=periodic",
                "parthenon/mesh/ox2_bc=periodic",
            ],
        )


def _load(filename, variable):
    data = phdf(filename)
    alpha = data.Get(variable, True).reshape(-1)
    _, y, x = data.GetVolumeLocations(True)
    order = np.lexsort((x, y))
    return alpha[order]


def _metrics(initial, final):
    return {
        "l1": np.mean(np.abs(final - initial)),
        "mixed": np.mean(4.0 * final * (1.0 - final)),
        "area_drift": np.mean(final) - np.mean(initial),
    }


def analyze():
    logger.debug("Analyzing test " + __name__)
    base = os.environ.get("RIOT_TEST_OUTPUT_DIR", "build/src")
    results = {}
    reference_initial = None
    passed = True

    for case, problem_id in problem_ids.items():
        initial_file = os.path.join(base, problem_id + ".out1.00000.phdf")
        final_file = os.path.join(base, problem_id + ".out1.final.phdf")
        current_initial = _load(initial_file, fraction_variable)
        final = _load(final_file, fraction_variable)
        if reference_initial is None:
            reference_initial = current_initial
        elif not np.allclose(
            reference_initial, current_initial, rtol=0.0, atol=1.0e-14
        ):
            logger.warning("slotted-cylinder initial conditions differ between cases")
            passed = False
        if (
            not np.all(np.isfinite(final))
            or np.min(final) < -1.0e-12
            or np.max(final) > 1.0 + 1.0e-12
        ):
            logger.warning("%s produced non-finite or unbounded fractions", problem_id)
            passed = False
        results[case] = _metrics(current_initial, final)

        initial_masses = [_load(initial_file, var) for var in material_mass_variables]
        final_masses = [_load(final_file, var) for var in material_mass_variables]
        results[case]["mass_drift"] = np.mean(final_masses[1]) - np.mean(
            initial_masses[1]
        )
        results[case]["partition_error"] = np.max(
            np.abs(np.add.reduce(final_masses) - 1.0)
        )

        initial_pressure = _load(initial_file, pressure_variable)
        final_pressure = _load(final_file, pressure_variable)
        results[case]["pressure_error"] = max(
            np.max(np.abs(initial_pressure - 1.0)),
            np.max(np.abs(final_pressure - 1.0)),
        )

    for mode in advection_modes:
        off = results[(mode, False)]
        on = results[(mode, True)]
        logger.info(
            "\nslotted cylinder %s:\n"
            "   L1 PLM/THINC %.8e %.8e; mixed %.8e %.8e; \n"
            "   area drift %.3e %.3e; \n"
            "   mass drift %.3e %.3e; \n"
            "   partition error %.3e %.3e; \n"
            "   pressure error %.3e %.3e\n",
            mode,
            off["l1"],
            on["l1"],
            off["mixed"],
            on["mixed"],
            off["area_drift"],
            on["area_drift"],
            off["mass_drift"],
            on["mass_drift"],
            off["partition_error"],
            on["partition_error"],
            off["pressure_error"],
            on["pressure_error"],
        )

        if max(abs(off["mass_drift"]), abs(on["mass_drift"])) > 1.0e-10:
            logger.warning("%s material mass is not conserved", mode)
            passed = False
        if max(off["partition_error"], on["partition_error"]) > 1.0e-6:
            logger.warning("%s material densities do not sum to one", mode)
            passed = False
        if max(off["pressure_error"], on["pressure_error"]) > 1.0e-6:
            logger.warning("%s pressure is not uniform", mode)
            passed = False
        if on["l1"] >= off["l1"]:
            logger.warning("THINC did not reduce the %s one-period L1 error", mode)
            passed = False
        if on["mixed"] >= off["mixed"]:
            logger.warning("THINC did not reduce the %s mixed-cell measure", mode)
            passed = False
    return passed
