#!/usr/bin/env python3
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

import numpy as np
import riot


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input(
        "parthenon/job",
        problem_id="laser",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.electron_number_density",
            "c.c.bulk.laser_energy_density",
            "c.c.bulk.electron_temperature",
            "c.c.bulk.laser_deposition",
            "c.c.bulk.electron_internal_energy",
            "c.c.bulk.ionization_zbar",
            "c.c.bulk.velocity",
            "c.c.bulk.signal",
            "c.c.bulk.laser_tau_max",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=2.0e-11,  # time increment between outputs
        sparse_seed_nans=True,  # Seed sparse data with NaNs
    )

    riot.input(
        "parthenon/time",
        nlim=-10,  # cycle limit
        tlim=1.0e-9,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=1000,  # Number of zones in X1-direction
        x1min=0,  # minimum value of X1
        x1max=1.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=1,  # Number of zones in X2-direction
        x2min=-0.0886,  # minimum value of X2
        x2max=0.0886,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.0886,  # minimum value of X3
        x3max=0.0886,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input("materials", sparse_dealloc=True)

    riot.input(
        "material0",
        eos="ideal_eos",
        electron_eos="ideal_eos",
        max_bnd_level=0,
        max_mat_level=0,
    )

    const = riot.constants()

    riot.input(
        "ideal_eos",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1.5 * 7 * const.kb / (12.0 * const.amu),
        mean_atomic_number=6.0,
        mean_atomic_mass=12.0,
        zsplit=True,
    )

    riot.input("material1", eos="weird", electron_eos="weird")

    riot.input(
        "weird",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=3.0e9,
        mean_atomic_number=1.0,
        mean_atomic_mass=12.0,
        zsplit=True,
    )

    p0 = 1.0e6

    riot.input(
        "region0",
        name="ablator",
        mask_type="inside_rectangle",
        x0=0.0,
        matid=0,
        c_m_rho=1.0e-3,
        c_m_pressure=p0,
    )

    riot.input(
        "physics",
        hydro=True,
        fixed_fluid=True,
        lasers=True,
        ionization=True,
        sparse_physics=False,
    )

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input(
        "ionization",
        fully_ionized=True,
        electron_ion_coupling=False,
        advect_electron_entropy=False,
    )

    riot.input(
        "laser",
        enable_deposition=True,
        dt_safety=0.7,
        dt_edot_floor=1.0,
        dt_tau_cutoff=0.2,
    )

    beam_width = 0.01

    # 2 kJ over 1 ns
    power0 = 2.0e3 / 1.0e-9

    riot.input(
        "laser0",
        lens_x=[riot.input["parthenon/mesh"]["x1min"], 0.0, 0.0],
        target_x=[riot.input["parthenon/mesh"]["x1max"], 0.0, 0.0],
        target_size_ratio=1.0,
        phi=0.0,
        phi_axis="z",
        power_semi_major_axis=beam_width,
        power_semi_minor_axis=beam_width,
        distribution="flat",
        grid_type="equal_area",
        nr=1,
        ntarget=10,
        time_ns=[0.0, 1.0],
        power_watts=[power0, power0],
        wavelength_nm=530.0,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
