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
            "n.electron_number_density",
            "c.c.bulk.laser_energy_density",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=0.01,  # time increment between outputs
        sparse_seed_nans=True,  # Seed sparse data with NaNs
    )

    riot.input(
        "parthenon/time",
        nlim=1,  # cycle limit
        tlim=0.5,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=200,  # Number of zones in X1-direction
        x1min=0,  # minimum value of X1
        x1max=16 * np.pi,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=50,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=10.0,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-1.0,  # minimum value of X3
        x3max=1.0,  # maximum value of X3
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

    riot.input(
        "ideal_eos",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=3.0e7,
        mean_atomic_number=1.0,
        mean_atomic_mass=1.00784,
        zsplit=True,
    )

    riot.input(
        "region0",
        name="trough",
        mask_type="background",
        matid=0,
        c_m_pressure=1.0e-6,
        # c_m_rho=1.0,
    )

    riot.input(
        "trough/params",
        mean_atomic_number=riot.input["ideal_eos"]["mean_atomic_number"],
        mean_atomic_mass=riot.input["ideal_eos"]["mean_atomic_mass"],
        wavelength=351.0e-7,
    )

    riot.input("physics", hydro=True, lasers=True, ionization=True)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input("laser", enable_deposition=False)

    beam_width = 1.0e-12

    riot.input(
        "laser0",
        lens_x=[riot.input["parthenon/mesh"]["x1min"], 8.025, 0.1],
        target_x=[riot.input["parthenon/mesh"]["x1max"], 8.025, 0.1],
        target_size_ratio=1.0,
        phi=0.0,
        phi_axis="z",
        power_semi_major_axis=beam_width,
        power_semi_minor_axis=beam_width,
        distribution="flat",
        grid_type="equal_area",
        nr=5,
        ntarget=10000,
        time_ns=[0.0, 1.0],
        power_watts=[1.0, 1.0],
    )


const = riot.constants()


class trough:
    def __init__(self):
        pass

    def c_m_rho(self, pos, rho):
        ncrit = np.pi * const.me * const.c**2 / (self.wavelength**2 * const.qe**2)
        nt = 0.5 * ncrit
        to_rho = const.amu * self.mean_atomic_mass / self.mean_atomic_number
        rho[:] = to_rho * (nt + (ncrit - nt) * np.power((pos[:, self.y] - 5.0) / 5, 2))


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
