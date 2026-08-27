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
        problem_id="noh",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.rho", "c.c.bulk.pressure"],
        file_type="hdf5",  # Tabular data dump
        dt=0.01,  # time increment between outputs
        sparse_seed_nans=True,  # Seed sparse data with NaNs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=0.5,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=1024,  # Number of zones in X1-direction
        x1min=-0.5,  # minimum value of X1
        x1max=0.5,  # maximum value of X1
        ix1_bc="reflecting",  # Inner-X1 boundary condition flag
        ox1_bc="reflecting",  # Outer-X1 boundary condition flag
        nx2=1024,  # Number of zones in X2-direction
        x2min=-0.5,  # minimum value of X2
        x2max=0.5,  # maximum value of X2
        ix2_bc="reflecting",  # Inner-X2 boundary condition flag
        ox2_bc="reflecting",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.5,  # minimum value of X3
        x3max=0.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=512,
        nx2=256,
        nx3=1,
    )

    riot.input("materials", sparse_dealloc=True)

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=3.0e7,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        name="noh",
        mask_type="background",
        matid=0,
        c_m_pressure=1.0e-6,
        c_m_rho=1.0,
    )

    riot.input("physics", hydro=True)

    riot.input("hydro", riemann="hllc", recon="plm", cfl=0.5, amr_interface=False)


class noh:
    def __init__(self):
        pass

    def c_c_bulk_velocity(self, pos, vel):
        r = np.sqrt(np.power(pos[:, self.x], 2) + np.power(pos[:, self.y], 2))
        vel[:, self.x] = -pos[:, self.x] / r
        vel[:, self.y] = -pos[:, self.y] / r
        vel[:, self.z] = 0.0


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
