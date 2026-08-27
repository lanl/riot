#!/usr/bin/env python3
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
# This file was made in part with generative AI.

import riot


def make_input():

    riot.input(
        "riot",
        problem="taylor_green",
    )

    riot.input(
        "parthenon/job",
        problem_id="tg",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output0",
        file_type="rst",
        dt=10,
    )

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.rho", "c.c.bulk.velocity"],
        file_type="hdf5",  # Tabular data dump
        dt=1.0,  # time increment between outputs
    )

    riot.input(
        "parthenon/output2",
        file_type="hst",
        dt=0.01,
        packages="hydro",
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=6.0,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=2,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        # refinement  = adaptive,
        # numlevel    = 3,
        # derefine_count = 5,
        # sparse_init = true,
        nghost=4,
        nx1=32,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=6.2831853071796,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=32,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=6.2831853071796,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=32,  # Number of zones in X3-direction
        x3min=0,  # minimum value of X3
        x3max=6.2831853071796,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,
        nx2=32,
        nx3=32,
    )

    riot.input(
        "materials",
        sparse_dealloc=False,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=1.0e-3,
        # max_bnd_level = 0,
        # max_mat_level = 0,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "hydro",
        recon="weno5",
        cfl=0.8,
        # riemann = hllc,
        amr_interface=True,
    )

    riot.input(
        "taylor_green",
        P0=100,  # => M < 0.1 (approx)
        amplitude_x=1,
        amplitude_y=-1,
        amplitude_z=0,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
