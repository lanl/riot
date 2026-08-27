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
        problem="shock_tube",  # name of the pgen
    )

    riot.input(
        "parthenon/job",
        problem_id="shock_tube",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
            "c.c.volume_fraction",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=0.05,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=0.2,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        # refinement  = adaptive,
        # numlevel    = 2,
        # derefine_count = 5,
        nx1=128,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=1.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=1,  # Number of zones in X2-direction
        x2min=-0.5,  # minimum value of X2
        x2max=0.5,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.5,  # minimum value of X3
        x3max=0.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    # riot.input(
    #     "parthenon/meshblock",
    #     nx1="16  # In practice this gives us four mesh blocks of size 64x1x1 each",
    #     nx2=16,
    #     nx3=1,
    # )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=1.0e12,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        # riemann = hllc,
        # amr_interface = true,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
