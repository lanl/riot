#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
        problem="linear_modes",  # name of the pgen
    )

    riot.input(
        "parthenon/job",
        problem_id="linear_modes",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.bulk.momentum",
            "c.c.mat.rho",
            "c.c.bulk.total_material_energy",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=0.1,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=-1.0,  # time limit (to be reset by problem generator)
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        refinement="none",
        numlevel=3,
        # derefine_count = 5,
        nx1=128,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=3.0,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=64,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=1.5,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=64,  # Number of zones in X3-direction
        x3min=0.0,  # minimum value of X3
        x3max=1.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,  # Number of cells in each MeshBlock, X1-dir
        nx2=32,  # Number of cells in each MeshBlock, X2-dir
        nx3=32,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.66666666667,
        Cv=1.0e12,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.9,
        amr_interface=False,
        # riemann = hllc,
    )

    riot.input(
        "problem",
        iprob=1,  # 1: single material linear wave, 2: two materials
        nperiod=1,  # Number of wave crossing periods to evolve simulation
        wave_flag=0,  # Wave family number ([0-4] for adiabatic hydro)
        amp=1.0e-6,  # Wave Amplitude
        vflow=0.0,  # background flow velocity
        along_x1=False,  # set to 'true' for wave along x1-axis
        along_x2=False,  # set to 'true' for wave along x2-axis
        along_x3=False,  # set to 'true' for wave along x3-axis
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
