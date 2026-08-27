#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
        problem="region_pgen",  # name of the pgen
    )

    riot.input(
        "parthenon/job",
        problem_id="sedov2d",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.rho", "c.c.bulk.velocity", "c.c.bulk.pressure"],
        file_type="hdf5",  # Tabular data dump
        dt=0.1,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1.5,  # time limit
        integrator="rk4",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        refinement="none",
        numlevel=0,
        nx1=400,  # Number of zones in X1-direction
        x1min=-0.5,  # minimum value of X1
        x1max=0.5,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=600,  # Number of zones in X2-direction
        x2min=-0.75,  # minimum value of X2
        x2max=0.75,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.5,  # minimum value of X3
        x3max=0.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=200,  # Number of cells in each MeshBlock, X1-dir
        nx2=120,  # Number of cells in each MeshBlock, X2-dir
        nx3=1,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "materials",
        sparse_dealloc=False,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.6666666667,
        Cv=1.0e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        name="region0",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=0.1,
    )

    riot.input(
        "region1",
        name="region1",
        mask_type="inside_cylinder",
        matid=0,
        x0=0.0,
        y0=0.0,
        z0=-0.75,
        x1=0.0,
        y1=0.0,
        z1=0.75,
        radius=0.1,
        c_m_rho=1.0,
        c_m_pressure=10.0,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "hydro",
        recon="weno5",
        cfl=1.5,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
