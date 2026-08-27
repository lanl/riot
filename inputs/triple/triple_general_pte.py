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
        problem="region_pgen",  # name of the pgen
    )

    riot.input(
        "parthenon/job",
        problem_id="triple",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output0",
        file_type="hst",
        dt=0.05,
    )

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.rho", "c.c.bulk.velocity", "c.c.bulk.pressure"],
        file_type="hdf5",  # Tabular data dump
        dt=0.1,  # time increment between outputs
    )

    riot.input(
        "parthenon/output2",
        variables=["c.c.mat.rho", "c.c.mat.volume_fraction"],
        file_type="hdf5",  # Tabular data dump
        dt=0.1,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=5.0,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=3,
        derefine_count=10,
        nx1=112,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=7.0,  # maximum value of X1
        ix1_bc="reflecting",  # Inner-X1 boundary condition flag
        ox1_bc="reflecting",  # Outer-X1 boundary condition flag
        nx2=48,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=3.0,  # maximum value of X2
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
        nx1=16,  # Number of cells in each MeshBlock, X1-dir
        nx2=16,  # Number of cells in each MeshBlock, X2-dir
        nx3=1,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "materials",
        sparse_init=True,
        sparse_dealloc=True,
        use_general_pte=True,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material2",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=1.0e-3,
        max_bnd_level=-1,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
    )

    riot.input(
        "region1",
        mask_type="inside_rectangle",
        matid=1,
        x0=1.0,
        y0=1.5,
        c_m_rho=0.125,
        c_m_pressure=0.1,
    )

    riot.input(
        "region2",
        mask_type="inside_rectangle",
        matid=2,
        x0=1.0,
        y1=1.5,
        c_m_rho=1.0,
        c_m_pressure=0.1,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "diagnostics",
        packages="masses",
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        # riemann = hllc,
        amr_interface=True,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
