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
        problem="region_pgen",
    )

    riot.input(
        "parthenon/job",
        problem_id="sie_advection",  # problem ID: basename of output filenames
    )

    # riot.input(
    #     "parthenon/output0",
    #     file_type="rst",
    #     dt=0.9,
    # )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.mat.volume_fraction",
            "c.c.bulk.internal_energy",
            "c.c.bulk.pressure",
            "c.c.bulk.bulk_modulus",
            "c.c.bulk.electron_internal_energy",
            "c.c.bulk.electron_pressure",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=2.0e-2,  # time increment between outputs
        sparse_seed_nans=True,
        # ghost_zones = true,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1.0,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        # refinement  = adaptive,
        # numlevel    = 3,
        # deref_count = 5,
        nghost=4,
        nx1=128,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=1.0,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=128,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=1.0,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.5,  # minimum value of X3
        x3max=0.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
        num_threads=1,  # maximum number of OMP threads
    )

    riot.input(
        "parthenon/meshblock",
        nx1=64,
        nx2=64,
        nx3=1,
    )

    riot.input(
        "materials",
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos="air",
        electron_eos="air",
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos="air",
        electron_eos="air",
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "air",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
    )

    riot.input(
        "region0",
        name="region0",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=2.0,
        c_c_bulk_electron_temperature=2.0e3,
        c_c_bulk_velocity=[1, 1, 0],
    )

    riot.input(
        "region1",
        name="region1",
        mask_type="inside_rectangle",
        matid=1,
        x0=0.25,
        y0=0.25,
        z0=-0.50,
        x1=0.75,
        y1=0.75,
        z1=0.50,
        c_m_rho=1.0,
        c_m_pressure=1.0,
        c_c_bulk_electron_temperature=4.0e3,
        c_c_bulk_velocity=[1, 1, 0],
    )

    riot.input(
        "physics",
        hydro=True,
        ionization=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.1,
        # riemann = hllc,
        amr_interface=True,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
