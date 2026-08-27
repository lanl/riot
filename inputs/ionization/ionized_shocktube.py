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
        problem="region_pgen",
    )

    riot.input(
        "parthenon/job",
        problem_id="ionized_sod",
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
            "c.c.bulk.total_material_energy",
            "c.c.bulk.electron_internal_energy",
            "c.c.bulk.electron_pressure",
            "c.c.mat.internal_energy",
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
        nghost=2,
        nx1=1024,  # Number of zones in X1-direction
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

    riot.input(
        "materials",
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos="air",
        electron_eos="air",
    )

    riot.input(
        "air",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=1.0,
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=1,
        c_m_pressure=1.0,
        # press_e = 0.5,
    )

    riot.input(
        "region1",
        mask_type="inside_rectangle",
        x0=0.5,
        matid=0,
        c_m_rho=0.125,
        c_m_pressure=0.1,
        # press_e = 0.05,
    )

    riot.input(
        "physics",
        hydro=True,
        ionization=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        amr_interface=True,
    )

    riot.input(
        "ionization",
        root_tol=1e-20,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
