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
# Set up sinusoidal density profile with uniform pressure and accelerate it
# through periodic boundaries using constant gravity
# gamma = 1.4, P0 = 2.5
#
# will travel 1 cm in sqrt(2/gacc) s.

import riot


def make_input():

    riot.input(
        "riot",
        problem="gacc",
    )

    riot.input(
        "parthenon/job",
        problem_id="gacceleration",
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.mat.rho",
            "c.c.mat.volume_fraction",
            "c.c.bulk.velocity",
            "c.c.bulk.total_material_energy",
            "c.c.bulk.rho",
            "c.c.bulk.pressure",
            "c.c.bulk.bulk_modulus",
            "c.m.pressure",
        ],
        file_type="hdf5",
        dt=0.2,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=0.4472135955,
        ncycle_out=20,
        integrator="rk2",
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        refinement="none",
        numlevel=0,
        nx1=20,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=0.2,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=100,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=1.0,  # maximum value of X2
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
        nx1=20,
        nx2=100,
        nx3=1,
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
        max_mat_level=0,
        max_bnd_level=0,
    )

    riot.input(
        "physics",
        hydro=True,
        gravity=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
    )

    riot.input(
        "gravity",
        gravity_dim=1,
        gravity_g=-10.0,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
