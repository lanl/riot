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
        problem_id="sedov",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.mat.rho",
            "c.c.mat.volume_fraction",
            "c.c.cell_delta",
            "c.c.bulk.internal_energy",
            "c.c.bulk.total_material_energy",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=5.0e-4,  # time increment between outputs
        sparse_seed_nans=True,
    )

    riot.input(
        "parthenon/time",
        nlim=-160,  # cycle limit
        tlim=0.01,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
        # balancer = automatic,
        # tolerance = 0.05,
    )

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=2,
        derefine_count=5,
        nghost=2,
        pack_size=1,
        nx1=128,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=0.3,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=256,  # Number of zones in X2-direction
        x2min=-0.3,  # minimum value of X2
        x2max=0.3,  # maximum value of X2
        ix2_bc="outflow",  # Inner-X2 boundary condition flag
        ox2_bc="outflow",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=0.0,  # minimum value of X3
        x3max=6.2831853071795864769,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,  # Number of cells in each MeshBlock, X1-dir
        nx2=32,  # Number of cells in each MeshBlock, X2-dir
        nx3=1,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "parthenon/refinement0",
        method="derivative_order_1",
        field="c.c.bulk.pressure",
    )

    riot.input(
        "materials",
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.4,
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
        c_m_pressure=1.0e-3,
    )

    riot.input(
        "region1",
        name="region1",
        mask_type="inside_rectangle",
        matid=0,
        x0=0.0,
        x1=0.00234375,
        z0=-0.00234375,
        z1=0.00234375,
        c_m_rho=1.0,
        c_m_sie=9271447.4226404,
    )

    riot.input(
        "physics",
        hydro=True,
        sparse_physics=True,
        sparse_physics_threshold=1.0e-30,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.5,
        riemann="lhllc",
        amr_interface=True,
        lm_correction=False,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
