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
        problem="region_pgen",
    )

    riot.input(
        "parthenon/job",
        problem_id="taylor_bar",
    )

    riot.input(
        "parthenon/output0",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.internal_energy",
            "c.c.bulk.pressure",
            "c.c.bulk.temperature",
            "c.c.mat.rho",
            "c.m.equivalent_plastic_strain",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=5.0e-5,  # time increment between outputs
        sparse_seed_nans=True,  # Seed sparse data with NaNs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=5.0e-3,
        integrator="rk2",
        ncycle_out=1,
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",
        numlevel=0,
        # derefine_count = 5,
        nghost=2,
        nx1=128,  # Number of zones in X1-direction
        x1min=-150.0,  # minimum value of X1
        x1max=150.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=640,  # Number of zones in X2-direction
        x2min=-750.0,  # minimum value of X2
        x2max=750.0,  # maximum value of X2
        ix2_bc="outflow",  # Inner-X2 boundary condition flag
        ox2_bc="outflow",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-3.5,  # minimum value of X3
        x3max=3.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,
        nx2=32,
        nx3=1,
    )

    riot.input(
        "materials",
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.1,
        Cv=3.69e7,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        label="aluminum",
        strong=True,
        strength_model="aluminum_strength",
        eos="aluminum_eos",
        max_bnd_level=-1,
        max_mat_level=-1,
    )

    riot.input(
        "aluminum_strength",
        modelname="epp",
        G0=2.76e11,
        Y0=3.0e9,
        rho_fail=1.0,
    )

    riot.input(
        "aluminum_eos",
        eos_type="Gruneisen",
        C0=5.328e5,
        s1=1.338,
        s2=0.0,
        s3=0.0,
        G0=2.0,
        b=0.0,
        rho0=2.785,
        T0=0.0,
        P0=0.0,
        Cv=1.0e3,
    )

    riot.input(
        "region0",
        name="background",
        mask_type="background",
        matid=0,
        c_m_pressure=1.0e6,
        c_m_rho=1.0e-3,
    )

    riot.input(
        "region1",
        name="top",
        mask_type="inside_rectangle",
        x0=-50.0,
        x1=50.0,
        y0=0.0,
        y1=500.0,
        matid=1,
        c_m_pressure=1.0e6,
        c_m_rho=2.785,
        c_c_bulk_velocity=[0.0, -1.5e4, 0.0],
    )

    riot.input(
        "region2",
        name="bottom",
        mask_type="inside_rectangle",
        x0=-50.0,
        x1=50.0,
        y0=-500.0,
        y1=0.0,
        matid=1,
        c_m_pressure=1.0e6,
        c_m_rho=2.785,
        c_c_bulk_velocity=[0.0, 1.5e4, 0.0],
    )

    riot.input(
        "physics",
        hydro=True,
        strength=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.9,
        amr_interface=False,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
