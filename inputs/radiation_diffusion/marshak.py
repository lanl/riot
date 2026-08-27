#!/usr/bin/env python3
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

import numpy as np
import riot

Tbound = 1.16045e7
T_background = 1.16045e2
rho_background = 3.0
rho_sphere = 3.0
initial_equilibrium_radiation = True
Er_background = 1.0e-1


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input(
        "parthenon/job",
        problem_id="marshak",  # problem ID: basename of output filenames
    )

    riot.input("parthenon/output0", file_type="rst", dt=1.0e-3)

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.temperature", "rmg.Eg"],
        file_type="hdf5",  # Tabular data dump
        dt=5.0e-4,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-20,  # cycle limit
        tlim=4.0e-3,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=100,  # interval for stdout summary info
        dt_init=1.0e-10,
    )

    riot.input(
        "parthenon/mesh",
        refinement="static",
        numlevel=3,
        derefine_count=10,
        multigrid=True,
        base_block_coarsenings=1,
        pack_size=-1,
        nghost=2,
        nx1=128,  # Number of zones in X1-direction
        x1min=-1.0,  # minimum value of X1
        x1max=10.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=1,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=10.0,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-1.0,  # minimum value of X3
        x3max=1.0,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,
        nx2=1,
        nx3=1,
    )

    riot.input("materials", sparse_dealloc=True)

    # We mock up a constant material temperature boundary
    # with a region of material with extremely high
    # specific heat. This is much easier than implementing
    # such a boundary condition explicitly
    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=8.61733e200,
        opac_a="powerlaw",
        kappa0_a=1.56272e22,
        kappa_Tpower_a=-3.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=8.61733e10,
        opac_a="powerlaw",
        kappa0_a=1.56272e22,
        kappa_Tpower_a=-3.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=3.0,
        c_c_bulk_velocity=[0.0, 0.0, 0.0],
        c_m_temperature=Tbound,
    )

    riot.input(
        "region1",
        mask_type="inside_rectangle",
        matid=1,
        x0=0.0,
        c_m_rho=3.0,
        c_c_bulk_velocity=[0.0, 0.0, 0.0],
        c_m_temperature=T_background,
    )

    riot.input(
        "physics",
        hydro=True,
        fixed_fluid=True,
        multigroup_diffusion=True,
        sparse_physics=False,
    )

    riot.input("hydro", recon="plm", cfl=0.8, amr_interface=True)

    riot.input(
        "diffusion",
        update_temperature=True,
        nr_tolerance=1.0e-3,
        nriter=20,
        local_nriter=0,
        print_per_nr_step=False,
        opacity_temp_min=0.0,
        boundary_condition="constant_temperature",
        boundary_T=[Tbound, T_background, 1.0, 1.0, 1.0, 1.0],
        # Timestep control parameters
        timestep_temperature_scale=1.16e6,
        temperature_fractional_change_target=0.2,
        timestep_min_temperature=1.0e5,
        maximum_timestep_reduction_factor=1.0,
    )

    riot.input(
        "diffusion/linear_solver_params",
        relative_residual_tolerance=1e-5,
        absolute_residual_tolerance=1e-12,
        volume_weight=True,
        max_coarsenings=100000,
        precondition=True,
        preconditioner="Multigrid",
        max_iterations=1000,
        presmoother="SRJ1",
        postsmoother="SRJ2",
        print_per_step=False,
        block_interior_prolongation="Constant",
    )

    riot.input(
        "parthenon/static_refinement0",
        x1min=0.0,
        x1max=1.0,
        x2min=0.0,
        x2max=10.0,
        x3min=-0.25,
        x3max=0.25,
        level=3,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
