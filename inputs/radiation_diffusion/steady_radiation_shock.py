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
        problem_id="steady_radiation_shock",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.bulk.temperature",
            "rmg.Egroup",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=0.01,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=0.05,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=100,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        multigrid=True,
        refinement="adaptive",
        numlevel=6,
        derefine_count=10,
        nx1=64,  # Number of zones in X1-direction
        x1min=-0.04,  # minimum value of X1
        x1max=0.04,  # maximum value of X1
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
        "parthenon/meshblock",
        nx1=64,  # Number of cells in each MeshBlock, X1-dir
        nx2=1,  # Number of cells in each MeshBlock, X2-dir
        nx3=1,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "materials",
        sparse_dealloc=True,
        use_general_pte=True,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.666666,
        Cv=1.5,
        opac_a="powerlaw",
        kappa0_a=577.35,
        kappa_Tpower_a=0.0,
        kappa_Rhopower_a=-1.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=1.6666666,
        Cv=1.5,
        opac_a="powerlaw",
        kappa0_a=577.35,
        kappa_Tpower_a=0.0,
        kappa_Rhopower_a=-1.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        mask_type="inside_rectangle",
        matid=0,
        x1=0.0005,
        c_m_rho=1.0,
        c_c_bulk_velocity=[2.0, 0, 0],
        c_m_temperature=0.6,
    )

    riot.input(
        "region1",
        mask_type="inside_rectangle",
        matid=1,
        x0=0.0005,
        c_m_rho=2.285714,
        c_c_bulk_velocity=[0.875000, 0, 0],
        c_m_temperature=1.246875,
    )

    riot.input(
        "physics",
        hydro=True,
        multigroup_diffusion=True,
        fixed_fluid=False,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        # riemann = hllc,
        amr_interface=True,
        temp_floor=1.0e-20,
    )

    riot.input(
        "diffusion",
        update_temperature=True,
        nr_tolerance=1.0e-4,
        nriter=15,
        local_nriter=0,
        print_per_nr_step=False,
        opacity_temp_min=1.0e0,
        a_radiation=7.716e-4,
        c_light=1732.05,
        boundary_condition="constant_temperature",
        boundary_T=[0.6, 1.246875, 1.0, 1.0, 1.0, 1.0],
        timestep_temperature_scale=1.0e5,
        temperature_fractional_change_target=0.1,
        timestep_min_temperature=1.0e2,
        cfl=5e5,
        use_amr_criteria=True,
        amr_min_temperature=0.0,
        amr_min_density=0.0,
        amr_threshold=0.01,
    )

    riot.input(
        "diffusion/linear_solver_params",
        relative_residual_tolerance=1e-8,
        absolute_residual_tolerance=1e-16,
        volume_weight=True,
        max_coarsenings=100000,
        precondition=True,
        preconditioner="Multigrid",
        max_iterations=100,
        presmoother="SRJ1",
        postsmoother="SRJ2",
        block_interior_prolongation="Constant",
        print_per_step=False,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
