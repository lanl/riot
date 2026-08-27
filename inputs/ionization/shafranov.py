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
import singularity_eos

const = riot.constants()


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="shafranov")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.mat.ionization_zbar",
            "c.c.bulk.pressure",
            "c.c.mat.volume_fraction",
            "c.c.bulk.electron_entropy",
            "c.c.bulk.temperature",
            "c.c.bulk.electron_temperature",
            "c.c.bulk.electron_pressure",
            "c.c.bulk.electron_number_density",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=1.0e-10,  # time increment between outputs
        ghost_zones=False,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1e-10,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=20,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        multigrid=True,
        refinement="adaptive",
        numlevel=4,
        derefine_count=1,
        nx1=128,  # Number of zones in X1-direction
        x1min=-0.002,  # good for shock width if really is stationary
        x1max=0.015,
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

    riot.input("parthenon/meshblock", nx1=16, nx2=1, nx3=1)

    riot.input(
        "parthenon/refinement1",
        field="c.c.bulk.temperature",  # the name of the variable we want to refine on
        method="derivative_order_1",  # selects the first derivative method
        refine_tol=1.0e-1,  # tag for refinement if |(dfield/dx)/field| > refine_tol
        derefine_tol=6e-2,  # tag for derefinement if |(dfield/dx)/field| < derefine_tol
        max_level=4,  # if set, limits refinement level from this criteria to no greater than max_level
    )

    riot.input("materials", sparse_dealloc=False)

    riot.input("material0", eos="ideal_eos0", electron_eos="ideal_eos0")

    riot.input("material1", eos="ideal_eos1", electron_eos="ideal_eos1")

    riot.input(
        "ideal_eos0",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1.5 * const.kb * const.na * 2,  # 249443092.41996,  # 3./2.*kberg*na*2
        mean_atomic_mass=1,
        mean_atomic_number=1,
        zsplit=True,
    )

    riot.input(
        "ideal_eos1",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1.5
        * 3.0
        * const.kb
        * const.na
        / 4.0,  # 93541159.657485 # 3/2 (zbar + 1) * kberg * N_A / A
        mean_atomic_mass=4,
        mean_atomic_number=2,
        zsplit=True,
    )

    riot.input(
        "physics",
        hydro=True,
        ionization=True,
        sparse_physics=False,
        strength=False,
        fixed_fluid=False,
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=0.004491770812594265,
        c_m_temperature=1013258.0049343018,
        c_c_bulk_velocity=[12798687.555072872, 0.0, 0.0],
    )

    riot.input(
        "region1",
        mask_type="inside_rectangle",
        x0=0.0,
        matid=1,
        c_m_rho=0.002,
        c_m_temperature=500000.0,
    )

    riot.input("hydro", recon="plm", riemann="hllc", cfl=0.8, amr_interface=True)

    riot.input(
        "ionization",
        root_tol=1e-20,
        fully_ionized=True,
        # ei coupling
        electron_ion_coupling=True,
        electron_ion_coupling_model="landau_spitzer",
        coulomb_logarithm="brysk",
        advect_electron_entropy=True,
        # conduction
        electron_thermal_conduction=True,
        electron_conductivity_model="spitzer_volume_average_arithmetic",
        ion_thermal_conduction=True,
        ion_conductivity_model="braginskii",
        timestep_control="relative",
        fractional_change_scale=0.5,
        # viscosity
        plasma_viscosity=True,
        ion_viscosity_model="fokker_planck_landau",
    )

    riot.input(
        "ionization/linear_solver_params",
        residual_tolerance=1.0e-3,
        max_coarsenings=10000,
        precondition=True,
        max_iterations=200,
        smoother="SRJ2",
        print_per_step=False,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
