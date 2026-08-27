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
        problem="conduction_analytic",
        trap_fpes=False,
        verbose=False,
    )

    riot.input(
        "parthenon/job",
        problem_id="conduction_analytic",
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.mat.ionization_zbar",
            "c.c.bulk.electron_internal_energy",
            "c.c.bulk.electron_entropy",
            "c.c.bulk.temperature",
            "c.c.bulk.electron_temperature",
            "c.c.bulk.electron_pressure",
            "c.c.bulk.electron_number_density",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=1e-2,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1e-1,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=20,  # interval for stdout summary info
        dt_force=1e-3,
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        multigrid=True,
        nx1=128,  # Number of zones in X1-direction
        x1min=-5.0,  # minimum value of X1
        x1max=5.0,  # maximum value of X1
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
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
        nx1=64,
        nx2=1,
        nx3=1,
    )

    riot.input(
        "materials",
        sparse_dealloc=False,
    )

    riot.input(
        "material0",
        eos="eos",
        electron_eos="eos",
    )

    riot.input(
        "eos",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=2.0,
        mean_atomic_mass=1,
        mean_atomic_number=1,
    )

    riot.input(
        "conduction_analytic",
        rho0=4.0,
        nx=4,
        ny=0,
        nz=0,
        amplitude=0.5,
        constant=1.0,
    )

    riot.input(
        "physics",
        hydro=True,
        ionization=True,
        sparse_physics=False,
        strength=False,
        fixed_fluid=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        riemann="hllc",
        cfl=0.8,
        amr_interface=True,
    )

    riot.input(
        "ionization",
        root_tol=1e-20,
        fully_ionized=True,
        electron_ion_coupling=False,
        electron_thermal_conduction=True,
        electron_conductivity_model="constant",
        electron_conductivity=5.0,
    )

    riot.input(
        "ionization/linear_solver_params",
        residual_tolerance=1.0e-12,
        max_coarsenings=10000,
        precondition=True,
        max_iterations=200,
        smoother="SRJ2",
        print_per_step=False,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
