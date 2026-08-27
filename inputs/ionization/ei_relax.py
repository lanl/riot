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
        problem="ei_relax",
        trap_fpes=False,
    )

    riot.input(
        "parthenon/job",
        problem_id="ei_relax",
    )

    riot.input(
        "ei_relax",
        rho=1.0,
        Te0=2.320905001233140e07,  # 2000 eV
        Ti0=1.160452500616570e07,  # 1000 eV
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.mat.ionization_zbar",
            "c.c.bulk.pressure",
            "c.c.volume_fraction",
            "c.c.bulk.total_material_energy",
            "c.c.bulk.electron_internal_energy",
            "c.c.bulk.temperature",
            "c.c.bulk.electron_temperature",
            "c.c.bulk.electron_pressure",
            "c.c.bulk.electron_number_density",
            "c.c.mat.internal_energy",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=0.1e-11,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=2.0e-11,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1000,  # interval for stdout summary info
        dt_force=1e-15,
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        nx1=16,  # Number of zones in X1-direction
        x1min=-0.5,  # minimum value of X1
        x1max=0.5,  # maximum value of X1
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
        "materials",
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos="hydrogen",
        electron_eos="hydrogen",
    )

    riot.input(
        "hydrogen",
        eos_type="IdealGas",
        zsplit=True,
        Cv=70155830.33230154,
        Gamma=1.66666666666667,
        mean_atomic_mass=16,
        mean_atomic_number=8,
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
        "ionization",
        root_tol=1e-20,
        fully_ionized=True,
        electron_ion_coupling=True,
        electron_ion_coupling_model="landau_spitzer",
        coulomb_logarithm="brysk",
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
