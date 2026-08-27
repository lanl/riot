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
import singularity_eos as eos

const = riot.constants()


def make_input():

    riot.input(
        "riot",
        problem="region_pgen",
    )

    riot.input("parthenon/job", problem_id="rose")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "my_bulk_scalar",
            "prim.my_mat_scalar",
            "c.c.bulk.Er",
            "c.c.bulk.temperature",
            "c.c.mat.rho",
        ],
        file_type="hdf5",
        dt=0.1,
        sparse_seed_nans=True,
    )

    riot.input("parthenon/time", nlim=1, tlim=0.1, integrator="rk2", ncycle_out=1)

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=5,
        derefine_count=5,
        nghost=2,
        pack_size=1,
        nx1=32,  # Number of zones in X1-direction
        x1min=-1.2,  # minimum value of X1
        x1max=1.2,  # maximum value of X1
        ix1_bc="reflecting",  # Inner-X1 boundary condition flag
        ox1_bc="reflecting",  # Outer-X1 boundary condition flag
        nx2=32,  # Number of zones in X2-direction
        x2min=-1.2,  # minimum value of X2
        x2max=1.2,  # maximum value of X2
        ix2_bc="reflecting",  # Inner-X2 boundary condition flag
        ox2_bc="reflecting",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.01,  # minimum value of X3
        x3max=0.01,  # maximum value of X3
        ix3_bc="reflecting",  # Inner-X3 boundary condition flag
        ox3_bc="reflecting",  # Outer-X3 boundary condition flag
        multigrid=True,
    )

    riot.input("parthenon/meshblock", nx1=32, nx2=32, nx3=1)

    riot.input(
        "parthenon/refinement1",
        method="derivative_order_1",
        field="my_bulk_scalar",
        refine_tol=0.1,
        derefine_tol=0.01,
    )

    riot.input("materials", sparse_dealloc=True)

    riot.input(
        "material0",
        nphase=2,
        eos0="Eos0",
        eos1="Eos1",
        # eos_type = "IdealGas",
        # Gamma = 1.5,
        # Cv = 1.e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input("Eos0", eos_type="IdealGas", Gamma=1.5, Cv=1.0e-3)

    riot.input("Eos1", eos_type="IdealGas", Gamma=1.5, Cv=1.0e-3)

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "regions",
        nlev_min=0,
        nlev_max=5,
        # c_c_bulk_Er = 1.0,
        # equilibrium_radiation = False
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
        # passive_scalars = "my_bulk_scalar",
        c_m_phase_fraction=[0.0, 1.0],
        # c_c_bulk_Er = 1.0,
        # equilibrium_radiation = False
    )

    riot.input(
        "region1",
        name="rose",
        mask_type="python",
        # file = __file__,
        # matid = 1,
        matid=0,
        # c_m_rho=2.0,
        # c_m_pressure=1.0,
        passive_scalars="my_bulk_scalar",
        prefer_python_init=False,
        c_m_phase_fraction=[1.0, 0.0],
    )

    riot.input(
        "physics",
        hydro=True,
        scalars=True,
        sparse_physics=True,
        sparse_physics_threshold=1.0e-30,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        riemann="hllc",
        amr_interface=True,
        lm_correction=True,
    )

    riot.input("scalars0", label="my_bulk_scalar")

    riot.input(
        "scalars1",
        label="my_mat_scalar",
        matid=1,
    )

    riot.input(
        "diffusion",
        nriter=5,
        print_per_nr_step=True,
        flux_limit=True,
    )

    riot.input(
        "diffusion/linear_solver_params",
        residual_tolerance=1.0e-5,
        max_coarsenings=10000,
        precondition=True,
        max_iterations=20,
        smoother="SRJ2",
        print_per_step=False,
    )

    riot.input("rose/params", mode="sin", a=0.6, k=3)


class rose:
    def __init__(self):
        make_input()
        self.eos = riot.EOS("material1")
        # pass

    def mask(self, pos):
        r = np.hypot(pos[:, self.x], pos[:, self.y])
        theta = np.atan2(pos[:, self.y], pos[:, self.x])

        if self.mode == "cos":
            base = np.cos(self.k * theta)
        elif self.mode == "sin":
            base = np.sin(self.k * theta)
        else:
            raise ValueError("mode must be 'cos' or 'sin'")

        rose_radius = np.abs(np.abs(self.a) * base)  # |a*f(kθ)| fills all petals
        inclusive = True
        eps = 1.0e-12
        if inclusive:
            return r <= rose_radius + eps
        else:
            return r < rose_radius - eps

    def rho_func(self, pos):
        return 1.0 + 0.5 * (np.power(pos[:, self.x], 2) + np.power(pos[:, self.y], 2))

    def c_m_rho(self, pos, rho):
        rho[:] = self.rho_func(pos)

    def c_m_pressure(self, pos, press):
        rho = self.rho_func(pos)
        self.eos.PressureFromDensityInternalEnergy(rho, 2.0 / rho, press)
        # press[:] = 1.0

    def c_c_bulk_Er(self, pos, erad):
        erad[:] = (
            const.ar * 2000.0**4
        )  # * (1.0 + np.power(pos[:, self.x], 2) + np.power(pos[:, self.y], 2))


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
