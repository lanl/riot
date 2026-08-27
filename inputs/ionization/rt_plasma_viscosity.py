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


import numpy as np
import riot
import singularity_eos

const = riot.constants()

x1min = -0.5
x1max = 0.5
x2min = -0.5
x2max = 1.0
x3min = -0.5
x3max = 0.5
Lx = x1max - x1min
Ly = x2max - x2min
Lz = x3max - x3min
nx1 = 32
nx2 = 48
nx3 = 1


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="rt_viscous")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.strain_rate",
            "c.c.bulk.face_velocity",
            "c.c.bulk.ion_shear_viscosity",
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
        dt=1e-4,  # time increment between outputs
        ghost_zones=False,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1.5e-3,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=200,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        multigrid=False,
        refinement="none",
        numlevel=4,
        derefine_count=1,
        nx1=nx1,  # Number of zones in X1-direction
        x1min=x1min,  # good for shock width if really is stationary
        x1max=x1max,
        ix1_bc="periodic",  # Inner-X1 boundary condition flag
        ox1_bc="periodic",  # Outer-X1 boundary condition flag
        nx2=nx2,  # Number of zones in X2-direction
        x2min=x2min,  # minimum value of X2
        x2max=x2max,  # maximum value of X2
        ix2_bc="reflecting",  # Inner-X2 boundary condition flag
        ox2_bc="outflow",  # Outer-X2 boundary condition flag
        nx3=nx3,  # Number of zones in X3-direction
        x3min=x3min,  # minimum value of X3
        x3max=x3max,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    riot.input(
        "parthenon/refinement1",
        field="c.c.bulk.temperature",  # the name of the variable we want to refine on
        method="derivative_order_1",  # selects the first derivative method
        refine_tol=1.0e-1,  # tag for refinement if |(dfield/dx)/field| > refine_tol
        derefine_tol=6e-2,  # tag for derefinement if |(dfield/dx)/field| < derefine_tol
        max_level=4,  # if set, limits refinement level from this criteria to no greater than max_level
    )

    riot.input("materials", sparse_dealloc=False)

    riot.material(
        0,
        name="hydrogen0",
        rho=0.002,
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=2.4e8,
        max_level=0,
        mean_atomic_mass=1,
        mean_atomic_number=1,
    )

    riot.material(
        1,
        name="hydrogen1",
        rho=0.010,
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=9.35e7,
        max_level=0,
        mean_atomic_mass=1,
        mean_atomic_number=1,
    )

    for id in range(2):
        name = riot.material[id]["name"]
        eos_name = name + "EOS"
        riot.input(
            "material" + str(id),
            name=riot.material[id]["name"],
            eos=eos_name,
            electron_eos=eos_name,
            max_bnd_level=riot.material[id]["max_level"],
            max_mat_level=riot.material[id]["max_level"],
        )
        mat_eos = riot.material[id]["eos_type"]
        riot.input(
            eos_name,
            eos_type=mat_eos,
            zsplit=True,
            Gamma=riot.material[id]["Gamma"],
            Cv=riot.material[id]["Cv"],
            mean_atomic_mass=riot.material[id]["mean_atomic_mass"],
            mean_atomic_number=riot.material[id]["mean_atomic_number"],
        )

    riot.input(
        "ideal_eos0",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1e-3,
        zsplit=True,
    )

    riot.input(
        "ideal_eos1",
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=1e-3,
        mean_atomic_mass=4,
        mean_atomic_number=2,
        zsplit=True,
    )

    riot.input(
        "physics",
        hydro=True,
        ionization=True,
        gravity=True,
        sparse_physics=False,
        strength=False,
        fixed_fluid=False,
    )

    riot.input(
        "regions",
        nlev_max=4,
        nlev_min=4,
    )

    riot.input("hydro", recon="plm", riemann="hllc", cfl=0.8, amr_interface=True)

    riot.input("gravity", gravity_dim=1, gravity_g=-5e6)

    riot.input(
        "ionization",
        root_tol=1e-20,
        fully_ionized=True,
        # ei coupling
        electron_ion_coupling=True,
        electron_ion_coupling_model="constant",
        tau_ei=1.0e-3,
        coulomb_logarithm="brysk",
        advect_electron_entropy=False,
        # conduction
        electron_thermal_conduction=False,
        electron_conductivity_model="spitzer_volume_average_arithmetic",
        ion_thermal_conduction=False,
        ion_conductivity_model="braginskii",
        # viscosity
        plasma_viscosity=True,
        ion_viscosity_model="constant",
        ion_shear_viscosity=1.0e-1,
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

    riot.input(
        "region0",
        name="smooth",
        mask_type="background",
        matid=[riot.material["hydrogen0"]["id"], riot.material["hydrogen1"]["id"]],
        c_m_rho_hydrogen0=riot.material["hydrogen0"]["rho"],
        c_m_rho_hydrogen1=riot.material["hydrogen1"]["rho"],
        c_m_pressure_hydrogen0=1e6,
        c_m_pressure_hydrogen1=1e6,
    )


class smooth:
    def __init__(self):
        self.A = 0.02
        self.K = 1
        self.f = 2.0 * np.pi * self.K
        self.rho0 = 0.002
        self.rho1 = 0.004
        self.P = 1.0
        self.vx = 0.0
        self.vy = 3e3
        self.vz = 0.0
        self.phase = -np.pi * 0.5
        self.width = 0.1
        self.x = 0
        self.y = 1
        self.z = 2
        self.temperature = 1.0

    # def mask(self, pos):
    # return np.ones(len(pos[:,0]), dtype=bool)

    def c_c_mat_volume_fraction_hydrogen0(self, pos, alpha):
        theta = self.f * (pos[:, self.x] - x1min) / Lx + self.phase
        y_int = 0.50 + self.A * np.sin(theta)
        dydx = self.A * self.f * np.cos(theta)
        # approximate signed distance to interface
        d = (pos[:, self.y] - y_int) / np.sqrt(1.0 + dydx**2 + 1e-15)

        # Smooth Heaviside using tanh; gives exactly 0.5 at d = 0
        s = -1.0
        alpha[:] = 0.5 * (1.0 + s * np.tanh(2.0 * d / self.width))

    def c_c_mat_volume_fraction_hydrogen1(self, pos, alpha):
        self.c_c_mat_volume_fraction_hydrogen0(pos, alpha)
        alpha[:] = 1.0 - alpha[:]

    # def c_c_bulk_temperature(self, pos, temperature):
    # temperature[:] = self.temperature

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = self.vx
        vel[:, self.y] = self.vy
        vel[:, self.z] = self.vz

    # def make_vis_mesh(self):
    # x = np.linspace(x1min, x1max, nx1)
    # y = np.linspace(x2min, x2max, nx2)
    # z = np.linspace(x3min, x3max, nx3)

    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    # return pos, X, Y, Z

    # def visualize(self):
    # pos, X, Y, Z = self.make_vis_mesh()
    # alpha0 = np.zeros_like(pos[:,0])
    # self.c_c_mat_volume_fraction_0(pos, alpha0)
    # alpha0_grid = alpha0.reshape(nx1, nx2)
    # figure(1)
    # clf()
    # pcolormesh(X[:,:,0], Y[:,:,0], alpha0_grid, shading="auto")
    # gca().set_aspect("equal")
    # colorbar()


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
    # s = smooth()
    # s.visualize()
