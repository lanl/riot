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

ndim = 1
x1min = 0.0
x1max = 1.0
x2min = 0.0
x2max = 1.0
x3min = 0.0
x3max = 1.0
Lx = x1max - x1min
Ly = x2max - x2min
Lz = x3max - x3min


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="advection")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
            "c.c.mat.volume_fraction",
        ],
        file_type="hdf5",
        dt=0.05,
    )

    riot.input("parthenon/time", nlim=-1, tlim=1.0, integrator="rk2", ncycle_out=10)

    riot.input(
        "parthenon/mesh",
        nghost=4,
        nx1=128,
        x1min=x1min,
        x1max=x1max,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=128 if ndim > 1 else 1,
        x2min=x2min,
        x2max=x2max,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=128 if ndim == 3 else 1,
        x3min=x3min,
        x3max=x3max,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input("material0", eos_type="IdealGas", Gamma=1.4, Cv=1.0e12)

    riot.input("material1", eos_type="IdealGas", Gamma=1.4, Cv=1.0e12)

    riot.input("physics", hydro=True)

    riot.input("hydro", recon="weno5", cfl=0.8)

    riot.input(
        "region0", name="smooth", mask_type="background", matid=[0, 1], file=__file__
    )


class smooth:
    def __init__(self):
        self.A = 0.25
        self.K = 1
        self.f = 2.0 * np.pi * self.K
        self.rho = 1.0
        self.P = 1.0
        self.vx = 1.0
        self.vy = 0.0
        self.vz = 0.0

    def sx(self, pos):
        xoff = pos[:, self.x] - x1min
        yoff = pos[:, self.y] - x2min
        zoff = pos[:, self.z] - x3min
        val = self.A * np.sin(self.f * xoff / Lx)
        if ndim > 1:
            val *= np.sin(self.f * yoff / Ly)
        if ndim > 2:
            val *= np.sin(self.f * zoff / Lz)
        return val

    def c_c_mat_volume_fraction_0(self, pos, alpha):
        alpha[:] = 0.5 + self.sx(pos)

    def c_c_mat_volume_fraction_1(self, pos, alpha):
        self.c_c_mat_volume_fraction_0(pos, alpha)
        alpha[:] = 1.0 - alpha[:]

    def c_m_rho_0(self, pos, rho):
        rho[:] = self.rho

    def c_m_rho_1(self, pos, rho):
        rho[:] = self.rho

    def c_m_pressure_0(self, pos, press):
        press[:] = self.P

    def c_m_pressure_1(self, pos, press):
        press[:] = self.P

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = self.vx
        vel[:, self.y] = self.vy
        vel[:, self.z] = self.vz


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
