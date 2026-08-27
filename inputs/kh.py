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
import argparse


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", action="store_true", help="Enable mix")
    parser.add_argument(
        "--K0", required=False, default=1.0e-2, help="Initial turbulent kinetic energy"
    )
    parser.add_argument(
        "--S0", required=False, default=1.0e-4, help="Sets initial S_T and S_D"
    )
    parser.add_argument(
        "--amp",
        required=False,
        default=1.0e-2,
        help="Amplitude of initial velocity perturbation",
    )
    args = parser.parse_args()

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="kh")

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
        dt=0.1,
    )

    riot.input("parthenon/time", nlim=-1, tlim=5.0, integrator="rk2", ncycle_out=10)

    riot.input(
        "parthenon/mesh",
        nx1=128,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=128,
        x2min=0.0,
        x2max=1.0,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input("parthenon/meshblock", nx1=64, nx2=32, nx3=1)

    riot.input("material0", eos_type="IdealGas", Gamma=1.4, Cv=1.0e12)

    riot.input("material1", eos_type="IdealGas", Gamma=1.4, Cv=1.0e12)

    riot.input("physics", hydro=True, mix=args.mix)

    riot.input("hydro", recon="plm", cfl=0.8)

    riot.input("mix", K0=args.K0, S0=args.S0)

    riot.input(
        "regions",
        nlev_max=5,
        c_m_pressure=1.0,
        c_c_bulk_reynolds_stress=[
            2.0 / 3.0 * args.K0,
            2.0 / 3.0 * args.K0,
            2.0 / 3.0 * args.K0,
            0.0,
            0.0,
            0.0,
        ],
        c_c_bulk_bhr_ST=args.S0,
        c_c_bulk_bhr_SD=args.S0,
    )

    riot.input(
        "region0",
        name="lower",
        mask_type="inside_rectangle",
        y1=0.5,
        matid=0,
        c_m_rho=1.0,
    )

    riot.input(
        "region1",
        name="upper",
        mask_type="inside_rectangle",
        y0=0.5,
        matid=1,
        c_m_rho=2.0,
    )

    dy = (
        riot.input["parthenon/mesh"]["x2max"] - riot.input["parthenon/mesh"]["x2min"]
    ) / riot.input["parthenon/mesh"]["nx2"]
    riot.input("lower/params", amp=args.amp, dy=dy)
    riot.input("upper/params", amp=args.amp, dy=dy)


def bfunc(y, dy):
    bval = 0.0 * y[:]
    bval[np.abs(y - 0.5) < dy] = 0.125
    return bval


class lower:
    def __init__(self):
        self.vx = -0.5

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = self.vx
        vel[:, self.y] = (
            self.amp
            * np.sin(2.0 * np.pi * pos[:, self.x])
            * np.sin(2.0 * np.pi * pos[:, self.y])
        )
        vel[:, self.z] = 0.0

    def c_c_bulk_bhr_b(self, pos, b):
        b[:] = bfunc(pos[:, self.y], self.dy)


class upper:
    def __init__(self):
        self.vx = 0.5

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = self.vx
        vel[:, self.y] = (
            self.amp
            * np.sin(2.0 * np.pi * pos[:, self.x])
            * np.sin(2.0 * np.pi * pos[:, self.y])
        )
        vel[:, self.z] = 0.0

    def c_c_bulk_bhr_b(self, pos, b):
        b[:] = bfunc(pos[:, self.y], self.dy)


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
