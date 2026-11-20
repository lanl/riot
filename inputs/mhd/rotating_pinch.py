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
#
# Rotating screw-pinch equilibrium in cylindrical (r, z) coordinates (requires a build
# with PARTHENON_COORDINATES=UniformCylindrical). Uniform density in rigid rotation
# v_phi = Omega r, with uniform axial current, B_phi = B0 r, and uniform B_z. Radial force
# balance,
#   dp/dr = rho v_phi^2 / r - (B_phi / r) d(r B_phi)/dr = rho Omega^2 r - 2 B0^2 r,
# gives p = P0 + (rho Omega^2 / 2 - B0^2) r^2. The centrifugal and magnetic hoop stress
# geometric source terms hold the equilibrium, which is uniform in z.

import argparse
import numpy as np
import riot

RHO = 1.0
OMEGA = 1.0
B0 = 0.5
BZ = 1.0
P0 = 2.0


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=64, help="Number of cells in r")
    parser.add_argument("--tlim", type=float, default=0.2, help="Final time")
    args = parser.parse_args()

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="rotating_pinch")

    riot.input("parthenon/output0", file_type="hst", dt=0.01)

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.bulk.bcc",
            "c.c.bulk.divb",
        ],
        file_type="hdf5",
        dt=args.tlim,
    )

    riot.input(
        "parthenon/time", nlim=-1, tlim=args.tlim, integrator="rk2", ncycle_out=100
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        nx1=args.nr,
        x1min=0.5,
        x1max=1.5,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=16,
        x2min=0.0,
        x2max=0.25,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=0.0,
        x3max=2.0 * np.pi,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.0)

    riot.input("physics", hydro=True)

    riot.input("hydro", recon="plm", cfl=0.4, mhd=True)

    riot.input("regions", file=__file__, name="rotating_pinch")

    riot.input(
        "region0",
        name="rotating_pinch",
        mask_type="background",
        matid=0,
        file=__file__,
    )


class rotating_pinch:
    # Cartesian convention in the x-z plane: pos = (r, 0, z) and vector components
    # (v_r, v_phi, v_z), indexed by self.x, self.y, self.z
    def c_m_rho_0(self, pos, rho):
        rho[:] = RHO

    def c_m_pressure_0(self, pos, press):
        r = pos[:, self.x]
        press[:] = P0 + (0.5 * RHO * OMEGA**2 - B0**2) * r**2

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, :] = 0.0
        vel[:, self.y] = OMEGA * pos[:, self.x]

    def f_b_bfield(self, pos, b):
        b[:, :] = 0.0
        b[:, self.y] = B0 * pos[:, self.x]
        b[:, self.z] = BZ


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
