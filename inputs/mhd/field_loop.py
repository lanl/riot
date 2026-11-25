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
# Gardiner & Stone field loop advection: a weak loop of radius R in uniform flow, set from
# a vector potential A = A0 max(R - r, 0) along the loop axis.
#   2D (GS05): [-1,1]x[-0.5,0.5], flow (2, 1), A along z. --ny sets the number of cells in
#     y (default nx/2, i.e. square cells).
#   3D (--3d, GS08): [-0.5,0.5]^2x[-1,1], flow (1, 1, 2). The loop axis is tilted in the
#     x-z plane so that it is not aligned with the grid, and the loop crosses the domain
#     periodically. The field along the loop axis must stay zero.
# The loop returns to its initial position at t = 1. With --amr the mesh refines on the
# magnetic field so the loop crosses fine/coarse boundaries.

import argparse
import numpy as np
import riot

A0 = 1.0e-3
R = 0.3


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128, help="Number of cells in x")
    parser.add_argument(
        "--ny", type=int, default=None, help="Number of cells in y (2D only)"
    )
    parser.add_argument("--tlim", type=float, default=2.0, help="Final time")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="3D loop")
    parser.add_argument("--amr", action="store_true", help="Refine on the field")
    args = parser.parse_args()

    if args.three_d:
        nx = [args.nx, args.nx, 2 * args.nx]
        xmin = [-0.5, -0.5, -1.0]
        xmax = [0.5, 0.5, 1.0]
    else:
        nx = [args.nx, args.ny if args.ny is not None else args.nx // 2, 1]
        xmin = [-1.0, -0.5, -0.5]
        xmax = [1.0, 0.5, 0.5]

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="field_loop")

    riot.input("parthenon/output0", file_type="hst", dt=0.01)

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.bulk.bfield",
            "c.c.bulk.divb",
        ],
        file_type="hdf5",
        dt=0.5,
    )

    riot.input(
        "parthenon/time", nlim=-1, tlim=args.tlim, integrator="rk2", ncycle_out=100
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        nx1=nx[0],
        x1min=xmin[0],
        x1max=xmax[0],
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=nx[1],
        x2min=xmin[1],
        x2max=xmax[1],
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=nx[2],
        x3min=xmin[2],
        x3max=xmax[2],
        ix3_bc="periodic",
        ox3_bc="periodic",
    )
    if args.amr:
        riot.input("parthenon/mesh", refinement="adaptive", numlevel=2)

    riot.input(
        "parthenon/meshblock",
        nx1=16,
        nx2=16,
        nx3=16 if args.three_d else 1,
    )

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.0)

    riot.input("physics", hydro=True, mhd=True)

    riot.input("hydro", recon="plm", cfl=0.4)

    if args.amr:
        # Refine where B_x > 0 or B_y > 0, so fine/coarse boundaries cut through the loop
        for n in range(2):
            riot.input(
                "parthenon/refinement" + str(n),
                field="c.c.bulk.bfield",
                vector_i=n,
                method="magnitude",
                refine_tol=0.2 * A0,
                derefine_tol=0.05 * A0,
            )

    riot.input("regions", file=__file__, name="field_loop")

    riot.input(
        "region0",
        name="field_loop",
        mask_type="background",
        matid=0,
        file=__file__,
    )

    # Tilt of the 3D loop axis from z toward x, chosen so that the loop is periodic
    tilt = np.arctan((xmax[0] - xmin[0]) / (xmax[2] - xmin[2])) if args.three_d else 0.0
    riot.input(
        "field_loop/params",
        three_d=int(args.three_d),
        sin_tilt=np.sin(tilt),
        cos_tilt=np.cos(tilt),
        wavelength=(xmax[0] - xmin[0]) * np.cos(tilt),
    )


class field_loop:
    def c_m_rho_0(self, pos, rho):
        rho[:] = 1.0

    def c_m_pressure_0(self, pos, press):
        press[:] = 1.0

    def c_c_bulk_velocity(self, pos, vel):
        if self.three_d:
            vel[:, self.x] = 1.0
            vel[:, self.y] = 1.0
            vel[:, self.z] = 2.0
        else:
            vel[:, self.x] = 2.0
            vel[:, self.y] = 1.0
            vel[:, self.z] = 0.0

    def f_b_vector_potential(self, pos, a):
        # distance from the loop axis, which passes through the origin along
        # (-sin_tilt, 0, cos_tilt); in 3D the axis repeats every wavelength in x'
        xp = self.cos_tilt * pos[:, self.x] + self.sin_tilt * pos[:, self.z]
        if self.three_d:
            xp = (xp + 0.5 * self.wavelength) % self.wavelength - 0.5 * self.wavelength
        r = np.sqrt(xp**2 + pos[:, self.y] ** 2)
        amag = A0 * np.maximum(R - r, 0.0)
        a[:, self.x] = -self.sin_tilt * amag
        a[:, self.y] = 0.0
        a[:, self.z] = self.cos_tilt * amag


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
