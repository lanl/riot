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
# Orszag-Tang vortex on [0,1]^2 (Heaviside-Lorentz units, as in Athena++), with the
# field set from an edge-centered vector potential so that div B = 0 initially. With
# --amr the mesh refines on density gradients up to --numlevel levels; --nblock sets the
# meshblock size (default nx/2) and --dt the hdf5 output interval.

import argparse
import numpy as np
import riot

gamma = 5.0 / 3.0
B0 = 1.0 / np.sqrt(4.0 * np.pi)
d0 = 25.0 / (36.0 * np.pi)
p0 = 5.0 / (12.0 * np.pi)


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128, help="Number of cells per side")
    parser.add_argument("--tlim", type=float, default=0.5, help="Final time")
    parser.add_argument(
        "--nblock", type=int, default=None, help="Meshblock size (default nx/2)"
    )
    parser.add_argument(
        "--amr", action="store_true", help="Refine on density gradients"
    )
    parser.add_argument(
        "--numlevel", type=int, default=3, help="Number of refinement levels"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="hdf5 output interval")
    args = parser.parse_args()

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="orszag_tang")

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
        dt=args.dt,
    )

    riot.input(
        "parthenon/time", nlim=-1, tlim=args.tlim, integrator="rk2", ncycle_out=100
    )

    riot.input(
        "parthenon/mesh",
        nghost=4,
        nx1=args.nx,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=args.nx,
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

    if args.amr:
        riot.input("parthenon/mesh", refinement="adaptive", numlevel=args.numlevel)
        riot.input(
            "parthenon/refinement0",
            method="derivative_order_1",
            field="c.c.bulk.rho",
            refine_tol=0.1,
            derefine_tol=0.03,
        )

    nb = args.nblock if args.nblock is not None else args.nx // 2
    riot.input("parthenon/meshblock", nx1=nb, nx2=nb, nx3=1)

    riot.input("material0", eos_type="IdealGas", Gamma=gamma, Cv=1.0)

    riot.input("physics", hydro=True, mhd=True)

    riot.input("hydro", recon="plm", cfl=0.4)

    riot.input("regions", file=__file__, name="orszag_tang")

    riot.input(
        "region0",
        name="orszag_tang",
        mask_type="background",
        matid=0,
        file=__file__,
    )


class orszag_tang:
    def c_m_rho_0(self, pos, rho):
        rho[:] = d0

    def c_m_pressure_0(self, pos, press):
        press[:] = p0

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = -np.sin(2.0 * np.pi * pos[:, self.y])
        vel[:, self.y] = np.sin(2.0 * np.pi * pos[:, self.x])
        vel[:, self.z] = 0.0

    def f_b_vector_potential(self, pos, a):
        x = pos[:, self.x]
        y = pos[:, self.y]
        a[:, :] = 0.0
        a[:, self.z] = B0 * (
            np.cos(4.0 * np.pi * x) / (4.0 * np.pi)
            + np.cos(2.0 * np.pi * y) / (2.0 * np.pi)
        )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
