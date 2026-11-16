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
# Gardiner & Stone (2005) field loop advection on [-1,1]x[-0.5,0.5]: a weak loop of
# radius R in uniform flow, set from A_z = A0 max(R - r, 0). With --amr the mesh refines
# on the magnetic field so the loop crosses fine/coarse boundaries.

import argparse
import numpy as np
import riot


A0 = 1.0e-3
R = 0.3


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--tlim", type=float, default=2.0)
    parser.add_argument("--amr", action="store_true")
    args, _ = parser.parse_known_args()

    riot.input("riot", problem="region_pgen")
    riot.input("parthenon/job", problem_id="field_loop")
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
        dt=0.5,
    )
    riot.input(
        "parthenon/time", nlim=-1, tlim=args.tlim, integrator="rk2", ncycle_out=100
    )
    mesh = dict(
        nghost=4,
        nx1=args.nx,
        x1min=-1.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=args.nx // 2,
        x2min=-0.5,
        x2max=0.5,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )
    if args.amr:
        mesh.update(refinement="adaptive", numlevel=2)
    riot.input("parthenon/mesh", **mesh)
    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.0)
    riot.input("physics", hydro=True)
    riot.input("hydro", recon="plm", cfl=0.4, mhd=True)
    if args.amr:
        # Refine where B_x > 0 or B_y > 0, so fine/coarse boundaries cut through the loop
        for n in range(2):
            riot.input(
                "parthenon/refinement" + str(n),
                field="c.c.bulk.bcc",
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


class field_loop:
    def c_m_rho_0(self, pos, rho):
        rho[:] = 1.0

    def c_m_pressure_0(self, pos, press):
        press[:] = 1.0

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, self.x] = 2.0
        vel[:, self.y] = 1.0
        vel[:, self.z] = 0.0

    def f_b_vector_potential(self, pos, a):
        r = np.sqrt(pos[:, self.x] ** 2 + pos[:, self.y] ** 2)
        a[:, :] = 0.0
        a[:, self.z] = A0 * np.maximum(R - r, 0.0)


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
