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
# Field loop advection in cylindrical (r, z) coordinates (requires a build with
# PARTHENON_COORDINATES=UniformCylindrical): a weak poloidal loop of radius R centered at
# (r, z) = (1, 0), set from A_phi = A0 max(R - d, 0) with d the distance to the center,
# advected in z through a periodic domain. The cells are non-square. With --amr the mesh
# refines on the magnetic field so the loop crosses fine/coarse boundaries.

import argparse
import numpy as np
import riot

A0 = 1.0e-3
R = 0.3
R0 = 1.0


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=64, help="Number of cells in r")
    parser.add_argument("--nz", type=int, default=64, help="Number of cells in z")
    parser.add_argument("--tlim", type=float, default=2.0, help="Final time")
    parser.add_argument("--amr", action="store_true", help="Refine on the field")
    args = parser.parse_args()

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="field_loop_rz")

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

    riot.input(
        "parthenon/mesh",
        nghost=4,
        nx1=args.nr,
        x1min=R0 - 0.5,
        x1max=R0 + 0.5,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=args.nz,
        x2min=-1.0,
        x2max=1.0,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=0.0,
        x3max=2.0 * np.pi,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    if args.amr:
        riot.input("parthenon/mesh", refinement="adaptive", numlevel=2)

    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.0)

    riot.input("physics", hydro=True)

    riot.input("hydro", recon="plm", cfl=0.4, mhd=True)

    if args.amr:
        # Refine where B_r > 0 or B_z > 0, so fine/coarse boundaries cut through the loop
        for n in range(2):
            riot.input(
                "parthenon/refinement" + str(n),
                field="c.c.bulk.bcc",
                vector_i=n,
                method="magnitude",
                refine_tol=0.2 * A0,
                derefine_tol=0.05 * A0,
            )

    riot.input("regions", file=__file__, name="field_loop_rz")

    riot.input(
        "region0",
        name="field_loop_rz",
        mask_type="background",
        matid=0,
        file=__file__,
    )


class field_loop_rz:
    # Cartesian convention in the x-z plane: pos = (r, 0, z) and vector components
    # (v_r, v_phi, v_z), indexed by self.x, self.y, self.z
    def c_m_rho_0(self, pos, rho):
        rho[:] = 1.0

    def c_m_pressure_0(self, pos, press):
        press[:] = 1.0

    def c_c_bulk_velocity(self, pos, vel):
        vel[:, :] = 0.0
        vel[:, self.z] = 1.0

    def f_b_vector_potential(self, pos, a):
        d = np.sqrt((pos[:, self.x] - R0) ** 2 + pos[:, self.z] ** 2)
        a[:, :] = 0.0
        a[:, self.y] = A0 * np.maximum(R - d, 0.0)


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
