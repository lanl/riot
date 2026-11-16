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
# Brio & Wu (1988) MHD shock tube (Heaviside-Lorentz units), along direction --dir.
# The normal field is uniform and the tangential field lies along the next cyclic
# direction, so the X2 and X3 runs are cyclic permutations of the X1 run.

import argparse
import riot


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=int, default=1, choices=[1, 2, 3], help="Shock direction"
    )
    parser.add_argument(
        "--nx", type=int, default=256, help="Number of cells along the tube"
    )
    parser.add_argument("--tlim", type=float, default=0.1, help="Final time")
    parser.add_argument(
        "--full3d", action="store_true", help="Use 4 cells in all transverse directions"
    )
    args = parser.parse_args()

    # Shock along nd; transverse directions below nd need > 1 cell so that the X2 and
    # X3 runs see the same grid as the X1 run. Transverse cells are cubic.
    nd = args.dir - 1
    nx = [
        args.nx if d == nd else (4 if (d < nd or args.full3d) else 1) for d in range(3)
    ]
    xmax = [nx[d] / args.nx for d in range(3)]
    bc = ["outflow" if d == nd else "periodic" for d in range(3)]

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="brio_wu")

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
        nx1=nx[0],
        x1min=0.0,
        x1max=xmax[0],
        ix1_bc=bc[0],
        ox1_bc=bc[0],
        nx2=nx[1],
        x2min=0.0,
        x2max=xmax[1],
        ix2_bc=bc[1],
        ox2_bc=bc[1],
        nx3=nx[2],
        x3min=0.0,
        x3max=xmax[2],
        ix3_bc=bc[2],
        ox3_bc=bc[2],
    )

    riot.input("parthenon/meshblock", nx1=nx[0], nx2=nx[1], nx3=nx[2])

    riot.input("material0", eos_type="IdealGas", Gamma=2.0, Cv=1.0)

    riot.input("physics", hydro=True)

    riot.input("hydro", recon="plm", cfl=0.4, mhd=True)

    riot.input("regions", file=__file__, name="brio_wu")

    axis = ["x", "y", "z"][nd]
    riot.input(
        "region0",
        name="left",
        mask_type="inside_rectangle",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
        **{axis + "1": 0.5},
    )

    riot.input(
        "region1",
        name="right",
        mask_type="inside_rectangle",
        matid=0,
        c_m_rho=0.125,
        c_m_pressure=0.1,
        **{axis + "0": 0.5},
    )

    riot.input("brio_wu/params", nd=nd)


class brio_wu:
    def f_b_bfield(self, pos, b):
        comp = [self.x, self.y, self.z]
        n = comp[self.nd]
        t = comp[(self.nd + 1) % 3]
        b[:, :] = 0.0
        b[:, n] = 0.75
        b[:, t] = 1.0
        b[pos[:, n] > 0.5, t] = -1.0


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
