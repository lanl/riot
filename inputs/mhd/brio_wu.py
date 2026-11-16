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


axes = ["x", "y", "z"]


def make_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--tlim", type=float, default=0.1)
    parser.add_argument(
        "--full3d", action="store_true", help="Use 4 cells in all transverse directions"
    )
    args, _ = parser.parse_known_args()

    nd = args.dir - 1
    riot.input("riot", problem="region_pgen")
    riot.input("parthenon/job", problem_id="brio_wu")
    riot.input(
        "parthenon/output0",
        file_type="hst",
        dt=0.01,
    )
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

    mesh = {}
    block = {}
    for d in range(3):
        n = str(d + 1)
        if d == nd:
            mesh.update(
                {
                    "nx" + n: args.nx,
                    "x" + n + "min": 0.0,
                    "x" + n + "max": 1.0,
                    "ix" + n + "_bc": "outflow",
                    "ox" + n + "_bc": "outflow",
                }
            )
            block["nx" + n] = args.nx
        else:
            # transverse directions below the shock direction need > 1 cell
            ntr = 4 if (d < nd or args.full3d) else 1
            mesh.update(
                {
                    "nx" + n: ntr,
                    "x" + n + "min": 0.0,
                    "x" + n + "max": ntr / args.nx,
                    "ix" + n + "_bc": "periodic",
                    "ox" + n + "_bc": "periodic",
                }
            )
            block["nx" + n] = ntr
    riot.input("parthenon/mesh", nghost=4, **mesh)
    riot.input("parthenon/meshblock", **block)

    riot.input("material0", eos_type="IdealGas", Gamma=2.0, Cv=1.0)
    riot.input("physics", hydro=True)
    riot.input("hydro", recon="plm", cfl=0.4, mhd=True)

    riot.input("regions", file=__file__, name="brio_wu")
    riot.input(
        "region0",
        name="left",
        mask_type="inside_rectangle",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
        **{axes[nd] + "1": 0.5},
    )
    riot.input(
        "region1",
        name="right",
        mask_type="inside_rectangle",
        matid=0,
        c_m_rho=0.125,
        c_m_pressure=0.1,
        **{axes[nd] + "0": 0.5},
    )
    riot.input("brio_wu/params", nd=nd)


class brio_wu:
    def f_b_bfield(self, pos, b):
        nt = (self.nd + 1) % 3
        b[:, :] = 0.0
        b[:, self.nd] = 0.75
        b[:, nt] = 1.0
        b[pos[:, self.nd] > 0.5, nt] = -1.0


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
