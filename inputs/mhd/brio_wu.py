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
"""Brio & Wu (1988) coplanar MHD shock tube at t = 0.08, gamma = 2, 512 zones.

The reference solution contains a compound structure -- a slow shock attached to a
rotational discontinuity -- that is sensitive to the Riemann solver, which is what makes
this a real test rather than a smoke test.

`mhd/mu0 = 1` puts the field in the donor's scale-free normalization, so results are
directly comparable with Artemis AND with the independent Athena++ reference solution
(`artemis/tst/scripts/mhd/athena_bw.std`) with no unit reconciliation. See
plan_histories/artemis_mhd_port/adr/001-units.md.

Structural invariants this setup must preserve exactly, not just approximately: Bx stays at
0.75 everywhere for all time, and Bz, vz and div B stay identically zero. They are checked
in plan_histories/artemis_mhd_port/TEST_LEDGER.md (G3/G4.1, P01.4).

Run `python3 brio_wu.py` to emit brio_wu.rin.
"""

import riot

OUTPUT_VARIABLES = [
    "c.c.bulk.rho",
    "c.c.bulk.pressure",
    "c.c.bulk.velocity",
    "c.c.bulk.magnetic_field",
    "c.c.bulk.div_magnetic_field",
    "c.c.bulk.magnetic_energy",
    "c.c.bulk.internal_energy",
    "c.c.bulk.total_material_energy",
    "c.c.bulk.temperature",
]


def make_input():

    riot.input("riot", problem="mhd_shock_tube")

    riot.input("parthenon/job", problem_id="brio_wu")

    riot.input(
        "parthenon/output1",
        variables=OUTPUT_VARIABLES,
        file_type="hdf5",
        dt=0.08,
    )

    riot.input("parthenon/time", nlim=-1, tlim=0.08, integrator="rk2", ncycle_out=100)

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=512,
        x1min=-0.5,
        x1max=0.5,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=1,
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

    # Two mesh blocks even in 1D, so the face exchange is exercised by the default deck.
    riot.input("parthenon/meshblock", nx1=256, nx2=1, nx3=1)

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=2.0,
        Cv=1.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input("physics", hydro=True, mhd=True)

    riot.input("mhd", mu0=1.0, monitor_divb=True)

    riot.input("hydro", riemann="mhd_hlle", recon="plm", cfl=0.4, amr_interface=False)

    riot.input(
        "mhd_shock_tube",
        x0=0.0,
        rho_l=1.0,
        P_l=1.0,
        rho_r=0.125,
        P_r=0.1,
        bx=0.75,
        by_l=1.0,
        by_r=-1.0,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
