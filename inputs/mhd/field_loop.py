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
"""Gardiner & Stone (2005) advected field loop, 2D, one full diagonal crossing.

tlim = 1 with v = (2, 1) on a 2 x 1 periodic box is exactly one crossing in each direction.
Runs on 4 mesh blocks by default, so div B is checked ACROSS block boundaries -- a
single-block run cannot detect a shared-face or edge-ownership error.

Exact solution: pure advection, since the field is dynamically negligible (beta ~ 1e6). The
loop must return to its initial position with its shape and magnetic energy intact and div B
at roundoff. The two failure modes this exposes are ones no 1D test can see: an EMF
averaging error diffuses the loop (energy decays) and an orientation or upwinding error
distorts it anisotropically even though the advection velocity is uniform.

With x3 collapsed this takes the COLLAPSED branch of MHD::AssembleEdgeEMF, not the upwind
one, so it is a distinct code path from field_loop_3d.py rather than a subset of it.

`mhd/mu0 = 1` for donor parity; see plan_histories/artemis_mhd_port/adr/001-units.md.

Run `python3 field_loop.py` to emit field_loop.rin.
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

    riot.input("riot", problem="mhd_field_loop")

    riot.input("parthenon/job", problem_id="field_loop")

    riot.input(
        "parthenon/output1",
        variables=OUTPUT_VARIABLES,
        file_type="hdf5",
        dt=1.0,
    )

    riot.input("parthenon/time", nlim=-1, tlim=1.0, integrator="rk2", ncycle_out=100)

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=128,
        x1min=-1.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=64,
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

    riot.input("parthenon/meshblock", nx1=64, nx2=32, nx3=1)

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.6666666666666667,
        Cv=1.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input("physics", hydro=True, mhd=True)

    riot.input("mhd", mu0=1.0, monitor_divb=True)

    riot.input("hydro", riemann="mhd_hlle", recon="plm", cfl=0.4, amr_interface=False)

    riot.input(
        "mhd_field_loop",
        rho=1.0,
        P=1.0,
        v1=2.0,
        v2=1.0,
        v3=0.0,
        amp=1.0e-3,
        r_loop=0.3,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
