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
"""Orszag & Tang (1979) vortex on the unit square, 64 x 64 on 4 mesh blocks, to t = 0.5.

THE POINT OF THIS TEST: smooth initial data steepens into a network of interacting MHD
shocks, so it is the only case in this port where constrained transport has to hold
div B = 0 WHILE limiters are firing and the EMF stencil straddles discontinuities. The field
loop is smooth for all time and Brio-Wu is one-dimensional; neither can catch a CT defect
that only appears at a shock.

Matches artemis/inputs/orszag_tang/orszag_tang.in (gamma = 5/3, mu0 = 1) except for the
Riemann solver: the donor uses HLLD, which this port does not have yet, so this deck uses
mhd_hlle. That difference is why the acceptance criteria are conserved global quantities and
the local energy split rather than a pointwise reference -- the shock network is chaotic, so
two correct codes disagree pointwise at t = 0.5 and a gold file would encode this build
rather than the physics. See claude_sessions/mhd_runs/analyze_orszag_tang.py and
plan_histories/artemis_mhd_port/TEST_LEDGER.md (G4.4).

The parameter values are the standard closed forms rho0 = 25/(36 pi), P0 = 5/(12 pi),
b0 = 1/sqrt(4 pi), written out as decimals here to match the donor deck byte for byte.

`mhd/mu0 = 1` for donor parity; see plan_histories/artemis_mhd_port/adr/001-units.md.

Run `python3 orszag_tang.py` to emit orszag_tang.rin.
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
]


def make_input():

    riot.input("riot", problem="mhd_orszag_tang")

    riot.input("parthenon/job", problem_id="orszag_tang")

    riot.input(
        "parthenon/output1",
        variables=OUTPUT_VARIABLES,
        file_type="hdf5",
        dt=0.1,
    )

    riot.input("parthenon/time", nlim=-1, tlim=0.5, integrator="rk2", ncycle_out=100)

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=64,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=64,
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

    riot.input("parthenon/meshblock", nx1=32, nx2=32, nx3=1)

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
        "mhd_orszag_tang",
        rho0=0.22104853207207686,
        P0=0.13262911924324611,
        v0=1.0,
        b0=0.28209479177387814,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
