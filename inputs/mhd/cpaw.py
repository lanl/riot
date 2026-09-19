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
"""Circularly polarized Alfven wave (Toth 2000; Gardiner & Stone 2005 section 5.2).

An EXACT NONLINEAR solution of ideal MHD, used here to measure ORDER OF ACCURACY. Every
other MHD test in this port checks either an identity that holds at roundoff (div B, the
axial field, conservation) or agreement with a reference at a single resolution. None of them
would notice a scheme that is stable, conservative, divergence free and merely FIRST order --
which is what a subtly wrong EMF average or reconstruction produces. Fitting the L1 slope
across resolutions is what closes that gap.

It is an analytic oracle rather than a donor one, deliberately: the donor has no CPAW. Its
`linwave` test uses 7 linear eigenmodes at amplitude 1e-6, which probes only the linearized
system, and a scheme can be second order there while losing it at finite amplitude. Here the
perpendicular field rotates at constant magnitude, so |B| is uniform, the magnetic pressure
gradient vanishes identically, and the wave translates at v_A with no steepening at
amplitude 0.1.

`tlim` is one full period: v_A = b_par/sqrt(rho) = 1 and the wavelength is the box length
along the wave direction, so T = 1. The exact solution at t = 1 is therefore the initial
condition again, and L1 can be measured directly against the t = 0 snapshot with no analytic
evaluator needed at t > 0.

`mhd_cpaw/wave_dir` (1/2/3) rotates the wave onto each axis, so every face direction carries
the nonuniform components in turn; `travel_sign` (+1/-1) reverses propagation, which is what
catches an induction sign error that a standing pattern would hide.

LIMITATION, stated rather than glossed: this is an AXIS-ALIGNED wave. A wave along a box
diagonal additionally couples all three EMF components and is a strictly stronger test; it is
not covered by this deck.

Run `python3 cpaw.py` to emit cpaw.rin. Resolution is swept from the command line, e.g.
    riot -i cpaw.rin parthenon/mesh/nx1=32 parthenon/meshblock/nx1=16
see claude_sessions/mhd_runs/analyze_cpaw.py for the driver.
"""

import riot

OUTPUT_VARIABLES = [
    "c.c.bulk.rho",
    "c.c.bulk.pressure",
    "c.c.bulk.velocity",
    "c.c.bulk.magnetic_field",
    "c.c.bulk.div_magnetic_field",
    "c.c.bulk.magnetic_energy",
    "c.c.bulk.total_material_energy",
]


def make_input():

    riot.input("riot", problem="mhd_cpaw")

    riot.input("parthenon/job", problem_id="cpaw")

    # Only t=0 and t=tlim are needed, and tlim is one period, so dt = tlim.
    riot.input(
        "parthenon/output1",
        variables=OUTPUT_VARIABLES,
        file_type="hdf5",
        dt=1.0,
    )

    riot.input("parthenon/time", nlim=-1, tlim=1.0, integrator="rk2", ncycle_out=200)

    # The wave direction spans [0, 1] so the wavelength is 1 and, with v_A = 1, one period is
    # tlim = 1. The transverse extents are deliberately short and coarse: the solution is
    # uniform across them, so resolving them buys nothing and only costs wall time -- but they
    # are kept NONDEGENERATE so the run is genuinely 3D and the transverse face exchanges and
    # EMF components are all live.
    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=32,
        x1min=0.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=4,
        x2min=0.0,
        x2max=0.125,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=4,
        x3min=0.0,
        x3max=0.125,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    # Split along the wave direction so block boundaries are crossed by the wave.
    riot.input("parthenon/meshblock", nx1=16, nx2=4, nx3=4)

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.6666666666666667,
        Cv=1.0,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input("physics", hydro=True, mhd=True)

    riot.input("mhd", mu0=1.0)

    riot.input("hydro", riemann="mhd_hlle", recon="plm", cfl=0.4, amr_interface=False)

    riot.input(
        "mhd_cpaw",
        rho=1.0,
        P=0.1,
        b_par=1.0,
        amp=0.1,
        wave_dir=1,
        travel_sign=1.0,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
