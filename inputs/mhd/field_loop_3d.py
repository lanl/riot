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
"""Gardiner & Stone (2005) section 5.4: a field loop advected in 3D with a velocity
component ALONG the loop axis. 64 x 32 x 32 on 8 mesh blocks, v = (2, 1, 1) on a
2 x 1 x 1 periodic box, so tlim = 1 is exactly one crossing in every direction.

THE POINT OF THIS TEST. With the loop axis along x3 the field is B = (B1, B2, 0) and the
axial component evolves as

    d_t B3 = -(d_1 E2 - d_2 E1) = v3 (d_1 B1 + d_2 B2) = v3 * div B = 0,

so B3 must stay EXACTLY zero -- but only because E1 = v3 B2 and E2 = -v3 B1 cancel against
each other. Any inconsistency between how E1 and E2 are assembled shows up immediately as a
spurious axial field with nothing to hide behind. That makes max|B3| a direct, quantitative
check on MHD::UpwindEMF<X1DIR> and <X2DIR>, which are UNREACHABLE in 2D (there the collapsed
branch runs instead).

Sweep `mhd_field_loop/loop_axis` over 1, 2, 3 to put the check on each pair of edge
directions in turn. This matters for more than EMF coverage: with loop_axis=3 the field has
B3 = 0, so the F3 face exchange carries only zeros and a defect in it would be invisible.

`mhd/mu0 = 1` for donor parity; see plan_histories/artemis_mhd_port/adr/001-units.md.

Run `python3 field_loop_3d.py` to emit field_loop_3d.rin.
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

    riot.input("riot", problem="mhd_field_loop")

    riot.input("parthenon/job", problem_id="field_loop_3d")

    riot.input(
        "parthenon/output1",
        variables=OUTPUT_VARIABLES,
        file_type="hdf5",
        dt=1.0,
    )

    riot.input("parthenon/time", nlim=-1, tlim=1.0, integrator="rk2", ncycle_out=200)

    riot.input(
        "parthenon/mesh",
        refinement="none",
        nghost=2,
        nx1=64,
        x1min=-1.0,
        x1max=1.0,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=32,
        x2min=-0.5,
        x2max=0.5,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=32,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    # 8 blocks: divides evenly across 1/2/4/8 ranks, and unevenly across 3 or 5, which is a
    # deliberately different ownership pattern (see TEST_LEDGER P01.1).
    riot.input("parthenon/meshblock", nx1=32, nx2=16, nx3=16)

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
        "mhd_field_loop",
        rho=1.0,
        P=1.0,
        v1=2.0,
        v2=1.0,
        v3=1.0,
        amp=1.0e-3,
        r_loop=0.3,
        loop_axis=3,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
