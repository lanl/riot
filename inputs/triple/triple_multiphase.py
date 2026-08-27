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


import numpy as np
import riot


def make_input():
    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="triple")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
            "c.c.mat.volume_fraction",
        ],
        file_type="hdf5",
        dt=0.1,
    )

    riot.input("parthenon/time", nlim=-1, tlim=5.0, integrator="rk2", ncycle_out=10)

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=3,
        derefine_count=5,
        nx1=112,
        x1min=0.0,
        x1max=7.0,
        ix1_bc="reflecting",
        ox1_bc="reflecting",
        nx2=48,
        x2min=0.0,
        x2max=3.0,
        ix2_bc="reflecting",
        ox2_bc="reflecting",
        nx3=1,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    riot.input(
        "parthenon/refinement1",
        method="derivative_order_1",
        field="c.c.bulk.rho",
        refine_tol=0.1,
        derefine_tol=0.01,
    )

    riot.input("materials", sparse_dealloc=True, use_general_pte=True)

    riot.input("material0", eos_type="IdealGas", Gamma=1.5, Cv=1.0e-3)

    riot.input(
        "material1",
        nphase=2,
        eos0="ideal1",
        eos1="ideal2",
    )

    riot.input("ideal1", eos_type="IdealGas", Gamma=1.5, Cv=1.0e-3)

    riot.input("ideal2", eos_type="IdealGas", Gamma=1.4, Cv=1.0e-3)

    riot.input("physics", hydro=True, sparse_physics=True)

    riot.input("hydro", recon="plm", cfl=0.8, amr_interface=True)

    riot.input(
        "regions",
        nlev_max=5,
    )

    riot.input(
        "region0",
        name="left",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
    )

    riot.input(
        "region1",
        name="upper",
        mask_type="inside_rectangle",
        x0=1.0,
        y0=1.5,
        matid=1,
        c_m_phase_fraction=[1.0, 0.0],
        c_m_rho=0.125,
        c_m_pressure=0.1,
    )

    riot.input(
        "region2",
        name="lower",
        mask_type="inside_rectangle",
        x0=1.0,
        y1=1.5,
        matid=1,
        c_m_phase_fraction=[0.0, 1.0],
        c_m_rho=1.0,
        c_m_pressure=0.1,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
