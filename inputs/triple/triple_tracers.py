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

"""
Generate triple-point problem with both lagrangian and eulerian swarms.
"""

import riot


def make_input():

    # Problem specification
    riot.input("riot", problem="region_pgen")
    riot.input("parthenon/job", problem_id="triple_tracers")

    # Output configuration - separate outputs for each swarm to avoid warnings
    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
        ],
        swarms=["lagrangian"],
        swarm_variables=["swarm.id"],
        auto_swarm_sample_fields=True,
        write_swarm_xdmf=True,
        file_type="hdf5",
        dt=0.01,
    )

    riot.input(
        "parthenon/output2",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
        ],
        swarms=["eulerian"],
        swarm_variables=["swarm.id"],
        auto_swarm_sample_fields=True,
        write_swarm_xdmf=True,
        file_type="hdf5",
        dt=0.01,
    )

    # Time integration parameters
    riot.input("parthenon/time", nlim=-1, tlim=5.0, integrator="rk2", ncycle_out=10)

    # Mesh configuration - same as original triple problem
    riot.input(
        "parthenon/mesh",
        refinement="none",
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
        num_threads=1,
    )

    riot.input("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

    # Material definitions - three ideal gas regions
    riot.input("material0", name="air0", eos_type="IdealGas", Gamma=1.5, Cv=1.0e12)
    riot.input("material1", name="air1", eos_type="IdealGas", Gamma=1.5, Cv=1.0e12)
    riot.input("material2", name="air2", eos_type="IdealGas", Gamma=1.5, Cv=1.0e12)

    # Region 0: Background - high pressure, unit density
    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=1.0,
        c_m_pressure=1.0,
    )

    # Region 1: Low density, low pressure (lower left, x > 1.0, y > 1.5)
    riot.input(
        "region1",
        mask_type="inside_rectangle",
        matid=1,
        x0=1.0,
        y0=1.5,
        c_m_rho=0.125,
        c_m_pressure=0.1,
    )

    # Region 2: Unit density, low pressure (upper left, x > 1.0, y < 1.5)
    riot.input(
        "region2",
        mask_type="inside_rectangle",
        matid=2,
        x0=1.0,
        y1=1.5,
        c_m_rho=1.0,
        c_m_pressure=0.1,
    )

    # Physics modules
    riot.input("physics", hydro=True, tracers=True)

    # Hydro configuration
    riot.input("hydro", recon="plm", cfl=0.9, amr_interface=True)

    # Tracer configuration - two swarms
    import numpy as np

    n_particles = 100000
    rng = np.random.default_rng(seed=42)

    # Lagrangian particles - uniformly distributed, advect with flow
    x1_lag = rng.uniform(0.0, 7.0, n_particles).tolist()
    x2_lag = rng.uniform(0.0, 3.0, n_particles).tolist()
    x3_lag = [0.0] * n_particles

    # Eulerian probes - uniformly distributed, stay fixed
    x1_eul = rng.uniform(0.0, 7.0, n_particles).tolist()
    x2_eul = rng.uniform(0.0, 3.0, n_particles).tolist()
    x3_eul = [0.0] * n_particles

    # Lagrangian swarm - moves with flow, samples velocity
    riot.input(
        "tracers/lagrangian",
        x1=x1_lag,
        x2=x2_lag,
        x3=x3_lag,
        advect=True,
        sample_fields=["c.c.bulk.velocity"],
    )

    # Eulerian swarm - stays fixed, samples rho and pressure
    riot.input(
        "tracers/eulerian",
        x1=x1_eul,
        x2=x2_eul,
        x3=x3_eul,
        advect=False,
        sample_fields=["c.c.bulk.rho", "c.c.bulk.pressure"],
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
