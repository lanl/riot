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

from pathlib import Path
import numpy as np
import riot

ifile_dir = Path(__file__).resolve().parent

# This problem compresses a bubble of gas by heating the atmosphere
# around it slowly enough to be considered adiabatic. For
# adiabatic/isentropic flow, the ideal gas
#
# P = (Gamma - 1) rho C_v T
#
# behaves as a polytropic EOS
#
# P = K rho^Gamma
#
# This setup here starts with a bubble of some radius r_start in a
# periodic box with sides of length L and computes the amount of
# energy to add to the ambient gas to shrink the bubble to r_end. Then
# we write the added energy as a time history for the adiabatic
# sources physics package.


def cyl_vol(r):
    return np.pi * (r**2)


def make_input():
    L = 2

    Gamma1 = 4.0 / 3.0
    Gamma2 = 4.0 / 3.0
    Cv1 = 100
    Cv2 = 1

    P_start = 1

    r_start = 0.5
    r_end = 0.25

    vol2_start = cyl_vol(r_start)
    vol1_start = L**2 - vol2_start

    vol2_end = cyl_vol(r_end)
    vol1_end = L**2 - vol2_end

    rho1_start = 1
    rho2_start = 1 * rho1_start

    K_2 = P_start / rho2_start**Gamma2

    M1 = rho1_start * vol1_start
    M2 = rho2_start * vol2_start

    rho1_end = M1 / vol1_end
    rho2_end = M2 / vol2_end

    e1_start = P_start / ((Gamma1 - 1) * rho1_start)
    e2_start = P_start / ((Gamma2 - 1) * rho2_start)

    P_end = K_2 * rho2_end**Gamma2

    e1_end = P_end / ((Gamma1 - 1) * rho1_end)
    de1 = (e1_end - e1_start) * M1

    cs2_1 = Gamma1 * (Gamma1 - 1) * e1_start
    ncrossings = 5

    tf = ncrossings * L / np.sqrt(cs2_1)

    nout = 20
    dtout = tf / nout

    sources_filename = ifile_dir / "bubble_source.dat"
    times = np.array([0, 0.1 * tf, 0.4 * tf, 0.5 * tf, 0.75 * tf, 0.9 * tf])
    x = times / (0.9 * tf)
    smoothstep = 3 * (x**2) - 2 * (x**3)
    e_added = de1 * smoothstep
    sourcetable = np.vstack((times, e_added)).transpose()
    np.savetxt(sources_filename, sourcetable)

    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="compression")

    riot.input(
        "parthenon/output0",
        file_type="hst",
        dn=5,
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.pressure",
            "c.c.mat.rho",
            "c.c.mat.volume_fraction",
            "c.m.rho",
            "c.m.sie",
        ],
        file_type="hdf5",
        dt=dtout,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=tf,
        integrator="rk2",
        ncycle_out=1,
    )

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=1,
        derefine_count=10,
        nghost=2,
        nx1=270,
        x1min=-1,
        x1max=1,
        ix1_bc="periodic",
        ox1_bc="periodic",
        nx2=270,
        x2min=-1,
        x2max=1,
        ix2_bc="periodic",
        ox2_bc="periodic",
        nx3=1,
        x3min=-0.5,
        x3max=0.5,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )

    riot.input(
        "parthenon/meshblock",
        nx1=90,
        nx2=90,
        nx3=1,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=Gamma1,
        Cv=Cv1,
        max_bnd_level=-1,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=Gamma2,
        Cv=Cv2,
        max_bnd_level=-1,
        max_mat_level=-1,
    )

    riot.input(
        "physics",
        hydro=True,
        prescribed_sources=True,
        sparse_physics=False,
    )

    riot.input(
        "diagnostics",
        packages=["masses", "energies"],
    )

    riot.input(
        "energy_source0",
        material=0,
        cumulative_energies=sources_filename,
    )

    riot.input("hydro", recon="plm", cfl=0.8, amr_interface=True)

    riot.input(
        "region0",
        name="ambient",
        mask_type="background",
        matid=0,
        c_m_rho=rho1_start,
        c_m_pressure=P_start,
    )

    riot.input(
        "region1",
        name="bubble",
        mask_type="inside_cylinder",
        radius=r_start,
        matid=1,
        c_m_rho=rho2_start,
        c_m_pressure=P_start,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
