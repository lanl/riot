#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

import numpy as np
import riot


def make_input():

    riot.input("riot", problem="region_pgen")

    riot.input(
        "parthenon/job",
        problem_id="bend",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.internal_energy",
            "c.c.bulk.pressure",
            "c.c.bulk.temperature",
            "c.c.mat.rho",
            "c.m.strength_j2",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=1.0e-6,  # time increment between outputs
        sparse_seed_nans=True,  # Seed sparse data with NaNs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=4.2e-5,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=3,
        nghost=2,
        nx1=80,  # Number of zones in X1-direction
        x1min=-4.0,  # minimum value of X1
        x1max=4.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=80,  # Number of zones in X2-direction
        x2min=-4.0,  # minimum value of X2
        x2max=4.0,  # maximum value of X2
        ix2_bc="outflow",  # Inner-X2 boundary condition flag
        ox2_bc="outflow",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-3.5,  # minimum value of X3
        x3max=3.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=8,
        nx2=8,
        nx3=1,
    )

    riot.input("materials", sparse_init=True, sparse_dealloc=True)

    riot.input(
        "material0",
        name="air",
        eos_type="IdealGas",
        Gamma=1.1,
        Cv=3.0e7,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        name="beryllium",
        strong=True,
        strength_model="beryllium_strength",
        eos="beryllium_eos",
        max_bnd_level=-1,
        max_mat_level=-1,
    )

    riot.input(
        "beryllium_strength",
        modelname="epp",
        G0=1.519e12,
        Y0=1.0e100,  # 3.30e9
        rho_fail=0.5,
    )

    riot.input(
        "beryllium_eos",
        eos_type="Gruneisen",
        C0=1.287e6,
        s1=1.124,
        s2=0.0,
        s3=0.0,
        G0=2.0,
        b=0.0,
        rho0=1.845,
        T0=0.0,
        P0=0.0,
        Cv=1.0e3,
    )

    riot.input(
        "region0",
        name="background",
        mask_type="background",
        matid=0,
        c_m_rho=1.0e-3,
        c_m_pressure=1.0e6,
    )

    riot.input(
        "region1",
        name="beam",
        mask_type="inside_rectangle",
        matid=1,
        x0=-3.2,
        x1=3.2,
        y0=-0.48,
        y1=0.48,
        c_m_rho=1.845,
        c_m_pressure=1.0e6,
    )

    riot.input(
        "beam/params",
        aa=-4.1301249831098646e-1,
        om=1.9917289273647353e5,
        aj=0.5,
        bj=-4.91251107288119e-1,
        ll=6.4,
        kn=7.390688663847975e-1,
    )

    riot.input("physics", hydro=True, strength=True)

    riot.input("hydro", recon="plm", cfl=0.9, amr_interface=True)


class beam:
    def __init__(self):
        pass

    def c_c_bulk_velocity(self, pos, v):
        v[:, self.x] = 0.0
        v[:, self.y] = (
            self.aa
            * self.om
            * (
                self.aj
                * (
                    np.cosh(self.kn * (pos[:, self.x] + self.ll / 2.0))
                    + np.cos(self.kn * (pos[:, self.x] + self.ll / 2.0))
                )
                + self.bj
                * (
                    np.sinh(self.kn * (pos[:, self.x] + self.ll / 2.0))
                    + np.sin(self.kn * (pos[:, self.x] + self.ll / 2.0))
                )
            )
        )
        v[:, self.z] = 0.0


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
