#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

from pathlib import Path
import riot

ifile_dir = Path(__file__).resolve().parent


def make_input():

    isotope_filename = ifile_dir / "dummy_tn_data.hdf5"

    riot.input(
        "riot",
        problem="tn_test",
    )

    riot.input(
        "parthenon/job",
        problem_id="tn_test_multireaction",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.mat.rho",
            "c.c.mat.iso",
            "c.c.bulk.velocity",
            "c.c.mat.internal_energy",
            "c.m.tn_specific_reactions",
        ],
        file_type="hdf5",  # Tabular data dump
        dt=1.0e-8,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1.0e-7,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        refinement="none",
        numlevel=3,
        # derefine_count = 5,
        nx1=4,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=4.0,  # maximum value of X1
        ix1_bc="reflecting",  # Inner-X1 boundary condition flag
        ox1_bc="reflecting",  # Outer-X1 boundary condition flag
        nx2=1,  # Number of zones in X2-direction
        x2min=-0.5,  # minimum value of X2
        x2max=0.5,  # maximum value of X2
        ix2_bc="periodic",  # Inner-X2 boundary condition flag
        ox2_bc="periodic",  # Outer-X2 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=-0.5,  # minimum value of X3
        x3max=0.5,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=4,
        nx2=1,
        nx3=1,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.4,
        Cv=3.0e12,
        isotope0=1002,
        isotope1=1003,
        isotope2=2004,
        isotope3="0001",
        # isotope4 = 1001,
        # isotope5 = 2003,
    )

    riot.input(
        "physics",
        hydro=True,
        scalars=False,
        tn=True,
        sparse_physics=False,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.9,
        amr_interface=True,
        # riemann = hllc,
    )

    riot.input(
        "isotope_data",
        filename=isotope_filename,
    )

    riot.input(
        "tnburn",
        reaction0="d+t->n+a",
        reaction1="t+t->2n+a",
        reaction2="d+d->n+he3",
        reaction3="d+d->p+t",
        reaction4="p+t->n+he3",
        reaction5="n+he3->p+t",
        reaction6="d+he3->p+a",
        reaction7="he3+he3->2p+a",
        reaction8="t+he3->n+p+a",
        reaction9="t+he3->d+a",
        # reaction0 = d+he3->p+a,
        # reaction1 = t+he3->n+p+a,
        deposit_locally_1=True,  # neutrons dissapear from problem (default)
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
