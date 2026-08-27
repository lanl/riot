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
#
# NOTE(@pdmullen):
#
# - Homogeneous sphere test.  A hot spherical source region emits into vacuum.
#  **This problem requires 1D UniformSpherical coordinates.**
#
# - fixed_fluid=True freezes the hydro; coupling=True enables the radiation source term.
#
# ----------------------------------------------------------------------------------------
# USAGE
#
# This file is dual-purpose: it generates the riot input deck AND plots/analyzes the
# resulting simulation solution.  The mode is selected by command-line arguments.
#
#   Input-deck mode (default; no arguments):
#       python3 sphere.py
#     Writes sphere.rin next to this script.  This is also the entry point used by the
#     test harness via riot.generate("radiation_transport/sphere/sphere.py").
#
#   Analysis/plot mode (--plot):
#       python3 sphere.py --plot                  # analyze the default dump
#       python3 sphere.py --plot my_dump.phdf     # analyze a specific dump
#       python3 sphere.py --plot --out spec.png   # choose the output image path
#     Reads c.c.rad.moments from the dump, plot as a function of radius and overplots the
#     analytic solution.  Run this from the directory containing the .phdf dump.
#
#   All-in-one wrapper (--run): generate the deck, run riot, then plot -- equivalent to
#       python3 sphere.py                                   # generate sphere.rin
#       mpiexec -n 1 ./riot -i <sphere.rin>                 # run, dumps into CWD
#       python3 sphere.py --plot sphere.out1.final.phdf
#     driven by a single command:
#       python3 sphere.py --run ./riot                      # exe path, defaults np=1
#       python3 sphere.py --run ./riot --np 4               # 4 ranks

import os
import sys

# Path of the input deck written by generate_input() (co-located with this script).
rin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere.rin")

# Default simulation dump to analyze, resolved relative to the current working directory
# (i.e. the directory the dump was written to, such as build/src).
default_datafile = "sphere.out1.final.phdf"

# Homogeneous sphere density and temperature (dimensionless)
rad_sphere = 1.0
dens_sphere = 1.0
temp_sphere = 1.0
opac_sphere = 10.0


def generate_input():
    """Emit the sphere.rin input deck (input-file mode).

    Imports riot lazily so the analysis/plotting mode does not require the build's
    (singularity-eos-backed) riot python module."""
    import riot

    riot.input(
        "riot",
        problem="region_pgen",
    )

    riot.input(
        "parthenon/job",
        problem_id="sphere",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.temperature",
            "c.c.rad.aa",
            "c.c.rad.moments",
        ],
        file_type="hdf5",  # HDF5 data dump
        dt=10.0,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=100.0,  # time limit
        integrator="rk1",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",  # refinement type
        numlevel=2,  # number of AMR levels
        nx1=1200,  # Number of zones in X2-direction
        x1min=0.05,  # minimum value of X2
        x1max=10.0,  # maximum value of X2
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=1,  # Number of zones in X1-direction
        x2min=0.0,  # minimum value of X1
        x2max=3.141592653589793,  # maximum value of X1
        ix2_bc="periodic",  # Inner-X1 boundary condition flag
        ox2_bc="periodic",  # Outer-X1 boundary condition flag
        nx3=1,  # Number of zones in X3-direction
        x3min=0.0,  # minimum value of X3
        x3max=6.283185307179586,  # maximum value of X3
        ix3_bc="periodic",  # Inner-X3 boundary condition flag
        ox3_bc="periodic",  # Outer-X3 boundary condition flag
    )

    riot.input(
        "parthenon/meshblock",
        nx1=120,  # meshblock size in X1-direction
        nx2=1,  # meshblock size in X2-direction
        nx3=1,  # meshblock size in X3-direction
    )

    riot.input(
        "materials",
        sparse_init=False,  # enable sparse material initialization
        sparse_dealloc=False,  # enable sparse material deallocation
    )

    riot.input(
        "material0",
        label="mat0",  # material label
        eos_type="IdealGas",  # equation of state type
        Gamma=2.0,  # adiabatic index
        Cv=1.0,  # specific heat at constant volume
        opac_a="constant",  # tabular absorption opacity
        kappa_a=0.0,
        max_bnd_level=-1,  # maximum boundary AMR level
        max_mat_level=0,  # maximum material AMR level
    )

    riot.input(
        "material1",
        label="mat1",  # material label
        eos_type="IdealGas",  # equation of state type
        Gamma=2.0,  # adiabatic index
        Cv=1.0,  # specific heat at constant volume
        opac_a="constant",  # tabular absorption opacity
        kappa_a=opac_sphere,
        max_bnd_level=-1,  # maximum boundary AMR level
        max_mat_level=0,  # maximum material AMR level
    )

    riot.input(
        "region0",
        name="thin",  # region name
        mask_type="background",  # region mask type
        matid=0,  # material ID
        c_m_rho=1.0e-7 * dens_sphere,  # material density
        c_m_temperature=1.0e-3 * temp_sphere,  # material temperature
    )

    riot.input(
        "region1",
        name="region1",  # region name
        mask_type="inside_sphere",  # region mask type
        radius=rad_sphere,
        matid=1,  # material ID
        c_m_rho=dens_sphere,  # material density
        c_m_temperature=temp_sphere,  # material temperature
    )

    riot.input(
        "physics",
        hydro=True,  # enable hydrodynamics
        radiation_transport=True,  # enable radiation transport
        fixed_fluid=True,  # fix fluid (no hydro update)
        sparse_physics=False,  # enable sparse physics allocation
    )

    riot.input(
        "hydro",
        cfl=0.8,  # CFL number for hydro
        amr_interface=False,  # use AMR interface reconstruction
    )

    riot.input(
        "radiation_transport",
        do_explicit=False,  # enable explicit transport
        do_jacobi=True,  # enable implicit Jacobi solver
        nlevel=8,  # level of geodesic mesh
        beta=0.0,  # parameter controlling tauc in Rusanov rad-flux
        coupling=True,  # flag to enable radiation source term
        affect_fluid=False,  # feedback on the fluid
        cfl=0.8,  # The Courant, Friedrichs, & Lewy (CFL) Number
        units_override=True,  # Units system override
        fixed_temp_rhs=True,  # temp advanced == temp old
    )

    riot.input(
        "radiation_transport/jacobi",
        err_thr=1.0e-8,  # error threshhold
        niter_limit=10000,  # iteration limit
        dt_ratio_hyperbolic=1.0e6,  # Multiple of light-crossing time
        verbose=2,  # Verbosity
    )

    riot.input(
        "radiation_transport/explicit",
        dt_ratio_hyperbolic=1.0e6,  # Multiple of light-crossing time
        verbose=2,  # Verbosity
    )

    riot.input(
        "radiation_transport/drive",
        trad_bc=1.0,  # uniform boundary source
        ix1_bc="drive",
    )

    riot.input(
        "radiation_transport/init",
        initialization="zero",
    )

    # Writes sphere.rin next to this script (and prints its path to stdout).
    riot.input.generate_input()
    return rin_path


def analyze(datafile=default_datafile, outfile="sphere.png"):
    """Plot the emergent homogeneous sphere solution again the exact one (plot mode)."""
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../external/parthenon/scripts/python/packages/"
            "parthenon_tools/parthenon_tools",
        ),
    )
    import h5py
    from phdf import phdf

    # Plotting style
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=18)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Get data
    data = phdf(datafile)
    rc = np.array(
        [0.5 * (data.xng[i][1:] + data.xng[i][:-1]) for i in range(0, len(data.xng))]
    ).flatten()
    urad = data.Get("c.c.rad.moments").flatten()

    # Analytic
    aa = 1.0
    r0 = rad_sphere
    rk = opac_sphere
    tt = temp_sphere

    def g(r, n):
        return np.sqrt(np.maximum(0, 1 - (r / r0) ** 2 * (1 - n * n)))

    def s(r, n):
        r, n = np.asarray(r)[:, None], np.asarray(n)[None, :]
        G = g(r, n)
        inside = r * n + r0 * G
        outside = np.where(
            n >= np.sqrt(np.maximum(0, 1 - (r0 / r) ** 2)), 2 * r0 * G, 0
        )
        return np.where(r < r0, inside, outside)

    def i(r, n):
        return aa * tt**4 / (4 * np.pi) * (1 - np.exp(-rk * s(r, n)))

    def e(r):
        n = np.linspace(-1, 1, 10000)
        return 2 * np.pi * np.trapezoid(i(r, n), n)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(rc, e(rc), lw=4, alpha=0.75, color="black", label="Exact")
    ax1.scatter(rc, urad, color=colors[1], s=20, label="$S_N$", zorder=10)
    ax1.set_xlabel("$r$")
    ax1.set_ylabel("$u$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.5)
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")
    ax1.tick_params(which="both", direction="in")
    ax1.legend(loc="lower left")
    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    print("wrote {} from {}".format(outfile, datafile))


def run(exe, np=1, outfile="sphere.png"):
    """All-in-one wrapper: generate the deck, run riot under MPI, then plot the result.

    All outputs (the .phdf dumps and the plot) are written directly into the current
    working directory.

    exe     : path to the riot executable (invoke this from its directory, e.g. build/src).
    np      : number of MPI ranks.
    outfile : output image for the plot step.
    """
    import subprocess

    rin = generate_input()  # write sphere.rin, get its path
    cmd = ["mpiexec", "-n", str(np), exe, "-i", rin]
    print("running: " + " ".join(cmd))
    subprocess.check_call(cmd)
    analyze(default_datafile, outfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Homogeneous sphere problem: generate the riot input deck "
        "(default), or analyze/plot a simulation dump with --plot."
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=default_datafile,
        default=None,
        metavar="DUMP",
        help="Plot mode: analyze the given .phdf dump (default: %(default)s -> "
        "'{}') and write sphere.png, instead of generating the input deck.".format(
            default_datafile
        ),
    )
    parser.add_argument(
        "--out", default="sphere.png", help="Output image path for --plot/--run mode."
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="RIOT_EXE",
        help="All-in-one mode: generate the deck, run the given riot executable under "
        "MPI (see --np), then plot the result.  All outputs land in the current "
        "working directory.",
    )
    parser.add_argument(
        "--np", type=int, default=1, help="Number of MPI ranks for --run mode."
    )
    args = parser.parse_args()

    if args.run is not None:
        run(args.run, args.np, args.out)
    elif args.plot is not None:
        analyze(args.plot, args.out)
    else:
        generate_input()
