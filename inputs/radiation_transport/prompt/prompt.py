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
# - Graziani prompt-spectrum test.  A hot source region (temp_s) drives an initially cold
#   background (temp_c) through a frequency-dependent opacity spectrum, and the emergent
#   radiation spectrum is the quantity of interest.  Reference: Computational Methods in
#   Transport: Verification and Validation, Frank Graziani (ed.), LNCSE 62.
#
# - We reuse the "shock" pgen: it initializes a left/right static state and its custom
#   inner-/outer-x1 "ic" boundaries inject an isotropic Planckian intensity into the ghost
#   cells.  The source region is the L state injected at inner-x1; the background is the
#   R state that fills the interior and the outer-x1 wall.  Placing the discontinuity at
#   the left edge (xd = x1min) starts the whole interior cold.
#
# - Opacities come from a singularity-opac multigroup table (prompt_opacity.sp5, generated
#   by make_prompt_opacity_table.py).  Both the absorption and (all-zero) scattering
#   opacities are tabular, exercising the full tabular multigroup pathway.
#
# - fixed_fluid=True freezes the hydro; coupling=True enables the radiation source term.
#
# ----------------------------------------------------------------------------------------
# USAGE
#
# This file is dual-purpose: it generates the riot input deck AND plots/analyzes the
# resulting simulation spectrum.  The mode is selected by command-line arguments.
#
#   Input-deck mode (default; no arguments):
#       python3 prompt.py
#     Writes prompt.rin next to this script.  This is also the entry point used by the
#     test harness via riot.generate("radiation_transport/prompt/prompt.py").
#
#   Analysis/plot mode (--plot):
#       python3 prompt.py --plot                  # analyze the default dump
#       python3 prompt.py --plot my_dump.phdf     # analyze a specific dump
#       python3 prompt.py --plot --out spec.png   # choose the output image path
#     Reads c.c.rad.moments from the dump, weights by dnu, overlays the Graziani exact
#     solution (read from prompt_exact.txt), and writes prompt.png.  Run this from the
#     directory containing the .phdf dump.
#
#   All-in-one wrapper (--run): generate the deck, run riot, then plot -- equivalent to
#       python3 prompt.py                                   # generate prompt.rin
#       mpiexec -n 1 ./riot -i <prompt.rin>                 # run, dumps into CWD
#       python3 prompt.py --plot prompt.out1.final.phdf
#     driven by a single command:
#       python3 prompt.py --run ./riot                      # exe path, defaults np=1
#       python3 prompt.py --run ./riot --np 4               # 4 ranks
#
# Companion files (all co-located with this script):
#   make_prompt_opacity_table.py  generator for the opacity table
#   prompt_opacity.sp5            committed singularity-opac multigroup opacity table
#   prompt_exact.txt              Graziani exact spectrum (reference for --plot)

import os
import sys

# Absolute path to the opacity table (co-located with this input file).
opac_table = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_opacity.sp5"
)

# Path of the input deck written by generate_input() (co-located with this script).
rin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.rin")

# Default simulation dump to analyze, resolved relative to the current working directory
# (i.e. the directory the dump was written to, such as build/src).
default_datafile = "prompt.out1.final.phdf"

# Temperature conversion [K per keV] used to express the source/background temperatures
# in the convenient keV units of the Graziani reference.
keV_to_K = 1.160706432e7


def generate_input():
    """Emit the prompt.rin input deck (input-file mode).

    Imports riot lazily so the analysis/plotting mode does not require the build's
    (singularity-eos-backed) riot python module."""
    import riot

    riot.input(
        "riot",
        problem="shock",
    )

    riot.input(
        "parthenon/job",
        problem_id="prompt",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.temperature",
            "c.c.bulk.pressure",
            "c.c.rad.aa",
            "c.c.rad.moments",
        ],
        file_type="hdf5",  # HDF5 data dump
        dt=1.0e-10,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=1.35e-12,  # time limit
        integrator="rk1",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        nx1=12,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=0.02526315789473684,  # maximum value of X1
        ix1_bc="ic",  # Inner-X1 boundary condition flag (hot source wall)
        ox1_bc="ic",  # Outer-X1 boundary condition flag (cold background wall)
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
        nx1=12,  # meshblock size in X1-direction
        nx2=1,  # meshblock size in X2-direction
        nx3=1,  # meshblock size in X3-direction
    )

    riot.input(
        "materials",
        sparse_init=False,  # enable sparse material initialization
        sparse_dealloc=False,  # enable sparse material deallocation
        # frequency group structure (ngroups + bounds) discovered from the opacity table
    )

    riot.input(
        "material0",
        label="mat0",  # material label
        eos_type="IdealGas",  # equation of state type
        Gamma=2.0,  # adiabatic index
        Cv=8.61733e42,  # specific heat at constant volume
        opac_a="table",  # tabular absorption opacity
        opac_filename=opac_table,  # singularity-opac multigroup table
        opac_s="table",  # tabular scattering opacity (all zeros)
        opac_material="mat0",  # shared HDF5 material group
        max_bnd_level=-1,  # maximum boundary AMR level
        max_mat_level=0,  # maximum material AMR level
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
        do_explicit=True,  # enable explicit transport
        do_jacobi=False,  # enable implicit Jacobi solver
        nlevel=4,  # level of geodesic mesh
        beta=5.0,  # parameter controlling tauc in Rusanov rad-flux
        coupling=True,  # flag to enable radiation source term
        affect_fluid=False,  # feedback on the fluid
        troot_tol=1.0e-6,  # temperature root find tolerance
        cfl=0.8,  # The Courant, Friedrichs, & Lewy (CFL) Number
    )

    riot.input(
        "radiation_transport/jacobi",
        err_thr=1.0e-6,  # implicit residual threshold
        dt_ratio_hyperbolic=2.0,  # Multiple of light-crossing time
    )

    riot.input(
        "problem",
        xd=0.0,  # discontinuity at left edge -> whole interior starts cold (background)
        # L State (hot source, injected at inner-x1)
        rho_l=0.0916,  # source density [g/cm^3]
        vx_l=0.0,  # source vel-x (static)
        temp_l=0.3 * keV_to_K,  # source temperature (0.3 keV)
        # R State (cold background, fills interior and outer-x1 wall)
        rho_r=0.0916,  # background density [g/cm^3]
        vx_r=0.0,  # background vel-x (static)
        temp_r=0.03 * keV_to_K,  # background temperature (0.03 keV)
    )

    # Writes prompt.rin next to this script (and prints its path to stdout).
    riot.input.generate_input()
    return rin_path


def analyze(datafile=default_datafile, outfile="prompt.png"):
    """Plot the emergent radiation spectrum against the Graziani reference (plot mode)."""
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

    # Frequency group structure: read the [Hz] group bounds straight from the opacity
    # table (the single source of truth) and convert to photon energy [keV] via nu*h.  The
    # first (0) and last (+inf) edges bound the zero-opacity tail groups and are dropped,
    # leaving the 51 finite edges of the 50 interior groups plotted below.
    h_kev_s = 4.135667696e-18  # Planck constant [keV s]
    with h5py.File(opac_table, "r") as f:
        nu_hz = f["group bounds/data"][()]
    nuf = h_kev_s * nu_hz[1:-1]
    dnu = nuf[1:] - nuf[:-1]
    nuv = np.sqrt((nuf[1:] * nuf[1:] + nuf[:-1] * nuf[:-1]) / 2.0)

    # Get data
    data = phdf(datafile)
    urad = data.Get("c.c.rad.moments")[0, :, 9][1:-1]
    # Weight by dnu as in Graziani...
    urad = urad / dnu

    # Graziani exact solution, one value per group-center frequency, read from the
    # committed reference file (co-located with this script).  See prompt_exact.txt.
    exact_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompt_exact.txt"
    )
    exact = np.loadtxt(exact_file)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(nuv, exact, lw=4, alpha=0.75, color="black", label="Exact", ls="--")
    ax1.scatter(
        nuv, urad, color=colors[1], s=80, label="$S_N$", zorder=10, edgecolor="black"
    )
    ax1.set_xlim(nuv[0], nuv[-5])
    ax1.set_ylim(max(exact) * 1e-6, max(exact) * 2)
    ax1.set_xlabel("$\\nu$ (keV)")
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


def run(exe, np=1, outfile="prompt.png"):
    """All-in-one wrapper: generate the deck, run riot under MPI, then plot the result.

    All outputs (the .phdf dumps and the plot) are written directly into the current
    working directory.

    exe     : path to the riot executable (invoke this from its directory, e.g. build/src).
    np      : number of MPI ranks.
    outfile : output image for the plot step.
    """
    import subprocess

    rin = generate_input()  # write prompt.rin, get its path
    cmd = ["mpiexec", "-n", str(np), exe, "-i", rin]
    print("running: " + " ".join(cmd))
    subprocess.check_call(cmd)
    analyze(default_datafile, outfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graziani prompt-spectrum problem: generate the riot input deck "
        "(default), or analyze/plot a simulation dump with --plot."
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=default_datafile,
        default=None,
        metavar="DUMP",
        help="Plot mode: analyze the given .phdf dump (default: %(default)s -> "
        "'{}') and write prompt.png, instead of generating the input deck.".format(
            default_datafile
        ),
    )
    parser.add_argument(
        "--out", default="prompt.png", help="Output image path for --plot/--run mode."
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
