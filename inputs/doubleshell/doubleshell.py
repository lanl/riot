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

"""Riot input generator for the double-shell ICF capsule (demonstration deck).

The double-shell capsule is a nested pair of shells.  From the inside out the
nominal 1-D material stack is::

    DT gas  ->  inner (hi-z) shell  ->  tamper  ->  low-density foam cushion
            ->  ablator  (outermost)

The *outer* (ablator) shell additionally carries a complex gap: a horizontal
inner slab, a vertical run, and an outward-angled section, with a thin gold
liner sputter-coated on the inner surface of the gap.  In 3-D (and in 2-D when
the fill tube is placed on the pole) a conical glass fill tube is glued into
bore holes drilled through the shells.

This deck is parameterized with ``argparse`` and can be driven with grey
radiation using either diffusion or discrete-ordinates (SN) transport, with a
constant-temperature radiation drive.

    IMPORTANT -- the spatial coordinate system is a COMPILE-TIME choice
    (``PARTHENON_COORDINATES``).  Pick a binary that matches ``--dim``:

        --dim 1  ->  PARTHENON_COORDINATES=UniformSpherical    (r)
        --dim 2  ->  PARTHENON_COORDINATES=UniformCylindrical   (R, Z)   [RZ]
        --dim 3  ->  PARTHENON_COORDINATES=UniformCartesian      (x, y, z)

    The custom Python region masks receive Cartesian sample positions in a
    ``pos`` array of shape ``(N, 3)``.  Under the region_pgen machinery those
    positions are ``(r, 0, 0)`` in 1-D spherical, ``(R, 0, Z)`` in 2-D
    cylindrical, and ``(x, y, z)`` in 3-D Cartesian (see
    ``src/riot_pgen/regions.cpp::to_cartesian``).

Usage examples::

    # generate a 2-D RZ diffusion deck
    python doubleshell.py --dim 2 --rad diffusion

    # generate a 3-D transport deck with the fill tube on the equator
    python doubleshell.py --dim 3 --rad transport --do_fill_tube --fill_tube_axis 0

"""

import argparse
from pathlib import Path

import numpy as np

# NOTE: ``riot`` (and its compiled singularity_eos dependency) is imported lazily inside
# make_input().  The region-mask classes and build_geometry() below are pure numpy, so
# they could be used by external tools for visualization.

ifile_dir = Path(__file__).resolve().parent

# Conversion factor from electron-volts to Kelvin (radiation temperatures).
EV_TO_K = 1.16045e4

# ----------------------------------------------------------------------------------------
# Material catalog.
#
# ``kappa0`` for the grey powerlaw absorption opacity follows the patriot double-shell
# decks: kappa = kappa0 * rho * T^-3.5, with kappa0 = C * Z^2 and C = 5e22 (motivated by
# TOPS opacities for Al/Be over T in [200 eV, 2 keV]).  Helium is treated as a
# transparent ideal-gas background (kappa0 = 0) when using --rad transport.
# ----------------------------------------------------------------------------------------
OPAC_C = 5.0e22


def _kappa0(z):
    return OPAC_C * z * z


MATERIALS = {
    #  name        : sesame_id, density (g/cc), Z (for grey opacity), eos_type
    "Helium": {"sesame_id": 5762, "density": 1.0e-5, "z": 2, "eos": "IdealGas"},
    "Aluminum": {
        "sesame_id": 3720,
        "density": 2.70,
        "z": 13,
        "eos": "SpinerEOSDependsRhoT",
    },
    "CH": {"sesame_id": 7592, "density": 1.02, "z": 5, "eos": "SpinerEOSDependsRhoT"},
    "CHFoam": {
        "sesame_id": 7592,
        "density": 0.035,
        "z": 5,
        "eos": "SpinerEOSDependsRhoT",
    },
    "Beryllium": {
        "sesame_id": 2024,
        "density": 1.62,
        "z": 4,
        "eos": "SpinerEOSDependsRhoT",
    },
    "Tungsten": {
        "sesame_id": 3541,
        "density": 19.2367,
        "z": 74,
        "eos": "SpinerEOSDependsRhoT",
    },
    "Molybdenum": {
        "sesame_id": 3539,
        "density": 10.2,
        "z": 42,
        "eos": "SpinerEOSDependsRhoT",
    },
    "DT": {"sesame_id": 1018, "density": 0.20, "z": 1, "eos": "SpinerEOSDependsRhoT"},
    "Gold": {
        "sesame_id": 2705,
        "density": 19.3,
        "z": 79,
        "eos": "SpinerEOSDependsRhoT",
    },
    "Glass": {
        "sesame_id": 7387,
        "density": 2.204,
        "z": 10,
        "eos": "SpinerEOSDependsRhoT",
    },
}

# Fixed material index layout used throughout the deck.  Kept explicit so the region
# blocks, the mask classes, and plot_mesh.py all agree.
MATID = {
    "background": 0,  # Helium
    "ablator": 1,  # args.ablator_material
    "foam": 2,  # CHFoam (also used for glue: full-density CH)
    "tamper": 3,  # args.tamper_material
    "inner_shell": 4,  # args.inner_shell_material
    "dt": 5,  # DT
    "gold": 6,  # Gold
    "glass": 7,  # Glass fill tube
}
# Reverse lookup (matid index -> logical name), used to key the refinement table below.
MATID_NAME = {v: k for k, v in MATID.items()}

# ----------------------------------------------------------------------------------------
# Per-material AMR refinement levels (max_bnd_level / max_mat_level).
#
# These cap how deeply the mesh refines around each material's interfaces and are the
# primary knob for trading resolution against memory when scaling the problem up.  They are
# meant to stay FIXED across a suite of runs; only a dedicated resolution study would sweep
# them.  Edit them here rather than exposing them on the command line.
#
# A value of -1 means "the finest level available for this run" -- it is mapped to
# (parthenon/mesh numlevel - 1) in src/materials/materials.cpp, so it tracks --nlevels
# automatically no matter how far the resolution is turned up.  The background (helium) is
# never refined.
#
# In 2-D (and 1-D) we can afford to drive every material interface to the finest level.
# In 3-D memory forces a graduated scheme, but the gas cavity (DT) and the thin gold liner
# are ALWAYS pinned to the finest level (-1) so they keep tracking the resolution even when
# --nlevels is raised; the tamper / inner shell / glass sit at a fixed level below that, and
# the ablator and foam are left coarser still (~4 um at the nominal grid).  The fixed levels
# are deliberate holds -- bump them by hand if a run needs them finer.
REFINE_LEVELS = {
    #  logical name : (2-D / 1-D level, 3-D level);  99 == finest available
    "background": (0, 0),
    "ablator": (99, 3),
    "foam": (99, 4),
    "tamper": (99, 5),
    "inner_shell": (99, 5),
    "dt": (99, 99),
    "gold": (99, 99),
    "glass": (99, 6),
}


def refine_level(matid, dim):
    """Return the AMR refinement level for a material index at the requested dimension."""
    lev2d, lev3d = REFINE_LEVELS[MATID_NAME[matid]]
    return lev3d if dim == 3 else lev2d


# Regions initialize in PRESSURE equilibrium (rho, pressure) rather than temperature
# equilibrium: every condensed material starts at its solid/fill density and a common
# pressure INIT_P, so the EOS sets each material's initial temperature.  The helium
# background/void is the exception: it is initialized from
# (rho, temperature) at BACKGROUND_T so the transparent gas has a well-defined warm state.
INIT_P = 5.0e8  # initial pressure for condensed materials (dyne/cm^2)
BACKGROUND_T = 1000.0  # initial helium background/void temperature (K)
COLD_T = 298.0  # boundary/guess reference temperature (K); the drive heats inward

# Thermonuclear burn: DT fuel isotopes (ZAID) and atomic masses (amu).  The region
# isotope<N>_mfrac inputs are MASS fractions; to start 50/50 by NUMBER density we weight
# the mass fractions by atomic mass (mfrac_i = m_i / sum_j m_j for equal number).
DT_ISOTOPES = {
    "D": {"zaid": 1002, "mass": 2.0141},
    "T": {"zaid": 1003, "mass": 3.0155},
    "HE4": {"zaid": 2004, "mass": 4.0026},
}
_dt_mtot = DT_ISOTOPES["D"]["mass"] + DT_ISOTOPES["T"]["mass"]
DT_MFRAC = {
    "D": DT_ISOTOPES["D"]["mass"] / _dt_mtot,  # ~0.4004
    "T": DT_ISOTOPES["T"]["mass"] / _dt_mtot,
}  # ~0.5996


# ----------------------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Double-shell ICF capsule input deck.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    fid = parser.add_argument_group("riot fidelities")
    fid.add_argument(
        "--dim",
        required=True,
        type=int,
        choices=[1, 2, 3],
        help="Dimensionality (must match the binary's PARTHENON_COORDINATES)",
    )
    fid.add_argument(
        "--nlevels",
        type=int,
        default=3,
        help="Number of AMR refinement levels."
        "   nlevels = 6 corresponds to 1 um finest resolution; default base mesh is 32 um",
    )
    fid.add_argument(
        "--rad",
        required=True,
        choices=["diffusion", "transport"],
        help="Radiation model (grey)",
    )
    fid.add_argument(
        "--solver",
        choices=["jacobi", "explicit"],
        default="jacobi",
        help="Transport solver (ignored for diffusion)",
    )
    fid.add_argument(
        "--do_tn",
        action="store_true",
        default=False,
        help="Enable thermonuclear burn (DT fuel initialized 50/50 D/T by "
        "number density; requires the isotope data file, see --tn_data)",
    )
    fid.add_argument(
        "--tn_data",
        default=None,
        help="Path to the TN isotope/reaction HDF5 data file "
        "(default: eos/isotope_data.hdf5 next to this deck)",
    )

    geom = parser.add_argument_group("capsule geometry (cm)")
    geom.add_argument(
        "--gas_radius", type=float, default=0.0215, help="DT gas cavity radius"
    )
    geom.add_argument(
        "--inner_shell_thickness",
        type=float,
        default=0.0040,
        help="Inner (hi-z) shell thickness",
    )
    geom.add_argument(
        "--tamper_thickness",
        type=float,
        default=0.0090,
        help="Tamper thickness (90 um; Sauppe et al., Phys. Plasmas 33, 072705)",
    )
    geom.add_argument(
        "--ablator_outer_radius",
        type=float,
        default=0.111,
        help="Outer radius of the ablator",
    )
    geom.add_argument(
        "--ablator_thickness", type=float, default=0.0140, help="Ablator thickness"
    )
    geom.add_argument(
        "--ablator_material",
        default="Aluminum",
        choices=["Aluminum", "CH"],
        help="Ablator material",
    )
    geom.add_argument(
        "--tamper_material",
        default="CH",
        choices=["Beryllium", "CH"],
        help="Tamper material (CH at 1.02 g/cc per the reference design)",
    )
    geom.add_argument(
        "--inner_shell_material",
        default="Tungsten",
        choices=["Tungsten", "Molybdenum"],
        help="Inner shell material",
    )

    gap = parser.add_argument_group("ablator gap (2-D / 3-D)")
    gap.add_argument(
        "--do_gap", action="store_true", default=True, help="Include the ablator gap"
    )
    gap.add_argument(
        "--no_gap", dest="do_gap", action="store_false", help="Disable the ablator gap"
    )
    gap.add_argument(
        "--inner_gap_width",
        type=float,
        default=2.0e-4,
        help="Horizontal (inner) gap full width (2 um)",
    )
    gap.add_argument(
        "--vertical_gap_width",
        type=float,
        default=2.0e-4,
        help="Vertical gap full width (2 um)",
    )
    gap.add_argument(
        "--outer_gap_width",
        type=float,
        default=2.0e-4,
        help="Outer (angled) gap full width (2 um)",
    )
    gap.add_argument(
        "--gap_vertical_offset",
        type=float,
        default=0.0,
        help="Axial offset of the inner (horizontal) gap slab",
    )
    gap.add_argument(
        "--vertical_gap_distance",
        type=float,
        default=0.0042,
        help="Axial length of the vertical gap run",
    )
    gap.add_argument(
        "--outer_gap_angle",
        type=float,
        default=15.0,
        help="Outer gap opening angle (degrees)",
    )
    gap.add_argument(
        "--outer_gap_depth",
        type=float,
        default=0.003,
        help="Radial depth (from ablator outer radius) of the vertical gap",
    )
    gap.add_argument(
        "--gold_coating_thickness",
        type=float,
        default=2.5e-5,
        help="Gold liner thickness on the inner gap surface (250 nm; 0 disables)",
    )

    tube = parser.add_argument_group("fill tube / glue (3-D, or 2-D on the pole)")
    tube.add_argument(
        "--do_fill_tube",
        action="store_true",
        default=False,
        help="Include the fill tube, bore holes, and glue blobs",
    )
    tube.add_argument(
        "--fill_tube_axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Axis the fill tube is aligned with (0=x, 1=y equator; "
        "2=z pole). Default: x in 3-D (equator), z in 2-D RZ (pole). "
        "2-D RZ only supports the z-axis (pole).",
    )
    tube.add_argument(
        "--filltube_outer_diameter_inner",
        type=float,
        default=5.0e-4,
        help="Fill-tube outer diameter at the inner shell (5 um)",
    )
    tube.add_argument(
        "--filltube_outer_diameter_ablator",
        type=float,
        default=1.2e-3,
        help="Fill-tube outer diameter at the ablator (12 um)",
    )
    tube.add_argument(
        "--filltube_wall_thickness",
        type=float,
        default=5.0e-5,
        help="Fill-tube wall thickness",
    )
    tube.add_argument(
        "--inner_shell_bore_diameter",
        type=float,
        default=7.0e-4,
        help="Inner-shell bore-hole diameter",
    )
    tube.add_argument(
        "--ablator_bore_diameter",
        type=float,
        default=1.7e-3,
        help="Ablator bore-hole diameter",
    )
    tube.add_argument(
        "--inner_glue_depth", type=float, default=5.0e-4, help="Inner glue blob depth"
    )
    tube.add_argument(
        "--inner_glue_radius", type=float, default=1.4e-3, help="Inner glue blob radius"
    )
    tube.add_argument(
        "--inner_glue_height", type=float, default=5.0e-4, help="Inner glue blob height"
    )
    tube.add_argument(
        "--outer_glue_depth", type=float, default=5.0e-4, help="Outer glue blob depth"
    )
    tube.add_argument(
        "--outer_glue_radius", type=float, default=3.4e-3, help="Outer glue blob radius"
    )
    tube.add_argument(
        "--outer_glue_height", type=float, default=5.0e-4, help="Outer glue blob height"
    )

    drive = parser.add_argument_group("radiation drive")
    drive.add_argument(
        "--tr_drive",
        type=float,
        default=200.0,
        help="Constant radiation drive temperature (eV)",
    )

    return parser


def resolve_args(args):
    """Fill dimension-dependent defaults and validate cross-argument constraints.

    The fill tube defaults to an equatorial axis (x) in 3-D and to the pole (z) in 2-D
    RZ.  2-D RZ can only represent a pole-aligned (z) tube; an equatorial tube there is
    not axisymmetric and is rejected.
    """
    if args.fill_tube_axis is None:
        args.fill_tube_axis = 2 if args.dim == 2 else 0
    if args.dim == 2 and args.do_fill_tube and args.fill_tube_axis != 2:
        raise SystemExit(
            "2-D RZ only supports a pole-aligned fill tube (--fill_tube_axis 2)."
        )
    if args.dim == 1 and args.do_fill_tube:
        raise SystemExit("The fill tube requires --dim 2 (pole) or --dim 3.")
    return args


# ----------------------------------------------------------------------------------------
# Derived geometry (shared by make_input and plot_mesh.py)
# ----------------------------------------------------------------------------------------
def build_geometry(args):
    """Return a dict of derived radii and the fill-tube availability flag."""
    gas_r = args.gas_radius
    shell_outer = gas_r + args.inner_shell_thickness
    tamper_outer = shell_outer + args.tamper_thickness
    ablator_outer = args.ablator_outer_radius
    ablator_inner = ablator_outer - args.ablator_thickness
    cushion_inner = tamper_outer
    cushion_outer = ablator_inner

    # The fill tube is meaningful in 3-D, or in 2-D RZ when placed on the pole (Z axis).
    fill_tube_active = args.do_fill_tube and (
        args.dim == 3 or (args.dim == 2 and args.fill_tube_axis == 2)
    )
    # The gap is only meaningful in 2-D / 3-D.
    gap_active = args.do_gap and args.dim > 1
    gold_active = gap_active and args.gold_coating_thickness > 0.0

    return {
        "gas_r": gas_r,
        "shell_outer": shell_outer,
        "tamper_outer": tamper_outer,
        "ablator_inner": ablator_inner,
        "ablator_outer": ablator_outer,
        "cushion_inner": cushion_inner,
        "cushion_outer": cushion_outer,
        "fill_tube_active": fill_tube_active,
        "gap_active": gap_active,
        "gold_active": gold_active,
    }


# ----------------------------------------------------------------------------------------
# Mesh helper
# ----------------------------------------------------------------------------------------
def _drive_faces(args):
    """Return the per-face ``radiation_transport/drive`` flags for transport.

    These mark the physical outer faces (left at ``outflow`` in the mesh block) where the
    constant-temperature radiation drive BC is enrolled; symmetry/axis and periodic faces
    are omitted so they keep their mesh-level BC.  Mirrors the driven faces in
    _mesh_params.
    """
    if args.dim == 1:
        return {"ox1_bc": "drive"}
    if args.dim == 2:
        return {"ox1_bc": "drive", "ix2_bc": "drive", "ox2_bc": "drive"}
    return {
        "ix1_bc": "drive",
        "ox1_bc": "drive",
        "ix2_bc": "drive",
        "ox2_bc": "drive",
        "ix3_bc": "drive",
        "ox3_bc": "drive",
    }


def _mesh_params(args, rmax):
    """Build the parthenon/mesh extent + BC dict for the requested dimensionality.

    Physical outer faces get ``outflow`` at the mesh (hydro) level; symmetry/axis faces
    get reflecting; the azimuthal wedge is periodic.  For transport the radiation drive is
    layered on top of the outflow faces via the ``radiation_transport/drive`` block (see
    make_input / _drive_faces).
    """
    drive_flag = "outflow"

    if args.dim == 1:
        # 1-D spherical: x1 = r >= 0.
        return {
            "nx1": rmax["nx1"],
            "x1min": 0.0,
            "x1max": rmax["r"],
            "ix1_bc": "reflecting",
            "ox1_bc": drive_flag,
            "nx2": 1,
            "x2min": 0.0,
            "x2max": np.pi,
            "ix2_bc": "periodic",
            "ox2_bc": "periodic",
            "nx3": 1,
            "x3min": 0.0,
            "x3max": 2.0 * np.pi,
            "ix3_bc": "periodic",
            "ox3_bc": "periodic",
        }
    if args.dim == 2:
        # 2-D RZ: x1 = R >= 0, x2 = Z.
        return {
            "nx1": rmax["nx1"],
            "x1min": 0.0,
            "x1max": rmax["r"],
            "ix1_bc": "reflecting",
            "ox1_bc": drive_flag,
            "nx2": 2 * rmax["nx1"],
            "x2min": -rmax["r"],
            "x2max": rmax["r"],
            "ix2_bc": drive_flag,
            "ox2_bc": drive_flag,
            "nx3": 1,
            "x3min": 0.0,
            "x3max": 2.0 * np.pi,
            "ix3_bc": "periodic",
            "ox3_bc": "periodic",
        }
    # 3-D Cartesian.
    n = 2 * rmax["nx1"]
    return {
        "nx1": n,
        "x1min": -rmax["r"],
        "x1max": rmax["r"],
        "ix1_bc": drive_flag,
        "ox1_bc": drive_flag,
        "nx2": n,
        "x2min": -rmax["r"],
        "x2max": rmax["r"],
        "ix2_bc": drive_flag,
        "ox2_bc": drive_flag,
        "nx3": n,
        "x3min": -rmax["r"],
        "x3max": rmax["r"],
        "ix3_bc": drive_flag,
        "ox3_bc": drive_flag,
    }


# ----------------------------------------------------------------------------------------
# Deck construction
# ----------------------------------------------------------------------------------------
def make_input(args):
    import riot

    geo = build_geometry(args)

    ablator = MATERIALS[args.ablator_material]
    tamper = MATERIALS[args.tamper_material]
    inner_shell = MATERIALS[args.inner_shell_material]

    # "thermal" initialization); diffusion uses a package-level constant-temperature
    # radiation boundary.
    riot.input(
        "riot",
        problem="region_pgen",
        sparse_init=False,
        sparse_dealloc=True,
        verbose=False,
        use_mpi_shared_memory=True,
    )
    riot.input("comment", problem="Double Shell ICF Capsule")
    riot.input("parthenon/job", problem_id="doubleshell")

    rad_output = "rmg.Egroup" if args.rad == "diffusion" else "c.c.rad.moments"
    output_vars = [
        "c.c.bulk.rho",
        "c.c.bulk.pressure",
        "c.c.bulk.temperature",
        "c.c.mat.rho",
        "c.c.mat.volume_fraction",
        rad_output,
    ]
    if args.do_tn:
        output_vars += ["c.c.mat.iso", "c.c.mat.tn_reaction_density"]
    riot.input("parthenon/output0", file_type="hst", dt=1.0e-12)
    riot.input(
        "parthenon/output1",
        variables=output_vars,
        file_type="hdf5",
        dt=1.0e-10,
        sparse_seed_nans=True,
    )
    riot.input("parthenon/output2", file_type="rst", dt=1.0e-9)
    riot.input("parthenon/time", nlim=-1, tlim=1.5e-8, integrator="rk2", ncycle_out=1)

    # +/- rmax cm domain; base mesh resolution nx1 radial cells.
    rmax = {"r": 0.2880, "nx1": 90}
    meshparams = _mesh_params(args, rmax)
    # Diffusion needs geometric multigrid for its implicit solve.
    mesh_extra = {"multigrid": True} if args.rad == "diffusion" else {}
    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=args.nlevels,
        derefine_count=10,
        nghost=2,
        task_collection_timeout_in_seconds=10000,
        **meshparams,
        **mesh_extra,
    )
    riot.input(
        "parthenon/meshblock",
        nx1=18,
        nx2=18 if args.dim > 1 else 1,
        nx3=18 if args.dim > 2 else 1,
    )
    refine_field = 1 if args.dim == 3 else 0
    for n, m in enumerate(REFINE_LEVELS):
        riot.input(
            f"parthenon/refinement{n}",
            method="magnitude",
            comparator="greater_than",
            field=f"c.c.mat.rho_{n}",
            refine_tol=1.5,
            derefine_tol=0.5,
            max_level=REFINE_LEVELS[m][refine_field],
        )
    # Try a small initial timestep to avoid any issues related to operator-splitting
    # at very early times while the radiation is coming in from the hot boundary.
    riot.input(
        "parthenon/time", dt_init=1.0e-14 * 0.5 ** (args.nlevels - 3), dt_factor=1.2
    )

    # --- materials + EOS ---------------------------------------------------------------
    riot.input("materials", sparse_init=False, sparse_dealloc=True)

    def add_material(idx, catalog_name, density=None, isotopes=None):
        info = MATERIALS[catalog_name]
        rho = info["density"] if density is None else density
        # Refinement level comes from the REFINE_LEVELS table (keyed by matid), which is
        # the single place to tune resolution vs. memory for a suite of runs.
        max_level = refine_level(idx, args.dim)
        eos_name = f"{catalog_name}EOS"
        opac = {
            "opac_a": "powerlaw",
            "kappa0_a": (
                0.0
                if (catalog_name == "Helium" and args.rad == "transport")
                else _kappa0(info["z"])
            ),
            # "kappa0_a": _kappa0(info["z"]),
            "kappa_Tpower_a": -3.5,
            "kappa_Rhopower_a": 1.0,
        }
        # For TN burn, declare the isotopes (ZAIDs) carried by this material.
        iso = (
            {}
            if isotopes is None
            else (
                {f"isotope{n}": z[0] for n, z in enumerate(isotopes)}
                | {f"isotope{n}_mfrac": z[1] for n, z in enumerate(isotopes)}
            )
        )
        riot.input(
            f"material{idx}",
            label=catalog_name.lower(),
            eos=eos_name,
            max_bnd_level=max_level,
            max_mat_level=max_level,
            **opac,
            **iso,
        )
        if info["eos"] == "IdealGas":
            riot.input(
                eos_name, eos_type="IdealGas", Gamma=1.65767, Cv=31158319.97244526
            )
        else:
            riot.input(
                eos_name,
                eos_type="SpinerEOSDependsRhoT",
                sesame_id=info["sesame_id"],
                filename=str(ifile_dir / "eos" / "materials.sp5"),
            )
        return rho

    add_material(MATID["background"], "Helium")
    add_material(MATID["ablator"], args.ablator_material)
    add_material(MATID["foam"], "CHFoam")
    add_material(MATID["tamper"], args.tamper_material)
    add_material(MATID["inner_shell"], args.inner_shell_material)
    dt_isotopes = (
        [
            (DT_ISOTOPES["D"]["zaid"], DT_MFRAC["D"]),
            (DT_ISOTOPES["T"]["zaid"], DT_MFRAC["T"]),
            (DT_ISOTOPES["HE4"]["zaid"], 0.0),
        ]
        if args.do_tn
        else None
    )

    add_material(MATID["dt"], "DT", isotopes=dt_isotopes)
    add_material(MATID["gold"], "Gold")
    add_material(MATID["glass"], "Glass")

    # --- regions -----------------------------------------------------------------------
    # NOTE: region priority increases with index -- when masks overlap, the HIGHEST
    # region index wins (src/riot_pgen/regions.cpp::set_mask).  The primitive spherical
    # shells come first (low indices); the custom Python masks (gap, gold, fill tube)
    # come at higher indices so they overwrite the shells they cut into.
    riot.input("regions", nlev_min=0, nlev_max=5)

    bg_temp = (1.0e4) if (args.rad == "diffusion") else BACKGROUND_T

    riot.input(
        "region0",
        name="background",
        mask_type="background",
        matid=MATID["background"],
        c_m_rho=MATERIALS["Helium"]["density"],
        c_m_temperature=bg_temp,
    )

    riot.input(
        "region1",
        name="Ablator",
        mask_type="inside_spherical_shell",
        matid=MATID["ablator"],
        inner_radius=geo["ablator_inner"],
        outer_radius=geo["ablator_outer"],
        c_m_rho=ablator["density"],
        c_m_pressure=INIT_P,
    )

    riot.input(
        "region2",
        name="Cushion",
        mask_type="inside_spherical_shell",
        matid=MATID["foam"],
        inner_radius=geo["cushion_inner"],
        outer_radius=geo["cushion_outer"],
        c_m_rho=MATERIALS["CHFoam"]["density"],
        c_m_pressure=INIT_P,
    )

    riot.input(
        "region3",
        name="Tamper",
        mask_type="inside_spherical_shell",
        matid=MATID["tamper"],
        inner_radius=geo["shell_outer"],
        outer_radius=geo["tamper_outer"],
        c_m_rho=tamper["density"],
        c_m_pressure=INIT_P,
    )

    riot.input(
        "region4",
        name="CapsuleShell",
        mask_type="inside_spherical_shell",
        matid=MATID["inner_shell"],
        inner_radius=geo["gas_r"],
        outer_radius=geo["shell_outer"],
        c_m_rho=inner_shell["density"],
        c_m_pressure=INIT_P,
    )

    riot.input(
        "region5",
        name="DTFuel",
        mask_type="inside_sphere",
        matid=MATID["dt"],
        radius=geo["gas_r"],
        c_m_rho=MATERIALS["DT"]["density"],
        c_m_pressure=INIT_P,
    )

    next_region = 6

    # Gap + gold liner (custom Python masks).  GoldLiner uses the full gap half-widths;
    # Gap subtracts the gold thickness and sits at a HIGHER index so it overwrites the
    # liner interior, leaving gold as a thin rim on the gap surface.
    if geo["gap_active"]:
        gold_thickness = args.gold_coating_thickness if geo["gold_active"] else 0.0
        gap_params = dict(
            inner_gap_width=args.inner_gap_width,
            vertical_gap_width=args.vertical_gap_width,
            outer_gap_width=args.outer_gap_width,
            outer_gap_angle=args.outer_gap_angle * np.pi / 180.0,
            outer_gap_depth=args.outer_gap_depth,
            gap_vertical_offset=args.gap_vertical_offset,
            vertical_gap_distance=args.vertical_gap_distance,
            ablator_inner=geo["ablator_inner"],
            ablator_outer=geo["ablator_outer"],
            dim=args.dim,
        )
        if geo["gold_active"]:
            riot.input(
                f"region{next_region}",
                name="GoldLiner",
                mask_type="python",
                matid=MATID["gold"],
                c_m_rho=MATERIALS["Gold"]["density"],
                c_m_pressure=INIT_P,
            )
            riot.input("GoldLiner/params", gold_thickness=0.0, **gap_params)
            next_region += 1

        riot.input(
            f"region{next_region}",
            name="Gap",
            mask_type="python",
            matid=MATID["background"],
            c_m_rho=MATERIALS["Helium"]["density"],
            c_m_temperature=BACKGROUND_T,
        )
        riot.input("Gap/params", gold_thickness=gold_thickness, **gap_params)
        next_region += 1

    # Fill tube, bore holes, glue blobs (custom Python masks).
    if geo["fill_tube_active"]:
        tube_common = dict(
            fill_tube_axis=args.fill_tube_axis,
            shell_inner=geo["gas_r"],
            tamper_outer=geo["tamper_outer"],
            ablator_inner=geo["ablator_inner"],
            ablator_outer=geo["ablator_outer"],
            inner_shell_bore_diameter=args.inner_shell_bore_diameter,
            ablator_bore_diameter=args.ablator_bore_diameter,
        )
        filltube_geom = dict(
            fill_tube_axis=args.fill_tube_axis,
            filltube_r0=geo["gas_r"],
            filltube_r1=geo["ablator_outer"],
            filltube_outer_diameter_inner=args.filltube_outer_diameter_inner,
            filltube_outer_diameter_ablator=args.filltube_outer_diameter_ablator,
            filltube_wall_thickness=args.filltube_wall_thickness,
        )

        # The bore holes are modeled as filled with glue (full-density CH), matching the
        # glue blobs, rather than mixing in the hohlraum (helium) background gas.
        riot.input(
            f"region{next_region}",
            name="BoreHoles",
            mask_type="python",
            matid=MATID["foam"],
            c_m_rho=MATERIALS["CH"]["density"],
            c_m_pressure=INIT_P,
        )
        riot.input("BoreHoles/params", **tube_common)
        next_region += 1

        riot.input(
            f"region{next_region}",
            name="GlueBlobs",
            mask_type="python",
            matid=MATID["foam"],
            c_m_rho=MATERIALS["CH"]["density"],
            c_m_pressure=INIT_P,
        )
        riot.input(
            "GlueBlobs/params",
            inner_glue_depth=args.inner_glue_depth,
            inner_glue_radius=args.inner_glue_radius,
            inner_glue_height=args.inner_glue_height,
            outer_glue_depth=args.outer_glue_depth,
            outer_glue_radius=args.outer_glue_radius,
            outer_glue_height=args.outer_glue_height,
            **tube_common,
        )
        next_region += 1

        riot.input(
            f"region{next_region}",
            name="FillTube",
            mask_type="python",
            matid=MATID["glass"],
            c_m_rho=MATERIALS["Glass"]["density"],
            c_m_pressure=INIT_P,
        )
        riot.input("FillTube/params", **filltube_geom)
        next_region += 1

        riot.input(
            f"region{next_region}",
            name="DTInTube",
            mask_type="python",
            matid=MATID["dt"],
            c_m_rho=MATERIALS["DT"]["density"],
            c_m_pressure=INIT_P,
        )
        riot.input("DTInTube/params", **filltube_geom)
        next_region += 1

    # --- physics -----------------------------------------------------------------------
    if args.rad == "diffusion":
        riot.input(
            "physics",
            hydro=True,
            multigroup_diffusion=True,
            tn=args.do_tn,
            sparse_physics=False,
        )
    else:
        riot.input(
            "physics",
            hydro=True,
            radiation_transport=True,
            tn=args.do_tn,
            sparse_physics=False,
        )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.3,
        riemann="chllc",
        amr_interface=False,
        mass_frac_thresh=1.0e-8,
    )

    # --- radiation configuration (grey) ------------------------------------------------
    # Grey: leave materials/group_bounds unset -> ngroups defaults to 1.
    tr_drive_K = args.tr_drive * EV_TO_K

    if args.rad == "diffusion":
        # boundary_T order: [ix1, ox1, ix2, ox2, ix3, ox3].  Drive the physical outer
        # faces; hold symmetry/axis faces at the cold background temperature.
        if args.dim == 1:
            boundary_T = [COLD_T, tr_drive_K, COLD_T, COLD_T, COLD_T, COLD_T]
        elif args.dim == 2:
            boundary_T = [COLD_T, tr_drive_K, tr_drive_K, tr_drive_K, COLD_T, COLD_T]
        else:
            boundary_T = [tr_drive_K] * 6
        # opacity_temp_min floors the temperature used to evaluate the Kramers
        # kappa ~ T^-3.5 opacity; without it the cold (~298 K) capsule interior gives a
        # divergent opacity and the first implicit solve NaNs the central cell.
        # A FINITE diffusion cfl is essential: it activates the P1 light-crossing dt vote
        # (sqrt3*cfl*dx/c_light).
        # We do some things to try to control time-steps carefully, but this is fragile.
        riot.input(
            "diffusion",
            update_temperature=True,
            nr_tolerance=1.0e-3,
            nriter=20,
            local_nriter=0,
            print_per_nr_step=True,
            opacity_temp_min=COLD_T,
            cfl=1.0e100,
            boundary_condition="constant_temperature",
            boundary_T=boundary_T,
            timestep_temperature_scale=0.0,
            temperature_fractional_change_target=1.0,
            timestep_min_temperature=1.0e3,
            amr_min_temperature=1e100,
            amr_min_density=1e100,
            maximum_timestep_reduction_factor=1.0,
        )
        riot.input(
            "diffusion/linear_solver_params",
            relative_residual_tolerance=1.0e-5,
            absolute_residual_tolerance=1.0e-12,
            volume_weight=True,
            max_coarsenings=100000,
            precondition=True,
            preconditioner="Multigrid",
            max_iterations=200,
            presmoother="SRJ1",
            postsmoother="SRJ2",
            block_interior_prolongation="Constant",
            print_per_step=False,
        )
    else:
        riot.input(
            "radiation_transport",
            do_jacobi=(args.solver == "jacobi"),
            do_explicit=(args.solver == "explicit"),
            nlevel=5 - args.dim,  # This sets number of angles.
            coupling=True,
            affect_fluid=True,
            troot_tol=1.0e-4,
            troot_max_iter=50,
            rotate_geo=1,
        )
        riot.input(
            "radiation_transport/jacobi", err_thr=1.0e-3, niter_limit=10000, verbose=2
        )
        riot.input(
            "radiation_transport/explicit",
            dt_ratio_hyperbolic=1000.0,
            integrator="rk2",
            verbose=2,
        )
        # Radiation drive boundary (src/radiation_transport/algorithms/radiation_bcs.hpp):
        # the per-face ix?_bc/ox?_bc="drive" flags enroll the drive BC on the physical
        # outer faces (left as "outflow" in the mesh block), and trad_bc sets the constant
        # drive temperature in KELVIN.  The interior intensity initialization is left at
        # its default ("thermal" -- the local Planckian) from radiation_transport/init.
        riot.input(
            "radiation_transport/drive", trad_bc=tr_drive_K, **_drive_faces(args)
        )

    # --- thermonuclear burn ------------------------------------------------------------
    if args.do_tn:
        tn_data = (
            Path(args.tn_data)
            if args.tn_data
            else ifile_dir / "eos" / "isotope_data.hdf5"
        )
        riot.input("isotope_data", filename=str(tn_data))
        # Primary DT reaction
        riot.input("tnburn", reaction0="d+t->n+a")


# ========================================================================================
# Custom Python region masks.
#
# Each class name matches the ``name=`` of its region block.  The region_pgen machinery
# instantiates the class (zero-arg __init__), injects axis indices ``self.x=0``,
# ``self.y=1``, ``self.z=2``, and injects every key from the matching ``<Name>/params``
# block as a typed attribute.  ``mask(self, pos)`` returns a boolean numpy array; ``pos``
# has shape ``(N, 3)`` holding Cartesian coordinates (see module docstring).
# ========================================================================================


class _GapBase:
    """Shared geometry for the ablator gap.

    The gap has three OR'd pieces (following the historical C++ pgen):
      (A) an inner horizontal slab centered at ``gap_vertical_offset``,
      (B) a vertical run at cylindrical radius ``vertical_gap_center``, and
      (C) an outward-angled section.

    ``gold_shrink`` subtracts the gold-liner thickness from every half-width so that a
    separate, full-width GoldLiner region left underneath shows through as a thin rim.
    """

    def _derived(self):
        g = self.gold_shrink
        slope = -np.tan(self.outer_gap_angle)
        in_half = 0.5 * self.inner_gap_width - g
        vert_half = 0.5 * self.vertical_gap_width - g
        out_half_z = (0.5 * self.outer_gap_width - g) * np.cos(self.outer_gap_angle)
        # The vertical gap is centered a depth ``outer_gap_depth`` inside the ablator.
        vertical_gap_center = self.ablator_outer - self.outer_gap_depth
        return in_half, vert_half, out_half_z, slope, vertical_gap_center

    def _gap_mask(self, pos):
        if self.dim == 1:
            return np.zeros(pos.shape[0], dtype=bool)
        x = pos[:, self.x]
        y = pos[:, self.y]
        z = pos[:, self.z]
        r_cylin = np.hypot(x, y)
        r_sph = np.hypot(r_cylin, z)
        in_half, vert_half, out_half_z, slope, vgc = self._derived()
        vo = self.gap_vertical_offset

        cond1 = (z > vo - in_half) & (z < vo + in_half) & (r_cylin < vgc + vert_half)
        cond2 = (
            (r_cylin < vgc + vert_half)
            & (r_cylin > vgc - vert_half)
            & (z < vo + in_half)
            & (z > vo - self.vertical_gap_distance - out_half_z)
        )
        cond3 = (
            (z > vo - self.vertical_gap_distance - out_half_z + slope * (r_cylin - vgc))
            & (
                z
                < vo - self.vertical_gap_distance + out_half_z + slope * (r_cylin - vgc)
            )
            & (r_cylin > vgc - vert_half)
        )

        within_shell = (r_sph >= self.ablator_inner) & (r_sph <= self.ablator_outer)
        return (cond1 | cond2 | cond3) & within_shell


class Gap(_GapBase):
    """Void (helium) filling the ablator gap, minus the gold liner thickness."""

    def __init__(self):
        pass

    def mask(self, pos):
        self.gold_shrink = self.gold_thickness
        return self._gap_mask(pos)


class GoldLiner(_GapBase):
    """Full-width gap volume in gold; the Gap region overwrites its interior."""

    def __init__(self):
        pass

    def mask(self, pos):
        self.gold_shrink = 0.0
        return self._gap_mask(pos)


class _TubeBase:
    """Shared helpers for the fill-tube family of masks."""

    def _cyl_radius(self, pos):
        """Cylindrical radius about the fill-tube axis."""
        axis = self.fill_tube_axis
        if axis == self.x:
            return np.hypot(pos[:, self.y], pos[:, self.z])
        if axis == self.y:
            return np.hypot(pos[:, self.x], pos[:, self.z])
        return np.hypot(pos[:, self.x], pos[:, self.y])

    def _sph_radius(self, pos):
        return np.sqrt(pos[:, self.x] ** 2 + pos[:, self.y] ** 2 + pos[:, self.z] ** 2)

    def _tube_outer_radius(self, pos):
        """Outer radius of the conical tube as a function of axial position.

        The outer diameter interpolates linearly from ``filltube_outer_diameter_inner``
        at the inner-shell end (``d = 0``) to ``filltube_outer_diameter_ablator`` at the
        ablator end (``d = span``).
        """
        d = pos[:, self.fill_tube_axis] - self.filltube_r0
        span = self.filltube_r1 - self.filltube_r0
        frac = np.clip(d / span, 0.0, 1.0)
        diam = self.filltube_outer_diameter_inner + frac * (
            self.filltube_outer_diameter_ablator - self.filltube_outer_diameter_inner
        )
        return d, 0.5 * diam


class FillTube(_TubeBase):
    """Conical glass tube wall surrounding the DT fill channel."""

    def __init__(self):
        pass

    def mask(self, pos):
        r = self._cyl_radius(pos)
        d, rout = self._tube_outer_radius(pos)
        rin = rout - self.filltube_wall_thickness
        return (d >= 0.0) & (r < rout) & (r >= rin)


class DTInTube(_TubeBase):
    """DT gas filling the interior of the conical fill tube."""

    def __init__(self):
        pass

    def mask(self, pos):
        r = self._cyl_radius(pos)
        d, rout = self._tube_outer_radius(pos)
        rin = rout - self.filltube_wall_thickness
        return (d >= 0.0) & (r < rin)


class BoreHoles(_TubeBase):
    """Bore holes drilled through the inner shell and ablator for the fill tube."""

    def __init__(self):
        pass

    def mask(self, pos):
        r = self._cyl_radius(pos)
        r_sph = self._sph_radius(pos)
        # The bore is drilled from the +axis pole inward for the fill tube; the
        # cylindrical radius alone is symmetric in the axial coordinate, so without
        # this half-space restriction the hole is punched through BOTH poles (and,
        # in 2-D RZ, straight down the -Z axis and through the core).  Match the
        # +axis convention used by FillTube/DTInTube (d >= 0) and GlueBlobs.
        on_tube_side = pos[:, self.fill_tube_axis] > 0.0
        # The inner bore is drilled through the SOLID stack (inner shell + tamper) only,
        # from the gas-cavity surface (shell_inner) out to the tamper surface.  Without the
        # lower bound r_sph >= shell_inner the "hole" would be a solid helium cylinder
        # running through the gas cavity and all the way to the origin on the axis.
        inner = (
            (r < 0.5 * self.inner_shell_bore_diameter)
            & (r_sph >= self.shell_inner)
            & (r_sph <= self.tamper_outer)
        )
        outer = (
            (r < 0.5 * self.ablator_bore_diameter)
            & (r_sph >= self.ablator_inner)
            & (r_sph <= self.ablator_outer)
        )
        return (inner | outer) & on_tube_side


class GlueBlobs(_TubeBase):
    """Conical glue blobs sealing the fill tube against the shell surfaces."""

    def __init__(self):
        pass

    def mask(self, pos):
        r = self._cyl_radius(pos)
        r_sph = self._sph_radius(pos)
        axis = pos[:, self.fill_tube_axis]

        d_inner = axis - self.tamper_outer
        inner_cone = (
            r < self.inner_glue_radius * (1.0 - d_inner / self.inner_glue_height)
        ) & (d_inner > -self.inner_glue_depth)
        inner_blob = inner_cone & (r_sph >= self.tamper_outer)

        d_outer = axis - self.ablator_outer
        outer_cone = (
            r < self.outer_glue_radius * (1.0 - d_outer / self.outer_glue_height)
        ) & (d_outer > -self.outer_glue_depth)
        outer_blob = outer_cone & (r_sph >= self.ablator_outer)

        return inner_blob | outer_blob


if __name__ == "__main__":
    import riot

    make_input(resolve_args(build_parser().parse_args()))
    riot.input.generate_input()
