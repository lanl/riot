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

# Gardiner & Stone (2005) advected field loop: the multi-dimensional constrained-transport
# test. Two configurations, which are different CODE PATHS and not two resolutions of one
# thing: with x3 collapsed MHD::AssembleEdgeEMF takes its collapsed branch, and only the 3D
# case reaches the upwind branch of UpwindEMF<X1DIR> and <X2DIR>.
#
# THE ORACLE IS ANALYTIC, not a reference file. The loop is dynamically negligible
# (beta ~ 1e6), so the exact solution is pure advection, and tlim is exactly one crossing in
# every direction on a periodic box -- so the exact final state IS the initial state. That
# makes the shape error |B_final - B_initial| directly measurable with no reference solution
# and no interpolation.
#
# What each criterion catches, since "div B is small" is not the whole test:
#   - shape error: an EMF averaging error diffuses the loop; an orientation or upwinding error
#     distorts it anisotropically even though the advection velocity is uniform.
#   - axial field: with the loop axis along x3 the field is (B1, B2, 0) and the axial component
#     obeys d_t B3 = v3 (d_1 B1 + d_2 B2) = v3 div B = 0 -- but only because E1 = -v3 B2 and
#     E2 = +v3 B1 cancel against each other. Any inconsistency between how the two EMF
#     components are assembled appears immediately as a spurious axial field, with nothing to
#     hide behind. This is why v3 is nonzero in BOTH cases: with v3 = 0 the check is vacuous.
#   - energy retention: a scheme can hold div B at roundoff and still be far too diffusive.
#
# THRESHOLDS ARE THE DONOR'S OWN, taken verbatim from
# artemis/tst/scripts/mhd/field_loop.py (max_divb 1e-10, max_axial_field 1e-10, fine shape
# error 0.6, fine energy retention 0.5, plus the fine-improves-on-coarse pair), together with
# its metric DEFINITIONS -- a threshold is meaningless without the normalization it was
# calibrated against. They are the donor's demonstrated performance rather than numbers
# invented here.
#
# COVERAGE DIFFERENCE, stated rather than glossed: the donor's second case is a loop tilted
# onto a box DIAGONAL. RIOT's mhd_field_loop generator offers loop_axis (1/2/3), which rotates
# the loop onto each axis instead, so the 3D case here sweeps all three axes -- which covers
# every pair of EMF components -- but no test in this suite propagates a loop obliquely to the
# grid. That gap is recorded in plan_histories/artemis_mhd_port/OPEN_CONCERNS.md.

# Modules
import logging
import numpy as np
import h5py
import scripts.utils.riot as riot
from scripts.utils.mhd_analysis import Checks, assemble, field_snapshot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

# Frozen thresholds, quoted from the donor script.
max_divb = 1.0e-10
max_axial_field = 1.0e-10
max_fine_shape_error = 0.6
min_fine_energy_retention = 0.5
shape_improvement_factor = 0.9
max_conservation_error = 1.0e-10

# (name, input deck, resolutions, axial component index, extra arguments)
#
# The 2D case runs with v3 = 1.0, overriding the deck's 0.0, to make the axial-field check
# non-vacuous -- see the header. The 3D deck already has v3 = 1.0.
cases = [
    {
        "name": "planar",
        "input_id": "mhd/field_loop",
        "resolutions": (64, 128),
        "shape": lambda n: ((n, n // 2, 1), (n // 4, n // 4, 1)),
        "extra": ["mhd_field_loop/v3=1.0"],
        "axis": 3,
    },
    {
        "name": "loop3d",
        "input_id": "mhd/field_loop_3d",
        "resolutions": (16, 32),
        "shape": lambda n: ((2 * n, n, n), (n, n // 2, n // 2)),
        "extra": [],
        "axis": 3,
    },
]

# Loop axes swept at the COARSE 3D resolution only. With loop_axis=3 the field has B3 = 0, so
# the F3 face exchange carries nothing but zeros and a defect in it would be invisible; the
# sweep is what makes all three face directions carry the nonzero components in turn.
loop_axis_sweep = (1, 2)


def _tag(case, resolution, axis=None):
    if axis is None:
        return "fl_{}_{}".format(case["name"], resolution)
    return "fl_{}_{}_axis{}".format(case["name"], resolution, axis)


def _args(case, resolution, axis=None):
    nx, nmb = case["shape"](resolution)
    args = [
        "parthenon/job/problem_id=" + _tag(case, resolution, axis),
        "parthenon/time/nlim=100000",
        "mhd/monitor_divb=false",
        "parthenon/time/ncycle_out=100000",
    ]
    for i in range(3):
        args.append("parthenon/mesh/nx{}={:d}".format(i + 1, nx[i]))
        args.append("parthenon/meshblock/nx{}={:d}".format(i + 1, nmb[i]))
    args += case["extra"]
    if axis is not None:
        args.append("mhd_field_loop/loop_axis={:d}".format(axis))
    return args


def run(**kwargs):
    logger.debug("Running test " + __name__)
    for case in cases:
        riot.generate(case["input_id"] + ".py")
        for resolution in case["resolutions"]:
            riot.mpirun(1, case["input_id"] + ".rin", _args(case, resolution))
    # Axis sweep, coarse 3D only.
    case = cases[1]
    for axis in loop_axis_sweep:
        riot.mpirun(
            1,
            case["input_id"] + ".rin",
            _args(case, case["resolutions"][0], axis),
        )


def _min_cell_width(path):
    """Smallest cell width in the run, over all blocks and all live directions.

    A collapsed direction holds a single cell spanning the whole extent, which is typically
    much WIDER than the live cells; including it would inflate the div B normalization and
    weaken the test, so widths are taken per direction and the minimum is over live ones only.
    """
    with h5py.File(path, "r") as f:
        widths = []
        for axis in ("x", "y", "z"):
            edges = f["Locations/" + axis][...]
            if edges.shape[-1] > 2:  # more than one cell in this direction
                widths.append(np.min(np.diff(edges, axis=-1)))
        return min(widths)


def _snapshot(tag, snap):
    path = field_snapshot(tag, snap)
    return {
        "rho": assemble(path, "c.c.bulk.rho"),
        "P": assemble(path, "c.c.bulk.pressure"),
        "vel": assemble(path, "c.c.bulk.velocity", 3),
        "E": assemble(path, "c.c.bulk.total_material_energy"),
        "B": assemble(path, "c.c.bulk.magnetic_field", 3),
        "emag": assemble(path, "c.c.bulk.magnetic_energy"),
        "divB": assemble(path, "c.c.bulk.div_magnetic_field"),
        "dx_min": _min_cell_width(path),
    }


def _metrics(initial, final, axis):
    b_magnitude = np.linalg.norm(initial["B"], axis=0)
    b_scale = b_magnitude.max()
    return {
        # Donor's definition: mean over cells of the vector difference norm, normalized by the
        # mean initial field magnitude.
        "shape_error": np.mean(np.linalg.norm(final["B"] - initial["B"], axis=0))
        / np.mean(b_magnitude),
        "retention": final["emag"].mean() / initial["emag"].mean(),
        "relative_divb": np.abs(final["divB"]).max() * final["dx_min"] / b_scale,
        "axial_field": np.abs(final["B"][axis - 1]).max() / b_scale,
    }


def _check_one(tag, case, resolution, axis=None):
    """Per-run criteria: the ones that are meaningful at a single resolution."""
    initial = _snapshot(tag, "00000")
    final = _snapshot(tag, "final")
    label = tag
    c = Checks(logger, label)

    c.finite(
        "all finite", final["rho"], final["P"], final["B"], final["E"], final["divB"]
    )
    c.gt("min rho", final["rho"].min(), 0.0)
    c.gt("min P", final["P"].min(), 0.0)

    # Conservation. Momentum is checked PER COMPONENT, which is stricter than the donor's
    # lumped mean; the threshold is still the donor's 1e-10.
    for name, get in (
        ("mass", lambda s: s["rho"].sum()),
        ("total energy", lambda s: s["E"].sum()),
    ):
        old, new = get(initial), get(final)
        c.le(
            "rel change in {}".format(name),
            abs(new - old) / abs(old),
            max_conservation_error,
        )
    mom0 = (initial["rho"] * initial["vel"]).sum(axis=(1, 2, 3))
    mom1 = (final["rho"] * final["vel"]).sum(axis=(1, 2, 3))
    scale = max(np.abs(mom0).max(), 1.0)
    for comp in range(3):
        c.le(
            "rel change in momentum {}".format(comp + 1),
            abs(mom1[comp] - mom0[comp]) / scale,
            max_conservation_error,
        )

    metrics = _metrics(initial, final, axis if axis is not None else case["axis"])
    c.le("relative divB", metrics["relative_divb"], max_divb)
    c.le("axial field", metrics["axial_field"], max_axial_field)
    return c.ok, metrics


def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    for case in cases:
        metrics = []
        for resolution in case["resolutions"]:
            ok, m = _check_one(_tag(case, resolution), case, resolution)
            analyze_status &= ok
            metrics.append(m)

        # Convergence criteria, which need both resolutions.
        coarse, fine = metrics
        c = Checks(logger, "{} convergence".format(case["name"]))
        c.le("fine shape error", fine["shape_error"], max_fine_shape_error)
        c.le(
            "fine shape error / coarse",
            fine["shape_error"] / coarse["shape_error"],
            shape_improvement_factor,
        )
        c.ge("fine energy retention", fine["retention"], min_fine_energy_retention)
        c.ge(
            "fine retention / coarse",
            fine["retention"] / coarse["retention"],
            1.0,
        )
        analyze_status &= c.ok

    # Axis sweep: the axial-field cancellation must hold with the loop on each axis in turn.
    case = cases[1]
    for axis in loop_axis_sweep:
        ok, _ = _check_one(
            _tag(case, case["resolutions"][0], axis),
            case,
            case["resolutions"][0],
            axis=axis,
        )
        analyze_status &= ok

    return bool(analyze_status)
