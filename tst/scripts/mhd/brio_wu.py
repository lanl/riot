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

# Brio & Wu (1988) MHD shock tube: shock capturing, and a comparison of all three MHD
# Riemann solvers.
#
# WHY THERE IS NO REFERENCE SOLUTION FILE HERE. Brio-Wu has no closed-form solution, and its
# compound wave (a slow shock attached to a rotational discontinuity) is exactly the feature a
# gold file would freeze at this build's resolution and solver. The port WAS verified against
# an independent Athena++ solution during development (see
# plan_histories/artemis_mhd_port/TEST_LEDGER.md, G3/G4.1 and G5.2), but that reference lives
# outside this repository. Rather than vendor it, this test asserts only criteria that need no
# external oracle:
#
#   1. Structural invariants that are EXACT statements, not tolerances. Bx is uniform along x
#      so div B = d_1 B1 = 0 identically and Bx must stay 0.75 for all time; the problem is
#      coplanar so nothing can generate vz or Bz. Each was measured at exactly 0.0.
#   2. Positivity of rho and P, i.e. no floor had to rescue the solve.
#   3. The solver diffusivity ordering HLLD < HLLE < LLF, which follows from wave structure
#      (five resolved waves vs two vs none) and is therefore oracle-independent. See the
#      comment on SELF-REFERENCE BIAS below for how it is measured without a reference
#      solution and without stacking the deck.
#   4. Rank invariance.
#
# Criterion 3 is what catches a silently degraded solver -- e.g. an HLLD whose degeneracy
# guards fire everywhere, which would leave every other check in this file passing.

# Modules
import logging
import numpy as np
import scripts.utils.riot as riot
from scripts.utils.mhd_analysis import Checks, assemble, field_snapshot, restrict1d

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

input_id = "mhd/brio_wu"

# Solvers under test, in order of expected INCREASING diffusivity.
solvers = ["mhd_hlld", "mhd_hlle", "mhd_llf"]

nx_coarse = 512
# The reference resolution must be an integer multiple of the coarse one so that the fine
# solution can be restricted onto the coarse grid exactly (see restrict1d).
refine_factor = 4
nx_fine = nx_coarse * refine_factor

# Fields the ordering is measured on. All five, because a solver comparison that holds on
# density and fails on the transverse field is a partial result, not a pass.
profile_fields = [
    ("rho", "c.c.bulk.rho", None),
    ("press", "c.c.bulk.pressure", None),
    ("vx", "c.c.bulk.velocity", 0),
    ("vy", "c.c.bulk.velocity", 1),
    ("By", "c.c.bulk.magnetic_field", 1),
]

# Frozen thresholds. The first three were measured at exactly 0.0, so these are generous by
# many orders of magnitude and exist to catch a defect, not to accommodate one.
tol_exact = 1.0e-12
tol_divb = 1.0e-10
# Rank invariance was measured BITWISE (TEST_LEDGER P01.4). It is asserted at 1e-12 rather
# than bitwise because the CFL time step is a global MPI reduction, so a different reduction
# order on a different machine may legitimately change dt in its last bits. Whether the run
# was in fact bitwise is logged either way, since losing that property is worth knowing about
# even when the test still passes.
tol_rank = 1.0e-12


def _tag(solver, suffix=""):
    return "bw_{}{}".format(solver.replace("mhd_", ""), suffix)


def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(input_id + ".py")
    logger.debug("Running test " + __name__)

    base = [
        "parthenon/mesh/nx1=" + repr(nx_coarse),
        "parthenon/meshblock/nx1=" + repr(nx_coarse // 4),
        # The deck enables the div B monitor, which prints every step. div B is checked from
        # the dump instead, so switch it off and keep the log readable.
        "mhd/monitor_divb=false",
        "parthenon/time/ncycle_out=100000",
    ]

    for solver in solvers:
        args = base + [
            "parthenon/job/problem_id=" + _tag(solver),
            "hydro/riemann=" + solver,
        ]
        riot.mpirun(1, input_id + ".rin", args)

    # Fine reference runs. Two of them, deliberately: see the bias argument in analyze().
    for solver in ("mhd_hlle", "mhd_llf"):
        args = [
            "parthenon/job/problem_id=" + _tag(solver, "_fine"),
            "hydro/riemann=" + solver,
            "parthenon/mesh/nx1=" + repr(nx_fine),
            "parthenon/meshblock/nx1=" + repr(nx_fine // 4),
            "mhd/monitor_divb=false",
            "parthenon/time/ncycle_out=100000",
        ]
        riot.mpirun(1, input_id + ".rin", args)

    # Rank invariance, with shocks present. 4 blocks over 4 ranks.
    args = base + [
        "parthenon/job/problem_id=" + _tag("mhd_hlld", "_r4"),
        "hydro/riemann=mhd_hlld",
    ]
    riot.mpirun(4, input_id + ".rin", args)


def _profile(tag):
    """Every field of one run as a 1D array in global x order."""
    path = field_snapshot(tag)
    out = {}
    for name, key, comp in profile_fields:
        d = assemble(path, key, None if comp is None else 3)
        out[name] = d[0 if comp is None else comp, 0, 0, :]
    B = assemble(path, "c.c.bulk.magnetic_field", 3)
    V = assemble(path, "c.c.bulk.velocity", 3)
    out["Bx"] = B[0, 0, 0, :]
    out["Bz"] = B[2, 0, 0, :]
    out["vz"] = V[2, 0, 0, :]
    out["divB"] = assemble(path, "c.c.bulk.div_magnetic_field")[0, 0, 0, :]
    return out


def _l1(profile, reference):
    """Normalized L1 per field: mean absolute difference over the reference dynamic range, so
    fields with different magnitudes are directly comparable."""
    out = {}
    for name, _, _ in profile_fields:
        ref = restrict1d(reference[name], refine_factor)
        out[name] = np.mean(np.abs(profile[name] - ref)) / np.ptp(ref)
    return out


def analyze():
    logger.debug("Analyzing test " + __name__)
    profiles = {s: _profile(_tag(s)) for s in solvers}
    analyze_status = True

    # --- Per-solver invariants -------------------------------------------------------------
    for solver, p in profiles.items():
        c = Checks(logger, "{} invariants".format(solver))
        c.finite("all finite", p["rho"], p["press"], p["By"])
        c.le("max|Bx - 0.75|", np.abs(p["Bx"] - 0.75).max(), tol_exact)
        c.le("max|Bz| (out of plane)", np.abs(p["Bz"]).max(), tol_exact)
        c.le("max|vz| (out of plane)", np.abs(p["vz"]).max(), tol_exact)
        c.le("max|divB|", np.abs(p["divB"]).max(), tol_divb)
        c.gt("min rho", p["rho"].min(), 0.0)
        c.gt("min P", p["press"].min(), 0.0)
        analyze_status &= c.ok

    # --- Solver diffusivity ordering -------------------------------------------------------
    #
    # SELF-REFERENCE BIAS, and how it is handled. Measuring L1 against a fine run of solver X
    # flatters solver X: its own truncation error is self-similar across resolutions, so part
    # of it cancels. Each inequality is therefore asserted against the reference that biases
    # AGAINST it, making every pass conservative:
    #
    #   HLLD < HLLE  is checked against the fine HLLE run  (favours HLLE)
    #   HLLE < LLF   is checked against the fine LLF run   (favours LLF)
    #
    # Neither inequality is ever evaluated against a reference built with its own winner.
    references = {
        "mhd_hlle": _profile(_tag("mhd_hlle", "_fine")),
        "mhd_llf": _profile(_tag("mhd_llf", "_fine")),
    }
    comparisons = [
        ("mhd_hlld", "mhd_hlle", "mhd_hlle"),
        ("mhd_hlle", "mhd_llf", "mhd_llf"),
    ]
    for better, worse, ref_solver in comparisons:
        l1_better = _l1(profiles[better], references[ref_solver])
        l1_worse = _l1(profiles[worse], references[ref_solver])
        c = Checks(
            logger,
            "{} must be less diffusive than {} (reference: fine {}, which favours "
            "{})".format(better, worse, ref_solver, ref_solver),
        )
        for name, _, _ in profile_fields:
            ratio = l1_better[name] / l1_worse[name]
            c.record(
                "L1 ratio {}/{}: {}".format(better, worse, name),
                ratio,
                ratio < 1.0,
                "< 1  ({} {:.6e} vs {} {:.6e})".format(
                    better, l1_better[name], worse, l1_worse[name]
                ),
            )
        analyze_status &= c.ok

    # --- Rank invariance -------------------------------------------------------------------
    serial = profiles["mhd_hlld"]
    parallel = _profile(_tag("mhd_hlld", "_r4"))
    c = Checks(logger, "rank invariance, 1 vs 4 ranks with shocks")
    bitwise = True
    for name, _, _ in profile_fields:
        scale = np.ptp(serial[name])
        diff = np.abs(serial[name] - parallel[name])
        bitwise &= bool((diff == 0.0).all())
        c.le("rel diff {}".format(name), diff.max() / scale, tol_rank)
    logger.debug(
        "  1 vs 4 ranks was {}bitwise identical".format("" if bitwise else "NOT ")
    )
    analyze_status &= c.ok

    return bool(analyze_status)
