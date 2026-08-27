# ========================================================================================
#  (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

# Verification test for the nonlinear (equilibrium-diffusion) Marshak wave.
#
# This problem drives a constant-temperature boundary into a cold medium with a T^-3
# opacity. With fixed_fluid and single-group equilibrium diffusion it is the classic
# Zel'dovich-Raizer nonlinear thermal-conduction wave, whose solution is self-similar: the
# front advances as x_f ~ sqrt(t) and the temperature profile collapses onto a single
# curve in the similarity variable xi = x / sqrt(t).
#
# We verify that self-similarity directly. We check
#   - profile collapse across output times catches shape/physics breakage;
#   - the front rate x_f/sqrt(t) pins the absolute propagation speed

# Modules
import glob
import logging
import numpy as np
import scripts.utils.riot as riot

logger = logging.getLogger("riot" + __name__[7:])  # set logger name

file_id = "marshak"
amr = True
sparse = True
nranks = 1

var = "c.c.bulk.temperature"

# Max allowed relative spread of T(xi) across output times (self-similarity). Measured
# collapse is ~0.1% in the bulk and ~0.4% at the steep front; 1% leaves clear margin.
collapse_tol = 1.0e-2

# Analytic front rate x_f/sqrt(t)
a_radiation = 7.5657e-15  # radiation constant [erg cm^-3 K^-4], CGS
c_light = 2.99792458e10  # speed of light [cm/s], CGS
rho = 3.0  # wave-material density [g/cm^3]
Cv = 8.61733e10  # wave-material specific heat [erg/g/K] (air1)
kappa0 = 1.56272e22  # opacity scale [cm^2/g] (kappa0_a)
Tb = 1.16045e7  # boundary temperature [K]
z0 = 0.59863  # self-similar leading-front eigenvalue (hardcoded)
_D0 = 4.0 * a_radiation * c_light / (3.0 * rho**2 * Cv * kappa0)

# Expected leading-front rate x_f/sqrt(t) = z0 * sqrt(D0 * Tb^n).
front_rate_expected = z0 * (_D0 * Tb**6) ** 0.5

# Fraction of the temperature range defining the leading front for the measurement
# (matches the analytic f->0 front; 1% is on the steep edge but robust across dumps).
front_threshold_frac = 0.01

# Allowed fractional deviation. The discretized front lags the analytic value by ~1%
# (finite-cell smearing + startup transient); 5% leaves margin while still catching a
# wrong-magnitude (e.g. 2x diffusivity -> sqrt(2) rate) regression.
front_rate_tol = 0.05


def self_similar_collapse(
    dumps, var, similarity_exp=0.5, front_frac=0.01, xi_samples=None
):
    """Verify 1D Marshak similarity under variable xi = x / t**similarity_exp.

    For a self-similar solution (the nonlinear thermal-conduction / Marshak wave, whose
    front advances as x_f ~ sqrt(t)), the profile T(x, t) plotted against xi = x/sqrt(t)
    is time-independent. This is a machine-independent physics check: it compares a run to
    itself across output times, so it is immune to cross-architecture last-bit differences
    and to the sub-cell front-position wiggle that makes a pointwise gold comparison
    fragile.

    Parameters
    ----------
    dumps : list of phdf.phdf
        Output dumps at increasing times (t > 0).
    var : str
        Field name to test (e.g. "c.c.bulk.temperature").
    similarity_exp : float
        Exponent p in xi = x / t**p (0.5 for a diffusive sqrt(t) front).
    front_frac : float
        Fraction of the (min..max) range defining the leading front: the front is the
        outermost cell with value > vmin + front_frac*(vmax - vmin). This matches the
        analytic f->0 leading front.
    xi_samples : array_like or None
        Fractions of the front similarity coordinate at which to sample and compare the
        profile. If None, a default spanning the interior/front is used.

    Returns
    -------
    (max_rel_spread, front_rate) : (float, float)
        max_rel_spread : the largest (over xi_samples) relative peak-to-peak spread of the
            sampled value across the provided dumps -- 0 for a perfectly self-similar run.
        front_rate : x_f / sqrt(t) averaged over dumps, where x_f is the leading front;
            a physical invariant that pins the absolute propagation rate.
    """
    if xi_samples is None:
        xi_samples = np.linspace(
            0.5, 0.95, 8
        )  # fractions of the front similarity coord

    profiles = []  # (xi array, value array) per dump
    front_rates = []
    for d in dumps:
        t = d.Time
        x = np.asarray(d.x).flatten()
        v = np.asarray(d.Get(var)).flatten()
        order = np.argsort(x)
        x = x[order]
        v = v[order]
        xi = x / t**similarity_exp
        profiles.append((xi, v))
        # Leading front: outermost cell above a small fraction of the range.
        vthresh = v.min() + front_frac * (v.max() - v.min())
        above = np.where(v > vthresh)[0]
        if len(above) > 0:
            front_rates.append(xi[above[-1]])

    front_rate = float(np.mean(front_rates)) if front_rates else float("nan")

    # Sample each profile at absolute similarity coordinates (fractions of front rate),
    # then measure how much the sampled value varies across dumps.
    abs_xi = np.asarray(xi_samples) * front_rate
    max_rel_spread = 0.0
    for q in abs_xi:
        vals = np.array([np.interp(q, xi, v) for (xi, v) in profiles])
        mean = np.mean(np.abs(vals))
        if mean > 0:
            spread = (vals.max() - vals.min()) / mean
            max_rel_spread = max(max_rel_spread, spread)
    return max_rel_spread, front_rate


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate("radiation_diffusion/marshak.py")
    logger.debug("Runnning test " + __name__)
    args = [
        "parthenon/job/problem_id=" + file_id,
        "parthenon/mesh/refinement=" + ("static" if amr else "none"),
        "parthenon/mesh/sparse_init=" + str(sparse).lower(),
        "materials/sparse_init=" + str(sparse).lower(),
        "materials/sparse_dealloc=" + str(sparse).lower(),
    ]
    # Drive the simulation with the SAME physical constants used to compute the analytic
    # front rate above, so the two can never silently diverge (e.g. someone edits
    # marshak.rin's kappa0 without updating front_rate_expected). rho, Cv, and kappa0 are
    # the wave material (air1, matid 1); Tb is the hot-boundary temperature, which appears
    # both as the driven region's temperature and as the first entry of the diffusion
    # boundary_T vector (the cold background/other faces are left at the input defaults).
    T_background = 1.0e-5 * Tb
    args += [
        "diffusion/a_radiation=" + repr(a_radiation),
        "diffusion/c_light=" + repr(c_light),
        "material1/Cv=" + repr(Cv),
        "material1/kappa0_a=" + repr(kappa0),
        "region1/c_m_rho=" + repr(rho),
        "region0/c_m_rho=" + repr(rho),
        "region0/c_m_temperature=" + repr(Tb),
        "diffusion/boundary_T="
        + ",".join(repr(v) for v in (Tb, T_background, 1.0, 1.0, 1.0, 1.0)),
    ]
    riot.mpirun(nranks, "radiation_diffusion/marshak.rin", args)


# Analyze outputs
def analyze():
    import phdf

    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    # Intermediate time dumps (out1.00000, 00001, ...). Skip the first two: t=0 (no
    # similarity variable) and the earliest step (startup transient not yet self-similar).
    dump_files = sorted(glob.glob("build/src/" + file_id + ".out1.000[0-9][0-9].phdf"))[
        2:
    ]
    if len(dump_files) < 3:
        logger.warning(f"marshak: expected several time dumps, found {len(dump_files)}")
        return False
    dumps = [phdf.phdf(f) for f in dump_files]

    max_spread, front_rate = self_similar_collapse(
        dumps, var, front_frac=front_threshold_frac
    )

    if max_spread > collapse_tol:
        logger.warning(
            f"marshak: profile not self-similar: max relative spread of T(x/sqrt(t)) = "
            f"{max_spread:.3e} > {collapse_tol:.1e}"
        )
        analyze_status = False

    rate_dev = abs(front_rate - front_rate_expected) / front_rate_expected
    if rate_dev > front_rate_tol:
        logger.warning(
            f"marshak: front rate x_f/sqrt(t) = {front_rate:.4g} deviates "
            f"{rate_dev:.2%} from expected {front_rate_expected:.4g} (tol "
            f"{front_rate_tol:.0%})"
        )
        analyze_status = False

    logger.debug(
        f"marshak: max collapse spread = {max_spread:.3e}, front rate = {front_rate:.4g}"
    )
    return analyze_status
