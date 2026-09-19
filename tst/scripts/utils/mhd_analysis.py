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

# Shared analysis helpers for the tst/scripts/mhd suite.
#
# This lives under scripts/utils rather than scripts/mhd on purpose: run_tests.py treats
# EVERY module it finds in a suite directory as a test and calls run()/analyze() on it, so a
# helper module placed next to the tests would be collected and fail. scripts/utils is
# excluded from collection.

import numpy as np
import h5py


def assemble(path, key, ncomp=None):
    """Map every mesh block onto the global (comp, k, j, i) grid via LogicalLocations.

    WHY THIS IS NECESSARY. A .phdf holds one array per mesh BLOCK, and the block layout
    depends on the meshblock size and the rank count. Two runs of the same physical problem
    with different decompositions have different array shapes and different block orderings,
    so they can only be compared after mapping each block back onto the global grid.
    Comparing file hashes instead is meaningless: HDF5 containers carry timestamps, version
    strings, and the embedded input deck.

    A side benefit used throughout this suite: the result is in global index order, so for a
    1D problem the returned array is already sorted in x and needs no coordinate lookup.

    Assumes a uniform grid with equal-sized blocks and no refinement, which is the MHD
    package's supported scope. LogicalLocations column order is (i, j, k).
    """
    with h5py.File(path, "r") as f:
        ll = f["LogicalLocations"][:]
        d = f[key][:]
        if ncomp is None:
            nb, nk, nj, ni = d.shape
            d = d[:, None]
            ncomp = 1
        else:
            nb, ncomp, nk, nj, ni = d.shape
        shape = (
            ncomp,
            nk * (ll[:, 2].max() + 1),
            nj * (ll[:, 1].max() + 1),
            ni * (ll[:, 0].max() + 1),
        )
        out = np.full(shape, np.nan)
        for b in range(nb):
            i0, j0, k0 = ll[b, 0] * ni, ll[b, 1] * nj, ll[b, 2] * nk
            out[:, k0 : k0 + nk, j0 : j0 + nj, i0 : i0 + ni] = d[b]
    # A NaN left anywhere means the blocks did not tile the domain. Failing loudly here is
    # the difference between a detected harness bug and a silently wrong comparison.
    assert not np.isnan(out).any(), "blocks did not tile the global grid for " + key
    return out


def restrict1d(a, factor):
    """Conservative volume-average restriction of a 1D profile by an integer factor.

    Used to bring a fine reference run onto a coarse grid. On a uniform mesh the cell average
    over a group of `factor` fine cells IS the coarse cell average, so this is exact rather
    than an interpolation, and it introduces no error of its own into a convergence
    measurement. Interpolating the fine run instead would add an O(dx^2) term of unknown sign
    to every comparison.
    """
    assert a.size % factor == 0, "fine resolution must be an integer multiple of coarse"
    return a.reshape(-1, factor).mean(axis=1)


class Checks:
    """Accumulator for pass/fail criteria with uniform reporting.

    Exists so that a test reports EVERY criterion it evaluated and its measured value, not
    just the first failure. A regression suite that stops at the first failure hides how much
    else moved, and a suite that logs only failures gives no way to see a number drifting
    toward its threshold before it crosses.
    """

    def __init__(self, logger, title):
        self.logger = logger
        self.ok = True
        self.logger.debug("--- {} ---".format(title))

    def record(self, name, value, passed, criterion):
        self.ok &= bool(passed)
        msg = "  {:<34s} {:<16.8e} {:<6s} {}".format(
            name, float(value), "PASS" if passed else "FAIL", criterion
        )
        if passed:
            self.logger.debug(msg)
        else:
            self.logger.warning(msg)
        return bool(passed)

    def le(self, name, value, tol):
        return self.record(name, value, value <= tol, "<= {:.3e}".format(tol))

    def gt(self, name, value, bound):
        return self.record(name, value, value > bound, "> {:.3e}".format(bound))

    def ge(self, name, value, bound):
        return self.record(name, value, value >= bound, ">= {:.3e}".format(bound))

    def close(self, name, value, target, rtol):
        passed = bool(np.isclose(value, target, rtol=rtol, atol=0.0))
        return self.record(
            name, value, passed, "rtol {:.0e} of {:.17g}".format(rtol, target)
        )

    def finite(self, name, *arrays):
        allfinite = all(np.all(np.isfinite(a)) for a in arrays)
        return self.record(name, float(allfinite), allfinite, "no NaN or Inf")


def field_snapshot(tag, snap="final", d="build/src/"):
    """Path of one output snapshot of a run, in the directory run_tests.py leaves them in."""
    return "{}{}.out1.{}.phdf".format(d, tag, snap)
