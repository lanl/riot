#!/usr/bin/env python
# ========================================================================================
#  (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

import numpy as np
from scipy import integrate


def get_err(a, b, rel=False):
    out = 2.0 * np.abs(b - a)
    if rel:
        out /= np.abs(b + a) + 1e-20
    return out


def get_infty_norm_err(a, b, x, rel=False):
    return np.max(get_err(a, b, rel))


def get_l1_err(a, b, x, rel=False):
    err = np.abs(get_err(a, b, rel))
    dx = x[1] - x[0]
    return np.sum(err) / len(err)
    try:
        return integrate.romb(err, dx=dx)
    except:
        return integrate.simpson(err, dx=dx)


def get_l2_err(a, b, x, rel=False):
    err = get_err(a, b, rel)
    err2 = err * err
    dx = x[1] - x[0]
    # return np.sum(err2)/len(err2)
    try:
        return np.sqrt(integrate.romb(err2, dx=dx))
    except:
        return np.sqrt(integrate.simpson(err2, dx=dx))


def compare_files(name_a, name_b, field, norm="infty", rel=False, component=None):
    if norm == "infty":
        nfunc = get_infty_norm_err
    elif norm == "l1":
        nfunc = get_l1_err
    else:
        nfunc = get_l2_err
    import h5py

    with h5py.File(name_a, "r") as fa:
        with h5py.File(name_b, "r") as fb:
            x = fa["/VolumeLocations/x"][0]
            va = fa[field][:]
            vb = fb[field][:]
            if component is not None:
                va = va[component]
                vb = vb[component]
            va = va.reshape(x.shape[0])
            vb = vb.reshape(x.shape[0])
            return nfunc(va, vb, x, rel)
