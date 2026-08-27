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

import sys
import numpy as np
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc

from argparse import ArgumentParser

NON_FIELDS = set(
    [
        "Blocks",
        "Info",
        "Input",
        "Levels",
        "Locations",
        "LogicalLocations",
        "Params",
        "SparseInfo",
        "VolumeLocations",
    ]
)

parser = ArgumentParser(
    prog="snap1d",
    description="1D snapshot some number of fields",
)
parser.add_argument("-p", "--prefix", type=str, default="", help="Prefix for save name")
parser.add_argument("-f", "--fields", type=str, nargs="*", help="Which fields to plot")
parser.add_argument("files", type=str, nargs="+", help="Files to plot")
parser.add_argument(
    "-c", "--component", default=None, type=int, help="Components of fields to plot"
)


def plot_frame(f, x, fields, available_fields, prefix, i, component=None):
    avail_str = "Available fields are: {}".format(available_fields)
    nb = x.shape[0]
    nx = x.shape[1]
    if fields is None or len(fields) == 0:
        print("No field requested.\n" + avail_str)
        sys.exit()
    for varname in fields:
        if varname not in available_fields:
            raise ValueError("Field {} not available. ".format(varname) + avail_str)
        for b in range(nb):
            fdset = f[varname][b]
            numf = np.prod(fdset.shape)
            if component is not None and numf != nx:
                var = fdset[component, 0, 0, :].reshape(nx)
            else:
                var = fdset.reshape(nx)
            plt.plot(x[b], var, label=varname)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("field")
    savename = "{}{:04d}.png".format(prefix, i)
    plt.savefig(savename, bbox_inches="tight")
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    args = parser.parse_args()
    for i, fname in enumerate(args.files):
        with h5py.File(fname, "r") as f:
            available_fields = set(f.keys()) - NON_FIELDS
            x = f["/VolumeLocations/x"]
            print("Plotting frame", i)
            plot_frame(
                f, x, args.fields, available_fields, args.prefix, i, args.component
            )
