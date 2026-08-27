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

import sys

import os
import phdf
import numpy as np
import matplotlib.pyplot as plt
import glob

Lx = 4.0
Ly = 1.0


def read(name):
    return phdf.phdf(name)


def find_width(xg, v0, v1):
    i = 0
    vt = 0.5
    while v0[i] > vt and i < v0.shape[0] - 1:
        i += 1
    xright = np.sum(v0[i::] * xg[i::]) / np.sum(v0[i::])

    i = 0
    while v1[i] < vt and i < v1.shape[0] - 1:
        i += 1
    xleft = np.sum(v1[:i] * xg[:i]) / np.sum(v1[:i])

    return xright - xleft


def compute_averages(name):
    d = read(name)
    dx = d.xf[0, 1] - d.xf[0, 0]
    bsize = d.x[0, :].shape[0]
    dxblock = bsize * dx
    nxblock = round(Lx / dxblock)
    nyzblock = round(Ly / dxblock) ** 2
    nxtot = round(Lx / dx)
    av0 = np.zeros((nxtot), dtype=np.float64)
    av1 = np.zeros((nxtot), dtype=np.float64)
    xg = np.array([(i + 0.5) * dx for i in range(nxtot)])
    v0 = np.nan_to_num(d.Get("c.c.mat.volume_fraction_0", False))
    v1 = np.nan_to_num(d.Get("c.c.mat.volume_fraction_1", False))
    for b in range(v0.shape[0]):
        iblock = round(d.x[b][0] / dxblock)
        istart = iblock * bsize
        for i in range(istart, istart + bsize):
            av0[i] += np.sum(v0[b, 0, :, :, i - istart])
            av1[i] += np.sum(v1[b, 0, :, :, i - istart])
    av0 /= bsize * bsize * nyzblock
    av1 /= bsize * bsize * nyzblock
    return xg, av0, av1, find_width(xg, av0, av1), d.Time


def process_directory(name, with_profile_plots):
    root = os.getcwd()
    os.chdir(name)
    if os.path.isfile("width.txt"):
        print(name + ": using cached data")
        t = np.loadtxt("time.txt")
        w = np.loadtxt("width.txt")
        os.chdir(root)
        return t, w
    files = sorted(glob.glob("*.phdf"))
    nfiles = len(files)
    t = []
    w = []
    for idx, file in enumerate(files):
        print(
            name + ": ",
            str(idx + 1) + "/" + str(nfiles),
            "                         ",
            end="\r",
        )
        x, v0, v1, width, time = compute_averages(file)
        if with_profile_plots:
            fig, ax = plt.subplots()
            ax.plot(x, v0)
            ax.plot(x, v1)
            plt.tight_layout()
            plt.savefig(f"{idx:04}" + ".png", dpi=300)
            plt.close()
        t.append(time)
        w.append(width)
    print()
    t = np.array(t)
    w = np.array(w)
    np.savetxt("time.txt", t)
    np.savetxt("width.txt", w)
    os.chdir(root)
    return t, w


def main():
    tlo_half, wlo_half = process_directory("plm_half", True)
    tho_half, who_half = process_directory("ho_half", True)
    tlo, wlo = process_directory("plm_base", True)
    tho, who = process_directory("ho_base", False)
    fix, ax = plt.subplots()
    ax.plot(tlo_half, wlo_half, "r--", label="LO 1/2x")
    ax.plot(tho_half, who_half, "b--", label="HO 1/2x")
    ax.plot(tlo, wlo, "r-", label="LO 1x")
    ax.plot(tho, who, "b-", label="HO 1x")
    ax.set_xlabel("t")
    ax.set_ylabel("Mix width")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("mix_width.png", dpi=300)


if __name__ == "__main__":
    main()
