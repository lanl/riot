#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

import riot

import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path


class LetterShape:
    def __init__(self, letter, font="Avenir Next", size=1.0, normalize=False):
        """
        Create a geometric representation of a letter.

        If normalize=True, the letter's bounding box is mapped to
        [0, 1] x [0, 1].
        """
        prop = FontProperties(family=font)

        text_path = TextPath((0.0, 0.15), letter, size=size, prop=prop)

        # Convert compound glyph into separate closed contours.
        polygons = text_path.to_polygons()

        if normalize:
            bbox = text_path.get_extents()

            xmin, ymin = bbox.xmin, bbox.ymin
            width = bbox.width
            height = bbox.height

            polygons = [
                np.column_stack(((p[:, 0] - xmin) / width, (p[:, 1] - ymin) / height))
                for p in polygons
            ]

        self.paths = [Path(p, closed=True) for p in polygons]

    def contains(self, x, y):
        """
        Test whether point(s) (x, y) are inside the filled portion
        of the letter.

        x and y may be scalars or NumPy arrays.
        """
        x, y = np.broadcast_arrays(x, y)
        points = np.column_stack((x.ravel(), y.ravel()))

        # Even-odd fill rule:
        # crossing each contour toggles inside/outside.
        inside = np.zeros(len(points), dtype=bool)

        for path in self.paths:
            inside ^= path.contains_points(points)

        inside = inside.reshape(x.shape)

        if inside.ndim == 0:
            return bool(inside)

        return inside


def make_input():

    riot.input(
        "riot",
        problem="region_pgen",  # name of the pgen
    )

    riot.input(
        "parthenon/job",
        problem_id="logo",  # problem ID: basename of output filenames
    )

    riot.input(
        "parthenon/output1",
        variables=["c.c.bulk.rho", "c.c.bulk.velocity", "c.c.bulk.pressure"],
        file_type="hdf5",  # Tabular data dump
        dt=0.1,  # time increment between outputs
    )

    riot.input(
        "parthenon/time",
        nlim=-1,  # cycle limit
        tlim=0.5,  # time limit
        integrator="rk2",  # time integration algorithm
        ncycle_out=1,  # interval for stdout summary info
    )

    riot.input(
        "parthenon/mesh",
        refinement="none",
        numlevel=3,
        nghost=2,
        derefine_count=10,
        nx1=4096,  # Number of zones in X1-direction
        x1min=0.0,  # minimum value of X1
        x1max=4.0,  # maximum value of X1
        ix1_bc="outflow",  # Inner-X1 boundary condition flag
        ox1_bc="outflow",  # Outer-X1 boundary condition flag
        nx2=1024,  # Number of zones in X2-direction
        x2min=0.0,  # minimum value of X2
        x2max=1.0,  # maximum value of X2
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
        nx1=512,  # Number of cells in each MeshBlock, X1-dir
        nx2=512,  # Number of cells in each MeshBlock, X2-dir
        nx3=1,  # Number of cells in each MeshBlock, X3-dir
    )

    riot.input(
        "materials",
        sparse_init=True,
        sparse_dealloc=True,
    )

    riot.input(
        "material0",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
        max_bnd_level=0,
        max_mat_level=0,
    )

    riot.input(
        "material1",
        eos_type="IdealGas",
        Gamma=1.5,
        Cv=1.0e-3,
        max_bnd_level=-1,
        max_mat_level=0,
    )

    riot.input(
        "region0",
        mask_type="background",
        matid=0,
        c_m_rho=0.001,
        c_m_pressure=1.0,
        c_c_bulk_velocity=[2.5, 0, 0],
    )

    riot.input(
        "region1",
        name="logo",
        mask_type="python",
        matid=1,
        c_m_rho=1.0,
        c_m_pressure=1.0,
    )

    riot.input(
        "physics",
        hydro=True,
    )

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.9,
        # riemann = hllc,
        amr_interface=True,
    )


class logo:
    def __init__(self):
        self.word = LetterShape("RIOT")

    def mask(self, pos):
        return self.word.contains(pos[:, self.x], pos[:, self.y])


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
