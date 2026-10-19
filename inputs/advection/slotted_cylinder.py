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

import numpy as np
import riot

# Set False for one-period diagonal translation or True for one-period
# solid-body rotation about the origin.
solid_body_advection = True


def make_input():
    riot.input("riot", problem="region_pgen")

    riot.input("parthenon/job", problem_id="slotted_cylinder")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.velocity",
            "c.c.bulk.pressure",
            "c.c.mat.rho",
            "c.c.mat.volume_fraction",
        ],
        file_type="hdf5",
        dt=0.1,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        tlim=1.0,
        integrator="rk2",
        ncycle_out=50,
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        nx1=128,
        x1min=-0.5,
        x1max=0.5,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=128,
        x2min=-0.5,
        x2max=0.5,
        ix2_bc="outflow",
        ox2_bc="outflow",
        nx3=1,
        x3min=-1.0e-6,
        x3max=1.0e-6,
        ix3_bc="outflow",
        ox3_bc="outflow",
    )

    riot.input("parthenon/meshblock", nx1=32, nx2=32, nx3=1)

    riot.input("materials", sparse_dealloc=False)

    for material in ("material0", "material1"):
        riot.input(material, eos_type="IdealGas", Gamma=1.4, Cv=1.0e-3)

    riot.input(
        "regions",
        nlev_min=0,
        nlev_max=0,  # Required to maintain uniform pressure
    )

    common_state = {
        "c_m_rho": 1.0,
        "c_m_pressure": 1.0,
    }

    riot.input(
        "region0",
        name="slotted_cylinder",
        mask_type="background",
        matid=0,
        **common_state,
    )
    riot.input(
        "region1",
        name="slotted_cylinder",
        mask_type="python",
        matid=1,
        **common_state,
    )

    riot.input(
        "slotted_cylinder/params",
        advection="solid_body" if solid_body_advection else "diagonal",
        radius=0.15,
        slot_half_width=0.025,
        x_center=0.0,
        y_center=0.25,
        slot_opening_y=0.5,
        period=1.0,
    )

    riot.input("physics", hydro=True, thinc=False, kinematic_advection=True)
    riot.input("hydro", recon="plm", riemann="lhllc", cfl=0.4)


class slotted_cylinder:
    """Slot opens below the disk and points at the origin."""

    def __init__(self):
        pass

    def mask(self, pos):
        x = pos[:, self.x]
        y = pos[:, self.y]
        dx = x - self.x_center
        dy = y - self.y_center

        inside_disk = dx * dx + dy * dy <= self.radius * self.radius
        inside_slot = (
            (np.abs(dx) <= self.slot_half_width)
            & (y >= self.y_center)
            & (y <= self.slot_opening_y)
        )
        return inside_disk & ~inside_slot

    def c_c_bulk_velocity(self, pos, velocity):
        if self.advection == "diagonal":
            velocity[:, self.x] = 1.0 / self.period
            velocity[:, self.y] = 1.0 / self.period
        elif self.advection == "solid_body":
            omega = 2.0 * np.pi / self.period
            velocity[:, self.x] = -omega * pos[:, self.y]
            velocity[:, self.y] = omega * pos[:, self.x]
        else:
            raise ValueError("advection must be 'diagonal' or 'solid_body'")
        velocity[:, self.z] = 0.0


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
