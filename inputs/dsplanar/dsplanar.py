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


import argparse
from pathlib import Path
import numpy as np
import riot

ifile_dir = Path(__file__).resolve().parent


def make_input():
    parser = argparse.ArgumentParser()

    ### INPUTS
    def amprange(amp):
        amp = float(amp)
        if amp < 0.0 or amp > 0.00025:
            raise argparse.ArgumentTypeError("amp must be in range [0.0, 0.00025]")
        return amp

    def wavelengthrange(wavelength):
        wavelength = float(wavelength)
        if wavelength < 0.0 or wavelength > 0.0495:
            raise argparse.ArgumentTypeError(
                "wavelength must be in range [0.0, 0.0495]"
            )
        return wavelength

    def temprange(temp):
        temp = float(temp)
        if temp < 100:
            raise argparse.ArgumentTypeError("temp must be >= 100")
        return temp

    def betarange(beta):
        beta = float(beta)
        if beta > 1024.0:
            raise argparse.ArgumentTypeError("beta must be <= 1024")
        return beta

    riot_inputs = parser.add_argument_group("riot inputs")
    riot_inputs.add_argument(
        "--amp",
        required=True,
        type=amprange,
        help="Select perturbation amplitude in range (0, 0.00025)",
    )

    riot_inputs.add_argument(
        "--wavelength",
        required=True,
        type=wavelengthrange,
        help="Select perturbation wavelength in range (0, 0.0495)",
    )

    riot_inputs.add_argument(
        "--wavelength_z",
        required=False,
        type=wavelengthrange,
        default=0.0,
        help="Select z-direction perturbation wavelength in range (0, 0.0495)",
    )

    riot_inputs.add_argument(
        "--beta",
        required=True,
        type=betarange,
        help="Select gradient layer beta <= 1024",
    )

    riot_inputs.add_argument(
        "--temp",
        required=False,
        default=298.0,
        type=temprange,
        help="Initial temperature of high-z layer",
    )

    ### FIDELITIES
    riot_fidelities = parser.add_argument_group("riot fidelities")
    riot_fidelities.add_argument(
        "--dim",
        required=True,
        type=int,
        help="Select dimensionality 1, 2, or 3",
        choices=[1, 2, 3],
    )
    riot_fidelities.add_argument(
        "--nlevels",
        required=False,
        default=1,
        type=int,
        help="Select number of refinement levels",
    )
    riot_fidelities.add_argument(
        "--ionization",
        required=False,
        default=False,
        type=bool,
        help="Turn on ionization physics",
    )

    # Check for argument compatibility
    args = parser.parse_args()
    if args.dim == 1 and args.amp > 0.0:
        parser.error("1D requires amp=0!")

    riot.input(
        "riot",
        problem="region_pgen",
        sparse_init=True,
        sparse_dealloc=True,
        verbose=False,
    )

    riot.input("parthenon/job", problem_id="dsplanar")

    riot.input("diagnostics", packages=["dsplanar"])

    riot.input("parthenon/output0", file_type="hst", dt=0.01e-9)

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.rho",
            "c.c.bulk.momentum",
            "c.c.bulk.velocity",
            "c.c.mat.rho",
            "c.c.bulk.pressure",
            # "c.c.mat.volume_fraction",
            "c.c.bulk.temperature",
            "c.c.bulk.electron_temperature",
            "c.c.bulk.electron_number_density",
            "c.m.ionization_zbar",
            "c.m.sie",
        ],
        file_type="hdf5",
        dt=0.1e-9,
        sparse_seed_nans=True,
    )

    riot.input("parthenon/time", nlim=-1, tlim=15.0e-9, integrator="rk2", ncycle_out=1)

    riot.input(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=args.nlevels,
        derefine_count=10,
        nx1=128,
        x1min=-0.03,
        x1max=0.0724,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=192 if args.dim > 1 else 1,
        x2min=-0.0768,
        x2max=0.0768,
        ix2_bc="outflow",
        ox2_bc="outflow",
        nx3=192 if args.dim > 2 else 1,
        x3min=-0.0768,
        x3max=0.0768,
        ix3_bc="outflow",
        ox3_bc="outflow",
        multigrid=True,
    )

    riot.input(
        "parthenon/meshblock",
        nx1=32,
        nx2=32 if args.dim > 1 else 1,
        nx3=32 if args.dim > 1 else 1,
    )

    riot.material(
        0,
        name="Helium",
        rho=1.6e-4,
        eos_type="IdealGas",
        Gamma=5.0 / 3.0,
        Cv=3.1e7,
        max_level=0,
    )
    riot.material(
        1,
        name="Aluminum",
        rho=2.7,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=3720,
        max_level=3,
    )
    riot.material(
        2,
        name="CH",
        rho=0.035,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=7592,
        max_level=3,
    )
    riot.material(
        3,
        name="Beryllium",
        rho=1.845,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=2024,
        max_level=9,
    )
    riot.material(
        4,
        name="Tungsten",
        rho=19.2371,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=3541,
        max_level=9,
    )
    riot.material(
        5,
        name="Gold",
        rho=19.3,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=2705,
        max_level=3,
    )
    riot.material(
        6,
        name="CHCompress",
        rho=0.2,
        eos_type="SpinerEOSDependsRhoT",
        sesame_id=7592,
        max_level=3,
    )

    for id in range(7):
        name = riot.material[id]["name"]
        eos_name = name + "EOS"
        riot.input(
            "material" + str(id),
            name=riot.material[id]["name"],
            eos=eos_name,
            electron_eos=eos_name,
            max_bnd_level=riot.material[id]["max_level"],
            max_mat_level=riot.material[id]["max_level"],
        )
        mat_eos = riot.material[id]["eos_type"]
        if mat_eos == "IdealGas":
            riot.input(
                eos_name,
                eos_type=mat_eos,
                zsplit=True,
                Gamma=riot.material[id]["Gamma"],
                Cv=riot.material[id]["Cv"],
            )
        else:
            riot.input(
                eos_name,
                eos_type=mat_eos,
                zsplit=True,
                sesame_id=riot.material[id]["sesame_id"],
                filename=str(ifile_dir / "eos" / "materials.sp5"),
            )

    riot.input("materials", sparse_init=True, sparse_dealloc=True)

    riot.input(
        "regions",
        nlev_max=5,
        # c_m_pressure 1e6
    )

    standard_pressure = 1.0e6

    riot.input(
        "region0",
        mask_type="background",
        matid=riot.material["Helium"]["id"],
        c_m_rho=riot.material["Helium"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region1",
        name="ambient_ablator",
        mask_type="inside_cylinder",
        matid=riot.material["Aluminum"]["id"],
        x0=0.0,
        x1=0.005,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.055,
        c_m_rho=riot.material["Aluminum"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region2",
        name="hot_ablator",
        mask_type="inside_cylinder",
        matid=riot.material["Aluminum"]["id"],
        x0=0.0,
        x1=0.0025,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.045,
        c_m_rho=riot.material["Aluminum"]["rho"],
        c_m_temperature=2.5e6,
    )

    riot.input(
        "region3",
        name="low_density_foam",
        mask_type="inside_cylinder",
        matid=riot.material["CH"]["id"],
        x0=0.005,
        x1=0.017,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.05,
        c_m_rho=riot.material["CH"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region4",
        name="be_layer",
        mask_type="python",
        matid=riot.material["Beryllium"]["id"],
        x0=riot.input["low_density_foam"]["x1"],
        x1=riot.input["low_density_foam"]["x1"] + args.amp + 2.5e-4,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.0495,
        c_m_rho=riot.material["Beryllium"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region5",
        name="gold_foil",
        mask_type="python",
        matid=riot.material["Gold"]["id"],
        x0=riot.input["low_density_foam"]["x1"],
        c_m_rho=riot.material["Gold"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region6",
        name="gradient_layer",
        mask_type="inside_cylinder",
        matid=[riot.material["Beryllium"]["id"], riot.material["Tungsten"]["id"]],
        x0=riot.input["be_layer"]["x1"],
        x1=riot.input["be_layer"]["x1"] + 3.0e-3,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.0495,
        c_m_rho_Beryllium=riot.material["Beryllium"]["rho"],
        c_m_rho_Tungsten=riot.material["Tungsten"]["rho"],
        c_m_pressure_Beryllium=standard_pressure,
        c_m_pressure_Tungsten=standard_pressure,
    )

    eos = riot.EOS("Tungsten")
    riot.log(
        f"Initializing Tungsten with P = {eos.PressureFromDensityTemperature(riot.material["Tungsten"]["rho"], args.temp)}"
    )
    riot.log(f"Another thing I want to take not of...")

    riot.input(
        "region7",
        name="hi_z_layer",
        mask_type="inside_cylinder",
        matid=riot.material["Tungsten"]["id"],
        x0=riot.input["gradient_layer"]["x1"],
        x1=riot.input["gradient_layer"]["x1"] + 5.0e-4,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.0495,
        c_m_rho=riot.material["Tungsten"]["rho"],
        c_m_temperature=args.temp,
    )

    riot.input(
        "region8",
        name="medium_density_foam",
        mask_type="inside_cylinder",
        matid=riot.material["CHCompress"]["id"],
        x0=riot.input["hi_z_layer"]["x1"],
        x1=0.07,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        radius=0.05,
        c_m_rho=riot.material["CHCompress"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region9",
        name="can",
        mask_type="inside_cylindrical_shell",
        matid=riot.material["CH"]["id"],
        x0=riot.input["ambient_ablator"]["x1"],
        x1=0.07,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        inner_radius=0.05,
        outer_radius=0.055,
        c_m_rho=1.044,
        c_m_pressure=standard_pressure,
    )

    riot.input(
        "region10",
        name="hohlraum",
        mask_type="inside_cylindrical_shell",
        matid=riot.material["Gold"]["id"],
        x0=-0.003,
        x1=0.0,
        y0=0.0,
        y1=0.0,
        z0=0.0,
        z1=0.0,
        inner_radius=0.045,
        outer_radius=0.08,
        c_m_rho=riot.material["Gold"]["rho"],
        c_m_pressure=standard_pressure,
    )

    riot.input("physics", hydro=True, sparse_physics=False, ionization=args.ionization)

    riot.input(
        "hydro",
        recon="plm",
        cfl=0.8,
        riemann="lhllc",
        amr_interface=True,
        mass_frac_thresh=1.0e-8,
    )

    riot.input(
        "gradient_layer/params",
        beta=args.beta,
        grad_bot=riot.input["gradient_layer"]["x0"],
        grad_top=riot.input["gradient_layer"]["x1"],
    )

    riot.input(
        "gold_foil/params", x0=riot.input["low_density_foam"]["x1"], dim=args.dim
    )

    riot.input(
        "ionization",
        root_tol=1e-20,
        fully_ionized=False,
        # ei coupling
        electron_ion_coupling=True,
        electron_ion_coupling_model="landau_spitzer",
        coulomb_logarithm="brysk",
        advect_electron_entropy=False,
        # conduction
        electron_thermal_conduction=True,
        electron_conductivity_model="spitzer_volume_average_arithmetic",
        ion_thermal_conduction=True,
        ion_conductivity_model="braginskii",
    )

    riot.input(
        "ionization/linear_solver_params",
        residual_tolerance=1.0e-3,
        max_coarsenings=10000,
        precondition=True,
        max_iterations=200,
        smoother="SRJ2",
        print_per_step=False,
    )

    riot.input(
        "be_layer/params",
        be_bot=riot.input["low_density_foam"]["x1"],
        be_top=riot.input["gradient_layer"]["x0"],
        amp=args.amp,
        wavelength=args.wavelength,
        wavelength_z=args.wavelength_z,
        dim=args.dim,
    )


### Beginning of class definitions for regions


class gradient_layer:
    def __init__(self):
        pass

    def heaviside(self, x):
        if self.beta <= 0.0:
            return -0.5 * self.beta + x * (1.0 + self.beta)
        lx = 1.0 - 2.0 * x
        lx[x > 0.5] = -lx[x > 0.5]
        h = 0.5 * (np.exp(-self.beta * lx) - lx * np.exp(-self.beta))
        h[x > 0.5] = 1.0 - h[x > 0.5]
        h[x < 0] = 0.0
        h[x > 1.0] = 1.0
        return h

    # Tungsten volume fraction
    def c_c_mat_volume_fraction_Tungsten(self, pos, alpha):
        xh = (pos[:, self.x] - self.grad_bot) / (self.grad_top - self.grad_bot)
        alpha[:] = self.heaviside(xh)

    # Beryllium volume fraction
    def c_c_mat_volume_fraction_Beryllium(self, pos, alpha):
        self.c_c_mat_volume_fraction_Tungsten(pos, alpha)
        alpha[:] = 1.0 - alpha[:]


class gold_foil:
    def __init__(self):
        self.half_width = 0.025
        self.radius = 0.0495
        self.thickness = 0.003

    def mask(self, pos):
        if self.dim == 1:
            return np.zeros(pos.shape[0], dtype=bool)
        elif self.dim == 2:
            return np.zeros(pos.shape[0], dtype=bool)
        else:
            rcsq = np.power(pos[:, self.y], 2) + np.power(pos[:, self.z], 2)
            return (
                (np.abs(pos[:, self.z]) > self.half_width)
                * (rcsq < self.radius * self.radius)
                * (pos[:, self.x] > self.x0)
                * (pos[:, self.x] < self.x0 + self.thickness)
            )


class be_layer:
    def __init__(self):
        self.radius = 0.0495

    def mask(self, pos):
        ll = self.wavelength if self.wavelength > 0.0 else 1.0e-12
        mm = self.wavelength_z if self.wavelength_z > 0.0 else 1.0e-12
        if self.dim == 1:
            return (pos[:, self.x] > self.be_bot) * (pos[:, self.x] < self.be_top)
        elif self.dim == 2:
            pert_x = (
                self.amp * (np.cos(2.0 * np.pi * pos[:, self.y] / ll) + 1.0)
                + self.be_bot
            )
            rcsq = np.power(pos[:, self.y], 2)
            return (
                (pos[:, self.x] > pert_x)
                * (pos[:, self.x] < self.be_top)
                * (rcsq < self.radius * self.radius)
            )
        else:
            pert_x = (
                self.amp
                * (
                    np.cos(2.0 * np.pi * pos[:, self.y] / ll)
                    * np.cos(2.0 * np.pi * pos[:, self.z] / mm)
                    + 1.0
                )
                + self.be_bot
            )
            rcsq = np.power(pos[:, self.y], 2) + np.power(pos[:, self.z], 2)
            return (
                (pos[:, self.x] > pert_x)
                * (pos[:, self.x] < self.be_top)
                * (rcsq[:] < self.radius * self.radius)
            )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
