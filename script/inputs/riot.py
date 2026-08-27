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

import sys
import argparse
import atexit
import os, inspect, ast
from datetime import date
import singularity_eos
from math import pi

# The singularity_eos extension module carries its own Kokkos runtime.  Any
# array-based EOS call (e.g. PressureFromDensityInternalEnergy on numpy arrays,
# as used in region-init functions) dispatches to a Kokkos parallel_for and
# requires initialize() to have run first.  Bring the runtime up at import and
# tear it down at interpreter exit -- the latest point at which all
# Kokkos-dependent EOS work is guaranteed complete.  initialize() is idempotent
# and does not take ownership if Kokkos was already started elsewhere, so
# finalize() only shuts Kokkos down if singularity_eos actually started it.
singularity_eos.initialize()
atexit.register(singularity_eos.finalize)


def external_caller_file():
    me = __name__
    f = inspect.currentframe()
    try:
        f = f.f_back  # skip this function
        while f:
            mod = inspect.getmodule(f)
            if mod and mod.__name__ != me:
                return os.path.abspath(getattr(mod, "__file__", "")) or None
            f = f.f_back
    finally:
        del f


def defines_class_with_name(filepath, class_name):
    """
    Checks if a Python file defines a class with the given name without loading it as a module.

    Args:
        filepath (str): The path to the Python file.
        class_name (str): The name of the class to search for.

    Returns:
        bool: True if the class is found, False otherwise.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return False
    except SyntaxError as e:
        print(f"Error parsing file {filepath}: {e}")
        return False


def replace_extension_os(file_path, new_extension):
    """
    Provides the name of a file with it's extension replaced using the os module.

    Args:
        file_path (str): The path to the file.
        new_extension (str): The new extension (e.g., "txt", "exe").
    """
    base_name, old_extension = os.path.splitext(file_path)
    new_file_path = f"{base_name}.{new_extension}"
    return new_file_path


# this class is entirely static, and sort of behaves like a singleton
class material:
    mats = {}

    @staticmethod
    def __init__(material_id=None, **kwargs):
        if material_id is not None:
            if material_id in material.mats:
                material.mats.update(kwargs)
            else:
                material.mats[material_id] = {"id": material_id}
                material.mats[material_id].update(kwargs)

    def __class_getitem__(cls, key_want):
        if type(key_want) is int:
            return cls.mats[key_want]
        if type(key_want) is str:
            for v in cls.mats.values():
                if v["name"] == key_want:
                    return v


def replace_matnames_with_ids(d):
    for k, v in d.items():
        if (
            "c_m_rho_" in k
            or "c_m_sie_" in k
            or "c_m_temperature_" in k
            or "c_m_pressure_" in k
        ):
            suffix = k.rsplit("_", 1)[-1]
            for km in material.mats:
                if suffix == material.mats[km]["name"]:
                    res = d.pop(k)
                    id = materials[suffix]["id"]
                    new_name = k.rsplit("_", 1)[0] + "_" + str(id)
                    d[new_name] = res


# this class is entirely static, and sort of behaves like a singleton
class input:
    blocks = {}

    @staticmethod
    def __init__(block_name=None, **kwargs):
        if block_name is not None:
            # first check if block_name exists in a block as "name"
            for k, v in input.blocks.items():
                if "name" in v:
                    if v["name"] == block_name:
                        raise RuntimeError(
                            "Name of block and name parameter of another block collide.  Rename block."
                        )
            # replace_matnames_with_ids(kwargs)
            if block_name in input.blocks:
                input.blocks[block_name].update(kwargs)
            else:
                input.blocks[block_name] = kwargs
            if (
                "file" not in input.blocks[block_name]
                and "name" in input.blocks[block_name]
            ):
                origin = external_caller_file()
                if defines_class_with_name(origin, input.blocks[block_name]["name"]):
                    input.blocks[block_name]["file"] = origin

    def __class_getitem__(cls, key):
        if key in cls.blocks:
            return cls.blocks[key]
        else:
            for k, v in cls.blocks.items():
                if "name" in v:
                    if v["name"] == key:
                        return v
        return cls.blocks[key]

    @staticmethod
    def generate_input(*args):
        if len(args) == 0:
            name = replace_extension_os(external_caller_file(), "rin")
        elif len(args) == 1:
            name = args[0]
        else:
            raise RuntimeError(
                "Unexpected number of arguments provided to generate_input.  Should be 0 or 1."
            )
        with open(name, "w") as f:
            f.write(
                "# ========================================================================================\n"
            )
            f.write(
                f"# (C) (or copyright) {date.today().year}. Triad National Security, LLC. All rights reserved.\n"
            )
            f.write(
                "# This program was produced under U.S. Government contract 89233218CNA000001 for Los\n"
            )
            f.write(
                "# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC\n"
            )
            f.write(
                "# for the U.S. Department of Energy/National Nuclear Security Administration. All rights\n"
            )
            f.write(
                "# in the program are reserved by Triad National Security, LLC, and the U.S. Department\n"
            )
            f.write(
                "# of Energy/National Nuclear Security Administration. The Government is granted for\n"
            )
            f.write(
                "# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide\n"
            )
            f.write(
                "# license in this material to reproduce, prepare derivative works, distribute copies to\n"
            )
            f.write(
                "# the public, perform publicly and display publicly, and to permit others to do so.\n"
            )
            f.write(
                "# ========================================================================================\n\n"
            )
            f.write(f"# Input generated from {external_caller_file()}\n\n")
            f.write(input._input_format())
        print(name, end="")
        if len(log.record) > 0:
            name = replace_extension_os(external_caller_file(), "log")
            with open(name, "w") as f:
                f.write(log.record)

    @staticmethod
    def generate_cmd():
        print(input._cmd_format(), end="")

    @staticmethod
    def generate(input_file=""):
        if input_file != "":
            input.generate_input(input_file)
        else:
            input.generate_cmd()

    @staticmethod
    def _get_swarm_sample_fields(swarm_name):
        """Get sample_fields for a given swarm from tracers/{swarm_name} block.

        Args:
            swarm_name: Name of the swarm (e.g., "eulerian", "lagrangian")

        Returns:
            List of sample field names, or empty list if not found
        """
        tracer_block = f"tracers/{swarm_name}"
        if tracer_block in input.blocks:
            if "sample_fields" in input.blocks[tracer_block]:
                return input.blocks[tracer_block]["sample_fields"]
        return []

    @staticmethod
    def _process_auto_swarm_sample_fields():
        """Process output blocks with auto_swarm_sample_fields flag.

        When auto_swarm_sample_fields=True is set in an output block, this method
        automatically populates the swarm_variables list by:
        1. Finding all swarms listed in the output block
        2. Looking up sample_fields from each tracers/{swarm_name} block
        3. Transforming each field by prepending "particle.sample."
        4. Appending to swarm_variables (avoiding duplicates)

        This eliminates the need to manually duplicate sample field names between
        tracer configuration and output blocks.
        """
        for bkey, bval in input.blocks.items():
            # Check if this is an output block with auto_swarm_sample_fields
            if bkey.startswith("parthenon/output") and bval.get(
                "auto_swarm_sample_fields", False
            ):
                # Get the swarms list
                swarms = bval.get("swarms", [])
                if not swarms:
                    continue

                # Get existing swarm_variables or initialize empty list
                swarm_variables = bval.get("swarm_variables", [])
                if not isinstance(swarm_variables, list):
                    swarm_variables = [swarm_variables]
                else:
                    swarm_variables = swarm_variables.copy()

                # For each swarm, collect sample_fields and transform them
                for swarm_name in swarms:
                    sample_fields = input._get_swarm_sample_fields(swarm_name)
                    for field in sample_fields:
                        transformed = f"particle.sample.{field}"
                        if transformed not in swarm_variables:
                            swarm_variables.append(transformed)

                # Update the block with expanded swarm_variables
                input.blocks[bkey]["swarm_variables"] = swarm_variables

    @staticmethod
    def _input_format():
        # Process auto_swarm_sample_fields before generating output
        input._process_auto_swarm_sample_fields()

        s = ""
        for bkey, bval in input.blocks.items():
            s += f"<{bkey}>\n"
            for key, val in bval.items():
                # Skip Python-only parameters that should not appear in .rin files
                if key == "auto_swarm_sample_fields":
                    continue
                if type(val) == list:
                    vec = ""
                    for elem in val:
                        vec += f"{elem},"
                    vec = vec[:-1]
                    s += f"{key} = {vec}\n"
                else:
                    s += f"{key}={val}\n"
            s += "\n"
        return s

    @staticmethod
    def _cmd_format():
        # Process auto_swarm_sample_fields before generating output
        input._process_auto_swarm_sample_fields()

        s = ""
        for bkey, bval in input.blocks.items():
            for key, val in bval.items():
                # Skip Python-only parameters that should not appear in command format
                if key == "auto_swarm_sample_fields":
                    continue
                if type(val) == list:
                    vec = ""
                    for elem in val:
                        vec += f"{elem},"
                    vec = vec[:-1]
                    s += f"{bkey}/{key}={vec} "
                else:
                    s += f"{bkey}/{key}={val}"
                s += " "
        return s

    @staticmethod
    def __str__():
        return input._input_format()

    @staticmethod
    def make_eos(material_name):
        cls_name = input[material_name]["eos_type"]
        cls = getattr(singularity_eos, cls_name)
        my_args = {}
        # RIOT DOES NOT PROVIDE THE ACTUAL INPUTS TO INITIALIZE A GAMMA LAW!!
        my_args["Gamma"]
        return cls(*args, **kwargs)


class log:
    record = ""

    def __init__(self, s, end="\n"):
        log.record += s + end


class EOS:
    def __init__(self, material):
        eosblock = material
        if "eos" in input[material]:
            eosblock = input[material]["eos"]
        eos_type = input[eosblock]["eos_type"]
        p = input[eosblock]
        if eos_type == singularity_eos.IdealGas.EosType:
            eos_obj = singularity_eos.IdealGas(p["Gamma"] - 1, p["Cv"])
        elif eos_type == singularity_eos.Gruneisen.EosType:
            eos_obj = singularity_eos.Gruneisen(
                p["C0"],
                p["s1"],
                p["s2"],
                p["s3"],
                p["G0"],
                p["b"],
                p["rho0"],
                p["T0"],
                p["P0"],
                p["Cv"],
            )
        elif eos_type == singularity_eos.JWL.EosType:
            eos_obj = singularity_eos.JWL(
                p["A"], p["B"], p["R1"], p["R2"], p["w"], p["rho0"], p["Cv"]
            )
        elif eos_type == singularity_eos.DavisProducts.EosType:
            eos_obj = singularity_eos.DavisProducts(
                p["a"], p["b"], p["k"], p["n"], p["vc"], p["pc"], p["Cv"]
            )
        elif eos_type == singularity_eos.DavisReactants.EosType:
            eos_obj = singularity_eos.DavisReactants(
                p["rho0"],
                p["e0"],
                p["P0"],
                p["T0"],
                p["A"],
                p["B"],
                p["C"],
                p["G0"],
                p["Z"],
                p["alpha"],
                p["Cv0"],
            )
        elif eos_type == singularity_eos.SpinerEOSDependsRhoT.EosType:
            filename = p["filename"]
            reproducibility = (
                True
                if "reproducibility_mode" in p and p["reproducibility_mode"]
                else False
            )
            if "sesame_id" in p:
                eos_obj = singularity_eos.SpinerEOSDependsRhoT(
                    filename, p["sesame_id"], reproducibility
                )
            elif "sesame_name" in p:
                eos_obj = singularity_eos.SpinerEOSDependsRhoT(
                    filename, p["sesame_name"], reproducibility
                )
            else:
                print(f"Missing sesame_id or sesame_name in input block {eosblock}")
        elif eos_type == singularity_eos.SpinerEOSDependsRhoSie.EosType:
            filename = p["filename"]
            reproducibility = (
                True
                if "reproducibility_mode" in p and p["reproducibility_mode"]
                else False
            )
            if "sesame_id" in p:
                eos_obj = singularity_eos.SpinerEOSDependsRhoSie(
                    filename, p["sesame_id"], reproducibility
                )
            elif "sesame_name" in p:
                eos_obj = singularity_eos.SpinerEOSDependsRhoSie(
                    filename, p["sesame_name"], reproducibility
                )
            else:
                print(f"Missing sesame_id or sesame_name in input block {eosblock}")
        else:
            print(f"{eos_type} specified in input block {eosblock} not supported!")
        object.__setattr__(self, "eos", eos_obj)

    def __getattr__(self, name):
        return getattr(self.eos, name)

    def __setattr__(self, name, value):
        return setattr(self.eos, name, value)

    def __delattr__(self, name):
        return delattr(self.eos, name)

    def __repr__(self):
        return self.eos.EosType


class constants:
    def __init__(self, units="CGS"):
        if units == "CGS":
            self.length = 1.0e2
            self.mass = 1.0e3
            self.time = 1.0
            self.temperature = 1.0
            self.current = 1.0e-1
            self.charge = 2.997924580e9
            self.capacitance = 8.9831483395497e11
            self.angle = 1.0
        elif units == "SI":
            self.length = 1.0
            self.mass = 1.0
            self.time = 1.0
            self.temperature = 1.0
            self.current = 1.0
            self.charge = 1.0
            self.capacitance = 1.0
            self.angle = 1.0
        else:
            print(f"Invalid unit selection. {units} is not 'CGS' or 'SI'.")

        self.force = self.mass * self.length / (self.time * self.time)
        self.energy = self.force * self.length
        self.power = self.energy / self.time

        # Avogadro constant (CODATA 2010 value)
        self.avogadro = 6.02214129e23
        self.na = self.avogadro

        # Fine structure constant (CODATA 2010 value)
        self.fine_structure = 7.2973525698e-3
        self.alpha = self.fine_structure

        # Planck constant (CODATA 2010 value)
        self.planck = 6.62606957e-34 * self.energy * self.time
        self.h = self.planck

        # Reduced Planck constant
        self.reduced_planck = self.planck / (2.0 * pi)
        self.hbar = self.reduced_planck

        # Molar gas constant (CODATA 2010 value)
        self.gas_constant = 8.3144621 * self.energy / self.temperature
        self.r_gas = self.gas_constant

        # Boltzmann constant (CODATA 2010 value)
        self.boltzmann = 1.380648800e-23 * self.energy / self.temperature
        self.kb = self.boltzmann

        # Electron charge (CODATA 2018 exact value)
        self.electron_charge = 1.602176565e-19 * self.charge
        self.qe = self.electron_charge

        # Speed of light (CODATA 2018 exact value)
        self.speed_of_light = 2.99792458e8 * self.length / self.time
        self.c = self.speed_of_light

        # Gravitational constant (CODATA 2010 value)
        self.gravitational_constant = (
            6.67384e-11
            * self.length
            * self.length
            * self.length
            / (self.mass * self.time * self.time)
        )
        self.g_newt = self.gravitational_constant

        # Standard acceleration of gravity (CODATA 2010 value)
        self.acceleration_from_gravity = 9.80665 * self.length / (self.time * self.time)
        self.g_accel = self.acceleration_from_gravity

        # Electron rest mass (CODATA 2010 value)
        self.electron_mass = 9.10938291e-31 * self.mass
        self.me = self.electron_mass

        # Proton rest mass (CODATA 2010 value)
        self.proton_mass = 1.672621777e-27 * self.mass
        self.mp = self.proton_mass

        # Stefan-Boltzmann constant
        self.stefan_boltzmann = (
            2.0
            * pi
            * pi
            * pi
            * pi
            * pi
            * self.kb
            * self.kb
            * self.kb
            * self.kb
            / (15.0 * self.h * self.h * self.h * self.c * self.c)
        )
        self.sb = self.stefan_boltzmann

        # Radiation constant
        self.radiation_constant = 4.0 * self.stefan_boltzmann / self.speed_of_light
        self.ar = self.radiation_constant

        # Faraday constant
        self.faraday_constant = 96485.33645957 * self.capacitance
        self.faraday = self.faraday_constant

        # Permeability of free space
        self.vacuum_permeability = (
            4.0 * pi * 1.0e-7 * self.force / (self.current * self.current)
        )
        self.mu0 = self.vacuum_permeability

        # Permittivity of free space
        self.vacuum_permittivity = 8.85418782e-12 * self.capacitance / self.length
        self.eps0 = self.vacuum_permittivity

        # Classical electron radius
        self.classical_electron_radius = 2.81794033e-15 * self.length
        self.re = self.classical_electron_radius

        # Electron volt
        self.electron_volt = 1.602176565e-19 * self.energy
        self.eV = self.electron_volt

        # Atomic mass unit (CODATA 2010 value)
        self.atomic_mass_unit = 1.660538921e-27 * self.mass
        self.amu = self.atomic_mass_unit

        # Standard temperature
        self.standard_temperature = 273.15 * self.temperature
        # Standard pressure
        self.standard_pressure = 101325.0 * self.force / (self.length * self.length)

    def __repr__(self):
        s = "\n".join(f"{key} = {getattr(self, key)!r}" for key in self.__dict__)
        return s
