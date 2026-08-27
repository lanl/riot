#!/usr/bin/env python
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

"""
Generate fake TN data HDF5 files for testing purposes.

This script creates complete HDF5 files with isotope data (masses, charges) and
optionally reaction data in the format expected by RIOT. The data is physically
plausible but not real nuclear data - suitable for code testing only.

Usage:
    python make_fake_tn_data.py output.hdf5 --isotopes 1002,1003,2004,1 --reactions d+t->n+a
    python make_fake_tn_data.py output.hdf5 --isotopes 1002,1003,2004,1  # isotopes only
"""

import sys
import argparse
import numpy as np
import h5py

# Standard isotope data (ZAID -> (mass_amu, charge))
ISOTOPE_DATA = {
    1: (1.008664904, 0),  # neutron
    1001: (1.0078249887344393, 1),  # proton
    1002: (2.0141020803072, 1),  # deuterium
    1003: (3.0155005623387843, 1),  # tritium
    2003: (3.014931675332928, 2),  # helium-3
    2004: (4.002603236685975, 2),  # helium-4
}

# Standard reactions (name -> (reactants, products, product_multiplicities, Qval_MeV))
REACTION_DATA = {
    "d+t->n+a": ([1002, 1003], [1, 2004], [1, 1], 17.6),
    "t+t->2n+a": ([1003, 1003], [1, 2004], [2, 1], 11.3),
    "d+d->n+he3": ([1002, 1002], [1, 2003], [1, 1], 3.27),
    "d+d->p+t": ([1002, 1002], [1001, 1003], [1, 1], 4.03),
    "p+t->n+he3": ([1001, 1003], [1, 2003], [1, 1], -0.764),
    "n+he3->p+t": ([1, 2003], [1001, 1003], [1, 1], 0.764),
    "d+he3->p+a": ([1002, 2003], [1001, 2004], [1, 1], 18.3),
    "he3+he3->2p+a": ([2003, 2003], [1001, 2004], [2, 1], 12.86),
    "t+he3->n+p+a": ([1003, 2003], [1, 1001, 2004], [1, 1, 1], 12.1),
    "t+he3->d+a": ([1003, 2003], [1002, 2004], [1, 1], 14.3),
}


def make_fake_data(xpoint, ypoint, xpeak, ypeak, xpoints):
    """
    Generates data that looks like a Gaussian with peak value ypeak at xpeak,
    passing through (xpoint, ypoint). Useful for creating plausible cross-section curves.

    Args:
        xpoint: x-coordinate of point to pass through
        ypoint: y-coordinate of point to pass through
        xpeak: x-coordinate of peak
        ypeak: y-coordinate of peak
        xpoints: array of x-values to evaluate

    Returns:
        Array of y-values following Gaussian curve
    """
    if abs(ypeak - ypoint) > 1e-20:
        c = (xpeak - xpoint) ** 2 / np.log(ypeak / ypoint)
        y = ypeak * np.exp(-((xpoints - xpeak) ** 2) / c)
    else:
        y = xpoints * 0.0 + ypoint
    return y


def create_spiner_databox(group, name, data, abscissa=None, indexed=False):
    """
    Create a Spiner DataBox structure in HDF5.

    Args:
        group: HDF5 group to create DataBox in
        name: name of the DataBox
        data: 1D numpy array of data
        abscissa: (xmin, xmax) of the independent variable the data is indexed
            by. Spiner's uniform grid describes the *abscissa* over which the
            data is interpolated, NOT the range of the data values. For the
            temperature-indexed reaction tables this must be the log-temperature
            range, since the code interpolates with log(temperature). If None,
            fall back to the data's own min/max (only sensible for tables that
            are never interpolated, e.g. Temperatures itself).
        indexed: if True, write the axis as Spiner's Indexed type (index_types=2)
            with no uniform grid. Spiner only saves/loads grid_[1] for
            Interpolated axes, so the grids group is created empty. This matches
            how NDI/ndi2spiner store the Temperatures table (a plain lookup axis
            that the code never interpolates over). When True, abscissa is
            ignored.
    """
    db = group.create_group(name)
    db.create_dataset("data", data=data)

    # Create grids group. Spiner always opens this group on load, but only
    # populates grid_[1] for Interpolated axes; Indexed axes leave it empty.
    grids = db.create_group("grids")
    if not indexed:
        if abscissa is not None:
            xmin, xmax = float(abscissa[0]), float(abscissa[1])
        else:
            xmin = float(data.min())
            xmax = float(data.max())
        # Spiner spaces npoints samples inclusively over [xmin, xmax], so the
        # step is divided by (npoints - 1), not npoints.
        dx = (xmax - xmin) / (len(data) - 1) if len(data) > 1 else 1.0
        grid_dataset = grids.create_dataset(
            "grid_[1]", data=np.array([xmin, xmax, dx], dtype=np.float64)
        )

        # Add npoints attribute to grid (required by Spiner)
        grid_dataset.attrs["npoints"] = len(data)

    # Add attributes
    db.attrs["dims"] = len(data)
    db.attrs["rank"] = 1
    db.attrs["index_types"] = 2 if indexed else 0  # Indexed vs Interpolated


def parse_reaction(rxn_str):
    """
    Parse a reaction string like "d+t->n+a" into reactants and products.

    Args:
        rxn_str: reaction string

    Returns:
        tuple: (reactants, products, product_multiplicities, Qval)
    """
    if rxn_str in REACTION_DATA:
        return REACTION_DATA[rxn_str]
    else:
        raise ValueError(
            f"Unknown reaction: {rxn_str}. Known reactions: {list(REACTION_DATA.keys())}"
        )


def create_reaction_group(rxn_group, rxn_name):
    """
    Create a reaction group with fake data.

    Args:
        rxn_group: HDF5 reactions group
        rxn_name: name of reaction (e.g., "d+t->n+a")
    """
    reactants, products, product_mults, qval = parse_reaction(rxn_name)

    # Create reaction subgroup
    reaction = rxn_group.create_group(rxn_name)

    # ndi2spiner runs setup_arrays_for_code_units, which converts Qval from the
    # NDI-native MeV to erg (code units). The consumer reads Qval as-is, so the
    # fake file must store it in erg too or the code would behave differently
    # depending on where the data came from.
    ergs_per_mev = 1.602176634e-6  # 1e6 * eV in erg (CGS), matches ergs_per_MeV

    # Set attributes
    reaction.attrs["num_reactants"] = len(reactants)
    reaction.attrs["num_products"] = len(products)
    reaction.attrs["num_temp"] = 181
    reaction.attrs["Qval"] = qval * ergs_per_mev

    # Pad arrays to length 3 with zeros
    reactants_padded = reactants + [0] * (3 - len(reactants))
    products_padded = products + [0] * (3 - len(products))
    mults_padded = product_mults + [0] * (3 - len(product_mults))

    reaction.attrs["reactants"] = np.array(reactants_padded, dtype=np.int32)
    reaction.attrs["products"] = np.array(products_padded, dtype=np.int32)
    reaction.attrs["product_multiplicities"] = np.array(mults_padded, dtype=np.int32)

    # Generate fake temperature-dependent data
    mev_to_k = 1.16044e10
    temperatures_mev = np.exp(np.linspace(np.log(1e-6), np.log(1.0), 181))
    temperatures_k = temperatures_mev * mev_to_k

    # Create realistic-looking cross section curves
    ein = make_fake_data(0.001, 1.0e-8, 0.1, 2.0e-7, temperatures_mev)
    svbar = make_fake_data(0.001, 5.0e26, 0.1, 5.0e31, temperatures_mev)

    # The reaction tables are interpolated in log(temperature) (in Kelvin) by the
    # code, so the interpolated DataBoxes must carry that log-temperature range as
    # their abscissa.
    log_temp_range = (np.log(temperatures_k[0]), np.log(temperatures_k[-1]))

    # Create DataBoxes. Temperatures is a plain lookup axis (never interpolated
    # by the code), so it is written as Indexed to match NDI/ndi2spiner output.
    create_spiner_databox(reaction, "Temperatures", temperatures_k, indexed=True)
    create_spiner_databox(reaction, "InputEnergy", ein, abscissa=log_temp_range)
    create_spiner_databox(reaction, "SigmaVBar", svbar, abscissa=log_temp_range)

    # Energy out per product
    for j in range(len(products)):
        # Scale energy by Q-value
        energy_scale = qval / len(products)  # Split energy among products
        eout = make_fake_data(
            0.001, energy_scale * 1e-6, 0.1, energy_scale * 1e-5, temperatures_mev
        )
        create_spiner_databox(reaction, f"EnergyOut{j}", eout, abscissa=log_temp_range)


def create_fake_tn_file(filename, isotopes, reactions=None):
    """
    Create a complete fake TN data HDF5 file.

    Args:
        filename: output HDF5 filename
        isotopes: list of ZAIDs
        reactions: optional list of reaction strings
    """
    # Collect all isotopes (including those from reactions)
    all_isotopes = set(isotopes)
    if reactions:
        for rxn in reactions:
            reactants, products, _, _ = parse_reaction(rxn)
            all_isotopes.update(reactants)
            all_isotopes.update(products)

    all_isotopes = sorted(all_isotopes)

    # Get masses and charges
    masses = []
    charges = []
    for zaid in all_isotopes:
        if zaid in ISOTOPE_DATA:
            mass, charge = ISOTOPE_DATA[zaid]
        else:
            # Estimate from ZAID if not in table
            # ZAID format: ZZAAA where ZZ=charge, AAA=mass number
            charge = zaid // 1000
            mass_number = zaid % 1000
            # Rough estimate: mass = mass_number * amu
            mass = mass_number * 1.0
            print(
                f"Warning: Unknown ZAID {zaid}, estimating mass={mass}, charge={charge}"
            )

        masses.append(mass)
        charges.append(charge)

    # Create HDF5 file
    with h5py.File(filename, "w") as f:
        # Create reactions group
        if reactions:
            rxn_group = f.create_group("reactions")
            for rxn_str in reactions:
                print(f"Creating fake data for reaction: {rxn_str}")
                create_reaction_group(rxn_group, rxn_str)

        # Create Masses group
        mass_group = f.create_group("Masses")
        mass_group.attrs["num_isotopes"] = len(all_isotopes)
        mass_group.attrs["zaids"] = np.array(all_isotopes, dtype=np.int32)
        mass_group.attrs["masses"] = np.array(masses, dtype=np.float64)
        mass_group.attrs["charges"] = np.array(charges, dtype=np.int32)

    print(f"Created fake TN data file: {filename}")
    print(f"  Isotopes: {all_isotopes}")
    print(f"  Masses: {masses}")
    print(f"  Charges: {charges}")
    if reactions:
        print(f"  Reactions: {reactions}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake TN data HDF5 files for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create file with all default reactions from tn_test_multireaction
  %(prog)s output.hdf5

  # Create file with specific reactions only
  %(prog)s output.hdf5 --reactions d+t->n+a

  # Create file with isotopes only (for Materials without TNBurn)
  %(prog)s output.hdf5 --isotopes 1002,1003,2004,1 --reactions ""

Available reactions: d+t->n+a, t+t->2n+a, d+d->n+he3, d+d->p+t, p+t->n+he3,
                     n+he3->p+t, d+he3->p+a, he3+he3->2p+a, t+he3->n+p+a, t+he3->d+a
        """,
    )
    parser.add_argument("output", help="Output HDF5 filename")
    parser.add_argument(
        "--isotopes",
        default=None,
        help="Comma-separated list of isotope ZAIDs (default: all isotopes from reactions)",
    )
    parser.add_argument(
        "--reactions",
        default="d+t->n+a,t+t->2n+a,d+d->n+he3,d+d->p+t,p+t->n+he3,n+he3->p+t,d+he3->p+a,he3+he3->2p+a,t+he3->n+p+a,t+he3->d+a",
        help="Comma-separated list of reactions (default: all reactions from tn_test_multireaction)",
    )

    args = parser.parse_args()

    # Parse reactions
    reactions = None
    if args.reactions:
        reactions = [x.strip() for x in args.reactions.split(",")]

    # Parse isotopes (or derive from reactions)
    isotopes = []
    if args.isotopes:
        isotopes = [int(x.strip()) for x in args.isotopes.split(",")]

    # Create file
    create_fake_tn_file(args.output, isotopes, reactions)

    print("\nNote: This file contains FAKE DATA for testing only!")
    print("Do not use for production simulations.")


if __name__ == "__main__":
    main()
