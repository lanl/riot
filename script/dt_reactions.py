#!/usr/bin/env python3
"""D + T -> n + alpha: equal-temperature Maxwellian averages.

This file was written with the assistance of generative AI.

All public energies and temperatures are in MeV; temperature means k_B*T.
Reactivity is in cm^3/s, cross section in cm^2. Both species have zero bulk
velocity in the lab and the same temperature. Product energies use the
approximate 2:3:1:4 mass ratios and nonrelativistic kinematics discussed in
the derivation (roughly percent-level accuracy, not precision kinematics).

The cross section is the Bosch-Hale D-T fit:
H.-S. Bosch & G.M. Hale, Nuclear Fusion 32 (1992) 611.
https://doi.org/10.1088/0029-5515/32/4/I07
Coefficients taken from the reference implementation
Higginson, Link, and Schmidti, J Comp Phys. 388, 439 (2019)
https://doi.org/10.1016/j.jcp.2019.03.020
implemented here
https://github.com/llnl/pyPICfusion/blob/master/_BoschHale_library.py

Integration is restricted to the fit's collision-energy interval,
0.0005 <= E_cm <= 0.550 MeV; tails outside it are omitted, not
extrapolated.  Below k_B*T = 0.001 MeV all outputs are set to
zero. Above roughly 0.030 MeV the omitted tail above 0.550 MeV is no
longer negligible, so the results are increasingly truncated as T
rises. Reaction rate density = n_D*n_T*reactivity, with densities in
cm^-3, giving reactions/(cm^3 s). Nuclear power density is that rate
times Q, in MeV/(cm^3 s). Incoming thermal energy is not generated
energy.

The command line writes a reduced isotope_data.hdf5 containing only the
d+t->n+a reaction, in the format read by src/tnburn/ratelib/read_isotope_data.hpp.
That file is in code (CGS) units: temperatures in K, energies and Qval in erg,
and SigmaVBar = reactivity / (m_D * m_T) in cm^3/(s g^2). The interpolated
tables are on a uniform grid in ln(T/K).
"""

import argparse

import h5py
import numpy as np

# Match parthenon's CGS PhysicalConstants (CODATA 2010), which the code uses.
ERG_PER_MEV = 1.602176565e-6
K_B_ERG_PER_K = 1.3806488e-16
AMU_G = 1.660538921e-24

REACTION_NAME = "d+t->n+a"
REACTANTS = (1002, 1003)  # ZAIDs: D, T
PRODUCTS = (1, 2004)  # ZAIDs: n, alpha; EnergyOut0 and EnergyOut1 respectively
# ZAID -> (mass in amu, charge), as in isotope_data.hdf5
ISOTOPES = {
    1: (1.008664904, 0),
    1002: (2.0141020803071998, 1),
    1003: (3.0155005623387843, 1),
    2004: (4.0026032366859754, 2),
}

Q_MEV = 17.589
C_CM_S = 2.99792458e10
MU_C2_MEV = 1124.656  # D-T reduced mass times c^2
E_MIN_MEV, E_MAX_MEV = 0.0005, 0.550
T_MIN_MEV = 0.001


def sigma_cm2(energy_mev):
    """Bosch-Hale total cross section at CENTER-OF-MASS energy (MeV)."""
    energy_mev = np.asarray(energy_mev, dtype=float)
    if not np.all(np.isfinite(energy_mev)
                  & (energy_mev >= E_MIN_MEV) & (energy_mev <= E_MAX_MEV)):
        raise ValueError("Cross-section energy must be in [0.0005, 0.550] MeV")
    e = 1000.0 * energy_mev  # Original fit coefficients require keV internally.
    numerator = 6.927e4 + e * (7.454e8 + e * (2.050e6 + e * 5.2002e4))
    denominator = 1.0 + e * (63.8 + e * (-0.995 + e * (6.981e-5 + e * 1.728e-4)))
    sigma_millibarn = numerator / denominator / e * np.exp(-34.3827 / np.sqrt(e))
    return sigma_millibarn * 1e-27  # 1 millibarn = 1e-27 cm^2


def _quadrature_nodes(panels=64, order=16):
    """Composite Gauss-Legendre nodes (MeV) and weights in u = ln(E)."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(np.log(E_MIN_MEV), np.log(E_MAX_MEV), panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    u = (mid[:, None] + half[:, None] * x).ravel()
    return np.exp(u), (half[:, None] * w).ravel()


# The cross section does not depend on temperature, so tabulate it once.
_E_NODES_MEV, _U_WEIGHTS = _quadrature_nodes()
_SIGMA_BARN = sigma_cm2(_E_NODES_MEV) / 1e-24


def evaluate(temperatures_mev):
    """Evaluate an array of temperatures (MeV); averages are per fusion event.

    Returns (reactivity_cm3_s, mean_collision_energy_mev,
    mean_neutron_energy_mev, mean_alpha_energy_mev), each a numpy array
    with the same shape as temperatures_mev. Entries whose temperature is
    below T_MIN_MEV (or not finite) are set to zero.

    Use x = E/T and sigma in barns to keep quadrature integrands well scaled:
        J1 = integral sigma(T*x) * x   * exp(-x) dx
        J2 = integral sigma(T*x) * x^2 * exp(-x) dx
        <sigma v> = 1e-24*c*sqrt(8*T/(pi*mu*c^2))*J1
        <E>_fusion = T*J2/J1
    The integrals are done on a fixed Gauss-Legendre grid in u = ln(E), where
    dx = x du, so all temperatures are integrated at once.

    Center-of-mass kinetic energy of the reacting pair averages to 3*T/2.
    It is independent of the relative collision energy at equal temperatures.
    The product energies include it, but the returned input energy is the
    collision energy alone, matching the InputEnergy convention of
    isotope_data.hdf5.
    """
    t_all = np.asarray(temperatures_mev, dtype=float)
    valid = np.isfinite(t_all) & (t_all >= T_MIN_MEV)
    t = t_all[valid]

    x = _E_NODES_MEV / t[:, None]
    weighted = _SIGMA_BARN * _U_WEIGHTS * x**2 * np.exp(-x)  # sigma * x * e^-x dx
    j1 = weighted.sum(axis=-1)
    j2 = (weighted * x).sum(axis=-1)

    reactivity = 1e-24 * C_CM_S * np.sqrt(8.0 * t / (np.pi * MU_C2_MEV)) * j1
    collision = t * j2 / j1
    pair_cm = 1.5 * t
    neutron = 0.8 * (Q_MEV + collision) + 0.2 * pair_cm
    alpha = 0.2 * (Q_MEV + collision) + 0.8 * pair_cm

    outputs = []
    for values in (reactivity, collision, neutron, alpha):
        out = np.zeros_like(t_all)
        out[valid] = values
        outputs.append(out)
    return tuple(outputs)


def _int_attr(obj, name, values):
    obj.attrs.create(name, np.atleast_1d(np.asarray(values, dtype=np.int32)))


def _string_attr(obj, name, text):
    """Scalar null-terminated ASCII string attribute, as Spiner writes them."""
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(text) + 1)
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    obj.attrs.create(name, np.bytes_(text), dtype=h5py.Datatype(tid))


def _write_databox(parent, name, data, log_t_range=None):
    """Write a rank-1 Spiner DataBox.

    With log_t_range = (min, max) the axis is Interpolated on that uniform
    grid; without it the axis is Indexed and the grids group is left empty.
    """
    box = parent.create_group(name)
    box.create_dataset("data", data=np.asarray(data, dtype=np.float64))
    grids = box.create_group("grids")
    if log_t_range is not None:
        lo, hi = log_t_range
        grid = grids.create_dataset(
            "grid_[1]", data=np.array([lo, hi, (hi - lo) / (len(data) - 1)]))
        _int_attr(grid, "npoints", len(data))
        _string_attr(grid, "range columns", "[0]:min [1]:max [2]:dx")
    _int_attr(box, "dims", len(data))
    _int_attr(box, "rank", 1)
    _int_attr(box, "index_types", 2 if log_t_range is None else 0)
    _string_attr(box, "index_types_info", "Interpolated:0\nNamed:1\nIndexed:2")


def write_isotope_data(filename, temperatures_mev):
    """Write a d+t->n+a-only isotope_data.hdf5 tabulated at temperatures_mev."""
    t_mev = np.asarray(temperatures_mev, dtype=float)
    reactivity, collision, neutron, alpha = evaluate(t_mev)
    t_k = t_mev * ERG_PER_MEV / K_B_ERG_PER_K
    log_t_range = (np.log(t_k[0]), np.log(t_k[-1]))
    reactant_masses_g2 = np.prod([ISOTOPES[z][0] * AMU_G for z in REACTANTS])

    with h5py.File(filename, "w") as f:
        reaction = f.create_group("reactions").create_group(REACTION_NAME)
        _int_attr(reaction, "num_reactants", len(REACTANTS))
        _int_attr(reaction, "num_products", len(PRODUCTS))
        _int_attr(reaction, "num_temp", len(t_k))
        reaction.attrs.create("Qval", np.array([Q_MEV * ERG_PER_MEV]))
        # The reader expects length-3 arrays, zero padded.
        _int_attr(reaction, "reactants", [*REACTANTS, 0])
        _int_attr(reaction, "products", [*PRODUCTS, 0])
        _int_attr(reaction, "product_multiplicities", [1, 1, 0])
        _write_databox(reaction, "Temperatures", t_k)
        _write_databox(reaction, "InputEnergy", collision * ERG_PER_MEV, log_t_range)
        _write_databox(reaction, "SigmaVBar", reactivity / reactant_masses_g2, log_t_range)
        _write_databox(reaction, "EnergyOut0", neutron * ERG_PER_MEV, log_t_range)
        _write_databox(reaction, "EnergyOut1", alpha * ERG_PER_MEV, log_t_range)

        masses = f.create_group("Masses")
        zaids = sorted(ISOTOPES)
        _int_attr(masses, "num_isotopes", len(zaids))
        _int_attr(masses, "zaids", zaids)
        masses.attrs.create("masses", np.array([ISOTOPES[z][0] for z in zaids]))
        _int_attr(masses, "charges", [ISOTOPES[z][1] for z in zaids])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("t_min", type=float, help="minimum k_B*T in MeV")
    parser.add_argument("t_max", type=float, help="maximum k_B*T in MeV")
    parser.add_argument("-n", "--num-points", type=int, default=181,
                        help="number of temperatures, evenly spaced in ln(T); default: 181")
    parser.add_argument("-o", "--output", default="dt_isotope_data.hdf5",
                        help="output HDF5 file; default: dt_isotope_data.hdf5")
    args = parser.parse_args()
    if not 0.0 < args.t_min < args.t_max:
        parser.error("Temperatures must satisfy 0 < t_min < t_max")
    if args.num_points < 2:
        parser.error("--num-points must be at least 2")
    write_isotope_data(args.output,
                       np.geomspace(args.t_min, args.t_max, args.num_points))


if __name__ == "__main__":
    main()
