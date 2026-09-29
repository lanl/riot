#!/usr/bin/env python3
# ========================================================================================
# (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

import riot

riot.input(
    "riot",
    problem="region_pgen",  # name of the pgen
)

riot.input(
    "parthenon/job",
    problem_id="hohlraum",  # problem ID: basename of output filenames
)

riot.input(
    "parthenon/output1",
    variables=["c.c.rad.moments"],
    file_type="hdf5",  # HDF5 data dump
    dt=0.1,  # time increment between outputs
    use_final_label=True,  # do not add "final" label to last output
)

riot.input(
    "parthenon/time",
    nlim=-1,  # cycle limit
    tlim=0.75,  # time limit
    integrator="rk1",  # time integration algorithm
    ncycle_out=1,  # interval for stdout summary info
)

riot.input(
    "parthenon/mesh",
    refinement="none",  # refinement type
    nx1=128,  # Number of zones in X1-direction
    x1min=0.0,  # minimum value of X1
    x1max=2.0,  # maximum value of X1
    ix1_bc="outflow",  # Inner-X1 boundary condition flag
    ox1_bc="outflow",  # Outer-X1 boundary condition flag
    nx2=128,  # Number of zones in X2-direction
    x2min=0.0,  # minimum value of X2
    x2max=2.0,  # maximum value of X2
    ix2_bc="outflow",  # Inner-X2 boundary condition flag
    ox2_bc="outflow",  # Outer-X2 boundary condition flag
    nx3=1,  # Number of zones in X3-direction
    x3min=-0.5,  # minimum value of X3
    x3max=0.5,  # maximum value of X3
    ix3_bc="periodic",  # Inner-X3 boundary condition flag
    ox3_bc="periodic",  # Outer-X3 boundary condition flag
)

riot.input(
    "parthenon/meshblock",
    nx1=64,  # meshblock size in X1-direction
    nx2=64,  # meshblock size in X2-direction
    nx3=1,  # meshblock size in X3-direction
)

riot.input(
    "materials",
    sparse_init=False,  # enable sparse material initialization
    sparse_dealloc=False,  # enable sparse material deallocation
)

riot.input(
    "material0",
    label="mat0",  # material label
    eos_type="IdealGas",  # equation of state type
    Gamma=2.0,  # adiabatic index
    Cv=1.0,  # specific heat at constant volume
    opac_a="powerlaw",  # absorption opacity type
    kappa0_a=0.0,  # absorption opacity coefficient (vacuum)
    kappa_Tpower_a=0.0,  # absorption opacity temperature power
    kappa_Rhopower_a=0.0,  # absorption opacity density power
    max_bnd_level=-1,  # maximum boundary AMR level
    max_mat_level=0,  # maximum material AMR level
)

riot.input(
    "region0",
    name="vacuum",  # region name
    mask_type="background",  # region mask type
    matid=0,  # material ID
    c_m_rho=1.0,  # conserved material density
    c_m_temperature=1.0e-3,  # conserved material temperature
)

riot.input(
    "physics",
    hydro=True,  # enable hydrodynamics (materials owns group structure)
    radiation_transport=True,  # enable radiation transport
    fixed_fluid=True,  # fix fluid (no hydro update)
    sparse_physics=False,  # enable sparse physics allocation
)

riot.input(
    "hydro",
    cfl=0.8,  # CFL number for hydro
    amr_interface=False,  # use AMR interface reconstruction
)

riot.input(
    "radiation_transport",
    do_jacobi=False,  # enable implicit Jacobi solver
    do_explicit=True,  # enable explicit transport
    nlevel=2,  # level of geodesic mesh
    coupling=False,  # flag to enable radiation source term
    affect_fluid=False,  # feedback on the fluid
    fixed_pgen_opac=True,  # do not update opacities set in pgen
    cfl=0.5,  # The Courant, Friedrichs, & Lewy (CFL) Number
    units_override=True,  # Use custom units system (default c=1, a=1, kb=1, h=1)
)

riot.input(
    "radiation_transport/explicit",
    dt_ratio_hyperbolic=1000.0,  # Multiple of light-crossing time
    verbose=1,  # Verbosity
)

riot.input(
    "radiation_transport/jacobi",
    dt_ratio_hyperbolic=100.0,  # Multiple of light-crossing time
    dt_ratio_lag=-1,
    verbose=1,  # Verbosity
)

riot.input(
    "radiation_transport/drive",
    trad_bc=1.0,  # uniform boundary source
    ix1_bc="drive",
    ix2_bc="drive",
)

riot.input(
    "radiation_transport/init",
    initialization="zero",
)


if __name__ == "__main__":
    riot.input.generate_input()
