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
    problem="pipe",  # name of the pgen
)

riot.input(
    "parthenon/job",
    problem_id="pipe",  # problem ID: basename of output filenames
)

riot.input(
    "parthenon/output1",
    variables=[
        "c.c.bulk.rho",
        "c.c.bulk.temperature",
        "c.c.rad.aa",
        "c.c.rad.moments",
    ],
    file_type="hdf5",  # HDF5 data dump
    dt=1.0e-9,  # time increment between outputs
)

riot.input(
    "parthenon/time",
    nlim=-1,  # cycle limit
    tlim=1.0e-7,  # time limit
    integrator="rk1",  # time integration algorithm
    ncycle_out=1,  # interval for stdout summary info
)

riot.input(
    "parthenon/mesh",
    refinement="none",  # refinement type
    numlevel=2,  # number of AMR levels
    nx1=280,  # Number of zones in X1-direction
    x1min=-3.5,  # minimum value of X1
    x1max=3.5,  # maximum value of X1
    ix1_bc="ic",  # Inner-X1 boundary condition flag
    ox1_bc="ic",  # Outer-X1 boundary condition flag
    nx2=160,  # Number of zones in X2-direction
    x2min=-2.0,  # minimum value of X2
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
    nx1=14,  # meshblock size in X1-direction
    nx2=8,  # meshblock size in X2-direction
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
    Cv=8.61733e7,  # specific heat at constant volume
    opac_a="powerlaw",  # absorption opacity type (overwritten by pgen)
    kappa0_a=0.0,  # absorption opacity coefficient (overwritten by pgen)
    kappa_Tpower_a=0.0,  # absorption opacity temperature power (overwritten by pgen)
    kappa_Rhopower_a=0.0,  # absorption opacity density power (overwritten by pgen)
    max_bnd_level=-1,  # maximum boundary AMR level
    max_mat_level=0,  # maximum material AMR level
)

riot.input(
    "region0",
    name="thin",  # region name
    mask_type="background",  # region mask type
    matid=0,  # material ID
    c_m_rho=0.01,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region1",
    name="region1",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=-3.5,  # rectangle x-coordinate lower bound
    x1=-1.0,  # rectangle x-coordinate upper bound
    y0=0.5,  # rectangle y-coordinate lower bound
    y1=2.0,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region2",
    name="region2",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=-3.5,  # rectangle x-coordinate lower bound
    x1=-1.0,  # rectangle x-coordinate upper bound
    y0=-2.0,  # rectangle y-coordinate lower bound
    y1=-0.5,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region3",
    name="region3",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=1.0,  # rectangle x-coordinate lower bound
    x1=3.5,  # rectangle x-coordinate upper bound
    y0=0.5,  # rectangle y-coordinate lower bound
    y1=2.0,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region4",
    name="region4",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=1.0,  # rectangle x-coordinate lower bound
    x1=3.5,  # rectangle x-coordinate upper bound
    y0=-2.0,  # rectangle y-coordinate lower bound
    y1=-0.5,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region5",
    name="region5",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=-0.5,  # rectangle x-coordinate lower bound
    x1=0.5,  # rectangle x-coordinate upper bound
    y0=-1.0,  # rectangle y-coordinate lower bound
    y1=1.0,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region6",
    name="region6",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=-3.5,  # rectangle x-coordinate lower bound
    x1=3.5,  # rectangle x-coordinate upper bound
    y0=1.5,  # rectangle y-coordinate lower bound
    y1=2.0,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "region7",
    name="region7",  # region name
    mask_type="inside_rectangle",  # region mask type
    matid=0,  # material ID
    x0=-3.5,  # rectangle x-coordinate lower bound
    x1=3.5,  # rectangle x-coordinate upper bound
    y0=-2.0,  # rectangle y-coordinate lower bound
    y1=-1.5,  # rectangle y-coordinate upper bound
    c_m_rho=10.0,  # conserved material density
    c_m_temperature=5.803e5,  # conserved material temperature
)

riot.input(
    "physics",
    hydro=True,  # enable hydrodynamics
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
    do_jacobi=True,  # enable implicit Jacobi solver
    do_explicit=False,  # enable explicit transport
    nlevel=3,  # level of geodesic mesh
    beta=1.0,  # parameter controlling tauc in Rusanov rad-flux
    coupling=True,  # flag to enable radiation source term
    affect_fluid=True,  # feedback on the fluid
    fixed_pgen_opac=True,  # do not update opacities set in pgen
    troot_tol=1.0e-4,  # temperature root find tolerance
    cfl=0.8,  # The Courant, Friedrichs, & Lewy (CFL) Number
)

riot.input(
    "radiation_transport/jacobi",
    err_thr=1.0e-4,  # implicit residual threshold
    dt_ratio_hyperbolic=1.0e3,  # Multiple of light-crossing time
    verbose=2,  # Verbosity
)

if __name__ == "__main__":
    riot.input.generate_input()
