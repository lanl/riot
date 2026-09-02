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
#
# NOTE(@pdmullen):
#
# - This is the classic nonlinear (equilibrium-diffusion) Marshak wave / Zel'dovich-
#   Raizer thermal-conduction wave, here solved with SN (discrete-ordinates) transport.
#   A hot, static slab is driven by a constant-temperature radiation source at the
#   inner-x1 boundary and the resulting thermal wave propagates into initially cold,
#   static material.  The front advances self-similarly as x_f ~ sqrt(t).
#
# - We reuse the "shock" problem generator: the shock pgen initializes a left/right
#   static state and its custom inner-/outer-x1 "ic" boundaries inject an isotropic
#   Planckian intensity Emissivity(temp_l/temp_r) into the ghost angles.  For SN this
#   incident-blackbody wall IS the Marshak boundary condition, so (unlike the riot
#   equilibrium-diffusion setup) we do not need the infinite-heat-capacity reservoir
#   trick -- a single uniform cold material plus the hot inner-x1 wall suffices.  We
#   place the discontinuity at the left edge (xd = x1min) so the whole interior starts
#   cold and the hot wall drives the wave in +x.
#
# - Opacity follows a Kramers-like kappa = kappa0 * T^-3 (kappa_Tpower_a = -3), which
#   gives the temperature-dependent diffusivity responsible for the sharp, self-
#   sharpening front.  Physical parameters (CGS) match riot's radiation_diffusion
#   marshak test: kappa0 = 1.56272e22 cm^2/g, rho = 3 g/cm^3, Cv = 8.61733e10 erg/g/K,
#   Gamma = 1.5, Tbound = 1.16045e7 K (~1 keV), cold background ~1.16e2 K.
#
# - fixed_fluid=True freezes the hydro (no advection); coupling/affect_fluid=True let
#   the radiation source term heat the matter so the temperature wave can propagate.
#   The implicit (Jacobi) solver is used since the problem is diffusion-dominated.
#
# ========================================================================================

import riot

Tbound = 1.16045e7  # hot-wall (source) temperature [K] (~1 keV)
Tbackground = 1.16045e2  # cold initial/far-field temperature [K]
rho = 3.0  # uniform material density [g/cm^3]

riot.input(
    "riot",
    problem="shock",  # reuse the shock pgen (L/R static state + radiating ic walls)
)

riot.input(
    "parthenon/job",
    problem_id="marshak",  # problem ID: basename of output filenames
)

riot.input(
    "parthenon/output1",
    variables=[
        "c.c.bulk.temperature",
        "c.c.rad.moments",
    ],
    file_type="hdf5",  # HDF5 data dump
    dt=5.0e-4,  # time increment between outputs
)

riot.input(
    "parthenon/time",
    nlim=-1,  # cycle limit
    tlim=4.0e-3,  # time limit
    integrator="rk1",  # time integration algorithm
    ncycle_out=100,  # interval for stdout summary info
    dt_init=1.0e-9,  # first-cycle timestep (controller grows it thereafter)
)

riot.input(
    "parthenon/mesh",
    refinement="none",  # refinement type
    nx1=256,  # Number of zones in X1-direction
    x1min=0.0,  # minimum value of X1
    x1max=10.0,  # maximum value of X1
    ix1_bc="ic",  # Inner-X1 boundary condition flag (hot radiating wall)
    ox1_bc="ic",  # Outer-X1 boundary condition flag (cold radiating wall)
    nx2=1,  # Number of zones in X2-direction
    x2min=-0.5,  # minimum value of X2
    x2max=0.5,  # maximum value of X2
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
    nx1=64,  # meshblock size in X1-direction
    nx2=1,  # meshblock size in X2-direction
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
    Gamma=1.5,  # adiabatic index
    Cv=8.61733e10,  # specific heat at constant volume
    opac_a="powerlaw",  # absorption opacity type
    kappa0_a=1.56272e22,  # absorption opacity coefficient
    kappa_Tpower_a=-3.0,  # absorption opacity temperature power (Kramers-like)
    kappa_Rhopower_a=0.0,  # absorption opacity density power
    max_bnd_level=-1,  # maximum boundary AMR level
    max_mat_level=0,  # maximum material AMR level
)

riot.input(
    "physics",
    hydro=True,  # enable hydrodynamics (materials owns group structure)
    radiation_transport=True,  # enable radiation transport
    fixed_fluid=True,  # fix fluid (no hydro update); radiation still heats matter
    sparse_physics=False,  # enable sparse physics allocation
)

riot.input(
    "hydro",
    cfl=0.8,  # CFL number for hydro
    amr_interface=False,  # use AMR interface reconstruction
)

riot.input(
    "radiation_transport",
    do_explicit=False,  # enable explicit transport
    do_jacobi=True,  # enable implicit Jacobi solver (diffusion-dominated)
    fv_fix=False, # Use centroid unit normals
    nlevel=1,  # level of geodesic mesh
    beta=5.0,  # parameter controlling tauc in Rusanov rad-flux
    coupling=True,  # flag to enable radiation source term
    affect_fluid=True,  # feedback on the fluid (heats the static matter)
    troot_tol=1.0e-6,  # temperature root find tolerance
)

riot.input(
    "radiation_transport/jacobi",
    err_thr=1.0e-4,  # implicit residual threshold
    niter_limit=200,  # cap Jacobi iterations per step
    dt_ratio_hyperbolic=5.0e4,  # timestep controller for multiple of hyperbolic dt
    dt_ratio_lag=0.25,  # timestep controller for lagged opacities
)

riot.input(
    "problem",
    xd=0.0,  # discontinuity at left edge -> whole interior starts cold
    # L State (hot radiating wall injected at inner-x1)
    rho_l=rho,  # L-state density
    vx_l=0.0,  # L-state vel-x (static)
    temp_l=Tbound,  # L-state temperature (hot source)
    # R State (cold, static interior)
    rho_r=rho,  # R-state density
    vx_r=0.0,  # R-state vel-x (static)
    temp_r=Tbackground,  # R-state temperature (cold)
)

if __name__ == "__main__":
    riot.input.generate_input()
