.. _`chap:mix`:

Turbulent Mixing (BHR)
======================

The ``mix`` package models unresolved, subgrid turbulent mixing with the Besnard–Harlow–Rauenzahn (BHR) second-moment RANS closure — specifically the “BHR-3.1” variant. It is intended for variable-density mixing driven by Rayleigh–Taylor, Richtmyer–Meshkov, and Kelvin–Helmholtz instabilities, where the mixing layer is not resolved on the mesh. The package transports a set of turbulence moments, feeds their Reynolds stress and turbulent mass flux back onto the mean flow, and diffuses mass and energy across the mixing layer.

   **Cartesian only.** The BHR source terms and fluxes assume Cartesian gradients and divergences (no metric or curvature terms); the package aborts at initialization under cylindrical or spherical coordinates. It requires hydrodynamics.

Governing Equations
-------------------

Unlike a one-equation (:math:`K`) or two-equation (:math:`K`–:math:`\varepsilon`) model, BHR-3.1 is a *second-moment* closure: it transports the full turbulent Reynolds stress tensor together with the correlations that drive variable-density mixing. The evolved turbulence moments are, per cell (all *bulk* quantities),

.. container:: description

   the turbulent Reynolds stress tensor (six independent components), field ``ccbulk::reynolds_stress``;

   the turbulent mass flux — the density–velocity correlation that is the engine of variable-density mixing — field ``ccbulk::bhr_a``;

   the density–specific-volume correlation :math:`b=-\overline{\rho'\,(1/\rho)'}`, field ``ccbulk::bhr_b``;

   two turbulent length scales, a transport scale :math:`S_T` and a dissipation scale :math:`S_D`, fields ``ccbulk::bhr_ST`` and ``ccbulk::bhr_SD``.

The turbulent kinetic energy is not evolved separately; it is the half-trace of the Reynolds stress, :math:`K=\tfrac{1}{2}(R_{xx}+R_{yy}+R_{zz})`. Each moment is carried in density-weighted conservative form (e.g. :math:`\rho R_{ij}`, ``ccbulk::rho_reynolds_stress``) and advected with the flow; the primitive form is recovered by dividing by the bulk density.

Turbulent Viscosity
~~~~~~~~~~~~~~~~~~~

The closure defines an eddy viscosity from the transport length scale and the turbulent kinetic energy,

.. math::

   \begin{equation}
     \mu_t = c_\mu\,\rho\,S_T\,\sqrt{K},
   \end{equation}

which sets the gradient-diffusion coefficient for every turbulent transport term below.

Transport Equations
~~~~~~~~~~~~~~~~~~~

Let :math:`q` stand for any of the turbulence moments :math:`\{R_{ij},\,a_i,\,b,\,S_T,\,S_D\}`. Each is transported in density-weighted conservative form: the conserved quantity :math:`\rho q` is advected with the mean flow and evolved by an algebraic source :math:`\mathcal{S}_q` and a turbulent gradient-diffusion term,

.. math::

   \begin{equation}
     \frac{\partial \left(\rho\,q\right)}{\partial t} + \nabla\!\cdot\!\left(\rho\,q\,\bm{v}\right)
       = \mathcal{S}_q
         + \nabla\!\cdot\!\left(\frac{\mu_t}{\sigma_q}\,\nabla q\right),
     \label{eq:mix-transport}
   \end{equation}

where the eddy viscosity :math:`\mu_t` is given above and :math:`\sigma_q` is a quantity-specific Schmidt/Prandtl number (``sigma_k`` for :math:`R_{ij}`, ``sigma_a`` for :math:`a_i`, ``sigma_b`` for :math:`b`, ``sigma_epsilon`` for :math:`S_T`, ``sigma_visc`` for :math:`S_D`). The diffusion term is applied direction-by-direction. What distinguishes the moments is the algebraic source :math:`\mathcal{S}_q`, which combines production, redistribution, and dissipation. Writing :math:`\bm{a}` for the mass flux, :math:`\nabla p` for the pressure gradient, and using the production term :math:`\rho\,R_{ij}\partial_j u_i` and the buoyancy term :math:`\bm{a}\!\cdot\!\nabla p`, the sources are:

- **Reynolds stress :math:`R_{ij}`**: production from the mean shear and from the buoyancy correlation :math:`a_i\partial_i p`, pressure–strain redistribution toward isotropy, and a dissipation :math:`\propto \rho\sqrt{K}\,R_{ij}/S_D`. A realizability limiter prevents the pressure–strain terms from driving a diagonal component negative.

- **Mass flux :math:`a_i`**: buoyant production :math:`\propto b\,\partial_i p`, a :math:`R_{ij}\partial_j\rho` term, self-advection, and dissipation :math:`\propto \rho\sqrt{K}\,a_i/S_D`.

- **Density correlation :math:`b`**: production from :math:`\bm{a}\!\cdot\!\nabla\rho` and dissipation :math:`\propto \rho\,b\,\sqrt{K}/S_D`.

- **Length scales :math:`S_T`, :math:`S_D`**: each grows or decays with the local production-to-dissipation balance and the dilatation :math:`\nabla\!\cdot\!\bm{u}`. :math:`S_T` uses the coefficient set :math:`\{c_1,c_2,c_3,c_4\}` and :math:`S_D` the corresponding “v” set :math:`\{c_{1v},c_{2v},c_{3v},c_{4v}\}`.

In each case the dissipation carries the length scale :math:`S_D` in its denominator, and the algebraic source enters :math:`\mathcal{S}_q` in the equation above alongside the gradient-diffusion term.

Feedback on the Mean Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

The turbulence acts back on the resolved (bulk) hydrodynamics of Chapter :ref:`chap:hydro` through additional interface fluxes, applied in three stages:

.. container:: description

   The Reynolds stress :math:`\rho R_{ij}` is added to the bulk momentum flux, and the corresponding turbulent transport of energy (:math:`\rho\,\bm{v}\!\cdot\!R`) together with the pressure work of the mass flux (:math:`-p\,a_n`) is added to the total-energy flux.

   The eddy viscosity diffuses turbulent kinetic energy and per-material enthalpy into the energy flux, and deposits a per-material *diffusive mass flux* into a dedicated face register.

   That diffusive mass flux is then folded into each material’s density flux and used to advect *every* other per-material conserved quantity, so the turbulent mass diffusion transports all co-moving material quantities consistently, not just mass.

Numerical Method
----------------

The BHR fluxes are computed after the hydrodynamic fluxes and before flux correction, in the fixed order ``ComputeStressFluxes`` :math:`\to` ``ComputeViscousFluxes`` :math:`\to` ``ComputeAnonFluxes`` (the last consumes the diffusive mass flux the second deposits). The algebraic and gradient-diffusion moment sources are accumulated by ``CalculateMixSource`` and summed into the stage update alongside the other packages’ sources. After each update the primitive moments are recovered and floored: the Reynolds-stress diagonal is kept positive, the length scales non-negative, and :math:`b` clamped to :math:`[0,\rho]`. The stable time step is the minimum of a hyperbolic limit (:math:`|\bm{a}|/\Delta x`), a parabolic diffusion limit set by :math:`\mu_t` and the smallest Schmidt number, and a homogeneous source-decay limit.

Input Parameters
----------------

Turbulent mixing is enabled with ``mix`` ``= true`` in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``hydro`` must also be enabled. The model coefficients and initial conditions live in the ``<mix>`` block. The defaults below are the standard BHR-3.1 calibration; most problems change only the initial conditions ``K0`` and ``S0``.

.. list-table:: Closure coefficients in the ``<mix>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - c_1, c_2, c_3, c_4
     - Real
     - ``1.6``, ``1.77``, ``0.0``, ``1.1``
     - Transport length-scale (:math:`S_T`) source coefficients.
   * - c_1v, c_2v, c_3v, c_4v
     - Real
     - ``1.3``, ``1.77``, ``0.0``, ``1.24``
     - Dissipation length-scale (:math:`S_D`) source coefficients.
   * - c_a1, c_a3, c_ap, c_au
     - Real
     - ``2.8``, ``1.0``, ``0.1``, ``0.4``
     - Mass-flux (:math:`a_i`) dissipation, divergence, buoyancy, and advection coefficients.
   * - c_b2
     - Real
     - ``1.8``
     - Density-correlation (:math:`b`) dissipation coefficient.
   * - c_r1, c_r2, c_r4
     - Real
     - ``0.3``, ``0.6``, ``1.8``
     - Reynolds-stress pressure–strain, production, and return-to-isotropy coefficients.
   * - c_mu
     - Real
     - ``0.28``
     - Eddy-viscosity coefficient, :math:`\mu_t=c_\mu\rho S_T\sqrt{K}`.
   * - sigma_k, sigma_a, sigma_b, sigma_c
     - Real
     - ``1.0``
     - Schmidt/Prandtl numbers for diffusion of the Reynolds stress, mass flux, density correlation, and mass/enthalpy.
   * - sigma_visc
     - Real
     - ``0.6``
     - Schmidt number for :math:`S_D` diffusion.
   * - sigma_epsilon
     - Real
     - ``0.1``
     - Schmidt number for :math:`S_T` diffusion.

.. list-table:: Initial conditions in the ``<mix>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - K0
     - Real
     - ``0.01``
     - Initial (and floor) turbulent kinetic energy.
   * - S0
     - Real
     - ``0.0001``
     - Initial turbulent length scale.

..

   The coefficients ``c_a2``, ``c_ar`` are read for completeness but are not referenced by the current source terms.

Registered Fields
-----------------

The package registers the turbulence moments in the table below, each as a conserved density-weighted field (advected with fluxes) paired with a derived primitive field. All are *bulk* fields. A sparse per-material face field holds the turbulent diffusive mass flux that couples the three flux stages.

.. container:: fieldtable

   | Fields registered by the mix package.tab:mix-fields ccbulk::rho_reynolds_stress & :math:`\rho R_{ij}` & 6 & Cell, Independent, Intensive, Conserved, Vector, FillGhost, Advected, WithFluxes; conserved Reynolds stress.
   | ccbulk::reynolds_stress & :math:`R_{ij}` & 6 & Cell, Intensive, Vector, Derived, OneCopy; primitive Reynolds stress.
   | ccbulk::rho_bhr_a & :math:`\rho a_i` & 3 & Cell, Independent, Intensive, Conserved, Vector, FillGhost, Advected, WithFluxes; conserved mass flux.
   | ccbulk::bhr_a & :math:`a_i` & 3 & Cell, Intensive, Vector, Derived, OneCopy; primitive mass flux.
   | ccbulk::rho_bhr_b & :math:`\rho b` & 1 & Cell, Independent, Intensive, Conserved, FillGhost, Advected, WithFluxes; conserved density correlation.
   | ccbulk::rho_bhr_ST & :math:`\rho S_T` & 1 & Cell, Independent, Intensive, Conserved, FillGhost, Advected, WithFluxes; conserved transport length scale.
   | ccbulk::rho_bhr_SD & :math:`\rho S_D` & 1 & Cell, Independent, Intensive, Conserved, FillGhost, Advected, WithFluxes; conserved dissipation length scale.
   | ccbulk::bhr_b, bhr_ST, bhr_SD & :math:`b, S_T, S_D` & 1 & Cell, Intensive, Derived, OneCopy; primitive :math:`b` and length scales.
   | fm::diffusive_fluxes & — & 1/mat & Face, OneCopy, Sparse, CellMemAligned; per-material turbulent diffusive mass flux.

Example
-------

Enable BHR mixing with the default calibration, seeding a small initial turbulent kinetic energy and length scale:

.. code:: python

   riot.input("physics", hydro=True, mix=True)
   riot.input("mix", K0=1.0e-4, S0=1.0e-3)   # initial TKE and length scale
