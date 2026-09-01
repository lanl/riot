.. _`chap:radtransport`:

Radiation Transport
===================

The ``radiation_transport`` package solves the time-dependent equation of photon radiative transfer with a discrete-ordinates (:math:`S_N`) method. It transports the specific intensity along a fixed set of angular directions, couples the radiation field to the material internal energy through absorption and emission, and supports grey or multigroup frequency resolution. Two solver variants are provided: an *explicit* sub-cycled integrator and an *implicit* Jacobi iteration. Opacities are supplied per material (Chapter :ref:`chap:materials`).

Governing Equations
-------------------

Radiative Transfer Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time evolution of the frequency-dependent specific intensity :math:`I_\nu` is governed by the transport equation

.. math::

     \frac{\partial I_\nu}{\partial t} + c\,\vec{n}\!\cdot\!\nabla I_\nu = c\left(j_\nu - \alpha_\nu I_\nu\right),

where :math:`c` is the speed of light, :math:`\vec{n}` is a unit vector defining the photon propagation direction, and :math:`j_\nu` and :math:`\alpha_\nu` are the frequency-dependent emissivity and absorptivity, respectively.

Multigroup Formulation
~~~~~~~~~~~~~~~~~~~~~~

In the multigroup formalism the frequency domain is divided into :math:`N_\nu` bins by endpoints :math:`\nu_0,\nu_1,\dots,\nu_{N_\nu}`, with :math:`\nu_0 = 0` and :math:`\nu_{N_\nu}=\infty` covering all of frequency space. Integrating the transport equation over group :math:`f`, with frequency bounds :math:`[\nu_{f-1},\nu_f)`, gives

.. math::

     \frac{\partial I_f}{\partial t} + c\,\vec{n}\!\cdot\!\nabla I_f = c\left(j_f - \alpha_f I_f\right),

where the subscript :math:`f` denotes a quantity integrated over the group (e.g. :math:`I_f = \int_{\nu_{f-1}}^{\nu_f} I_\nu\,\mathrm{d}\nu` is the frequency-integrated specific intensity), and the group absorptivity :math:`\alpha_f` is a group-mean average.

Assuming a static background medium (material mass density :math:`\rho`, temperature :math:`T`), local thermodynamic equilibrium (LTE), and isotropic elastic scattering, the group transport equation expands to

.. math::

     \frac{\partial I_f}{\partial t} + c\,\vec{n}\!\cdot\!\nabla I_f
       = c\left[\sigma_{s,f}\left(J_f - I_f\right)
                + \sigma_{a,f}\left(\varepsilon_f - I_f\right)\right],

where :math:`\sigma_{a,f}` and :math:`\sigma_{s,f}` are the group-mean absorption and scattering coefficients of the cell. In a multi-material cell these are aggregated from the per-material opacities as given in Section :ref:`sec:rad-opacity`; for a single material :math:`m` they reduce to :math:`\sigma_{a,f}=\rho_m\kappa_{a,f,m}` and :math:`\sigma_{s,f}=\rho_m\kappa_{s,f,m}` in terms of the specific opacities :math:`\kappa_{a,f,m}`, :math:`\kappa_{s,f,m}`. The group mean intensity over solid angle :math:`\Omega` is

.. math::

     J_f = \frac{1}{4\pi}\int I_f\,\mathrm{d}\Omega ,

and is related to the group radiation energy density by :math:`E_f = \tfrac{4\pi}{c} J_f = \tfrac{1}{c}\int I_f\,\mathrm{d}\Omega`. The emission coefficient :math:`\varepsilon_f` is proportional to the integral of the Planck function :math:`B(\nu,T)` over the group band,

.. math::

     \varepsilon_f &= \frac{c}{4\pi}\int_{\nu_{f-1}}^{\nu_f} B(\nu,T)\,\mathrm{d}\nu,
       \qquad\text{where}\\
     B(\nu,T) &= \frac{8\pi h\nu^3}{c^3}\,
                 \frac{1}{\exp\!\left(h\nu/[kT]\right) - 1}.

For a single group with :math:`[\nu_0,\nu_{N_\nu}) \to [0,\infty)` (i.e. “grey”), :math:`\varepsilon_f` reduces to :math:`\varepsilon_{\text{grey}} = c\,a\,T^4/[4\pi]`, where :math:`a` is the radiation constant. The band integrals are evaluated in closed form using polylogarithms.

The frequency group structure (:math:`N_\nu` and the group boundaries in Hz) is *not* an input of the radiation package: it is owned by the materials package and resolved from the ``<materials>`` block, the opacity tables, or a grey default, exactly as described in Section :ref:`sec:mat-group-structure`. The radiation package reads the resolved structure from the materials package at initialization, so the transport and the opacities always share one group grid.

Stored Intensity and Energy Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For efficiency the transported field ``ccrad::intensity`` stores a scaled intensity that absorbs a factor of :math:`4\pi/c` relative to :math:`I_f`, so that the group radiation energy density is recovered directly as a weighted sum over the discrete directions,

.. math::

     E_f = \sum_{a} w_a\,I_{f,a},

where :math:`I_{f,a}` is the stored intensity in direction :math:`a` and the angular quadrature weights :math:`w_a` are normalized to :math:`\sum_a w_a = 1`. The streaming term is discretized with a Rusanov flux weighted by the local optical depth (controlled by ``beta`` and ``taumax``), so the scheme transitions smoothly between the optically thin (transport) and optically thick (diffusion) limits. In reduced-dimension curvilinear geometries (1D spherical, 2D cylindrical) an additional angular-flux divergence term (``ccrad::divfa``) accounts for the rotation of :math:`\vec{n}` along curved streaming paths; it is compiled in only for those coordinate systems.

Angular Discretization
~~~~~~~~~~~~~~~~~~~~~~

The unit sphere of directions is discretized into :math:`N_{\text{ang}}` ordinates, each carrying a solid-angle weight. Two quadratures are available: a nearly uniform *geodesic* grid built by subdividing an icosahedron, for which :math:`N_{\text{ang}} = 10\,n_{\text{level}}^2 + 2`, and a *latitude–longitude* product grid with :math:`N_{\text{ang}} = n_\theta\,n_\phi`. The geodesic grid may be rotated to avoid alignment with the spatial mesh. The quadrature and its resolution are set by the angular-grid parameters below.

.. _`sec:rad-opacity`:

Multi-Material Opacities
~~~~~~~~~~~~~~~~~~~~~~~~

The cell absorption and scattering coefficients that appear in the equation above are aggregated from the per-material opacities (Section :ref:`sec:mat-opacity`) as volume-fraction-weighted sums,

.. math::

     \sigma_{a,f} = \sum_m f_m\,\rho_m\,\kappa_{a,f,m}
                  = \sum_m f_m\,\sigma_{a,f,m}, \qquad
     \sigma_{s,f} = \sum_m f_m\,\sigma_{s,f,m},

where each material’s coefficient :math:`\sigma_{a,f,m} = \rho_m\,\kappa_{a,f,m}` is evaluated from its own opacity model at the material-averaged density :math:`\rho_m` and the (shared) cell temperature. This follows the volume-fraction aggregation of Section :ref:`sec:permat-bulk`, so a mixed cell presents a single set of group coefficients to the transport solve.

Matter–Radiation Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~

When coupling is enabled (``coupling``), energy exchanged with the radiation field is removed from (or added to) the material internal energy, conserving total energy,

.. math::

     \frac{\partial E_{\text{mat}}}{\partial t} = -\,\frac{\partial E_{\text{rad}}}{\partial t}, \qquad
     E_{\text{rad}} = \sum_f E_f .

The exchange is treated implicitly: an advanced material temperature :math:`T^{n+1}` is found by a non-linear root solve (tolerance ``troot_tol``, iteration cap ``troot_max_iter``) balancing the emission, absorption, and the change in material energy over the step. This keeps the strong emission/absorption coupling stable at large optical depth. The feedback onto the fluid energy can be suppressed independently with ``affect_fluid`` (e.g. to advance the radiation field against a fixed matter state), and the advanced-temperature solve itself can be bypassed with ``fixed_temp_rhs``.

Solvers
-------

Two integrators advance the transport equation. Radiation transport is enabled by setting ``radiation_transport`` ``= true`` in the ``<physics>`` block (Section :ref:`sec:physics-block`); the solver is then chosen in the ``<radiation_transport>`` block by ``do_explicit`` and ``do_jacobi``. Both default to ``false``, and RIOT requires that *exactly one* be enabled — it is an error to set both or neither.

.. container::
   :name: tab:rad-solvers

   .. table:: Radiation transport solvers.

      +-------------+------------------------------------------------------------------------------------------------------+
      | **Solver**  | **Description**                                                                                      |
      +=============+======================================================================================================+
      | do_explicit | Sub-cycled explicit integrator (Runge–Kutta), with an optical-depth-weighted Rusanov streaming flux. |
      +-------------+------------------------------------------------------------------------------------------------------+
      | do_jacobi   | Iterative implicit (Jacobi) solver; robust at large optical depth, iterated to a residual threshold. |
      +-------------+------------------------------------------------------------------------------------------------------+

Input Parameters
----------------

Radiation transport requires hydrodynamics and is incompatible with the ionization package, with sparse physics (Chapter :ref:`chap:sparse-physics`), and with the P1 radiation diffusion package (Chapter :ref:`chap:raddiffusion`), which provides an alternative radiation model — at most one of the two may be enabled. Because ``sparse_physics`` defaults to ``true``, a radiation run must set it ``false`` explicitly in the ``<physics>`` block.

Parameters are organized into a shared ``<radiation_transport>`` block and three nested blocks. The shared block holds everything common to both solvers (CFL, coupling, angular grid, unit overrides); the chosen solver’s algorithm-specific parameters live in ``<radiation_transport/explicit>`` or ``<radiation_transport/jacobi>``; the radiation initial state is set in ``<radiation_transport/init>``; and the drive boundary condition is configured in ``<radiation_transport/drive>``.

.. list-table:: Solver selection and shared parameters in the ``<radiation_transport>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - do_explicit
     - bool
     - ``false``
     - Use the explicit sub-cycled integrator.
   * - do_jacobi
     - bool
     - ``false``
     - Use the implicit Jacobi solver. Exactly one of these two must be ``true``.
   * - cfl
     - Real
     - ``0.8``
     - CFL number for the transport update.
   * - coupling
     - bool
     - ``true``
     - Enable the emission/absorption/scattering source.
   * - affect_fluid
     - bool
     - *=coupling*
     - Feed the radiation source back onto the fluid energy (requires ``coupling``).
   * - fixed_temp_rhs
     - bool
     - ``false``
     - Skip the advanced-temperature solve in the source term.
   * - beta
     - Real
     - *geom.*
     - Weight on the local optical depth in the Rusanov flux (:math:`1.0` in Cartesian, :math:`0.0` in curvilinear geometries).
   * - taumax
     - Real
     - *large*
     - Cap on the optical depth used in the Rusanov flux (default :math:`\approx` ``Real`` max).
   * - troot_tol
     - Real
     - ``1e-8``
     - Tolerance for the non-linear temperature root find.
   * - troot_max_iter
     - int
     - ``25``
     - Maximum iterations for the temperature root find.
   * - fixed_pgen_opac
     - bool
     - ``false``
     - Fix opacities to the values set in the problem generator.
   * - units_override
     - bool
     - ``false``
     - Use a custom (non-CGS) unit system for testing.
   * - c, arad, kb, h
     - Real
     - ``1.0``
     - Speed of light, radiation, Boltzmann, and Planck constants (only when ``units_override``).

The ``beta``, ``taumax``, ``troot_tol``, and ``troot_max_iter`` parameters are read only when ``coupling`` is ``true``.

.. list-table:: Angular-grid parameters (in the ``<radiation_transport>`` block).
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - angular_mesh
     - string
     - ``geodesic``
     - Angular quadrature: ``geodesic`` or ``latlon``.
   * - nlevel
     - int
     - ``1``
     - Geodesic refinement level; :math:`N_{\text{ang}} = 10\,n_{\text{level}}^2 + 2`.
   * - rotate_geo
     - int
     - ``1``
     - Geodesic rotation: 0 none, 1 automatic, 2 user angles.
   * - zpole, ppole
     - Real
     - *NaN*
     - Manual geodesic rotation angles (used when ``rotate_geo`` ``= 2``).
   * - ntheta
     - int
     - ``8``
     - Latitude bins (latlon grid).
   * - nphi
     - int
     - ``16``
     - Longitude bins (latlon grid).

The explicit solver adds sub-cycling controls:

.. list-table:: Parameters in the ``<radiation_transport/explicit>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - integrator
     - string
     - ``rk2``
     - Time integrator for sub-cycling: ``rk1``, ``rk2``, or ``rk3``.
   * - dt_ratio_hyperbolic
     - Real
     - ``-1.0``
     - Limit the global step to ``cfl``\ :math:`\times`\ this\ :math:`\times\min(\Delta x)/c`; :math:`-1` lets radiation sub-cycle without limiting the global step.
   * - verbose
     - int
     - ``0``
     - Diagnostic verbosity (0–2).

The Jacobi solver adds iteration and timestep controls:

.. list-table:: Parameters in the ``<radiation_transport/jacobi>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - niter_limit
     - int
     - ``1000``
     - Maximum Jacobi iterations per step.
   * - err_thr
     - Real
     - ``1e-8``
     - Residual threshold for convergence.
   * - split_g1
     - bool
     - ``true``
     - Split the Jacobi coefficient into positive/negative parts for robustness.
   * - dt_ratio_hyperbolic
     - Real
     - ``1e4``
     - Limit the global step to a multiple of the hyperbolic step; :math:`-1` disables this controller.
   * - dt_ratio_lag
     - Real
     - ``-1.0``
     - *Experimental* step limiter accounting for lagged opacities; :math:`-1` disables it.
   * - verbose
     - int
     - ``0``
     - Diagnostic verbosity (0–3; higher levels report per-iteration residuals, subcycling, and root-find failures).

.. _`sec:rad-jacobi-subcycle`:

Convergence Subcycling
~~~~~~~~~~~~~~~~~~~~~~

The Jacobi solver performs a single *global* implicit solve over the whole mesh each step, with the opacities lagged from the start of the step. This is a demanding nonlinear problem, and RIOT provides an escape hatch for the rare case in which an individual solve truly goes off the rails — its residual growing or stalling rather than settling. When a solve is judged to be diverging, it is discarded and the operator-split step is retried from the pristine start-of-step state with the timestep divided by ``reduce_factor``, repeating (reducing further as needed) until each subinterval completes and the full step is covered. It is a robustness backstop rather than part of the normal solve, and it is off by default.

Subcycling is distinct from the ordinary iteration count. Reaching ``niter_limit`` is *not* divergence: a solve that iterates smoothly and is simply cut off at the iteration cap is accepted and committed as usual. A solve is flagged as diverging only when its residual becomes non-finite, or — when the stall detector is enabled — when it fails to improve on its best residual so far for ``ndiverge_limit`` consecutive iterations. If subcycling is disabled (``nreduce_limit`` ``= 0``) or the reduction budget is exhausted, a genuinely diverging solve aborts the run.

.. list-table:: Convergence-subcycling parameters (in the ``<radiation_transport/jacobi>`` block).
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - nreduce_limit
     - int
     - ``0``
     - Maximum number of timestep reductions permitted on divergence; :math:`0` disables subcycling (a diverging solve aborts).
   * - reduce_factor
     - int
     - ``2``
     - Integer factor by which the subcycle timestep is divided at each reduction (:math:`>1`).
   * - ndiverge_limit
     - int
     - ``-1``
     - Consecutive non-improving iterations that count as divergence and trigger a reduction; :math:`-1` disables the stall detector, so only a non-finite residual fails a solve.

Initialization
~~~~~~~~~~~~~~

The radiation field’s initial state is chosen in the ``<radiation_transport/init>`` block.

.. list-table:: Parameters in the ``<radiation_transport/init>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - initialization
     - string
     - ``thermal``
     - Initial radiation field: ``thermal`` (in equilibrium with the matter, :math:`E=aT^4`), ``zero`` (empty), or ``none`` (leave the problem-generator-set intensity untouched).

.. _`sec:rad-drive-bc`:

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

By default the radiation intensity inherits the same face boundary conditions as the rest of the mesh (``ix1_bc`` …, Chapter :ref:`chap:parthenon`). Any face may instead be given a *drive* condition — a fixed incoming radiation temperature — by naming it in the ``<radiation_transport/drive>`` block.

.. list-table:: Parameters in the ``<radiation_transport/drive>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ix1_bc …ox3_bc
     - string
     - ``default``
     - Per-face selector; set to ``drive`` to impose the drive condition on that face.
   * - trad_bc
     - Real
     - ``0.0``
     - Uniform radiation temperature (K) injected by driven faces.
   * - force_upwind_flux_bc
     - bool
     - ``true``
     - Zero the boundary opacity to force a purely upwind flux at a driven face; if ``false``, copy the interior opacity.

Opacities are a *material* property, not a radiation input: absorption and scattering models are selected per material in each material’s opacity block (``opac_a``, ``opac_s``), and the frequency group structure on which the transport runs is likewise owned by the materials package. Both are documented in the materials chapter — the opacity models in Section :ref:`sec:mat-opacity` and the group structure in Section :ref:`sec:mat-group-structure`.

Registered Fields
-----------------

The package registers the radiation fields in the table below under the ``ccrad::`` prefix. The stored intensity is the transported field, carrying one component per (group, angle) pair; recall it absorbs the :math:`4\pi/c` factor so that :math:`E_f = \sum_a w_a I_{f,a}`. The remaining fields are derived opacities, moments, and per-solver scratch. Every field also carries the OperatorSplit flag and a per-solver user flag; the set differs between the two solvers, as noted below.

.. container:: fieldtable

   | Principal fields registered by the radiation package.tab:rad-fields ccrad::intensity & :math:`I_{f,a}` & :math:`N_\nu N_{\text{ang}}` & Cell, Independent, FillGhost, Intensive, Conserved, OperatorSplit; stored intensity per group and angle (:math:`4\pi/c`-scaled). The explicit solver adds WithFluxes; the Jacobi solver registers linear prolongation/restriction ops instead.
   | ccrad::aa & :math:`\sigma_{a,f}` & :math:`N_\nu` & Cell, Derived, OneCopy, FillGhost, OperatorSplit; cell absorption coefficient.
   | ccrad::ss & :math:`\sigma_{s,f}` & :math:`N_\nu` & Cell, Derived, OneCopy, FillGhost, OperatorSplit; cell scattering coefficient.
   | ccrad::moments & :math:`E_f` & :math:`N_\nu` & Cell, Derived, OneCopy, OperatorSplit; group radiation energy density and moments.
   | ccrad::s1, s2, s3 & — & :math:`N_\nu` & Cell, Derived, OneCopy, OperatorSplit; auxiliary moment/source scratch (both solvers).
   | ccrad::divfa & — & :math:`N_\nu N_{\text{ang}}` & Cell, Derived, OneCopy, OperatorSplit; angular-flux divergence. Explicit solver only, and only in curvilinear geometries.
   | ccrad::tauw & — & :math:`N_\nu` & Cell, Derived, OneCopy, OperatorSplit; optical-depth weight (Jacobi solver only).
   | ccrad::temperature & :math:`T^{n+1}` & 1 & Cell, Derived, OneCopy, OperatorSplit; advanced temperature (Jacobi solver only).

Example
-------

A grey, geodesic-grid transport run using the implicit Jacobi solver, coupled to a fixed (non-advecting) fluid, with a power-law absorption opacity on material ``0`` — the structure of a Marshak-wave problem:

.. code:: python

   riot.input("physics", hydro=True, radiation_transport=True,
              fixed_fluid=True, sparse_physics=False)

   riot.input("material0", label="mat0",
              eos_type="IdealGas", Gamma=1.5, Cv=8.61733e10,
              opac_a="powerlaw",       # absorption opacity model
              kappa0_a=1.56272e22,     # coefficient
              kappa_Tpower_a=-3.0)     # Kramers-like temperature power

   riot.input("radiation_transport",
              do_jacobi=True,          # select the implicit solver
              nlevel=1,                # geodesic grid: 12 angles
              coupling=True,
              affect_fluid=True,       # let radiation heat the matter
              beta=5.0)

   riot.input("radiation_transport/jacobi",
              err_thr=1.0e-4, niter_limit=200,
              dt_ratio_hyperbolic=5.0e4)

   # A hot radiating wall driving the inner-x1 face:
   riot.input("radiation_transport/drive", ix1_bc="drive", trad_bc=1.0)
   riot.input("radiation_transport/init", initialization="zero")
