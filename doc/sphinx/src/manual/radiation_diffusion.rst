.. _`chap:raddiffusion`:

Radiation Diffusion (P1)
========================

The ``multigroup_diffusion`` package is an implicit, multigroup radiation transport solver built on the *P1* (first-moment) closure. Despite its name, it is not a pure diffusion solver: it evolves both the group radiation energy density and the group radiation flux, coupling them so the scheme behaves as free-streaming transport in the optically thin limit and as diffusion in the optically thick limit. It couples the radiation field to the material internal energy through emission and absorption, and — like the discrete-ordinates transport package (Chapter :ref:`chap:radtransport`) — it takes its frequency group structure and its opacities from the materials package (Chapter :ref:`chap:materials`).

   Radiation diffusion and discrete-ordinates radiation transport (Chapter :ref:`chap:radtransport`) are mutually exclusive: at most one may be enabled. Both require hydrodynamics.

Governing Equations
-------------------

The P1 Moment System
~~~~~~~~~~~~~~~~~~~~

For each frequency group :math:`g` the package evolves two moments of the specific intensity: the group radiation energy density :math:`E_g` (cell-centered, field ``rmg::Egroup``) and the group radiation flux :math:`\bm{F}_g` (face-centered, field ``rmg::Fgroup``). Writing :math:`c` for the speed of light, :math:`a` for the radiation constant, and :math:`\sigma_{a,g}` for the cell group-mean absorption coefficient (aggregated from the per-material opacities exactly as in Section :ref:`sec:rad-opacity`), the moment system is

.. math::

   \begin{align}
     \frac{\partial E_g}{\partial t} + \nabla\!\cdot\!\bm{F}_g
       &= c\,\sigma_{a,g}\left(B_g(T) - E_g\right), \\[2pt]
     \frac{1}{c}\frac{\partial \bm{F}_g}{\partial t} + \frac{c}{3}\,\nabla E_g
       &= -\,\sigma_{t,g}\,\bm{F}_g ,
   \end{align}

where :math:`\sigma_{t,g}` is the group total (absorption plus scattering) coefficient and :math:`B_g(T)` is the group-integrated Planck function (below). The factor :math:`\tfrac{1}{3}` in the flux equation is the Eddington factor of the P1 closure — the assumption that the radiation pressure tensor is isotropic, :math:`\mathsf{P}_g = \tfrac{1}{3}E_g\,\mathsf{I}`. Retaining the time derivative of the flux keeps the system hyperbolic (a finite signal speed), so it reduces to free-streaming transport when :math:`\sigma_{t,g}` is small and to the diffusion limit :math:`\bm{F}_g \to -\tfrac{c}{3\sigma_{t,g}}\nabla E_g` when :math:`\sigma_{t,g}` is large.

Implicit Discretization
~~~~~~~~~~~~~~~~~~~~~~~

Both moments are advanced implicitly over the step :math:`\Delta t`. Eliminating the updated flux from the discretized momentum equation yields, per group, an implicit update for :math:`E_g` of the form

.. math::

   \begin{equation}
     E_g^{n+1} - \nabla\!\cdot\!\left(D_g\,\nabla E_g^{n+1}\right)
       = E_g^{n} + \text{(lagged flux, emission/absorption)} ,
   \end{equation}

where the effective diffusion coefficient carries the P1 flux relaxation in its denominator,

.. math::

   \begin{equation}
     D_g = \frac{(c\,\Delta t)^2}{3\left(1 + c\,\Delta t\,\sigma_{t,g}^{\text{face}}\right)} .
   \end{equation}

The face total opacity :math:`\sigma_{t,g}^{\text{face}}` is the harmonic mean of the two adjacent cell values. The updated flux is then reconstructed from the new energy density and the lagged flux, each attenuated by the same :math:`1/(1 + c\,\Delta t\,\sigma_{t,g}^{\text{face}})` factor. This relaxation is what limits the flux to physical values as the opacity grows; it plays the role a Levermore-style flux limiter would in a classical flux-limited-diffusion scheme.

Emission and Matter Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The emission term drives the radiation toward a Planck spectrum at the material temperature :math:`T`. The group-integrated Planck function is

.. math::

   \begin{equation}
     B_g(T) = a\,T^4\left[\Phi\!\left(\frac{h\nu_g}{k_B T}\right)
                        - \Phi\!\left(\frac{h\nu_{g-1}}{k_B T}\right)\right],
   \end{equation}

where :math:`\Phi(x) = \tfrac{15}{\pi^4}\int_0^{x} y^3/(e^y-1)\,\mathrm{d}y` is the normalized cumulative Planck fraction, evaluated in closed form with polylogarithms. Energy removed from (or added to) the radiation field is exchanged with the material internal energy so that total energy is conserved; the coupled matter temperature and the group energies are found together by the non-linear iteration described below.

The temperature the radiation couples to depends on the physics configuration. In a run without ionization the package couples to the bulk material temperature and updates the bulk internal energy; when the ionization package (Chapter :ref:`chap:ionization`) is active it instead couples to the *electron* temperature and electron internal energy, so radiation exchanges energy with the electrons.

Frequency Groups and Opacities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of frequency groups :math:`N_\nu` and the group boundaries are *not* inputs of this package: they are owned by the materials package and resolved from the ``<materials>`` block, the opacity tables, or a grey default, exactly as described in Section :ref:`sec:mat-group-structure`. The absorption and scattering coefficients :math:`\sigma_{a,g}`, :math:`\sigma_{t,g}` are likewise built from the per-material opacity models of Section :ref:`sec:mat-opacity` and combined into cell coefficients by the volume-fraction aggregation of Section :ref:`sec:rad-opacity`. A grey run is simply the single-group case.

Numerical Method
----------------

The package is applied as an *operator-split* step after the hydrodynamic update on each cycle. Within the step, the non-linear radiation–matter coupling is resolved by an outer Newton–Raphson iteration (at most ``nriter`` passes), converged on the relative change in temperature to ``nr_tolerance``. Each Newton pass assembles the linear system for the group energy densities and solves it with a Parthenon **BiCGSTAB** Krylov solver preconditioned by **geometric multigrid**; the linear solver’s own controls (tolerances, iteration and cycle limits) live in the ``<diffusion/linear_solver_params>`` block and are parsed by Parthenon. An optional set of zone-local Newton iterations (``local_nriter``, default none) can further relax the cell-local energy/temperature balance after the global solve.

Boundary conditions for the radiation energy are selected with ``boundary_condition``: a fixed-temperature (Robin) condition, a zero-flux (reflecting) condition, or a time-dependent ``double_shell`` drive.

Input Parameters
----------------

Radiation diffusion is enabled with ``multigroup_diffusion`` ``= true`` in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``hydro`` must also be enabled, and ``radiation_transport`` must be off. Its controls live in the ``<diffusion>`` block.

.. container:: paramtable

   | Principal parameters in the ``<diffusion>`` block. boundary_condition & string & ``constant_temperature`` & Radiation boundary: ``constant_temperature``, ``zero_flux``, or ``double_shell``.
   | boundary_T & list & ``1e5`` & Boundary temperature(s) in K; a single value applies to all six faces, or give one per face.
   | update_temperature & bool & ``false`` & Update the matter temperature from radiation exchange within the implicit solve (enables inter-group coupling terms).
   | nriter & int & ``3`` & Maximum outer Newton–Raphson iterations.
   | local_nriter & int & ``0`` & Additional zone-local Newton iterations after the global solve.
   | nr_tolerance & Real & ``1e-5`` & Convergence tolerance on the relative temperature change.
   | print_per_nr_step & bool & ``false`` & Print Newton-iteration convergence to screen.
   | opacity_rho_min & Real & *tiny* & Density floor used when evaluating opacities.
   | opacity_temp_min & Real & *tiny* & Temperature floor used when evaluating opacities.
   | a_radiation & Real & *CGS :math:`a`* & Radiation constant (defaults to the CGS value).
   | c_light & Real & *CGS :math:`c`* & Speed of light (defaults to the CGS value).
   | report_timings & bool & ``false`` & Report a solver timing breakdown at the end of the run.

.. container:: paramtable

   | Time-step controls in the ``<diffusion>`` block. cfl & Real & *large* & CFL number for the radiation step vote; the large default effectively lets the temperature heuristic below govern the step.
   | temperature_fractional_change_target & Real & ``0.1`` & Target fractional temperature change per step.
   | timestep_min_temperature & Real & ``1.0`` & Minimum zone temperature considered in the step vote.
   | timestep_temperature_scale & Real & ``0.0`` & Additive temperature offset in the step estimate.
   | maximum_timestep_reduction_factor & Real & ``2.0`` & Largest per-step reduction the radiation vote may impose.

.. container:: paramtable

   | Refinement controls in the ``<diffusion>`` block (adaptive runs). amr_threshold & Real & ``0.0`` & Dimensionless threshold on :math:`|\nabla T|/T` for tagging.
   | amr_min_temperature & Real & ``0.0`` & Minimum temperature for a cell to be tagged.
   | amr_min_density & Real & ``0.0`` & Minimum density for a cell to be tagged.
   | derefine_radius & Real & ``-1.0`` & Force derefinement outside this radius (:math:`\le 0` disables).

The linear solver is configured in a dedicated block; its keys are those of the Parthenon BiCGSTAB/multigrid solver.

.. container:: paramtable

   | Linear-solver parameters (``<diffusion/linear_solver_params>``). (various) & — & — & Krylov/multigrid tolerances, iteration and V-cycle limits, parsed by Parthenon’s solver classes.

Opacities and the frequency group structure are *material* properties, set in each material’s opacity block and the ``<materials>`` block respectively; see Section :ref:`sec:mat-opacity` and Section :ref:`sec:mat-group-structure`.

Registered Fields
-----------------

The package registers the fields in the table below under the ``rmg::`` (radiation-multigroup) prefix; all carry the OperatorSplit flag. The two evolved moments are ``Egroup`` (with fluxes) and the face-centered ``Fgroup``; the remainder are the diffusion coefficient, the linearized-operator scratch, the group-mean opacities, and cached geometry for the multigrid hierarchy. Each group field carries :math:`N_\nu` components. The matter fields it exchanges energy with (bulk or electron temperature and internal energy) are owned by the hydro and ionization packages.

.. container:: fieldtable

   | Principal fields registered by the radiation diffusion package.tab:raddiff-fields rmg::Egroup & :math:`E_g` & :math:`N_\nu` & Cell, Independent, FillGhost, WithFluxes, GMGRestrict, GMGProlongate, CommunicateOne, OperatorSplit; group radiation energy density.
   | rmg::Fgroup & :math:`\bm{F}_g` & :math:`N_\nu` & Face, Independent, Flux, CellMemAligned, OperatorSplit; group radiation flux.
   | rmg::D & :math:`D_g` & :math:`N_\nu` & Face, Independent, OneCopy, GMGRestrict, CellMemAligned, OperatorSplit; P1 diffusion coefficient.
   | rmg::diag_loc & — & :math:`N_\nu` & Cell, Independent, OneCopy, GMGRestrict, OperatorSplit; local diagonal of the linear operator.
   | rmg::sigma, dSdT & — & :math:`N_\nu` & Cell, Independent, OneCopy, GMGRestrict, OperatorSplit; source-linearization scratch.
   | rmg::kappa_cell & :math:`\sigma_{t,g}` & :math:`N_\nu` & Cell, Derived, OneCopy, OperatorSplit; cell group total opacity.
   | rmg::kappa_face & :math:`\sigma_{t,g}^{\text{face}}` & :math:`N_\nu` & Face, Derived, OneCopy, CellMemAligned, OperatorSplit; face group total opacity.
   | rmg::temperature0 & :math:`T^{n}` & 1 & Cell, Derived, OneCopy, OperatorSplit; matter temperature at the start of the step.
   | rmg::dTc & :math:`\Delta T` & 1 & Cell, Derived, OneCopy, OperatorSplit; Newton temperature increment.
   | rmg::face_area, DeltaX & — & 1 & Face, Derived, OneCopy, CellMemAligned, OperatorSplit; cached face area and cell spacing.
   | rmg::volume & — & 1 & Cell, Derived, OneCopy, OperatorSplit; cached cell volume.

..

   The ``flux_limit`` parameter (default ``true``) is accepted for historical reasons but does not currently alter the solve; the flux limiting in effect is the P1 relaxation :math:`1/(1+c\,\Delta t\,\sigma_{t,g})` built into the update above.

Example
-------

A grey radiation-diffusion run with a fixed-temperature boundary, driven into material ``0``:

.. code:: python

   riot.input("physics", hydro=True, multigroup_diffusion=True)

   riot.input("material0", label="mat0",
              eos_type="IdealGas", Gamma=1.6667, Cv=1.0e12,
              opac_a="constant", kappa_a=1.0e2)   # grey absorption opacity

   riot.input("diffusion",
              boundary_condition="constant_temperature",
              boundary_T=1.0e6,    # 10^6 K wall on every face
              nriter=5,
              nr_tolerance=1.0e-6)
