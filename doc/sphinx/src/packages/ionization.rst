.. _`chap:ionization`:

Ionization
==========

The ``ionization`` package extends RIOT to partially ionized plasmas with a two-temperature (2T) description in which the ions and free electrons carry separate temperatures and internal energies. Each material must supply a separate electron equation of state.

Enabling ionization does two distinct things, and the organization of this chapter follows that split. First, it *establishes the 2T plasma state*: the mean ionization state :math:`\bar{Z}`, the free-electron number density :math:`n_e`, and a separate electron energy and temperature :math:`T_e` alongside the ion temperature :math:`T_i`. No other package produces these quantities. Second, it provides a family of optional *transport processes defined on that state* — electron–ion energy exchange, electron and ion thermal conduction, and ion viscosity. These share a common character: each is an independent physical process with its own governing equation, but each closes on the *same* plasma variables (:math:`n_e`, :math:`T_e`, :math:`T_i`, :math:`\bar{Z}`), which is why they live in one package and one input block rather than in separate packages. Each is turned on by its own switch in the ``<ionization>`` block and, having no plasma state to act on otherwise, requires ionization to be enabled.

The subsections below treat the state-defining physics first (energy/pressure splitting and the mean ionization state), then each transport process in turn, each with the specific equation it solves.

Governing Equations
-------------------

Energy and Pressure Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ionization is a *per-material* property closed on a shared electron temperature. Each material :math:`m` carries its own mean ionization state :math:`\bar{Z}_m`, electron internal energy :math:`u_{e,m}`, and electron pressure :math:`p_{e,m}`; these aggregate to bulk electron quantities that split the bulk energy and pressure,

.. math::

     E = u_i + u_e + \tfrac{1}{2}\rho|\vec{v}|^2, \qquad
     p = p_i + p_e .

Here :math:`u_i`, :math:`u_e` are the bulk ion and electron internal-energy densities and :math:`p_i`, :math:`p_e` the corresponding bulk pressures. Following the aggregation rules of Section :ref:`sec:permat-bulk`, the bulk electron energy is a cell-averaged sum of per-material electron energy densities while the bulk electron pressure is a volume-fraction-weighted sum,

.. math::

     u_e = \sum_m \bar\rho_m\,e_{e,m}(\rho_m, T_e), \qquad
     p_e = \sum_m f_m\,p_{e,m}(\rho_m, T_e),

with each material’s electron specific energy :math:`e_{e,m}` and pressure :math:`p_{e,m}` supplied by its own electron equation of state.

Crucially, the electron temperature :math:`T_e` is a *single bulk* quantity shared by all materials: it is found by a root solve enforcing that the per-material electron energies sum to the transported bulk electron energy,

.. math::

     u_e = \sum_m \bar\rho_m\,e_{e,m}(\rho_m, T_e),

which is the electron-side analogue of the PTE temperature equilibrium.

Electron Energy Transport
~~~~~~~~~~~~~~~~~~~~~~~~~

By default the electron internal energy is advected with the flow and does :math:`p\,\mathrm{d}V` work against the electron pressure,

.. math::

     \frac{\partial u_e}{\partial t} + \nabla\!\cdot\!\left(u_e\vec{v}\right) = -\,p_e\,\nabla\!\cdot\!\vec{v} .

Alternatively the package can instead advance a conserved electron *entropy* density :math:`s_e` (which assumes an ideal electron gas and avoids the volumetric work source),

.. math::

     \frac{\partial s_e}{\partial t} + \nabla\!\cdot\!\left(s_e\vec{v}\right) = 0 ,

converting between :math:`s_e` and :math:`u_e` as needed. The choice is controlled by ``advect_electron_entropy``.

Electron–Ion Energy Exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``electron_ion_coupling`` is enabled, the (bulk) electron and ion temperatures relax toward a common value on a coupling time scale :math:`\tau_{ei}`,

.. math::

     \rho c_{v,i}\,\frac{\partial T_i}{\partial t} &= \frac{T_e - T_i}{\tau_{ei}}, \\
     \rho c_{v,e}\,\frac{\partial T_e}{\partial t} &= \frac{T_i - T_e}{\tau_{ei}} ,

integrated with a first-order exponential (analytic) update so the equilibrium :math:`T_e = T_i` is recovered stably for any step size. In a mixed cell the heat capacities :math:`\rho c_{v,i}`, :math:`\rho c_{v,e}` and the coupling rate are formed as volume-fraction-weighted sums over the per-material contributions (evaluated at :math:`T_e`, :math:`T_i` and each material’s :math:`\rho_m`, :math:`\bar{Z}_m`), so a single bulk relaxation represents the whole cell. The coupling time may be a user-supplied constant (``tau_ei``) or the Landau–Spitzer form (Blancard et al., *High Energy Density Phys.* **9**, 247, 2013),

.. math::

     \tau_{ei} = \frac{3\,m_e m_i}{8\sqrt{2\pi}\,n_i \bar{Z}^2 e^4}\,
                 \frac{\left(k_B T_e/m_e + k_B T_i/m_i\right)^{3/2}}{\ln\Lambda} ,

with the Coulomb logarithm :math:`\ln\Lambda` selected by ``coulomb_logarithm``.

Mean Ionization State
~~~~~~~~~~~~~~~~~~~~~

The mean ionization state is computed *per material*: for each material :math:`m`, :math:`\bar{Z}_m` is evaluated from the Thomas–Fermi statistical model with the fit of More (UCRL-84991, 1981; see Salzmann, *Atomic Physics in Hot Plasmas*, 1998), using that material’s atomic number :math:`Z_{\text{nuc},m}`, atomic mass, and material-averaged density :math:`\rho_m`, evaluated at the shared bulk electron temperature :math:`T_e`,

.. math::

     \bar{Z}_m = Z_{\text{nuc},m}\,f(x_m), \qquad x_m = \alpha\,Q_m^{\beta}.

The per-material ionization states set the free-electron number density :math:`n_e = \sum_m \bar\rho_m\,\bar{Z}_m/m_{\text{nuc},m}`. Setting ``fully_ionized`` forces :math:`\bar{Z}_m = Z_{\text{nuc},m}`.

Thermal Conduction
~~~~~~~~~~~~~~~~~~

Optionally, electron and/or ion heat conduction are solved as implicit diffusion equations on the shared bulk temperature,

.. math::

     \rho c_v\,\frac{\partial T}{\partial t} = \nabla\!\cdot\!\left(\kappa\,\nabla T\right) ,

using a BiCGSTAB linear solve. In a mixed cell the effective conductivity is a volume-fraction average (arithmetic or harmonic, selectable) of the per-material conductivities :math:`\kappa_m`, each evaluated at the shared bulk temperature with that material’s :math:`\rho_m` and :math:`\bar{Z}_m`. The per-material electron conductivity :math:`\kappa_{e,m}` defaults to the Spitzer–Härm form (Molvig, Simakov & Vold, *Phys. Plasmas* **21**, 092709, 2014),

.. math::

     \kappa_{e,m} = \frac{\alpha(\bar{Z}_m)}{\bar{Z}_m}\,\frac{8}{\pi^{3/2}}\,
                \frac{k_B^{7/2}}{e^4\sqrt{m_e}}\,\frac{T_e^{5/2}}{\ln\Lambda},
     \qquad \alpha(\bar{Z}_m) = \frac{1}{1 + 3.3/\bar{Z}_m},

and the per-material ion conductivity :math:`\kappa_{i,m}` to the Braginskii form. Either may instead be set to a constant.

Plasma Viscosity
~~~~~~~~~~~~~~~~

Optionally (``plasma_viscosity``), an ion viscous stress :math:`\mathsf{\sigma}_{\text{visc}}` is added to the bulk momentum and total-energy equations of Chapter :ref:`chap:hydro` as a divergence source,

.. math::

     \frac{\partial \left(\rho\vec{v}\right)}{\partial t}
       + \nabla\!\cdot\!\left(\rho\vec{v}\otimes\vec{v}+ p\,\mathsf{I}\right)
       &= \nabla\!\cdot\!\mathsf{\sigma}_{\text{visc}}, \\[2pt]
     \frac{\partial E}{\partial t}
       + \nabla\!\cdot\!\left[\left(E + p\right)\vec{v}\right]
       &= \nabla\!\cdot\!\left(\mathsf{\sigma}_{\text{visc}}\!\cdot\!\vec{v}\right),

so momentum diffuses and the associated viscous dissipation heats the fluid. The stress is Newtonian in the (bulk) strain rate :math:`\mathsf{e}` (Chapter :ref:`chap:strength`),

.. math::

     \mathsf{\sigma}_{\text{visc}} = 2\eta\,\mathsf{e}
       + \left(\eta_b - \tfrac{2}{3}\eta\right)(\nabla\!\cdot\!\vec{v})\,\mathsf{I},

with :math:`\eta` the shear viscosity and :math:`\eta_b` the bulk viscosity. The shear viscosity is set by ``ion_viscosity_model``: either a user-supplied constant (``ion_shear_viscosity``, ``ion_bulk_viscosity``), or the Fokker–Planck–Landau plasma model (Arnault, *High Energy Density Phys.* **9**, 711, 2013; Vold et al., *Phys. Plasmas* **24**, 042702, 2017),

.. math::

     \eta = \sum_i \frac{\alpha_{ij}\,n_i\,k_B T_i}{\sum_j \nu_{ij}},

where the ion–ion momentum-exchange rates :math:`\nu_{ij}` are summed over the ion species in the cell, :math:`n_i` is the ion number density, and :math:`T_i` the ion temperature. This is a bulk (whole-cell) transport coefficient built from the per-material ion states, in the same spirit as the conductivities above.

Input Parameters
----------------

Ionization is enabled with the ``ionization`` toggle in the ``<physics>`` block (Section :ref:`sec:physics-block`). When it is on, every material must provide an ``electron_eos`` block (Chapter :ref:`chap:materials`). The remaining controls live in the ``<ionization>`` block.

.. list-table:: Parameters in the ``<ionization>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - fully_ionized
     - bool
     - ``false``
     - Force :math:`\bar{Z} = Z_{\text{nuc}}` (fully ionized).
   * - root_tol
     - Real
     - ``1e-20``
     - Root-find tolerance for the electron temperature.
   * - advect_electron_entropy
     - bool
     - ``false``
     - Advect electron entropy density instead of solving the electron energy with a :math:`p\,\mathrm{d}V` source.
   * - electron_ion_coupling
     - bool
     - ``false``
     - Relax :math:`T_e` and :math:`T_i` toward equilibrium.
   * - electron_ion_coupling_model
     - string
     - ``landau_spitzer``
     - Coupling model: ``constant`` or ``landau_spitzer``.
   * - tau_ei
     - Real
     - ``0.0``
     - Constant coupling time (used when model is ``constant``).
   * - coulomb_logarithm
     - string
     - ``brysk``
     - :math:`\ln\Lambda` model: ``basic``, ``brysk``, ``lee_moore``, or ``bps``.
   * - electron_thermal_conduction
     - bool
     - ``false``
     - Solve electron heat conduction.
   * - electron_conductivity_model
     - string
     - ``spitzer_volume_average_arithmetic``
     - Electron conductivity model (Spitzer variants or ``constant``).
   * - electron_conductivity
     - Real
     - ``0.0``
     - Constant electron conductivity (if selected).
   * - ion_thermal_conduction
     - bool
     - ``false``
     - Solve ion heat conduction.
   * - ion_conductivity_model
     - string
     - ``braginskii``
     - Ion conductivity model: ``braginskii`` or ``constant``.
   * - ion_conductivity
     - Real
     - ``0.0``
     - Constant ion conductivity (if selected).
   * - plasma_viscosity
     - bool
     - ``false``
     - Add the ion viscous stress to the momentum and energy equations.
   * - ion_viscosity_model
     - string
     - ``fokker_planck_landau``
     - Viscosity model: ``fokker_planck_landau`` or ``constant``.
   * - ion_shear_viscosity
     - Real
     - ``1.0``
     - Constant shear viscosity :math:`\eta` (if model is ``constant``).
   * - ion_bulk_viscosity
     - Real
     - ``0.0``
     - Constant bulk viscosity :math:`\eta_b` (if model is ``constant``).
   * - timestep_control
     - string
     - ``relative``
     - Conduction step control: ``explicit`` or ``relative``.
   * - T_scale_floor
     - Real
     - ``1.0``
     - Temperature-scale floor for ``relative`` step control.
   * - fractional_change_scale
     - Real
     - ``0.1``
     - Allowed fractional :math:`T_e` change per conduction step.
   * - zbar_floor
     - Real
     - ``1e-6``
     - Floor on :math:`\bar{Z}` in microphysics evaluations.
   * - ion_number_density_floor
     - Real
     - ``1e11``
     - Floor on ion number density (cm\ :math:`^{-3}`).

Registered Fields
-----------------

The ionization package registers the bulk electron fields in the table below, and (through the materials package) the per-material :math:`\bar{Z}` and electron-energy fields. The transported electron quantity is either ``electron_internal_energy`` (the default) or, when ``advect_electron_entropy`` is set, ``electron_entropy``; the remaining bulk electron fields are derived. The implicit conduction solve additionally allocates operator-split auxiliary fields (diffusion coefficients, temperature deltas) that are internal to the solver and omitted here.

.. list-table:: Principal fields registered by the ionization package.
   :class: wraptable
   :header-rows: 1
   :widths: 30 14 16 40
   :name: tab:ionization-fields

   * - Field
     - Symbol
     - Components
     - Metadata / description
   * - ccbulk::electron_internal_energy
     - :math:`u_e`
     - 1
     - Cell, Independent, Intensive, FillGhost, Advected, WithFluxes; electron energy density (default transported quantity).
   * - ccbulk::electron_entropy
     - :math:`s_e`
     - 1
     - Cell, Independent, Intensive, FillGhost, Advected, WithFluxes, Conserved; electron entropy density (when ``advect_electron_entropy``).
   * - ccbulk::electron_temperature
     - :math:`T_e`
     - 1
     - Cell, Intensive, Derived, OneCopy; electron temperature.
   * - ccbulk::electron_pressure
     - :math:`p_e`
     - 1
     - Cell, Intensive, Derived, OneCopy; electron pressure.
   * - ccbulk::electron_number_density
     - :math:`n_e`
     - 1
     - Cell, Intensive, Derived, OneCopy; free-electron number density.
   * - ccbulk::electron_bulk_modulus
     - :math:`B_e`
     - 1
     - Cell, Intensive, Derived, OneCopy; electron bulk modulus.
   * - ccbulk::electron_gruneisen_parameter
     - :math:`\Gamma_e`
     - 1
     - Cell, Intensive, Derived, OneCopy; electron Grüneisen parameter.
   * - ccbulk::ion_shear_viscosity
     - :math:`\eta`
     - 1
     - Cell, Intensive, Derived, OneCopy; ion shear viscosity (when ``plasma_viscosity``).
   * - cm::ionization_zbar
     - :math:`\bar{Z}`
     - 1
     - Cell, Sparse, Derived, OneCopy, FillGhost; per-material mean ionization state.

Example
-------

A single ionized material with electron–ion coupling and electron conduction:

.. code:: python

   riot.input("physics", ionization=True)
   riot.input("ionization",
              electron_ion_coupling=True,
              electron_ion_coupling_model="landau_spitzer",
              coulomb_logarithm="brysk",
              electron_thermal_conduction=True)
   riot.input("material0", label="plasma",
              eos_type="IdealGas", Gamma=1.66667, Cv=1.0e12,
              electron_eos="plasma_electrons")
   riot.input("plasma_electrons", eos_type="IdealElectrons")
