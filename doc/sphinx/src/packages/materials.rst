.. _`chap:materials`:

Materials and Equations of State
================================

The ``materials`` package defines the materials present in a simulation and assigns each one an equation of state (EOS). It registers the per-material fields used by the multi-material hydrodynamics of Chapter :ref:`chap:hydro` (per-material densities, volume fractions, internal energies, and the material-averaged thermodynamic state) and configures the pressure–temperature equilibrium (PTE) solver used to close mixed cells. It also owns the optional per-material data consumed by other packages: strength models, opacities, ionization electron states, and burn isotopes.

Governing Equations
-------------------

The materials package does not itself advance a transport equation; it supplies the thermodynamic closure for the conservation laws solved elsewhere. Each material :math:`m` carries an equation of state that provides the pressure and specific internal energy as functions of the material-averaged density :math:`\rho_m` and temperature :math:`T`,

.. math::

   p_m & = p_m(\rho_m, T) \\
   e_m & = e_m(\rho_m, T)

supplied through the ``singularity-eos`` library. These relations close the per-material system and, together with the Amagat volume-additive constraint and the PTE conditions of Chapter :ref:`chap:hydro`, determine the equilibrium state of a mixed cell:

.. math::

   p_m(\rho_m,T) & = p \\
   T_m & = T \\
   \sum_m f_m & = 1 \\
   u & = \sum_m \bar\rho_m\,e_m(\rho_m,T)

for all materials :math:`m`, where :math:`u` is the bulk volumetric internal energy (internal energy per unit cell volume). For a cell containing a single material the EOS is evaluated directly; for a mixed cell the closure is solved by ``singularity-eos``. A mixed cell whose materials are all ideal gases admits a closed-form solution and bypasses the iterative solver unless ``use_general_pte`` is set.

.. _`sec:permat-bulk`:

Per-Material and Bulk Quantities
--------------------------------

The multi-material design of RIOT rests on a consistent distinction between three kinds of field, identified throughout the source (and this manual) by their namespace prefix. This section defines that distinction and the aggregation rules once; every subsequent chapter refers back to it. The prefixes are abbreviations of the fully-qualified namespaces in ``src/variables.hpp``:

``ccbulk::`` (``cell_variables::cell_averaged::bulk``)
  Cell-averaged *bulk* quantities: a single value per cell that describes the
  cell as a whole. The materials in a cell share one common velocity
  ``ccbulk::velocity``; the other bulk fields are aggregates that every material
  contributes to. For example, the bulk density ``ccbulk::rho`` is the sum of the
  per-material ``ccmat::rho``, and the bulk momentum ``ccbulk::momentum`` is that
  bulk density times the common velocity. RIOT evolves the bulk momentum and
  total energy (``ccbulk::total_material_energy``) directly.

``ccmat::`` (``cell_variables::cell_averaged::mat``)
  Cell-averaged *per-material* quantities. These are averaged over the whole
  cell volume, so a material that only partly fills a cell contributes in
  proportion to its volume fraction.

``cm::`` (``cell_variables::material_averaged``)
  *Material-averaged* per-material quantities: the intrinsic (physical) value a
  material carries within the sub-volume it actually occupies.

The distinction between ``ccmat::`` and ``cm::`` is fundamental. Consider a cell of volume :math:`V` and a material :math:`m` within it that has mass :math:`M_m` and occupies volume :math:`V_m`. Its *volume fraction* is :math:`f_m = V_m/V` (field ``ccmat::volume_fraction``), with :math:`\sum_m f_m = 1`. Then:

- :math:`\rho_m = M_m/V_m` — the *physical density* of the material, i.e. its mass divided by the volume it actually occupies (field ``cm::rho``). This is the density passed to the equation of state.

- :math:`\bar\rho_m = M_m/V` — the *cell-volume-averaged density* of the material, i.e. its mass divided by the *whole* cell volume rather than the sub-volume it occupies (field ``ccmat::rho``).

The two are related through the volume fraction,

.. math::

   \bar\rho_m = \rho_m\,\frac{V_m}{V} = f_m\,\rho_m

so ``ccmat::rho`` is always :math:`\le` ``cm::rho``, with equality only when the material fills the cell (:math:`f_m = 1`). The material-averaged pressure, temperature, specific internal energy, bulk modulus, and so on (:math:`p_m`, :math:`T_m`, :math:`e_m`, :math:`B_m`, …) are likewise ``cm::`` quantities, evaluated from the equation of state at the physical density :math:`\rho_m`.

RIOT integrates conservation laws for the cell-volume-averaged material densities :math:`\bar\rho_m` (``ccmat::rho``, one per material) together with the bulk momentum and total energy (``ccbulk::``); a single velocity :math:`\vec{v}` is common to all materials in a cell. After each update, the bulk quantities are reconstructed from the per-material states. The aggregation rules actually used (in ``multiphysics/fill_shared_derived.cpp``) fall into two kinds:

- **Direct sums.** Because ``ccmat::`` densities are already per-unit-cell-volume,
  the bulk density and bulk volumetric internal energy :math:`u` (internal energy
  per unit cell volume) are direct sums over materials:

  .. math::

     \rho & = \sum_m \bar\rho_m \\
     u & = \sum_m \bar\rho_m\,e_m

- **Equilibrium closure.** The bulk pressure :math:`p` and temperature :math:`T`
  are *not* averages: they are the common values returned by the
  pressure–temperature-equilibrium (PTE) solve, :math:`p_m(\rho_m,T)=p` and
  :math:`T_m=T` for all :math:`m` (Chapter :ref:`chap:hydro`).

The bulk velocity follows from the conserved momentum, :math:`\vec{v}= (\rho\vec{v})/\rho`.

This pattern — *per-material fields evolved or closed independently, then aggregated to a bulk field that feeds the hydrodynamics* — recurs in the strength, ionization, and burn packages, and each of those chapters states its own aggregation explicitly.

Equation-of-State Models
------------------------

Each material selects an EOS through the ``eos_type`` parameter. The available models and the parameters each requires are listed in the table below. All EOS blocks additionally accept ``mean_atomic_mass`` (default ``2.0``) and ``mean_atomic_number`` (default ``1.0``).

.. list-table:: Equation-of-state models.
   :header-rows: 1
   :widths: 27 19 54

   * - ``eos_type``
     - Model
     - Required parameters
   * - ``IdealGas``
     - Ideal gas (:math:`\gamma`-law)
     - ``Gamma``, ``Cv``
   * - ``Gruneisen``
     - Mie–Grüneisen
     - ``C0``, ``s1``, ``s2``, ``s3``, ``G0``, ``b``, ``rho0``, ``T0``,
       ``P0``, ``Cv``
   * - ``SpinerEOSDependsRhoT``
     - Tabular (SESAME), :math:`(\rho,T)` independent
     - ``filename``, ``sesame_id``, or ``sesame_name``
   * - ``SpinerEOSDependsRhoSie``
     - Tabular (SESAME), :math:`(\rho,e)` independent
     - ``filename``, ``sesame_id``, or ``sesame_name``
   * - ``IdealElectrons``
     - Ideal electron gas (ionization)
     - —

The analytic models (``IdealGas``, ``Gruneisen``) are provided directly by ``singularity-eos``. The tabular ``Spiner`` models read an HDF5 (SESAME/SP5) table selected either by numeric ``sesame_id`` (which takes precedence) or by ``sesame_name``, and accept the optional flags ``reproducibility_mode`` (default ``false``) and ``use_subtable`` (default ``false``).

Input Parameters
----------------

Defining Materials
~~~~~~~~~~~~~~~~~~

Materials are defined by contiguous, zero-based blocks ``<material0>``, ``<material1>``, …. The number of materials is set implicitly by how many such blocks are present; they must be numbered contiguously starting from ``0``. Each block selects an EOS and, optionally, enables strength, ionization, opacity, isotopes, or multiple phases.

.. list-table:: Per-material parameters in each ``<material``\ :math:`N`\ ``>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - label
     - string
     - —
     - Descriptive name for the material.
   * - eos_type
     - string
     - —
     - EOS model (the table below); required.
   * - eos
     - string
     - *this block*
     - Name of a separate block holding the EOS parameters; if absent, they are read from this material block.
   * - nphase
     - int
     - ``1``
     - Number of phases for a multi-phase material.
   * - max_mat_level
     - int
     - ``0``
     - Maximum AMR level to enforce within this material (:math:`-1` = finest).
   * - max_bnd_level
     - int
     - ``0``
     - Maximum AMR level to enforce around its interfaces.
   * - strong
     - bool
     - ``false``
     - Enable material strength (Chapter :ref:`chap:strength`).
   * - strength_model
     - string
     - —
     - Strength-model block (required if ``strong``).
   * - electron_eos
     - string
     - —
     - Electron-EOS block (required if ionization is on).
   * - opac
     - string
     - *this block*
     - Name of a separate block holding the opacity models (Section :ref:`sec:mat-opacity`); if absent, they are read from this material block. Multi-phase materials use ``opac0``, ``opac1``, …instead.
   * - isotope\ :math:`K`
     - string
     - —
     - ZAID of the :math:`K`\ th isotope (burn runs).
   * - isotope\ :math:`K`\ \_mfrac
     - Real
     - ``0.0``
     - Initial mass fraction of the :math:`K`\ th isotope.

Multi-Material Closure Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The global ``<materials>`` block and PTE controls tune how mixed cells are closed and diagnosed. The PTE tolerances apply when the general (iterative) solver is active; where the default column reads *library* below, the parameter is left unset and takes whatever default the ``singularity-eos`` library assigns it.

.. list-table:: Parameters in the ``<materials>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - use_general_pte
     - bool
     - ``false``
     - Use the iterative PTE solver even for cells of all ideal gases.
   * - track_pte_statistics
     - bool
     - ``false``
     - Report PTE-solver convergence diagnostics.
   * - pte_stats_mode
     - string
     - ``averaged_light``
     - Statistics mode: ``instantaneous``, ``averaged``, or ``averaged_light``.
   * - pte_max_iter_per_mat
     - int
     - *library*
     - Maximum PTE iterations per material.
   * - pte_rel_tolerance_p
     - Real
     - *library*
     - Relative pressure-residual tolerance.
   * - pte_rel_tolerance_e
     - Real
     - *library*
     - Relative energy-residual tolerance.
   * - pte_abs_tolerance_p
     - Real
     - *library*
     - Absolute pressure-residual tolerance.

.. _`sec:mat-group-structure`:

Frequency Group Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

For radiation runs the materials package *owns* the frequency group structure — the number of groups :math:`N_\nu` and the :math:`N_\nu+1` group boundaries (in Hz) — and supplies it to the radiation packages (Chapter :ref:`chap:radtransport`). It lives here, rather than in the radiation package, because the opacities are a material property (Section :ref:`sec:mat-opacity`) and the transport must use exactly the group grid on which the opacities are tabulated. The structure is resolved at initialization from the first of the following that applies:

#. **Explicit input.** If ``group_bounds`` is given in the ``<materials>`` block (a list of :math:`N_\nu+1` ascending frequency edges in Hz), it sets the structure directly. An optional ``ngroups`` may be given but must equal ``len(group_bounds)`` :math:`-\,1`.

#. **Opacity tables.** Otherwise, any tabular (``table``) opacity model is read and its built-in group structure adopted. All table-based materials must agree on the structure, or RIOT aborts.

#. **Grey default.** If neither is present — as in any non-radiating run — the structure defaults to a single group spanning :math:`[0,\infty)`.

All materials in a simulation therefore share one common group grid, checked for consistency across every table-based material.

.. list-table:: Group-structure parameters in the ``<materials>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - group_bounds
     - list
     - *tables/grey*
     - Ascending frequency-group edges (Hz), length :math:`N_\nu+1`. If omitted, taken from opacity tables, or a single grey group.
   * - ngroups
     - int
     - *derived*
     - Number of frequency groups; if given with ``group_bounds`` it must equal its length minus one.

.. _`sec:mat-opacity`:

Opacity Models
~~~~~~~~~~~~~~

For radiation runs (Chapter :ref:`chap:radtransport`) each material carries an *absorption* and a *scattering* opacity model, selected in its opacity block. By default this is the material block itself; the ``opac`` parameter (or ``opac``\ :math:`N` per phase, Section :ref:`sec:permat-bulk`) may redirect to a separately-named block so several materials can share one definition. The model is chosen with ``opac_a`` (absorption) and ``opac_s`` (scattering); both default to ``none``. RIOT builds a *multigroup group-mean* opacity table for every material — either by integrating the chosen analytic model over each frequency group (Section :ref:`sec:mat-group-structure`) or by reading a pre-tabulated SP5 (HDF5) file — so a grey run is simply the single-group case of the same machinery. All opacity input is interpreted in CGS.

.. code-block:: text

   @P0.16 P0.14 L0.56@ **Option & Applies to & Model and parameters
   none & abs., scat. & No opacity (zero coefficient). Default.
   constant & abs., scat. & Grey (frequency-independent) :math:`\kappa`: ``kappa_a`` (absorption) or ``kappa_s`` (scattering).
   powerlaw & abs. only & Power-law :math:`\kappa = \kappa_0\,\rho^{a}\,T^{b}\,(\nu/\nu_{\text{ref}})^{c}`: ``kappa0_a``, ``kappa_Rhopower_a`` (:math:`a`), ``kappa_Tpower_a`` (:math:`b`), ``kappa_Nupower_a`` (:math:`c`), ``kappa_Nuref_a`` (:math:`\nu_{\text{ref}}`).
   table & abs., scat. & Pre-tabulated group-mean opacity read from an SP5 (HDF5) file: ``opac_a_filename`` / ``opac_s_filename``. The file also carries the group structure, from which the run’s group grid can be inferred (Section \ \ **\ :ref:`sec:mat-group-structure`\ **\ \ ).
   **

For an analytic model (``constant``/``powerlaw``) RIOT generates the group-mean table at build time over a :math:`(\rho,T)` grid whose extent and resolution are set by the optional per-model parameters below (suffixed ``_a`` for absorption, ``_s`` for scattering); the defaults suffice when the model is density/temperature independent.

.. list-table:: Optional table-generation parameters in an opacity block (``_a``/``_s`` suffix).
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - lRhoMin, lRhoMax
     - Real
     - ``-1.0``, ``1.0``
     - :math:`\log_{10}` density bounds of the generated mean table.
   * - lTMin, lTMax
     - Real
     - ``-1.0``, ``1.0``
     - :math:`\log_{10}` temperature bounds of the generated mean table.
   * - NRho, NT
     - int
     - ``2``, ``2``
     - Number of density / temperature table entries.
   * - NNuPerGroup
     - int
     - ``64``
     - Frequency samples per group used in the group-mean integration.

The averaging weight (Rosseland vs. Planck) and the per-material evaluation inside a cell are described in Chapter :ref:`chap:singularity-opac`; the way the per-material coefficients combine into the cell coefficients used by the transport solve is given in Section :ref:`sec:rad-opacity`.

Registered Fields
-----------------

The materials package registers the per-material fields listed in the table below. These are *sparse*: they are allocated only in cells where the material is present, controlled by ``ccmat::rho``. Recall (Section :ref:`sec:permat-bulk`) that ``ccmat::`` denotes a cell-volume-averaged per-material quantity while ``cm::`` denotes a material-averaged (physical) quantity, related by the volume fraction as in the equation above. Additional per-material fields are registered here when strength, ionization, or burn are active, and are documented in those chapters.

.. container:: fieldtable

   | Core per-material fields registered by the materials package.tab:materials-fields ccmat::rho & :math:`\bar\rho_m` & 1 & Cell, Independent, Intensive, Conserved, Sparse, FillGhost, WithFluxes; cell-volume-averaged material density, :math:`M_m/V`.
   | ccmat::volume_fraction & :math:`f_m` & 1 & Cell, Intensive, Sparse, Derived, OneCopy, FillGhost, ForceRemeshComm, Restart; volume fraction :math:`V_m/V`.
   | ccmat::internal_energy & :math:`\bar\rho_m e_m` & 1 & Cell, Intensive, Sparse, Derived, OneCopy; cell-volume-averaged internal-energy density.
   | cm::rho & :math:`\rho_m` & 1 & Cell, Intensive, Sparse, Derived, OneCopy; physical (material-averaged) density :math:`M_m/V_m = \bar\rho_m/f_m`.
   | cm::sie & :math:`e_m` & 1 & Cell, Intensive, Sparse, Derived, OneCopy; specific internal energy.
   | cm::temperature & :math:`T_m` & 1 & Cell, Sparse, Derived, OneCopy; material temperature.
   | cm::pressure & :math:`p_m` & 1 & Cell, Sparse, Derived, OneCopy; material pressure.
   | cm::bulk_modulus & :math:`B_m` & 1 & Cell, Sparse, Derived, OneCopy; material bulk modulus.

Example
-------

Two ideal-gas materials with EOS parameters given inline:

.. code:: python

   riot.input("material0", label="air",
              eos_type="IdealGas", Gamma=1.4, Cv=1.0e-3)
   riot.input("material1", label="heavy",
              eos_type="IdealGas", Gamma=1.66667, Cv=1.0e-3)
