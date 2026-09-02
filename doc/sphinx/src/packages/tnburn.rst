.. _`chap:tnburn`:

Thermonuclear Burn
==================

The ``tnburn`` package models thermonuclear (fusion) reactions. It evaluates temperature-dependent reaction rates for a user-specified network of two-body reactions, evolves the isotopic composition, and couples the released energy back into the fluid. Reactivities and reaction energetics are supplied through tabulated nuclear data.

Governing Equations
-------------------

Burn in RIOT is *per-material*: reactions take place among the isotopes of a single material :math:`m`, driven by that material’s own isotope densities and phase/volume fractions. The isotope inventory is carried per material (the sparse field ``ccmat::iso``), and the reactions modify per-material cell-volume-averaged densities that then aggregate to the bulk density as in Section :ref:`sec:permat-bulk`. The released energy, by contrast, is deposited directly into the bulk energy equation.

Reaction Rate
~~~~~~~~~~~~~

Within material :math:`m`, for a two-body reaction :math:`r` consuming reactant isotopes :math:`1` and :math:`2`, the volumetric reaction rate is

.. math::

     R_r = \frac{\bar\rho_1}{m_1}\,\frac{\bar\rho_2}{m_2}\,
           \langle\sigma v\rangle_r(T)\,\frac{f_m^2}{f_{\text{vol},m}} ,

where :math:`\bar\rho_1,\bar\rho_2` are the reactant cell-volume-averaged densities within that material, :math:`m_1,m_2` their atomic masses, :math:`\langle\sigma v\rangle_r(T)` the temperature-dependent reactivity, :math:`f_m` the phase fraction, and :math:`f_{\text{vol},m}` the material volume fraction (the :math:`1/f_{\text{vol},m}` factor converts cell-volume-averaged densities to physical, material-intrinsic number densities). The reactivity :math:`\langle\sigma v\rangle_r(T)` is tabulated against :math:`\log T` and interpolated at run time; reactions are gated on the volume fraction exceeding ``vol_frac_thresh``.

Isotopic Composition
~~~~~~~~~~~~~~~~~~~~

Each participating per-material isotope density is sourced by every reaction it takes part in. In flux-conservation form, with :math:`\nu_{ir}` the (signed) stoichiometric coefficient of isotope :math:`i` in reaction :math:`r` (negative for reactants, positive for products),

.. math::

     \frac{\partial \bar\rho_i}{\partial t} + \nabla\!\cdot\!\left(\bar\rho_i\vec{v}\right)
       = \sum_r \nu_{ir}\,m_i\,R_r .

Summing the isotope sources within a material gives the source on that material’s cell-volume-averaged density :math:`\bar\rho_m`, and the bulk density follows by the cell-averaged sum :math:`\rho=\sum_m\bar\rho_m`; the per-reaction net :math:`\sum_i \nu_{ir}m_i` reflects the small mass defect converted to energy.

Energy Release
~~~~~~~~~~~~~~

The reaction energetics enter the total-energy equation as a source that removes the reactant input energy and deposits the product kinetic energies (both tabulated against temperature),

.. math::

     \frac{\partial E}{\partial t} + \nabla\!\cdot\!\left[\left(E + p\right)\vec{v}\right]
       = \sum_r \left[
           -R_r\,\varepsilon^{\text{in}}_r(T)
           + \sum_p R_r\,\varepsilon^{\text{out}}_{r,p}(T)\,\mu_{r,p}
         \right],

where :math:`\varepsilon^{\text{in}}_r` is the reactant input energy, :math:`\varepsilon^{\text{out}}_{r,p}` the energy carried by product :math:`p`, and :math:`\mu_{r,p}` its multiplicity. A product may be flagged to escape the problem rather than deposit locally (see ``deposit_locally_``), in which case its mass and energy are removed. The mass sources above additionally induce momentum and kinetic-energy sources :math:`\vec{v}\,(\partial_t\rho)_{\text{TN}}` and :math:`\tfrac{1}{2}|\vec{v}|^2(\partial_t\rho)_{\text{TN}}` so that the burned mass carries the local velocity.

Reaction Count
~~~~~~~~~~~~~~

A per-reaction count density :math:`Q_r` is advected as a conserved quantity sourced by the reaction rate,

.. math::

     \frac{\partial Q_r}{\partial t} + \nabla\!\cdot\!\left(Q_r\vec{v}\right) = R_r ,

providing an integrated diagnostic of the total number of reactions.

Input Parameters
----------------

Thermonuclear burn is enabled with the ``tn`` toggle in the ``<physics>`` block (Section :ref:`sec:physics-block`). Sparse physics (Chapter :ref:`chap:sparse-physics`) is not supported with burn, so ``sparse_physics`` must be set ``false``.

.. list-table:: Parameters in the ``<physics>`` block relevant to burn.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - tn
     - bool
     - ``false``
     - Enable thermonuclear burn.
   * - sparse_physics
     - bool
     - ``true``
     - Not supported with burn; set ``false``.

The reaction network is specified in the ``<tnburn>`` block by contiguous, zero-based ``reaction``\ :math:`K` entries in NDI reaction notation. Products may be individually flagged to deposit locally or escape.

.. list-table:: Parameters in the ``<tnburn>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - reaction\ :math:`K`
     - string
     - —
     - The :math:`K`\ th reaction, e.g. ``d+t->n+a``; numbered from ``0``.
   * - deposit_locally\_\ :math:`Z`
     - bool
     - ``true``
     - Deposit product with ZAID :math:`Z` locally; if ``false`` it escapes the problem.

The nuclear data table (isotope masses, charges, reactivities, and energetics) is selected in the ``<isotope_data>`` block:

.. list-table:: Parameters in the ``<isotope_data>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - filename
     - string
     - ``isotope_data.hdf5``
     - HDF5 nuclear-data file (produced by the ``ndi2spiner`` tool).

Each burning material lists its isotopes by ZAID in its ``<material>`` block via ``isotope``\ :math:`K` and initial mass fractions ``isotope``\ :math:`K`\ ``_mfrac`` (see Chapter :ref:`chap:materials`).

Registered Fields
-----------------

The burn package registers the per-reaction count fields in the table below; the isotope densities it evolves (``ccmat::iso``) are registered by the materials package when isotopes are present. All are sparse and carry the burn flag.

.. list-table:: Fields registered by the thermonuclear burn package.
   :class: wraptable
   :header-rows: 1
   :widths: 30 14 16 40
   :name: tab:tnburn-fields

   * - Field
     - Symbol
     - Components
     - Metadata / description
   * - ccmat::tn_reaction_density
     - :math:`Q_r`
     - :math:`N_{\text{rxn}}`
     - Cell, Independent, Intensive, Conserved, Sparse, FillGhost, Advected, WithFluxes, BurnFlag; per-reaction count density.
   * - cm::tn_specific_reactions
     - —
     - :math:`N_{\text{rxn}}`
     - Cell, Derived, OneCopy, Sparse, BurnFlag; specific reaction rate.
   * - ccmat::iso
     - :math:`\bar\rho_i`
     - :math:`N_{\text{iso}}`
     - Cell, Independent, Intensive, Conserved, Sparse, FillGhost, WithFluxes, Advected, BurnFlag; isotope partial densities (via materials).

Example
-------

A deuterium–tritium material burning via the DT reaction, with neutrons removed from the problem:

.. code:: python

   riot.input("physics", tn=True, sparse_physics=False)
   riot.input("material0", label="DT",
              eos_type="IdealGas", Gamma=1.4, Cv=3.0e12,
              isotope0=1002,   # deuterium
              isotope1=1003,   # tritium
              isotope2=2004,   # helium-4 (alpha)
              isotope3=1)      # neutron
   riot.input("tnburn", reaction0="d+t->n+a",
              deposit_locally_1=False)  # neutrons escape
   riot.input("isotope_data", filename="isotope_data.hdf5")
