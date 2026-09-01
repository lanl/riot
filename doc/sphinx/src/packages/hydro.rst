.. _`chap:hydro`:

Hydrodynamics
=============

The ``hydro`` package integrates the multi-material compressible Euler equations in conservative form. It is the core solver of RIOT: it performs reconstruction, evaluates interface fluxes with a Riemann solver, and applies the geometric source terms of curvilinear coordinates. It couples to the material strength, level-set, ionization, gravity, and other packages when those are enabled.

Governing Equations
-------------------

RIOT treats each cell as containing one or more materials. Before writing the conservation laws we fix the per-material decomposition and its notation; the per-material/bulk conventions used throughout are defined in Section :ref:`sec:permat-bulk`.

Amagat (Volume-Additive) Closure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each material :math:`m` carries a cell-volume-averaged density :math:`\bar\rho_m` (``ccmat::rho``, its mass per unit *cell* volume) and a volume fraction :math:`f_m` (Section :ref:`sec:permat-bulk`). RIOT uses the Amagat volume-additive closure, which assumes the materials occupy *distinct* sub-volumes of the cell — they do not interpenetrate, so no two materials share the same physical volume. Their volumes then partition the cell and sum to fill it,

.. math::

   \begin{equation}
     \sum_m f_m = 1 .
   \end{equation}

The physical (material-averaged) density is :math:`\rho_m = \bar\rho_m / f_m` (``cm::rho``), and the bulk density is the sum of the cell-volume-averaged densities,

.. math::

   \begin{equation}
     \rho= \sum_m \bar\rho_m = \sum_m f_m\,\rho_m .
   \end{equation}

The bulk volumetric internal energy :math:`u` (internal energy per unit cell volume) is the sum over materials of the per-material internal-energy densities,

.. math::

   \begin{equation}
     u = \sum_m \bar\rho_m\,e_m ,
   \end{equation}

where :math:`e_m` is the specific internal energy of material :math:`m`. The bulk velocity is :math:`\bm{v}= (\rho\bm{v})/\rho`, and the bulk pressure is set by the equilibrium closure of Section :ref:`sec:hydro-pte`.

Conservation Laws
~~~~~~~~~~~~~~~~~

The quantities that are actually integrated in time (advected with the interface fluxes) are the per-material cell-volume-averaged densities :math:`\bar\rho_m` and the *bulk* momentum and total energy. These obey

.. math::

   \begin{align}
     \frac{\partial \bar\rho_m}{\partial t} + \nabla\!\cdot\!\left(\bar\rho_m\bm{v}\right) &= 0,
       \qquad m = 1,\dots,N_{\text{mat}}, \\[2pt]
     \frac{\partial \left(\rho\bm{v}\right)}{\partial t}
       + \nabla\!\cdot\!\left(\rho\bm{v}\otimes\bm{v}+ p\,\bm{I} - \bm{s}\right)
       &= \bm{0}, \\[2pt]
     \frac{\partial E}{\partial t}
       + \nabla\!\cdot\!\left[\left(E + p\right)\bm{v}- \bm{s}\!\cdot\!\bm{v}\right]
       &= 0,
   \end{align}

where :math:`\bm{v}` is the single velocity common to all materials in the cell, :math:`p` the bulk pressure, :math:`E = u + \tfrac{1}{2}\rho|\bm{v}|^2` the total energy density (the sum of the bulk volumetric internal energy :math:`u` and the kinetic energy density), and :math:`\bm{s}` the *bulk* deviatoric stress tensor. Other packages contribute source terms to the right-hand sides above when enabled — for example the gravitational body force of Chapter :ref:`chap:gravity`. Each material partial density is advected by the common Riemann velocity from the bulk solver. The deviatoric stress is present only when the material strength package is active (Chapter :ref:`chap:strength`); it is itself an aggregate of the per-material deviatoric stresses, :math:`\bm{s}=\sum_m f_m\bm{s}_m`, and for pure hydrodynamics :math:`\bm{s}=\bm{0}`, reducing the momentum/energy equations to the standard compressible Euler form. Note that there is *no* independent per-material energy equation: only the bulk total energy is transported, and the individual material energies are recovered each step from the equilibrium closure below.

The bulk quantities appearing above (:math:`\rho`, :math:`\bm{v}`, :math:`p`, :math:`u`, :math:`\bm{s}`) are aggregated from the per-material states after each update according to the rules in Section :ref:`sec:permat-bulk`: :math:`\rho` and the volumetric internal energy :math:`u` are cell-averaged sums over materials, :math:`\bm{s}` is a volume-fraction-weighted sum, and :math:`p` (with :math:`T`) is the common equilibrium value from the PTE closure below.

.. _`sec:hydro-pte`:

Pressure–Temperature Equilibrium (PTE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The closure that determines the volume fractions :math:`f_m` and the material states is pressure–temperature equilibrium. Given the per-material cell-volume-averaged densities :math:`\bar\rho_m` and the bulk volumetric internal energy :math:`u`, RIOT solves for a common equilibrium pressure :math:`p` and temperature :math:`T` together with the volume fractions :math:`f_m` subject to

.. math::

   \begin{align}
     p_m(\rho_m, T) &= p \quad \text{for all } m
       &&\text{(pressure equilibrium)}, \\
     T_m &= T \quad \text{for all } m
       &&\text{(temperature equilibrium)}, \\
     \sum_m f_m &= 1
       &&\text{(volume additivity)}, \\
     u &= \sum_m \bar\rho_m\,e_m(\rho_m, T)
       &&\text{(energy consistency)},
   \end{align}

where each material’s pressure :math:`p_m(\rho_m, T)` and specific internal energy :math:`e_m(\rho_m, T)` are provided by its equation of state through ``singularity-eos``. In equilibrium every material shares the same pressure :math:`p` and temperature :math:`T`, so the common :math:`p` is the bulk pressure carried by the momentum and energy equations. This root-finding problem is handled by the ``singularity-eos`` ``PTESolverRhoT`` closure; a fixed-temperature solver is used as a fallback. For ideal-gas materials the equilibrium admits a closed-form solution and is evaluated analytically.

Materials whose volume or mass fraction falls below a threshold are removed from the cell (see ``vol_frac_thresh`` and ``mass_frac_thresh``).

Geometric Source Terms
~~~~~~~~~~~~~~~~~~~~~~

For reduced-dimension curvilinear runs (1D spherical, 2D cylindrical), the divergence operators above acquire geometric source terms that RIOT applies explicitly. These are activated by the mesh geometry rather than by a hydro parameter.

Numerical Method
----------------

Interface states are obtained by *reconstruction* of the primitive variables and combined into fluxes by a *Riemann solver*. The available reconstruction methods are listed in the table below and the Riemann solvers in the table below. The stable time step is limited by the CFL condition with the user-specified Courant number ``cfl``.

.. container::
   :name: tab:hydro-recon

   .. table:: Reconstruction methods (``recon``, ``vfrac_recon``).

      ========== ========= ==============================================
      **Option** **Order** **Description**
      ========== ========= ==============================================
      constant   1         Piecewise constant
      plm        2         Piecewise linear (MC limiter)
      ppm4       4         Piecewise parabolic (4th-order interpolation)
      weno5      5         Weighted essentially non-oscillatory (WENO5-Z)
      mp5        5         Monotonicity-preserving 5th-order
      ========== ========= ==============================================

.. container::
   :name: tab:hydro-riemann

   .. table:: Riemann solvers (``riemann``).

      ========== ====================================================
      **Option** **Description**
      ========== ====================================================
      hll        HLL solver; robust but more diffusive than HLLC
      hllc       HLLC (contact-restoring) solver; default
      chllc      Carbuncle-corrected HLLC (Minoshima & Miyoshi 2021)
      lhllc      Low-Mach and carbuncle-corrected HLLC (experimental)
      ========== ====================================================

When material strength is enabled, a strength-aware Riemann solver that accounts for the deviatoric stress contribution is selected automatically.

Input Parameters
----------------

All parameters below reside in the ``<hydro>`` block unless otherwise noted.

.. list-table:: Parameters in the ``<hydro>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - recon
     - string
     - ``plm``
     - Reconstruction method (the table below).
   * - vfrac_recon
     - string
     - ``=recon``
     - Reconstruction method used for volume fractions.
   * - riemann
     - string
     - ``hllc``
     - Riemann solver (the table below).
   * - cfl
     - Real
     - ``0.8``
     - Courant–Friedrichs–Lewy number for time-step control.
   * - lm_correction
     - bool
     - ``false``
     - Apply Thornber low-Mach correction (experimental).
   * - amr_interface
     - bool
     - ``true``
     - Refine on material interfaces when AMR is active.
   * - vol_frac_thresh
     - Real
     - ``1e-12``
     - Volume fraction below which a material is removed.
   * - mass_frac_thresh
     - Real
     - ``1e-12``
     - Mass fraction below which a material is excluded from PTE.
   * - temp_floor
     - Real
     - ``1.0``
     - Minimum allowed temperature (K).
   * - track_total_kinetic_energy
     - bool
     - ``false``
     - Record total kinetic energy in the history file.

Whether the hydro solver runs at all, and which extra terms it evaluates, is governed by the package toggles in the ``<physics>`` block (Section :ref:`sec:physics-block`). In particular, ``hydro`` enables the solver itself; ``strength`` adds the deviatoric stress :math:`\bm{s}` and its transport (Chapter :ref:`chap:strength`); ``gravity`` adds a gravitational body force (Chapter :ref:`chap:gravity`); and ``ionization`` augments the equation-of-state closure.

Registered Fields
-----------------

The hydro package registers the bulk fields listed in the table below. The two independent, conserved fields (``momentum`` and ``total_material_energy``) are transported with fluxes; the remaining bulk fields are derived and single-copy. Per-material fields are registered by the materials package (Chapter :ref:`chap:materials`).

.. container:: fieldtable

   | Fields registered by the hydro package.tab:hydro-fields ccbulk::rho & :math:`\rho` & 1 & Cell, Intensive, Conserved, Derived, OneCopy; bulk density :math:`\rho=\sum_m\bar\rho_m`.
   | ccbulk::momentum & :math:`\rho\bm{v}` & 3 & Cell, Independent, Intensive, Conserved, Vector, WithFluxes; bulk momentum.
   | ccbulk::total_material_energy & :math:`E` & 1 & Cell, Independent, Intensive, Conserved, WithFluxes; bulk total energy density :math:`E=u+\tfrac{1}{2}\rho|\bm{v}|^2`.
   | ccbulk::velocity & :math:`\bm{v}` & 3 & Cell, Intensive, Vector, Derived, OneCopy, FillGhost; bulk velocity.
   | ccbulk::pressure & :math:`p` & 1 & Cell, Intensive, Derived, OneCopy, FillGhost; bulk pressure.
   | ccbulk::temperature & :math:`T` & 1 & Cell, Intensive, Derived, OneCopy, FillGhost, ForceRemeshComm, Restart; temperature.
   | ccbulk::internal_energy & :math:`u` & 1 & Cell, Intensive, Derived, OneCopy; bulk volumetric internal energy :math:`u=\sum_m\bar\rho_m e_m`.
   | ccbulk::bulk_modulus & :math:`B` & 1 & Cell, Intensive, Derived, OneCopy; bulk modulus.
   | ccbulk::max_signal & — & 3 & Cell, Derived, OneCopy, FillGhost, ForceRemeshComm, Restart; per-direction maximum signal speeds.
   | ccbulk::face_signal & — & — & Face, Derived, Flux, OneCopy, CellMemAligned; face-centered signal speeds.

Example
-------

A minimal hydro configuration using PLM reconstruction and the HLLC solver:

.. code:: python

   riot.input("hydro", recon="plm", riemann="hllc", cfl=0.8)
   riot.input("physics", strength=False, ionization=False)
