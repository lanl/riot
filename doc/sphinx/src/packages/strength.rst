.. _`chap:strength`:

Material Strength
=================

The ``strength`` package augments the hydrodynamics of Chapter :ref:`chap:hydro` with an elastic–plastic constitutive response for solid materials. When enabled, materials support a deviatoric stress :math:`\bm{s}` that modifies the momentum and energy fluxes, evolves in time with the flow, and is limited by a yield condition. Strength is applied per material: a material must be flagged as *strong* and assigned a strength model.

Governing Equations
-------------------

Strength in RIOT is a *per-material* property: each strong material :math:`m` carries its own deviatoric stress :math:`\bm{s}_m`, shear modulus :math:`G_m`, and yield strength :math:`Y_m`, and evolves them independently. The hydrodynamics, however, sees a single *bulk* deviatoric stress :math:`\bm{s}` aggregated from the per-material stresses (Section :ref:`sec:permat-bulk`). This section follows that per-material :math:`\to` bulk path.

The full bulk Cauchy stress is decomposed into an isotropic (pressure) part and a traceless deviatoric part,

.. math::

   \begin{equation}
     \bm{\sigma} = -p\,\bm{I} + \bm{s}, \qquad \mathrm{tr}\,\bm{s} = 0 ,
   \end{equation}

where :math:`p` comes from the PTE closure (Chapter :ref:`chap:hydro`) and the bulk deviatoric stress :math:`\bm{s}` enters the bulk momentum and energy fluxes through the terms :math:`-\bm{s}` and :math:`-\bm{s}\!\cdot\!\bm{v}` (see the hydro equations in Chapter :ref:`chap:hydro`).

Per-Material Hypoelastic Stress Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each strong material :math:`m`, the conserved per-material deviatoric stress :math:`\bar\rho_m\bm{s}_m` is advected with the flow and sourced by a hypoelastic constitutive law using the Jaumann (corotational, objective) stress rate, which keeps the response frame-invariant under rigid rotation:

.. math::

   \begin{equation}
     \frac{\partial \left(\bar\rho_m\bm{s}_m\right)}{\partial t}
       + \nabla\!\cdot\!\left(\bar\rho_m\bm{s}_m\,\bm{v}\right)
     \;=\;
     2\bar\rho_m G_m\,\bm{e}
       + \bar\rho_m\!\left(\bm{w}\!\cdot\!\bm{s}_m
         - \bm{s}_m\!\cdot\!\bm{w}\right).
   \end{equation}

Here :math:`G_m` is the material’s shear modulus and :math:`\bar\rho_m` its cell-volume-averaged density, both per material. The strain-rate tensor :math:`\bm{e}` and spin tensor :math:`\bm{w}`, by contrast, are *bulk* quantities: they are computed once per cell from the single velocity field common to all materials in the cell. Thus every material in a cell is driven by the same velocity gradient but evolves its own stress with its own shear modulus. The strain-rate and spin tensors are the symmetric and antisymmetric parts of that velocity gradient,

.. math::

   \begin{align}
     e_{ij} &= \tfrac{1}{2}\!\left(\partial_j v_i + \partial_i v_j\right)
               - \tfrac{1}{3}\,(\nabla\!\cdot\!\bm{v})\,\delta_{ij}, \\
     w_{ij} &= \tfrac{1}{2}\!\left(\partial_j v_i - \partial_i v_j\right).
   \end{align}

Because :math:`\bm{s}_m` is symmetric and traceless, only five of its components are independent and stored (:math:`s_{xx}, s_{xy}, s_{xz}, s_{yy}, s_{yz}`); the sixth follows from :math:`s_{zz} = -s_{xx} - s_{yy}`.

Per-Material Yield and Radial Return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each material’s elastic stress is limited by its own von Mises yield criterion. Defining the second invariant of the per-material deviatoric stress,

.. math::

   \begin{equation}
     J_2 = \tfrac{1}{2}\,\bm{s}_m\!:\!\bm{s}_m
         = \tfrac{1}{2}\!\left(s_{xx}^2 + s_{yy}^2 + s_{zz}^2\right)
           + s_{xy}^2 + s_{xz}^2 + s_{yz}^2 ,
   \end{equation}

material :math:`m` yields when :math:`\bm{s}_m\!:\!\bm{s}_m` exceeds :math:`\tfrac{2}{3}Y_m^2`, where :math:`Y_m` is its yield strength. When the trial stress lies outside the yield surface, a radial-return correction scales it back onto the surface,

.. math::

   \begin{equation}
     \bm{s}_m \leftarrow
       \sqrt{\frac{2Y_m^2}{3\,\bm{s}_m\!:\!\bm{s}_m}}\;\bm{s}_m ,
   \end{equation}

and the material’s accumulated (equivalent) plastic strain :math:`\varepsilon_{p,m}` is incremented accordingly. The radial return and failure ramp are applied per material. The plastic work released by each material during its return is summed over materials and deposited into the *bulk* total energy, so plastic dissipation heats the cell.

Failure
~~~~~~~

As a material’s density falls toward its failure threshold :math:`\rho_{\text{fail},m}`, its deviatoric stress is ramped smoothly to zero,

.. math::

   \begin{equation}
     \bm{s}_m \leftarrow
       \min\!\left(\frac{\rho_m}{\rho_{\text{fail},m}},\,1\right)\bm{s}_m ,
   \end{equation}

so that a fully failed (e.g. spalled or rarefied) material carries no strength and reverts to hydrodynamic behavior.

Aggregation to the Bulk Stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hydrodynamic fluxes use a bulk deviatoric stress and bulk shear modulus formed as volume-fraction-weighted sums of the per-material values (Section :ref:`sec:permat-bulk`),

.. math::

   \begin{equation}
     \bm{s} = \sum_m f_m\,\bm{s}_m, \qquad
     G = \sum_m f_m\,G_m .
   \end{equation}

In practice the per-material stresses are reconstructed to cell faces and then combined with these :math:`f_m` weights, and the resulting bulk stress is passed to the strength-aware Riemann solver. This is the sole channel by which per-material strength influences the shared momentum and energy equations.

Strength Models
---------------

Each strong material is assigned a strength model that supplies the shear modulus :math:`G` and yield strength :math:`Y`. The available model is listed in the table below.

.. container::
   :name: tab:strength-models

   .. table:: Strength models (``modelname``).

      +------------+----------------------------+-----------------------------------------------------------------------------+
      | **Option** | **Name**                   | **Description**                                                             |
      +============+============================+=============================================================================+
      | epp        | Elastic–perfectly-plastic  | Constant shear modulus :math:`G_0` and constant yield strength :math:`Y_0`. |
      +------------+----------------------------+-----------------------------------------------------------------------------+

..

   A pressure- and temperature-dependent Steinberg–Guinan model (Steinberg, Cochran & Guinan, *J. Appl. Phys.* **51**, 1498, 1980) is present in the source but not yet enabled.

Input Parameters
----------------

Material strength is activated in three places: globally in the ``<physics>`` block, per material in each ``<material>`` block, and in a user-named block that defines the strength model itself.

.. container:: paramtable

   | Global toggle in the ``<physics>`` block. strength & bool & ``false`` & Enable the material strength package.

.. container:: paramtable

   | Per-material parameters in each ``<material``\ :math:`N`\ ``>`` block. strong & bool & ``false`` & Flag this material as having strength.
   | strength_model & string & — & Name of the input block defining this material’s strength model (required if ``strong`` is true).

The strength-model block is named by the ``strength_model`` parameter above. Its contents depend on the chosen ``modelname``.

.. container:: paramtable

   | Parameters in a strength-model block (for ``modelname = epp``). modelname & string & — & Model selector; use ``epp``.
   | G0 & Real & — & Shear modulus :math:`G_0` (required for ``epp``).
   | Y0 & Real & — & Yield strength :math:`Y_0` (required for ``epp``).
   | rho_fail & Real & ``0.0`` & Density below which the material loses strength.

Registered Fields
-----------------

When strength is enabled, the per-material stress fields in the table below are registered (sparse, controlled by ``ccmat::rho``), along with bulk strain-rate support fields. The conserved :math:`\rho\bm{s}` and :math:`\rho\varepsilon_p` are advected with fluxes; the material-averaged ``cm::`` counterparts and the strain-rate/face-velocity fields are derived.

.. container:: fieldtable

   | Fields registered when material strength is active.tab:strength-fields ccmat::deviatoric_stress & :math:`\bar\rho_m\bm{s}` & 5 & Cell, Independent, Intensive, Conserved, Sparse, FillGhost, WithFluxes; conserved deviatoric stress.
   | ccmat::equivalent_plastic_strain & :math:`\bar\rho_m\varepsilon_p` & 1 & Cell, Independent, Intensive, Conserved, Sparse, FillGhost, WithFluxes, Advected; plastic strain.
   | cm::deviatoric_stress & :math:`\bm{s}` & 5 & Cell, Intensive, Sparse, Derived, OneCopy; material-averaged deviatoric stress.
   | cm::equivalent_plastic_strain & :math:`\varepsilon_p` & 1 & Cell, Intensive, Sparse, Derived, OneCopy; equivalent plastic strain.
   | cm::shear_modulus & :math:`G` & 1 & Cell, Intensive, Sparse, Derived, OneCopy; shear modulus.
   | cm::strength_j2 & :math:`J_2` & 1 & Cell, Sparse, Derived, OneCopy; second stress invariant (diagnostic).
   | ccbulk::shear_modulus & :math:`G` & 1 & Cell, Intensive, Derived, OneCopy; volume-averaged shear modulus.
   | ccbulk::strain_rate & :math:`\bm{e}` & 6 & Cell, Intensive, Derived, OneCopy; strain-rate tensor.
   | ccbulk::face_velocity & — & 9 & Cell, Intensive, Derived, OneCopy; face velocities used to build :math:`\bm{e}`.

Example
-------

An elastic–perfectly-plastic beryllium, flagged as strong and pointed at a strength-model block:

.. code:: python

   riot.input("physics", strength=True)
   riot.input("material0", strong=True,
              strength_model="beryllium_strength")
   riot.input("beryllium_strength", modelname="epp",
              G0=1.519e12,   # shear modulus
              Y0=3.30e9,     # yield strength
              rho_fail=0.5)  # failure density
