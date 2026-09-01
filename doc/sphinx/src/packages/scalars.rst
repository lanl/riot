.. _`chap:scalars`:

Passive Scalars
===============

The ``scalars`` package advects passive scalar fields with the flow. Scalars are one-way coupled: they are transported by the hydrodynamic velocity but do not feed back on the dynamics. They are typically used as tracers, tagging a region or material with a value of one and zero elsewhere. Any number of scalars may be defined. The package requires hydrodynamics to be enabled.

Governing Equations
-------------------

RIOT supports two kinds of passive scalar.

Material-Tied Scalars
~~~~~~~~~~~~~~~~~~~~~

A material-tied scalar is a mass fraction :math:`c` carried by a particular material. Its conserved density :math:`C = \bar\rho_m\,c` is advected in flux-conservation form with no source,

.. math::

   \begin{equation}
     \frac{\partial \left(\bar\rho_m c\right)}{\partial t}
       + \nabla\!\cdot\!\left(\bar\rho_m c\,\bm{v}\right) = 0 ,
   \end{equation}

and the primitive mass fraction is recovered as :math:`c = C/\bar\rho_m`. Because it is tied to a material, the scalar is a sparse field that is allocated and deallocated together with that material.

Bulk-Tied Scalars
~~~~~~~~~~~~~~~~~

A bulk-tied scalar :math:`\psi` is not associated with any mass; it is a dense field advected directly with the bulk velocity,

.. math::

   \begin{equation}
     \frac{\partial \psi}{\partial t} + \nabla\!\cdot\!\left(\psi\,\bm{v}\right) = 0 .
   \end{equation}

No primitive/conserved distinction is required for a bulk-tied scalar.

Both kinds are advected with the same reconstruction and Riemann machinery as the hydrodynamic fields (Chapter :ref:`chap:hydro`).

Input Parameters
----------------

Passive scalars are enabled with the ``scalars`` toggle in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``hydro`` must also be enabled. Each scalar is defined in a contiguous, zero-based block ``<scalars0>``, ``<scalars1>``, …. Whether a scalar is material-tied or bulk-tied is determined by the presence of the ``matid`` parameter.

.. container:: paramtable

   | Per-scalar parameters in each ``<scalars``\ :math:`N`\ ``>`` block. label & string & *block* & Name of the scalar field (defaults to the block name).
   | matid & int & — & Material ID to tie the scalar to; if omitted, the scalar is bulk-tied.

Registered Fields
-----------------

For each material-tied scalar the package registers a conserved (advected) field and its derived primitive counterpart; a bulk-tied scalar registers a single advected field. The exact field names are taken from the user’s ``label``; the table below shows their structure.

.. container:: fieldtable

   | Fields registered per scalar (names taken from ``label``).tab:scalars-fields label (material-tied) & :math:`\bar\rho_m c` & 1 & Cell, Independent, Intensive, Sparse, FillGhost, Advected, WithFluxes; conserved scalar density.
   | prim.label & :math:`c` & 1 & Cell, Sparse, Derived, OneCopy; primitive mass fraction.
   | label (bulk-tied) & :math:`\psi` & 1 & Cell, Independent, Intensive, FillGhost, Advected, WithFluxes; bulk-advected scalar.

Example
-------

One material-tied tracer on material ``0`` and one bulk-tied tracer:

.. code:: python

   riot.input("physics", hydro=True, scalars=True)
   riot.input("scalars0", label="dye", matid=0)  # tied to material 0
   riot.input("scalars1", label="marker")        # no matid => bulk-tied
