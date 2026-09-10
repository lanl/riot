.. _`chap:levelsets`:

Level Sets
==========

The ``levelsets`` package tracks a material interface with a level-set function :math:`\phi`, whose zero contour :math:`\phi = 0` marks the interface. The field is advected with the flow and periodically reinitialized to remain a signed distance function near the interface. It provides a sharp interface representation complementing the diffuse volume-fraction description of Chapter :ref:`chap:hydro`.

.. warning:: 

   **Experimental:** The level-set package currently supports a single level set and requires hydrodynamics to be enabled.

Governing Equations
-------------------

Advection
~~~~~~~~~

The level-set function is advected by the hydrodynamic velocity using the same flux-conservative machinery (reconstruction and Riemann solver) as the other advected fields,

.. math::

     \frac{\partial \phi}{\partial t} + \nabla\!\cdot\!\left(\phi\vec{v}\right) = 0 .

By convention :math:`\phi > 0` in the designated *sharp* material (``sharp_mat``) and :math:`\phi < 0` elsewhere, so the interface is the :math:`\phi = 0` contour.

Reinitialization
~~~~~~~~~~~~~~~~

Advection distorts :math:`\phi` away from a signed distance function (:math:`|\nabla\phi| = 1`). Every ``reinit_modcyc`` cycles the package restores that property near the interface by evolving a hyperbolic Eikonal equation in a pseudo-time :math:`\tau` (Sussman & Fatemi, *SIAM J. Sci. Comput.* **20**, 1165, 1999),

.. math::

     \frac{\partial\phi}{\partial\tau}
       = \operatorname{sign}(\phi_0)\left(1 - |\nabla\phi|\right),

where :math:`\phi_0` is the level set at the start of reinitialization and :math:`\operatorname{sign}(\phi_0)` is a smoothed sign function. The gradient magnitude is formed with upwind differences selected by the sign of :math:`\phi_0` (forward differences where :math:`\phi_0 > 0`, backward where :math:`\phi_0 < 0`) using second-order minmod-limited reconstruction. Reinitialization is applied only within a band of half-width ``reinit_width`` cells around the interface; outside the band :math:`\phi` is held at a constant magnitude. The pseudo-time update runs for ``reinit_nstep`` steps, each propagating the correction roughly one cell.

Input Parameters
----------------

Level sets are enabled with the ``levelsets`` toggle in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``hydro`` must also be enabled. The remaining controls live in the ``<levelsets>`` block.

.. list-table:: Parameters in the ``<levelsets>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - sharp_mat
     - int
     - ``0``
     - Index of the material on the :math:`\phi > 0` side of the interface.
   * - reinit_modcyc
     - int
     - ``25``
     - Reinitialize every this many cycles (:math:`1` = every cycle).
   * - reinit_width
     - int
     - ``4``
     - Half-width (in cells) of the band kept as a signed distance function.
   * - reinit_nstep
     - int
     - ``15``
     - Number of pseudo-time steps per reinitialization (ideally :math:`>` ``reinit_width``).

Registered Fields
-----------------

The package registers the single advected level-set field in the table below, plus two single-copy scratch fields used only during reinitialization (:math:`\phi_0` storage and the pseudo-time right-hand side).

.. list-table:: Fields registered by the level-set package.
   :class: wraptable
   :header-rows: 1
   :widths: 30 14 16 40
   :name: tab:levelsets-fields

   * - Field
     - Symbol
     - Components
     - Metadata / description
   * - levelset
     - :math:`\phi`
     - 1
     - Cell, Independent, Intensive, FillGhost, Advected, WithFluxes; the level-set function.
   * - levelset0
     - :math:`\phi_0`
     - 1
     - Cell, Independent, Intensive, OneCopy; level set stored at the start of reinitialization.
   * - dudt_reinitialize
     - :math:`\partial_\tau\phi`
     - 1
     - Cell, Independent, Intensive, OneCopy; reinitialization right-hand side.

Example
-------

Track the interface of material ``0``, reinitializing every 25 cycles:

.. code:: python

   riot.input("physics", hydro=True, levelsets=True)
   riot.input("levelsets", sharp_mat=0, reinit_modcyc=25,
              reinit_width=4, reinit_nstep=15)
