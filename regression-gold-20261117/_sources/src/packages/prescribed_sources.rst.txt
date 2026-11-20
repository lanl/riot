.. _`chap:prescribed-sources`:

Prescribed Sources
==================

The ``prescribed_sources`` package deposits energy into a material
according to a user-supplied schedule — a time series of the
cumulative energy delivered to that material. It is the way to drive a
simulation with an externally specified energy input (for example a
modeled laser or radiation drive) without resolving the physical
source. Any number of independent sources may be defined, each
targeting one material. The package requires hydrodynamics.

Governing Equations
-------------------

Each source is specified by a monotonically increasing schedule
:math:`\mathcal{E}_m(t)` giving the *cumulative, total* energy
delivered to material :math:`m` up to time :math:`t`, integrated over
all of that material in the domain. Over a step from :math:`t` to
:math:`t+\Delta t` the total energy added to the material is the
increment

.. math::

     \Delta E_m = \max\!\left(\mathcal{E}_m(t+\Delta t) - \mathcal{E}_m(t),\ 0\right).

This total is converted to a *specific* energy increment by dividing
by the material’s current total mass :math:`M_m` (summed over the
whole domain, Section :ref:`sec:presrc-method`), and deposited
uniformly by mass: a cell holding a partial density :math:`\bar\rho_m`
of the material receives

.. math::

     \Delta\!\left(\text{bulk energy density}\right)
       = \bar\rho_m\,\frac{\Delta E_m}{M_m}.

The increment therefore adds the same specific energy :math:`\Delta
E_m/M_m` everywhere the material is present, so the source heats the
material uniformly per unit mass rather than per unit volume. The
energy is added to the bulk total energy
(``ccbulk::total_material_energy``); the per-material internal
energies, temperature, and pressure are then recovered through the
equation-of-state and PTE closure of Chapter :ref:`chap:materials`.

.. warning::

   The schedule is *cumulative*: the file records total energy
   delivered so far, not an instantaneous rate. Because the source is
   applied as a specific energy increment, a multi-phase material is
   heated with a single :math:`\mathrm{d}e/\mathrm{d}t` common to all
   its phases.

.. warning::

   The energies in the schedule have units of **total** energy,
   **not** specific, even though a specific energy is used internally
   by the actual source terms.

.. _`sec:presrc-method`:

Numerical Method
----------------

The package runs as an operator-split step after the hydrodynamic
update. On each step it adds the energy increment to the bulk total
energy, then recomputes the derived material state (internal energies
and a PTE solve), exchanges boundary data, and fills derived
quantities again. The domain-summed per-material masses :math:`M_m`
needed to convert total to specific energy are reduced before each
step (and once before the main loop, so they are present in the first
output).

The schedule file is read as an ASCII table and resampled onto a
uniform time grid at initialization; values are interpolated within
the tabulated range and held constant (constant extrapolation) outside
it. The package also votes on the time step: it limits :math:`\Delta
t` so that the energy deposited does not change a material’s
temperature too abruptly, scaling the estimate by ``dt_safety``.

Input Parameters
----------------

Prescribed sources are enabled with ``prescribed_sources`` ``= true``
in the ``<physics>`` block (Section :ref:`sec:physics-block`);
``hydro`` must also be enabled. A single global control lives in the
``<prescribed_sources>`` block, and each source is defined in its own
contiguous, zero-based ``<energy_source0>``, ``<energy_source1>``,
… block.

.. list-table:: Parameter in the ``<prescribed_sources>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - dt_safety
     - Real
     - ``0.9``
     - Safety factor applied to the source time-step limit.

.. list-table:: Per-source parameters in each ``<energy_source``\ :math:`N`\ ``>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - material
     - int
     - —
     - Index of the material this source drives; required.
   * - cumulative_energies
     - string
     - —
     - Path to an ASCII file giving the schedule; required.
   * - active
     - bool
     - ``true``
     - Set ``false`` to disable this source without removing the block.

The Schedule File
~~~~~~~~~~~~~~~~~

The ``cumulative_energies`` file is a two-column ASCII table: the
first column is time and the second is the cumulative total energy
delivered to the material by that time. The energies must be
monotonically increasing and non-negative, and the first energy entry
must be exactly ``0``. The table need not be uniformly spaced — RIOT
resamples it onto a uniform grid on read.

Example
-------

Drive material ``0`` with a tabulated energy schedule, and record the
per-material masses and energies in the history file:

.. code:: python

   riot.input("physics", hydro=True, prescribed_sources=True,
              sparse_physics=False)
   riot.input("energy_source0",
              material=0,
              cumulative_energies="drive_schedule.dat")  # time, cumulative energy
   riot.input("diagnostics", packages=["masses", "energies"])

A minimal ``drive_schedule.dat`` delivering :math:`100` (energy units)
linearly over the first microsecond and holding thereafter:

::

   # time      cumulative_energy
   0.0         0.0
   1.0e-6      100.0
   1.0e-3      100.0
