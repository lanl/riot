.. _`chap:diagnostics`:

Diagnostics
===========

*Diagnostics* are the mechanism through which RIOT computes derived, problem-specific quantities that probe a running simulation — typically global or regional reductions written to the history file — without altering the evolved state. They are deliberately open-ended: rather than a fixed menu of outputs, the diagnostics layer is an *extension point* where a developer registers a small, self-contained package that measures whatever a particular study needs (a total energy, a peak temperature, a shell mass, and so on). This chapter describes the mechanism and how to add a diagnostic, rather than cataloguing the specific diagnostics that happen to exist today, which come and go with the problems they serve.

Concept
-------

A diagnostic is a lightweight Parthenon package that registers one or more *history outputs*: reduction functions that Parthenon evaluates on a schedule and appends to the run’s history (``.hst``) file. Each reduction maps the whole mesh (across all blocks and MPI ranks) to a scalar or a small vector — for example a volume integral of a field, a maximum over the domain, or a per-material sum. Because a diagnostic only *reads* the state and writes to the history stream, it is side-effect free and can be enabled or disabled freely without changing the simulation result.

Diagnostics are distinct from the field *outputs* of Chapter :ref:`chap:parthenon` (full-mesh dumps of variables) and from the tracer particles of Chapter :ref:`chap:tracers` (Lagrangian sampling): they are the channel for *reduced*, whole-domain measurements recorded frequently over time.

Enabling Diagnostics
--------------------

Unlike the physics packages, diagnostics are not toggled in the ``<physics>`` block. Instead they are selected by name in the ``<diagnostics>`` block through the ``packages`` list; each named diagnostic is looked up in an internal registry and enrolled, and an unknown name is an error.

.. container:: paramtable

   | Parameter in the ``<diagnostics>`` block. packages & list & *empty* & Names of the diagnostic packages to enable this run.

.. code:: python

   riot.input("diagnostics", packages=["masses", "energies"])

A diagnostic may read additional parameters from its own block if it needs configuration, but many take none. The quantities each diagnostic registers appear as named columns in the history file.

Adding a Diagnostic
-------------------

A new diagnostic is a self-contained unit in ``src/diagnostics/`` with a single ``Initialize`` entry point, registered in the diagnostics registry so it can be requested by name. The pattern is:

#. Write one or more *reduction functions*. A reduction takes the mesh data and returns a scalar (or a vector, e.g. one entry per material), computed with a Parthenon parallel reduction over the interior cells. A volume integral weights each cell by its volume; a peak uses a max reduction; a per-material quantity reduces each material’s contribution separately.

#. In ``Initialize``, create a Parthenon ``StateDescriptor`` and attach the reductions as *history outputs*, each with a reduction operation (sum, max, …), the function, and a column label. Scalar and vector history outputs are registered through their respective history-output parameter keys.

#. Add the diagnostic’s name to the registry (the ``FOREACH_DIAG`` list in ``diagnostics.hpp``) so ``packages`` can find it.

Parthenon then performs the cross-rank reduction and writes the labeled result to the history file at the configured cadence; the diagnostic itself only supplies the per-rank measurement. Because the reduction runs on the device over variable packs, a diagnostic sees exactly the same field data — bulk, per-material, or sparse — as any physics kernel, so it can measure any registered quantity.

   The diagnostics currently in the tree (for example per-material mass and energy sums, and problem-specific shell diagnostics) are illustrative instances of this pattern and are tied to the problems they were written for; they are best read directly as templates for a new diagnostic rather than relied on as a stable, general-purpose set.
