.. _`chap:tracers`:

Tracer Particles
================

The ``tracers`` package carries massless marker particles through the flow and samples field values at their locations. Particles are one-way coupled: they read the mesh state but do not affect it. Two modes are supported per group of particles — *Lagrangian* tracers that advect with the fluid velocity, and *Eulerian* probes that stay fixed in space — and each group records a user-chosen list of fields interpolated to the particle position. Any number of independent particle groups (Parthenon *swarms*) may be defined.

Method
------

Swarms
~~~~~~

Each group of particles is a named swarm, defined by a ``<tracers/>``\ *name* block. The particles of a swarm share a common behavior (advecting or fixed) and a common list of sampled fields. Particles are seeded at initialization from explicit position lists and are distributed to the mesh blocks that contain them; as they move they are communicated between blocks and MPI ranks by Parthenon’s swarm machinery. Each particle carries a unique ``id`` so its trajectory can be followed across the run.

Transport
~~~~~~~~~

A Lagrangian swarm (``advect`` ``= true``) is pushed each step by the bulk velocity (``ccbulk::velocity``) with a second-order Runge–Kutta (midpoint) integrator: the velocity is reconstructed to the particle position with an MC-limited slope, a half-step midpoint position is formed, and the velocity there advances the particle over the full step. A swarm with ``advect`` ``= false`` is never moved — its particles act as fixed Eulerian probes.

Field Sampling
~~~~~~~~~~~~~~

After transport (and once at initialization), each swarm samples its ``sample_fields`` at every particle. The value is reconstructed from the containing cell with a second-order, MC-limited slope in each dimension, so the sampled value reflects the particle’s position within the cell rather than the plain cell average. Any registered field may be sampled, including multi-component and sparse per-material fields; the package resolves each field’s component layout automatically and stores the result in a matching per-particle swarm variable. Because the sampling reads other packages’ fields, the tracers package is initialized last, after every field it might sample has been registered.

Input Parameters
----------------

Tracers are enabled with ``tracers`` ``= true`` in the ``<physics>`` block (Section :ref:`sec:physics-block`). Each swarm is defined in its own ``<tracers/>``\ *name* block; the part of the block name after ``tracers/`` is the swarm’s name (used to select it for output).

.. container:: paramtable

   | Per-swarm parameters in each ``<tracers/``\ *name*\ ``>`` block. x1, x2, x3 & list & — & Initial particle positions, one entry per particle; the three lists must be the same length.
   | advect & bool & ``true`` & Advect with the flow (Lagrangian); ``false`` keeps particles fixed (Eulerian).
   | sample_fields & list & *empty* & Field names to interpolate onto the particles (e.g. ``c.c.bulk.velocity``).

The sampled values are written to output by requesting the swarm in a ``<parthenon/output>``\ :math:`N` block (``swarms``), together with the built-in ``swarm.id``; the sample-field swarm variables can be emitted automatically.

Registered Fields
-----------------

The package registers no mesh (cell) fields. For each defined swarm it registers the swarm itself, the built-in per-particle position and ``id``, and one swarm variable per sampled field (sized to that field’s component count). These per-particle variables are the package’s output.

Example
-------

The triple-point setup below defines two swarms: a Lagrangian group that advects with the flow and records the velocity, and an Eulerian group of fixed probes that record density and pressure. Here the initial positions are drawn at random with NumPy, illustrating that the position lists are ordinary Python lists.

.. code:: python

   import numpy as np
   import riot

   riot.input("physics", hydro=True, tracers=True)

   n = 100000
   rng = np.random.default_rng(seed=42)
   x1 = rng.uniform(0.0, 7.0, n).tolist()
   x2 = rng.uniform(0.0, 3.0, n).tolist()
   x3 = [0.0] * n

   # Lagrangian tracers: move with the flow, sample the velocity
   riot.input("tracers/lagrangian",
              x1=x1, x2=x2, x3=x3,
              advect=True,
              sample_fields=["c.c.bulk.velocity"])

   # Eulerian probes: stay fixed, sample density and pressure
   riot.input("tracers/eulerian",
              x1=x1, x2=x2, x3=x3,
              advect=False,
              sample_fields=["c.c.bulk.rho", "c.c.bulk.pressure"])

   # Emit each swarm to its own output
   riot.input("parthenon/output1", swarms=["lagrangian"],
              swarm_variables=["swarm.id"],
              auto_swarm_sample_fields=True, write_swarm_xdmf=True,
              file_type="hdf5", dt=0.01)
   riot.input("parthenon/output2", swarms=["eulerian"],
              swarm_variables=["swarm.id"],
              auto_swarm_sample_fields=True, write_swarm_xdmf=True,
              file_type="hdf5", dt=0.01)
