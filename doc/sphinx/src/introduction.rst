.. _`chap:introduction`:

Introduction
============

RIOT is a block adaptive mesh refinement (AMR) multi-material
hydrodynamics code built atop the Parthenon performance-portability
framework. Parthenon supplies the block-AMR infrastructure, load
balancing, and communication machinery but contains no physics; RIOT
supplies the hydrodynamics algorithm and the physics packages
documented in this manual.

Organization of the Documentation
---------------------------------

Each physics package in RIOT is documented in its own chapter. Every
chapter follows the same structure:

#. the *governing equations* solved by the package, and

#. the *user-tunable input parameters* that control it.

Input parameters are organized into *blocks*. The recommended way to
write inputs is a Python script that calls ``riot.input(block, …)``
for each block (Chapter :ref:`chap:python`); each call corresponds to
one block of the underlying text input deck (a ``.rin`` file). The
parameter tables in this manual list parameters by their block and
name (e.g. ``cfl`` in the ``<hydro>`` block); in a Python script these
are supplied as keyword arguments:

.. code:: python

   riot.input("hydro", recon="plm",   # reconstruction method
                       riemann="hllc",  # Riemann solver
                       cfl=0.8)         # CFL number

The same block in the equivalent text input deck reads:

::

   <hydro>
   recon   = plm      # reconstruction method
   riemann = hllc     # Riemann solver
   cfl     = 0.8      # CFL number

Each package chapter also includes a *Registered Fields* table listing
the Parthenon fields that package creates, the symbol each maps to in
the governing equations, its component count, and its metadata.

.. note::

   For readability, the metadata column of every *Registered Fields*
   table lists the salient flags (e.g. Independent, Conserved,
   WithFluxes, Sparse, Derived) rather than the complete flag set
   passed in the source.

Running RIOT
------------

After building RIOT, run the executable from ``build/src``. Supply an input
deck with ``-i`` and, optionally, select an output directory with ``-d``:

.. code-block:: bash

   cd build/src
   ./riot -i input.rin -d /path/to/output/
   mpiexec -n 4 ./riot -i input.rin -d /path/to/output/

The second command runs the same problem using four MPI ranks. See
:doc:`/src/building` for prerequisites, configuration, and installation details.

.. _`sec:physics-block`:

Enabling Physics: the ``<physics>`` Block
-----------------------------------------

Which packages are active in a run is controlled by boolean toggles in
the ``<physics>`` block. Hydrodynamics is on by default; the remaining
packages are off by default and are enabled here. Each package’s own
parameters live in its own block, documented in the corresponding
chapter.

.. list-table:: Package toggles in the ``<physics>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - hydro
     - bool
     - ``true``
     - Enable hydrodynamics.
   * - strength
     - bool
     - ``false``
     - Enable material strength.
   * - ionization
     - bool
     - ``false``
     - Enable partial ionization.
   * - levelsets
     - bool
     - ``false``
     - Enable level-set interface tracking.
   * - scalars
     - bool
     - ``false``
     - Enable passive scalars.
   * - mix
     - bool
     - ``false``
     - Enable the BHR RANS subgrid mixing model.
   * - tn
     - bool
     - ``false``
     - Enable thermonuclear burn.
   * - radiation_transport
     - bool
     - ``false``
     - Enable radiation transport.
   * - multigroup_diffusion
     - bool
     - ``false``
     - Enable P1 radiation diffusion.
   * - gravity
     - bool
     - ``false``
     - Enable a constant gravitational acceleration.
   * - prescribed_sources
     - bool
     - ``false``
     - Enable prescribed energy sources.
   * - lasers
     - bool
     - ``false``
     - Enable laser ray tracing and energy deposition.
   * - tracers
     - bool
     - ``false``
     - Enable Lagrangian tracer particles.
   * - fixed_fluid
     - bool
     - ``false``
     - Hold the fluid fixed (no hydro update).

.. _`sec:sparsity`:

Sparsity
--------

Many of RIOT’s per-material fields (cell-volume-averaged densities,
volume fractions, and the derived material state of
Chapter :ref:`chap:materials`) are registered as Parthenon *sparse*
fields. A sparse field is allocated only on the mesh blocks where it
is actually needed — for a material, only on blocks where that
material is present — rather than everywhere in the domain. In a
multi-material simulation where each material occupies a limited
region, this saves substantial memory, since a block that contains
none of a given material carries no storage for it. As materials move
through the mesh, Parthenon allocates a material’s fields on blocks it
enters and deallocates them on blocks it has left.

The deallocation step is controlled by ``sparse_dealloc`` in the
``<materials>`` block. When ``true`` (the default), the fields of a
material that is no longer present on a block are freed, reclaiming
memory; when ``false``, once-allocated fields persist for the rest of
the run.

.. list-table:: Sparsity parameter in the ``<materials>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - sparse_dealloc
     - bool
     - ``true``
     - Free a material’s fields on blocks it has left.

Submodules
----------

RIOT consumes several external libraries as submodules:

- **Parthenon** — block-AMR framework (mesh, communication, load balancing).

- **singularity-eos** — equation-of-state library.

- **singularity-opac** — opacity library.

These are summarized briefly in their own chapter. It also relies on
``Catch2`` for unit tests and ``kokkos-kernels`` for device-side
linear solvers. 
