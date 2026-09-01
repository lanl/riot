.. _`chap:python`:

The Python Interface
====================

The recommended way to set up a RIOT problem is with a Python input script. A script uses the ``riot`` Python module to declare the mesh, materials, physics, and regions, and can define regions whose geometry and initial state are arbitrary Python functions. This is strictly more powerful than the text input deck: it supports parameterized studies, programmatic geometry, and—through the ``singularity-eos`` Python bindings—initial states that are thermodynamically consistent with the equation of state.

Workflow
--------

A Python input serves two roles. Run as an ordinary script it *generates* a text input deck; at run time RIOT *calls back* into it for any region whose geometry or state is defined in Python.

#. Write a script (e.g. ``myproblem.py``) that imports the ``riot`` module, declares the problem, and finally calls ``generate_input()`` to write the deck.

#. Run it to emit the deck; then launch RIOT on the generated deck:

::

   python3 myproblem.py                 # writes myproblem.rin
   mpiexec -n 1 ./riot -i myproblem.rin -d /path/to/output/

The two steps can also be combined: because the script prints the path of the deck it writes, its output can be fed directly to the ``-i`` flag, so the deck is (re)generated and consumed in a single command:

::

   mpiexec -n 1 ./riot -i $(python3 myproblem.py) -d /path/to/output/

When RIOT encounters a region with ``mask_type`` ``= python``, it loads the script, instantiates the class named by the region, and calls its methods to fill the initial data.

Setting ``PYTHONPATH``
~~~~~~~~~~~~~~~~~~~~~~

Both the ``riot`` module (in ``script/inputs/``) and, if used, the ``singularity_eos`` bindings must be importable, so ``PYTHONPATH`` must include their locations. The exact paths depend on where RIOT is built and consumed as a dependency. As a concrete example, when running from the build tree at ``build/src/`` (the location of the executable):

::

   export PYTHONPATH="$(pwd)/../../external/riot/script/inputs/:\
   $(pwd)/../external/riot/singularity-eos/python:$PYTHONPATH"

Adjust these paths to match your own build and run directory; the two entries point at the ``riot`` input module and the ``singularity-eos`` Python bindings, respectively.

API Surface
-----------

The core entry points of the ``riot`` module are summarized in the table below.

.. code-block:: text

   @P0.40 L0.50@ **Call & Purpose
   riot.input(block, \**kwargs) & Add or update an input block (emits ``<block>`` in the deck).
   riot.material(id, \**kwargs) & Declare a material and its properties.
   riot.material[id_or_name] & Look up a declared material.
   riot.input[block] & Look up a previously declared block.
   riot.EOS(material) & Wrap a ``singularity-eos`` object for thermodynamic queries.
   riot.constants(units) & Physical constants (default CGS).
   riot.log(msg) & Write to the run log.
   riot.input.generate_input() & Write the ``.rin`` deck (name derived from the script).
   **

Python Regions
--------------

A region is made Python-driven by setting ``mask_type`` ``= python`` and giving it a ``name``; RIOT instantiates the class of that name from the script. The class may define:

- ``mask(self, pos)`` — returns a boolean array selecting the cells in the region. ``pos`` is an :math:`(N,3)` array of positions; the attributes ``self.x``, ``self.y``, ``self.z`` are set to the column indices :math:`0,1,2`. The columns are always *Cartesian* :math:`(x,y,z)`, even when the mesh uses a curvilinear coordinate system: RIOT converts each cell’s native coordinates to Cartesian before calling the Python region, so a region’s geometry is written in Cartesian space regardless of the run’s coordinates (see the caveat below).

- state setters such as ``c_m_rho(self, pos, rho)``, ``c_m_pressure(self, pos, press)``, and ``c_c_bulk_velocity(self, pos, vel)`` — each fills the output array in place from the positions.

Parameters for the class are passed through a companion block ``<name/params>``; each becomes an attribute of the instance. This is how geometry and state can vary arbitrarily in space, including perturbation seeds for instability studies.

   **Curvilinear caveat.** The Cartesian conversion of ``pos`` is currently partial in reduced-dimension curvilinear runs. In ``UniformCylindrical`` the mesh radius maps to :math:`x` and the axial coordinate to :math:`z`, with :math:`y=0` (i.e. the 2D :math:`r`–:math:`z` plane); in ``UniformSpherical`` the mesh radius maps to :math:`x` with :math:`y=z=0` (i.e. 1D radial). A Python region in these geometries should therefore use ``self.x`` as the radius (and, in cylindrical, ``self.z`` as the axial coordinate) and not rely on the dropped columns.

.. _`sec:cad`:

CAD Geometry
------------

When a region’s shape is too complex to express analytically or in a Python ``mask``, its geometry can be imported directly from a CAD model. Setting ``mask_type`` ``= cad`` makes RIOT read a solid from a STEP file and use it as the region mask; the region’s state is still set by the usual analytic or Python setters, so CAD supplies only the *shape*. This path is built on Open CASCADE (OCCT) and is compiled in only when RIOT is configured with ``-DRIOT_ENABLE_CAD``; a build without it will abort if a ``cad`` region is requested.

A CAD region reads the parameters in the table below. The ``name`` of the region doubles as the label of the solid to extract from the STEP file, so a single file may hold several named parts, each selected by a different region.

.. code-block:: text

   @P0.22 P0.10 P0.12 L0.42@ **Parameter & Type & Default & Description
   mask_type & string & — & Set to ``cad``.
   cadfile & string & — & Path to the STEP (``.step``/``.stp``) file.
   name & string & — & Region label; also the name of the solid to extract from the file.
   sample_dx_max & Real & ``-1`` & Optional cap on the base sampling cell size; :math:`\le 0` derives it from the mesh.
   **

At initialization RIOT loads the STEP file, heals the imported solid (to tolerate models that are not perfectly watertight), and builds a cell-based adaptive mesh that brackets the part: it samples an inside/outside point classifier on a coarse node grid and refines only those cells whose corners disagree, so the boundary is resolved to a depth tied to the mesh’s refinement levels (``nlev_min``, ``nlev_max`` in the ``<regions>`` block, and ``numlevel`` for adaptive runs). During domain initialization the region mask is then a fast interpolated lookup on this structure. Because refinement keys on node sampling, a feature sharp enough to pierce a cell face without touching a node can be missed; ``sample_dx_max`` forces a finer base grid when that matters.

.. code:: python

   riot.input("regions", nlev_max=4)
   riot.input("region1", mask_type="cad", name="bracket", matid=1,
              cadfile="assembly.step",
              c_m_rho=8.0, c_m_temperature=293.0)

Here the solid named ``bracket`` inside ``assembly.step`` defines the region’s shape, while its material state is set analytically; the same region could instead draw its state from a Python class exactly as in the previous section.

Consistent States with singularity-eos
--------------------------------------

Because the same equation of state is available in Python, initial states can be made thermodynamically consistent. For example, a region can set its pressure from a prescribed density and specific internal energy by calling the EOS directly:

.. code:: python

   self.eos = riot.EOS("material1")
   def c_m_pressure(self, pos, press):
       rho = self.rho_func(pos)
       self.eos.PressureFromDensityInternalEnergy(rho, 2.0 / rho, press)

The ``riot.EOS`` wrapper reads the material’s EOS block and constructs the matching ``singularity-eos`` object, so the initial condition uses exactly the EOS the simulation will run with.

Example
-------

The following complete script (``rose.py``) builds a 2D problem whose Python region carves a rose-curve (:math:`r = |a\sin(k\theta)|`) filled with a spatially varying density, using the EOS to set a consistent pressure.

.. code:: python

   import numpy as np
   import riot
   import singularity_eos

   const = riot.constants()

   def make_input():
       riot.input("riot", problem="region_pgen", sparse_physics=True)
       riot.input("parthenon/job", problem_id="python")
       riot.input("parthenon/time", nlim=1, tlim=0.1, integrator="rk2")
       riot.input("parthenon/mesh", refinement="adaptive", numlevel=5,
                  nx1=32, x1min=-1.2, x1max=1.2,
                  ix1_bc="reflecting", ox1_bc="reflecting",
                  nx2=32, x2min=-1.2, x2max=1.2,
                  ix2_bc="reflecting", ox2_bc="reflecting",
                  nx3=1, x3min=-0.01, x3max=0.01)
       riot.input("parthenon/meshblock", nx1=32, nx2=32, nx3=1)

       riot.material(1, label="air1", eos_type="IdealGas", Gamma=1.5, Cv=1.0e-3)

       riot.input("regions", nlev_max=5)
       riot.input("region0", mask_type="background", matid=0,
                  c_m_rho=1.0, c_m_pressure=1.0)
       riot.input("region1", name="rose", mask_type="python", matid=0)

       riot.input("physics", hydro=True)
       riot.input("hydro", recon="plm", cfl=0.8, riemann="hllc")

       riot.input("rose/params", mode="sin", a=0.6, k=3)

   class rose:
       def __init__(self):
           make_input()
           self.eos = riot.EOS("material1")

       def mask(self, pos):
           r = np.hypot(pos[:, self.x], pos[:, self.y])
           theta = np.atan2(pos[:, self.y], pos[:, self.x])
           base = np.sin(self.k * theta) if self.mode == "sin" else np.cos(self.k * theta)
           return r <= np.abs(self.a * base) + 1.0e-12

       def rho_func(self, pos):
           return 1.0 + 0.5 * (pos[:, self.x] ** 2 + pos[:, self.y] ** 2)

       def c_m_rho(self, pos, rho):
           rho[:] = self.rho_func(pos)

       def c_m_pressure(self, pos, press):
           rho = self.rho_func(pos)
           self.eos.PressureFromDensityInternalEnergy(rho, 2.0 / rho, press)

   if __name__ == "__main__":
       make_input()
       riot.input.generate_input()

Running ``python3 rose.py`` writes ``rose.rin``; the ``rose`` region carries ``mask_type`` ``= python``, so at run time RIOT instantiates the ``rose`` class and calls ``mask``, ``c_m_rho``, and ``c_m_pressure`` to initialize the petals.
