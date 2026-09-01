.. _`chap:regions`:

Regions
=======

Most users interact with RIOT by specifying *initial conditions*, and in RIOT initial conditions are built from *regions*. A region is a geometric subdomain assigned to one or more materials together with an initial thermodynamic and kinematic state. The problem generator paints the mesh by evaluating each region over every cell; where regions overlap, higher-numbered regions take precedence. This chapter describes the region model and its text input-deck parameters; the Python interface in Chapter :ref:`chap:python` offers a more powerful way to define the same regions.

Concept
-------

Regions are declared in numbered input blocks ``<region0>``, ``<region1>``, …; the numeric suffix is the region *id*. Each region selects a geometric *mask* (a shape), a list of materials (``matid``), and an initial state. The problem generator processes regions in order of increasing id so that a later (higher-id) region overrides earlier ones where they overlap. By convention ``<region0>`` uses the ``background`` mask (which covers the whole domain) to provide a default state that subsequent regions carve into.

Where two regions meet within a single cell, RIOT can adaptively subdivide the cell to compute accurate volume fractions; the depth of this subdivision is controlled by ``nlev_min`` and ``nlev_max`` in the global ``<regions>`` block. Defaults placed in the ``<regions>`` block apply to every region unless overridden locally.

Region Shapes
-------------

The geometric mask is chosen with the ``mask_type`` parameter. The available shapes and their defining parameters are listed in the table below. All center/coordinate parameters default to ``0``; radii default to ``1``.

.. code-block:: text

   @P0.24 L0.66@ **mask_type & Shape and defining parameters
   background & Entire domain (default state); no parameters.
   inside_sphere & Sphere: center ``x0,y0,z0``, ``radius``.
   inside_spherical_shell & Spherical shell: center ``x0,y0,z0``, ``inner_radius``, ``outer_radius``.
   inside_cylinder & Finite cylinder: axis endpoints ``x0,y0,z0`` to ``x1,y1,z1``, ``radius``.
   inside_cylindrical_shell & Cylindrical shell: axis ``x0,y0,z0`` to ``x1,y1,z1``, ``inner_radius``, ``outer_radius``.
   inside_ellipsoid & Ellipsoid: center ``x0,y0,z0``, semi-axes ``ax,ay,az``.
   inside_ellipsoidal_shell & Ellipsoidal shell: center ``x0,y0,z0``, ``inner_ax..az``, ``outer_ax..az``.
   inside_rectangle & Axis-aligned box: bounds ``x0,y0,z0`` to ``x1,y1,z1`` (defaults :math:`\pm\infty`, i.e. a half-space or slab if only some bounds are set).
   python & Mask supplied by a user Python function (Chapter \ **\ :ref:`chap:python`\ **\ ).
   cad & Solid imported from a STEP CAD file: ``cadfile``, ``name`` (Section \ **\ :ref:`sec:cad`\ **\ ).
   **

Initial State
-------------

A region’s thermodynamic state is set by providing *two* independent thermodynamic quantities per material; RIOT infers the initialization mode from which pair is given. The settable material-averaged quantities are density (``c_m_rho``), pressure (``c_m_pressure``), temperature (``c_m_temperature``), and specific internal energy (``c_m_sie``). The supported combinations are summarized in the table below. The equation of state (Chapter :ref:`chap:materials`) closes the remaining variables and produces the conserved state.

.. code-block:: text

   @P0.42 L0.48@ **Provide & Sets state from
   c_m_rho + c_m_temperature & density and temperature (most common).
   c_m_rho + c_m_pressure & density and pressure.
   c_m_rho + c_m_sie & density and specific internal energy.
   c_m_pressure + c_m_temperature & pressure and temperature.
   **

When ionization is active, the electron temperature may be set independently (``c_c_bulk_electron_temperature``) or placed in equilibrium with the ions. The bulk velocity is set with ``c_c_bulk_velocity`` (a three-vector). Passive scalars are tagged in a region with ``passive_scalars``. In multi-material regions, per-material state is given by suffixing the material label (e.g. ``c_m_rho_Tungsten``), and volume fractions default such that each material fills its region.

Input Parameters
----------------

.. list-table:: Per-region parameters in each ``<region``\ :math:`N`\ ``>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - mask_type
     - string
     - —
     - Region shape (the table below).
   * - matid
     - int/list
     - —
     - Material id(s) present in the region.
   * - name
     - string
     - —
     - Optional label (also the Python class name).
   * - c_m_rho
     - Real
     - —
     - Initial density (with a second state variable).
   * - c_m_pressure
     - Real
     - —
     - Initial pressure.
   * - c_m_temperature
     - Real
     - —
     - Initial temperature.
   * - c_m_sie
     - Real
     - —
     - Initial specific internal energy.
   * - c_c_bulk_velocity
     - list
     - ``0,0,0``
     - Initial velocity vector.
   * - passive_scalars
     - list
     - —
     - Passive scalars tagged in this region.

Shape-specific geometry parameters (``x0``, ``radius``, etc.) are listed in the table below. The global ``<regions>`` block holds defaults and the overlap-refinement controls:

.. list-table:: Parameters in the global ``<regions>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - nlev_min
     - int
     - ``0``
     - Minimum subdivision level in cells spanning multiple regions.
   * - nlev_max
     - int
     - ``0``
     - Maximum subdivision level for computing overlap volume fractions.

Example
-------

A Sedov-like setup: a uniform background of material ``0`` with a small high-pressure cylinder at the origin. The regions are shown here in the text input-deck form to illustrate the block syntax; the equivalent Python calls (``riot.input("region0", …)``, etc.) are the recommended way to write them (Chapter :ref:`chap:python`).

::

   <regions>
   nlev_max = 5

   <region0>
   mask_type = background
   matid     = 0
   c_m_rho      = 1.0
   c_m_pressure = 0.1

   <region1>
   mask_type = inside_cylinder
   matid     = 0
   x0 = 0.0
   y0 = 0.0
   z0 = -0.75
   x1 = 0.0
   y1 = 0.0
   z1 = 0.75
   radius = 0.1
   c_m_rho      = 1.0
   c_m_pressure = 10.0
