.. _`chap:laser`:

Laser Ray Tracing
=================

The ``lasers`` package models laser energy deposition by geometric-optics ray tracing. Each laser beam is discretized into many sample rays that are launched at the domain boundary, refracted through the plasma by the electron-density gradient, and attenuated by inverse-bremsstrahlung absorption; the absorbed energy is deposited into the electron internal energy. Because the absorption and refraction depend on the electron density, temperature, and mean ionization, the package *requires the ionization package* (Chapter :ref:`chap:ionization`) to be active.

   The package supports Cartesian meshes (1D, 2D, or 3D) and 2D (:math:`r`–:math:`z`) cylindrical meshes; 3D cylindrical and other coordinate systems are not supported.

Governing Equations
-------------------

Beams and Rays
~~~~~~~~~~~~~~

A *beam* is defined by a lens position and a target position (which set its centerline and focusing), a wavelength, a transverse power profile, and a power-versus-time history. Each beam is sampled into many *rays* on an equal-area polar grid over its spot, each ray carrying a fraction of the beam power set by the profile weight. Rays are tracked as Lagrangian particles (Parthenon swarm particles) that carry a position, direction, energy, and wavelength, and are communicated between mesh blocks and MPI ranks as they cross the domain.

Ray Refraction
~~~~~~~~~~~~~~

Each ray follows the geometric-optics trajectory of a medium with refractive index :math:`n_r = \sqrt{1 - n_e/n_c}`, where :math:`n_e` is the electron number density and :math:`n_c` is the critical density for the ray wavelength :math:`\lambda`,

.. math::

     n_c(\lambda) = \frac{\pi\,m_e\,c^2}{\lambda^2 q_e^2}.

The ray bends toward lower density under an acceleration proportional to the electron-density gradient,

.. math::

     \frac{\mathrm{d}\vec{v}}{\mathrm{d}s}
       = -\,\frac{\lambda^2 q_e^2}{2\pi\,m_e}\,\nabla n_e ,

and travels at the local group speed :math:`c\,\sqrt{1 - n_e/n_c}`. The trajectory is integrated cell by cell with a velocity–Verlet push that steps exactly to each cell face. A ray that reaches the critical density (:math:`n_e \ge n_c`) deposits its remaining energy locally; there is no explicit specular reflection — the density gradient alone turns rays back.

Inverse-Bremsstrahlung Absorption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Along its path a ray is absorbed by electron–ion collisions. The absorption coefficient is

.. math::

     \kappa \;\propto\; \frac{n_e^2\,\bar{Z}\,\ln\Lambda}{n_c\,T_e^{3/2}\,
                             \sqrt{1 - n_e/n_c}} ,

where :math:`\bar Z` is the mean ionization state, :math:`T_e` the electron temperature, and :math:`\ln\Lambda` the Coulomb logarithm (fixed at :math:`7`). Over a path length :math:`\Delta s` through a cell the optical depth is :math:`\tau = \kappa\,\Delta s` and the ray loses the fraction

.. math::

     \Delta\mathcal{E} = \mathcal{E}\left(1 - e^{-\tau}\right)

of its energy (linearized to :math:`\mathcal{E}\,\tau` for small :math:`\tau`). That energy is deposited into the cell: it is accumulated as a deposition *rate* (``ccbulk::laser_deposition``) and, at the end of the step, integrated into the electron internal energy :math:`u_e` and the bulk total energy :math:`E`,

.. math::

     u_e \mathrel{+}= \Delta t\,\dot{q}_{\text{laser}}, \qquad
     E   \mathrel{+}= \Delta t\,\dot{q}_{\text{laser}} .

Setting ``enable_deposition`` ``= false`` traces the rays but suppresses the energy deposition (useful for visualizing beam paths).

Power History
~~~~~~~~~~~~~

Each beam’s ``power_watts``-versus-``time_ns`` table is converted to CGS, resampled onto a uniform time grid, and integrated to a cumulative energy. The energy released over a step is distributed among the beam’s rays by their profile weights, which are normalized so the whole beam power is captured.

Numerical Method
----------------

The laser update runs at the top of each cycle, before the hydrodynamic step. It interpolates the electron density to mesh nodes, spawns rays on the boundary blocks where each beam enters the domain, and then traces every ray to completion: rays are pushed cell by cell, depositing energy as they go, and those that cross a block or rank boundary are handed to Parthenon’s swarm communication and continued on the next iteration until no rays remain active. Ray tracing uses a *fast-light* approximation — each ray crosses the whole mesh within a single cycle, so there is no finite ray transit time.

After the trace, the package votes on the time step to keep the electron heating per step controlled: it limits :math:`\Delta t` so the deposited energy does not change the electron energy too abruptly (floored by ``dt_edot_floor``, and applied only where the optical depth exceeds ``dt_tau_cutoff``) and, when the fluid is evolving, so the heating does not violate the subsequent hydrodynamic CFL condition. The estimate is scaled by ``dt_safety``. When the fluid is held fixed (``fixed_fluid``), the laser deposition is the entire step.

Input Parameters
----------------

Lasers are enabled with ``lasers`` ``= true`` in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``ionization`` must also be enabled. Global controls live in the ``<laser>`` block; each beam is defined in its own contiguous, zero-based ``<laser0>``, ``<laser1>``, … block.

.. list-table:: Global parameters in the ``<laser>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - enable_deposition
     - bool
     - ``true``
     - Deposit absorbed energy; ``false`` traces rays without heating.
   * - node_interp_order
     - int
     - ``2``
     - Order of the cell-to-node electron-density interpolation (:math:`2` or :math:`4`).
   * - dt_safety
     - Real
     - ``0.95``
     - Safety factor on the laser-limited time step.
   * - dt_edot_floor
     - Real
     - ``0.2``
     - Floor on the electron-heating time-step factor.
   * - dt_tau_cutoff
     - Real
     - ``0.2``
     - Optical-depth threshold below which the heating time-step limit is disabled.

Each ``<laser>``\ :math:`N` block defines one beam. A beam may instead point to a separate input file with ``laser_file`` (read under the block named by ``name``, defaulting to ``laser``\ :math:`N`), which is convenient for reusing a beam definition across problems.

.. list-table:: Per-beam parameters in each ``<laser``\ :math:`N`\ ``>`` block.
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - laser_file
     - string
     - —
     - Optional separate file holding this beam’s parameters.
   * - name
     - string
     - ``laser``\ :math:`N`
     - Block name to read within ``laser_file``.
   * - time_ns
     - list
     - —
     - Times (ns) of the power history; required, :math:`\ge 2` entries.
   * - power_watts
     - list
     - —
     - Beam power (W) at each ``time_ns``; same length.
   * - wavelength_nm
     - Real
     - ``351.0``
     - Laser wavelength (nm).
   * - lens_x
     - list
     - —
     - Lens center position :math:`(x,y,z)`; required.
   * - target_x
     - list
     - —
     - Target/focus center position :math:`(x,y,z)`; required.
   * - target_size_ratio
     - Real
     - —
     - Target spot size relative to the lens spot (sets convergence).
   * - phi
     - Real
     - —
     - Roll angle (rad) of the beam’s transverse axes about the centerline.
   * - phi_axis
     - string
     - —
     - Reference axis for ``phi``: ``x``, ``y``, or ``z``.
   * - distribution
     - string
     - —
     - Transverse power profile: ``flat`` or ``super`` (super-Gaussian).
   * - power_semi_major_axis
     - Real
     - —
     - Spot semi-major axis.
   * - power_semi_minor_axis
     - Real
     - —
     - Spot semi-minor axis.
   * - power_super_exp
     - Real
     - —
     - Super-Gaussian order (required when ``distribution`` ``= super``).
   * - power_sample_frac
     - Real
     - ``0.999``
     - Fraction of beam power the sampled super-Gaussian spot must capture.
   * - grid_type
     - string
     - —
     - Ray sampling grid; ``equal_area``.
   * - nr
     - int
     - —
     - Number of radial rings in the sample grid.
   * - ntarget
     - int
     - —
     - Target total number of sample rays.

Registered Fields
-----------------

The package registers the diagnostic and coupling fields in the table below (all single-copy), plus the node-centered electron density used by the ray push and a swarm of ray particles. The deposited energy is carried by ``ccbulk::laser_deposition`` (a rate), integrated into the electron and total energy each step.

.. container:: fieldtable

   | Fields registered by the laser package.tab:laser-fields ccbulk::laser_deposition & :math:`\dot{q}_{\text{laser}}` & 1 & Cell, OneCopy; laser energy deposition rate.
   | ccbulk::laser_energy_density & — & 1 & Cell, OneCopy; path-averaged in-cell laser energy density (diagnostic).
   | ccbulk::laser_tau_max & :math:`\tau_{\max}` & 1 & Cell, OneCopy; maximum per-cell optical depth (drives the time step).
   | nv::electron_number_density & :math:`n_e` & 1 & Node, OneCopy, CellMemAligned; node-interpolated electron density for the ray push.
   | particles::laser & — & — & Swarm of ray particles carrying position, direction, energy, and wavelength.

Example
-------

A single flat-profile beam entering from the :math:`-x` boundary and focused along the :math:`x` axis, delivering :math:`2\,\mathrm{kJ}` over :math:`1\,\mathrm{ns}` into an ionized plasma:

.. code:: python

   riot.input("physics", hydro=True, ionization=True, lasers=True)

   riot.input("laser", enable_deposition=True, dt_safety=0.7,
              dt_edot_floor=1.0, dt_tau_cutoff=0.2)

   riot.input("laser0",
              lens_x=[-1.0, 0.0, 0.0],       # beam origin
              target_x=[ 1.0, 0.0, 0.0],     # focus (defines pointing)
              target_size_ratio=1.0,
              phi=0.0, phi_axis="z",
              distribution="flat",
              power_semi_major_axis=0.01,    # spot half-widths
              power_semi_minor_axis=0.01,
              grid_type="equal_area", nr=1, ntarget=10,
              time_ns=[0.0, 1.0],            # power history
              power_watts=[2.0e12, 2.0e12],  # 2 kJ / 1 ns
              wavelength_nm=530.0)
