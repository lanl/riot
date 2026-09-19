.. This file was made in part with generative AI.

.. _`chap:mhd`:

Ideal Magnetohydrodynamics
==========================

The ``mhd`` package adds ideal magnetohydrodynamics to the hydro solver
of Chapter :ref:`chap:hydro`. The magnetic field is stored
**face-centered** — one component normal to each cell face — and is
advanced by **constrained transport** (CT), so that the discrete
divergence of the field is conserved to roundoff for the life of the
run rather than merely kept small by a cleaning step.

The capability is deliberately narrow. It is supported for
**Cartesian, uniform-grid, single-material, ideal-gas, single-temperature**
problems, and *every* other configuration is refused at startup with an
actionable message. Section :ref:`sec:mhd-support` lists what is
supported and what is rejected, and why. This is ideal MHD: there is no
resistivity, no Hall term, no Biermann battery, and no anisotropic
magnetized transport.

Governing Equations
-------------------

With a single material of density :math:`\rho` and a common velocity
:math:`\vec{v}`, RIOT integrates

.. math::

     \frac{\partial \rho}{\partial t} + \nabla\!\cdot\!\left(\rho\vec{v}\right) &= 0, \\[2pt]
     \frac{\partial \left(\rho\vec{v}\right)}{\partial t}
       + \nabla\!\cdot\!\left[\rho\vec{v}\otimes\vec{v}
         + \left(p + \frac{|\vec{B}|^2}{2\mu_0}\right)\mathsf{I}
         - \frac{\vec{B}\otimes\vec{B}}{\mu_0}\right] &= \vec{0}, \\[2pt]
     \frac{\partial E}{\partial t}
       + \nabla\!\cdot\!\left[\left(E + p + \frac{|\vec{B}|^2}{2\mu_0}\right)\vec{v}
         - \frac{\vec{B}\left(\vec{v}\cdot\vec{B}\right)}{\mu_0}\right] &= 0, \\[2pt]
     \frac{\partial \vec{B}}{\partial t}
       + \nabla\!\times\!\left(\vec{E}\right) &= \vec{0},
       \qquad \vec{E} = -\,\vec{v}\times\vec{B},

subject to the involution :math:`\nabla\!\cdot\!\vec{B}= 0`. The
induction equation is written in curl form because that is how it is
discretized: the field lives on faces, the electric field
:math:`\vec{E}` (the EMF) lives on edges, and the update is a discrete
Stokes theorem.

.. _`sec:mhd-energy`:

The Total-Energy Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   When MHD is enabled, ``ccbulk::total_material_energy`` includes the
   magnetic energy density:

   .. math::

        E = u + \tfrac{1}{2}\rho|\vec{v}|^2 + \frac{|\vec{B}|^2}{2\mu_0} .

   With MHD off it is the hydro quantity :math:`E = u +
   \tfrac{1}{2}\rho|\vec{v}|^2` of Chapter :ref:`chap:hydro`.
   **Analysis scripts must not add** :math:`|\vec{B}|^2/2\mu_0`
   **to the dumped total energy — it is already there.** The magnetic
   part is dumped separately as ``ccbulk::magnetic_energy`` so that the
   partition can be checked; the identity to verify is
   :math:`E - u - \tfrac{1}{2}\rho|\vec{v}|^2 - E_{\rm mag} = 0`, not a sum.

Folding the magnetic energy into the single evolved total is what keeps
total-energy conservation at machine precision: there is one
conservative update, not a hydro update plus a magnetic correction. The
internal energy handed to the equation of state is recovered by
*subtracting* both the kinetic and the magnetic parts, so magnetic
energy never reaches an EOS call as heat.

.. _`sec:mhd-units`:

Units and the ``mu0`` Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RIOT is CGS throughout, and the magnetic normalization is carried by
the single parameter ``mhd/mu0`` appearing as :math:`\mu_0` in every
equation above.

============= ============================= ==================================
``mhd/mu0``   Field units                   Magnetic energy density
============= ============================= ==================================
:math:`4\pi`  Gauss (Gaussian CGS, default) :math:`|\vec{B}|^2/8\pi`
:math:`1`     scale-free, normalized        :math:`|\vec{B}|^2/2`
============= ============================= ==================================

Both conventions are exercised by the unit tests, and identical
physical states expressed in the two conventions are verified to give
the same magnetic energy, fast speed, and time-step bound. The
normalized choice ``mu0 = 1`` is what the MHD regression problems use,
since it matches the convention of the standard MHD test problems in
the literature and so needs no unit reconciliation against published
reference solutions.

Numerical Method
----------------

Reconstruction and the Shared Normal Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three components of the cell-centered magnetic field are
reconstructed to the faces with the same method selected by
``hydro/recon``. The component *normal* to the face is then
**overwritten in both the left and right states by the single
face-centered value**, because that component is not a two-valued
interface quantity — it is continuous by construction. This is what
makes the normal-field flux vanish identically and keeps the EMF
assembly consistent with the field it updates.

All five reconstruction methods of Chapter :ref:`chap:hydro`
(``constant``, ``plm``, ``ppm4``, ``weno5``, ``mp5``) are certified for
use with MHD; none is excluded. Two measured caveats are worth knowing
before choosing one:

- ``weno5`` and ``mp5`` reach roughly **fourth** order, not fifth, on a
  smooth circularly polarized Alfvén wave. The cap is not the
  reconstruction (the two agree with each other to five digits) but the
  second-order corner averaging in the CT electromotive force.
- With the default ``rk2`` integrator at a fixed Courant number, the
  temporal error dominates once the spatial error is small, so
  ``weno5`` and ``mp5`` show only first-order *apparent* convergence
  and buy nothing over ``ppm4`` asymptotically. They remain
  dramatically more accurate at coarse resolution — a factor of
  60–80 lower :math:`L_1` at 16 cells per wavelength — which is usually
  where it matters. A user who needs fifth-order convergence needs a
  higher-order time integrator, or a Courant number that shrinks faster
  than :math:`\Delta x`.

Riemann Solvers
~~~~~~~~~~~~~~~

MHD uses its own Riemann solvers, selected by ``hydro/riemann``. The
hydro-only solvers have no magnetic terms and are rejected when MHD is
on; likewise an MHD solver is rejected when MHD is off.

.. container::
   :name: tab:mhd-riemann

   .. table:: MHD Riemann solvers (``hydro/riemann``).

      ========== ====================================================================
      **Option** **Description**
      ========== ====================================================================
      mhd_hlld   Five-wave HLLD (Miyoshi & Kusano 2005); least diffusive. Requires
                 an ideal gas. Falls back to HLLE on degenerate states.
      mhd_hlle   Two-wave HLLE; robust, works with any equation of state.
      mhd_llf    Local Lax–Friedrichs (Rusanov); most diffusive. A deliberately
                 simple reference solver, with no degeneracy guards.
      ========== ====================================================================

The diffusivity ordering ``mhd_hlld`` < ``mhd_hlle`` < ``mhd_llf`` is
verified to be strictly monotone on every field of the Brio & Wu shock
tube. ``mhd_hlld`` is the recommended default; use ``mhd_hlle`` if the
problem needs a non-ideal equation of state, and ``mhd_llf`` only to
diagnose whether a feature is physical or an artifact of a
lower-diffusivity solver.

``mhd_hlld`` contains six degeneracy guards, each of which falls back
to HLLE for that one face. This is correct behavior, not a failure —
but it means the solver actually used can differ from the solver
requested. The ``hlld_fallback`` counter reports how often
(Section :ref:`sec:mhd-diagnostics`), and it should be consulted before
any HLLD-versus-HLLE comparison is interpreted. On a problem where the
normal field vanishes over much of the mesh, HLLD is running as HLLE
almost everywhere.

Constrained Transport
~~~~~~~~~~~~~~~~~~~~~

The face field is updated by the discrete curl of an edge-centered
electromotive force,

.. math::

     \Phi_f \leftarrow \Phi_f - \Delta t \sum_{e \in \partial f} s_{fe}\, L_e\, \varepsilon_e ,

where :math:`\Phi_f` is the magnetic flux through face :math:`f`,
:math:`L_e` the length of edge :math:`e`, and :math:`s_{fe}` the
orientation sign. Because every edge is shared by exactly the faces
whose boundaries contain it, and each contributes with opposite sign to
each, the sum of fluxes over a closed cell surface is *algebraically*
unchanged by the update. The divergence is therefore preserved exactly,
whatever the EMF is — accuracy depends on the EMF, but the constraint
does not.

The edge EMF is the Gardiner & Stone (2005) upwind construction: each
of the four face-centered EMFs surrounding an edge is extrapolated to
that edge with a transverse derivative selected by the sign of the mass
flux at that same face, and the four results are averaged. Using the
mass flux as the upwind indicator is what couples the induction
upwinding to the hydro solve, so the field and the fluid see the same
characteristic direction.

The low-storage Runge–Kutta stage coefficients are applied to the face
update exactly as the hydro update applies them to cell-centered state,
so the field and the gas remain on the same stage at all times. Face
ghost values ride the existing boundary exchange, and shared faces
between blocks are single-valued, which is why the discrete divergence
is identical on both sides of a block boundary.

Time Step
~~~~~~~~~

The Courant condition is unchanged in form from Chapter
:ref:`chap:hydro` — the directional sum
:math:`\Delta t = \mathrm{cfl}\big/\sum_d (\lambda_d/\Delta x_d)` with
``hydro/cfl`` — but the acoustic speed is replaced by the **fast
magnetosonic speed**

.. math::

     c_f^2 = \tfrac{1}{2}\left(a^2 + v_A^2\right)
       + \tfrac{1}{2}\sqrt{\left(a^2 + v_A^2\right)^2 - 4\,a^2 v_{A,n}^2},

with :math:`a` the sound speed, :math:`v_A^2 =
|\vec{B}|^2/(\mu_0\rho)`, and :math:`v_{A,n}` the Alfvén speed built
from the field component along the sweep direction. The discriminant is
floored at zero so the exactly degenerate case :math:`a = v_A` is
finite rather than a NaN.

.. _`sec:mhd-diagnostics`:

Divergence and Solver Diagnostics
---------------------------------

``ccbulk::div_magnetic_field`` is available for output at all times. It
is computed as the net magnetic **flux** through the cell faces divided
by the cell volume — never from centered differences of the
cell-centered field, which would not vanish for a discretely
divergence-free field and so would be useless as a check.

Its magnitude scales like :math:`1/\Delta x`, so a raw
:math:`\max|\nabla\!\cdot\!\vec{B}|` is not comparable between meshes.
Setting ``mhd/monitor_divb`` reports three numbers once per step, after
the update:

- :math:`\max|\nabla\!\cdot\!\vec{B}|`,
- its volume-weighted mean, and
- the dimensionless :math:`\eta = |\nabla\!\cdot\!\vec{B}|\,
  \ell_{\rm cell} / \max(|\vec{B}|_{\rm cell}, \epsilon\,|\vec{B}|_{\max})`,
  which is the one to compare across resolutions and problems.

Expect :math:`\eta` at roundoff. A growing :math:`\eta` means a defect
in the port, not a tolerance to be relaxed; the correct response is
never to enable a divergence-cleaning step to conceal it.

.. warning::

   **Known defect.** With **four or more mesh blocks along a periodic
   axis**, the divergence constraint is violated at roundoff level
   :math:`\times 10^{8}`: :math:`\max|\nabla\!\cdot\!\vec{B}|` measures
   :math:`6\times10^{-8}` where the same problem on two blocks per axis
   gives :math:`7\times10^{-17}`. Adjacent blocks compute different
   updates for the shared face between them, and the block whose value
   is discarded no longer has a balanced divergence budget; constrained
   transport then preserves that error for the rest of the run.

   Until this is fixed, keep to **at most two mesh blocks along each
   periodic axis** — increase the meshblock size rather than splitting
   an axis further — and set ``mhd/monitor_divb`` to confirm
   :math:`\eta` stays at roundoff for your decomposition. Non-periodic
   axes are unaffected. The defect is independent of the
   reconstruction, the Riemann solver, the ghost width, and the MPI
   rank count.

Two solver-health counters are reported **unconditionally**, whenever
they increase, as a cumulative count and as a per-cell-per-step rate:

``hlld_fallback``
   How often ``mhd_hlld`` hit a degenerate state and used its HLLE
   fallback. A large rate means the run is effectively HLLE. This is
   not necessarily wrong, but a comparison between the two solvers on
   such a problem is meaningless.

``density_floor``
   How often a solver clamped a non-positive reconstructed density.
   Routine use of this floor is a **failed** result, not a rescued one:
   it means the reconstruction is producing unphysical states, and the
   resolution, Courant number, or reconstruction method should change.

Both counters are cumulative from :math:`t = 0` and are never reset.

Restarts
--------

Face-centered magnetic state is written to and read from checkpoints.
Reloading a checkpoint and integrating nothing reproduces the face
field, the conserved variables, and every magnetic derived quantity
**bitwise**; the derived primitives ``velocity`` and ``pressure`` differ
by one to two units in the last place, because they are recomputed from
the conserved state rather than read from the file. Continuing the
integration from a checkpoint agrees with the uninterrupted run to a
relative :math:`5.6\times10^{-15}`, and gives bit-for-bit the same
result whether or not the rank count changed across the restart.

Two restart configurations are refused:

- **Restarting across a change in** ``physics/mhd``, in either
  direction. Loading an MHD checkpoint into a hydro run reinterprets
  the magnetic energy as heat, and loading a hydro checkpoint into an
  MHD run starts from a zero field with a total energy that has no
  magnetic part; neither is a small error.
- **Restarting from a checkpoint whose field is identically zero**,
  because that is what the mistake above looks like. If a
  zero-field MHD run is genuinely intended, set
  ``mhd/allow_zero_field_restart``.

.. _`sec:mhd-support`:

Supported Configurations
------------------------

The certified matrix is Cartesian geometry, a uniform grid, a single
material, an ideal gas, and one temperature. Everything below is
**refused at startup** rather than run untested. These are not
statements that the coupling is impossible: each rejected package
writes or reads energy, adds a stress, or contributes a signal speed
that has not been audited against the magnetic terms.

.. list-table:: Configurations rejected when ``physics/mhd`` is true.
   :class: wraptable
   :header-rows: 1
   :widths: 34 66

   * - Configuration
     - Reason
   * - More than one material
     - The common-field, common-velocity model for a volume-additive mixture is a physics modeling assumption requiring scientific review.
   * - Non-Cartesian coordinates
     - Curvilinear face and edge metrics and the magnetic geometry source terms are not implemented.
   * - Mesh refinement (SMR/AMR)
     - Divergence-preserving prolongation and coarse/fine EMF correction are registered but untested, and there is no reference solution for them.
   * - General PTE closure, including any non-ideal ``eos_type``
     - The mixed-cell closure and the general-EOS acoustic derivative need separate validation. Note that a non-ideal ``eos_type`` turns this closure on regardless of the input flag.
   * - Fixed/frozen fluid
     - Induction transports the field with the fluid velocity, so a frozen background is not a consistent limit.
   * - Material strength
     - Combined deviatoric stress and signal speeds need review.
   * - Ionization / two temperature
     - Electron energy and entropy coupling to the magnetic terms needs separate validation.
   * - Mix, thermonuclear burn, level sets, multigroup diffusion, radiation transport, lasers, prescribed sources, gravity, passive scalars, tracer particles
     - Each needs its own energy, stress, or advection-consistency audit against the magnetic terms.
   * - Hydro-only Riemann solvers
     - ``hll``/``hllc``/``hllcf``/``chllc``/``lhllc`` contain no magnetic terms.

Not available at all, and not a matter of enabling a flag: resistive
induction, the Hall term, electron inertia, the Biermann battery,
ambipolar diffusion, Nernst transport, and anisotropic (Braginskii)
magnetized transport. Each is a new physics development.

GPU execution follows the same Kokkos portability rules as the rest of
RIOT and is reviewed, but the MHD package has not been run on a GPU.

Input Parameters
----------------

MHD is enabled with the ``mhd`` toggle in the ``<physics>`` block
(Section :ref:`sec:physics-block`); ``hydro`` must also be enabled, and
``hydro/riemann`` must be set to one of the MHD solvers. The remaining
controls live in the ``<mhd>`` block.

.. list-table:: Parameters in the ``<mhd>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 28 10 14 48

   * - Parameter
     - Type
     - Default
     - Description
   * - mu0
     - Real
     - :math:`4\pi`
     - Magnetic permeability in code units; magnetic energy density is :math:`|\vec{B}|^2/(2\mu_0)`. Must be positive. See Section :ref:`sec:mhd-units`.
   * - monitor_divb
     - bool
     - ``false``
     - Report max and volume-weighted-mean :math:`|\nabla\!\cdot\!\vec{B}|` and the dimensionless :math:`\eta` once per step, after the update. Costs two mesh-wide reductions per step.
   * - allow_zero_field_restart
     - bool
     - ``false``
     - Permit restarting from a checkpoint whose magnetic field is identically zero.

Registered Fields
-----------------

.. list-table:: Fields registered by the MHD package.
   :class: wraptable
   :header-rows: 1
   :widths: 30 12 14 44
   :name: tab:mhd-fields

   * - Field
     - Symbol
     - Components
     - Metadata / description
   * - fbulk::magnetic_field
     - :math:`\vec{B}`
     - 1 per face
     - Face, Independent, Conserved, WithFluxes, FillGhost; the evolved component normal to each face. Its automatically promoted edge-centered flux register *is* the EMF. This is the authoritative magnetic state.
   * - ccbulk::magnetic_field
     - :math:`\vec{B}`
     - 3
     - Cell, Derived, Intensive, OneCopy, FillGhost, WithFluxes, Vector; face-to-cell average, used by reconstruction. Its flux slots hold the transverse induction fluxes.
   * - ccbulk::magnetic_energy
     - :math:`|\vec{B}|^2/2\mu_0`
     - 1
     - Cell, Derived, Intensive, OneCopy, FillGhost, WithFluxes; magnetic energy density. Its flux slot holds the face magnetic pressure.
   * - ccbulk::div_magnetic_field
     - :math:`\nabla\!\cdot\!\vec{B}`
     - 1
     - Cell, Derived, OneCopy; diagnostic, from the face flux balance.

The face field is the state; the three cell-centered fields are
derived from it every time derived quantities are filled. Writing the
cell-centered field has no effect on the evolution.

Problem Generators
------------------

Four MHD problems are available, with input decks under ``inputs/mhd``:

============================= ======================================================
**Problem**                   **Description**
============================= ======================================================
``mhd_shock_tube``            Brio & Wu (1988) 1D shock tube, including the
                              compound wave. The standard MHD shock-capturing test.
``mhd_field_loop``            Advection of a weak field loop, planar 2D or tilted
                              3D. Tests EMF accuracy and the absence of spurious
                              out-of-plane field.
``mhd_orszag_tang``           Orszag & Tang (1979) vortex. Multi-dimensional CT
                              with shock formation.
``mhd_cpaw``                  Circularly polarized Alfvén wave, an exact nonlinear
                              solution. Used for order-of-accuracy measurement.
============================= ======================================================

Writing a New MHD Problem Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A generator initializes the gas state as usual and, in addition, must:

1. **Write the face field as a discrete curl of a vector potential.**
   Fill each face component from the circulation of :math:`\vec{A}`
   around that face. Doing so makes :math:`\nabla\!\cdot\!\vec{B}= 0`
   hold to machine precision at :math:`t = 0` *by construction*.
   Assigning an analytic :math:`\vec{B}` directly to the faces does not,
   in general, and CT will then preserve that initial error forever.
2. **Call** ``MHD::AddMagneticEnergyToTotal`` **last**, which converts
   ``ccbulk::total_material_energy`` from the hydro convention to the
   MHD convention of Section :ref:`sec:mhd-energy`. Omitting it leaves
   the run with too little total energy, which appears as a pressure
   error rather than as an obvious failure.

Everything else — the cell-centered field, the magnetic energy, and the
divergence — is derived automatically after the generator returns.

Example
-------

A 2D Orszag–Tang vortex with HLLD, normalized field units, and the
divergence monitor on:

.. code:: python

   riot.input("riot", problem="mhd_orszag_tang")
   riot.input("physics", hydro=True, mhd=True)
   riot.input("hydro", recon="plm", riemann="mhd_hlld", cfl=0.4)
   riot.input("mhd", mu0=1.0, monitor_divb=True)

   riot.input(
       "parthenon/output1",
       file_type="hdf5",
       variables=[
           "c.c.bulk.rho",
           "c.c.bulk.pressure",
           "c.c.bulk.total_material_energy",
           "c.c.bulk.magnetic_field",
           "c.c.bulk.magnetic_energy",
           "c.c.bulk.div_magnetic_field",
       ],
   )

Recall from Section :ref:`sec:mhd-energy` that the dumped
``total_material_energy`` already includes the magnetic contribution, so
``magnetic_energy`` is there to *decompose* it, not to be added to it.
