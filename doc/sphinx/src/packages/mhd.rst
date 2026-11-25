.. _`chap:mhd`:

Magnetohydrodynamics
====================

Setting ``mhd = true`` in the ``<physics>`` block extends the hydro
solver of Chapter :ref:`chap:hydro` to ideal MHD. The magnetic field
:math:`\vec{B}` is in Heaviside–Lorentz units, so the magnetic pressure
is :math:`B^2/2`, with no factors of :math:`4\pi` or :math:`\mu_0`. A
field :math:`\vec{B}_G` given in Gaussian units enters as :math:`\vec{B}
= \vec{B}_G/\sqrt{4\pi}`. With the magnetic terms the momentum and
energy equations become

.. math::

     \frac{\partial \left(\rho\vec{v}\right)}{\partial t}
       + \nabla\!\cdot\!\left[\rho\vec{v}\otimes\vec{v}
       + \left(p + \tfrac{1}{2}B^2\right)\mathsf{I}
       - \vec{B}\otimes\vec{B} - \mathsf{s}\right] &= \vec{0}, \\[2pt]
     \frac{\partial E}{\partial t}
       + \nabla\!\cdot\!\left[\left(E + p + \tfrac{1}{2}B^2\right)\vec{v}
       - \left(\vec{v}\!\cdot\!\vec{B}\right)\vec{B}
       - \mathsf{s}\!\cdot\!\vec{v}\right] &= 0, \\[2pt]
     \frac{\partial \vec{B}}{\partial t} + \nabla\!\times\!\vec{\mathcal{E}} &= \vec{0},
       \qquad \vec{\mathcal{E}} = -\vec{v}\times\vec{B},

where the total energy now includes the magnetic energy, :math:`E = u +
\tfrac{1}{2}\rho|\vec{v}|^2 + \tfrac{1}{2}B^2`. The bulk internal energy
passed to the PTE closure is recovered by subtracting both the kinetic
and the magnetic energy. The material densities, volume fractions and
the PTE closure are handled exactly as in the hydro case.

Constrained Transport
---------------------

The evolved field is face-centered: ``ffbulk::bfield`` stores the
component normal to each cell face. It is updated with constrained
transport: the EMF :math:`\vec{\mathcal{E}}` lives on cell edges, and
each face is updated by the circulation of :math:`\vec{\mathcal{E}}`
around its boundary (Stokes' theorem). This keeps the discrete
divergence of :math:`\vec{B}` at round-off for all time. The edge EMFs
are built from the face EMFs returned by the Riemann solver with the
upwind corner integration of Gardiner & Stone (2005). The upwind
direction at each face is chosen from the mass flux with the continuous
weight of Athena++. The EMFs are stored as the edge fluxes of
``ffbulk::bfield``, so Parthenon's flux correction keeps them consistent
across fine/coarse boundaries.

Riemann Solver and Reconstruction
---------------------------------

MHD always uses the HLLD solver. Any other ``riemann`` choice is
replaced by ``hlld`` with a warning, and ``hlld`` without ``mhd`` is an
error. The cell-centered field ``ccbulk::bfield`` (the average of the
face fields) is reconstructed with the other bulk primitives, using the
same ``recon`` and ``plm_slope_limit``. The field component normal to a
face is not reconstructed: both sides of the Riemann problem use the
face value from ``ffbulk::bfield``. The corner integration needs face
EMFs one cell beyond the block interior in each transverse direction, so
MHD needs one more ghost cell than the same reconstruction in pure
hydro. The time step uses the fast magnetosonic speed in each direction.

Geometry and Refinement
-----------------------

MHD runs in ``UniformCartesian`` and ``UniformCylindrical`` coordinates.
The CT update uses the true edge lengths and face areas, and in
cylindrical coordinates the cell-centered field is interpolated to the
cell centroid. The magnetic contributions to the geometric source terms
(the :math:`r`-momentum term :math:`\tfrac{1}{2}(B_r^2 + B_z^2 -
B_\phi^2)/r` and the corresponding :math:`B_r B_\phi` term in the
:math:`\phi`-momentum) are included. With static or adaptive refinement,
new fine faces are filled with the divergence-preserving prolongation of
Tóth & Roe (2002), and coarse faces are restricted by area averaging.
MHD cannot be combined with material strength, ionization, radiation
transport or the BHR mix model. Sparse physics is turned off
automatically when MHD is enabled.

Initial Conditions
------------------

The field is set from one of two optional methods of the Python class
named in the ``<regions>`` block (Chapter :ref:`chap:python`). Both fill
an :math:`(N,3)` output array in place:

- ``f_b_vector_potential(self, pos, a)`` returns the vector potential
  :math:`\vec{A}` at the cell edge centers. ``ffbulk::bfield`` is set to its
  discrete curl, so the initial :math:`\nabla\!\cdot\!\vec{B}` is zero to
  round-off. This is the preferred method.
- ``f_b_bfield(self, pos, b)`` returns :math:`\vec{B}` at the face
  centers; each face keeps its normal component. Initialization stops
  with an error unless the resulting discrete
  :math:`\nabla\!\cdot\!\vec{B}` vanishes to round-off. This suits fields
  that are uniform, or vary only along the face directions.

Without either method :math:`\vec{B} = 0`. Positions and vector
components follow the same Cartesian convention as the other state
setters: in cylindrical coordinates ``self.x``, ``self.y`` and
``self.z`` index the :math:`r`, :math:`\phi` and :math:`z` components.
The magnetic energy of the initial field is added to the total energy
automatically.

Diagnostics
-----------

The history file gains the total magnetic energy ``ME`` and the maximum
of :math:`|\nabla\!\cdot\!\vec{B}|\,\Delta x` (``max_divB_dx``, with
:math:`\Delta x` the smallest width of each cell), which should stay at
round-off. The cell-centered field ``ccbulk::bfield`` and the divergence
``ccbulk::divb`` can be added to the HDF5 outputs.

Example Decks
-------------

``inputs/mhd/`` holds the Brio–Wu shock tube (``brio_wu.py``), the
Orszag–Tang vortex (``orszag_tang.py``), field loop advection in 2D and
3D (``field_loop.py``, ``--3d`` for the loop inclined to the grid of
Gardiner & Stone 2008), and two cylindrical problems: field loop
advection in :math:`r`–:math:`z` (``field_loop_rz.py``) and a rotating
magnetized pinch equilibrium (``rotating_pinch.py``). Each deck takes
``--help``.

Input Parameters
----------------

.. list-table:: MHD parameter in the ``<physics>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - mhd
     - bool
     - ``false``
     - Evolve the ideal MHD equations with constrained transport.

The reconstruction (``recon``, ``plm_slope_limit``) and ``cfl`` of the
``<hydro>`` block apply to MHD as well (Chapter :ref:`chap:hydro`).

Registered Fields
-----------------

With MHD enabled the hydro package registers the fields below in
addition to those of Chapter :ref:`chap:hydro`. ``ffbulk`` is
``face_variables::face_averaged::bulk`` and ``ccbulk`` is
``cell_variables::cell_averaged::bulk``.

.. list-table:: Fields registered by the hydro package for MHD.
   :class: wraptable
   :header-rows: 1
   :widths: 30 14 16 40
   :name: tab:mhd-fields

   * - Field
     - Symbol
     - Components
     - Metadata / description
   * - ffbulk::bfield
     - :math:`\vec{B}\!\cdot\!\hat{n}`
     - —
     - Face, Independent, WithFluxes, FillGhost, Vector; face-normal magnetic field. The edge fluxes hold the EMFs.
   * - ccbulk::bfield
     - :math:`\vec{B}`
     - 3
     - Cell, Intensive, Vector, Derived, OneCopy; cell-centered magnetic field.
   * - ccbulk::divb
     - :math:`\nabla\!\cdot\!\vec{B}`
     - 1
     - Cell, Derived, OneCopy; discrete divergence of the field (diagnostic, filled before outputs).
   * - ffbulk::bnormal, ffbulk::vrho, ffbulk::emf
     - —
     - 1, 1, 2
     - Face, Derived, OneCopy, CellMemAligned; Riemann-solve scratch: a cell-aligned copy of the face-normal field, the upwind mass flux proxy and the two transverse face EMFs.
