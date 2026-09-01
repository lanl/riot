.. _`chap:gravity`:

Gravity
=======

The ``gravity`` package adds a gravitational body force to the hydrodynamics of Chapter :ref:`chap:hydro`. It sources the bulk momentum and total energy without introducing any transported field of its own, and requires hydrodynamics to be enabled.

Governing Equations
-------------------

Gravity enters through a gravitational potential :math:`\Phi`, which exerts an acceleration :math:`\bm{g} = -\nabla\Phi` on the fluid. This acceleration appears as a source on the right-hand sides of the bulk momentum and total-energy equations of Chapter :ref:`chap:hydro`,

.. math::

   \begin{align}
     \frac{\partial \left(\rho\bm{v}\right)}{\partial t}
       + \nabla\!\cdot\!\left(\rho\bm{v}\otimes\bm{v}+ p\,\bm{I} - \bm{s}\right)
       &= -\,\rho\,\nabla\Phi, \\[2pt]
     \frac{\partial E}{\partial t}
       + \nabla\!\cdot\!\left[\left(E + p\right)\bm{v}- \bm{s}\!\cdot\!\bm{v}\right]
       &= -\,\rho\,\bm{v}\!\cdot\!\nabla\Phi,
   \end{align}

where :math:`\rho` is the bulk density and :math:`\bm{v}` the common cell velocity. The momentum source is the body force per unit volume, :math:`-\rho\nabla\Phi`, and the energy source is the rate at which that force does work on the fluid, :math:`-\rho\,\bm{v}\!\cdot\!\nabla\Phi`. The gravitational force acts on the bulk fluid as a whole; it does not distinguish between the materials in a mixed cell.

RIOT currently supports a spatially uniform acceleration directed along a single mesh axis. Writing :math:`g` for the (signed) acceleration and :math:`\hat{\bm{e}}_d` for the chosen axis, the potential is :math:`\Phi = -g\,x_d` so that

.. math::

   \begin{equation}
     \bm{g} = -\nabla\Phi = g\,\hat{\bm{e}}_d ,
   \end{equation}

and the sources above reduce to :math:`\rho\,g` on the :math:`d`-component of the momentum and :math:`\rho\,v_d\,g` on the energy. The acceleration magnitude and direction are set by ``gravity_g`` and ``gravity_dim``, respectively.

Input Parameters
----------------

Gravity is enabled with the ``gravity`` toggle in the ``<physics>`` block (Section :ref:`sec:physics-block`); ``hydro`` must also be enabled. The acceleration is configured in the ``<gravity>`` block.

.. container:: paramtable

   | Parameters in the ``<gravity>`` block. gravity_dim & int & ``2`` & Axis along which gravity acts (:math:`0=x`, :math:`1=y`, :math:`2=z`).
   | gravity_g & Real & ``-9.7998e2`` & Signed gravitational acceleration :math:`g` (cm/s\ :math:`^2`).

Example
-------

A uniform downward gravitational acceleration along the :math:`y` axis:

.. code:: python

   riot.input("physics", hydro=True, gravity=True)
   riot.input("gravity", gravity_dim=1, gravity_g=-9.7998e2)
