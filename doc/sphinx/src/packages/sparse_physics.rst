.. _`chap:sparse-physics`:

Sparse Physics
==============

*Sparse physics* is a performance optimization distinct from the field sparsity of Section :ref:`sec:sparsity`: rather than saving memory where a material is absent, it saves *work* where the flow is not evolving. Large regions of many simulations are quiescent for long stretches — undisturbed ambient material, not-yet-shocked regions — and advancing them costs time without changing the solution. Sparse physics detects such regions and skips the physics update on the mesh blocks that cover them.

Method
------

The mechanism is built around a special sparse sentinel field, the cell delta :math:`\Delta` (``ccbulk::cell_delta``), whose allocation on a block marks that block as *active* (evolving). Each stage of the time integration accumulates into :math:`\Delta` a normalized measure of how much the conserved state changed over the update, summed over the state variables :math:`u`,

.. math::

   \begin{equation}
     \Delta = \sum_{u} \frac{\left|u^{\,\text{new}} - u^{\,\text{old}}\right|}
                           {\left|u^{\,\text{new}}\right| + 1} .
   \end{equation}

Where the solution is genuinely evolving, :math:`\Delta` is appreciable; where the flow is quiescent, :math:`\Delta` is nearly zero.

Parthenon’s sparse-field management then deallocates :math:`\Delta` on any block whose values remain below the deallocation threshold, and (re)allocates it where the change would exceed the allocation threshold. Because the physics tasks build their variable packs only over the blocks on which :math:`\Delta` is allocated, work is automatically confined to the active blocks: a task that finds no active blocks returns immediately, and loops over blocks visit only active ones. A quiescent block therefore carries no physics cost until activity reappears — for example, when a disturbance propagates in from an active neighbor — at which point :math:`\Delta` is reallocated and the block rejoins the update.

Input Parameters
----------------

Sparse physics is controlled in the ``<physics>`` block. The threshold sets the level of relative change required to keep a block active; its default value is appropriate for most problems, and it can be loosened to deactivate more aggressively or tightened to keep marginally-evolving regions active.

.. container:: paramtable

   | Parameters in the ``<physics>`` block. sparse_physics & bool & ``true`` & Skip the physics update on blocks where the flow is not evolving.
   | sparse_physics_threshold & Real & ``1e-12`` & Relative-change threshold :math:`\Delta` must exceed for a block to remain active.

Some packages are incompatible with sparse physics and require it to be disabled (``sparse_physics`` ``= false``); each notes this in its own chapter.

Example
-------

Enable sparse physics with a slightly looser activation threshold:

.. code:: python

   riot.input("physics", sparse_physics=True,
              sparse_physics_threshold=1.0e-10)
