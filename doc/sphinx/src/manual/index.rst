RIOT User Manual
================

This is the Sphinx/RST edition of the legacy LaTeX RIOT user manual. The
chapter source files were converted with Pandoc and are organized below to
retain the structure of the original manual.

.. figure:: _images/cover.png
   :alt: RIOT User Manual cover
   :width: 70%
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: User Manual

   introduction

.. toctree::
   :maxdepth: 1
   :caption: Physics Packages

   materials
   hydro
   strength
   ionization
   laser
   levelsets
   scalars
   gravity
   prescribed_sources
   tracers
   mix
   radiation_transport
   radiation_diffusion
   tnburn
   sparse_physics

.. toctree::
   :maxdepth: 1
   :caption: Problem Setup

   regions
   python_interface

.. toctree::
   :maxdepth: 1
   :caption: Programmer Guide

   programmer_guide
   diagnostics
   acknowledgements
