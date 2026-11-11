Rshow PHDF plot viewer
======================

``rshow.py`` is an interactive viewer for one- and two-dimensional
Parthenon PHDF output. It supports multiple plot panels, dump playback,
publication-quality figure output, and MP4 or GIF movie export.

Requirements
------------

The viewer requires Python with Tkinter support, NumPy, Matplotlib, and the
Parthenon ``phdf`` reader. The reader must be importable as either ``phdf`` or
``parthenon_tools.phdf``.

MP4 export additionally requires ``ffmpeg``. On macOS it can be installed
with::

   brew install ffmpeg

GIF export requires Pillow::

   python -m pip install pillow

Running the viewer
------------------

Start with an empty viewer::

   python path/to/rshow.py

or open a particular dump immediately::

   python path/to/rshow.py path/to/problem.out0.00000.phdf

Opening a dump also discovers all ``*.phdf`` files in the same directory.
They become available in the **Dump** menu, through **Previous** and **Next**,
and for playback and movie export.

Basic plotting
--------------

1. Select **Open dump...** and choose a PHDF file.
2. Choose a field from **Variable**.
3. If the field has component or slice dimensions, enter one comma-separated
   integer per dimension in **Selector indices**. For example, ``4,0`` selects
   component 4 and z-plane 0. The field shape and number of selectable axes
   are displayed above the entry.
4. Adjust the limits or plotting options, then select **Plot**.

The viewer detects the mesh dimensionality. A one-dimensional mesh is drawn
as a line plot; a two-dimensional mesh is drawn with ``pcolormesh``. Selecting
a different dump or variable recalculates the ranges automatically.

Plot controls
-------------

``Color min`` and ``Color max`` set the color limits for a two-dimensional
plot. Equal limits and constant-valued fields are expanded automatically.
For one-dimensional plots, ``y min`` and ``y max`` set the vertical limits.
Blank limit fields use automatically determined values.

**Log scale** applies logarithmic color normalization in two dimensions and a
logarithmic y-axis in one dimension. Values at or below **Log floor** are not
used to determine an automatic logarithmic range.

**Block outlines** draws the boundary of each mesh block. **Equal aspect**
uses equal physical scaling in x and y. **Auto-scale plot styling** adjusts
fonts, tick marks, axes frames, and line widths as the window is resized. The
**Base font size** controls the reference size.

Use **Auto ranges** to restore data and coordinate limits. **Save figure...**
writes PDF, SVG, or PNG; PDF and SVG are preferable for publication figures.

Multiple panels
---------------

Use **Add...** under **Additional panels** to select another variable and its
selector indices. Additional panels work for both line and pseudocolor plots.

Select a panel and use **Edit...**, or double-click its list entry, to change
its variable or selector indices. **Remove** deletes the selected panel.
**Layout** can arrange panels left-to-right, top-to-bottom, or automatically
according to the window shape.

Playback and movie export
-------------------------

Set **FPS** and select **Play** to step through every PHDF file in the dump
directory. **Loop** restarts at the first dump after the last one.

Select **Export movie...** to render all discovered dumps with the current
variables, panels, limits, and styling. Use a filename ending in ``.mp4`` for
an MP4 movie through ffmpeg, or ``.gif`` for an animated GIF.

