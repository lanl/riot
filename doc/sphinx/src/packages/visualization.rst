.. _`chap:visualization`:

Visualization
=============

RIOT can render images directly during a run through its built-in *volume ray
tracer*, ``riot_viz``. Rather than writing full-mesh dumps for an external tool,
``riot_viz`` casts a ray through the domain for every pixel of one or more
virtual *cameras*, integrates color and opacity along each ray on the device,
and writes finished PNG images on a schedule. It is the in-situ visualization
path: it only reads the evolved state, so it can be enabled or disabled without
changing the simulation result. Volume renderings, slice planes, and iso-value
contours are all produced by the same ray march and can be composited together
in a single image.

Concept
-------

``riot_viz`` is registered as a diagnostic (Chapter :ref:`chap:diagnostics`) and
is enabled by adding it to the ``packages`` list of the ``<diagnostics>`` block:

.. code:: python

   riot.input("diagnostics", packages=["riot_viz"])

Configuration then lives in the ``<riot_viz>`` block and its sub-blocks. A run
defines one or more *cameras* in numbered blocks ``<riot_viz/camera0>``,
``<riot_viz/camera1>``, …; each camera produces one image (or image sequence).
A camera references a list of *layers*, and each layer is its own input block
that describes one thing to draw — a volume rendering, a slice, or a set of
contours. Layers are composited front-to-back in the order listed, so a later
layer draws over earlier ones.

At the scheduled times, RIOT seeds a ray-tracer particle for every pixel of
every camera at the point where its ray enters the domain, marches those
particles cell-by-cell accumulating color and opacity, reduces the finished
pixels across MPI ranks, and writes each camera's image as
``<name>NNNN.png``, where ``NNNN`` is a zero-padded, four-digit dump index.
The ray march supports Cartesian, cylindrical, and spherical coordinates.

Scheduling
----------

The render cadence is set in the ``<riot_viz>`` block. Provide **either** a
fixed interval ``dt`` **or** an explicit list of times ``t`` — setting both is
an error, and setting neither is an error.

.. list-table:: Parameters in the ``<riot_viz>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 22 12 16 50

   * - Parameter
     - Type
     - Default
     - Description
   * - dt
     - Real
     - ``-1``
     - Render every ``dt`` in simulation time. Mutually exclusive with ``t``.
   * - t
     - list
     - *empty*
     - Explicit list of render times. Mutually exclusive with ``dt``.
   * - timebar
     - bool
     - ``false``
     - Draw a progress bar and time stamp across the bottom of each image.
   * - timebar_rgb
     - list
     - ``1,1,1``
     - RGB color (each in :math:`[0,1]`) of the time bar.

Cameras
-------

Each ``<riot_viz/cameraN>`` block places a pinhole camera in the domain and
lists the layers it draws. The camera looks from ``location`` toward ``focus``;
``up`` fixes the roll, and ``target_width`` / ``target_height`` set the size of
the focal-plane window (in simulation length units) that is sampled by
``nwidth`` × ``nheight`` pixels. A single point light is placed at
``light_x,y,z`` and shades slices and contours through the ``light_ambient`` and
``light_diffuse`` terms; volume layers are emissive and ignore the light.

.. list-table:: Parameters in each ``<riot_viz/camera``\ :math:`N`\ ``>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 24 12 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - name
     - string
     - —
     - Output filename stem; images are ``<name>NNNN.png``.
   * - layers
     - list
     - —
     - Names of the layer blocks to draw, composited in order.
   * - location
     - list
     - —
     - Camera position ``x,y,z``.
   * - focus
     - list
     - —
     - Point the camera looks at.
   * - up
     - list
     - ``0,0,1``
     - Up vector (fixes camera roll).
   * - target_width
     - Real
     - —
     - Width of the focal-plane window in length units.
   * - target_height
     - Real
     - —
     - Height of the focal-plane window in length units.
   * - nwidth
     - int
     - —
     - Image width in pixels.
   * - nheight
     - int
     - —
     - Image height in pixels.
   * - opacity_threshold
     - Real
     - ``1e-2``
     - Remaining transparency at which a ray stops marching (early ray
       termination).
   * - light_x, light_y, light_z
     - Real
     - ``0``
     - Position of the point light used to shade slices and contours.
   * - light_ambient
     - Real
     - ``1``
     - Ambient (unshaded) lighting term.
   * - light_diffuse
     - Real
     - ``0``
     - Diffuse lighting coefficient (falls off as :math:`1/r^2`).
   * - colorbar_thickness
     - int
     - ``10``
     - Height in pixels of each color bar appended below the image.

Layers
------

A layer block is named by the string that appears in a camera's ``layers``
list; its ``type`` selects what is drawn:

``volume``
    Emission/absorption volume rendering. Along each ray the field is sampled,
    mapped to color and opacity through the transfer functions, and composited.

``slice``
    A single planar cut. The plane is defined by ``slice_location`` and
    ``slice_normal``; where the ray crosses the plane the field is colored and
    lit.

``contour``
    Iso-surfaces of the field. Each value in ``contours`` is drawn as a lit,
    colored surface with its own opacity.

``contour_slice``
    Like ``contour``, but the surfaces are colored from the transfer function
    (the color-map) rather than from per-contour colors.

Fields and scaling
~~~~~~~~~~~~~~~~~~~

The scalar drawn by a layer is named with ``field``. A separate field may drive
opacity through ``field_alpha`` (it defaults to ``field``), which lets a volume
be colored by one quantity and made transparent by another. Either field may be
sampled as its value, its gradient magnitude, or the magnitude of the gradient
of its logarithm, via ``field_use_grad`` / ``field_alpha_use_grad``. The color
axis may be linear or logarithmic through ``field_scale``.

Transfer functions
~~~~~~~~~~~~~~~~~~~

Color and opacity are piecewise-linear transfer functions given as lists of
control points that are spread evenly over the mapped data range. The ``red``,
``green``, ``blue`` lists (each value in :math:`[0,1]`) define the color map
over ``[min_value, max_value]``; the ``alpha`` list defines opacity over
``[min_alpha_value, max_alpha_value]``. The three color lists need not have the
same length as one another or as ``alpha``.

.. list-table:: Common layer parameters.
   :class: wraptable
   :header-rows: 1
   :widths: 26 12 16 46

   * - Parameter
     - Type
     - Default
     - Description
   * - type
     - string
     - —
     - ``volume``, ``slice``, ``contour``, or ``contour_slice``.
   * - field
     - string
     - —
     - Name of the field to draw (e.g. ``c.c.bulk.rho``).
   * - field_alpha
     - string
     - ``field``
     - Field that drives opacity.
   * - field_use_grad
     - string
     - ``none``
     - ``none``, ``magnitude``, or ``log_magnitude`` transform of ``field``.
   * - field_alpha_use_grad
     - string
     - ``none``
     - Same transform options, applied to ``field_alpha``.
   * - field_scale
     - string
     - ``linear``
     - ``linear`` or ``log`` scaling of the color axis.
   * - label
     - string
     - ``field``
     - Text label drawn on the layer's color bar.
   * - colorbar
     - bool
     - ``false``
     - Append a labeled color bar for this layer below the image.
   * - red, green, blue
     - list
     - ``0,0``
     - Color-map control points in :math:`[0,1]`.
   * - alpha
     - list
     - ``0,0``
     - Opacity control points.
   * - min_value, max_value
     - Real
     - ``0``, ``1``
     - Data range mapped by the color map.
   * - min_alpha_value, max_alpha_value
     - Real
     - ``min_value``, ``max_value``
     - Data range mapped by the opacity map.
   * - masks
     - list
     - *empty*
     - Region blocks that restrict where the layer is drawn (see below).

.. list-table:: Slice and contour parameters.
   :class: wraptable
   :header-rows: 1
   :widths: 26 12 16 46

   * - Parameter
     - Type
     - Default
     - Description
   * - slice_location
     - list
     - —
     - A point ``x,y,z`` on the slice plane (``slice`` layers).
   * - slice_normal
     - list
     - —
     - Normal ``x,y,z`` of the slice plane (``slice`` layers).
   * - slice_alpha
     - Real
     - ``1``
     - Opacity of a ``slice`` layer, clamped to :math:`[0,1]`.
   * - contours
     - list
     - *empty*
     - Iso-values to draw (``contour`` / ``contour_slice``).
   * - contour_red, contour_green, contour_blue
     - list
     - *empty*
     - Per-contour colors; each must match the length of ``contours``.
   * - contour_alpha
     - list
     - *empty*
     - Per-contour opacities in :math:`[0,1]`; must match ``contours``.

.. note::

   When ``contours`` is non-empty, ``contour_red``, ``contour_green``,
   ``contour_blue``, and ``contour_alpha`` must each have exactly the same
   number of entries as ``contours``, or initialization fails.

Masks
~~~~~

A layer may be restricted to part of the domain by listing one or more region
blocks in its ``masks`` parameter. Each named block is a geometric region using
the same ``mask_type`` shapes and parameters described in Chapter
:ref:`chap:regions` (spheres, shells, cylinders, boxes, and so on, each with an
optional ``invert``). The layer is drawn only where **all** listed masks are
satisfied, which makes cutaways and half-domain views straightforward.

Example
-------

A single camera that composites a volume rendering of density (colored by
density, made transparent by pressure) over a density slice through the
mid-plane, each restricted to one half of the domain by a rectangular mask:

.. code:: python

   riot.input("diagnostics", packages=["riot_viz"])

   riot.input("riot_viz", dt=0.002)

   riot.input(
       "riot_viz/camera0",
       name="image",
       location=[5, -2, 0],
       focus=[0, 0, 0],
       up=[0, 0, 1],
       target_width=2.1,
       target_height=2.1,
       nwidth=1024,
       nheight=1024,
       opacity_threshold=1.e-2,
       layers=["rho_volume", "rho_slice"],
       light_x=5.0, light_y=-2.0, light_z=1.0,
       light_ambient=0.5, light_diffuse=30.0,
   )

   riot.input(
       "rho_volume",
       type="volume",
       field="c.c.bulk.rho",
       field_alpha="c.c.bulk.pressure",
       red=[0.0, 1.0],
       green=[0.0, 0.0, 1.0, 0.0, 0.0],
       blue=[1.0, 0.0],
       alpha=[0.0, 4.0, 6.0],
       min_value=1.0, max_value=4.0,
       min_alpha_value=1.0, max_alpha_value=30.0,
       masks="right_half",
   )

   riot.input(
       "rho_slice",
       type="slice",
       field="c.c.bulk.rho",
       slice_location=[0.0, 0.0, 0.0],
       slice_normal=[1.0, 0.0, 0.0],
       red=[0.0, 1.0],
       green=[0.0, 0.0, 1.0, 0.0, 0.0],
       blue=[1.0, 0.0],
       min_value=1.0, max_value=4.0,
       masks="left_half",
   )

   riot.input("right_half", mask_type="inside_rectangle", y0=0.0)
   riot.input("left_half",  mask_type="inside_rectangle", y1=0.0)

To draw iso-value contours instead, give a ``contour`` layer a list of
``contours`` with matching per-contour colors and opacities:

.. code:: python

   riot.input(
       "rho_contours",
       type="contour",
       field="c.c.bulk.rho",
       contours=[0.75, 2.0, 4.0],
       contour_red=[1, 0, 0],
       contour_green=[0, 1, 0],
       contour_blue=[0, 0, 1],
       contour_alpha=[0.3, 0.7, 1.0],
   )
