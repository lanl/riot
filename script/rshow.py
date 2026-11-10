#!/usr/bin/env python3
# ========================================================================================
# (C) (or copyright) 2026-2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

"""Interactive PHDF viewer for 1-D/2-D Matplotlib plots, with run comparison.

3-D dumps are shown as 2-D slices: pick the xy, xz or yz plane and move the
slice through the domain with the slider (or type a position / step one cell).

Run with

    python phdf_plot_gui.py
    python phdf_plot_gui.py path/to/dump.phdf
    python phdf_plot_gui.py path/to/runA/dump.phdf path/to/runB/dump.phdf

The PHDF reader must be importable as either ``phdf`` or
``parthenon_tools.phdf``. Tkinter is included with most python.org and Conda
Python installations on macOS.

MP4 export requires the ``ffmpeg`` executable (e.g. ``brew install ffmpeg``).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import traceback
import math
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import h5py
import numpy as np
from cycler import cycler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

try:
    import phdf
except ImportError:
    try:
        from parthenon_tools import phdf
    except ImportError as exc:
        raise SystemExit(
            "Could not import the PHDF reader. Add Parthenon's "
            "scripts/python/packages to PYTHONPATH, or install parthenon_tools."
        ) from exc


COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

RUN_LABELS = ("A", "B")

# 3-D slicing: plane name -> (horizontal axis, vertical axis, normal axis),
# with axes numbered 0=x, 1=y, 2=z.
SLICE_PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
AXIS_NAMES = ("x", "y", "z")

# Resolution used when rendering movie frames (independent of savefig.dpi,
# which is intentionally very high for publication figures).
MOVIE_DPI = 200

# Largest raster (pixels per side) built for one 2-D panel.
MAX_RASTER_PIXELS = 8192

# Number of sliced/selected block sets kept per dump (slider positions etc.).
BLOCK_CACHE_SIZE = 8

# A valid movie with real frames should be larger than this many bytes.
MIN_VALID_MOVIE_BYTES = 1024

# Places to look for ffmpeg if it is not on the PATH seen by the GUI process.
FFMPEG_FALLBACK_PATHS = (
    "/opt/homebrew/bin/ffmpeg",  # Apple-silicon Homebrew
    "/usr/local/bin/ffmpeg",  # Intel Homebrew / manual installs
    "/usr/bin/ffmpeg",
    "/opt/local/bin/ffmpeg",  # MacPorts
)


def configure_ffmpeg() -> None:
    """Point Matplotlib at an ffmpeg executable if one can be found."""
    found = shutil.which("ffmpeg")
    if found is None:
        for candidate in FFMPEG_FALLBACK_PATHS:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                found = candidate
                break
    if found is not None:
        mpl.rcParams["animation.ffmpeg_path"] = found


def configure_publication_style() -> None:
    """Apply compact, vector-output-friendly plotting defaults."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "axes.linewidth": 1.0,
            "axes.prop_cycle": cycler(color=COLORS),
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Compact tick labels: "×10^7" offsets instead of long numbers.
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-3, 4),
        }
    )
    configure_ffmpeg()


def block_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges (block, cell + 1) from monotonic (block, cell) centers."""
    c = np.asarray(centers, dtype=float)
    if c.ndim == 1:
        c = c[None, :]
    if c.shape[1] == 1:
        return np.concatenate((c - 0.5, c + 0.5), axis=1)
    mids = 0.5 * (c[:, :-1] + c[:, 1:])
    first = c[:, :1] - (mids[:, :1] - c[:, :1])
    last = c[:, -1:] + (c[:, -1:] - mids[:, -1:])
    return np.concatenate((first, mids, last), axis=1)


def rasterize_blocks(
    blocks: list[np.ndarray],
    edges: list[tuple[np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_shape: tuple[int, int],
):
    """Composite per-block 2-D arrays into one image covering the view.

    Drawing one image is far cheaper than one QuadMesh per block, which is what
    dominates plotting time for AMR meshes (and remote X11 sessions).

    When the finest cells in view fit within ``max_shape`` the raster is aligned
    to the finest cell grid, so every cell (at every AMR level) maps onto whole
    pixels and the picture is exact. Otherwise the raster is capped at
    ``max_shape`` (roughly screen resolution) and cells are sampled at pixel
    centers; zooming in rebuilds the raster for the new view.

    Returns ``(image, (x0, x1, y0, y1))`` or ``None`` if no block is in view.
    Pixels not covered by any block are NaN (drawn transparent).
    """
    if not blocks:
        return None
    x0, x1 = sorted(float(v) for v in xlim)
    y0, y1 = sorted(float(v) for v in ylim)
    hlo = np.array([h[0] for h, _ in edges])
    hhi = np.array([h[-1] for h, _ in edges])
    vlo = np.array([v[0] for _, v in edges])
    vhi = np.array([v[-1] for _, v in edges])
    in_view = (hhi > x0) & (hlo < x1) & (vhi > y0) & (vlo < y1)
    if not in_view.any():
        return None
    visible = np.flatnonzero(in_view)
    x0, x1 = max(x0, hlo[visible].min()), min(x1, hhi[visible].max())
    y0, y1 = max(y0, vlo[visible].min()), min(y1, vhi[visible].max())
    if x1 <= x0 or y1 <= y0:
        return None
    dx = min((edges[b][0][-1] - edges[b][0][0]) / (len(edges[b][0]) - 1) for b in visible)
    dy = min((edges[b][1][-1] - edges[b][1][0]) / (len(edges[b][1]) - 1) for b in visible)

    def axis_grid(lo, hi, cell, origin, cap):
        count = math.ceil((hi - lo) / cell - 1e-6)
        if count <= cap:
            # Snap to the finest cell grid so pixel centers are cell centers.
            k0 = math.floor((lo - origin) / cell + 1e-6)
            k1 = math.ceil((hi - origin) / cell - 1e-6)
            lo, hi, count = origin + k0 * cell, origin + k1 * cell, max(k1 - k0, 1)
        else:
            count = cap
        return lo, hi, lo + (np.arange(count) + 0.5) * ((hi - lo) / count)

    x0, x1, xc = axis_grid(x0, x1, dx, hlo.min(), max_shape[0])
    y0, y1, yc = axis_grid(y0, y1, dy, vlo.min(), max_shape[1])
    image = np.full((yc.size, xc.size), np.nan)
    for b in visible:
        he, ve = edges[b]
        i0, i1 = np.searchsorted(xc, (he[0], he[-1]))
        j0, j1 = np.searchsorted(yc, (ve[0], ve[-1]))
        if i0 >= i1 or j0 >= j1:
            continue
        ii = np.clip(np.searchsorted(he, xc[i0:i1], side="right") - 1, 0, he.size - 2)
        jj = np.clip(np.searchsorted(ve, yc[j0:j1], side="right") - 1, 0, ve.size - 2)
        image[j0:j1, i0:i1] = blocks[b][np.ix_(jj, ii)]
    return image, (x0, x1, y0, y1)


def dump_series(path: Path) -> list[Path]:
    """Dumps of the same output series as ``path`` (e.g. ``radflow.out1.*.phdf``).

    A plain ``*.phdf`` glob would mix in other output blocks or problems that
    share the directory (``out2``, ``radflow_3d`` ...), which may have different
    variables or dimensionality.
    """
    match = re.fullmatch(r"(.*?)(\d+)", path.stem)
    if match is None:
        return sorted(path.parent.glob("*.phdf"))
    prefix = match.group(1)
    pattern = re.compile(re.escape(prefix) + r"\d+")
    series = [
        candidate
        for candidate in path.parent.glob(f"{glob_escape(prefix)}*.phdf")
        if pattern.fullmatch(candidate.stem)
    ]
    return sorted(series) or [path]


def glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


def dump_time(path: Path) -> float | None:
    """Simulation time stored in a dump, read cheaply (``None`` if unreadable)."""
    try:
        with h5py.File(path, "r") as handle:
            return float(handle["Info"].attrs["Time"])
    except Exception:
        return None


def dump_cache(dump) -> dict:
    """Per-dump scratch cache; it is discarded together with the dump object."""
    cache = getattr(dump, "_rshow_cache", None)
    if cache is None:
        cache = {}
        try:
            dump._rshow_cache = cache
        except AttributeError:
            pass
    return cache


def optional_float(text: str) -> float | None:
    text = text.strip()
    return None if text == "" else float(text)


def parse_indices(text: str, count: int) -> tuple[int, ...]:
    """Parse indices for all axes between block and the final y,x axes."""
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values and count:
        values = (0,) * count
    if len(values) != count:
        raise ValueError(
            f"This field needs {count} selector index/indices before (y,x); "
            f"enter {count} comma-separated values."
        )
    return values


def plottable_variables(dump) -> list[str]:
    """Return PHDF variables backed by array-like HDF5 datasets.

    Some PHDF reader versions expose metadata groups alongside field datasets
    through ``Variables``.  An h5py Group accepts name lookup but cannot be
    sliced with ``[:]``, which is what ``phdf.Get`` expects to do.
    """
    variables = []
    for raw_name in dump.Variables:
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        try:
            node = dump.fid[name]
            shape = getattr(node, "shape", None)
            # phdf.Get() always inspects the final three axes (z,y,x), even
            # for a 1-D mesh where z and y are singleton dimensions. Exclude
            # lower-rank block metadata before it reaches phdf.Get().
            if shape is None or len(shape) < 3:
                continue
            # Exclude coordinate/metadata arrays that are not block fields.
            if int(shape[0]) != int(dump.NumBlocks):
                continue
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        variables.append(name)
    return sorted(variables)


def dump_dimension(dump) -> int:
    """Infer whether a mesh is 1-D, 2-D, or 3-D from cells per block."""
    y = np.asarray(dump.y)
    if y.ndim < 2 or y.shape[-1] == 1:
        return 1
    z = np.asarray(dump.z)
    return 3 if z.ndim >= 2 and z.shape[-1] > 1 else 2


def dump_coordinate(dump, axis: int) -> np.ndarray:
    """Cell-center coordinates (block, cell) along axis 0=x, 1=y, 2=z."""
    return np.asarray((dump.x, dump.y, dump.z)[axis])


def dump_edges(dump, axis: int) -> np.ndarray:
    """Cached cell-edge coordinates (block, cell + 1) along axis 0=x, 1=y, 2=z."""
    cache = dump_cache(dump)
    key = ("edges", axis)
    if key not in cache:
        cache[key] = block_edges(dump_coordinate(dump, axis))
    return cache[key]


def exception_details(exc: BaseException) -> str:
    """Return the exception text plus any captured subprocess (ffmpeg) output."""
    detail = str(exc)
    extra = getattr(exc, "stderr", None) or getattr(exc, "output", None)
    if isinstance(extra, bytes):
        extra = extra.decode(errors="replace")
    if extra:
        extra = str(extra).strip()
        if extra and extra not in detail:
            detail += "\n\nffmpeg output (last lines):\n" + extra[-1500:]
    return detail


class PhdfPlotGui(tk.Tk):
    def __init__(
        self, initial_file: str | None = None, compare_file: str | None = None
    ) -> None:
        super().__init__()
        self.title("PHDF Plot Viewer")
        self.geometry("1050x900")
        self.minsize(850, 600)

        self.dump = None
        self.dump_path: Path | None = None
        self.dump_paths: list[Path] = []

        # Comparison run (run B)
        self.dump2 = None
        self.dump2_path: Path | None = None
        self.dump2_paths: list[Path] = []
        self._dump2_times: dict[Path, float | None] = {}

        self.field_data: np.ndarray | None = None
        self.colorbar = None
        self.colorbars = []
        self.main_axes = None
        self.mesh_lines = []
        self.data_lines = []
        self.line_panels: list[dict[str, str]] = []
        self.line_panel_axes = []
        self.figure_title = None
        self._resize_job = None
        self._last_panel_grid_shape = None
        self._play_job = None
        self.playing = False
        self.exporting = False

        self.dump_var = tk.StringVar()
        self.variable_var = tk.StringVar()
        self.indices_var = tk.StringVar(value="0")
        self.shape_var = tk.StringVar(value="No field loaded")
        self.cmap_var = tk.StringVar(value="viridis")
        self.font_size_var = tk.StringVar(value="14")
        self.auto_style_var = tk.BooleanVar(value=True)
        self.fps_var = tk.StringVar(value="5")
        self.loop_var = tk.BooleanVar(value=True)
        self.panel_layout_var = tk.StringVar(value="Auto")
        self.log_var = tk.BooleanVar(value=False)
        self.mesh_var = tk.BooleanVar(value=False)
        self.equal_aspect_var = tk.BooleanVar(value=True)
        self.lock_colorscale_var = tk.BooleanVar(value=False)
        # Locked value ranges, keyed by (variable, indices, log): set the first
        # time a panel is drawn while "Lock value scale" is on.
        self._locked_ranges: dict[tuple, tuple[float, float]] = {}
        # Layout gutters only grow while the panel structure is unchanged, so
        # playback and movies do not jitter as tick labels change width.
        self._layout_memory: tuple | None = None
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()
        self.floor_var = tk.StringVar(value="1e-30")
        self.status_var = tk.StringVar(value="Choose a PHDF dump to begin")

        # Comparison controls
        self.compare_var = tk.BooleanVar(value=False)
        self.sync_time_var = tk.BooleanVar(value=True)
        self.dump2_label_var = tk.StringVar(value="Run B: (none)")

        # 3-D slice controls
        self.slice_plane_var = tk.StringVar(value="xy")
        self.slice_swap_var = tk.BooleanVar(value=False)
        self.slice_pos_var = tk.DoubleVar(value=0.0)
        self.slice_pos_text = tk.StringVar(value="")
        self.slice_axis_label = tk.StringVar(value="Position")
        self._slice_job = None
        self._slice_dx = 0.0
        self._slice_initialized = False

        # Rendering: one image per 2-D panel instead of one mesh per block.
        self.fast_render_var = tk.BooleanVar(value=True)
        self._raster_panels: list[dict] = []
        self._raster_job = None
        self._raster_dpi: float | None = None  # higher-res rasters for save/export

        self._build_controls()
        self._build_figure()

        self.protocol("WM_DELETE_WINDOW", self.close)
        if initial_file:
            self.after(50, lambda: self._startup(initial_file, compare_file))

    def _startup(self, initial_file: str, compare_file: str | None) -> None:
        """Load run A, then optionally run B given on the command line."""
        if compare_file:
            self._open_dump2(Path(compare_file), replot=False)
        self.load_dump(Path(initial_file))

    # ------------------------------------------------------------------ GUI
    def _build_controls(self) -> None:
        panel = ttk.Frame(self, padding=6)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.columnconfigure(1, weight=1)

        row = 0
        ttk.Button(panel, text="Open dump…", command=self.choose_dump).grid(
            row=row, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(panel, text="Reload", command=self.reload_dump).grid(
            row=row, column=1, sticky="ew", padx=2, pady=2
        )
        row += 1

        ttk.Label(panel, text="Dump (run A)").grid(
            row=row, column=0, sticky="w", pady=(8, 2)
        )
        row += 1
        self.dump_combo = ttk.Combobox(
            panel, textvariable=self.dump_var, state="readonly", width=33
        )
        self.dump_combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.dump_combo.bind("<<ComboboxSelected>>", self._dump_selected)
        row += 1

        nav = ttk.Frame(panel)
        nav.grid(row=row, column=0, columnspan=2, sticky="ew")
        nav.columnconfigure((0, 1), weight=1)
        ttk.Button(nav, text="Previous", command=lambda: self.step_dump(-1)).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        ttk.Button(nav, text="Next", command=lambda: self.step_dump(1)).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )
        row += 1

        playback = ttk.Frame(panel)
        playback.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        playback.columnconfigure(0, weight=1)
        self.play_button = ttk.Button(
            playback, text="▶ Play", command=self.toggle_playback
        )
        self.play_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Label(playback, text="FPS").grid(row=0, column=1, padx=(0, 2))
        ttk.Spinbox(
            playback,
            from_=0.1,
            to=60.0,
            increment=1.0,
            textvariable=self.fps_var,
            width=5,
        ).grid(row=0, column=2, padx=(0, 5))
        ttk.Checkbutton(playback, text="Loop", variable=self.loop_var).grid(
            row=0, column=3
        )
        row += 1

        # ---- Comparison run -------------------------------------------------
        compare = ttk.LabelFrame(panel, text="Comparison run (B)", padding=4)
        compare.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        compare.columnconfigure((0, 1), weight=1)
        ttk.Button(
            compare, text="Open second run…", command=self.choose_dump2
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(compare, text="Clear", command=self.clear_dump2).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )
        ttk.Label(
            compare, textvariable=self.dump2_label_var, wraplength=250
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Checkbutton(
            compare,
            text="Show side-by-side",
            variable=self.compare_var,
            command=self._compare_toggled,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            compare,
            text="Match B to A by time (else by index)",
            variable=self.sync_time_var,
            command=self._compare_toggled,
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(panel, text="Variable").grid(
            row=row, column=0, sticky="w", pady=(8, 2)
        )
        row += 1
        self.variable_combo = ttk.Combobox(
            panel, textvariable=self.variable_var, state="readonly", width=33
        )
        self.variable_combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.variable_combo.bind("<<ComboboxSelected>>", self._variable_selected)
        row += 1

        ttk.Label(panel, textvariable=self.shape_var, wraplength=260).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1
        self._entry_row(panel, row, "Selector indices", self.indices_var)
        row += 1
        ttk.Label(
            panel,
            text="Example: component,z = 4,0 (2-D) or component (3-D)",
            foreground="#666666",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        # ---- 3-D slice --------------------------------------------------------
        slicing = ttk.LabelFrame(panel, text="3-D slice", padding=4)
        slicing.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        slicing.columnconfigure(1, weight=1)
        ttk.Label(slicing, text="Plane").grid(row=0, column=0, sticky="w")
        self.slice_plane_combo = ttk.Combobox(
            slicing,
            textvariable=self.slice_plane_var,
            values=tuple(SLICE_PLANES),
            state="readonly",
            width=6,
        )
        self.slice_plane_combo.grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.slice_plane_combo.bind("<<ComboboxSelected>>", self._slice_plane_changed)
        self.slice_swap_check = ttk.Checkbutton(
            slicing,
            text="Swap axes",
            variable=self.slice_swap_var,
            command=self._slice_swap_toggled,
        )
        self.slice_swap_check.grid(row=0, column=2, sticky="e")
        self.slice_scale = ttk.Scale(
            slicing,
            orient=tk.HORIZONTAL,
            variable=self.slice_pos_var,
            from_=0.0,
            to=1.0,
            command=self._slice_scale_moved,
        )
        self.slice_scale.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        position = ttk.Frame(slicing)
        position.grid(row=2, column=0, columnspan=3, sticky="ew")
        position.columnconfigure(1, weight=1)
        ttk.Label(position, textvariable=self.slice_axis_label).grid(
            row=0, column=0, sticky="w"
        )
        self.slice_entry = ttk.Entry(
            position, textvariable=self.slice_pos_text, width=12
        )
        self.slice_entry.grid(row=0, column=1, sticky="ew", padx=4)
        self.slice_entry.bind("<Return>", self._slice_entry_set)
        self.slice_prev_button = ttk.Button(
            position, text="◀", width=2, command=lambda: self._slice_step(-1)
        )
        self.slice_prev_button.grid(row=0, column=2)
        self.slice_next_button = ttk.Button(
            position, text="▶", width=2, command=lambda: self._slice_step(1)
        )
        self.slice_next_button.grid(row=0, column=3)
        self._slice_widgets = (
            self.slice_plane_combo,
            self.slice_swap_check,
            self.slice_scale,
            self.slice_entry,
            self.slice_prev_button,
            self.slice_next_button,
        )
        self._set_slice_controls_enabled(False)
        row += 1

        line_panels = ttk.LabelFrame(panel, text="Additional panels", padding=4)
        line_panels.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        line_panels.columnconfigure((0, 1, 2), weight=1)
        ttk.Label(line_panels, text="Layout").grid(row=0, column=0, sticky="w")
        panel_layout = ttk.Combobox(
            line_panels,
            textvariable=self.panel_layout_var,
            values=("Auto", "Left–right", "Top–bottom"),
            state="readonly",
            width=15,
        )
        panel_layout.grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(0, 3)
        )
        panel_layout.bind("<<ComboboxSelected>>", self._panel_layout_changed)
        self.line_panel_list = tk.Listbox(
            line_panels, height=2, exportselection=False, activestyle="dotbox"
        )
        self.line_panel_list.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.line_panel_list.bind("<Double-Button-1>", self.edit_line_panel)
        ttk.Button(line_panels, text="Add…", command=self.add_line_panel).grid(
            row=2, column=0, sticky="ew", padx=(0, 2), pady=(3, 0)
        )
        ttk.Button(line_panels, text="Edit…", command=self.edit_line_panel).grid(
            row=2, column=1, sticky="ew", padx=2, pady=(3, 0)
        )
        ttk.Button(line_panels, text="Remove", command=self.remove_line_panel).grid(
            row=2, column=2, sticky="ew", padx=(2, 0), pady=(3, 0)
        )
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(panel, text="Color map").grid(row=row, column=0, sticky="w")
        self.cmap_combo = ttk.Combobox(
            panel,
            textvariable=self.cmap_var,
            values=(
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "RdBu_r",
                "coolwarm",
            ),
            width=16,
        )
        self.cmap_combo.grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        row += 1
        self._entry_row(panel, row, "Base font size", self.font_size_var)
        row += 1
        self._entry_row(panel, row, "Log floor", self.floor_var)
        row += 1

        checks = ttk.Frame(panel)
        checks.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            checks,
            text="Auto-scale plot styling",
            variable=self.auto_style_var,
            command=self._apply_plot_style,
        ).pack(anchor="w")
        ttk.Checkbutton(
            checks,
            text="Log scale",
            variable=self.log_var,
            command=self._log_toggled,
        ).pack(anchor="w")
        ttk.Checkbutton(checks, text="Block outlines", variable=self.mesh_var).pack(
            anchor="w"
        )
        ttk.Checkbutton(
            checks, text="Equal aspect", variable=self.equal_aspect_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            checks,
            text="Lock value scale across dumps",
            variable=self.lock_colorscale_var,
            command=self._lock_toggled,
        ).pack(anchor="w")
        ttk.Checkbutton(
            checks,
            text="Fast rendering (one image per panel)",
            variable=self.fast_render_var,
            command=self.plot,
        ).pack(anchor="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        for label, variable in (
            ("x min", self.xmin_var),
            ("x max", self.xmax_var),
            ("y min", self.ymin_var),
            ("y max", self.ymax_var),
        ):
            self._entry_row(panel, row, label, variable)
            row += 1

        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Plot", command=self._replot_fresh).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        ttk.Button(buttons, text="Auto ranges", command=self.auto_ranges).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )
        row += 1
        ttk.Button(panel, text="Save figure / movie…", command=self.save_figure).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )
        row += 1
        ttk.Button(panel, text="Export movie…", command=self.export_movie).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )

    @staticmethod
    def _entry_row(
        parent: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=17).grid(
            row=row, column=1, sticky="ew", padx=2, pady=2
        )

    def _build_figure(self) -> None:
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # Axes positions are managed by _layout_figure (measured gutters, uniform
        # aspect-correct panels) rather than by a Matplotlib layout engine.
        self.figure = Figure(figsize=(6, 5), dpi=100, layout="none")
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        # ``add='+'`` is essential: FigureCanvasTkAgg already owns a Configure
        # binding that resizes the Matplotlib figure with the Tk widget.
        # Replacing that binding leaves a fixed 600x500 figure in a large canvas.
        self.canvas_widget.bind("<Configure>", self._canvas_resized, add="+")
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        ttk.Label(right, textvariable=self.status_var, anchor="w").pack(
            fill=tk.X, padx=5
        )

    def _style_metrics(self, grid_shape: tuple[int, int] | None = None) -> tuple[float, float]:
        """Return responsive font and line sizes for the canvas and panel grid."""
        base_font = float(self.font_size_var.get())
        if base_font <= 0:
            raise ValueError("Base font size must be positive")
        scale = 1.0
        if self.auto_style_var.get():
            width = max(self.canvas_widget.winfo_width(), 1)
            height = max(self.canvas_widget.winfo_height(), 1)
            # 700 x 650 is approximately the plotting area in the default window.
            scale = float(
                np.clip(np.sqrt((width * height) / (700.0 * 650.0)), 0.85, 1.6)
            )
            # Smaller text when the canvas is split into more panels.
            rows, columns = grid_shape or self._last_panel_grid_shape or (1, 1)
            scale *= max(0.66, 1.0 - 0.12 * (max(rows, columns) - 1))
        font_size = base_font * scale
        line_width = 1.1 * scale
        return font_size, line_width

    def _canvas_resized(self, _event=None) -> None:
        """Debounce resize events so styling follows the window smoothly."""
        if self.exporting:
            return
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self._resize_job = None
        if (
            self.panel_layout_var.get() == "Auto"
            and self.dump is not None
            and (self.line_panels or self._comparison_active())
        ):
            new_shape = self._panel_grid_shape(self._panel_count())
            if new_shape != self._last_panel_grid_shape:
                self.plot()
                return
        self._apply_plot_style(draw=False)
        self._layout_figure()
        if self._raster_panels:
            self._rerender_rasters(draw=False)
        self.canvas.draw_idle()

    def _apply_plot_style(self, draw=True) -> None:
        """Scale plot text, ticks, spines, and outlines as one visual system."""
        self._resize_job = None
        if self.main_axes is None or self.main_axes not in self.figure.axes:
            return
        try:
            font_size, line_width = self._style_metrics()
        except (TypeError, ValueError):
            return
        tick_font = max(font_size - 1.5, 1.0)
        major_length = 0.38 * font_size
        minor_length = 0.22 * font_size

        all_axes = list(self.figure.axes)
        for axes in self.figure.axes:
            all_axes += axes.child_axes  # colorbars are attached as child axes
        for axes in all_axes:
            axes.tick_params(
                axis="both",
                which="major",
                labelsize=tick_font,
                length=major_length,
                width=line_width,
                pad=0.25 * font_size,
            )
            axes.tick_params(
                axis="both",
                which="minor",
                length=minor_length,
                width=0.72 * line_width,
            )
            for spine in axes.spines.values():
                spine.set_linewidth(line_width)
            axes.xaxis.label.set_fontsize(font_size)
            axes.yaxis.label.set_fontsize(font_size)
            axes.xaxis.offsetText.set_fontsize(tick_font)
            axes.yaxis.offsetText.set_fontsize(tick_font)
            axes.title.set_fontsize(0.9 * font_size)

        for colorbar in self.colorbars:
            colorbar.ax.yaxis.label.set_fontsize(font_size)
            colorbar.outline.set_linewidth(line_width)
        if self.figure_title is not None:
            self.figure_title.set_fontsize(font_size)
        for line in self.mesh_lines:
            line.set_linewidth(max(0.45, 0.45 * line_width))
        for line in self.data_lines:
            line.set_linewidth(max(1.4, 1.4 * line_width))

        if draw:
            self.canvas.draw_idle()

    def choose_dump(self) -> None:
        self.pause_playback()
        filename = filedialog.askopenfilename(
            title="Open a PHDF dump",
            filetypes=(("PHDF dumps", "*.phdf"), ("All files", "*")),
        )
        if filename:
            self.load_dump(Path(filename))

    def _dump_selected(self, _event=None) -> None:
        self.pause_playback()
        selected = self.dump_var.get()
        for path in self.dump_paths:
            if path.name == selected:
                self.load_dump(path)
                return

    def step_dump(self, amount: int) -> None:
        self.pause_playback()
        if not self.dump_paths:
            return
        try:
            index = self.dump_paths.index(self.dump_path)
        except ValueError:
            index = 0
        index = min(max(index + amount, 0), len(self.dump_paths) - 1)
        self.load_dump(self.dump_paths[index])

    def _fps(self) -> float:
        fps = float(self.fps_var.get())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be a positive number")
        return fps

    def toggle_playback(self) -> None:
        if self.playing:
            self.pause_playback()
            return
        try:
            self._fps()
            if len(self.dump_paths) < 2:
                raise ValueError("Open a directory containing at least two PHDF dumps")
            self.playing = True
            self.play_button.configure(text="⏸ Pause")
            self._schedule_next_frame()
        except Exception as exc:
            self._show_error("Could not start playback", exc)

    def pause_playback(self) -> None:
        self.playing = False
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="▶ Play")

    def _schedule_next_frame(self) -> None:
        if self.playing:
            delay_ms = max(1, round(1000.0 / self._fps()))
            self._play_job = self.after(delay_ms, self._play_next_frame)

    def _play_next_frame(self) -> None:
        self._play_job = None
        if not self.playing or not self.dump_paths:
            return
        try:
            try:
                index = self.dump_paths.index(self.dump_path)
            except ValueError:
                index = -1
            next_index = index + 1
            if next_index >= len(self.dump_paths):
                if self.loop_var.get():
                    next_index = 0
                else:
                    self.pause_playback()
                    return
            self.load_dump(self.dump_paths[next_index])
            self._schedule_next_frame()
        except Exception as exc:
            self.pause_playback()
            self._show_error("Playback stopped", exc)

    def reload_dump(self) -> None:
        self.pause_playback()
        if self.dump_path:
            # Force run B to be reopened as well.
            self.dump2_path = None
            self._dump2_times.clear()
            self.load_dump(self.dump_path)

    # ---------------------------------------------------- comparison run (B)
    def _comparison_active(self) -> bool:
        return bool(self.compare_var.get() and self.dump2 is not None)

    def _panel_count(self) -> int:
        count = 1 + len(self.line_panels)
        return 2 * count if self._comparison_active() else count

    def choose_dump2(self) -> None:
        self.pause_playback()
        filename = filedialog.askopenfilename(
            title="Open comparison PHDF dump (run B)",
            filetypes=(("PHDF dumps", "*.phdf"), ("All files", "*")),
        )
        if filename:
            self._open_dump2(Path(filename), replot=True)

    def _open_dump2(self, path: Path, replot: bool) -> None:
        try:
            path = path.expanduser().resolve()
            self.dump2_paths = dump_series(path)
            self._dump2_times.clear()
            self.dump2_path = None  # force (re)open
            self.compare_var.set(True)
            self._sync_comparison(initial=path)
            if replot and self.dump is not None:
                self._refit_after_change()
                self.plot()
        except Exception as exc:
            self._show_error("Could not open comparison run", exc)

    def clear_dump2(self) -> None:
        self.pause_playback()
        self._close_dump2()
        self.dump2_paths = []
        self._dump2_times.clear()
        self.compare_var.set(False)
        self.dump2_label_var.set("Run B: (none)")
        if self.dump is not None:
            self._refit_after_change()
            self.plot()

    def _close_dump2(self) -> None:
        if self.dump2 is not None and hasattr(self.dump2, "fid"):
            try:
                self.dump2.fid.close()
            except Exception:
                pass
        self.dump2 = None
        self.dump2_path = None

    def _compare_toggled(self) -> None:
        if self.dump is None:
            return
        try:
            self._sync_comparison()
            self._refit_after_change()
            self.plot()
        except Exception as exc:
            self._show_error("Could not update comparison", exc)

    def _refit_after_change(self) -> None:
        """Re-fit spatial ranges so they cover whichever runs are currently shown."""
        if self.dump is None or not self.variable_var.get():
            return
        self._auto_spatial_range(redraw=False)

    def _time_of(self, path: Path) -> float | None:
        """Cached run-B dump time; ``None`` for files that cannot be read yet."""
        if self._dump2_times.get(path) is None:
            self._dump2_times[path] = dump_time(path)
        return self._dump2_times[path]

    def _sync_comparison(self, initial: Path | None = None) -> None:
        """Select the run-B dump that corresponds to the current run-A dump.

        Matching by time works for runs with different numbers of dumps or
        output cadences: run B shows its dump closest in time to run A (its
        last dump once run A goes past the end of run B). Unreadable run-B files
        (e.g. a dump still being written) are skipped instead of aborting, and
        if no run-B dump can be opened the previous one stays on screen.
        """
        if not self.compare_var.get() or not self.dump2_paths:
            return
        if initial is not None:
            self._open_dump2_file(initial, strict=True)
            return
        if self.dump is None or self.dump_path is None:
            return

        candidates: list[Path] = []
        if self.sync_time_var.get():
            t = float(getattr(self.dump, "Time", 0.0))
            timed = [(p, self._time_of(p)) for p in self.dump2_paths]
            timed = [(p, tb) for p, tb in timed if tb is not None]
            candidates = [p for p, tb in sorted(timed, key=lambda item: abs(item[1] - t))]
        if not candidates:
            # Index matching (or no readable times): clamp to run B's length.
            try:
                i = self.dump_paths.index(self.dump_path)
            except ValueError:
                i = 0
            i = min(i, len(self.dump2_paths) - 1)
            candidates = [self.dump2_paths[i]] + self.dump2_paths[i - 1 :: -1][:2]

        if self.dump2 is not None and candidates[0] == self.dump2_path:
            return
        problems = []
        for target in candidates[:4]:
            try:
                self._open_dump2_file(target, strict=False)
                if problems:
                    self.status_var.set(
                        f"Run B: skipped unreadable {', '.join(problems)}"
                    )
                return
            except Exception as exc:
                problems.append(f"{target.name} ({exc})")
                self._dump2_times[target] = None
        self.status_var.set(
            "Run B: could not open a matching dump; showing the previous one. "
            + "; ".join(problems)
        )

    def _open_dump2_file(self, target: Path, strict: bool) -> None:
        """Open ``target`` as the run-B dump (raising if it is unusable)."""
        new_dump = phdf.phdf(str(target))
        if self.dump is not None and dump_dimension(new_dump) != dump_dimension(
            self.dump
        ):
            try:
                new_dump.fid.close()
            except Exception:
                pass
            raise ValueError(
                "Run B has a different dimensionality than run A; "
                "both runs must be 1-D, 2-D, or 3-D."
            )
        old = self.dump2
        self.dump2 = new_dump
        self.dump2_path = target
        self.dump2_label_var.set(f"Run B: {target.parent.name}/{target.name}")
        if old is not None and hasattr(old, "fid"):
            try:
                old.fid.close()
            except Exception:
                pass

    def load_dump(self, path: Path) -> None:
        try:
            path = path.expanduser().resolve()
            self.status_var.set(f"Loading {path.name}…")
            self.update_idletasks()
            old_dump = self.dump
            self.dump = phdf.phdf(str(path))
            if old_dump is not None and hasattr(old_dump, "fid"):
                try:
                    old_dump.fid.close()
                except Exception:
                    pass

            self.dump_path = path
            self.dump_paths = dump_series(path)
            self.dump_combo["values"] = [p.name for p in self.dump_paths]
            self.dump_var.set(path.name)

            # Keep run B in step with run A (no-op if comparison is off).
            self._sync_comparison()
            self._update_slice_range()

            variables = plottable_variables(self.dump)
            self.variable_combo["values"] = variables
            old_variable = self.variable_var.get()
            self.variable_var.set(
                old_variable
                if old_variable in variables
                else (variables[0] if variables else "")
            )
            self.field_data = None
            if variables:
                self.load_variable()
                # A newly selected dump gets its own spatial ranges (in 1-D the
                # y range is held while the value scale is locked).
                self._auto_spatial_range(redraw=False)
                self.plot()
            else:
                self.status_var.set(f"No plottable field datasets found in {path.name}")
        except Exception as exc:
            if self.exporting:
                # Let export_movie abort rather than silently repeating a frame.
                raise
            self._show_error("Could not load dump", exc)

    def _variable_selected(self, _event=None) -> None:
        try:
            self.load_variable()
            self._refit_after_change()
            self.plot()
        except Exception as exc:
            self._show_error("Could not load variable", exc)

    # ------------------------------------------------------ additional panels
    def add_line_panel(self) -> None:
        self._open_panel_dialog(None)

    def edit_line_panel(self, _event=None) -> None:
        selection = self.line_panel_list.curselection()
        if not selection:
            return
        self._open_panel_dialog(selection[0])

    def _open_panel_dialog(self, panel_index: int | None) -> None:
        """Add a panel or edit an existing panel's variable and selectors."""
        if self.dump is None:
            return

        variables = list(self.variable_combo["values"])
        if not variables:
            return
        editing = panel_index is not None
        if editing:
            current = self.line_panels[panel_index]
            preferred = current["variable"]
            initial_indices = current["indices"]
        else:
            preferred = next(
                (name for name in variables if name != self.variable_var.get()),
                variables[0],
            )
            initial_indices = ""
        dialog = tk.Toplevel(self)
        dialog.title("Edit panel" if editing else "Add panel")
        dialog.transient(self)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        variable = tk.StringVar(value=preferred)
        indices = tk.StringVar(value=initial_indices)
        shape_text = tk.StringVar()
        ttk.Label(frame, text="Variable").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(
            frame, textvariable=variable, values=variables, state="readonly", width=34
        )
        combo.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(frame, text="Selector indices").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=indices, width=20).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=2
        )
        ttk.Label(frame, textvariable=shape_text).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 6)
        )

        def update_shape(_event=None, reset_indices=True) -> None:
            try:
                data = self._field_source(self.dump, variable.get())
                spatial_dimension = self._plot_dimension()
                leading_shape = data.shape[1:-spatial_dimension]
                shape_text.set(
                    f"Shape: {data.shape}; selectable axes: {leading_shape or 'none'}"
                )
                if reset_indices:
                    indices.set(",".join("0" for _ in leading_shape))
            except Exception as exc:
                shape_text.set(f"Could not inspect variable: {exc}")

        def accept() -> None:
            try:
                data = self._field_source(self.dump, variable.get())
                selector_count = data.ndim - 1 - self._plot_dimension()
                parse_indices(indices.get(), selector_count)
                updated = {
                    "variable": variable.get(),
                    "indices": indices.get().strip(),
                }
                if editing:
                    self.line_panels[panel_index] = updated
                else:
                    self.line_panels.append(updated)
                self._refresh_line_panel_list()
                dialog.destroy()
                self.plot()
            except Exception as exc:
                action = "edit" if editing else "add"
                messagebox.showerror(
                    f"Could not {action} panel", str(exc), parent=dialog
                )

        combo.bind("<<ComboboxSelected>>", lambda event: update_shape(event, True))
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons, text="Save" if editing else "Add", command=accept).pack(
            side=tk.LEFT
        )
        update_shape(reset_indices=not editing)
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()
        combo.focus_set()

    def remove_line_panel(self) -> None:
        selection = self.line_panel_list.curselection()
        if not selection:
            return
        del self.line_panels[selection[0]]
        self._refresh_line_panel_list()
        if self.dump is not None:
            self.plot()

    def _refresh_line_panel_list(self) -> None:
        self.line_panel_list.delete(0, tk.END)
        for panel in self.line_panels:
            suffix = f" [{panel['indices']}]" if panel["indices"] else ""
            self.line_panel_list.insert(tk.END, f"{panel['variable']}{suffix}")

    def _panel_layout_changed(self, _event=None) -> None:
        if self.dump is not None:
            self.plot()

    def _panel_grid_shape(self, panel_count: int) -> tuple[int, int]:
        """Choose rows and columns from the explicit or automatic layout setting."""
        if panel_count <= 1:
            return 1, 1
        # Comparison mode: panels are interleaved A,B,A,B,... so use two
        # columns (run A left, run B right) and one row per variable.
        if self._comparison_active() and panel_count % 2 == 0:
            return panel_count // 2, 2
        layout = self.panel_layout_var.get()
        if layout == "Left–right":
            return 1, panel_count
        if layout == "Top–bottom":
            return panel_count, 1

        width = max(self.canvas_widget.winfo_width(), 1)
        height = max(self.canvas_widget.winfo_height(), 1)
        aspect = width / height
        if panel_count == 2:
            return (1, 2) if aspect >= 1.0 else (2, 1)
        columns = max(1, min(panel_count, math.ceil(math.sqrt(panel_count * aspect))))
        rows = math.ceil(panel_count / columns)
        return rows, columns

    def _log_toggled(self) -> None:
        """Apply log color scaling in 2-D or a log y axis in 1-D."""
        if self.dump is not None and self.variable_var.get():
            try:
                self._refit_after_change()
                self.plot()
            except Exception as exc:
                self._show_error("Could not update scale", exc)

    # --------------------------------------------------------- data handling
    def load_variable(self) -> None:
        if self.dump is None or not self.variable_var.get():
            return
        self.status_var.set(f"Loading {self.variable_var.get()}…")
        self.update_idletasks()
        self.field_data = self._field_source(self.dump, self.variable_var.get())
        if self.field_data.ndim < 2:
            raise ValueError(
                f"Expected at least (block,x), got shape {self.field_data.shape}"
            )
        spatial_dimension = self._plot_dimension()
        leading_shape = self.field_data.shape[1:-spatial_dimension]
        self.shape_var.set(
            f"Shape: {self.field_data.shape}; {spatial_dimension}-D plot; "
            f"selectable axes: {leading_shape or 'none'}"
        )
        count = len(leading_shape)
        try:
            parse_indices(self.indices_var.get(), count)
        except ValueError:
            self.indices_var.set(",".join("0" for _ in range(count)))

    def _plot_dimension(self) -> int:
        """Infer whether the current mesh should be plotted as 1-D or 2-D."""
        if self.dump is None:
            return 2
        return dump_dimension(self.dump)

    def _selected_blocks(self) -> list[np.ndarray]:
        if self.field_data is None:
            self.load_variable()
        return self._load_blocks(self.dump, self.variable_var.get(), self.indices_var.get())

    def _field_source(self, dump, variable: str):
        """Array-like (block, [components], z, y, x) data for ``variable``.

        3-D cell-centered data is returned as the lazy HDF5 dataset so that only
        the slice plane of the blocks it cuts is ever read from disk; everything
        else goes through ``phdf.Get`` (which reads and caches the whole field).
        """
        if dump_dimension(dump) == 3 and not getattr(dump, "IncludesGhost", 0):
            try:
                dataset = dump.fid[variable]
                topology = dataset.attrs.get("TopologicalLocation", "Cell")
                if isinstance(topology, bytes):
                    topology = topology.decode()
                if str(topology) == "Cell" and len(dataset.shape) >= 4:
                    return dataset
            except (KeyError, TypeError, AttributeError):
                pass
        data = dump.Get(variable, flatten=False)
        if data is None:
            raise ValueError(f"Could not read '{variable}'")
        return np.asarray(data)

    def _load_blocks(self, dump, variable: str, indices_text: str) -> list[np.ndarray]:
        """Return one plottable array per block (cached per dump and slice)."""
        key = (variable, indices_text.strip())
        if dump_dimension(dump) == 3:
            key += (
                self.slice_plane_var.get(),
                self.slice_swap_var.get(),
                float(self.slice_pos_var.get()),
            )
        cache = dump_cache(dump).setdefault("blocks", {})
        if key in cache:
            return cache[key]
        try:
            field_data = self._field_source(dump, variable)
        except Exception as exc:
            raise ValueError(f"Could not read '{variable}': {exc}") from exc
        blocks = self._blocks_from_field_data(field_data, indices_text, dump)
        if len(cache) >= BLOCK_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = blocks
        return blocks

    def _selected_blocks_all_runs(self) -> list[np.ndarray]:
        """Primary-variable blocks from run A (and run B when comparing)."""
        blocks = list(self._selected_blocks())
        if self._comparison_active():
            blocks += self._load_blocks(
                self.dump2, self.variable_var.get(), self.indices_var.get()
            )
        return blocks

    def _blocks_from_field_data(
        self, field_data: np.ndarray, indices_text: str, dump
    ) -> list[np.ndarray]:
        """Select component indices and return one plottable array per block.

        For 3-D dumps only the blocks cut by the current slice plane are
        returned, each reduced to the 2-D cut (rows = vertical plot axis).
        The order matches ``_plane_coords``.
        """
        spatial_dimension = self._plot_dimension()
        selector_count = field_data.ndim - 1 - spatial_dimension
        indices = parse_indices(indices_text, selector_count)
        spatial_slices = (slice(None),) * spatial_dimension
        if spatial_dimension == 3:
            _, _, normal = SLICE_PLANES[self.slice_plane_var.get()]
            blocks = []
            for block, index in self._slice_selection(dump):
                # Array axes are (z, y, x), so the normal axis n lives at 2 - n.
                # Indexing the plane directly reads only that plane from HDF5.
                plane = [slice(None)] * 3
                plane[2 - normal] = int(index)
                cut = np.asarray(field_data[(int(block),) + indices + tuple(plane)])
                if cut.ndim != 2:
                    raise ValueError(
                        f"Selected data is not 3-D; resulting slice shape is {cut.shape}"
                    )
                blocks.append(cut.T if self.slice_swap_var.get() else cut)
            return blocks
        blocks = []
        for block in range(field_data.shape[0]):
            array = np.asarray(field_data[(block,) + indices + spatial_slices])
            if array.ndim != spatial_dimension:
                raise ValueError(
                    f"Selected data is not {spatial_dimension}-D; resulting shape is "
                    f"{array.shape}"
                )
            blocks.append(array)
        return blocks

    # ------------------------------------------------------------ 3-D slicing
    def _plane_axes(self) -> tuple[int, int, int]:
        """(horizontal, vertical, normal) axes; plain x/y/z for 1-D and 2-D."""
        if self._plot_dimension() != 3:
            return 0, 1, 2
        horizontal, vertical, normal = SLICE_PLANES[self.slice_plane_var.get()]
        if self.slice_swap_var.get():
            horizontal, vertical = vertical, horizontal
        return horizontal, vertical, normal

    def _axis_labels(self) -> tuple[str, str]:
        horizontal, vertical, _ = self._plane_axes()
        return AXIS_NAMES[horizontal], AXIS_NAMES[vertical]

    def _slice_selection(self, dump) -> list[tuple[int, int]]:
        """(block, cell index along the normal) for every block the plane cuts.

        Blocks are tested on half-open [lo, hi) extents so a plane on a block
        boundary picks exactly one block per in-plane footprint (also with AMR).
        """
        _, _, normal = SLICE_PLANES[self.slice_plane_var.get()]
        position = float(self.slice_pos_var.get())
        edges = dump_edges(dump, normal)
        inside = (edges[:, 0] <= position) & (position < edges[:, -1])
        if not inside.any():
            # Only reachable exactly on the domain's upper face.
            inside = (edges[:, 0] <= position) & (position <= edges[:, -1])
        selection = []
        for block in np.flatnonzero(inside):
            index = int(np.searchsorted(edges[block], position, side="right")) - 1
            selection.append((int(block), min(max(index, 0), edges.shape[1] - 2)))
        if not selection:
            raise ValueError(
                f"The {AXIS_NAMES[normal]} = {position:.6g} slice does not cut the mesh"
            )
        return selection

    def _plane_coords(self, dump) -> list[tuple[np.ndarray, np.ndarray]]:
        """(horizontal, vertical) cell edges per plotted block, in block order."""
        horizontal, vertical, _ = self._plane_axes()
        hedges = dump_edges(dump, horizontal)
        vedges = dump_edges(dump, vertical)
        if self._plot_dimension() != 3:
            return list(zip(hedges, vedges))
        return [(hedges[block], vedges[block]) for block, _ in self._slice_selection(dump)]

    def _set_slice_controls_enabled(self, enabled: bool) -> None:
        for widget in self._slice_widgets:
            if widget is self.slice_plane_combo:
                widget.configure(state="readonly" if enabled else "disabled")
            else:
                widget.state(["!disabled"] if enabled else ["disabled"])

    def _update_slice_range(self, reset: bool = False) -> None:
        """Fit the slider to the normal-axis extent of the current dump."""
        if self.dump is None or self._plot_dimension() != 3:
            self._set_slice_controls_enabled(False)
            self.slice_axis_label.set("Position")
            self.slice_pos_text.set("")
            return
        self._set_slice_controls_enabled(True)
        _, _, normal = SLICE_PLANES[self.slice_plane_var.get()]
        centers = dump_coordinate(self.dump, normal)
        lo, hi = float(np.nanmin(centers)), float(np.nanmax(centers))
        spacing = np.diff(centers, axis=1)
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        self._slice_dx = float(spacing.min()) if spacing.size else (hi - lo)
        self.slice_scale.configure(from_=lo, to=hi)
        position = float(self.slice_pos_var.get())
        if reset or not self._slice_initialized or not lo <= position <= hi:
            position = 0.5 * (lo + hi)
            self._slice_initialized = True
        self.slice_pos_var.set(position)
        self.slice_axis_label.set(f"{AXIS_NAMES[normal]} =")
        self.slice_pos_text.set(f"{position:.6g}")

    def _slice_changed(self, refit_space: bool) -> None:
        """Replot after the slice plane or position changed."""
        self._slice_job = None
        if self.dump is None or self._plot_dimension() != 3:
            return
        try:
            if refit_space:
                self._auto_spatial_range(redraw=False)
            self.plot()
        except Exception as exc:
            self._show_error("Could not update slice", exc)

    def _slice_plane_changed(self, _event=None) -> None:
        self.pause_playback()
        self._update_slice_range(reset=True)
        self._slice_changed(refit_space=True)

    def _slice_swap_toggled(self) -> None:
        self._slice_changed(refit_space=True)

    def _slice_scale_moved(self, value: str) -> None:
        """Track the slider continuously but debounce the (heavier) replot."""
        self.slice_pos_text.set(f"{float(value):.6g}")
        if self._slice_job is not None:
            self.after_cancel(self._slice_job)
        self._slice_job = self.after(120, lambda: self._slice_changed(False))

    def _set_slice_position(self, position: float) -> None:
        lo = float(self.slice_scale.cget("from"))
        hi = float(self.slice_scale.cget("to"))
        position = min(max(position, lo), hi)
        self.slice_pos_var.set(position)
        self.slice_pos_text.set(f"{position:.6g}")
        self._slice_changed(refit_space=False)

    def _slice_entry_set(self, _event=None) -> None:
        try:
            self._set_slice_position(float(self.slice_pos_text.get()))
        except ValueError as exc:
            self._show_error("Invalid slice position", exc)

    def _slice_step(self, cells: int) -> None:
        self._set_slice_position(float(self.slice_pos_var.get()) + cells * self._slice_dx)

    def _finite_range(
        self, blocks: list[np.ndarray], positive=False
    ) -> tuple[float, float]:
        values = np.concatenate([np.asarray(a).ravel() for a in blocks])
        values = values[np.isfinite(values)]
        if positive:
            floor = max(float(self.floor_var.get()), 0.0)
            values = values[values > floor]
        if values.size == 0:
            if positive:
                # Nothing positive to show on a log scale (all zero/negative).
                # Fall back to a valid dummy decade so plotting and movie export
                # continue; the panel will simply appear empty.
                floor = float(self.floor_var.get())
                lo = floor if floor > 0 else 1.0e-30
                return lo, lo * 10.0
            raise ValueError("The selected field contains no finite values")
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            if positive:
                lo, hi = lo / 1.1, hi * 1.1
            else:
                delta = max(abs(lo) * 1.0e-6, 1.0e-6)
                lo, hi = lo - delta, hi + delta
        return lo, hi

    def _auto_spatial_range(self, redraw=True) -> None:
        """Fit x/y limits from the mesh and field extrema of all shown runs."""
        if self.dump is None:
            return
        blocks = self._selected_blocks_all_runs()
        is_1d = self._plot_dimension() == 1
        lo, hi = self._finite_range(blocks, positive=self.log_var.get())

        dumps = [self.dump]
        if self._comparison_active():
            dumps.append(self.dump2)
        xmin = min(float(np.nanmin(d.x)) for d in dumps)
        xmax = max(float(np.nanmax(d.x)) for d in dumps)
        self.xmin_var.set(f"{xmin:.8g}")
        self.xmax_var.set(f"{xmax:.8g}")
        if is_1d and self.lock_colorscale_var.get() and self.ymin_var.get().strip():
            pass  # value scale locked: keep the current y range
        elif is_1d:
            if self.log_var.get():
                padding_factor = (hi / lo) ** 0.03
                self.ymin_var.set(f"{lo / padding_factor:.8g}")
                self.ymax_var.set(f"{hi * padding_factor:.8g}")
            else:
                padding = 0.03 * (hi - lo)
                self.ymin_var.set(f"{lo - padding:.8g}")
                self.ymax_var.set(f"{hi + padding:.8g}")
        else:
            # Fit to cell edges so boundary cells are shown in full.
            horizontal, vertical = self._plane_axes()[:2]
            xmin = min(float(np.nanmin(dump_edges(d, horizontal))) for d in dumps)
            xmax = max(float(np.nanmax(dump_edges(d, horizontal))) for d in dumps)
            ymin = min(float(np.nanmin(dump_edges(d, vertical))) for d in dumps)
            ymax = max(float(np.nanmax(dump_edges(d, vertical))) for d in dumps)
            self.xmin_var.set(f"{xmin:.8g}")
            self.xmax_var.set(f"{xmax:.8g}")
            self.ymin_var.set(f"{ymin:.8g}")
            self.ymax_var.set(f"{ymax:.8g}")
        if redraw:
            self.plot()

    def auto_ranges(self, redraw=True) -> None:
        """Refit spatial ranges (and release any locked value scale)."""
        try:
            self._locked_ranges.clear()
            if self.lock_colorscale_var.get():
                self.ymin_var.set("")
            self._auto_spatial_range(redraw=False)
            if redraw:
                self.plot()
        except Exception as exc:
            self._show_error("Could not determine ranges", exc)

    def _lock_toggled(self) -> None:
        """Locking captures the scale shown now; unlocking resumes auto-fitting."""
        self._locked_ranges.clear()
        if self.dump is not None and self.variable_var.get():
            try:
                self._auto_spatial_range(redraw=False)
                self.plot()
            except Exception as exc:
                self._show_error("Could not update scale", exc)

    def _replot_fresh(self) -> None:
        """Plot button: re-measure the layout from scratch, then draw."""
        self._layout_memory = None
        self.plot()

    def _value_range(self, spec: dict, positive: bool, padded: bool) -> tuple[float, float]:
        """Auto (or locked) value range shared by the run A/B panels of a variable."""
        if padded:
            lo, hi = self._line_y_limits(spec["group_blocks"], positive)
        else:
            lo, hi = self._finite_range(spec["group_blocks"], positive=positive)
        if not self.lock_colorscale_var.get():
            return lo, hi
        key = (spec["variable"], spec["indices"].strip(), positive, padded)
        return self._locked_ranges.setdefault(key, (lo, hi))

    def _line_y_limits(
        self, blocks: list[np.ndarray], positive: bool
    ) -> tuple[float, float]:
        lo, hi = self._finite_range(blocks, positive=positive)
        if positive:
            padding_factor = (hi / lo) ** 0.03
            return lo / padding_factor, hi * padding_factor
        padding = 0.03 * (hi - lo)
        return lo - padding, hi + padding

    def _panel_specs(self, main_blocks: list[np.ndarray]) -> list[dict]:
        """Build panel descriptions for the current dump(s).

        Each spec has:
          variable, indices, blocks   - what to draw
          main                        - True for the primary variable (A or B)
          dump                        - the dump supplying x/y coordinates
          run, tag                    - 0/1 and a title string (comparison only)
          group_blocks                - blocks of every run for this variable, so
                                        paired panels share one color/y scale.
        When comparing, panels are interleaved A, B, A, B, ... so a two-column
        grid puts run A on the left and run B on the right.
        """
        base = [
            {
                "variable": self.variable_var.get(),
                "indices": self.indices_var.get(),
                "blocks": main_blocks,
                "main": True,
            }
        ]
        available = set(self.variable_combo["values"])
        for panel in self.line_panels:
            variable = panel["variable"]
            if variable not in available:
                raise ValueError(
                    f"Additional panel variable '{variable}' is absent from "
                    f"{self.dump_path.name}"
                )
            base.append(
                {
                    "variable": variable,
                    "indices": panel["indices"],
                    "blocks": self._load_blocks(self.dump, variable, panel["indices"]),
                    "main": False,
                }
            )

        comparing = self._comparison_active()
        specs = []
        for item in base:
            if comparing:
                blocks_b = self._load_blocks(
                    self.dump2, item["variable"], item["indices"]
                )
                group = list(item["blocks"]) + list(blocks_b)
                specs.append(
                    {
                        **item,
                        "dump": self.dump,
                        "run": 0,
                        "tag": f"{RUN_LABELS[0]}: {self.dump_path.parent.name}",
                        "group_blocks": group,
                    }
                )
                specs.append(
                    {
                        **item,
                        "blocks": blocks_b,
                        "dump": self.dump2,
                        "run": 1,
                        "tag": f"{RUN_LABELS[1]}: {self.dump2_path.parent.name}",
                        "group_blocks": group,
                    }
                )
            else:
                specs.append(
                    {
                        **item,
                        "dump": self.dump,
                        "run": 0,
                        "tag": "",
                        "group_blocks": item["blocks"],
                    }
                )
        return specs

    # ---------------------------------------------------------------- plotting
    def _plot_1d_panels(
        self,
        main_blocks: list[np.ndarray],
        positive: bool,
        line_width: float,
        title: str,
    ) -> None:
        """Draw the primary field and all requested extra fields as panels."""
        specs = self._panel_specs(main_blocks)
        comparing = self._comparison_active()

        rows, columns = self._panel_grid_shape(len(specs))
        self._last_panel_grid_shape = (rows, columns)
        grid = self.figure.add_gridspec(rows, columns)
        axes = []
        self.data_lines = []
        self.mesh_lines = []
        self.colorbar = None
        self.colorbars = []
        self._raster_panels = []
        for panel_index, spec in enumerate(specs):
            row, column = divmod(panel_index, columns)
            # Run B shares run A's y axis (same limits), so its y ticks are hidden.
            share_y = axes[-1] if comparing and spec["run"] == 1 else None
            axes_object = self.figure.add_subplot(
                grid[row, column], sharex=axes[0] if axes else None, sharey=share_y
            )
            axes.append(axes_object)
            panel_blocks = spec["blocks"]
            dump = spec["dump"]
            # One Line2D per panel (blocks separated by NaN) instead of one per
            # block: far fewer artists to build and draw.
            xs, ys = [], []
            for block, data in enumerate(panel_blocks):
                x = np.asarray(dump.x[block, :])
                if data.shape != x.shape:
                    raise ValueError(
                        f"{spec['variable']}, block {block}: data shape {data.shape} "
                        f"does not match x-coordinate shape {x.shape}"
                    )
                xs += [x, [np.nan]]
                ys += [np.asarray(data, dtype=float), [np.nan]]
            (data_line,) = axes_object.plot(
                np.concatenate(xs),
                np.concatenate(ys),
                color=COLORS[spec["run"] % len(COLORS)],
                linewidth=max(1.4, 1.4 * line_width),
            )
            self.data_lines.append(data_line)
            if self.mesh_var.get():
                xedges = dump_edges(dump, 0)
                mesh_line = axes_object.vlines(
                    np.unique(np.concatenate((xedges[:, 0], xedges[:, -1]))),
                    0.0,
                    1.0,
                    transform=axes_object.get_xaxis_transform(),
                    color="black",
                    linewidth=max(0.45, 0.45 * line_width),
                )
                self.mesh_lines.append(mesh_line)

            if share_y is None:
                axes_object.set_ylabel(spec["variable"])
            else:
                axes_object.tick_params(labelleft=False)
            axes_object.set_yscale("log" if positive else "linear")
            if comparing and row == 0:
                axes_object.set_title(self._run_header(spec["run"]))
            # Paired panels share limits computed over both runs.
            auto_ymin, auto_ymax = self._value_range(spec, positive, padded=True)
            if spec["main"]:
                ymin = optional_float(self.ymin_var.get())
                ymax = optional_float(self.ymax_var.get())
                ymin = auto_ymin if ymin is None else ymin
                ymax = auto_ymax if ymax is None else ymax
            else:
                ymin, ymax = auto_ymin, auto_ymax
            if positive and ymin <= 0:
                raise ValueError(
                    f"Log y scaling requires a positive y minimum for {spec['variable']}"
                )
            axes_object.set_ylim(ymin, ymax)

        self.line_panel_axes = axes
        self.main_axes = axes[0]
        xmin = optional_float(self.xmin_var.get())
        xmax = optional_float(self.xmax_var.get())
        if xmin is not None or xmax is not None:
            axes[0].set_xlim(left=xmin, right=xmax)
        self._label_bottom_edges(axes, columns, "x")
        self._finish_figure(title, rows, columns, aspect=None)
        self.status_var.set(
            f"{len(specs)} line panel{'s' if len(specs) != 1 else ''}  |  "
            f"{self._status_names()}"
        )

    def _color_norm(self, spec: dict, positive: bool) -> mcolors.Normalize:
        """Color normalization for a panel (shared by its run A/B pair)."""
        vmin, vmax = self._value_range(spec, positive, padded=False)
        if positive:
            vmin = max(vmin, float(self.floor_var.get()))
            if vmin <= 0:
                raise ValueError("Log color scaling requires a positive color minimum")
            if vmin >= vmax:
                vmin, vmax = vmax / 10.0, vmax
        return (
            mcolors.LogNorm(vmin=vmin, vmax=vmax)
            if positive
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
        )

    def _plot_2d_panels(
        self,
        main_blocks: list[np.ndarray],
        positive: bool,
        line_width: float,
        title: str,
    ) -> None:
        """Draw primary and additional 2-D fields with independent colorbars."""
        specs = self._panel_specs(main_blocks)
        comparing = self._comparison_active()
        rows, columns = self._panel_grid_shape(len(specs))
        self._last_panel_grid_shape = (rows, columns)
        grid = self.figure.add_gridspec(rows, columns)
        fast = self.fast_render_var.get()
        # Cell edges per run (all panels of one run share the same blocks).
        plane_edges = {}
        for spec in specs:
            if spec["run"] not in plane_edges:
                plane_edges[spec["run"]] = self._plane_coords(spec["dump"])
        all_edges = [pair for pairs in plane_edges.values() for pair in pairs]
        view = []
        for entry, pick, which in (
            (self.xmin_var, min, (0, 0)),
            (self.xmax_var, max, (0, -1)),
            (self.ymin_var, min, (1, 0)),
            (self.ymax_var, max, (1, -1)),
        ):
            value = optional_float(entry.get())
            if value is None:
                value = pick(float(pair[which[0]][which[1]]) for pair in all_edges)
            view.append(value)
        xlim, ylim = (view[0], view[1]), (view[2], view[3])
        # Data aspect (height / width) of every panel; they all share limits.
        aspect = (
            abs(ylim[1] - ylim[0]) / max(abs(xlim[1] - xlim[0]), 1e-300)
            if self.equal_aspect_var.get()
            else None
        )
        # Colorbar size relative to the panel width, so tall/narrow panels
        # still get a reasonably sized bar.
        shape_ratio = float(np.clip(aspect if aspect else rows / columns, 0.4, 4.0))
        colorbar_width = 0.05 * shape_ratio
        colorbar_pad = 0.035 * shape_ratio

        axes = []
        self.mesh_lines = []
        self.data_lines = []
        self.colorbars = []
        self._raster_panels = []
        for panel_index, spec in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axes_object = self.figure.add_subplot(
                grid[row, column],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
            axes.append(axes_object)
            panel_blocks = spec["blocks"]
            # The norm is built from both runs' data for this variable, so run
            # A and run B panels of a pair always share an identical scale.
            norm = self._color_norm(spec, positive)
            edges = plane_edges[spec["run"]]
            for block, (data, (xedges, yedges)) in enumerate(zip(panel_blocks, edges)):
                expected_shape = (len(yedges) - 1, len(xedges) - 1)
                if data.shape != expected_shape:
                    raise ValueError(
                        f"{spec['variable']}, block {block}: data shape {data.shape} "
                        f"does not match coordinate shape {expected_shape}"
                    )
            if fast:
                # Placeholder now; the raster is built once, after the layout.
                pcol = self._draw_raster(axes_object, panel_blocks, edges, norm, xlim, ylim, defer=True)
            else:
                pcol = None
                for data, (xedges, yedges) in zip(panel_blocks, edges):
                    pcol = axes_object.pcolormesh(
                        xedges,
                        yedges,
                        data,
                        norm=norm,
                        cmap=self.cmap_var.get(),
                        shading="flat",
                        rasterized=True,
                    )
            if self.mesh_var.get() and edges:
                # All block outlines as a single collection.
                h0 = np.array([e[0][0] for e in edges])
                h1 = np.array([e[0][-1] for e in edges])
                v0 = np.array([e[1][0] for e in edges])
                v1 = np.array([e[1][-1] for e in edges])
                outlines = np.stack(
                    (
                        np.stack((h0, h1, h1, h0, h0), axis=1),
                        np.stack((v0, v0, v1, v1, v0), axis=1),
                    ),
                    axis=2,
                )
                mesh_line = LineCollection(
                    outlines, colors="black", linewidths=max(0.45, 0.45 * line_width)
                )
                axes_object.add_collection(mesh_line, autolim=False)
                self.mesh_lines.append(mesh_line)

            if pcol is None:
                pcol = mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap_var.get())
            # When comparing, the A/B pair shares one scale, so only the run-B
            # panel carries the (single) colorbar for that row.
            if not comparing or spec["run"] == 1:
                # Attached in axes coordinates, so it follows the panel exactly.
                colorbar_axes = axes_object.inset_axes(
                    [1.0 + colorbar_pad, 0.0, colorbar_width, 1.0]
                )
                colorbar = self.figure.colorbar(pcol, cax=colorbar_axes)
                colorbar.set_label(spec["variable"])
                colorbar.ax.minorticks_off()
                self.colorbars.append(colorbar)
            if comparing and row == 0:
                axes_object.set_title(self._run_header(spec["run"]))
            if column == 0:
                axes_object.set_ylabel(self._axis_labels()[1])
            else:
                axes_object.tick_params(labelleft=False)
            axes_object.set_aspect("equal" if self.equal_aspect_var.get() else "auto")

        self.colorbar = self.colorbars[0]
        self.line_panel_axes = axes
        self.main_axes = axes[0]
        axes[0].set_xlim(*xlim)
        axes[0].set_ylim(*ylim)
        self._label_bottom_edges(axes, columns, self._axis_labels()[0])
        self._finish_figure(title, rows, columns, aspect=aspect)
        if fast:
            # Rasters were built before the final layout; size them to it.
            self._rerender_rasters(draw=False)
            # Rebuild the rasters for the new view after toolbar zoom/pan.
            axes[0].callbacks.connect("xlim_changed", self._view_changed)
            axes[0].callbacks.connect("ylim_changed", self._view_changed)
        self.canvas.draw_idle()
        self.status_var.set(
            f"{len(specs)} 2-D panel{'s' if len(specs) != 1 else ''}  |  "
            f"{self._status_names()}"
        )

    def _raster_shape(self, axes_object) -> tuple[int, int]:
        """Raster size cap: the panel's size in output pixels."""
        bbox = axes_object.get_window_extent()
        scale = (self._raster_dpi or self.figure.dpi) / self.figure.dpi
        return (
            int(np.clip(bbox.width * scale + 1, 16, MAX_RASTER_PIXELS)),
            int(np.clip(bbox.height * scale + 1, 16, MAX_RASTER_PIXELS)),
        )

    def _draw_raster(self, axes_object, blocks, edges, norm, xlim, ylim, defer=False):
        result = (
            None
            if defer
            else rasterize_blocks(blocks, edges, xlim, ylim, self._raster_shape(axes_object))
        )
        if result is None:
            image_data, extent = np.full((1, 1), np.nan), (*xlim, *ylim)
        else:
            image_data, extent = result
        image = axes_object.imshow(
            image_data,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            norm=norm,
            cmap=self.cmap_var.get(),
            aspect="auto",
        )
        image.set_visible(result is not None and not defer)
        self._raster_panels.append(
            {"axes": axes_object, "image": image, "blocks": blocks, "edges": edges}
        )
        return image

    def _view_changed(self, _axes=None) -> None:
        if self._raster_job is not None:
            self.after_cancel(self._raster_job)
        self._raster_job = self.after(150, self._rerender_rasters)

    def _rerender_rasters(self, draw: bool = True) -> None:
        """Rebuild each panel's raster for its current view limits."""
        self._raster_job = None
        for panel in self._raster_panels:
            axes_object = panel["axes"]
            if axes_object not in self.figure.axes:
                continue
            result = rasterize_blocks(
                panel["blocks"],
                panel["edges"],
                axes_object.get_xlim(),
                axes_object.get_ylim(),
                self._raster_shape(axes_object),
            )
            panel["image"].set_visible(result is not None)
            if result is not None:
                panel["image"].set_data(result[0])
                panel["image"].set_extent(result[1])
        if draw:
            self.canvas.draw_idle()

    def _status_names(self) -> str:
        if self._comparison_active():
            return f"A: {self.dump_path.name}  |  B: {self.dump2_path.name}"
        return self.dump_path.name

    def _make_title(self) -> str:
        """Figure title. When comparing, each column header carries its run."""
        lines = []
        if not self._comparison_active():
            text = self.dump_path.name if self.dump_path else "PHDF"
            time = getattr(self.dump, "Time", None)
            if time is not None:
                text += f"    t = {float(time):.3e} s"
            lines.append(text)
        if self._plot_dimension() == 3:
            plane = self.slice_plane_var.get()
            normal = AXIS_NAMES[SLICE_PLANES[plane][2]]
            lines.append(
                f"{plane} slice at {normal} = {float(self.slice_pos_var.get()):.4g}"
            )
        return "\n".join(lines)

    def _run_header(self, run: int) -> str:
        """Column header for run A/B: directory, then dump time."""
        dump, path = (self.dump, self.dump_path) if run == 0 else (self.dump2, self.dump2_path)
        header = f"{RUN_LABELS[run]}: {path.parent.name}"
        time = getattr(dump, "Time", None)
        if time is not None:
            header += f"\nt = {float(time):.3e} s"
            if run == 1 and self.sync_time_var.get():
                last = max(
                    (t for t in (self._time_of(p) for p in self.dump2_paths) if t is not None),
                    default=None,
                )
                if last is not None and float(getattr(self.dump, "Time", 0.0)) > last:
                    header += " (end of run)"
        return header

    @staticmethod
    def _label_bottom_edges(axes: list, columns: int, xlabel: str) -> None:
        """x tick labels and label only on panels with no panel below them."""
        count = len(axes)
        for index, axes_object in enumerate(axes):
            if index + columns < count:
                axes_object.tick_params(labelbottom=False)
                axes_object.set_xlabel("")
            else:
                axes_object.tick_params(labelbottom=True)
                axes_object.set_xlabel(xlabel)

    def _finish_figure(self, title: str, rows: int, columns: int, aspect) -> None:
        """Title, styling, and the measured layout shared by 1-D and 2-D plots."""
        self.figure_title = (
            self.figure.suptitle(title, y=1.0, va="top") if title else None
        )
        self._layout_aspect = aspect
        self._apply_plot_style(draw=False)
        self._layout_figure()

    def _fit_colorbar_labels(self, renderer, font_size: float) -> None:
        """Keep each rotated colorbar label within the length of its bar."""
        for colorbar in self.colorbars:
            label = colorbar.ax.yaxis.label
            full_text = label.get_text()
            bar_height = colorbar.ax.get_window_extent(renderer).height
            for _ in range(10):
                if label.get_window_extent(renderer).height <= bar_height:
                    break
                if label.get_fontsize() > 0.7 * font_size:
                    label.set_fontsize(label.get_fontsize() * 0.9)
                elif label.get_text() == full_text and full_text.count(".") > 1:
                    # e.g. "c.c.bulk.temperature" -> "bulk.temperature"
                    label.set_text(".".join(full_text.split(".")[-2:]))
                elif label.get_fontsize() > 0.5 * font_size:
                    label.set_fontsize(label.get_fontsize() * 0.9)
                else:
                    break

    @staticmethod
    def _thin_ticks(axes_list: list, renderer, font_size: float) -> None:
        """Use fewer ticks wherever neighboring tick labels would touch."""
        min_gap = 0.5 * font_size * renderer.points_to_pixels(1.0)
        targets = []
        for axes_object in axes_list:
            targets += [(axes_object.xaxis, 0), (axes_object.yaxis, 1)]
            for child in axes_object.child_axes:  # colorbars
                targets.append((child.yaxis, 1))
        for axis, along in targets:
            for _ in range(4):
                try:
                    ticks = axis._update_ticks()
                except Exception:
                    break
                boxes = [
                    tick.label1.get_window_extent(renderer)
                    for tick in ticks
                    if tick.label1.get_visible() and tick.label1.get_text()
                ]
                if len(boxes) < 3:
                    break
                if along == 0:
                    spans = sorted((b.x0, b.x1) for b in boxes)
                else:
                    spans = sorted((b.y0, b.y1) for b in boxes)
                gap = min(nxt[0] - prev[1] for prev, nxt in zip(spans, spans[1:]))
                if gap >= min_gap:
                    break
                count = max(2, len(boxes) - 2)
                if axis.get_scale() == "log":
                    axis.set_major_locator(mticker.LogLocator(numticks=count + 1))
                else:
                    axis.set_major_locator(
                        mticker.MaxNLocator(
                            nbins=count, steps=[1, 2, 2.5, 4, 5, 10], min_n_ticks=1
                        )
                    )

    def _layout_figure(self) -> None:
        """Place panels on a uniform grid with gutters measured from their text.

        Each panel's decorations (tick labels, axis labels, titles, colorbar and
        offset text) are measured, column/row gutters are sized to the largest
        of them, panels get one common size that honors the data aspect, and
        the grid is centered. Gutters only grow while the panel structure stays
        the same, so stepping through dumps never makes the panels jitter.
        """
        axes_list = getattr(self, "line_panel_axes", None) or []
        axes_list = [a for a in axes_list if a in self.figure.axes]
        rows, columns = self._last_panel_grid_shape or (1, 1)
        if not axes_list:
            return
        figure = self.figure
        renderer = figure.canvas.get_renderer()
        dpi = figure.dpi
        width_in, height_in = figure.get_size_inches()
        font_size, _ = self._style_metrics()
        pad = 0.35 * font_size / 72.0  # breathing room around the figure (inches)
        gap = 0.8 * font_size / 72.0  # space between neighboring panels' text
        aspect = getattr(self, "_layout_aspect", None)

        structure = (
            rows,
            columns,
            len(axes_list),
            round(width_in, 3),
            round(height_in, 3),
            round(font_size, 2),
            aspect is not None and round(aspect, 6),
            self.log_var.get(),
            self.figure_title.get_text().count("\n") if self.figure_title else -1,
        )

        title_height = 0.0
        if self.figure_title is not None:
            # Shrink an over-long title instead of letting it run off the canvas.
            for _ in range(6):
                extent = self.figure_title.get_window_extent(renderer)
                if extent.width / dpi <= width_in - 2 * pad:
                    break
                self.figure_title.set_fontsize(self.figure_title.get_fontsize() * 0.9)
            title_height = extent.height / dpi + pad

        def measure() -> tuple[list, list, list, list]:
            left = [0.0] * columns
            right = [0.0] * columns
            top = [0.0] * rows
            bottom = [0.0] * rows
            for index, axes_object in enumerate(axes_list):
                row, column = divmod(index, columns)
                box = axes_object.get_window_extent(renderer)
                tight = axes_object.get_tightbbox(renderer)
                left[column] = max(left[column], (box.x0 - tight.x0) / dpi)
                right[column] = max(right[column], (tight.x1 - box.x1) / dpi)
                top[row] = max(top[row], (tight.y1 - box.y1) / dpi)
                bottom[row] = max(bottom[row], (box.y0 - tight.y0) / dpi)
            return left, right, top, bottom

        def place(left, right, top, bottom) -> None:
            avail_w = width_in - 2 * pad - sum(left) - sum(right) - gap * (columns - 1)
            avail_h = (
                height_in - 2 * pad - title_height - sum(top) - sum(bottom)
                - 0.5 * gap * (rows - 1)
            )
            panel_w = max(avail_w / columns, 0.3)
            panel_h = max(avail_h / rows, 0.3)
            if aspect is not None:
                if panel_h / panel_w > aspect:
                    panel_h = panel_w * aspect
                else:
                    panel_w = panel_h / aspect
            used_w = sum(left) + sum(right) + columns * panel_w + gap * (columns - 1)
            used_h = sum(top) + sum(bottom) + rows * panel_h + 0.5 * gap * (rows - 1)
            x_start = pad + max(width_in - 2 * pad - used_w, 0.0) / 2.0
            y_start = height_in - pad - title_height - max(
                height_in - 2 * pad - title_height - used_h, 0.0
            ) / 2.0
            x_edges = []
            x = x_start
            for column in range(columns):
                x += left[column]
                x_edges.append(x)
                x += panel_w + right[column] + gap
            y_tops = []
            y = y_start
            for row in range(rows):
                y -= top[row]
                y_tops.append(y)
                y -= panel_h + bottom[row] + 0.5 * gap
            for index, axes_object in enumerate(axes_list):
                row, column = divmod(index, columns)
                axes_object.set_position(
                    [
                        x_edges[column] / width_in,
                        (y_tops[row] - panel_h) / height_in,
                        panel_w / width_in,
                        panel_h / height_in,
                    ]
                )

        remembered = (
            self._layout_memory[1]
            if self._layout_memory is not None and self._layout_memory[0] == structure
            else None
        )
        # Start from the remembered layout (playback: same structure) or from
        # a first measurement, so the panels are at (nearly) their final size.
        place(*(remembered or measure()))
        self._fit_colorbar_labels(renderer, font_size)
        self._thin_ticks(axes_list, renderer, font_size)
        gutters = measure()  # tick counts depend on panel size: measure again
        if remembered is not None:
            gutters = tuple(
                [max(a, b) for a, b in zip(new, old)]
                for new, old in zip(gutters, remembered)
            )
        self._layout_memory = (structure, gutters)
        place(*gutters)
        # Final panel heights may differ slightly: make sure labels still fit.
        self._fit_colorbar_labels(renderer, font_size)

    def plot(self) -> None:
        try:
            if self.dump is None:
                return
            if self.field_data is None:
                self.load_variable()
            blocks = self._selected_blocks()
            is_1d = self._plot_dimension() == 1
            positive = self.log_var.get()

            line_width = self._style_metrics(
                self._panel_grid_shape(self._panel_count())
            )[1]
            with warnings.catch_warnings():
                # Clearing shared log axes briefly resets them to (0, 1).
                warnings.simplefilter("ignore", UserWarning)
                self.figure.clear()
            self.figure_title = None
            title = self._make_title()
            if is_1d:
                self._plot_1d_panels(blocks, positive, line_width, title)
            else:
                self._plot_2d_panels(blocks, positive, line_width, title)
        except Exception as exc:
            if self.exporting:
                # Propagate to load_dump/export_movie so the export aborts.
                raise
            self._show_error("Could not plot field", exc)

    # ------------------------------------------------------------ saving/export
    def save_figure(self) -> None:
        """Save the current figure; .mp4/.gif are routed to the movie exporter."""
        if self.dump_path is None:
            return
        suggested = (
            f"{self.dump_path.stem}_{self.variable_var.get().replace('.', '_')}"
        )
        if self._comparison_active():
            suggested += "_compare"
        suggested += ".pdf"
        filename = filedialog.asksaveasfilename(
            title="Save figure or movie",
            initialfile=suggested,
            defaultextension=".pdf",
            filetypes=(
                ("PDF vector figure", "*.pdf"),
                ("SVG vector figure", "*.svg"),
                ("PNG image", "*.png"),
                ("MP4 movie (all dumps)", "*.mp4"),
                ("Animated GIF (all dumps)", "*.gif"),
                ("All files", "*"),
            ),
        )
        if not filename:
            return
        if Path(filename).suffix.lower() in {".mp4", ".gif"}:
            self.export_movie(filename)
            return
        try:
            if self._raster_panels:
                # Rebuild the rasters at the output resolution for this save.
                dpi = mpl.rcParams["savefig.dpi"]
                self._raster_dpi = self.figure.dpi if dpi == "figure" else float(dpi)
                self._rerender_rasters(draw=False)
            self.figure.savefig(filename)
            self.status_var.set(f"Saved {filename}")
        except Exception as exc:
            self._show_error("Could not save figure", exc)
        finally:
            if self._raster_dpi is not None:
                self._raster_dpi = None
                self._rerender_rasters()

    def export_movie(self, filename: str | None = None) -> None:
        """Render all run-A dumps (with matching run-B dumps) to MP4 or GIF."""
        if self.dump_path is None:
            return
        self.pause_playback()
        try:
            fps = self._fps()
            if len(self.dump_paths) < 2:
                raise ValueError("Open a directory containing at least two PHDF dumps")
        except Exception as exc:
            self._show_error("Could not export movie", exc)
            return

        variable_slug = self.variable_var.get().replace(".", "_")
        if self._comparison_active():
            variable_slug += "_compare"
        if not filename:
            filename = filedialog.asksaveasfilename(
                title="Export all dumps as a movie",
                initialfile=f"{variable_slug}.mp4",
                defaultextension=".mp4",
                filetypes=(("MP4 movie", "*.mp4"), ("Animated GIF", "*.gif")),
            )
        if not filename:
            return

        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".gif"}:
            self._show_error(
                "Could not export movie",
                ValueError("Movie filename must end in .mp4 or .gif"),
            )
            return
        if suffix == ".mp4" and not animation.writers.is_available("ffmpeg"):
            self._show_error(
                "FFmpeg is not available",
                RuntimeError(
                    "MP4 export requires ffmpeg. Install it with one of:\n"
                    "  macOS:   brew install ffmpeg\n"
                    "  Ubuntu:  sudo apt install ffmpeg\n"
                    "  Conda:   conda install -c conda-forge ffmpeg\n\n"
                    "Make sure it is on the PATH of the shell that launches this "
                    "GUI, or export an animated GIF instead."
                ),
            )
            return
        if suffix == ".gif" and not animation.writers.is_available("pillow"):
            self._show_error(
                "Pillow is not available",
                RuntimeError(
                    "GIF export requires Pillow. Install it with:\n"
                    "python -m pip install pillow"
                ),
            )
            return

        original_path = self.dump_path
        frame_paths = list(self.dump_paths)
        original_size = self.figure.get_size_inches().copy()
        was_resizable = self.resizable()
        self.resizable(False, False)  # keep the frame size constant
        self.update_idletasks()
        self.exporting = True
        self._raster_dpi = MOVIE_DPI
        movie_saved = False
        export_failed = False

        # Fix the movie frame size once: an even number of pixels in each
        # direction at MOVIE_DPI (H.264/yuv420p requires even dimensions).
        # It is re-applied before every frame because Tk <Configure> events
        # (e.g. triggered by status-label updates) can otherwise resize the
        # Matplotlib figure mid-export, which corrupts the raw-video stream
        # fed to ffmpeg.
        w_in, h_in = original_size
        w_px = max(2, int(round(w_in * MOVIE_DPI)) // 2 * 2)
        h_px = max(2, int(round(h_in * MOVIE_DPI)) // 2 * 2)
        locked_size = (w_px / MOVIE_DPI, h_px / MOVIE_DPI)

        try:
            if suffix == ".mp4":
                writer = animation.FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    extra_args=[
                        # Broad player compatibility (QuickTime, browsers).
                        "-pix_fmt",
                        "yuv420p",
                        # Put the index at the front so players/browsers can
                        # start without reading the entire file.
                        "-movflags",
                        "+faststart",
                    ],
                    metadata={"title": variable_slug, "artist": "PHDF Plot Viewer"},
                )
            else:
                writer = animation.PillowWriter(fps=fps)

            self.figure.set_size_inches(*locked_size, forward=False)

            # savefig.bbox="tight" would make each frame a slightly different
            # size, which breaks video encoders, so disable it for the export.
            with mpl.rc_context({"savefig.bbox": None}):
                with writer.saving(self.figure, filename, dpi=MOVIE_DPI):
                    for frame_number, path in enumerate(frame_paths, start=1):
                        self.status_var.set(
                            f"Exporting frame {frame_number}/{len(frame_paths)}: "
                            f"{path.name}"
                        )
                        self.update_idletasks()
                        self.load_dump(path)
                        # Undo any Tk-driven resize before grabbing the frame.
                        self.figure.set_size_inches(*locked_size, forward=False)
                        self.canvas.draw()
                        writer.grab_frame()

            size_bytes = os.path.getsize(filename) if os.path.exists(filename) else 0
            if size_bytes < MIN_VALID_MOVIE_BYTES:
                raise RuntimeError(
                    f"The movie file is only {size_bytes} bytes, so no video "
                    "frames were written. Check the terminal for ffmpeg errors."
                )
            movie_saved = True
        except Exception as exc:
            export_failed = True
            self.exporting = False
            # Do not leave an unplayable, truncated file behind.
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except OSError:
                pass
            self._show_error("Could not export movie", exc)
        finally:
            self.exporting = False
            self._raster_dpi = None
            self.resizable(*was_resizable)
            try:
                self.figure.set_size_inches(*original_size, forward=False)
            except Exception:
                pass
            if original_path is not None and original_path.exists():
                self.load_dump(original_path)
            if movie_saved:
                self.status_var.set(f"Saved movie: {filename}")
            elif export_failed:
                self.status_var.set("Movie export failed (see error dialog)")

    def _show_error(self, heading: str, exc: Exception) -> None:
        detail = exception_details(exc)
        self.status_var.set(f"Error: {exc}")
        traceback.print_exc()
        messagebox.showerror(heading, detail, parent=self)

    def close(self) -> None:
        self.pause_playback()
        if self.dump is not None and hasattr(self.dump, "fid"):
            try:
                self.dump.fid.close()
            except Exception:
                pass
        self._close_dump2()
        self.destroy()


def main() -> None:
    configure_publication_style()
    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    compare_file = sys.argv[2] if len(sys.argv) > 2 else None
    app = PhdfPlotGui(initial_file, compare_file)
    app.mainloop()


if __name__ == "__main__":
    main()