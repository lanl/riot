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

"""Interactive PHDF viewer for 2-D Matplotlib plots.

Run with

    python phdf_plot_gui.py
    python phdf_plot_gui.py path/to/dump.phdf

The PHDF reader must be importable as either ``phdf`` or
``parthenon_tools.phdf``. Tkinter is included with most python.org and Conda
Python installations on macOS.
"""

from __future__ import annotations

import sys
import traceback
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from cycler import cycler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        }
    )


def coordinate_edges(centers: np.ndarray) -> np.ndarray:
    """Convert monotonic cell centers to edges, including nonuniform grids."""
    centers = np.asarray(centers, dtype=float).squeeze()
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Coordinates must be a nonempty 1-D array")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - (mids[0] - centers[0])],
            mids,
            [centers[-1] + (centers[-1] - mids[-1])],
        )
    )


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


class PhdfPlotGui(tk.Tk):
    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.title("PHDF Plot Viewer")
        self.geometry("1050x850")
        self.minsize(850, 600)

        self.dump = None
        self.dump_path: Path | None = None
        self.dump_paths: list[Path] = []
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
        self.vmin_var = tk.StringVar()
        self.vmax_var = tk.StringVar()
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()
        self.floor_var = tk.StringVar(value="1e-30")
        self.status_var = tk.StringVar(value="Choose a PHDF dump to begin")

        self._build_controls()
        self._build_figure()

        self.protocol("WM_DELETE_WINDOW", self.close)
        if initial_file:
            self.after(50, lambda: self.load_dump(Path(initial_file)))

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

        ttk.Label(panel, text="Dump").grid(row=row, column=0, sticky="w", pady=(8, 2))
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
        ttk.Label(panel, text="Example: component,z = 4,0", foreground="#666666").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
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
        self._entry_row(panel, row, "Color min", self.vmin_var)
        row += 1
        self._entry_row(panel, row, "Color max", self.vmax_var)
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
        ttk.Button(buttons, text="Plot", command=self.plot).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        ttk.Button(buttons, text="Auto ranges", command=self.auto_ranges).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )
        row += 1
        ttk.Button(panel, text="Save figure…", command=self.save_figure).grid(
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
        # Axes positions are managed explicitly below so changing colorbar tick
        # label widths cannot shift the data axes during playback.
        self.figure = Figure(figsize=(6, 5), dpi=100, constrained_layout=False)
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

    def _style_metrics(self) -> tuple[float, float]:
        """Return responsive font and line sizes for the current canvas."""
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
        font_size = base_font * scale
        line_width = 1.1 * scale
        return font_size, line_width

    def _canvas_resized(self, _event=None) -> None:
        """Debounce resize events so styling follows the window smoothly."""
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self._resize_job = None
        if (
            self.panel_layout_var.get() == "Auto"
            and self.dump is not None
            and self.line_panels
        ):
            new_shape = self._panel_grid_shape(1 + len(self.line_panels))
            if new_shape != self._last_panel_grid_shape:
                self.plot()
                return
        self._apply_plot_style()

    def _apply_plot_style(self, draw=True) -> None:
        """Scale plot text, ticks, spines, and outlines as one visual system."""
        self._resize_job = None
        if self.main_axes is None or self.main_axes not in self.figure.axes:
            return
        try:
            font_size, line_width = self._style_metrics()
        except (TypeError, ValueError):
            return
        tick_font = max(font_size - 1.0, 1.0)
        major_length = 0.42 * font_size
        minor_length = 0.25 * font_size

        for axes in self.figure.axes:
            axes.tick_params(
                axis="both",
                which="major",
                labelsize=tick_font,
                length=major_length,
                width=line_width,
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
            axes.title.set_fontsize(font_size)

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
            self.load_dump(self.dump_path)

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
            self.dump_paths = sorted(path.parent.glob("*.phdf"))
            self.dump_combo["values"] = [p.name for p in self.dump_paths]
            self.dump_var.set(path.name)

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
                # A newly selected dump gets its own data and coordinate ranges.
                self.auto_ranges(redraw=False)
                self.plot()
            else:
                self.status_var.set(f"No plottable field datasets found in {path.name}")
        except Exception as exc:
            self._show_error("Could not load dump", exc)

    def _variable_selected(self, _event=None) -> None:
        self.load_variable()
        self.auto_ranges(redraw=False)
        self.plot()

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
                data = np.asarray(self.dump.Get(variable.get(), flatten=False))
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
                data = np.asarray(self.dump.Get(variable.get(), flatten=False))
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
            self.auto_ranges(redraw=True)

    def load_variable(self) -> None:
        if self.dump is None or not self.variable_var.get():
            return
        self.status_var.set(f"Loading {self.variable_var.get()}…")
        self.update_idletasks()
        self.field_data = np.asarray(
            self.dump.Get(self.variable_var.get(), flatten=False)
        )
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
        y = np.asarray(self.dump.y)
        return 1 if y.ndim < 2 or y.shape[-1] == 1 else 2

    def _selected_blocks(self) -> list[np.ndarray]:
        if self.field_data is None:
            self.load_variable()
        assert self.field_data is not None
        return self._blocks_from_field_data(self.field_data, self.indices_var.get())

    def _blocks_from_field_data(
        self, field_data: np.ndarray, indices_text: str
    ) -> list[np.ndarray]:
        """Select component/slice indices and return one spatial array per block."""
        spatial_dimension = self._plot_dimension()
        selector_count = field_data.ndim - 1 - spatial_dimension
        indices = parse_indices(indices_text, selector_count)
        spatial_slices = (slice(None),) * spatial_dimension
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

    def _finite_range(
        self, blocks: list[np.ndarray], positive=False
    ) -> tuple[float, float]:
        values = np.concatenate([np.asarray(a).ravel() for a in blocks])
        values = values[np.isfinite(values)]
        if positive:
            floor = float(self.floor_var.get())
            values = values[values > max(floor, 0.0)]
        if values.size == 0:
            qualifier = "positive " if positive else "finite "
            raise ValueError(f"The selected field contains no {qualifier}values")
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            if positive:
                lo, hi = lo / 1.1, hi * 1.1
            else:
                # Give a constant linear field a visible, nonzero color span.
                # The absolute floor is useful for fields that are identically
                # zero or very close to it.
                delta = max(abs(lo) * 1.0e-6, 1.0e-6)
                lo, hi = lo - delta, hi + delta
        return lo, hi

    def auto_ranges(self, redraw=True) -> None:
        try:
            blocks = self._selected_blocks()
            is_1d = self._plot_dimension() == 1
            lo, hi = self._finite_range(blocks, positive=self.log_var.get())
            self.vmin_var.set(f"{lo:.8g}")
            self.vmax_var.set(f"{hi:.8g}")
            if self.dump is not None:
                self.xmin_var.set(f"{np.nanmin(self.dump.x):.8g}")
                self.xmax_var.set(f"{np.nanmax(self.dump.x):.8g}")
                if is_1d:
                    if self.log_var.get():
                        padding_factor = (hi / lo) ** 0.03
                        self.ymin_var.set(f"{lo / padding_factor:.8g}")
                        self.ymax_var.set(f"{hi * padding_factor:.8g}")
                    else:
                        padding = 0.03 * (hi - lo)
                        self.ymin_var.set(f"{lo - padding:.8g}")
                        self.ymax_var.set(f"{hi + padding:.8g}")
                else:
                    self.ymin_var.set(f"{np.nanmin(self.dump.y):.8g}")
                    self.ymax_var.set(f"{np.nanmax(self.dump.y):.8g}")
            if redraw:
                self.plot()
        except Exception as exc:
            self._show_error("Could not determine ranges", exc)

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
        """Load the primary and additional variables for the current dump."""
        specs = [
            {
                "variable": self.variable_var.get(),
                "indices": self.indices_var.get(),
                "blocks": main_blocks,
                "primary": True,
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
            field_data = np.asarray(self.dump.Get(variable, flatten=False))
            specs.append(
                {
                    "variable": variable,
                    "indices": panel["indices"],
                    "blocks": self._blocks_from_field_data(
                        field_data, panel["indices"]
                    ),
                    "primary": False,
                }
            )
        return specs

    def _plot_1d_panels(
        self,
        main_blocks: list[np.ndarray],
        positive: bool,
        line_width: float,
        title: str,
    ) -> None:
        """Draw the primary field and all requested extra fields as stacked panels."""
        specs = self._panel_specs(main_blocks)

        rows, columns = self._panel_grid_shape(len(specs))
        self._last_panel_grid_shape = (rows, columns)
        grid = self.figure.add_gridspec(
            rows,
            columns,
            left=0.11,
            right=0.96,
            bottom=0.08,
            top=0.91,
            hspace=0.14,
            wspace=0.28,
        )
        axes = []
        self.data_lines = []
        self.mesh_lines = []
        self.colorbar = None
        self.colorbars = []
        for panel_index, spec in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axes_object = self.figure.add_subplot(
                grid[row, column], sharex=axes[0] if axes else None
            )
            axes.append(axes_object)
            panel_blocks = spec["blocks"]
            for block, data in enumerate(panel_blocks):
                x = np.asarray(self.dump.x[block, :])
                if data.shape != x.shape:
                    raise ValueError(
                        f"{spec['variable']}, block {block}: data shape {data.shape} "
                        f"does not match x-coordinate shape {x.shape}"
                    )
                (data_line,) = axes_object.plot(
                    x,
                    data,
                    color=COLORS[0],
                    linewidth=max(1.4, 1.4 * line_width),
                )
                self.data_lines.append(data_line)
                if self.mesh_var.get():
                    xedges = coordinate_edges(x)
                    for edge in (xedges[0], xedges[-1]):
                        mesh_line = axes_object.axvline(
                            edge,
                            color="black",
                            linewidth=max(0.45, 0.45 * line_width),
                        )
                        self.mesh_lines.append(mesh_line)

            axes_object.set_ylabel(spec["variable"])
            axes_object.set_yscale("log" if positive else "linear")
            if spec["primary"]:
                ymin = optional_float(self.ymin_var.get())
                ymax = optional_float(self.ymax_var.get())
                auto_ymin, auto_ymax = self._line_y_limits(panel_blocks, positive)
                ymin = auto_ymin if ymin is None else ymin
                ymax = auto_ymax if ymax is None else ymax
            else:
                ymin, ymax = self._line_y_limits(panel_blocks, positive)
            if positive and ymin <= 0:
                raise ValueError(
                    f"Log y scaling requires a positive y minimum for {spec['variable']}"
                )
            axes_object.set_ylim(ymin, ymax)
            if row < rows - 1:
                axes_object.tick_params(labelbottom=False)

        self.line_panel_axes = axes
        self.main_axes = axes[0]
        xmin = optional_float(self.xmin_var.get())
        xmax = optional_float(self.xmax_var.get())
        if xmin is not None or xmax is not None:
            axes[0].set_xlim(left=xmin, right=xmax)
        for panel_index, axes_object in enumerate(axes):
            row, _ = divmod(panel_index, columns)
            if row == rows - 1:
                axes_object.set_xlabel("x")
        self.figure_title = self.figure.suptitle(title, y=0.985)
        self._apply_plot_style(draw=False)
        self.canvas.draw_idle()
        self.status_var.set(
            f"{len(specs)} line panel{'s' if len(specs) != 1 else ''}  |  "
            f"{self.dump_path.name}"
        )

    def _color_norm(
        self,
        blocks: list[np.ndarray],
        positive: bool,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[mcolors.Normalize, float, float]:
        auto_lo, auto_hi = self._finite_range(blocks, positive=positive)
        vmin = auto_lo if vmin is None else vmin
        vmax = auto_hi if vmax is None else vmax
        if positive:
            vmin = max(vmin, float(self.floor_var.get()))
            if vmin <= 0:
                raise ValueError("Log color scaling requires a positive color minimum")
        if vmin == vmax:
            # Equal explicit limits (or values rounded equal in the GUI) are
            # harmless: expand them here so Normalize never sees a zero-width
            # interval. Keep logarithmic limits strictly positive.
            if positive:
                vmin, vmax = vmin / 1.1, vmax * 1.1
            else:
                delta = max(abs(vmin) * 1.0e-6, 1.0e-6)
                vmin, vmax = vmin - delta, vmax + delta
        elif vmin > vmax:
            raise ValueError("Color min must be smaller than color max")
        norm = (
            mcolors.LogNorm(vmin=vmin, vmax=vmax)
            if positive
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        return norm, vmin, vmax

    def _plot_2d_panels(
        self,
        main_blocks: list[np.ndarray],
        positive: bool,
        line_width: float,
        title: str,
    ) -> None:
        """Draw primary and additional 2-D fields with independent colorbars."""
        specs = self._panel_specs(main_blocks)
        rows, columns = self._panel_grid_shape(len(specs))
        self._last_panel_grid_shape = (rows, columns)
        grid = self.figure.add_gridspec(
            rows,
            columns,
            left=0.09,
            # Leave a fixed outer gutter for the last colorbar's tick labels
            # and rotated variable label. AxesDivider keeps this stable even
            # when scientific-notation widths change between frames.
            right=0.87,
            bottom=0.07,
            top=0.91,
            hspace=0.16,
            wspace=0.40,
        )
        axes = []
        self.mesh_lines = []
        self.data_lines = []
        self.colorbars = []
        for panel_index, spec in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axes_object = self.figure.add_subplot(
                grid[row, column],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
            axes.append(axes_object)
            panel_blocks = spec["blocks"]
            requested_vmin = (
                optional_float(self.vmin_var.get()) if spec["primary"] else None
            )
            requested_vmax = (
                optional_float(self.vmax_var.get()) if spec["primary"] else None
            )
            norm, _, _ = self._color_norm(
                panel_blocks, positive, requested_vmin, requested_vmax
            )
            pcol = None
            for block, data in enumerate(panel_blocks):
                xedges = coordinate_edges(self.dump.x[block, :])
                yedges = coordinate_edges(self.dump.y[block, :])
                expected_shape = (len(yedges) - 1, len(xedges) - 1)
                if data.shape != expected_shape:
                    raise ValueError(
                        f"{spec['variable']}, block {block}: data shape {data.shape} "
                        f"does not match coordinate shape {expected_shape}"
                    )
                pcol = axes_object.pcolormesh(
                    xedges,
                    yedges,
                    data,
                    norm=norm,
                    cmap=self.cmap_var.get(),
                    shading="flat",
                    rasterized=True,
                )
                if self.mesh_var.get():
                    (mesh_line,) = axes_object.plot(
                        [xedges[0], xedges[-1], xedges[-1], xedges[0], xedges[0]],
                        [yedges[0], yedges[0], yedges[-1], yedges[-1], yedges[0]],
                        color="black",
                        linewidth=max(0.45, 0.45 * line_width),
                    )
                    self.mesh_lines.append(mesh_line)

            assert pcol is not None
            divider = make_axes_locatable(axes_object)
            colorbar_axes = divider.append_axes("right", size="4%", pad="3%")
            colorbar = self.figure.colorbar(pcol, cax=colorbar_axes)
            colorbar.set_label(spec["variable"])
            self.colorbars.append(colorbar)
            if column == 0:
                axes_object.set_ylabel("y")
            else:
                axes_object.tick_params(labelleft=False)
            axes_object.set_aspect("equal" if self.equal_aspect_var.get() else "auto")
            if row < rows - 1:
                axes_object.tick_params(labelbottom=False)

        self.colorbar = self.colorbars[0]
        self.line_panel_axes = axes
        self.main_axes = axes[0]
        xmin = optional_float(self.xmin_var.get())
        xmax = optional_float(self.xmax_var.get())
        ymin = optional_float(self.ymin_var.get())
        ymax = optional_float(self.ymax_var.get())
        if xmin is not None or xmax is not None:
            axes[0].set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            axes[0].set_ylim(bottom=ymin, top=ymax)
        for panel_index, axes_object in enumerate(axes):
            row, _ = divmod(panel_index, columns)
            if row == rows - 1:
                axes_object.set_xlabel("x")
        self.figure_title = self.figure.suptitle(title, y=0.985)
        self._apply_plot_style(draw=False)
        self.canvas.draw_idle()
        self.status_var.set(
            f"{len(specs)} 2-D panel{'s' if len(specs) != 1 else ''}  |  "
            f"{self.dump_path.name}"
        )

    def plot(self) -> None:
        try:
            if self.dump is None:
                return
            _, line_width = self._style_metrics()
            if self.field_data is None:
                self.load_variable()
            blocks = self._selected_blocks()
            is_1d = self._plot_dimension() == 1
            positive = self.log_var.get()

            self.figure.clear()
            self.figure_title = None
            time = getattr(self.dump, "Time", None)
            title = self.dump_path.name if self.dump_path else "PHDF"
            if time is not None:
                title += f"    t = {float(time):.3e} s"
            if is_1d:
                self._plot_1d_panels(blocks, positive, line_width, title)
            else:
                self._plot_2d_panels(blocks, positive, line_width, title)
        except Exception as exc:
            self._show_error("Could not plot field", exc)

    def save_figure(self) -> None:
        if self.dump_path is None:
            return
        suggested = (
            f"{self.dump_path.stem}_{self.variable_var.get().replace('.', '_')}.pdf"
        )
        filename = filedialog.asksaveasfilename(
            title="Save figure",
            initialfile=suggested,
            defaultextension=".pdf",
            filetypes=(
                ("PDF vector figure", "*.pdf"),
                ("SVG vector figure", "*.svg"),
                ("PNG image", "*.png"),
                ("All files", "*"),
            ),
        )
        if filename:
            try:
                self.figure.savefig(filename)
                self.status_var.set(f"Saved {filename}")
            except Exception as exc:
                self._show_error("Could not save figure", exc)

    def export_movie(self) -> None:
        """Render all dumps in the selected directory to MP4 or animated GIF."""
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
                    "MP4 export requires ffmpeg. On macOS, install it with:\n"
                    "brew install ffmpeg\n\nAlternatively, export an animated GIF."
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
        self.exporting = True
        movie_saved = False
        try:
            writer = (
                animation.FFMpegWriter(
                    fps=fps,
                    metadata={"title": variable_slug, "artist": "PHDF Plot Viewer"},
                )
                if suffix == ".mp4"
                else animation.PillowWriter(fps=fps)
            )
            with writer.saving(self.figure, filename, dpi=self.figure.dpi):
                for frame_number, path in enumerate(frame_paths, start=1):
                    self.status_var.set(
                        f"Exporting frame {frame_number}/{len(frame_paths)}: {path.name}"
                    )
                    self.update_idletasks()
                    self.load_dump(path)
                    self.canvas.draw()
                    writer.grab_frame()
            movie_saved = True
        except Exception as exc:
            self._show_error("Could not export movie", exc)
        finally:
            self.exporting = False
            if original_path is not None and original_path.exists():
                self.load_dump(original_path)
            if movie_saved:
                self.status_var.set(f"Saved movie: {filename}")

    def _show_error(self, heading: str, exc: Exception) -> None:
        self.status_var.set(f"Error: {exc}")
        traceback.print_exc()
        messagebox.showerror(heading, str(exc), parent=self)

    def close(self) -> None:
        self.pause_playback()
        if self.dump is not None and hasattr(self.dump, "fid"):
            try:
                self.dump.fid.close()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    configure_publication_style()
    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    app = PhdfPlotGui(initial_file)
    app.mainloop()


if __name__ == "__main__":
    main()
