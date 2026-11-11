"""Matplotlib rendering onto a caller-owned figure."""

from collections.abc import Callable
from dataclasses import dataclass
import math

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .errors import InvalidStyleError, RenderError
from .models import DEFAULT_LINE_COLORS, PanelMargins, PlotDimension, PlotRequest, RenderResult
from .ranges import coordinate_edges, finite_range, line_y_limits, panel_grid, validate_limits


COLORS = DEFAULT_LINE_COLORS


@dataclass(frozen=True)
class RenderArtifacts:
    figure: object
    data_axes: tuple[object, ...]
    colorbars: tuple[object, ...]
    title_artist: object
    dimension: PlotDimension


RenderCustomizer = Callable[[RenderArtifacts, PlotRequest], None]


@dataclass(frozen=True)
class _Context:
    font_size: float
    title_size: float
    axis_label_size: float
    tick_label_size: float
    colorbar_label_size: float
    colorbar_tick_size: float
    font_family: tuple[str, ...]
    title_weight: str
    axis_label_weight: str
    data_line_width: float
    block_outline_width: float
    spine_width: float
    colorbar_outline_width: float
    major_tick_width: float
    minor_tick_width: float
    major_tick_length: float
    minor_tick_length: float
    figure_background: str
    axes_background: str
    text_color: str
    title_color: str
    axis_label_color: str
    tick_color: str
    spine_color: str
    grid_color: str
    block_outline_color: str
    colorbar_outline_color: str
    line_palette: tuple[str, ...]
    colormap: object
    grid_visible: bool
    grid_axis: str
    grid_which: str
    grid_line_style: object
    grid_line_width: float
    grid_alpha: float
    one_dimensional_margins: PanelMargins
    two_dimensional_margins: PanelMargins
    title_y: float
    colorbar_size: float
    colorbar_padding: float
    rasterize_mesh: bool


class RshowRenderer:
    """Render a request onto a supplied Matplotlib figure.

    Explicit metrics participate in responsive scaling when
    ``auto_scale_style`` is enabled. Disable it to use fixed metrics.
    """

    def __init__(self, customizer: RenderCustomizer | None = None):
        self._customizer = customizer

    def render(self, figure, dataset, request: PlotRequest, canvas_size=None) -> RenderResult:
        try:
            context = self._context(request, canvas_size)
            specs = [(request.primary, dataset.select_blocks(request.primary), True)]
            specs.extend(
                (selection, dataset.select_blocks(selection), False)
                for selection in request.additional_panels
            )
            figure.clear()
            figure.patch.set_facecolor(context.figure_background)
            title = dataset.path.name
            if dataset.info.time is not None:
                title += f"    t = {dataset.info.time:.3e} s"
            if dataset.info.dimension is PlotDimension.ONE_D:
                axes, colorbars = self._render_1d(figure, dataset, request, specs, context)
            else:
                axes, colorbars = self._render_2d(figure, dataset, request, specs, context)
            title_artist = self._finish(figure, axes, colorbars, title, context)
            artifacts = RenderArtifacts(
                figure, tuple(axes), tuple(colorbars), title_artist, dataset.info.dimension
            )
            if self._customizer is not None:
                try:
                    self._customizer(artifacts, request)
                except Exception as exc:
                    raise RenderError(f"Render customizer failed: {exc}") from exc
            return RenderResult(len(specs), dataset.info.dimension, title)
        except InvalidStyleError:
            raise
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(str(exc)) from exc

    def _context(self, request, canvas_size):
        style = request.style
        appearance = style.appearance
        base_font = self._positive("base_font_size", style.base_font_size)
        scale = 1.0
        if style.auto_scale_style and canvas_size:
            scale = float(np.clip(np.sqrt(max(canvas_size[0], 1) * max(canvas_size[1], 1) / (700.0 * 650.0)), 0.85, 1.6))
        font_size = base_font * scale
        line_width = 1.1 * scale
        typography = appearance.typography
        strokes = appearance.strokes
        colors = appearance.colors
        grid = appearance.grid
        layout = appearance.layout

        if not colors.line_palette:
            raise InvalidStyleError("colors.line_palette must not be empty")
        color_fields = {
            "colors.figure_background": colors.figure_background,
            "colors.axes_background": colors.axes_background,
            "colors.text": colors.text,
            "colors.title": colors.title,
            "colors.axis_label": colors.axis_label,
            "colors.tick": colors.tick,
            "colors.spine": colors.spine,
            "colors.grid": colors.grid,
            "colors.block_outline": colors.block_outline,
            "colors.colorbar_outline": colors.colorbar_outline,
            **{f"colors.line_palette[{i}]": color for i, color in enumerate(colors.line_palette)},
        }
        for name, color in (
            ("colors.colormap_bad", colors.colormap_bad),
            ("colors.colormap_under", colors.colormap_under),
            ("colors.colormap_over", colors.colormap_over),
        ):
            if color is not None:
                color_fields[name] = color
        for name, color in color_fields.items():
            self._validate_color(name, color)

        one_dimensional_margins = self._validate_margins(
            "layout.one_dimensional", layout.one_dimensional
        )
        two_dimensional_margins = self._validate_margins(
            "layout.two_dimensional", layout.two_dimensional
        )
        title_y = self._finite("layout.title_y", layout.title_y)
        colorbar_size = self._positive("layout.colorbar_size", layout.colorbar_size)
        colorbar_padding = self._nonnegative("layout.colorbar_padding", layout.colorbar_padding)

        if grid.axis not in {"x", "y", "both"}:
            raise InvalidStyleError("grid.axis must be 'x', 'y', or 'both'")
        if grid.which not in {"major", "minor", "both"}:
            raise InvalidStyleError("grid.which must be 'major', 'minor', or 'both'")
        grid_line_width = self._positive("grid.line_width", grid.line_width) * scale
        grid_alpha = self._finite("grid.alpha", grid.alpha)
        if not 0.0 <= grid_alpha <= 1.0:
            raise InvalidStyleError("grid.alpha must be between 0 and 1")
        try:
            Line2D([], [], linestyle=grid.line_style)
        except Exception as exc:
            raise InvalidStyleError(f"grid.line_style is invalid: {grid.line_style!r}") from exc
        for name, weight in (
            ("typography.title_weight", typography.title_weight),
            ("typography.axis_label_weight", typography.axis_label_weight),
        ):
            try:
                FontProperties(weight=weight)
            except Exception as exc:
                raise InvalidStyleError(f"{name} is invalid: {weight!r}") from exc
        if any(not isinstance(family, str) or not family for family in typography.font_family):
            raise InvalidStyleError("typography.font_family entries must be nonempty strings")
        try:
            colormap = mpl.colormaps.get_cmap(style.colormap).copy()
        except Exception as exc:
            raise InvalidStyleError(f"colormap is invalid: {style.colormap!r}") from exc
        extremes = {
            key: value
            for key, value in (
                ("bad", colors.colormap_bad),
                ("under", colors.colormap_under),
                ("over", colors.colormap_over),
            )
            if value is not None
        }
        if extremes:
            colormap = colormap.with_extremes(**extremes)

        def metric(name, value, default):
            return default if value is None else self._positive(name, value) * scale

        return _Context(
            font_size,
            metric("typography.title_size", typography.title_size, font_size),
            metric("typography.axis_label_size", typography.axis_label_size, font_size),
            metric("typography.tick_label_size", typography.tick_label_size, max(font_size - 1.0, 1.0)),
            metric("typography.colorbar_label_size", typography.colorbar_label_size, font_size),
            metric("typography.colorbar_tick_size", typography.colorbar_tick_size, max(font_size - 1.0, 1.0)),
            tuple(typography.font_family),
            typography.title_weight,
            typography.axis_label_weight,
            metric("strokes.data_line_width", strokes.data_line_width, max(1.4, 1.4 * line_width)),
            metric("strokes.block_outline_width", strokes.block_outline_width, max(0.45, 0.45 * line_width)),
            metric("strokes.spine_width", strokes.spine_width, line_width),
            metric("strokes.colorbar_outline_width", strokes.colorbar_outline_width, line_width),
            metric("strokes.major_tick_width", strokes.major_tick_width, line_width),
            metric("strokes.minor_tick_width", strokes.minor_tick_width, 0.72 * line_width),
            metric("strokes.major_tick_length", strokes.major_tick_length, 0.42 * font_size),
            metric("strokes.minor_tick_length", strokes.minor_tick_length, 0.25 * font_size),
            colors.figure_background, colors.axes_background, colors.text, colors.title,
            colors.axis_label, colors.tick, colors.spine, colors.grid,
            colors.block_outline, colors.colorbar_outline, tuple(colors.line_palette),
            colormap, bool(grid.visible), grid.axis, grid.which, grid.line_style,
            grid_line_width, grid_alpha, one_dimensional_margins, two_dimensional_margins,
            title_y, colorbar_size, colorbar_padding, bool(appearance.rasterize_mesh),
        )

    def _validate_color(self, name, color):
        try:
            mcolors.to_rgba(color)
        except (TypeError, ValueError) as exc:
            raise InvalidStyleError(f"{name} is not a valid color: {color!r}") from exc

    def _finite(self, name, value):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidStyleError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise InvalidStyleError(f"{name} must be a finite number")
        return result

    def _positive(self, name, value):
        result = self._finite(name, value)
        if result <= 0:
            raise InvalidStyleError(f"{name} must be positive")
        return result

    def _nonnegative(self, name, value):
        result = self._finite(name, value)
        if result < 0:
            raise InvalidStyleError(f"{name} must be nonnegative")
        return result

    def _validate_margins(self, name, margins):
        left = self._finite(f"{name}.left", margins.left)
        right = self._finite(f"{name}.right", margins.right)
        bottom = self._finite(f"{name}.bottom", margins.bottom)
        top = self._finite(f"{name}.top", margins.top)
        self._nonnegative(f"{name}.horizontal_space", margins.horizontal_space)
        self._nonnegative(f"{name}.vertical_space", margins.vertical_space)
        if not 0 <= left < right <= 1:
            raise InvalidStyleError(f"{name} requires 0 <= left < right <= 1")
        if not 0 <= bottom < top <= 1:
            raise InvalidStyleError(f"{name} requires 0 <= bottom < top <= 1")
        return PanelMargins(
            left,
            right,
            bottom,
            top,
            float(margins.horizontal_space),
            float(margins.vertical_space),
        )

    def _grid(self, figure, count, request, dimension, context):
        width, height = figure.get_size_inches()
        rows, columns = panel_grid(count, request.style.panel_layout, width / max(height, 0.1))
        margins = context.one_dimensional_margins if dimension is PlotDimension.ONE_D else context.two_dimensional_margins
        return rows, columns, figure.add_gridspec(
            rows, columns, left=margins.left, right=margins.right,
            bottom=margins.bottom, top=margins.top,
            hspace=margins.vertical_space, wspace=margins.horizontal_space,
        )

    def _new_axis(self, figure, subplot_spec, context, **kwargs):
        axis = figure.add_subplot(subplot_spec, **kwargs)
        axis.set_facecolor(context.axes_background)
        axis.set_prop_cycle(cycler(color=context.line_palette))
        return axis

    def _render_1d(self, figure, dataset, request, specs, context):
        rows, columns, grid = self._grid(figure, len(specs), request, PlotDimension.ONE_D, context)
        axes = []
        for panel_index, (selection, blocks, primary) in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axis = self._new_axis(figure, grid[row, column], context, sharex=axes[0] if axes else None)
            axes.append(axis)
            block_series = []
            for block_index, data in enumerate(blocks):
                x = np.asarray(dataset.x_coordinates[block_index, :])
                if data.shape != x.shape:
                    raise RenderError(f"{selection.variable}, block {block_index}: data shape {data.shape} does not match x-coordinate shape {x.shape}")
                edges = coordinate_edges(x)
                block_series.append((edges[0], edges[-1], edges[1] - edges[0], edges[-1] - edges[-2], x, data))
                axis.plot(x, data, color=context.line_palette[0], linewidth=context.data_line_width)
                if request.style.show_block_outlines:
                    for edge in (edges[0], edges[-1]):
                        axis.axvline(edge, color=context.block_outline_color, linewidth=context.block_outline_width)
            block_series.sort(key=lambda series: series[0])
            for left, right in zip(block_series, block_series[1:]):
                scale = max(abs(left[1]), abs(right[0]), 1.0)
                tolerance = max(32.0 * np.finfo(float).eps * scale, 0.05 * min(abs(left[3]), abs(right[2])))
                if abs(left[1] - right[0]) <= tolerance:
                    axis.plot([left[4][-1], right[4][0]], [left[5][-1], right[5][0]], color=context.line_palette[0], linewidth=context.data_line_width)
            axis.set_ylabel(selection.variable)
            axis.set_yscale("log" if request.style.logarithmic else "linear")
            if primary and (request.limits.y_min is not None or request.limits.y_max is not None):
                auto_low, auto_high = line_y_limits(blocks, request.style.logarithmic, request.style.log_floor)
                low = auto_low if request.limits.y_min is None else request.limits.y_min
                high = auto_high if request.limits.y_max is None else request.limits.y_max
            else:
                low, high = line_y_limits(blocks, request.style.logarithmic, request.style.log_floor)
            validate_limits(low, high, request.style.logarithmic)
            axis.set_ylim(low, high)
            if row < rows - 1:
                axis.tick_params(labelbottom=False)
        self._apply_shared_limits(axes[0], request, False)
        self._set_x_labels(axes, rows, columns)
        return axes, []

    def _norm(self, blocks, request, primary):
        low, high = finite_range(blocks, request.style.logarithmic, request.style.log_floor)
        if primary:
            low = low if request.limits.value_min is None else request.limits.value_min
            high = high if request.limits.value_max is None else request.limits.value_max
        if low == high:
            if request.style.logarithmic:
                low, high = low / 1.1, high * 1.1
            else:
                delta = max(abs(low) * 1.0e-6, 1.0e-6)
                low, high = low - delta, high + delta
        low = max(low, request.style.log_floor) if request.style.logarithmic else low
        validate_limits(low, high, request.style.logarithmic)
        return (mcolors.LogNorm if request.style.logarithmic else mcolors.Normalize)(vmin=low, vmax=high)

    def _render_2d(self, figure, dataset, request, specs, context):
        rows, columns, grid = self._grid(figure, len(specs), request, PlotDimension.TWO_D, context)
        axes, colorbars = [], []
        for panel_index, (selection, blocks, primary) in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axis = self._new_axis(figure, grid[row, column], context, sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
            axes.append(axis)
            norm, mesh = self._norm(blocks, request, primary), None
            for block_index, data in enumerate(blocks):
                x_edges = coordinate_edges(dataset.x_coordinates[block_index, :])
                y_edges = coordinate_edges(dataset.y_coordinates[block_index, :])
                expected = (len(y_edges) - 1, len(x_edges) - 1)
                if data.shape != expected:
                    raise RenderError(f"{selection.variable}, block {block_index}: data shape {data.shape} does not match coordinate shape {expected}")
                mesh = axis.pcolormesh(x_edges, y_edges, data, norm=norm, cmap=context.colormap, shading="flat", rasterized=context.rasterize_mesh)
                if request.style.show_block_outlines:
                    axis.plot([x_edges[0], x_edges[-1], x_edges[-1], x_edges[0], x_edges[0]], [y_edges[0], y_edges[0], y_edges[-1], y_edges[-1], y_edges[0]], color=context.block_outline_color, linewidth=context.block_outline_width)
            divider = make_axes_locatable(axis)
            colorbar_axis = divider.append_axes("right", size=f"{100.0 * context.colorbar_size:g}%", pad=f"{100.0 * context.colorbar_padding:g}%")
            colorbar_axis.set_facecolor(context.axes_background)
            colorbar = figure.colorbar(mesh, cax=colorbar_axis)
            colorbar.set_label(selection.variable)
            colorbars.append(colorbar)
            if column == 0:
                axis.set_ylabel("y")
            else:
                axis.tick_params(labelleft=False)
            axis.set_aspect("equal" if request.style.equal_aspect else "auto")
            if row < rows - 1:
                axis.tick_params(labelbottom=False)
        self._apply_shared_limits(axes[0], request, True)
        self._set_x_labels(axes, rows, columns)
        return axes, colorbars

    def _apply_shared_limits(self, axis, request, include_y):
        limits = request.limits
        if limits.x_min is not None or limits.x_max is not None:
            axis.set_xlim(left=limits.x_min, right=limits.x_max)
        if include_y and (limits.y_min is not None or limits.y_max is not None):
            axis.set_ylim(bottom=limits.y_min, top=limits.y_max)

    def _set_x_labels(self, axes, rows, columns):
        for index, axis in enumerate(axes):
            row, _ = divmod(index, columns)
            if row == rows - 1:
                axis.set_xlabel("x")

    def _finish(self, figure, axes, colorbars, title, context):
        for axis in axes:
            self._style_data_axis(axis, context)
        for colorbar in colorbars:
            self._style_colorbar(colorbar, context)
        title_artist = figure.suptitle(title, y=context.title_y, fontsize=context.title_size, fontweight=context.title_weight, color=context.title_color)
        self._set_font_family(title_artist, context.font_family)
        return title_artist

    def _style_data_axis(self, axis, context):
        axis.set_facecolor(context.axes_background)
        axis.set_prop_cycle(cycler(color=context.line_palette))
        axis.tick_params(axis="both", which="major", colors=context.tick_color, labelcolor=context.tick_color, labelsize=context.tick_label_size, length=context.major_tick_length, width=context.major_tick_width)
        axis.tick_params(axis="both", which="minor", colors=context.tick_color, labelcolor=context.tick_color, length=context.minor_tick_length, width=context.minor_tick_width)
        for spine in axis.spines.values():
            spine.set_color(context.spine_color)
            spine.set_linewidth(context.spine_width)
        for label in (axis.xaxis.label, axis.yaxis.label):
            self._style_text(label, context.axis_label_color, context.axis_label_size, context.font_family, context.axis_label_weight)
        self._style_text(axis.title, context.text_color, context.title_size, context.font_family, context.title_weight)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            self._style_text(label, context.tick_color, context.tick_label_size, context.font_family)
        for offset in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            self._style_text(offset, context.tick_color, context.tick_label_size, context.font_family)
        if context.grid_visible:
            axis.grid(True, axis=context.grid_axis, which=context.grid_which, color=context.grid_color, linestyle=context.grid_line_style, linewidth=context.grid_line_width, alpha=context.grid_alpha)
        else:
            axis.grid(False, axis="both", which="both")

    def _style_colorbar(self, colorbar, context):
        axis = colorbar.ax
        axis.set_facecolor(context.axes_background)
        colorbar.outline.set_edgecolor(context.colorbar_outline_color)
        colorbar.outline.set_linewidth(context.colorbar_outline_width)
        axis.tick_params(axis="both", which="major", colors=context.tick_color, labelcolor=context.tick_color, labelsize=context.colorbar_tick_size, length=context.major_tick_length, width=context.major_tick_width)
        axis.tick_params(axis="both", which="minor", colors=context.tick_color, labelcolor=context.tick_color, length=context.minor_tick_length, width=context.minor_tick_width)
        for spine in axis.spines.values():
            spine.set_color(context.colorbar_outline_color)
            spine.set_linewidth(context.colorbar_outline_width)
        for label in (axis.xaxis.label, axis.yaxis.label):
            self._style_text(label, context.axis_label_color, context.colorbar_label_size, context.font_family, context.axis_label_weight)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            self._style_text(label, context.tick_color, context.colorbar_tick_size, context.font_family)
        for offset in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            self._style_text(offset, context.tick_color, context.colorbar_tick_size, context.font_family)

    def _style_text(self, artist, color, size, family, weight=None):
        artist.set_color(color)
        artist.set_fontsize(size)
        if family:
            artist.set_fontfamily(family)
        if weight is not None:
            artist.set_fontweight(weight)

    def _set_font_family(self, artist, family):
        if family:
            artist.set_fontfamily(family)
