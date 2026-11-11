"""Matplotlib rendering onto a caller-owned figure."""

from dataclasses import dataclass

import matplotlib.colors as mcolors
import numpy as np
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .errors import InvalidRangeError, RenderError
from .models import PlotDimension, PlotRequest, RenderResult
from .ranges import coordinate_edges, finite_range, line_y_limits, panel_grid, validate_limits


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")


@dataclass
class _Context:
    font_size: float
    line_width: float


class RshowRenderer:
    """Clear and render a request onto a supplied Matplotlib Figure."""

    def render(self, figure, dataset, request: PlotRequest, canvas_size: tuple[int, int] | None = None) -> RenderResult:
        try:
            context = self._context(request, canvas_size)
            specs = [(request.primary, dataset.select_blocks(request.primary), True)]
            specs.extend((selection, dataset.select_blocks(selection), False) for selection in request.additional_panels)
            figure.clear()
            title = dataset.path.name
            if dataset.info.time is not None:
                title += f"    t = {dataset.info.time:.3e} s"
            if dataset.info.dimension is PlotDimension.ONE_D:
                self._render_1d(figure, dataset, request, specs, title, context)
            else:
                self._render_2d(figure, dataset, request, specs, title, context)
            return RenderResult(len(specs), dataset.info.dimension, title)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(str(exc)) from exc

    def _context(self, request, canvas_size):
        base = float(request.style.base_font_size)
        if not np.isfinite(base) or base <= 0:
            raise InvalidRangeError("Base font size must be positive")
        scale = 1.0
        if request.style.auto_scale_style and canvas_size:
            scale = float(np.clip(np.sqrt(max(canvas_size[0], 1) * max(canvas_size[1], 1) / (700 * 650)), 0.85, 1.6))
        return _Context(base * scale, 1.1 * scale)

    def _grid(self, figure, count, request, is_1d):
        width, height = figure.get_size_inches()
        rows, columns = panel_grid(count, request.style.panel_layout, width / max(height, 0.1))
        return rows, columns, figure.add_gridspec(
            rows, columns, left=0.11 if is_1d else 0.09, right=0.96 if is_1d else 0.87,
            bottom=0.08 if is_1d else 0.07, top=0.91, hspace=0.14 if is_1d else 0.16,
            wspace=0.28 if is_1d else 0.40,
        )

    def _render_1d(self, figure, dataset, request, specs, title, context):
        rows, columns, grid = self._grid(figure, len(specs), request, True)
        axes = []
        for panel_index, (selection, blocks, primary) in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axis = figure.add_subplot(grid[row, column], sharex=axes[0] if axes else None)
            axes.append(axis)
            block_series = []
            data_line_width = max(1.4, 1.4 * context.line_width)
            for block_index, data in enumerate(blocks):
                x = np.asarray(dataset.x_coordinates[block_index, :])
                if data.shape != x.shape:
                    raise RenderError(f"{selection.variable}, block {block_index}: data shape {data.shape} does not match x-coordinate shape {x.shape}")
                edges = coordinate_edges(x)
                block_series.append((edges[0], edges[-1], edges[1] - edges[0], edges[-1] - edges[-2], x, data))
                axis.plot(x, data, color=COLORS[0], linewidth=data_line_width)
                if request.style.show_block_outlines:
                    for edge in (edges[0], edges[-1]):
                        axis.axvline(edge, color="black", linewidth=max(0.45, 0.45 * context.line_width))
            block_series.sort(key=lambda series: series[0])
            for left, right in zip(block_series, block_series[1:]):
                scale = max(abs(left[1]), abs(right[0]), 1.0)
                boundary_tolerance = max(
                    32.0 * np.finfo(float).eps * scale,
                    0.05 * min(abs(left[3]), abs(right[2])),
                )
                if abs(left[1] - right[0]) <= boundary_tolerance:
                    axis.plot(
                        [left[4][-1], right[4][0]],
                        [left[5][-1], right[5][0]],
                        color=COLORS[0],
                        linewidth=data_line_width,
                    )
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
        self._finish(figure, axes, rows, columns, title, context)

    def _norm(self, blocks, request, primary):
        low, high = finite_range(blocks, request.style.logarithmic, request.style.log_floor)
        if primary:
            low = low if request.limits.value_min is None else request.limits.value_min
            high = high if request.limits.value_max is None else request.limits.value_max
        if low == high:
            if request.style.logarithmic:
                low, high = low / 1.1, high * 1.1
            else:
                delta = max(abs(low) * 1e-6, 1e-6)
                low, high = low - delta, high + delta
        low = max(low, request.style.log_floor) if request.style.logarithmic else low
        validate_limits(low, high, request.style.logarithmic)
        norm_type = mcolors.LogNorm if request.style.logarithmic else mcolors.Normalize
        return norm_type(vmin=low, vmax=high)

    def _render_2d(self, figure, dataset, request, specs, title, context):
        rows, columns, grid = self._grid(figure, len(specs), request, False)
        axes = []
        for panel_index, (selection, blocks, primary) in enumerate(specs):
            row, column = divmod(panel_index, columns)
            axis = figure.add_subplot(grid[row, column], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
            axes.append(axis)
            norm = self._norm(blocks, request, primary)
            mesh = None
            for block_index, data in enumerate(blocks):
                x_edges = coordinate_edges(dataset.x_coordinates[block_index, :])
                y_edges = coordinate_edges(dataset.y_coordinates[block_index, :])
                expected = (len(y_edges) - 1, len(x_edges) - 1)
                if data.shape != expected:
                    raise RenderError(f"{selection.variable}, block {block_index}: data shape {data.shape} does not match coordinate shape {expected}")
                mesh = axis.pcolormesh(x_edges, y_edges, data, norm=norm, cmap=request.style.colormap, shading="flat", rasterized=True)
                if request.style.show_block_outlines:
                    axis.plot([x_edges[0], x_edges[-1], x_edges[-1], x_edges[0], x_edges[0]], [y_edges[0], y_edges[0], y_edges[-1], y_edges[-1], y_edges[0]], color="black", linewidth=max(0.45, 0.45 * context.line_width))
            divider = make_axes_locatable(axis)
            colorbar = figure.colorbar(mesh, cax=divider.append_axes("right", size="4%", pad="3%"))
            colorbar.set_label(selection.variable)
            if column == 0:
                axis.set_ylabel("y")
            else:
                axis.tick_params(labelleft=False)
            axis.set_aspect("equal" if request.style.equal_aspect else "auto")
            if row < rows - 1:
                axis.tick_params(labelbottom=False)
        self._apply_shared_limits(axes[0], request, True)
        self._finish(figure, axes, rows, columns, title, context)

    def _apply_shared_limits(self, axis, request, include_y):
        limits = request.limits
        if limits.x_min is not None or limits.x_max is not None:
            axis.set_xlim(left=limits.x_min, right=limits.x_max)
        if include_y and (limits.y_min is not None or limits.y_max is not None):
            axis.set_ylim(bottom=limits.y_min, top=limits.y_max)

    def _finish(self, figure, axes, rows, columns, title, context):
        for index, axis in enumerate(axes):
            row, _ = divmod(index, columns)
            if row == rows - 1:
                axis.set_xlabel("x")
        figure.suptitle(title, y=0.985, fontsize=context.font_size)
        for axis in figure.axes:
            axis.set_prop_cycle(cycler(color=COLORS))
            axis.tick_params(axis="both", which="major", labelsize=max(context.font_size - 1, 1), length=0.42 * context.font_size, width=context.line_width)
            axis.tick_params(axis="both", which="minor", length=0.25 * context.font_size, width=0.72 * context.line_width)
            axis.xaxis.label.set_fontsize(context.font_size)
            axis.yaxis.label.set_fontsize(context.font_size)
            for spine in axis.spines.values():
                spine.set_linewidth(context.line_width)
