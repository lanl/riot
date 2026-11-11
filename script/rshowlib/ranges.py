"""Pure selection, range, and layout helpers."""

from collections.abc import Iterable, Sequence
import math

import numpy as np

from .errors import InvalidRangeError, InvalidSelectionError
from .models import AxisLimits, FieldSelection, PanelLayout, PlotDimension, SuggestedRanges


def coordinate_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float).squeeze()
    if centers.ndim != 1 or centers.size == 0:
        raise InvalidRangeError("Coordinates must be a nonempty 1-D array")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - (mids[0] - centers[0])], mids, [centers[-1] + (centers[-1] - mids[-1])]))


def parse_indices(text: str, expected_count: int) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise InvalidSelectionError("Selector indices must be integers") from exc
    if not values and expected_count:
        values = (0,) * expected_count
    if len(values) != expected_count:
        raise InvalidSelectionError(
            f"This field needs {expected_count} selector index/indices before its spatial axes; "
            f"enter {expected_count} comma-separated values."
        )
    return values


def format_indices(indices: Sequence[int]) -> str:
    return ",".join(str(index) for index in indices)


def finite_range(blocks: Iterable[np.ndarray], positive: bool = False, floor: float = 0.0) -> tuple[float, float]:
    arrays = [np.asarray(block).ravel() for block in blocks]
    if not arrays:
        raise InvalidRangeError("No field blocks are available")
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > max(float(floor), 0.0)]
    if values.size == 0:
        qualifier = "positive " if positive else "finite "
        raise InvalidRangeError(f"The selected field contains no {qualifier}values")
    low, high = float(values.min()), float(values.max())
    if low == high:
        if positive:
            low, high = low / 1.1, high * 1.1
        else:
            delta = max(abs(low) * 1.0e-6, 1.0e-6)
            low, high = low - delta, high + delta
    return low, high


def line_y_limits(blocks: Iterable[np.ndarray], positive: bool, floor: float) -> tuple[float, float]:
    low, high = finite_range(blocks, positive, floor)
    if positive:
        padding = (high / low) ** 0.03
        return low / padding, high * padding
    padding = 0.03 * (high - low)
    return low - padding, high + padding


def validate_limits(low: float, high: float, logarithmic: bool = False) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high):
        raise InvalidRangeError("Plot limits must be finite")
    if low >= high:
        raise InvalidRangeError("The minimum plot limit must be less than the maximum")
    if logarithmic and low <= 0:
        raise InvalidRangeError("Logarithmic scaling requires a positive minimum")
    return low, high


def panel_grid(panel_count: int, layout: PanelLayout, aspect: float) -> tuple[int, int]:
    if panel_count < 1:
        raise InvalidRangeError("At least one panel is required")
    if layout is PanelLayout.HORIZONTAL:
        return 1, panel_count
    if layout is PanelLayout.VERTICAL:
        return panel_count, 1
    if panel_count == 2:
        return (1, 2) if aspect >= 1.0 else (2, 1)
    columns = max(1, min(panel_count, math.ceil(math.sqrt(panel_count * max(aspect, 0.1)))))
    rows = math.ceil(panel_count / columns)
    return rows, columns


def suggest_ranges(dataset, selection: FieldSelection, logarithmic: bool, floor: float) -> SuggestedRanges:
    blocks = dataset.select_blocks(selection)
    low, high = finite_range(blocks, logarithmic, floor)
    x = dataset.x_coordinates
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if dataset.info.dimension is PlotDimension.ONE_D:
        y_min, y_max = line_y_limits(blocks, logarithmic, floor)
    else:
        y = dataset.y_coordinates
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    return SuggestedRanges(AxisLimits(x_min, x_max, y_min, y_max, low, high))
