"""Backend-neutral request and result models for rshow."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


DEFAULT_LINE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


class PlotDimension(Enum):
    ONE_D = 1
    TWO_D = 2


class PanelLayout(Enum):
    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True)
class FieldSelection:
    variable: str
    indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class AxisLimits:
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    value_min: float | None = None
    value_max: float | None = None


@dataclass(frozen=True)
class PlotColors:
    """Colors used for figure, axes, data, grids, and colorbars."""

    figure_background: str = "white"
    axes_background: str = "white"
    text: str = "black"
    title: str = "black"
    axis_label: str = "black"
    tick: str = "black"
    spine: str = "black"
    grid: str = "#b0b0b0"
    block_outline: str = "black"
    colorbar_outline: str = "black"
    line_palette: tuple[str, ...] = DEFAULT_LINE_COLORS
    colormap_bad: str | None = None
    colormap_under: str | None = None
    colormap_over: str | None = None


@dataclass(frozen=True)
class PlotTypography:
    """Font families, sizes, and weights for renderer-created text."""

    font_family: tuple[str, ...] = ()
    title_size: float | None = None
    axis_label_size: float | None = None
    tick_label_size: float | None = None
    colorbar_label_size: float | None = None
    colorbar_tick_size: float | None = None
    title_weight: str = "normal"
    axis_label_weight: str = "normal"


@dataclass(frozen=True)
class PlotStrokes:
    """Line widths and tick metrics; ``None`` retains legacy formulas."""

    data_line_width: float | None = None
    block_outline_width: float | None = None
    spine_width: float | None = None
    colorbar_outline_width: float | None = None
    major_tick_width: float | None = None
    minor_tick_width: float | None = None
    major_tick_length: float | None = None
    minor_tick_length: float | None = None


@dataclass(frozen=True)
class GridStyle:
    """Grid visibility and line presentation."""

    visible: bool = False
    axis: str = "both"
    which: str = "major"
    line_style: str = "-"
    line_width: float = 0.8
    alpha: float = 1.0


@dataclass(frozen=True)
class PanelMargins:
    """Normalized subplot margins and inter-panel spacing."""

    left: float
    right: float
    bottom: float
    top: float
    horizontal_space: float
    vertical_space: float


@dataclass(frozen=True)
class PlotLayoutStyle:
    """Dimension-specific subplot, title, and colorbar layout."""

    one_dimensional: PanelMargins = field(
        default_factory=lambda: PanelMargins(0.11, 0.96, 0.08, 0.91, 0.28, 0.14)
    )
    two_dimensional: PanelMargins = field(
        default_factory=lambda: PanelMargins(0.09, 0.87, 0.07, 0.91, 0.40, 0.16)
    )
    title_y: float = 0.985
    colorbar_size: float = 0.04
    colorbar_padding: float = 0.03


@dataclass(frozen=True)
class PlotAppearance:
    """Complete backend-neutral declarative plot appearance."""

    colors: PlotColors = field(default_factory=PlotColors)
    typography: PlotTypography = field(default_factory=PlotTypography)
    strokes: PlotStrokes = field(default_factory=PlotStrokes)
    grid: GridStyle = field(default_factory=GridStyle)
    layout: PlotLayoutStyle = field(default_factory=PlotLayoutStyle)
    rasterize_mesh: bool = True


@dataclass(frozen=True)
class PlotStyle:
    colormap: str = "viridis"
    logarithmic: bool = False
    log_floor: float = 1.0e-30
    show_block_outlines: bool = False
    equal_aspect: bool = True
    auto_scale_style: bool = True
    base_font_size: float = 14.0
    panel_layout: PanelLayout = PanelLayout.AUTO
    appearance: PlotAppearance = field(default_factory=PlotAppearance)


@dataclass(frozen=True)
class PlotRequest:
    primary: FieldSelection
    additional_panels: tuple[FieldSelection, ...] = ()
    limits: AxisLimits = field(default_factory=AxisLimits)
    style: PlotStyle = field(default_factory=PlotStyle)


@dataclass(frozen=True)
class VariableInfo:
    name: str
    shape: tuple[int, ...]
    selectable_shape: tuple[int, ...]


@dataclass(frozen=True)
class DumpInfo:
    path: Path
    neighboring_dumps: tuple[Path, ...]
    variables: tuple[VariableInfo, ...]
    dimension: PlotDimension
    time: float | None


@dataclass(frozen=True)
class SuggestedRanges:
    limits: AxisLimits


@dataclass(frozen=True)
class RenderResult:
    panel_count: int
    dimension: PlotDimension
    title: str
