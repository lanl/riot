"""Backend-neutral request and result models for rshow."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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
class PlotStyle:
    colormap: str = "viridis"
    logarithmic: bool = False
    log_floor: float = 1.0e-30
    show_block_outlines: bool = False
    equal_aspect: bool = True
    auto_scale_style: bool = True
    base_font_size: float = 14.0
    panel_layout: PanelLayout = PanelLayout.AUTO


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
