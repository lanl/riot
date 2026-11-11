"""Reusable, backend-neutral implementation of the rshow PHDF viewer."""

RSHOW_API_VERSION = 1
RSHOW_CAPABILITIES = frozenset({"plot_appearance_v1", "render_customizer_v1"})

from .errors import (
    DumpOpenError, ExportCancelled, ExportDependencyError, InvalidRangeError,
    InvalidSelectionError, InvalidStyleError, NoPlottableVariablesError, PhdfDependencyError,
    RenderError, RshowError, UnknownVariableError,
)
from .exporting import export_movie, save_figure
from .models import (
    DEFAULT_LINE_COLORS, AxisLimits, DumpInfo, FieldSelection, GridStyle,
    PanelLayout, PanelMargins, PlotAppearance, PlotColors, PlotDimension,
    PlotLayoutStyle, PlotRequest, PlotStrokes, PlotStyle, PlotTypography,
    RenderResult, SuggestedRanges, VariableInfo,
)
from .phdf_source import (
    DumpDataset, default_reader_factory, neighboring_dumps, phdf_series_key,
)
from .ranges import coordinate_edges, finite_range, format_indices, panel_grid, parse_indices, suggest_ranges
from .rendering import RenderArtifacts, RshowRenderer
from .session import RshowSession

__all__ = [
    "RSHOW_API_VERSION", "RSHOW_CAPABILITIES", "DEFAULT_LINE_COLORS", "AxisLimits",
    "DumpDataset", "DumpInfo", "FieldSelection", "GridStyle",
    "DumpOpenError", "ExportCancelled", "ExportDependencyError", "InvalidRangeError",
    "InvalidSelectionError", "InvalidStyleError", "NoPlottableVariablesError",
    "PhdfDependencyError", "PanelLayout", "PanelMargins", "PlotAppearance",
    "PlotColors", "PlotDimension", "PlotLayoutStyle", "PlotRequest", "PlotStrokes",
    "PlotStyle", "PlotTypography", "RenderArtifacts", "RenderResult", "RenderError",
    "RshowError", "RshowRenderer", "RshowSession", "SuggestedRanges",
    "UnknownVariableError", "VariableInfo",
    "coordinate_edges", "default_reader_factory", "export_movie", "finite_range",
    "format_indices", "neighboring_dumps", "panel_grid", "parse_indices",
    "phdf_series_key", "save_figure", "suggest_ranges",
]
