"""Reusable, backend-neutral implementation of the rshow PHDF viewer."""

RSHOW_API_VERSION = 1

from .errors import (
    DumpOpenError, ExportCancelled, ExportDependencyError, InvalidRangeError,
    InvalidSelectionError, NoPlottableVariablesError, PhdfDependencyError,
    RenderError, RshowError, UnknownVariableError,
)
from .exporting import export_movie, save_figure
from .models import AxisLimits, DumpInfo, FieldSelection, PanelLayout, PlotDimension, PlotRequest, PlotStyle, RenderResult, SuggestedRanges, VariableInfo
from .phdf_source import DumpDataset, default_reader_factory
from .ranges import coordinate_edges, finite_range, format_indices, panel_grid, parse_indices, suggest_ranges
from .rendering import RshowRenderer
from .session import RshowSession

__all__ = [
    "RSHOW_API_VERSION", "AxisLimits", "DumpDataset", "DumpInfo", "FieldSelection",
    "DumpOpenError", "ExportCancelled", "ExportDependencyError", "InvalidRangeError",
    "InvalidSelectionError", "NoPlottableVariablesError", "PhdfDependencyError",
    "PanelLayout", "PlotDimension", "PlotRequest", "PlotStyle", "RenderResult",
    "RenderError", "RshowError", "RshowRenderer", "RshowSession", "SuggestedRanges",
    "UnknownVariableError", "VariableInfo",
    "coordinate_edges", "default_reader_factory", "export_movie", "finite_range",
    "format_indices", "panel_grid", "parse_indices", "save_figure", "suggest_ranges",
]
