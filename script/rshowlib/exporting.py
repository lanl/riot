"""Backend-neutral still-image and animation export."""

import math
from pathlib import Path

import matplotlib.animation as animation

from .errors import ExportCancelled, ExportDependencyError, RshowError
from .phdf_source import DumpDataset, default_reader_factory


def save_figure(figure, destination: Path) -> None:
    try:
        figure.savefig(Path(destination))
    except Exception as exc:
        raise RshowError(f"Could not save figure to {destination}: {exc}") from exc


def export_movie(paths, destination: Path, fps: float, figure, request, renderer, reader_factory=default_reader_factory, on_progress=None, is_cancelled=None) -> None:
    paths = tuple(Path(path) for path in paths)
    destination = Path(destination)
    if not math.isfinite(fps) or fps <= 0:
        raise ExportDependencyError("Frames per second must be finite and positive")
    if len(paths) < 2:
        raise ExportDependencyError("Movie export requires at least two PHDF dumps")
    suffix = destination.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ExportDependencyError("Movie filename must end in .mp4 or .gif")
    writer_name = "ffmpeg" if suffix == ".mp4" else "pillow"
    if not animation.writers.is_available(writer_name):
        instruction = "install ffmpeg" if suffix == ".mp4" else "install Pillow"
        raise ExportDependencyError(f"{writer_name} is unavailable; {instruction} to export {suffix} files")
    writer = animation.FFMpegWriter(fps=fps, metadata={"artist": "PHDF Plot Viewer"}) if suffix == ".mp4" else animation.PillowWriter(fps=fps)
    try:
        with writer.saving(figure, str(destination), dpi=figure.dpi):
            for number, path in enumerate(paths, start=1):
                if is_cancelled is not None and is_cancelled():
                    raise ExportCancelled("Movie export was cancelled")
                with DumpDataset(path, reader_factory) as dataset:
                    renderer.render(figure, dataset, request)
                    writer.grab_frame()
                if on_progress is not None:
                    on_progress(number, len(paths), path)
    except (ExportCancelled, RshowError):
        raise
    except Exception as exc:
        raise ExportDependencyError(f"Could not export movie: {exc}") from exc
