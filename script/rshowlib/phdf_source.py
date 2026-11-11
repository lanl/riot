"""PHDF discovery and field selection without GUI dependencies."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .errors import DumpOpenError, InvalidSelectionError, NoPlottableVariablesError, PhdfDependencyError, UnknownVariableError
from .models import DumpInfo, FieldSelection, PlotDimension, VariableInfo


class PhdfReader(Protocol):
    Variables: Sequence[object]
    NumBlocks: int
    x: object
    y: object
    Time: object
    fid: object

    def Get(self, name: str, flatten: bool = False) -> object: ...


PhdfReaderFactory = Callable[[Path], PhdfReader]


def phdf_series_key(path: Path) -> str | None:
    """Return the Parthenon output-series stem for a recognized PHDF frame."""
    path = Path(path)
    if path.suffix.lower() != ".phdf":
        return None
    series, separator, frame = path.stem.rpartition(".")
    if not separator or not series or not frame:
        return None
    if not (frame.isdigit() or frame in {"final", "now"}):
        return None
    return series


def neighboring_dumps(path: Path) -> tuple[Path, ...]:
    """Discover sorted PHDF frames belonging to the same output series."""
    resolved = Path(path).expanduser().resolve()
    key = phdf_series_key(resolved)
    if key is None:
        return (resolved,)
    candidates = {resolved}
    candidates.update(
        candidate.resolve()
        for candidate in resolved.parent.glob("*.phdf")
        if phdf_series_key(candidate) == key
    )
    return tuple(sorted(candidates))


def default_reader_factory(path: Path) -> PhdfReader:
    try:
        try:
            import phdf
        except ImportError:
            from parthenon_tools import phdf
    except ImportError as exc:
        raise PhdfDependencyError(
            "Could not import the PHDF reader. Add Parthenon's scripts/python/packages "
            "to PYTHONPATH, or install parthenon_tools."
        ) from exc
    return phdf.phdf(str(path))


def plottable_variables(reader: PhdfReader) -> tuple[str, ...]:
    variables = []
    for raw_name in reader.Variables:
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        try:
            shape = getattr(reader.fid[name], "shape", None)
            if shape is None or len(shape) < 3 or int(shape[0]) != int(reader.NumBlocks):
                continue
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        variables.append(name)
    return tuple(sorted(variables))


class DumpDataset:
    def __init__(self, path: Path, reader_factory: PhdfReaderFactory = default_reader_factory):
        self.path = Path(path).expanduser().resolve()
        self._reader = None
        try:
            self._reader = reader_factory(self.path)
            names = plottable_variables(self._reader)
            if not names:
                raise NoPlottableVariablesError(f"No plottable fields found in {self.path.name}")
            self._dimension = self._infer_dimension()
            self._variables = tuple(self._make_variable_info(name) for name in names)
        except (PhdfDependencyError, NoPlottableVariablesError):
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise DumpOpenError(f"Could not open PHDF dump {self.path}: {exc}") from exc

    def _infer_dimension(self) -> PlotDimension:
        y = np.asarray(self._reader.y)
        if y.ndim < 2 or y.shape[-1] == 1:
            return PlotDimension.ONE_D
        return PlotDimension.TWO_D

    def _make_variable_info(self, name: str) -> VariableInfo:
        shape = tuple(int(value) for value in np.asarray(self._reader.Get(name, flatten=False)).shape)
        selectable_count = len(shape) - 1 - self._dimension.value
        return VariableInfo(name, shape, shape[1 : 1 + max(selectable_count, 0)])

    @property
    def info(self) -> DumpInfo:
        time = getattr(self._reader, "Time", None)
        try:
            time = None if time is None else float(time)
        except (TypeError, ValueError):
            time = None
        return DumpInfo(
            self.path,
            neighboring_dumps(self.path),
            self._variables,
            self._dimension,
            time,
        )

    @property
    def x_coordinates(self) -> np.ndarray:
        return np.asarray(self._reader.x)

    @property
    def y_coordinates(self) -> np.ndarray:
        return np.asarray(self._reader.y)

    def variable_info(self, name: str) -> VariableInfo:
        try:
            return next(info for info in self._variables if info.name == name)
        except StopIteration as exc:
            raise UnknownVariableError(f"Variable '{name}' is absent from {self.path.name}") from exc

    def select_blocks(self, selection: FieldSelection) -> tuple[np.ndarray, ...]:
        info = self.variable_info(selection.variable)
        if len(selection.indices) != len(info.selectable_shape):
            raise InvalidSelectionError(
                f"{selection.variable} needs {len(info.selectable_shape)} selector index/indices; "
                f"received {len(selection.indices)}"
            )
        for axis, (index, size) in enumerate(zip(selection.indices, info.selectable_shape)):
            if index < 0 or index >= size:
                raise InvalidSelectionError(
                    f"Selector {axis} for {selection.variable} is {index}; valid range is 0..{size - 1}"
                )
        try:
            field = np.asarray(self._reader.Get(selection.variable, flatten=False))
            spatial = (slice(None),) * self._dimension.value
            blocks = tuple(
                np.asarray(field[(block,) + selection.indices + spatial])
                for block in range(field.shape[0])
            )
        except InvalidSelectionError:
            raise
        except Exception as exc:
            raise InvalidSelectionError(f"Could not select {selection.variable}: {exc}") from exc
        if len(blocks) != int(self._reader.NumBlocks) or any(block.ndim != self._dimension.value for block in blocks):
            raise InvalidSelectionError(f"Selected data for {selection.variable} is not {self._dimension.value}-D")
        return blocks

    def close(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None and hasattr(reader, "fid"):
            try:
                reader.fid.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
