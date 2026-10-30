#!/usr/bin/env python3
# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
"""Plot a two-dimensional mesh field from a Parthenon openPMD/BP output.

For example, from the Parthenon build directory::

    python3 ../scripts/python/plot_openpmd_2d.py \
        'my_problem.out0.%05T.bp' --field c.c.mat.volume_fraction_0

The input may also be one concrete output file, e.g. ``my_problem.out0.00000.bp``;
the ``%05T`` series pattern is inferred.  Use ``--list-fields`` to inspect the
record and component names stored in an output.

Parthenon stores only populated chunks for AMR and sparse fields.  This script reads
those chunks into a NaN-filled array rather than loading unpopulated portions of the
record.  With the default ``--amr composite`` it overlays all available AMR levels on a
uniform grid at the finest available resolution.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

try:
    import openpmd_api as io
except ModuleNotFoundError:  # pragma: no cover - depends on user environment
    io = None


LEVEL_RE = re.compile(r"^(?P<base>.+)_lvl(?P<level>\d+)$")
FILE_RE = re.compile(r"^(?P<prefix>.*\.)(?P<iteration>\d+)(?P<suffix>\.bp)$")


@dataclass
class FieldData:
    """One populated component and its openPMD grid metadata."""

    record: str
    component: str
    level: int
    values: np.ndarray
    spacing: np.ndarray  # Stored order: y, x
    offset: np.ndarray  # Stored order: y, x
    position: np.ndarray  # Stored order: y, x

    @property
    def left_edges(self) -> np.ndarray:
        return self.offset + (self.position - 0.5) * self.spacing

    @property
    def right_edges(self) -> np.ndarray:
        return self.left_edges + np.asarray(self.values.shape) * self.spacing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "series",
        help="openPMD series pattern (normally '*.%%05T.bp') or one '*.bp' output",
    )
    parser.add_argument(
        "--field",
        default="c.c.mat.volume_fraction_0",
        help="Parthenon field/base record to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--component",
        help="openPMD record component; inferred when the record has one component",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="output iteration to plot (default: the last available iteration)",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="only use this AMR level (also disables AMR compositing)",
    )
    parser.add_argument(
        "--amr",
        choices=("composite", "finest"),
        default="composite",
        help="how to handle multiple AMR levels (default: %(default)s)",
    )
    parser.add_argument(
        "--reader",
        choices=("auto", "openpmd", "bpls"),
        default="auto",
        help=(
            "reader to use (default: %(default)s); bpls works around malformed "
            "non-mesh attributes in some ADIOS2 outputs"
        ),
    )
    parser.add_argument(
        "--bpls",
        help="path to the ADIOS2 bpls executable (used by --reader bpls or fallback)",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="print mesh records/components in the selected iteration and exit",
    )
    parser.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    parser.add_argument("--vmin", type=float, help="lower color limit")
    parser.add_argument("--vmax", type=float, help="upper color limit")
    parser.add_argument("--log", action="store_true", help="use logarithmic colors")
    parser.add_argument("--title", help="custom plot title")
    parser.add_argument(
        "--output", type=Path, default=Path("openpmd_2d.png"), help="output PNG path"
    )
    parser.add_argument("--show", action="store_true", help="also show an interactive window")
    return parser.parse_args()


def normalize_series_path(path: str) -> str:
    """Turn a concrete Parthenon BP file name into its openPMD series pattern."""
    if "%T" in path or re.search(r"%0\d+T", path):
        return path
    match = FILE_RE.match(path)
    if match is None:
        raise ValueError(
            "Expected an openPMD pattern containing %T (usually %05T), or a file name "
            "ending in .NNNNN.bp."
        )
    width = len(match.group("iteration"))
    return f"{match.group('prefix')}%0{width}T{match.group('suffix')}"


def bpls_executable(args: argparse.Namespace) -> str:
    """Find the ADIOS2 command-line reader used for the compatibility fallback."""
    executable = args.bpls or shutil.which("bpls")
    if executable is None:
        raise ValueError(
            "openPMD could not read this output and ADIOS2's `bpls` was not found. "
            "Activate the ADIOS2 environment or pass --bpls /path/to/bpls."
        )
    return executable


def run_bpls(args: argparse.Namespace, *arguments: str) -> str:
    """Run bpls without invoking a shell, so file and mesh names remain literal."""
    try:
        result = subprocess.run(
            [bpls_executable(args), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ValueError(f"Could not run bpls: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise ValueError(f"bpls could not read the output: {detail}") from exc
    return result.stdout


def bpls_iteration_path(series_path: str, requested: int | None) -> tuple[Path, int]:
    """Resolve an openPMD pattern to one file, since bpls reads one BP file at a time."""
    if "%T" not in series_path and not re.search(r"%0\d+T", series_path):
        path = Path(series_path)
        match = FILE_RE.match(path.name)
        return path, int(match.group("iteration")) if match else 0

    token = re.search(r"%0(?P<width>\d+)T|%T", series_path)
    assert token is not None  # checked above
    width = int(token.group("width") or 0)
    before, after = series_path[: token.start()], series_path[token.end() :]
    directory = Path(before).parent
    prefix = Path(before).name
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"{prefix}*{after}"):
        match = re.fullmatch(
            re.escape(prefix) + r"(\d+)" + re.escape(after), path.name
        )
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(f"No BP files match '{series_path}'.")
    iteration = max(number for number, _ in candidates) if requested is None else requested
    for number, path in candidates:
        if number == iteration:
            return path, number
    expected = f"{prefix}{iteration:0{width}d}{after}" if width else f"{prefix}{iteration}{after}"
    raise ValueError(f"Iteration {iteration} is unavailable; expected '{directory / expected}'.")


def choose_iteration(series: io.Series, requested: int | None):
    available = sorted(series.iterations)
    if not available:
        raise ValueError("The openPMD series contains no iterations.")
    iteration = available[-1] if requested is None else requested
    if iteration not in series.iterations:
        raise ValueError(f"Iteration {iteration} is unavailable; found {available}.")
    return iteration, series.iterations[iteration]


def record_level(name: str) -> int:
    match = LEVEL_RE.match(name)
    return int(match.group("level")) if match else 0


def select_records(meshes, requested: str, level: int | None) -> list[str]:
    """Resolve a Parthenon base field to its level-specific openPMD records."""
    exact = [name for name in meshes if name == requested]
    with_levels = [
        name
        for name in meshes
        if (match := LEVEL_RE.match(name)) and match.group("base") == requested
    ]
    records = exact + with_levels
    if level is not None:
        records = [name for name in records if record_level(name) == level]
    if not records:
        available = ", ".join(sorted(meshes))
        raise ValueError(
            f"No mesh record for field '{requested}'. Available records: {available}"
        )
    return sorted(set(records), key=record_level)


def component_label(component) -> str:
    """Return a terminal- and command-line-friendly component name."""
    return str(component).strip()


def select_component(mesh, record_name: str, requested: str | None):
    components = list(mesh)
    if requested is not None:
        matching = [name for name in components if component_label(name) == requested]
        if not matching:
            raise ValueError(
                f"Component '{requested}' is not in record '{record_name}'; "
                f"available: {', '.join(component_label(name) for name in components)}"
            )
        return matching[0]
    if len(components) == 1:
        return components[0]
    scalar = [name for name in components if component_label(name).lower() == "scalar"]
    if len(scalar) == 1:
        return scalar[0]
    raise ValueError(
        f"Record '{record_name}' has multiple components "
        f"({', '.join(component_label(name) for name in components)}). "
        "Select one with --component."
    )


def load_populated_chunks(component, series: io.Series) -> np.ndarray:
    """Load only chunks that exist on disk, leaving absent AMR/sparse cells as NaN."""
    shape = tuple(component.shape)
    if len(shape) != 2:
        raise ValueError(
            f"The selected component has shape {shape}; this tool plots only 2-D outputs. "
            "Use a Parthenon slice output for a 3-D simulation."
        )
    data = np.full(shape, np.nan, dtype=float)
    chunks = list(component.available_chunks())
    if not chunks:
        raise ValueError("The selected component has no populated chunks.")
    for chunk in chunks:
        stop = [offset + extent for offset, extent in zip(chunk.offset, chunk.extent)]
        selection = tuple(slice(start, end) for start, end in zip(chunk.offset, stop))
        data[selection] = component[selection]
        series.flush()
    return data


def read_fields(series: io.Series, iteration, args: argparse.Namespace) -> list[FieldData]:
    records = select_records(iteration.meshes, args.field, args.level)
    fields = []
    for record in records:
        mesh = iteration.meshes[record]
        component_name = select_component(mesh, record, args.component)
        component = mesh[component_name]
        values = load_populated_chunks(component, series)
        spacing = np.asarray(mesh.grid_spacing, dtype=float)
        offset = np.asarray(mesh.grid_global_offset, dtype=float)
        position = np.asarray(component.position, dtype=float)
        if not (len(spacing) == len(offset) == len(position) == 2):
            raise ValueError(
                f"Record '{record}' does not describe a 2-D Cartesian grid."
            )
        fields.append(
            FieldData(
                record,
                component_label(component_name),
                record_level(record),
                values,
                spacing,
                offset,
                position,
            )
        )
    return fields


BPLS_MESH_RE = re.compile(
    r"^\s*\S+\s+(?P<path>/data/(?P<iteration>\d+)/meshes/"
    r"(?P<record>[^/\s]+)(?:/(?P<component>[^/\s]+))?)\s+"
    r"\{(?P<shape>[^}]*)\}",
    re.MULTILINE,
)
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
    re.IGNORECASE,
)
BPLS_BLOCK_RE = re.compile(
    r"^\s*block\s+\d+:\s+\[\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*,"
    r"\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*\]$",
    re.MULTILINE,
)


def bpls_mesh_records(
    args: argparse.Namespace, path: Path, iteration_index: int
) -> dict[str, dict[str | None, tuple[int, ...]]]:
    """Return mesh records, their components, and their array shapes from bpls."""
    # bpls's '*' does not cross '/', so list the file and retain only this
    # iteration's mesh variables; this includes record components such as /0.
    listing = run_bpls(args, "-l", str(path))
    records: dict[str, dict[str | None, tuple[int, ...]]] = {}
    for match in BPLS_MESH_RE.finditer(listing):
        records.setdefault(match.group("record"), {})[match.group("component")] = tuple(
            int(value.strip()) for value in match.group("shape").split(",")
        )
    if not records:
        raise ValueError(f"bpls found no mesh records in '{path}'.")
    return records


def bpls_component_label(component: str | None) -> str:
    return "scalar" if component is None else component


def select_bpls_component(
    components: dict[str | None, tuple[int, ...]], record: str, requested: str | None
) -> str | None:
    if requested is not None:
        for component in components:
            if bpls_component_label(component) == requested:
                return component
        available = ", ".join(bpls_component_label(component) for component in components)
        raise ValueError(
            f"Component '{requested}' is not in record '{record}'; available: {available}"
        )
    if len(components) == 1:
        return next(iter(components))
    available = ", ".join(bpls_component_label(component) for component in components)
    raise ValueError(
        f"Record '{record}' has multiple components ({available}). Select one with --component."
    )


def bpls_attribute(metadata: str, mesh_path: str, name: str) -> np.ndarray:
    """Extract a numeric mesh attribute from bpls's stable, human-readable output."""
    pattern = re.compile(
        rf"^\s*\S+\s+{re.escape(mesh_path)}/{re.escape(name)}\s+attr\s+=\s*(?P<value>.+)$",
        re.MULTILINE,
    )
    match = pattern.search(metadata)
    if match is None:
        raise ValueError(f"bpls did not report required attribute '{name}' for '{mesh_path}'.")
    values = np.asarray(NUMBER_RE.findall(match.group("value")), dtype=float)
    if values.size == 0:
        raise ValueError(f"bpls reported a non-numeric '{name}' attribute for '{mesh_path}'.")
    return values


def bpls_attribute_or_default(
    metadata: str, mesh_path: str, name: str, default: np.ndarray
) -> np.ndarray:
    try:
        return bpls_attribute(metadata, mesh_path, name)
    except ValueError:
        return default


def bpls_populated_chunks(args: argparse.Namespace, path: Path, mesh_path: str):
    """Return the extents ADIOS2 actually stored for a sparse mesh record."""
    decomposition = run_bpls(args, "-D", str(path), mesh_path)
    chunks = [
        (
            slice(int(match.group("y0")), int(match.group("y1")) + 1),
            slice(int(match.group("x0")), int(match.group("x1")) + 1),
        )
        for match in BPLS_BLOCK_RE.finditer(decomposition)
    ]
    if not chunks:
        raise ValueError(f"bpls found no populated chunks for '{mesh_path}'.")
    return chunks


def read_fields_bpls(args: argparse.Namespace) -> tuple[int, list[FieldData]]:
    """Read 2-D meshes without parsing unrelated openPMD attributes.

    A few Parthenon restart files contain zero-length application attributes that
    older openPMD-api releases cannot deserialize. ADIOS2 itself can still read
    their mesh arrays, so this deliberately narrow fallback keeps those outputs
    plottable without modifying them.
    """
    path, iteration_index = bpls_iteration_path(args.series, args.iteration)
    meshes = bpls_mesh_records(args, path, iteration_index)
    all_records = select_records(meshes, args.field, None)
    component = select_bpls_component(
        meshes[all_records[0]], all_records[0], args.component
    )
    records = [
        record for record in all_records if args.level is None or record_level(record) == args.level
    ]
    if not records:
        raise ValueError(f"No mesh record for field '{args.field}' at level {args.level}.")

    fields = []
    for record in records:
        mesh_path = f"/data/{iteration_index}/meshes/{record}"
        if component not in meshes[record]:
            raise ValueError(
                f"Component '{bpls_component_label(component)}' is not in record '{record}'."
            )
        component_path = mesh_path if component is None else f"{mesh_path}/{component}"
        metadata = run_bpls(args, "-la", str(path), f"{mesh_path}/*")
        raw = run_bpls(args, "-d", "-y", str(path), component_path)
        raw_values = np.fromstring(raw.partition("\n")[2], sep=" ")
        shape = meshes[record][component]
        if len(shape) != 2:
            raise ValueError(
                f"The selected component has shape {shape}; this tool plots only 2-D outputs."
            )
        if raw_values.size != int(np.prod(shape)):
            raise ValueError(
                f"bpls read {raw_values.size} values for '{record}', expected {int(np.prod(shape))}."
            )
        raw_image = raw_values.reshape(shape)
        values = np.full(shape, np.nan)
        for chunk in bpls_populated_chunks(args, path, component_path):
            values[chunk] = raw_image[chunk]
        fields.append(
            FieldData(
                record=record,
                component=bpls_component_label(component),
                level=record_level(record),
                values=values,
                spacing=bpls_attribute(metadata, mesh_path, "gridSpacing"),
                offset=bpls_attribute(metadata, mesh_path, "gridGlobalOffset"),
                position=bpls_attribute_or_default(
                    metadata, component_path, "position", np.full(2, 0.5)
                ),
            )
        )
    return iteration_index, fields


def print_fields_bpls(args: argparse.Namespace) -> None:
    path, iteration_index = bpls_iteration_path(args.series, args.iteration)
    print(f"Iteration {iteration_index} mesh records:")
    for record, components in sorted(bpls_mesh_records(args, path, iteration_index).items()):
        labels = sorted(bpls_component_label(component) for component in components)
        print(f"  {record}: {', '.join(labels)}")


def composite_amr(fields: list[FieldData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay coarse-to-fine records on a finest-resolution uniform image."""
    finest_spacing = np.min(np.stack([field.spacing for field in fields]), axis=0)
    left = np.min(np.stack([field.left_edges for field in fields]), axis=0)
    right = np.max(np.stack([field.right_edges for field in fields]), axis=0)
    shape_float = (right - left) / finest_spacing
    shape = np.rint(shape_float).astype(int)
    if not np.allclose(shape_float, shape):
        raise ValueError("AMR levels do not align on a common Cartesian grid.")
    image = np.full(tuple(shape), np.nan, dtype=float)

    # Larger cell widths are coarser, so fine populated data overwrites coarse data.
    for field in sorted(fields, key=lambda item: np.prod(item.spacing), reverse=True):
        factors_float = field.spacing / finest_spacing
        factors = np.rint(factors_float).astype(int)
        start_float = (field.left_edges - left) / finest_spacing
        start = np.rint(start_float).astype(int)
        if not np.allclose(factors_float, factors) or not np.allclose(start_float, start):
            raise ValueError("AMR levels do not align on a common Cartesian grid.")
        expanded = np.repeat(np.repeat(field.values, factors[0], axis=0), factors[1], axis=1)
        end = start + np.asarray(expanded.shape)
        target = image[start[0] : end[0], start[1] : end[1]]
        valid = np.isfinite(expanded)
        target[valid] = expanded[valid]
    return image, left, right


def prepare_image(fields: list[FieldData], args: argparse.Namespace):
    if args.level is not None:
        field = fields[0]
        return field.values, field.left_edges, field.right_edges, field
    if args.amr == "finest":
        field = min(fields, key=lambda item: np.prod(item.spacing))
        return field.values, field.left_edges, field.right_edges, field
    image, left, right = composite_amr(fields)
    return image, left, right, max(fields, key=lambda item: item.level)


def print_fields(iteration, iteration_index: int) -> None:
    print(f"Iteration {iteration_index} mesh records:")
    for record_name in sorted(iteration.meshes):
        mesh = iteration.meshes[record_name]
        print(
            f"  {record_name}: "
            f"{', '.join(component_label(component) for component in mesh)}"
        )


def main() -> None:
    args = parse_args()
    series = None
    try:
        use_bpls = args.reader == "bpls"
        if args.reader == "openpmd" and io is None:
            raise ValueError("openpmd_api is not installed.")
        if args.reader == "auto" and io is None:
            use_bpls = True

        if not use_bpls:
            try:
                series_path = normalize_series_path(args.series)
                series = io.Series(series_path, io.Access.read_only)
                iteration_index, iteration = choose_iteration(series, args.iteration)
                iteration.open()
                if args.list_fields:
                    print_fields(iteration, iteration_index)
                    return
                fields = read_fields(series, iteration, args)
            except RuntimeError as exc:
                if args.reader != "auto":
                    raise
                if series is not None:
                    series.close()
                    series = None
                print(
                    f"openPMD reader failed ({exc}); retrying with ADIOS2 bpls.",
                    file=sys.stderr,
                )
                use_bpls = True

        if use_bpls:
            if args.list_fields:
                print_fields_bpls(args)
                return
            iteration_index, fields = read_fields_bpls(args)

        image, left, right, reference = prepare_image(fields, args)
        if not np.isfinite(image).any():
            raise ValueError("No finite values were read from the selected field.")

        kwargs = {"cmap": args.cmap, "origin": "lower", "aspect": "equal"}
        if args.log:
            positive = image[np.isfinite(image) & (image > 0)]
            if positive.size == 0:
                raise ValueError("--log requires at least one positive field value.")
            kwargs["norm"] = LogNorm(
                vmin=args.vmin or positive.min(), vmax=args.vmax or positive.max()
            )
        else:
            kwargs.update(vmin=args.vmin, vmax=args.vmax)

        fig, axis = plt.subplots(constrained_layout=True)
        plot = axis.imshow(
            image,
            extent=(left[1], right[1], left[0], right[0]),
            **kwargs,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        default_title = f"{args.field}, iteration {iteration_index}"
        if len(fields) > 1 and args.level is None:
            default_title += (
                " (AMR composite)" if args.amr == "composite" else " (finest level)"
            )
        axis.set_title(args.title or default_title)
        fig.colorbar(plot, ax=axis, label=args.field)
        fig.savefig(args.output, dpi=200)
        print(
            f"Wrote {args.output} from {reference.record}/{reference.component} "
            f"(iteration {iteration_index})."
        )
        if args.show:
            plt.show()
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    finally:
        if series is not None:
            series.close()


if __name__ == "__main__":
    main()
