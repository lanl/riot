"""Toolkit-independent PHDF session lifecycle."""

from pathlib import Path

from .errors import DumpOpenError
from .phdf_source import DumpDataset, PhdfReaderFactory, default_reader_factory


class RshowSession:
    def __init__(self, reader_factory: PhdfReaderFactory = default_reader_factory):
        self._reader_factory = reader_factory
        self._dataset: DumpDataset | None = None

    @property
    def dataset(self) -> DumpDataset | None:
        return self._dataset

    @property
    def dump_info(self):
        return None if self._dataset is None else self._dataset.info

    def open(self, path: Path):
        replacement = DumpDataset(path, self._reader_factory)
        previous, self._dataset = self._dataset, replacement
        if previous is not None:
            previous.close()
        return replacement.info

    def reload(self):
        if self._dataset is None:
            raise DumpOpenError("No dump is open")
        return self.open(self._dataset.path)

    def step(self, amount: int):
        if self._dataset is None:
            raise DumpOpenError("No dump is open")
        paths = self._dataset.info.neighboring_dumps
        try:
            current = paths.index(self._dataset.path)
        except ValueError:
            current = 0
        return self.open(paths[max(0, min(len(paths) - 1, current + amount))])

    def close(self):
        dataset, self._dataset = self._dataset, None
        if dataset is not None:
            dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
