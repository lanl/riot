"""Typed errors raised by the reusable rshow implementation."""


class RshowError(Exception):
    """Base class for errors suitable for presentation by a frontend."""


class PhdfDependencyError(RshowError):
    pass


class DumpOpenError(RshowError):
    pass


class NoPlottableVariablesError(RshowError):
    pass


class UnknownVariableError(RshowError):
    pass


class InvalidSelectionError(RshowError):
    pass


class InvalidRangeError(RshowError):
    pass


class RenderError(RshowError):
    pass


class ExportDependencyError(RshowError):
    pass


class ExportCancelled(RshowError):
    pass
