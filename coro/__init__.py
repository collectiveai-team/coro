"""coro — packaged ASR diarization server."""

from importlib.metadata import PackageNotFoundError, metadata

# Single source for the metadata stamped into both published contracts (OpenAPI
# and AsyncAPI ``info``). Everything is read from the installed distribution so
# pyproject.toml stays the only place it is declared; a source tree with no
# install has no distribution metadata, so fall back rather than failing at
# import.
try:
    _metadata = metadata("coro-asr")
    __version__ = _metadata["Version"]
    __summary__ = _metadata["Summary"]
    __license__ = _metadata["License-Expression"]
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0"
    __summary__ = ""
    __license__ = ""

__all__ = ["__license__", "__summary__", "__version__"]
