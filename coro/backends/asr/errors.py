"""Typed ASR adapter errors that are not concurrency-specific.

``AsrCapacityError`` (in :mod:`coro.backends.asr.concurrency`) set the pattern
this package follows: a backend raises a typed error carrying structured
attributes, and every request surface translates it the same way instead of
reporting a server fault. This module holds the same-pattern errors that do
not belong to the Adapter Concurrency Policy, so that module stays
single-concern.
"""

from __future__ import annotations

from collections.abc import Iterable


# MARK: Unsupported Language
class AsrUnsupportedLanguageError(ValueError):
    """Raised when a request's language resolves to none the model supports.

    Carries the requested language and the supported set (derived from the
    loaded vocab by the raising backend, never hardcoded), which every request
    surface renders into a 400 naming both -- the forced-language analogue of
    how :class:`~coro.backends.asr.concurrency.AsrCapacityError` reaches the
    API layer.
    """

    def __init__(self, language: str, *, supported_languages: Iterable[str]) -> None:
        supported = sorted(supported_languages)
        supported_text = ", ".join(supported) if supported else "<none>"
        message = (
            f"Language {language!r} is not supported by this model (supported: {supported_text})."
        )
        super().__init__(message)
        self.message = message
        self.language = language
        self.supported_languages = tuple(supported)
