"""Per-request ASR Windowing overrides shared by the vendor endpoints.

Both endpoints accept ``coro_window_seconds`` and ``coro_overlap_seconds`` as
extra request parameters. The ``coro_`` prefix keeps them apart from the vendor
contracts: OpenAI and Deepgram clients do not send them, and the server-wide
defaults (``CORO_ASR_WINDOW_SECONDS``, ``CORO_ASR_WINDOW_OVERLAP_SECONDS``)
apply when a request leaves them out.
"""

from __future__ import annotations

from dataclasses import dataclass

from coro.core.protocols import TranscriptionPipeline
from coro.pipelines.windowing import ASRWindowing

WINDOW_SECONDS_PARAM = "coro_window_seconds"
OVERLAP_SECONDS_PARAM = "coro_overlap_seconds"

WINDOW_SECONDS_DOC = (
    "coro extension. ASR Windowing window length in seconds for this request; "
    "defaults to CORO_ASR_WINDOW_SECONDS."
)
OVERLAP_SECONDS_DOC = (
    "coro extension. ASR Windowing overlap in seconds for this request; must be shorter "
    "than the window. Defaults to CORO_ASR_WINDOW_OVERLAP_SECONDS."
)


class InvalidWindowOverride(ValueError):
    """A request asked for window geometry the pipeline cannot use."""

    def __init__(self, message: str, *, param: str) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


@dataclass(frozen=True)
class WindowOverrides:
    """The window geometry one request asked for; ``None`` keeps the server default."""

    window_seconds: float | None = None
    overlap_seconds: float | None = None

    @property
    def requested(self) -> bool:
        """Whether the request overrides any part of the geometry."""
        return self.window_seconds is not None or self.overlap_seconds is not None


def pipeline_for_request(
    pipeline: TranscriptionPipeline, overrides: WindowOverrides
) -> TranscriptionPipeline:
    """Return the pipeline that serves a request with its window overrides applied.

    The configured pipeline is shared by every request, so an override builds
    a new pipeline over the same adapters instead of changing the shared one.

    Raises:
        InvalidWindowOverride: If a value is out of range, the overlap is not
            shorter than the window, or the pipeline has no window geometry.

    """
    if not overrides.requested:
        return pipeline
    with_windowing = getattr(pipeline, "with_windowing", None)
    windowing = getattr(pipeline, "windowing", None)
    if with_windowing is None or windowing is None:
        raise InvalidWindowOverride(
            "The configured pipeline does not support window overrides.",
            param=WINDOW_SECONDS_PARAM,
        )
    return with_windowing(windowing_for_request(windowing, overrides))


def windowing_for_request(base: ASRWindowing, overrides: WindowOverrides) -> ASRWindowing:
    """Return ``base`` with the request's window overrides applied.

    Raises:
        InvalidWindowOverride: If a value is out of range or the overlap is not
            shorter than the window.

    """
    if not overrides.requested:
        return base
    _reject_out_of_range(overrides)
    try:
        return base.with_overrides(
            window_seconds=overrides.window_seconds,
            overlap_seconds=overrides.overlap_seconds,
        )
    except ValueError as exc:
        raise InvalidWindowOverride(
            f"{OVERLAP_SECONDS_PARAM} must be shorter than {WINDOW_SECONDS_PARAM}.",
            param=OVERLAP_SECONDS_PARAM,
        ) from exc


def _reject_out_of_range(overrides: WindowOverrides) -> None:
    """Reject a non-positive window or a negative overlap."""
    if overrides.window_seconds is not None and not overrides.window_seconds > 0:
        raise InvalidWindowOverride(
            f"{WINDOW_SECONDS_PARAM} must be greater than 0.", param=WINDOW_SECONDS_PARAM
        )
    if overrides.overlap_seconds is not None and not overrides.overlap_seconds >= 0:
        raise InvalidWindowOverride(
            f"{OVERLAP_SECONDS_PARAM} must be 0 or greater.", param=OVERLAP_SECONDS_PARAM
        )
