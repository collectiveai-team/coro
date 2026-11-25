"""The ASR Windowing settings: window length and overlap, read as ``CORO_`` variables."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from coro.settings.selection import DEFAULT_OVERLAP_SECONDS, DEFAULT_WINDOW_SECONDS


class ASRWindowSettings(BaseSettings):
    """Server-wide ASR Windowing geometry; ``ServerSettings`` extends it with its own prefix."""

    asr_window_seconds: float = Field(
        default=DEFAULT_WINDOW_SECONDS,
        gt=0,
        description="ASR Windowing window length in seconds. A request can override it "
        "with coro_window_seconds.",
    )
    asr_window_overlap_seconds: float = Field(
        default=DEFAULT_OVERLAP_SECONDS,
        ge=0,
        description="ASR Windowing overlap between consecutive windows in seconds; must "
        "be shorter than asr_window_seconds. A request can override it with "
        "coro_overlap_seconds.",
    )

    @model_validator(mode="after")
    def reject_overlap_not_shorter_than_window(self) -> ASRWindowSettings:
        """Reject an ASR Windowing overlap that does not leave a positive step."""
        if self.asr_window_overlap_seconds >= self.asr_window_seconds:
            msg = (
                f"CORO_ASR_WINDOW_OVERLAP_SECONDS ({self.asr_window_overlap_seconds}) must be "
                f"shorter than CORO_ASR_WINDOW_SECONDS ({self.asr_window_seconds})."
            )
            raise ValueError(msg)
        return self
