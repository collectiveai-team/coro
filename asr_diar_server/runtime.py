"""App-scoped RuntimeState.

Contains settings, ASR adapter, optional diarization adapter, and
backend-owned resources such as locks. All state is explicit and
test-injectable — no module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# MARK: Singleton Runtime State
@dataclass
class RuntimeState:
    """Holds all live resources for one running asr_diar_server instance.

    An instance with ``asr_adapter=None`` is considered not ready.
    The runtime is injected into the FastAPI app via ``app.state``.

    ``pipeline`` is typed as ``Any`` so protocol-compatible fakes can be used
    in tests without subclassing.
    """

    pipeline: Any | None = None
    pipeline_selector: str = "full-memory"
    asr_provider: str = "whisperlivekit"
    asr_model: str = "openai/whisper-medium"
    diarization_provider: str = "none"
    diarization_model: str | None = None
    asr_adapter: Any | None = None
    diarization_adapter: Any | None = None
    warmup_ready: bool = False
    _extra: dict = field(default_factory=dict, repr=False)

    # Capability Readiness --------------------------------------------------
    @property
    def ready(self) -> bool:
        """Return True when the ASR adapter is loaded and available."""
        return self.asr_adapter is not None
