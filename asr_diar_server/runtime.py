"""App-scoped RuntimeState.

Contains settings, ASR adapter, optional diarization adapter, and
backend-owned resources such as locks. All state is explicit and
test-injectable — no module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    """Holds all live resources for one running asr_diar_server instance.

    An instance with ``asr_adapter=None`` is considered not ready.
    The runtime is injected into the FastAPI app via ``app.state``.

    ``v1_pipeline`` and ``v2_pipeline`` are optional pipeline instances
    injected at startup or in tests.  They are typed as ``Any`` so that
    protocol-compatible fakes can be used in tests without subclassing.
    """

    backend: str = "whisper"
    asr_adapter: Any | None = None
    diarization_adapter: Any | None = None
    v1_pipeline: Any | None = None
    v2_pipeline: Any | None = None
    _extra: dict = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        """Return True when the ASR adapter is loaded and available."""
        return self.asr_adapter is not None
