"""Package-owned server settings using pydantic-settings.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
module must not mutate global logging policy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PipelineSelector = Literal["full-memory", "chunked-file"]
ASRBackendProvider = Literal["whisperlivekit"]
DiarizationBackendProvider = Literal[None, "whisperlivekit"]


class ServerSettings(BaseSettings):
    """Runtime-injectable settings for the asr_diar_server package."""
    model_config = SettingsConfigDict(env_prefix="ASR_DIAR_")

    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins.")

    pipeline: PipelineSelector = Field(
        default="full-memory", description="Configured Transcription Pipeline selector."
    )
    backend_asr: ASRBackendProvider = Field(
        default="whisperlivekit", description="ASR Backend Provider selector."
    )
    model_asr: str = Field(
        default="openai/whisper-medium", description="ASR Model Selection."
    )
    backend_diarization: DiarizationBackendProvider = Field(
        default=None,
        description="Diarization Backend Provider selector.",
    )
    model_diarization: str | None = Field(
        default=None, description="Diarization Model Selection."
    )
    log_level: str = Field(default="info", description="Log level (for CLI use only).")

    @model_validator(mode="after")
    def default_enabled_diarization_model(self) -> ServerSettings:
        if self.backend_diarization == "whisperlivekit" and self.model_diarization is None:
            self.model_diarization = "nvidia/diar_sortformer_4spk-v1"
        return self
