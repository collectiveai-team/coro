"""Package-owned server settings using pydantic-settings.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
module must not mutate global logging policy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# MARK: Startup Selector Types
PipelineSelector = Literal["full-memory", "streaming"]
ASRBackendProvider = Literal["faster-whisper", "onnx-asr", "onnx-genai"]
DiarizationBackendProvider = Literal["none", "nemo"]
ASRDevice = Literal["auto", "cuda", "cpu"]
DiarizationDevice = Literal["auto", "cuda", "cpu"]
DiarizationLatencyTier = Literal["very-high", "high", "low", "ultra-low"]


# MARK: Server Settings
class ServerSettings(BaseSettings):
    """Runtime-injectable settings for the asr_diar_server package."""

    model_config = SettingsConfigDict(env_prefix="ASR_DIAR_", case_sensitive=False)

    # Process Settings ------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins.")

    # Transcription Selectionmmms ----------------------------------------------
    pipeline: PipelineSelector = Field(
        default="full-memory", description="Configured Transcription Pipeline selector."
    )
    backend_asr: ASRBackendProvider = Field(
        default="faster-whisper", description="ASR Backend Provider selector."
    )
    model_asr: str = Field(
        default="openai/whisper-medium", description="ASR Model Selection."
    )
    asr_device: ASRDevice = Field(
        default="auto", description="Faster Whisper device selection."
    )
    asr_compute_type: str = Field(
        default="default",
        description="Faster Whisper compute type selection (ignored by the onnx-asr backend).",
    )
    asr_quantization: str | None = Field(
        default=None,
        description="onnx-asr model quantization selector (e.g. 'int8'); ignored by "
        "the faster-whisper backend.",
    )
    backend_diarization: DiarizationBackendProvider = Field(
        default="none",
        description="Diarization Backend Provider selector.",
    )
    model_diarization: str | None = Field(
        default=None, description="Diarization Model Selection."
    )
    diarization_device: DiarizationDevice = Field(
        default="auto", description="NeMo diarization device selection."
    )
    transcript_spill_dir: str | None = Field(
        default=None,
        description="Directory for the streaming pipeline's per-request transcript "
        "spill store. MUST be on real disk for flat host RAM: a tmpfs path (e.g. "
        "/tmp on many systems) keeps the transcript in memory and defeats the spill. "
        "None uses the system temp dir.",
    )
    log_level: str = Field(default="info", description="Log level (for CLI use only).")

    diarization_latency: DiarizationLatencyTier = Field(
        default="very-high",
        description="Diarization Latency Selection tier for streaming Sortformer.",
    )

    # Server Warmup ---------------------------------------------------------
    warmup: Literal["enabled", "disabled"] = Field(
        default="enabled",
        description="Server Warmup runs the Configured Transcription Pipeline against "
        "the Warmup Audio Asset at startup. Set to 'disabled' to skip warmup.",
    )

    # TLS ------------------------------------------------------------------
    ssl_certfile: str | None = Field(default=None, description="TLS certificate file path.")
    ssl_keyfile: str | None = Field(default=None, description="TLS private key file path.")

    # Derived Defaults ------------------------------------------------------
    @model_validator(mode="after")
    def default_enabled_diarization_model(self) -> ServerSettings:
        if self.backend_diarization == "nemo" and self.model_diarization is None:
            self.model_diarization = "nvidia/diar_streaming_sortformer_4spk-v2"
        return self
