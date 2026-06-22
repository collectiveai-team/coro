"""Package-owned server settings using pydantic-settings.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
module must not mutate global logging policy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# MARK: Startup Selector Types
PipelineSelector = Literal["full-memory", "streaming"]
ASRBackendProvider = Literal["faster-whisper", "onnx-asr", "onnx-genai"]
DiarizationBackendProvider = Literal["none", "nemo", "pyannote"]
ASRDevice = Literal["auto", "cuda", "cpu"]
OnnxVadSelector = Literal["enabled", "disabled"]
DiarizationDevice = Literal["auto", "cuda", "cpu"]
DiarizationLatencyTier = Literal["very-high", "high", "low", "ultra-low"]


# MARK: Server Settings
class ServerSettings(BaseSettings):
    """Runtime-injectable settings for the coro package."""

    model_config = SettingsConfigDict(
        env_prefix="CORO_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
    model_asr: str = Field(default="openai/whisper-medium", description="ASR Model Selection.")
    asr_device: ASRDevice = Field(default="auto", description="Faster Whisper device selection.")
    asr_compute_type: str = Field(
        default="default",
        description="Faster Whisper compute type selection (ignored by the onnx-asr backend).",
    )
    asr_quantization: str | None = Field(
        default=None,
        description="onnx-asr model quantization selector (e.g. 'int8'); ignored by "
        "the faster-whisper backend.",
    )
    asr_onnx_vad: OnnxVadSelector = Field(
        default="disabled",
        description="Enable Silero VAD speech segmentation for the onnx-asr backend "
        "(via onnx_asr.load_vad('silero')). Ignored by the faster-whisper and "
        "onnx-genai backends.",
    )
    asr_onnx_vad_threshold: float | None = Field(
        default=None,
        description="Optional Silero VAD speech probability threshold for the onnx-asr "
        "backend; only used when asr_onnx_vad is 'enabled'. None uses onnx-asr's default.",
    )
    backend_diarization: DiarizationBackendProvider = Field(
        default="none",
        description="Diarization Backend Provider selector.",
    )
    model_diarization: str | None = Field(default=None, description="Diarization Model Selection.")
    diarization_device: DiarizationDevice = Field(
        default="auto", description="Diarization device selection."
    )
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CORO_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        description="HuggingFace access token for gated diarization models (e.g. the "
        "pyannote community-1 pipeline). Read from CORO_HF_TOKEN, HF_TOKEN, or "
        "HUGGING_FACE_HUB_TOKEN; masked in logs.",
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
        if self.model_diarization is None:
            if self.backend_diarization == "nemo":
                self.model_diarization = "nvidia/diar_streaming_sortformer_4spk-v2"
            elif self.backend_diarization == "pyannote":
                self.model_diarization = "pyannote/speaker-diarization-community-1"
        return self

    @model_validator(mode="after")
    def reject_streaming_pyannote(self) -> ServerSettings:
        """Reject the Streaming Pipeline for the batch-only pyannote backend."""
        if self.backend_diarization == "pyannote" and self.pipeline == "streaming":
            msg = (
                "The 'pyannote' diarization backend is batch-only and cannot run "
                "with the 'streaming' pipeline. Use CORO_PIPELINE=full-memory, or "
                "select a streaming-capable diarization backend (e.g. 'nemo')."
            )
            raise ValueError(msg)
        return self
