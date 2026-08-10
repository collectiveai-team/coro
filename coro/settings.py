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
        default="onnx-asr", description="ASR Backend Provider selector."
    )
    model_asr: str = Field(default="nemo-parakeet-tdt-0.6b-v3", description="ASR Model Selection.")
    asr_device: ASRDevice = Field(default="auto", description="Faster Whisper device selection.")
    asr_compute_type: str = Field(
        default="default",
        description="Faster Whisper compute type selection (ignored by the onnx-asr backend).",
    )
    asr_quantization: str | None = Field(
        default=None,
        description="onnx-asr model quantization selector (e.g. 'int8'); ignored by "
        "the faster-whisper backend. Left unset on purpose: int8 is a memory-fitting "
        "tool for the default transducer ASR Model Selection, not a speed tool "
        "(measured: no throughput gain, small WER cost). See docs/benchmark.md.",
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
        description="Diarization Backend Provider selector. Defaults to 'none' as an "
        "explicit product decision, not by omission: an ASR-Only Server is a valid "
        "configuration, and enabling streaming Sortformer by default would cost ~24% "
        "Transcription Throughput, ~1 GB peak Process-Tree PSS and a ~500 MB model "
        "download on first start, while capping the server at 4 speakers. "
        "See docs/benchmark.md.",
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
    diarization_postprocessing: str | None = Field(
        default=None,
        description="Diarization Post-Processing Configuration for NeMo Sortformer: "
        "a vendored preset name ('dihard3-dev', 'callhome-part1') or a path to a "
        "custom YAML in the same schema. None keeps NeMo's own unconfigured baseline "
        "unchanged. Ignored by the pyannote backend. See ADR 0010 — neither preset is "
        "a coro recommendation; choosing one is a per-deployment operator decision.",
    )

    # Server Warmup ---------------------------------------------------------
    warmup: Literal["enabled", "disabled"] = Field(
        default="enabled",
        description="Server Warmup runs the Configured Transcription Pipeline against "
        "the Warmup Audio Asset at startup. Set to 'disabled' to skip warmup.",
    )

    # Adapter Concurrency Policy --------------------------------------------
    asr_max_concurrency: int = Field(
        default=0,
        ge=0,
        description="Maximum ASR inference calls allowed to run at once. 0 (default) "
        "auto-sizes from the host core count so total backend thread demand stays "
        "near it. Ignored by the onnx-genai backend, whose Adapter Concurrency "
        "Policy fixes the permit count at 1.",
    )
    asr_max_queue_depth: int = Field(
        default=32,
        ge=0,
        description="Maximum ASR inference calls allowed to wait for a concurrency "
        "permit. Requests beyond this cap are rejected with an OpenAI-Style Error "
        "(HTTP 429) carrying a Retry-After hint instead of being queued indefinitely.",
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

    @model_validator(mode="after")
    def resolve_transcript_spill_dir(self) -> ServerSettings:
        """Resolve the Streaming Pipeline's transcript spill directory to real disk.

        The spill store exists to keep host memory flat, so a RAM-backed
        directory (``/tmp`` is ``tmpfs`` on most Linux distributions) silently
        defeats it. When unset, a real-disk default is chosen; when explicitly
        set to a RAM-backed path, startup fails loudly. Only the Streaming
        Pipeline spills, so no other pipeline selector is affected.
        """
        if self.pipeline != "streaming":
            return self
        from coro.pipelines.spill import resolve_spill_dir

        self.transcript_spill_dir = resolve_spill_dir(self.transcript_spill_dir)
        return self

    @model_validator(mode="after")
    def reject_enabled_diarization_without_model(self) -> ServerSettings:
        """Reject an enabled diarization Backend Provider with no Diarization Model Selection.

        Runs after ``default_enabled_diarization_model``, so a model is only
        missing here when it was explicitly set to an empty value. Without this
        check the server silently degrades to an ASR-Only Server, producing
        single-speaker hypotheses that look like a diarization quality
        regression rather than a configuration error.
        """
        if self.backend_diarization != "none" and not (self.model_diarization or "").strip():
            msg = (
                f"Diarization Backend Provider '{self.backend_diarization}' is selected but "
                "the Diarization Model Selection is empty. Set CORO_MODEL_DIARIZATION to a "
                "model id, or set CORO_BACKEND_DIARIZATION=none for an ASR-Only Server."
            )
            raise ValueError(msg)
        return self
