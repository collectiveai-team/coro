"""Package-owned server settings using pydantic-settings.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
module must not mutate global logging policy.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import AliasChoices, Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# MARK: Startup Selector Types
PipelineSelector = Literal["full-memory", "streaming"]
# License note (see CONTEXT.md's Sortformer-v1 precedent for the same policy
# applied to a Diarization Model Selection): "nemo" and "onnx-parakeet-prompt"
# both exist to drive parakeet-rnnt-1.1b-multilingual-prompt, which is
# licensed under the NVIDIA Community Model License and gated behind an
# NVIDIA NIM runtime / AI Enterprise subscription for production use -- see
# ``coro/backends/asr/nemo.py``'s and ``coro/backends/asr/onnx_parakeet_prompt.py``'s
# module docstrings. Comparative-reference backends only: never the default
# (``onnx-canary-split`` is, see ADR 0019), never recommended, and their
# weights/derivative ONNX exports must not be redistributed. "onnx-canary-split"
# drives `nvidia/canary-1b-v2` (CC-BY-4.0, no NIM/redistribution restriction)
# -- see ``coro/backends/asr/onnx_canary_split.py``'s module docstring and
# ADR 0019 for why it is the default.
ASRBackendProvider = Literal[
    "faster-whisper", "onnx-asr", "onnx-genai", "nemo", "onnx-parakeet-prompt", "onnx-canary-split"
]
DiarizationBackendProvider = Literal["none", "nemo", "pyannote"]
ASRDevice = Literal["auto", "cuda", "cpu"]
OnnxVadSelector = Literal["enabled", "disabled"]
DiarizationDevice = Literal["auto", "cuda", "cpu"]
DiarizationLatencyTier = Literal["very-high", "high", "low", "ultra-low"]


# MARK: Model Slug Registry
class _SlugConfig(TypedDict):
    """One slug's complete ASR configuration -- see ``resolve_model_slug``."""

    backend_asr: ASRBackendProvider
    model_asr: str
    asr_quantization: str | None
    asr_decoder_quantization: str | None


# A slug names a complete, known-good ASR configuration so a `model_asr` value
# alone can select backend + model + quantization together. See ADR 0020.
MODEL_SLUGS: dict[str, _SlugConfig] = {
    "canary-1b-v2": {
        "backend_asr": "onnx-canary-split",
        "model_asr": "collectiveai/canary-1b-v2-onnx-split-int8",
        "asr_quantization": "static_qdq_v4_pct_excl",
        "asr_decoder_quantization": "dynamic_v1_quint8",
    },
    "parakeet-tdt-0.6b-v3": {
        "backend_asr": "onnx-asr",
        "model_asr": "nemo-parakeet-tdt-0.6b-v3",
        "asr_quantization": None,
        "asr_decoder_quantization": None,
    },
    "whisper-large-v3-turbo": {
        "backend_asr": "faster-whisper",
        "model_asr": "large-v3-turbo",
        "asr_quantization": None,
        "asr_decoder_quantization": None,
    },
    "whisper-large-v3": {
        "backend_asr": "faster-whisper",
        "model_asr": "large-v3",
        "asr_quantization": None,
        "asr_decoder_quantization": None,
    },
}

DEFAULT_MODEL_SLUG = "canary-1b-v2"

# The sentinel that turns a slug's quantization off. Distinct from `None`
# (unset -- let the slug or the backend's own default decide) because a slug
# default needs an explicit way to say "no quantization" rather than merely
# "no opinion".
FP32_QUANTIZATION_SENTINEL = "fp32"


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
    # pyrefly mis-widens the declared type to `str` once ASRBackendProvider's
    # Literal grows past 4 members (reproduced in isolation, unrelated to
    # pydantic-settings specifics) -- verified false positive, not a real
    # type error: "onnx-asr" is one of the Literal's own members.
    # This class-level default is a pydantic-typing placeholder only, never
    # actually read: resolve_model_slug always overwrites it (from a
    # recognised model_asr slug) or raises (a non-slug model_asr with no
    # explicit backend_asr) before any other code sees this field. See that
    # validator and ADR 0020.
    backend_asr: ASRBackendProvider = Field(  # pyrefly: ignore[bad-assignment]
        default="onnx-asr",
        description="ASR Backend Provider selector. Derived from model_asr's slug when "
        "not given explicitly; required explicitly alongside a non-slug model_asr.",
    )
    model_asr: str = Field(
        default=DEFAULT_MODEL_SLUG,
        description="ASR Model Selection: a slug resolving backend_asr and quantization "
        f"together ({', '.join(sorted(MODEL_SLUGS))}; default: {DEFAULT_MODEL_SLUG!r}), or "
        "a raw model id/path for the ASR Backend Provider given via backend_asr.",
    )
    asr_device: ASRDevice = Field(default="auto", description="Faster Whisper device selection.")
    asr_compute_type: str = Field(
        default="default",
        description="Faster Whisper compute type selection (ignored by the onnx-asr backend).",
    )
    asr_quantization: str | None = Field(
        default=None,
        description="Encoder quantization selector (e.g. 'int8' for onnx-asr, "
        "'static_qdq_v4_pct_excl' for onnx-canary-split); ignored by the "
        "faster-whisper backend. None means 'let the resolved model slug decide': "
        "the default canary-1b-v2 slug fills this with 'static_qdq_v4_pct_excl' "
        "-- a real win for that backend (norm cpWER 0.0508 vs fp32's 0.0513, "
        "+4.0% RTFx), not just a memory trade -- see ADR 0019 and "
        "coro/backends/asr/onnx_canary_split.py's module docstring. The "
        "parakeet-tdt-0.6b-v3 slug leaves it unset instead: int8 is a "
        "memory-fitting tool for that transducer, not a speed tool (measured: no "
        "throughput gain, small WER cost). See docs/benchmark.md. Explicit "
        "'fp32' always turns a slug's own default back off.",
    )
    asr_decoder_quantization: str | None = Field(
        default=None,
        description="onnx-canary-split decoder_step.onnx quantization selector (e.g. "
        "'dynamic_v1_quint8'); ignored by every other ASR Backend Provider. Distinct "
        "from asr_quantization (which selects the encoder's quantization for this "
        "same backend) because the decoder graph needed a different technique: "
        "static-QDQ INT8 (usable for the encoder) caused real word-level WER damage "
        "on the autoregressive decoder, while dynamic INT8 did not. The default "
        "canary-1b-v2 slug fills this with 'dynamic_v1_quint8' (+3.1% relative "
        "cpWER, +43.8% RTFx). See coro/backends/asr/onnx_canary_split.py's module "
        "docstring for the measured quality/speed numbers and ADR 0019 for the "
        "default decision. Explicit 'fp32' turns it off, independently of "
        "asr_quantization.",
    )
    asr_fallback_language: str = Field(
        default="en",
        description="Language used when a request gives no language hint and no "
        "detection is available. Consulted today by the onnx-canary-split backend "
        "only (its Canary checkpoint has no auto-detection); slice 05's language "
        "detection will take precedence over this fallback where available, and "
        "only consult it when detection never yields a language. Server Warmup "
        "also passes it explicitly.",
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
        "custom YAML in the same schema. None, '' and 'none' all keep NeMo's own "
        "unconfigured baseline unchanged. Ignored by the pyannote backend. See ADR "
        "0010 — neither preset is a coro recommendation; choosing one is a "
        "per-deployment operator decision.",
    )
    diarization_postprocessing_max_speakers: int = Field(
        default=4,
        ge=1,
        description="Speaker-Count Post-Processing Gate ceiling. Above this estimated "
        "speaker count the Diarization Post-Processing Configuration is bypassed in "
        "favour of NeMo's baseline, because tuned short-segment deletion is reported "
        "to degrade DER for five or more speakers. Has no observable effect while a "
        "4-speaker Diarization Model Selection is configured, since the estimate can "
        "never exceed 4. See ADR 0010.",
    )

    # ASR Window Cache ------------------------------------------------------
    asr_cache: Literal["enabled", "disabled"] = Field(
        default="disabled",
        description="Reuse previously-computed ASR window results from disk, so "
        "re-running audio that has already been transcribed skips the model "
        "entirely. Keyed on the canonical PCM of each window plus everything "
        "that can change a prediction; response-formatting and diarization "
        "options are excluded, so changing those still hits. Disabled by "
        "default because it introduces disk growth to a service that has none.",
    )
    asr_cache_dir: str | None = Field(
        default=None,
        description="Directory for the ASR window cache. MUST be on real disk: a "
        "tmpfs path both competes for the memory the cache saves and loses every "
        "entry on restart. None uses a directory under the user cache root.",
    )
    asr_cache_max_mb: int = Field(
        default=1024,
        ge=0,
        description="Size cap for the ASR window cache, in megabytes. Least-"
        "recently-accessed entries are evicted on write once the cap is exceeded. "
        "0 disables the cap. Only digests and transcript tokens are stored — "
        "never audio — so a few megabytes covers an hour of audio.",
    )
    asr_cache_ttl_days: float = Field(
        default=30.0,
        ge=0,
        description="Lifetime of an ASR window cache entry, in days, applied "
        "lazily on lookup: an expired entry is a miss and is removed. 0 disables "
        "expiry, leaving the size cap as the only bound.",
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

    # Fields the operator actually set (CLI flag, env var, or constructor
    # kwarg), captured before resolve_model_slug's own mutations -- pydantic's
    # own `model_fields_set` would otherwise also pick up those mutations
    # (setting an attribute in a `mode="after"` validator adds it), which is
    # exactly what would make warn_ignored_asr_settings misattribute a
    # slug-filled value to the operator. Never read before that validator
    # has run (every successful construction runs it).
    _explicit_fields: frozenset[str] = PrivateAttr(default=frozenset())

    # Derived Defaults ------------------------------------------------------
    @model_validator(mode="after")
    def resolve_model_slug(self) -> ServerSettings:
        """Resolve model_asr's slug into backend_asr, model_asr and quantization.

        Precedence: an operator-set value always wins over the slug's own
        default; the slug only fills fields the operator left unset. A
        non-slug model_asr passes through verbatim as the model id/path for
        backend_asr, which must then be given explicitly -- backward
        compatible with ``--backend-asr onnx-asr --model-asr
        nemo-parakeet-tdt-0.6b-v3``. ``fp32`` for either quantization selector
        always collapses to ``None`` (backends never see the string), the
        only way to turn a slug's quantization off. See ``MODEL_SLUGS`` and
        ADR 0020.

        Raises:
            ValueError: If model_asr is not a recognised slug and backend_asr
                was not given explicitly.

        """
        self._explicit_fields = frozenset(self.model_fields_set)

        slug = MODEL_SLUGS.get(self.model_asr)
        if slug is not None:
            if "backend_asr" not in self._explicit_fields:
                self.backend_asr = slug["backend_asr"]
            if "asr_quantization" not in self._explicit_fields:
                self.asr_quantization = slug["asr_quantization"]
            if "asr_decoder_quantization" not in self._explicit_fields:
                self.asr_decoder_quantization = slug["asr_decoder_quantization"]
            self.model_asr = slug["model_asr"]
        elif "backend_asr" not in self._explicit_fields:
            msg = (
                f"ASR Model Selection model_asr={self.model_asr!r} is not a recognised "
                f"slug ({', '.join(sorted(MODEL_SLUGS))}) and no backend_asr was given. "
                "Set CORO_BACKEND_ASR (or --backend-asr) alongside a raw model id/path, "
                "or use one of the slugs above."
            )
            raise ValueError(msg)

        if self.asr_quantization == FP32_QUANTIZATION_SENTINEL:
            self.asr_quantization = None
        if self.asr_decoder_quantization == FP32_QUANTIZATION_SENTINEL:
            self.asr_decoder_quantization = None
        return self

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
    def resolve_asr_cache_dir(self) -> ServerSettings:
        """Resolve the ASR window cache directory to real disk, when enabled.

        Strict Startup Validation, not lazy validation on first request: an
        operator who mistypes a cache directory should find out when the server
        starts. The Streaming Pipeline's spill resolution cannot cover this — it
        only runs for one pipeline selector, while the cache applies to all.
        """
        if self.asr_cache != "enabled":
            return self
        from coro.cache.directory import resolve_cache_dir

        self.asr_cache_dir = resolve_cache_dir(self.asr_cache_dir)
        return self

    @model_validator(mode="after")
    def collapse_blank_fallback_language(self) -> ServerSettings:
        """Collapse a blank asr_fallback_language to the default.

        Blank means unset (the same forgiving contract the request surfaces
        give their optional language fields), and an empty string would
        otherwise reach the adapter as "no language at all".
        """
        if not self.asr_fallback_language.strip():
            self.asr_fallback_language = "en"
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
