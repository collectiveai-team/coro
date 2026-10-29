"""Startup selector types and the ASR Model Selection slug registry (ADR 0020)."""

from __future__ import annotations

from typing import Literal, TypedDict

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
