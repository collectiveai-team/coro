"""Key derivation for the ASR window cache.

A cached window may only be served when it is provably what a fresh run would
produce. That splits into two parts:

- The **fingerprint** covers everything *outside* the request that can change a
  prediction: the ASR Backend Provider, the ASR Model Selection, the
  provider-specific knobs that provider actually honours, the resolved device,
  the inference runtime version, the accelerator identity, the ASR Windowing
  geometry, whether the backend honours prompts, and an explicit cache format
  version.
- The **window key** adds the request inputs that can change a prediction: the
  canonical PCM of the window, a normalised language, and — only for a backend
  that honours it — the carried prompt.

Runtime and accelerator identity are in the fingerprint because determinism is
not portable across them. Measured: CPU and GPU agree on text and timestamps but
diverge on token probabilities by up to 4e-4, and those probabilities are part
of the response.

Keying on canonical PCM (post-decode, post-resample, 16 kHz mono s16le) is what
makes container, encoding, declared sample rate, channel count and filename
automatically irrelevant: the same audio in a different container decodes to the
same bytes and therefore hits the same entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coro.pipelines.windowing import DEFAULT_OVERLAP_SECONDS, DEFAULT_WINDOW_SECONDS

if TYPE_CHECKING:
    from coro.settings import ServerSettings

CACHE_FORMAT_VERSION = 1
"""Stored-shape version. Bump to invalidate every existing entry.

Entries are never migrated: a fingerprint mismatch is simply a miss, and a miss
costs one recomputation rather than a wrong answer.
"""

# Provider-specific settings that can change a *prediction*, with the ASR Backend
# Providers that honour each. Deliberately narrower than the factory's
# leakage-warning table: the concurrency knobs there are honoured but cannot
# change a prediction (determinism under concurrency was measured), so including
# them would cost hit rate for no correctness gain.
PREDICTION_AFFECTING_SETTINGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("asr_compute_type", frozenset({"faster-whisper"})),
    ("asr_quantization", frozenset({"onnx-asr"})),
    ("asr_onnx_vad", frozenset({"onnx-asr"})),
    ("asr_onnx_vad_threshold", frozenset({"onnx-asr"})),
)

_ONNX_PROVIDERS = frozenset({"onnx-asr", "onnx-genai"})
_NVIDIA_PROC_ROOT = Path("/proc/driver/nvidia/gpus")


# MARK: Language Normalisation
def normalise_language(language: str | None) -> str:
    """Return a canonical language value for key derivation.

    The three surfaces disagree today: one validates and strips, one forwards
    the raw string, one coerces empty to absent. Normalising here means
    equivalent spellings share an entry. This affects key derivation only —
    never what is passed to the model — so it cannot change a prediction.

    Args:
        language: Language value as supplied by a request surface.

    Returns:
        The lowercased, stripped language, with underscores folded to hyphens,
        or ``""`` when absent or blank.

    """
    if language is None:
        return ""
    return language.strip().lower().replace("_", "-")


# MARK: Runtime And Accelerator Identity
def _module_version(name: str) -> str | None:
    """Return an installed module's version without importing the module."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_identity(provider: str) -> str:
    """Return the inference runtime identity for an ASR Backend Provider.

    Read from distribution metadata rather than by importing the runtime, so
    computing a fingerprint stays cheap enough for a fully-cached run that never
    loads a model.

    Args:
        provider: ASR Backend Provider selector.

    Returns:
        A stable identity string; ``"<name> unknown"`` when the distribution is
        not installed, which still differs from any real version.

    """
    if provider in _ONNX_PROVIDERS:
        names = ("onnxruntime-gpu", "onnxruntime")
    elif provider == "faster-whisper":
        names = ("faster-whisper", "ctranslate2")
    elif provider == "nemo":
        names = ("torch", "nemo-toolkit")
    else:
        names = ()

    parts = [f"{name} {_module_version(name) or 'absent'}" for name in names]
    return ", ".join(parts) if parts else f"{provider} unknown"


def _nvidia_gpu_names() -> list[str]:
    """Return NVIDIA GPU model names from ``/proc``, empty when unavailable.

    Reading ``/proc`` avoids importing torch or shelling out to ``nvidia-smi``
    just to fingerprint a run.
    """
    names: list[str] = []
    try:
        entries = sorted(_NVIDIA_PROC_ROOT.iterdir())
    except OSError:
        return names
    for entry in entries:
        try:
            text = (entry / "information").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "Model":
                names.append(value.strip())
                break
    return names


def _cuda_is_available(provider: str) -> bool:
    """Return whether the configured provider would pick CUDA under ``auto``."""
    if provider in _ONNX_PROVIDERS:
        return _module_version("onnxruntime-gpu") is not None and bool(_nvidia_gpu_names())
    if provider in ("faster-whisper", "nemo"):
        return bool(_nvidia_gpu_names())
    return False


def resolve_device(device: str, provider: str) -> str:
    """Resolve an ASR device selector to the device that will actually be used.

    ``auto`` is not a device: the same ``auto`` setting means CPU on one host and
    CUDA on another, and those two produce different token probabilities. It is
    resolved here so the fingerprint records what ran, not what was requested.

    Args:
        device: Configured selector (``"auto"``, ``"cuda"``, ``"cpu"``).
        provider: ASR Backend Provider selector, which decides how ``auto`` resolves.

    Returns:
        ``"cpu"`` or ``"cuda"``.

    """
    if device != "auto":
        return device
    return "cuda" if _cuda_is_available(provider) else "cpu"


def accelerator_identity(device: str) -> str:
    """Return the identity of the accelerator a resolved device names.

    Args:
        device: A resolved device (``"cpu"`` or ``"cuda"``).

    Returns:
        ``"cpu"`` for the CPU path, otherwise the visible GPU models and the
        ``CUDA_VISIBLE_DEVICES`` mask that selects among them.

    """
    if device != "cuda":
        return "cpu"
    names = _nvidia_gpu_names() or ["unknown"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return f"cuda[{';'.join(names)}]visible={visible}"


# MARK: Fingerprint
@dataclass(frozen=True)
class FingerprintComponents:
    """Every input that contributes to an ASR fingerprint.

    ``provider_settings`` holds only the knobs the configured ASR Backend
    Provider actually honours, as ordered name/value pairs. A knob the provider
    ignores is absent rather than present-and-null, so setting one costs nothing
    in hit rate.
    """

    cache_format_version: int
    backend_asr: str
    model_asr: str
    device: str
    runtime: str
    accelerator: str
    window_seconds: float
    overlap_seconds: float
    honours_prompt: bool
    provider_settings: tuple[tuple[str, Any], ...] = ()


def fingerprint_components(
    settings: ServerSettings,
    *,
    honours_prompt: bool,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> FingerprintComponents:
    """Return the fingerprint's inputs, for logging and for testing derivation.

    Args:
        settings: Server Startup Selection describing the configured backend.
        honours_prompt: Whether the ASR Backend Provider honours the prompt.
        window_seconds: ASR Windowing window length.
        overlap_seconds: ASR Windowing overlap length.

    Returns:
        Every input that contributes to the fingerprint.

    """
    provider = settings.backend_asr
    device = resolve_device(settings.asr_device, provider)
    return FingerprintComponents(
        cache_format_version=CACHE_FORMAT_VERSION,
        backend_asr=provider,
        model_asr=settings.model_asr,
        device=device,
        runtime=runtime_identity(provider),
        accelerator=accelerator_identity(device),
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        honours_prompt=honours_prompt,
        provider_settings=tuple(
            (name, getattr(settings, name))
            for name, honouring_providers in PREDICTION_AFFECTING_SETTINGS
            if provider in honouring_providers
        ),
    )


def asr_fingerprint(
    settings: ServerSettings,
    *,
    honours_prompt: bool,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> str:
    """Return the digest of everything outside a request that can change a prediction.

    Args:
        settings: Server Startup Selection describing the configured backend.
        honours_prompt: Whether the ASR Backend Provider honours the prompt.
        window_seconds: ASR Windowing window length.
        overlap_seconds: ASR Windowing overlap length.

    Returns:
        A hex digest that changes whenever any contributing input changes.

    """
    components = fingerprint_components(
        settings,
        honours_prompt=honours_prompt,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    payload = json.dumps(asdict(components), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


# MARK: Window Key
def window_key(
    pcm: bytes,
    *,
    fingerprint: str,
    language: str | None,
    prompt: str | None,
) -> str:
    """Return the cache key for one ASR Windowing window.

    ``prompt`` is only meaningful for a backend that honours it; callers pass
    ``None`` for one that does not, which gives independent per-window keys so a
    missing window stays local instead of invalidating its neighbours.

    Args:
        pcm: Canonical PCM bytes of the window (s16le mono 16 kHz).
        fingerprint: Digest from :func:`asr_fingerprint`.
        language: Language value as supplied by the request surface.
        prompt: Carried prompt, or ``None`` when the backend ignores it.

    Returns:
        A hex digest identifying this window under this configuration.

    """
    digest = hashlib.blake2b(digest_size=32)
    digest.update(fingerprint.encode("utf-8"))
    digest.update(b"\x00lang\x00")
    digest.update(normalise_language(language).encode("utf-8"))
    digest.update(b"\x00prompt\x00")
    if prompt is not None:
        digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00pcm\x00")
    digest.update(pcm)
    return digest.hexdigest()
