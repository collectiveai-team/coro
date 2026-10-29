"""Artifact contract resolution for the onnx-canary-split backend.

Resolves ``model_asr`` (a local directory or a Hugging Face repo id) to the
encoder, split-decoder, vocab and config paths; see
``coro/backends/asr/onnx_canary_split.py``'s module docstring (artifact
directory contract and Hugging Face resolution sections).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Artifact directory contract -- see module docstring. Names match
# `.tmp/split_canary_decoder.py`'s and the upstream `canary-1b-v2-onnx`
# repo's own filenames exactly (not a convention invented here).
_ENCODER_BASENAME = "encoder-model"
_XATTN_KV_FILENAME = "xattn_kv.onnx"
_DECODER_STEP_BASENAME = "decoder_step"
_VOCAB_FILENAME = "vocab.txt"
_CONFIG_FILENAME = "config.json"

# Default HF repo holding the whole contract except the fp32 encoder, and the
# upstream export the fp32 encoder (and the split graphs' source) came from.
# See the module docstring's Hugging Face resolution section.
_DEFAULT_SPLIT_REPO = "collectiveai/canary-1b-v2-onnx-split-int8"
_FP32_ENCODER_REPO = "istupakov/canary-1b-v2-onnx"
# Explicit sentinel for "no quantization"; the slug registry (ticket 06)
# normalises it at the settings layer, this builder accepts it directly too.
_FP32_SELECTOR = "fp32"


def _is_fp32(quantization: str | None) -> bool:
    """Whether a quantization selector names the unquantized fp32 graph."""
    return quantization is None or quantization == _FP32_SELECTOR


def _encoder_filename(quantization: str | None) -> str:
    """Return the encoder ONNX filename for a quantization selector."""
    if _is_fp32(quantization):
        return f"{_ENCODER_BASENAME}.onnx"
    return f"{_ENCODER_BASENAME}.{quantization}.onnx"


def _decoder_step_filename(decoder_quantization: str | None) -> str:
    """Return the decoder_step ONNX filename for a decoder quantization selector."""
    if _is_fp32(decoder_quantization):
        return f"{_DECODER_STEP_BASENAME}.onnx"
    return f"{_DECODER_STEP_BASENAME}.{decoder_quantization}.onnx"


def _require_file(directory: Path, filename: str, *, repo_id: str | None = None) -> Path:
    """Return the artifact path, raising this backend's FileNotFoundError shape.

    ``repo_id`` is included in the message when resolution came from the hub,
    so a missing post-download file names the repo and the filename.
    """
    path = directory / filename
    if not path.is_file():
        suffix = f" (repo {repo_id})" if repo_id else ""
        msg = f"Missing required onnx-canary-split artifact: {path}{suffix}"
        raise FileNotFoundError(msg)
    return path


@dataclass(frozen=True)
class _SplitArtifacts:
    """Resolved artifact paths for one build (local directory or HF snapshots).

    The encoder and the split-decode graphs may live in different HF cache
    snapshots (fp32-encoder rule -- see the module docstring), which is why
    resolution yields named paths rather than one directory.
    """

    encoder: Path
    xattn_kv: Path
    decoder_step: Path
    vocab: Path
    config: Path | None = None


def _optional_config(directory: Path) -> Path | None:
    """Return ``config.json`` when present (optional in both local and hub contracts)."""
    config_path = directory / _CONFIG_FILENAME
    return config_path if config_path.is_file() else None


def _artifacts_from_directory(
    directory: Path, *, quantization: str | None, decoder_quantization: str | None
) -> _SplitArtifacts:
    """Resolve the artifact contract against a local artifact directory.

    Raises:
        FileNotFoundError: If a required artifact is missing (path in the message).

    """
    return _SplitArtifacts(
        encoder=_require_file(directory, _encoder_filename(quantization)),
        xattn_kv=_require_file(directory, _XATTN_KV_FILENAME),
        decoder_step=_require_file(directory, _decoder_step_filename(decoder_quantization)),
        vocab=_require_file(directory, _VOCAB_FILENAME),
        config=_optional_config(directory),
    )


def _artifacts_from_hub(
    repo_id: str,
    *,
    quantization: str | None,
    decoder_quantization: str | None,
    hf_token: str | None,
) -> _SplitArtifacts:
    """Resolve the artifact contract from HF snapshots, downloading only what is needed.

    ``allow_patterns`` are derived from the selected quantizations so the
    default INT8/INT8 selection pulls ~1.29 GB, never the 609 MB fp32
    ``decoder_step.onnx``. The fp32 encoder is absent from the split repos, so
    selecting it means a second ``snapshot_download`` from
    ``_FP32_ENCODER_REPO`` and resolved paths spanning two snapshots -- see
    the module docstring's Hugging Face resolution section.

    Raises:
        FileNotFoundError: If a required artifact is still missing after the
            download, with the repo id and filename in the message.

    """
    from huggingface_hub import snapshot_download

    encoder_name = _encoder_filename(quantization)
    shared_patterns = [
        _decoder_step_filename(decoder_quantization),
        _XATTN_KV_FILENAME,
        _VOCAB_FILENAME,
        _CONFIG_FILENAME,
    ]
    if _is_fp32(quantization):
        encoder_repo_id = _FP32_ENCODER_REPO
        encoder_snapshot = Path(
            snapshot_download(
                encoder_repo_id,
                allow_patterns=[encoder_name, f"{encoder_name}.data"],
                token=hf_token,
            )
        )
        split_snapshot = Path(
            snapshot_download(repo_id, allow_patterns=shared_patterns, token=hf_token)
        )
    else:
        encoder_repo_id = repo_id
        encoder_snapshot = split_snapshot = Path(
            snapshot_download(
                repo_id,
                allow_patterns=[encoder_name, f"{encoder_name}.data", *shared_patterns],
                token=hf_token,
            )
        )

    return _SplitArtifacts(
        encoder=_require_file(encoder_snapshot, encoder_name, repo_id=encoder_repo_id),
        xattn_kv=_require_file(split_snapshot, _XATTN_KV_FILENAME, repo_id=repo_id),
        decoder_step=_require_file(
            split_snapshot, _decoder_step_filename(decoder_quantization), repo_id=repo_id
        ),
        vocab=_require_file(split_snapshot, _VOCAB_FILENAME, repo_id=repo_id),
        config=_optional_config(split_snapshot),
    )


def _resolve_artifacts(
    model_asr: str,
    *,
    quantization: str | None,
    decoder_quantization: str | None,
    hf_token: str | None,
) -> _SplitArtifacts:
    """Resolve the artifact contract to concrete paths: local directory else HF hub.

    Follows ``onnx-genai``'s resolution pattern: an existing local directory
    is used as-is (no hub call at all); anything else is a Hugging Face repo
    id. The default repo is ``_DEFAULT_SPLIT_REPO``, but it must be named
    explicitly in ``model_asr`` -- flipping the default is ticket 06.
    """
    directory = Path(model_asr)
    if directory.is_dir():
        return _artifacts_from_directory(
            directory, quantization=quantization, decoder_quantization=decoder_quantization
        )
    return _artifacts_from_hub(
        repo_id=model_asr,
        quantization=quantization,
        decoder_quantization=decoder_quantization,
        hf_token=hf_token,
    )
