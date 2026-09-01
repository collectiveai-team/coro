"""NeMo ASR Model Integration (nemo backend).

Evaluation vehicle for language-constrained NeMo transducers — primarily the
Parakeet RNNT 1.1B multilingual *Prompt* checkpoints — which the onnx-asr
backend cannot drive: their encoder takes a ``prompt_indices`` language one-hot
input that onnx-asr's ``nemo-conformer-rnnt`` path never feeds, and onnx-asr's
``language`` argument is silently dropped outside its Whisper/Canary paths.

This adapter runs the PyTorch checkpoint through NeMo's ``transcribe``, where
language forcing is native: the request language is resolved against the
checkpoint's ``prompt_dictionary`` and passed as ``target_lang``. Token
timestamps come from NeMo's RNNT timestamp computation, so the project-owned
``TranscriptToken`` contract (per-word start/end, chronological order) holds.

Accepted trade-off: PyTorch eager CPU inference of a 1.1B checkpoint is far
slower than ONNX Runtime with int8 quantisation. This backend exists to
measure *quality* (forced ``es``/``es-US``/``es-ES`` vs auto detection); if
the measurements justify adoption, the production path is an ONNX export
(baked-prompt or upstream ``prompt_indices`` support), not this adapter.

Note:
    Prompt checkpoints whose ``prompt_dictionary`` has no ``"auto"`` entry
    cannot transcribe without an explicit language, so Server Warmup (which
    sends none) fails loudly. Run such checkpoints with ``CORO_WARMUP=disabled``
    until a language-aware warmup exists.

"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.onnx_asr import _LAST_WORD_PAD, _group_subwords, _words_from_text
from coro.core.models import TranscriptToken

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32


def resolve_target_language(language: str | None, prompt_dictionary: dict[str, int]) -> str | None:
    """Resolve a request language to a checkpoint prompt-dictionary key.

    Args:
        language: Request language (e.g. ``"es"``, ``"es-US"``, ``"es-ES"``),
            or None to leave the model's default behaviour (auto detection
            where the checkpoint supports it).
        prompt_dictionary: The checkpoint's target-language -> prompt-id map.

    Returns:
        The dictionary key to pass as ``target_lang``, or None when no
        language was requested.

    Raises:
        ValueError: If the language matches no dictionary key. Listing the
            supported keys turns a silent English fallback (the failure mode
            this backend exists to avoid) into an explicit configuration error.

    """
    if not language:
        return None
    if language in prompt_dictionary:
        return language
    primary = language.split("-")[0]
    for key in prompt_dictionary:
        if key.split("-")[0] == primary:
            return key
    available = ", ".join(sorted(prompt_dictionary) or ["<none>"])
    msg = (
        f"Language {language!r} is not supported by this model's prompt dictionary "
        f"(supported: {available})."
    )
    raise ValueError(msg)


def _tokens_from_hypothesis(hyp, tokenizer, *, span_end: float) -> list[TranscriptToken]:
    """Convert one NeMo RNNT hypothesis into word-level TranscriptTokens.

    ``hyp.timestamp`` carries one emission time per predicted subword token
    (SentencePiece pieces using the ``▁`` word-start marker), so words are
    reconstructed with the same grouping as the onnx-asr converter: a word's
    ``start`` is its first subword's time, its ``end`` the next word's start
    (final word padded by ``_LAST_WORD_PAD``). No per-word probability source
    exists on the hypothesis, so ``probability`` stays None rather than being
    manufactured. When the checkpoint emits no timestamps, timings are spread
    evenly over the clip span via the shared text fallback.
    """
    timestamps = [float(t) for t in (getattr(hyp, "timestamp", None) or [])]
    pieces: list[str] = []
    if timestamps:
        token_ids = list(getattr(hyp, "y", None) or ())
        if tokenizer is not None and token_ids:
            pieces = list(tokenizer.convert_ids_to_tokens(token_ids))
    if pieces and len(pieces) == len(timestamps):
        groups = _group_subwords(pieces, timestamps, None)
        out: list[TranscriptToken] = []
        for i, group in enumerate(groups):
            start = group["start"]
            end = groups[i + 1]["start"] if i + 1 < len(groups) else start + _LAST_WORD_PAD
            out.append(
                TranscriptToken(
                    start=round(start, 3),
                    end=round(max(end, start), 3),
                    text=group["text"],
                    probability=None,
                )
            )
        return out

    text = (getattr(hyp, "text", "") or "").strip()
    return _words_from_text(text, 0.0, span_end or None)


class NemoASRAdapter:
    """ASRAdapter that wraps a NeMo transducer (EncDecRNNTBPEModel family).

    Adapter Concurrency Policy: **serialised**. A PyTorch module is not
    documented thread-safe for concurrent ``transcribe`` calls, and NeMo's
    decoding mutates model-level decoding state. Serialising is expressed as
    a one-permit :class:`AdmissionController` rather than a ``threading.Lock``,
    so waiting happens on the event loop and overload past the queue-depth cap
    is rejected with a retry hint — the same policy and precedent as the
    onnx-genai backend.
    """

    honours_prompt: bool = False
    """A transducer has no text input port, so the carried prompt is inert.

    Language forcing — the reason this backend exists — is separate: it flows
    through the ``language`` argument and the checkpoint's prompt dictionary,
    not through the carried prompt.
    """

    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        prompt_dictionary: dict[str, int] | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_dictionary = prompt_dictionary
        self._admission = admission or build_admission_controller(
            max_concurrency=1, max_queue_depth=_DEFAULT_QUEUE_DEPTH, serialized=True
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored (no
            text input port on a transducer).

        Raises:
            AsrCapacityError: If the admission queue is full.
            ValueError: If an explicit language matches no prompt-dictionary
                key, or the checkpoint carries no prompt dictionary at all.

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if language and self._prompt_dictionary is None:
            msg = (
                "This model has no prompt dictionary, so language forcing is not "
                "supported by the checkpoint. Send no language, or use a Prompt "
                "variant checkpoint."
            )
            raise ValueError(msg)
        target_lang = resolve_target_language(language, self._prompt_dictionary or {})

        duration = len(audio) / _SAMPLE_RATE
        async with self._admission.admit():
            return await asyncio.to_thread(self._transcribe, audio, target_lang, duration)

    def _transcribe(
        self, audio: np.ndarray, target_lang: str | None, duration: float
    ) -> list[TranscriptToken]:
        """Run NeMo ``transcribe`` on a temp wav file and convert the hypothesis.

        NeMo's ``transcribe`` takes file paths, so the window PCM is written to
        a temporary wav (16 kHz, s16le) and removed afterwards; the write is
        negligible next to transducer inference.
        """
        import soundfile as sf

        kwargs: dict = {
            "timestamps": True,
            "return_hypotheses": True,
            "batch_size": 1,
            "verbose": False,
        }
        if target_lang is not None:
            kwargs["target_lang"] = target_lang

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, _SAMPLE_RATE, subtype="PCM_16")
            hypotheses = self._model.transcribe([path], **kwargs)
        finally:
            Path(path).unlink(missing_ok=True)

        hyp = hypotheses[0] if hypotheses else SimpleNamespace(text="", timestamp=[])
        return _tokens_from_hypothesis(hyp, self._tokenizer, span_end=duration)


def build_nemo_asr_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> NemoASRAdapter:
    """Construct and return a NemoASRAdapter.

    Args:
        model_asr: NeMo model name (``nvidia/parakeet-rnnt-1.1b-...``) or path
            to a local ``.nemo`` checkpoint.
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``); ``auto``
            uses CUDA when torch reports it available.
        max_queue_depth: Calls allowed to wait for the single permit before
            rejection. The permit count is fixed at 1 by this backend's
            Adapter Concurrency Policy, so ``asr_max_concurrency`` does not
            apply.

    Returns:
        Initialised adapter ready for use.

    """
    import nemo.collections.asr as nemo_asr
    import torch

    model: Any
    local = Path(model_asr)
    if local.suffix == ".nemo" and local.exists():
        model = nemo_asr.models.ASRModel.restore_from(str(local))
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_asr)

    resolved = device
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(resolved)

    model_defaults = model.cfg.get("model_defaults", {}) or {}
    prompt_dictionary = dict(model_defaults.get("prompt_dictionary", {}) or {}) or None
    tokenizer = getattr(model.tokenizer, "tokenizer", model.tokenizer)

    logger.info(
        "Loaded NeMo ASR model '%s' (device=%s, prompt_dictionary=%s).",
        model_asr,
        resolved,
        sorted(prompt_dictionary) if prompt_dictionary else None,
    )
    return NemoASRAdapter(
        model,
        tokenizer=tokenizer,
        prompt_dictionary=prompt_dictionary,
        admission=build_admission_controller(
            max_concurrency=1, max_queue_depth=max_queue_depth, serialized=True
        ),
    )
