"""NeMo ASR Model Integration (nemo backend).

Evaluation vehicle for language-constrained NeMo transducers — primarily the
Parakeet RNNT 1.1B multilingual *Prompt* checkpoints — which the onnx-asr
backend cannot drive today: it has no registry entry for this checkpoint
class, and its ``language`` argument is silently dropped outside its
Whisper/Canary paths. (Earlier notes here claimed the encoder itself needed
an unsupported ``prompt_indices`` input; verified against a real checkpoint
on 2026-09-01, that is wrong -- the encoder is plain and onnx-asr-compatible.
Prompt conditioning is a small *post*-encoder MLP, ``model.prompt_kernel``,
which is why the production path is closer than previously assessed; see
``.scratch/issue-64-language-constrained-asr/findings.md``.)

This adapter runs the PyTorch checkpoint through NeMo's ``transcribe``, where
language forcing works via the request language resolved against the
checkpoint's ``prompt_dictionary`` and passed as ``target_lang``. Token
timestamps come from NeMo's RNNT timestamp computation, so the project-owned
``TranscriptToken`` contract (per-word start/end, chronological order) holds
-- except on the forced-language path, where a NeMo bug forces a fallback
(see the two caveats below).

Two NeMo-level quirks, discovered against a real ``EncDecHybridRNNTCTCBPEModelWithPrompt``
checkpoint (verified 2026-09-01; see ``.scratch/issue-64-language-constrained-asr/findings.md``),
shape how this adapter must call ``transcribe()``:

1. **``target_lang`` as a bare kwarg is silently ignored.** ``transcribe()``'s
   default dataloader (``use_lhotse=True``) reads the target language from a
   manifest supervision's ``language`` field, not from the transcribe config
   -- a bare-file-path call has no such field, so the request is dropped
   without error and the checkpoint falls back to whatever the (missing)
   default resolves to. ``target_lang`` only reaches the model when passed
   via an explicit ``override_config=<TranscribeConfig subclass>(...,
   use_lhotse=False)`` -- only the non-Lhotse dataloader threads
   ``target_lang`` into the per-utterance one-hot prompt tensor. The
   ``override_config`` dataclass type is discovered via introspection
   (:func:`_resolve_override_config_type`) rather than imported by name, so
   this works for any Prompt-conditioned NeMo ASR model class, not only
   ``EncDecHybridRNNTCTCBPEModelWithPrompt``.
2. **``timestamps=True`` crashes on the forced-language path, and even
   ``timestamps=False`` doesn't yield None/empty.** With ``timestamps=True``,
   ``process_timestamp_outputs`` crashes downstream
   (``if 'word' in timestamp`` raises ``RuntimeError: Tensor.__contains__
   only supports Tensor or scalar``) because ``hyp.timestamp`` is a raw
   per-token frame-index ``Tensor`` instead of the expected structured dict.
   With ``timestamps=False`` requested at *both* the top-level kwarg and
   inside ``override_config`` (both are required -- the decoding-strategy
   reset that would normally clear stale timestamp state is gated on the
   top-level kwarg alone, evaluated before ``override_config`` is even
   consulted), the crash is avoided but ``hyp.timestamp`` still comes back
   as that same raw frame-index ``Tensor``, verified directly against the
   checkpoint -- not None, not seconds, not the structured dict. Treating
   those frame indices as seconds would silently produce garbled word
   timing, so this adapter explicitly discards ``hyp.timestamp`` on the
   forced-language path and relies on :func:`_tokens_from_hypothesis`'s
   evenly-spaced text fallback, trading true per-word timing for
   correctness (no crash, no garbled timing) until NeMo's timestamp
   post-processing is fixed upstream for this checkpoint class. The
   auto-detection path (no ``language`` requested) is unaffected and keeps
   true timestamps -- though it has not been exercised against a real
   checkpoint this session (this Prompt checkpoint has no ``"auto"``
   dictionary entry and cannot use it at all).

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


def _resolve_override_config_type(model: Any) -> type | None:
    """Discover a model's ``transcribe(override_config=...)`` dataclass type.

    NeMo checkpoints with prompt conditioning declare their own
    ``TranscribeConfig`` subclass (e.g. ``HybridRNNTCTCPromptTranscribeConfig``)
    as the ``override_config`` parameter's type annotation on their
    ``transcribe()`` override. Discovering it via introspection -- rather
    than importing one specific class by name -- lets this adapter drive any
    such subclass without hard-coding to
    ``EncDecHybridRNNTCTCBPEModelWithPrompt``.

    Returns:
        The dataclass type, or None if ``transcribe`` has no typed
        ``override_config`` parameter (e.g. a plain non-prompt ASR model, or
        a test double without one).

    """
    import typing

    # get_type_hints (not inspect.signature(...).annotation) is required:
    # both this project's and NeMo's modules use
    # `from __future__ import annotations` (PEP 563), which stores
    # annotations as unevaluated strings -- inspect.signature would hand
    # back the literal string "Optional[HybridRNNTCTCPromptTranscribeConfig]"
    # instead of the type. get_type_hints resolves it against the function's
    # own module globals regardless of which annotation style is in effect.
    try:
        hints = typing.get_type_hints(model.transcribe)
    except (NameError, TypeError, AttributeError):
        return None
    annotation = hints.get("override_config")
    if annotation is None:
        return None
    for candidate in typing.get_args(annotation) or (annotation,):
        if candidate is not type(None) and isinstance(candidate, type):
            return candidate
    return None


def _build_forced_language_config(model: Any, target_lang: str) -> Any:
    """Build the ``override_config`` needed to reliably force ``target_lang``.

    See the module docstring's two NeMo-quirk caveats: ``use_lhotse=False``
    is required for ``target_lang`` to reach the model at all;
    ``timestamps=False`` works around a downstream crash on this code path.
    ``num_workers=0`` avoids a ``DataLoader`` multiprocessing crash
    (``OSError: AF_UNIX path too long``) that surfaces whenever the resolved
    working/temp directory path is long enough to overflow the platform's
    Unix-socket path limit -- deterministic given a fixed checkout path, not
    caller-input-dependent, so it is always disabled rather than detected.

    Raises:
        ValueError: If the model exposes no typed ``override_config``
            parameter, so a caller-supplied ``language`` cannot be forced
            reliably. This should not happen for a checkpoint with a prompt
            dictionary; surfacing loudly matches this adapter's no-silent-
            fallback policy for language handling.

    """
    config_cls = _resolve_override_config_type(model)
    if config_cls is None:
        msg = (
            "This checkpoint's transcribe() has no typed override_config "
            "parameter, so language forcing cannot be driven reliably. This "
            "should not happen for a checkpoint with a prompt dictionary -- "
            "please check the checkpoint's model class."
        )
        raise ValueError(msg)
    return config_cls(
        batch_size=1,
        return_hypotheses=True,
        timestamps=False,
        verbose=False,
        target_lang=target_lang,
        use_lhotse=False,
        num_workers=0,
    )


def _tokens_from_hypothesis(hyp, tokenizer, *, span_end: float) -> list[TranscriptToken]:
    """Convert one NeMo RNNT hypothesis into word-level TranscriptTokens.

    ``hyp.timestamp`` carries one emission time (in seconds) per predicted
    subword token (SentencePiece pieces using the ``▁`` word-start marker),
    so words are reconstructed with the same grouping as the onnx-asr
    converter: a word's ``start`` is its first subword's time, its ``end``
    the next word's start (final word padded by ``_LAST_WORD_PAD``). No
    per-word probability source exists on the hypothesis, so ``probability``
    stays None rather than being manufactured. When the checkpoint emits no
    timestamps, timings are spread evenly over the clip span via the shared
    text fallback.

    A raw ``Tensor`` on ``hyp.timestamp`` (rather than None, a flat sequence
    of second-valued floats, or a structured dict) is treated as untrusted
    and discarded -- verified against a real checkpoint, this holds
    per-token *frame indices*, not seconds; converting the two would need
    the model's ``window_stride``/``subsampling_factor``, and this
    model-agnostic helper doesn't have access to either, so treating the raw
    values as seconds would silently produce garbled (not just imprecise)
    word timing. Falling back to evenly-spaced text timing is the safe
    choice, matching how callers already discard ``hyp.timestamp``
    proactively on paths known to return this shape (see
    :func:`NemoASRAdapter._transcribe`).
    """
    raw_timestamp = getattr(hyp, "timestamp", None)
    if hasattr(raw_timestamp, "numel"):  # torch.Tensor duck-type, no top-level torch import
        raw_timestamp = None
    timestamps = [float(t) for t in (raw_timestamp or [])]
    pieces: list[str] = []
    if timestamps:
        token_ids = getattr(hyp, "y_sequence", None)
        if token_ids is None:
            token_ids = getattr(hyp, "y", None)  # legacy/test-fixture attribute name
        token_ids = (
            list(token_ids.tolist()) if hasattr(token_ids, "tolist") else list(token_ids or ())
        )
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
                key, the checkpoint carries no prompt dictionary at all, or
                (forced-language calls only) the checkpoint's ``transcribe``
                exposes no typed ``override_config`` to force it through.

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

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, _SAMPLE_RATE, subtype="PCM_16")
            if target_lang is None:
                hypotheses = self._model.transcribe(
                    [path],
                    timestamps=True,
                    return_hypotheses=True,
                    batch_size=1,
                    verbose=False,
                )
            else:
                override_config = _build_forced_language_config(self._model, target_lang)
                # `timestamps` must ALSO be passed as a top-level kwarg here,
                # duplicating override_config.timestamps: transcribe()'s
                # decoding-strategy reset (compute_timestamps/preserve_align-
                # ments) is gated on the *top-level* `timestamps is not None`
                # check, evaluated before override_config is even consulted.
                # Without it, decoding keeps whatever compute_timestamps was
                # last configured to, and hyp.timestamp comes back as a raw
                # multi-element Tensor (verified against the real checkpoint).
                hypotheses = self._model.transcribe(
                    [path], timestamps=False, override_config=override_config
                )
                # Even with timestamps=False at both levels, hyp.timestamp
                # still comes back as that same raw per-token frame-index
                # Tensor rather than None -- verified directly. Discard it
                # explicitly (rather than relying solely on
                # _tokens_from_hypothesis's defensive Tensor guard) so the
                # intent is unambiguous at the call site: these values are
                # known-untrustworthy on this path, not merely "unusual
                # shape, handle defensively".
                for hyp in hypotheses or []:
                    hyp.timestamp = None
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
