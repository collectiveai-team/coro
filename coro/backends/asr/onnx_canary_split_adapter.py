"""ASRAdapter for the onnx-canary-split backend.

See ``coro/backends/asr/onnx_canary_split.py``'s module docstring for the
Adapter Concurrency Policy and the forced-language and auto-LID design.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_canary_split_language import (
    _partial_prompt_lid,
    resolve_canary_language,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coro.backends.asr.onnx_canary_split_language import CanaryLidModel

    import numpy.typing as npt

    from coro.core.models import TranscriptToken

_SAMPLE_RATE = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32


class SplitCanaryAsr(Protocol):
    """The split-decode ``NemoConformerAED`` surface the adapter calls."""

    language_token_ids: dict[str, int]

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object,
    ) -> Iterator[object]:
        """Recognize a waveform batch, one result per waveform."""
        ...


class OnnxCanarySplitASRAdapter:
    """ASRAdapter wrapping the split-decode Canary pipeline.

    Adapter Concurrency Policy: **concurrent** -- the split-decode K/V cache is
    thread-local and the ONNX Runtime sessions are shared and thread-safe; see
    the module docstring. Load is bounded by an :class:`AdmissionController`
    sized from ``asr_max_concurrency`` (0 = auto-size from the core count).
    """

    honours_prompt: bool = False
    """Canary's decoder has no carried-prompt input port (only `language`/`pnc`
    prefix tokens); this matches `onnx-asr`'s own Canary integration.
    """

    def __init__(
        self,
        asr: SplitCanaryAsr,
        *,
        admission: AdmissionController | None = None,
        fallback_language: str = "en",
    ) -> None:
        self._asr = asr
        # The ASR instance's vocab-derived `<|xx|>` map -- this checkpoint's
        # supported-language set, never a hardcoded list.
        self._language_tokens: dict[str, int] = asr.language_token_ids
        self._fallback_language = fallback_language
        self._admission = admission or build_admission_controller(
            max_concurrency=0, max_queue_depth=_DEFAULT_QUEUE_DEPTH
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    @property
    def supported_languages(self) -> frozenset[str]:
        """Language codes this checkpoint can decode with, derived from its vocab."""
        return frozenset(self._language_tokens)

    @property
    def fallback_language(self) -> str:
        """Language used when a request carries none (``asr_fallback_language``).

        Public so the ASR Window Cache decorator can key an absent request
        language the same way :meth:`resolve_language` decodes it (base
        subtag of this value), without the cache layer knowing anything else
        about this backend -- see ``coro/cache/adapter.py``.
        """
        return self._fallback_language

    def resolve_language(self, language: str | None) -> str:
        """Resolve a request language to a vocab-backed Canary code.

        A blank or absent request language resolves to the adapter's
        ``fallback_language`` (``asr_fallback_language`` via the factory), so
        the decode prefix is never left to onnx_asr's hardcoded ``<|en|>``.
        Detection (slice 05) will take precedence by passing a detected
        language explicitly -- explicit beats the fallback here.

        Raises:
            AsrUnsupportedLanguageError: If the requested language (or the
                fallback, when the request carries none) matches no ``<|xx|>``
                token in the loaded vocab.

        """
        requested = language if (language or "").strip() else self._fallback_language
        resolved = resolve_canary_language(requested, self._language_tokens)
        if resolved is None:  # blank request and blank fallback
            raise AsrUnsupportedLanguageError("", supported_languages=self._language_tokens)
        return resolved

    async def detect_language(self, pcm: bytes) -> str | None:
        """Detect one window's source language via the Canary2 partial-prompt LID probe.

        Not part of the :class:`~coro.core.protocols.ASRAdapter` protocol --
        the auto-LID sticky pipeline layer (``coro/pipelines/windowing.py``)
        duck-types this (``getattr(asr, "detect_language", None)``), so every
        other backend is unaffected.

        Returns:
            The base-subtag language code (e.g. ``"es"``) when the probe's
            second decoder step emitted one of this checkpoint's ``<|xx|>``
            language tokens, else ``None``. Cross-checked against
            ``self._language_tokens`` (the same vocab-derived set
            :meth:`resolve_language` uses) rather than trusting
            :func:`_partial_prompt_lid`'s own regex match in isolation.

        """

        def _detect() -> str | None:
            lid_model = cast("CanaryLidModel", self._asr)  # noqa: BTTOOL004 -- the split-decode class has these private NemoConformerAED internals; SplitCanaryAsr does not name them
            code = _partial_prompt_lid(lid_model, pcm)
            return code if code in self._language_tokens else None

        async with self._admission.admit():
            return await asyncio.to_thread(_detect)

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes with an optional forced language.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored --
            same as plain ``onnx-asr``'s Canary integration: no text input
            port for a carried prompt, only `language`/`target_language`/`pnc`
            prefix tokens.

        Raises:
            AsrCapacityError: If the admission queue is full.
            AsrUnsupportedLanguageError: If the language (or the fallback,
                when the request carries none) matches no ``<|xx|>`` vocab
                token. Raised at resolution time, before admission, so every
                request surface gets it for free.

        """
        resolved = self.resolve_language(language)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / _SAMPLE_RATE

        def _recognize() -> object:
            waveforms = audio[None, :]
            waveforms_len = np.array([len(audio)], dtype=np.int64)
            return next(
                iter(self._asr.recognize_batch(waveforms, waveforms_len, language=resolved))
            )

        async with self._admission.admit():
            result = await asyncio.to_thread(_recognize)
        # No per-token timestamps for this AED model (same as `onnx-asr`'s Canary
        # and Whisper integrations) -- `convert_onnx_asr_result` already falls
        # back to `words_from_text` when `result.timestamps` is None, which
        # `_decoding`'s `None` second yield element guarantees here too. Do not
        # regress or paper over that known limitation (see findings.md).
        return convert_onnx_asr_result(result, span_end=duration)
