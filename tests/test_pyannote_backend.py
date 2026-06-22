"""pyannote diarization backend adapter using fake pipeline objects.

Tests verify that the PyannoteDiarizationAdapter:
- Converts 16-bit PCM bytes into a float32 ``(1, num_samples)`` waveform.
- Extracts the speaker-diarization Annotation from both result shapes
  (``output.speaker_diarization`` and a bare Annotation).
- Converts pyannote ``(segment, track, label)`` tracks into 1-indexed
  SpeakerSegment values clamped to the audio duration.
- Surfaces an informative error when the pipeline fails to load.

No real model inference or network access is performed. The two builder
tests that patch ``pyannote.audio.Pipeline`` are skipped when the optional
``diar-pyannote`` extra is not installed.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coro.backends.diarization.pyannote import (
    PyannoteDiarizationAdapter,
    _extract_annotation,
    _pcm_to_waveform,
    build_pyannote_diarization_adapter,
)
from coro.core.models import SpeakerSegment

# 1 second of silence at 16 kHz mono 16-bit.
_FAKE_PCM = struct.pack("<16000h", *([0] * 16000))


class _FakeAnnotation:
    """Minimal stand-in for a pyannote.core Annotation."""

    def __init__(self, tracks):
        # tracks: list of (start, end, label)
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for start, end, label in self._tracks:
            segment = SimpleNamespace(start=start, end=end)
            if yield_label:
                yield segment, "_track", label
            else:
                yield segment, "_track"


class _FakePipeline:
    """Callable stand-in that returns a community-1-style result object."""

    def __init__(self, annotation, *, bare=False):
        self._annotation = annotation
        self._bare = bare
        self.received = None

    def __call__(self, payload):
        self.received = payload
        if self._bare:
            return self._annotation
        return SimpleNamespace(speaker_diarization=self._annotation)

    def to(self, device):  # pragma: no cover - trivial
        return self


# ---------------------------------------------------------------------------
# _pcm_to_waveform
# ---------------------------------------------------------------------------


def test_pcm_to_waveform_shape_and_dtype():
    waveform = _pcm_to_waveform(_FAKE_PCM)
    assert tuple(waveform.shape) == (1, 16000)
    assert str(waveform.dtype) == "torch.float32"


def test_pcm_to_waveform_normalizes_full_scale():
    pcm = struct.pack("<2h", 32767, -32768)
    waveform = _pcm_to_waveform(pcm)
    assert waveform[0, 0].item() == pytest.approx(32767 / 32768.0, abs=1e-6)
    assert waveform[0, 1].item() == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _extract_annotation
# ---------------------------------------------------------------------------


def test_extract_annotation_from_result_object():
    annotation = _FakeAnnotation([])
    output = SimpleNamespace(speaker_diarization=annotation)
    assert _extract_annotation(output) is annotation


def test_extract_annotation_from_bare_annotation():
    annotation = _FakeAnnotation([])
    assert _extract_annotation(annotation) is annotation


# ---------------------------------------------------------------------------
# PyannoteDiarizationAdapter.diarize_pcm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diarize_pcm_converts_tracks_to_segments():
    annotation = _FakeAnnotation([(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")])
    adapter = PyannoteDiarizationAdapter(_FakePipeline(annotation))
    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert all(isinstance(s, SpeakerSegment) for s in timeline)
    assert [s.speaker for s in timeline] == [1, 2]  # 0-indexed labels -> 1-indexed
    assert timeline[0].start == 0.0
    assert timeline[1].end == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_diarize_pcm_passes_waveform_payload():
    pipeline = _FakePipeline(_FakeAnnotation([]))
    adapter = PyannoteDiarizationAdapter(pipeline)
    await adapter.diarize_pcm(_FAKE_PCM)

    assert pipeline.received is not None
    assert pipeline.received["sample_rate"] == 16000
    assert tuple(pipeline.received["waveform"].shape) == (1, 16000)


@pytest.mark.asyncio
async def test_diarize_pcm_clamps_end_to_duration():
    # 1 second of audio but a track extending to 5s should be clamped.
    annotation = _FakeAnnotation([(0.0, 5.0, "SPEAKER_00")])
    adapter = PyannoteDiarizationAdapter(_FakePipeline(annotation))
    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert timeline[0].end <= 1.0


@pytest.mark.asyncio
async def test_diarize_pcm_supports_bare_annotation_output():
    annotation = _FakeAnnotation([(0.0, 1.0, "SPEAKER_03")])
    adapter = PyannoteDiarizationAdapter(_FakePipeline(annotation, bare=True))
    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert timeline[0].speaker == 4


# ---------------------------------------------------------------------------
# build_pyannote_diarization_adapter (requires the diar-pyannote extra)
# ---------------------------------------------------------------------------


def test_build_adapter_wraps_loaded_pipeline():
    pytest.importorskip("pyannote.audio")
    fake_pipeline = _FakePipeline(_FakeAnnotation([]))
    # The builder imports ``Pipeline`` lazily from pyannote.audio, so patching
    # ``from_pretrained`` on the real class is sufficient (no model download).
    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=fake_pipeline):
        adapter = build_pyannote_diarization_adapter(
            "pyannote/speaker-diarization-community-1",
            device="cpu",
            hf_token="tok",
        )
    assert isinstance(adapter, PyannoteDiarizationAdapter)


def test_build_adapter_raises_on_failed_load():
    pytest.importorskip("pyannote.audio")
    with (
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=None),
        pytest.raises(RuntimeError, match="HF_TOKEN"),
    ):
        build_pyannote_diarization_adapter(
            "pyannote/speaker-diarization-community-1",
            device="cpu",
            hf_token=None,
        )
