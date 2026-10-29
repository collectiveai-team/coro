"""Streaming Pipeline driven through in-memory ASR and Streaming Diarizer adapters.

The upload is a real WAV that the pipeline decodes with ffmpeg, so nothing in
the pipeline is patched. Each test calls ``transcribe`` or ``stream`` and checks
the result, the events, what the adapters received, or that the spooled upload
file is gone from disk.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from coro.audio import AudioInput
from coro.core.models import (
    SpeakerSegment,
    TranscriptDeltaEvent,
    TranscriptToken,
)
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing
from support.adapters import RecordingASR, RecordingStreamingDiarizerFactory
from support.done_frame import render_done_frame
from support.factories import make_wav
from support.windowing import BYTES_PER_SECOND

RESPONSE_KEYS = {
    "segments",
    "word_segments",
    "transcript",
    "diarization",
    "raw_words",
    "detected_language",
}
UPLOAD_SECONDS = 2.5
UPLOAD_PCM_BYTES = int(UPLOAD_SECONDS * BYTES_PER_SECOND)
HELLO = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]


def upload() -> AudioInput:
    """Return a 2.5 s silent WAV upload."""
    return AudioInput(make_wav(frames=UPLOAD_PCM_BYTES // 2), filename="clip.wav")


async def stream_events(pipeline: StreamingPipeline, audio: AudioInput) -> list:
    """Drain ``pipeline.stream`` and release the spill store of its done frame."""
    events = [event async for event in pipeline.stream(audio)]
    for event in events:
        if isinstance(event, StreamingDoneFrame):
            render_done_frame(event)
    return events


@pytest.mark.asyncio
async def test_streaming_pipeline_returns_response_shape():
    result = await StreamingPipeline(asr=RecordingASR()).transcribe(upload())

    assert set(asdict(result)) == RESPONSE_KEYS


@pytest.mark.asyncio
async def test_streaming_pipeline_passes_prompt_to_asr():
    asr = RecordingASR()

    await StreamingPipeline(asr=asr).transcribe(upload(), prompt="mi prompt")

    assert asr.prompts[0] == "mi prompt"


@pytest.mark.asyncio
async def test_streaming_pipeline_passes_language_to_asr():
    asr = RecordingASR()

    await StreamingPipeline(asr=asr).transcribe(upload(), language="es")

    assert set(asr.languages) == {"es"}


@pytest.mark.asyncio
async def test_streaming_pipeline_uses_diarization_when_provided():
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    factory = RecordingStreamingDiarizerFactory([SpeakerSegment(start=0.0, end=2.0, speaker=2)])
    pipeline = StreamingPipeline(asr=RecordingASR(tokens), streaming_diarizer_factory=factory)

    result = await pipeline.transcribe(upload())

    assert result.segments[0].speaker == "2"


@pytest.mark.asyncio
async def test_streaming_pipeline_empty_tokens_no_crash():
    result = await StreamingPipeline(asr=RecordingASR([])).transcribe(upload())

    assert (result.segments, result.raw_words) == ([], [])


@pytest.mark.asyncio
async def test_stream_emits_delta_events():
    events = await stream_events(StreamingPipeline(asr=RecordingASR(HELLO)), upload())

    deltas = [e.delta for e in events if isinstance(e, TranscriptDeltaEvent)]
    assert deltas[0] == "hello."


@pytest.mark.asyncio
async def test_stream_emits_exactly_one_done_frame():
    events = [e async for e in StreamingPipeline(asr=RecordingASR(HELLO)).stream(upload())]

    done_frames = [e for e in events if isinstance(e, StreamingDoneFrame)]
    assert len(done_frames) == 1
    assert render_done_frame(done_frames[0]).keys() >= RESPONSE_KEYS


@pytest.mark.asyncio
async def test_stream_done_frame_comes_after_deltas():
    events = await stream_events(StreamingPipeline(asr=RecordingASR(HELLO)), upload())

    assert isinstance(events[0], TranscriptDeltaEvent)
    assert isinstance(events[-1], StreamingDoneFrame)


@pytest.mark.asyncio
async def test_stream_emits_no_progress_events():
    events = await stream_events(StreamingPipeline(asr=RecordingASR(HELLO)), upload())

    assert [e for e in events if "progress" in getattr(e, "type", "")] == []


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_success():
    audio = upload()
    spooled = Path(await audio.temp_path())

    await StreamingPipeline(asr=RecordingASR()).transcribe(audio)

    assert not spooled.exists()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_error():
    audio = upload()
    spooled = Path(await audio.temp_path())
    pipeline = StreamingPipeline(asr=RecordingASR(error=RuntimeError("ASR failed")))

    with pytest.raises(RuntimeError, match="ASR failed"):
        await pipeline.transcribe(audio)

    assert not spooled.exists()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_stream_cleans_up_temp_file_on_success():
    audio = upload()
    spooled = Path(await audio.temp_path())

    await stream_events(StreamingPipeline(asr=RecordingASR(HELLO)), audio)

    assert not spooled.exists()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_stream_cleans_up_temp_file_on_error():
    audio = upload()
    spooled = Path(await audio.temp_path())
    pipeline = StreamingPipeline(asr=RecordingASR(error=RuntimeError("ASR failed")))

    with pytest.raises(RuntimeError, match="ASR failed"):
        await stream_events(pipeline, audio)

    assert not spooled.exists()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_asr_never_called_with_more_than_window_bytes():
    asr = RecordingASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    await StreamingPipeline(asr=asr, windowing=windowing).transcribe(upload())

    assert len(asr.window_sizes) > 1
    assert max(asr.window_sizes) <= BYTES_PER_SECOND


@pytest.mark.asyncio
async def test_streaming_diarizer_ingests_the_whole_upload():
    factory = RecordingStreamingDiarizerFactory()

    await StreamingPipeline(asr=RecordingASR(), streaming_diarizer_factory=factory).transcribe(
        upload()
    )

    assert sum(factory.diarizers[0].chunk_sizes) == UPLOAD_PCM_BYTES


@pytest.mark.asyncio
async def test_no_full_pcm_accumulation_during_transcribe():
    factory = RecordingStreamingDiarizerFactory()

    await StreamingPipeline(asr=RecordingASR(), streaming_diarizer_factory=factory).transcribe(
        upload()
    )

    assert max(factory.diarizers[0].chunk_sizes) < UPLOAD_PCM_BYTES


@pytest.mark.asyncio
async def test_streaming_diarizer_factory_produces_one_instance_per_request():
    factory = RecordingStreamingDiarizerFactory()
    pipeline = StreamingPipeline(asr=RecordingASR(), streaming_diarizer_factory=factory)

    await pipeline.transcribe(upload())
    await pipeline.transcribe(upload())

    assert len(factory.diarizers) == 2


@pytest.mark.asyncio
async def test_streaming_diarizer_finalize_provides_timeline():
    tokens = [TranscriptToken(start=0.0, end=0.2, text=" hola.", probability=0.9)]
    factory = RecordingStreamingDiarizerFactory([SpeakerSegment(start=0.0, end=0.2, speaker=1)])
    pipeline = StreamingPipeline(asr=RecordingASR(tokens), streaming_diarizer_factory=factory)

    result = await pipeline.transcribe(upload())

    assert result.segments[0].speaker == "1"
