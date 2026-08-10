"""Deepgram-native WebSocket /v1/listen streams live audio.

This is real streaming, not buffer-then-transcribe: audio pushed as binary
frames reaches the same ASRWindowing the Streaming Pipeline uses, and Results
frames come back as windows complete.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.settings import ServerSettings
from conftest import make_app

# One second of silence at the canonical rate.
ONE_SECOND = b"\x00\x00" * SAMPLE_RATE
# ASRWindowing emits a window at 30 s by default; send enough to force one.
WINDOW_SECONDS = 31


class _FakeASR:
    """Returns a fixed token batch for every window."""

    def __init__(self, tokens: list[TranscriptToken] | None = None) -> None:
        self._tokens = tokens or [
            TranscriptToken(start=0.0, end=0.5, text=" hola", probability=0.91),
            TranscriptToken(start=0.5, end=1.0, text=" mundo.", probability=0.83),
        ]
        self.calls = 0
        self.max_pcm = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.calls += 1
        self.max_pcm = max(self.max_pcm, len(pcm))
        return list(self._tokens)


class _FakeDiarizer:
    def __init__(self, timeline):
        self._timeline = timeline
        self.processed_chunks = 0

    def ingest_pcm_chunk(self, chunk):
        self.processed_chunks += 1

    def finalize(self):
        return list(self._timeline)


def _app(asr=None, *, timeline=None):
    application = make_app(pipeline=object(), settings=ServerSettings(_env_file=None))
    application.state.runtime.asr_adapter = asr or _FakeASR()
    if timeline is not None:
        application.state.runtime.streaming_diarizer_factory = lambda: _FakeDiarizer(timeline)
    return application


def _drain(ws) -> list[dict]:
    """Read frames until the closing Metadata frame."""
    frames: list[dict] = []
    while True:
        frame = json.loads(ws.receive_text())
        frames.append(frame)
        if frame.get("type") == "Metadata":
            return frames


def _stream(app, query: str = "", *, seconds: int = WINDOW_SECONDS) -> list[dict]:
    with TestClient(app).websocket_connect(f"/v1/listen{query}") as ws:
        for _ in range(seconds):
            ws.send_bytes(ONE_SECOND)
        ws.send_text(json.dumps({"type": "CloseStream"}))
        return _drain(ws)


class TestLiveResults:
    def test_streaming_audio_yields_results_frames(self):
        frames = _stream(_app())
        assert [f["type"] for f in frames if f["type"] == "Results"]

    def test_results_frames_carry_a_transcript(self):
        frames = _stream(_app())
        results = [f for f in frames if f["type"] == "Results"]
        assert results[0]["channel"]["alternatives"][0]["transcript"] == "hola mundo."

    def test_results_frames_carry_words_with_timings_and_confidence(self):
        results = [f for f in _stream(_app()) if f["type"] == "Results"]
        word = results[0]["channel"]["alternatives"][0]["words"][0]
        assert word["word"] == "hola"
        assert word["confidence"] == 0.91
        assert (word["start"], word["end"]) == (0.0, 0.5)

    def test_stream_ends_with_a_metadata_frame(self):
        frames = _stream(_app())
        assert frames[-1]["type"] == "Metadata"
        assert frames[-1]["request_id"]
        assert frames[-1]["channels"] == 1

    def test_metadata_reports_the_audio_duration_actually_ingested(self):
        frames = _stream(_app(), seconds=WINDOW_SECONDS)
        assert frames[-1]["duration"] == pytest.approx(WINDOW_SECONDS, abs=1.0)

    def test_audio_reaches_the_asr_incrementally_not_as_one_buffer(self):
        # A buffer-then-transcribe imitation would hand the ASR everything at
        # once; windowing must cap what any single call sees.
        asr = _FakeASR()
        _stream(_app(asr), seconds=WINDOW_SECONDS)
        assert asr.calls >= 1
        assert asr.max_pcm <= 30 * SAMPLE_RATE * BYTES_PER_SAMPLE

    def test_results_arrive_before_the_client_closes_the_stream(self):
        # The load-bearing test for this feature. A buffer-then-transcribe
        # implementation cannot pass it: it has nothing to send until the
        # client signals end of stream.
        with TestClient(_app()).websocket_connect("/v1/listen") as ws:
            for _ in range(WINDOW_SECONDS):
                ws.send_bytes(ONE_SECOND)
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "Results"
            ws.send_text(json.dumps({"type": "CloseStream"}))
            _drain(ws)

    def test_a_second_window_produces_a_second_results_frame(self):
        with TestClient(_app()).websocket_connect("/v1/listen") as ws:
            for _ in range(WINDOW_SECONDS * 2):
                ws.send_bytes(ONE_SECOND)
            first = json.loads(ws.receive_text())
            second = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "CloseStream"}))
            _drain(ws)
        assert first["type"] == second["type"] == "Results"


class TestControlFrames:
    def test_close_stream_terminates_the_session(self):
        frames = _stream(_app(), seconds=WINDOW_SECONDS)
        assert frames[-1]["type"] == "Metadata"

    def test_keepalive_does_not_end_the_stream(self):
        app = _app()
        with TestClient(app).websocket_connect("/v1/listen") as ws:
            ws.send_text(json.dumps({"type": "KeepAlive"}))
            for _ in range(WINDOW_SECONDS):
                ws.send_bytes(ONE_SECOND)
            ws.send_text(json.dumps({"type": "CloseStream"}))
            frames = _drain(ws)
        assert any(f["type"] == "Results" for f in frames)

    def test_finalize_closes_the_stream(self):
        app = _app()
        with TestClient(app).websocket_connect("/v1/listen") as ws:
            for _ in range(WINDOW_SECONDS):
                ws.send_bytes(ONE_SECOND)
            ws.send_text(json.dumps({"type": "Finalize"}))
            frames = _drain(ws)
        assert frames[-1]["type"] == "Metadata"

    def test_unparseable_control_frame_does_not_kill_the_socket(self):
        app = _app()
        with TestClient(app).websocket_connect("/v1/listen") as ws:
            ws.send_text("not json")
            for _ in range(WINDOW_SECONDS):
                ws.send_bytes(ONE_SECOND)
            ws.send_text(json.dumps({"type": "CloseStream"}))
            frames = _drain(ws)
        assert frames[-1]["type"] == "Metadata"


class TestDiarization:
    def test_diarize_true_emits_a_final_frame_with_per_word_speakers(self):
        timeline = [SpeakerSegment(start=0.0, end=60.0, speaker=2)]
        frames = _stream(_app(timeline=timeline), "?diarize=true")
        results = [f for f in frames if f["type"] == "Results"]
        speakers = [w.get("speaker") for w in results[-1]["channel"]["alternatives"][0]["words"]]
        assert speakers and set(speakers) == {2}

    def test_interim_frames_carry_no_speaker(self):
        # The timeline is incomplete while audio is still arriving; a label
        # there would be a guess a later frame contradicts.
        timeline = [SpeakerSegment(start=0.0, end=60.0, speaker=2)]
        frames = _stream(_app(timeline=timeline), "?diarize=true")
        results = [f for f in frames if f["type"] == "Results"]
        first_words = results[0]["channel"]["alternatives"][0]["words"]
        assert all("speaker" not in w for w in first_words)

    def test_without_diarize_no_frame_carries_a_speaker(self):
        timeline = [SpeakerSegment(start=0.0, end=60.0, speaker=2)]
        raw = json.dumps(_stream(_app(timeline=timeline)))
        assert '"speaker"' not in raw


class TestAudioFormatDeclaration:
    @pytest.mark.parametrize(
        "query", ["?encoding=mulaw", "?encoding=opus", "?encoding=flac", "?sample_rate=999"]
    )
    def test_unsupported_declaration_is_refused_before_audio(self, query: str):
        with TestClient(_app()).websocket_connect(f"/v1/listen{query}") as ws:
            frame = json.loads(ws.receive_text())
        assert frame["type"] == "Error"
        assert "not supported" in frame["message"] or "range" in frame["message"]

    def test_multichannel_audio_is_refused(self):
        with TestClient(_app()).websocket_connect("/v1/listen?channels=2") as ws:
            frame = json.loads(ws.receive_text())
        assert frame["type"] == "Error"

    def test_linear16_is_accepted(self):
        frames = _stream(_app(), "?encoding=linear16&sample_rate=16000")
        assert frames[-1]["type"] == "Metadata"

    def test_non_canonical_sample_rate_is_resampled(self):
        # 8 kHz in must still produce transcription; duration is reported in
        # canonical seconds after resampling.
        app = _app()
        with TestClient(app).websocket_connect(
            "/v1/listen?encoding=linear16&sample_rate=8000"
        ) as ws:
            for _ in range(WINDOW_SECONDS * 2):
                ws.send_bytes(b"\x00\x00" * 8000)
            ws.send_text(json.dumps({"type": "CloseStream"}))
            frames = _drain(ws)
        assert frames[-1]["type"] == "Metadata"
        assert frames[-1]["duration"] > 0


class TestReadiness:
    def test_missing_asr_adapter_is_reported_and_closed(self):
        app = make_app(pipeline=object(), settings=ServerSettings(_env_file=None))
        app.state.runtime.asr_adapter = None
        with TestClient(app).websocket_connect("/v1/listen") as ws:
            frame = json.loads(ws.receive_text())
        assert frame["type"] == "Error"
        assert "not ready" in frame["description"].lower()
