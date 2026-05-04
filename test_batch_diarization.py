#!/usr/bin/env python3
"""Regression checks for batch diarization helper behavior."""

import asyncio
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakeOnlineDiarization:
    def __init__(self):
        self.audio = np.array([], dtype=np.float32)
        self.calls = 0

    def insert_audio_chunk(self, chunk):
        self.audio = np.concatenate([self.audio, chunk.copy()])

    async def diarize(self):
        self.calls += 1
        duration = len(self.audio) / 16000
        if duration < 1.0:
            return []
        return [
            SimpleNamespace(start=0.0, end=min(0.5, duration), speaker=0),
            SimpleNamespace(start=0.5, end=min(1.0, duration), speaker=1),
        ]

    def close(self):
        pass


class BatchDiarizationTests(unittest.TestCase):
    def test_batch_diarize_returns_one_indexed_speaker_timeline(self):
        original_argv = sys.argv[:]
        sys.argv = ["custom_server.py"]
        try:
            import custom_server
        finally:
            sys.argv = original_argv

        fake_engine = SimpleNamespace(
            args=SimpleNamespace(diarization=True, diarization_backend="sortformer"),
            config=SimpleNamespace(diarization=True),
            diarization_model=object(),
        )
        pcm = (np.zeros(16000, dtype=np.int16)).tobytes()

        with patch.object(custom_server, "transcription_engine", fake_engine), patch(
            "whisperlivekit.core.online_diarization_factory",
            return_value=FakeOnlineDiarization(),
        ):
            diarization = asyncio.run(custom_server._batch_diarize(pcm))

        self.assertEqual(
            diarization,
            [
                {"start": 0.0, "end": 0.5, "speaker": 1},
                {"start": 0.5, "end": 1.0, "speaker": 2},
            ],
        )

    def test_clamps_batch_segments_to_avoid_adjacent_overlaps(self):
        original_argv = sys.argv[:]
        sys.argv = ["custom_server.py"]
        try:
            import custom_server
        finally:
            sys.argv = original_argv

        segments = [
            custom_server.Segment(start=0.0, end=10.0, text="first", speaker=1),
            custom_server.Segment(start=9.0, end=12.0, text="second", speaker=2),
        ]

        clamped = custom_server._clamp_segment_overlaps(segments)

        self.assertEqual(clamped[0].end, 9.0)
        self.assertEqual(clamped[1].start, 9.0)

    def test_batch_backed_sse_emits_deltas_and_final_whisperx_json(self):
        original_argv = sys.argv[:]
        sys.argv = ["custom_server.py"]
        try:
            import custom_server
        finally:
            sys.argv = original_argv

        async def collect_events():
            pcm = (np.zeros(16000, dtype=np.int16)).tobytes()
            token_batches = [
                [custom_server.ASRToken(start=0.0, end=0.4, text="hello ", probability=0.9)],
                [custom_server.ASRToken(start=0.5, end=0.9, text="world", probability=0.8)],
            ]
            diarization = [{"start": 0.0, "end": 1.0, "speaker": 1}]

            with patch.object(custom_server, "_iter_batch_transcribe_chunks", return_value=iter(token_batches)), patch.object(
                custom_server,
                "_batch_diarize",
                return_value=diarization,
            ):
                return [event async for event in custom_server._stream_batch_transcription(pcm, "en", 1.0)]

        events = asyncio.run(collect_events())
        payloads = [event.removeprefix("data: ").strip() for event in events]

        delta_events = [json.loads(payload) for payload in payloads if payload.startswith("{")]
        deltas = [event["delta"] for event in delta_events if event.get("type") == "transcript.text.delta"]
        done_events = [event for event in delta_events if event.get("type") == "transcript.text.done"]

        self.assertEqual(deltas, ["hello ", "world"])
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(len(done_events), 1)

        final = json.loads(done_events[0]["text"])
        self.assertEqual(final["segments"][0]["text"], "hello world")
        self.assertEqual(final["word_segments"][0]["word"], "hello")
        self.assertEqual(final["word_segments"][1]["word"], "world")


if __name__ == "__main__":
    unittest.main()
