"""StreamingTranscriptFinalizer grouping, online speakers, and batch parity."""

from __future__ import annotations

from asr_diar_server.core.response import build_transcription_response
from asr_diar_server.core.types import SpeakerSegment, TranscriptToken
from asr_diar_server.pipelines.finalizer import (
    StreamingTranscriptFinalizer,
    build_streaming_response,
)
from asr_diar_server.pipelines.transcript_store import TranscriptSpillStore


def _tok(start, end, text, prob=1.0):
    return TranscriptToken(start=start, end=end, text=text, probability=prob)


# Three punctuation-bounded segments, strictly in order, no overlap.
_TOKENS = [
    _tok(0.0, 0.4, " hola"),
    _tok(0.4, 0.8, " mundo."),
    _tok(0.8, 1.2, " como"),
    _tok(1.2, 1.6, " estas?"),
    _tok(1.6, 2.0, " bien"),
    _tok(2.0, 2.4, " gracias."),
]


def test_finalizer_matches_batch_builder_without_diarization(tmp_path):
    """Streaming assembly equals build_transcription_response for in-order input."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        # Feed tokens in two batches to exercise cross-batch open runs.
        finalizer.add_tokens(_TOKENS[:3])
        finalizer.add_tokens(_TOKENS[3:])
        finalizer.finish()
        streamed = build_streaming_response(store)

    batch = build_transcription_response(_TOKENS, [], duration=2.4)
    assert streamed["segments"] == batch["segments"]
    assert streamed["word_segments"] == batch["word_segments"]
    assert streamed["transcript"] == batch["transcript"]
    assert streamed["raw_words"] == batch["raw_words"]


def test_finalizer_emits_three_segments(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        assert store.segment_count == 3


def test_finalizer_applies_online_speaker_resolver(tmp_path):
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=2),
        SpeakerSegment(start=1.0, end=2.4, speaker=3),
    ]

    def resolver(start, end):
        # Pick the speaker whose span contains the segment midpoint.
        mid = (start + end) / 2
        for s in timeline:
            if s.start <= mid < s.end:
                return s.speaker
        return 1

    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store, speaker_resolver=resolver)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        speakers = [s["speaker"] for s in store.iter_segments()]

    assert speakers == ["2", "3", "3"]


def test_finalizer_flushes_unterminated_tail(tmp_path):
    """Tokens with no closing punctuation still finalize on finish()."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens([_tok(0.0, 0.4, " sin"), _tok(0.4, 0.8, " punto")])
        assert store.segment_count == 0  # nothing finalized yet
        finalizer.finish()
        segments = list(store.iter_segments())

    assert len(segments) == 1
    assert segments[0]["text"] == "sin punto"


def test_finalizer_open_buffer_stays_bounded(tmp_path):
    """Open run never retains more than the current unterminated segment."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        max_open = 0
        for i in range(500):
            finalizer.add_tokens([_tok(i, i + 0.5, f" w{i}.")])
            max_open = max(max_open, len(finalizer._open))
        finalizer.finish()

    # Each batch is a single punctuation-terminated token, so the open run is
    # flushed every batch and never accumulates.
    assert max_open <= 1
    assert store.segment_count == 500


def test_finalizer_clamps_overlapping_segments(tmp_path):
    """iter_response_segments clamps an earlier segment's end to the next start."""
    tokens = [
        _tok(0.0, 1.5, " uno."),
        _tok(1.0, 2.0, " dos."),  # starts before previous ended
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        streamed = build_streaming_response(store)

    assert streamed["segments"][0]["end"] <= streamed["segments"][1]["start"]
