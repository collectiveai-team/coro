"""StreamingTranscriptFinalizer grouping, online speakers, and batch parity."""

from __future__ import annotations

from coro.core.response import build_transcription_response
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.finalizer import (
    StreamingTranscriptFinalizer,
    build_streaming_response,
)
from coro.pipelines.transcript_store import TranscriptSpillStore


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
    assert streamed.segments == batch.segments
    assert streamed.word_segments == batch.word_segments
    assert streamed.transcript == batch.transcript
    assert streamed.raw_words == batch.raw_words


def test_finalizer_matches_batch_builder_with_diarization(tmp_path):
    """Deferred speaker assignment equals the batch global pass, in-order input."""
    timeline = [
        SpeakerSegment(start=0.0, end=0.8, speaker=2),
        SpeakerSegment(start=0.8, end=1.6, speaker=3),
        SpeakerSegment(start=1.6, end=2.4, speaker=2),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(_TOKENS, timeline, duration=2.4)
    assert streamed.segments == batch.segments
    assert streamed.word_segments == batch.word_segments
    assert streamed.transcript == batch.transcript
    assert streamed.diarization == batch.diarization
    assert streamed.raw_words == batch.raw_words


def test_speaker_boundary_split_matches_batch(tmp_path):
    """Both pipelines split one punctuation segment at the same turn change.

    The Full-Memory Pipeline splits during segmentation with the tokens in
    hand; the Streaming Pipeline splits at assembly against stored words. They
    must still emit identical segments.
    """
    # One punctuation-bounded segment, speaker changes after the third word.
    tokens = [
        _tok(0.0, 0.5, " uno"),
        _tok(0.5, 1.0, " dos"),
        _tok(1.0, 1.5, " tres"),
        _tok(1.5, 2.0, " cuatro"),
        _tok(2.0, 2.5, " cinco"),
        _tok(2.5, 3.0, " seis."),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=1.5, speaker=2),
        SpeakerSegment(start=1.5, end=3.0, speaker=3),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(tokens, timeline, duration=3.0)

    assert len(batch.segments) == 2, "the turn change must split the segment"
    assert [s.speaker for s in batch.segments] == ["2", "3"]
    assert [s.text for s in batch.segments] == ["uno dos tres", "cuatro cinco seis."]
    assert streamed.segments == batch.segments
    assert streamed.word_segments == batch.word_segments
    assert streamed.transcript == batch.transcript
    assert streamed.diarization == batch.diarization


def test_backchannel_does_not_split_either_pipeline(tmp_path):
    """A one-word interruption stays with the surrounding speaker, on both paths."""
    tokens = [
        _tok(0.0, 0.5, " uno"),
        _tok(0.5, 1.0, " dos"),
        _tok(1.0, 1.2, " mm"),
        _tok(1.2, 1.7, " tres"),
        _tok(1.7, 2.2, " cuatro."),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=2),
        SpeakerSegment(start=1.0, end=1.2, speaker=3),
        SpeakerSegment(start=1.2, end=2.2, speaker=2),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(tokens, timeline, duration=2.2)

    assert len(batch.segments) == 1
    assert streamed.segments == batch.segments


def test_persisted_segment_words_carry_measured_starts(tmp_path):
    """Words spilled before diarization exists hold real token times.

    The Streaming Pipeline commits a segment before the speaker timeline is
    known, so by assembly time the stored words are the only sub-segment
    structure a Speaker Boundary Split can cut on. Interpolated starts would
    put the cut in the wrong place.
    """
    tokens = [
        _tok(0.0, 0.4, " hola"),
        _tok(0.4, 8.0, " mundo"),
        _tok(8.0, 9.0, " otra."),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        stored = list(store.iter_segments())

    words = stored[0].words
    assert [w.word for w in words] == ["hola", "mundo", "otra."]
    assert [w.start for w in words] == [0.0, 0.4, 8.0]


def test_finalizer_marks_segments_beyond_timeline_unknown(tmp_path):
    """Segments past the diarization horizon get speaker -1, matching batch."""
    timeline = [SpeakerSegment(start=0.0, end=1.0, speaker=2)]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(_TOKENS, timeline, duration=2.4)
    assert [s.speaker for s in streamed.segments] == [s.speaker for s in batch.segments]
    assert "-1" in [s.speaker for s in streamed.segments]


def test_finalizer_emits_three_segments(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        assert store.segment_count == 3


def test_finalizer_defers_speaker_assignment_to_assembly(tmp_path):
    """Finalizer spills provisional speaker 1; assembly assigns from timeline."""
    timeline = [
        SpeakerSegment(start=0.0, end=0.8, speaker=2),
        SpeakerSegment(start=0.8, end=2.4, speaker=3),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        # Stored provisionally as speaker 1 before assembly.
        assert [s.speaker for s in store.iter_segments()] == ["1", "1", "1"]
        streamed = build_streaming_response(store, timeline)

    assert [s.speaker for s in streamed.segments] == ["2", "3", "3"]


def test_finalizer_flushes_unterminated_tail(tmp_path):
    """Tokens with no closing punctuation still finalize on finish()."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens([_tok(0.0, 0.4, " sin"), _tok(0.4, 0.8, " punto")])
        assert store.segment_count == 0  # nothing finalized yet
        finalizer.finish()
        segments = list(store.iter_segments())

    assert len(segments) == 1
    assert segments[0].text == "sin punto"


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

    assert streamed.segments[0].end <= streamed.segments[1].start
