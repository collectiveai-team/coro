"""StreamingTranscriptFinalizer grouping, online speakers, and batch parity."""

from __future__ import annotations

import pytest

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
    """Finalizer spills bare tokens; assembly attributes them from the timeline."""
    timeline = [
        SpeakerSegment(start=0.0, end=0.8, speaker=2),
        SpeakerSegment(start=0.8, end=2.4, speaker=3),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        # The store holds tokens only — no speaker is decided at spill time.
        assert [len(run) for run in store.iter_segment_tokens()] == [2, 2, 2]
        streamed = build_streaming_response(store, timeline)

    assert [s.speaker for s in streamed.segments] == ["2", "3", "3"]


def test_finalizer_flushes_unterminated_tail(tmp_path):
    """Tokens with no closing punctuation still finalize on finish()."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens([_tok(0.0, 0.4, " sin"), _tok(0.4, 0.8, " punto")])
        assert store.segment_count == 0  # nothing finalized yet
        finalizer.finish()
        streamed = build_streaming_response(store)

    assert len(streamed.segments) == 1
    assert streamed.segments[0].text == "sin punto"


def test_finalizer_word_timings_match_batch_across_the_overlap_clamp(tmp_path):
    """Both paths agree on word timings even where the clamp shortens a segment.

    Inherited from the streaming-correctness work, which asserted the last word
    ended exactly at the clamped segment end because words were *interpolated*
    over that span. Word timings are now the backend's own (ADR 0008), so a word
    may legitimately extend past its clamped segment end and the tiling
    assumption no longer holds. The invariant that still matters — and the one
    the test existed for — is that batch and streaming do not disagree.
    """
    tokens = [_tok(0.0, 1.5, " uno."), _tok(1.0, 2.0, " dos.")]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        streamed = build_streaming_response(store)

    batch = build_transcription_response(tokens, [], duration=2.0)
    assert streamed.word_segments == batch.word_segments
    assert streamed.segments == batch.segments
    # Real timings, not interpolated: the clamp moves the segment, not the word.
    first = streamed.segments[0]
    assert first.words[-1].end == pytest.approx(1.5, abs=1e-9)
    assert first.end == pytest.approx(1.0, abs=1e-9)


def test_finalizer_open_buffer_stays_bounded(tmp_path):
    """Open run never retains more than the current unterminated segment."""
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        max_open = 0
        for i in range(500):
            finalizer.add_tokens([_tok(i, i + 0.5, f" w{i}.")])
            max_open = max(max_open, len(finalizer.open_tokens))
        finalizer.finish()

    # Each batch is a single punctuation-terminated token, so the open run is
    # flushed every batch and never accumulates.
    assert max_open <= 1
    assert store.segment_count == 500


def test_finalizer_matches_batch_when_a_speaker_changes_mid_run(tmp_path):
    """Word-level splitting inside one segment run is identical in both paths.

    The turn is a clean two-way split with no punctuation support, and flicker
    correction leaves it alone because it is not sandwiched (see
    ``test_core_realignment.py``).
    """
    tokens = [
        _tok(0.0, 0.4, " hola"),
        _tok(0.4, 0.8, " mundo"),
        _tok(2.0, 2.4, " adios"),
        _tok(2.4, 2.8, " amigo."),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=2),
        SpeakerSegment(start=1.5, end=3.0, speaker=3),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        assert store.segment_count == 1  # one punctuation-bounded run
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(tokens, timeline, duration=3.0)
    assert streamed.segments == batch.segments
    assert streamed.word_segments == batch.word_segments
    assert streamed.diarization == batch.diarization
    assert [s.speaker for s in streamed.segments] == ["2", "3"]


def test_finalizer_matches_batch_for_overlapped_speech(tmp_path):
    timeline = [
        SpeakerSegment(start=0.0, end=2.4, speaker=2),
        SpeakerSegment(start=0.6, end=2.4, speaker=3),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(_TOKENS)
        finalizer.finish()
        streamed = build_streaming_response(store, timeline)

    batch = build_transcription_response(_TOKENS, timeline, duration=2.4)
    assert streamed.segments == batch.segments
    assert any(s.overlap for s in streamed.segments)


def test_finalizer_matches_batch_with_real_word_timings(tmp_path):
    """Uneven per-word timings and confidences survive the spill round-trip."""
    tokens = [
        _tok(0.0, 0.2, " muy", prob=0.9),
        _tok(3.0, 4.0, " tarde.", prob=0.4),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        finalizer = StreamingTranscriptFinalizer(store)
        finalizer.add_tokens(tokens)
        finalizer.finish()
        streamed = build_streaming_response(store)

    batch = build_transcription_response(tokens, [], duration=4.0)
    assert streamed.word_segments == batch.word_segments
    assert [(w.start, w.end, w.score) for w in streamed.word_segments] == [
        (0.0, 0.2, 0.9),
        (3.0, 4.0, 0.4),
    ]


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
