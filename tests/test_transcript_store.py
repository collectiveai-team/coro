"""TranscriptSpillStore on-disk round-trip, WAL mode, and cleanup."""

from __future__ import annotations

from pathlib import Path

from coro.core.types import RawWord, ResponseSegment, TranscriptWord
from coro.pipelines.transcript_store import TranscriptSpillStore


def _segment(start, end, text, speaker, words=None):
    return ResponseSegment(start=start, end=end, text=text, speaker=speaker, words=words or [])


def test_store_round_trips_segments_in_order(tmp_path):
    word = TranscriptWord(word="hola.", start=0.0, end=1.0, score=1.0, speaker="1")
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        store.append_segment(_segment(0.0, 1.0, "hola.", "1", [word]))
        store.append_segment(_segment(1.0, 2.0, "mundo.", "2"))
        segments = list(store.iter_segments())

    assert [s.text for s in segments] == ["hola.", "mundo."]
    assert [s.speaker for s in segments] == ["1", "2"]
    assert segments[0].words == [word]
    assert segments[0].start == 0.0 and segments[1].end == 2.0


def test_store_round_trips_raw_words_in_order(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        store.append_raw_words(
            [
                RawWord(word=" hola", start=0.0, end=0.5, score=0.9),
                RawWord(word=" mundo", start=0.5, end=1.0, score=0.8),
            ]
        )
        store.append_raw_words([RawWord(word=" !", start=1.0, end=1.1, score=1.0)])
        words = list(store.iter_raw_words())

    assert [w.word for w in words] == [" hola", " mundo", " !"]
    assert words[0].score == 0.9
    assert store.raw_word_count == 3


def test_store_counts_track_appends(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        assert store.segment_count == 0
        store.append_segment(_segment(0.0, 1.0, "a.", "1"))
        store.append_segment(_segment(1.0, 2.0, "b.", "1"))
        assert store.segment_count == 2


def test_store_uses_wal_journal_mode(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    try:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        store.close()


def test_append_empty_raw_words_is_noop(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        store.append_raw_words([])
        assert store.raw_word_count == 0
        assert list(store.iter_raw_words()) == []


def test_close_deletes_database_and_sidecars(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    store.append_segment(_segment(0.0, 1.0, "a.", "1"))
    db_path = Path(store.path)
    assert db_path.exists()
    store.close()
    assert not db_path.exists()
    assert not Path(store.path + "-wal").exists()
    assert not Path(store.path + "-shm").exists()


def test_store_file_lands_in_requested_directory(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        assert Path(store.path).parent == tmp_path
