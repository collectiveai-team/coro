"""TranscriptSpillStore on-disk round-trip, WAL mode, and cleanup."""

from __future__ import annotations

from pathlib import Path

from coro.core.models import RawWord, TranscriptToken
from coro.pipelines.transcript_store import TranscriptSpillStore


def _append(store, start, end, text, tokens=None):
    store.append_segment_tokens(
        tokens if tokens is not None else [TranscriptToken(start=start, end=end, text=text)],
        start=start,
        end=end,
        text=text,
    )


def test_store_round_trips_segment_tokens_in_order(tmp_path):
    tokens = [
        TranscriptToken(start=0.0, end=0.5, text=" hola", probability=0.9),
        TranscriptToken(start=0.5, end=1.0, text=" mundo.", probability=0.8),
    ]
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        _append(store, 0.0, 1.0, " hola mundo.", tokens)
        _append(store, 1.0, 2.0, " adios.")
        runs = list(store.iter_segment_tokens())

    assert runs[0] == tokens
    assert [t.text for run in runs for t in run] == [" hola", " mundo.", " adios."]
    assert runs[0][0].probability == 0.9


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
        _append(store, 0.0, 1.0, " a.")
        _append(store, 1.0, 2.0, " b.")
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
    _append(store, 0.0, 1.0, " a.")
    db_path = Path(store.path)
    assert db_path.exists()
    store.close()
    assert not db_path.exists()
    assert not Path(store.path + "-wal").exists()
    assert not Path(store.path + "-shm").exists()


def test_store_file_lands_in_requested_directory(tmp_path):
    with TranscriptSpillStore(directory=str(tmp_path)) as store:
        assert Path(store.path).parent == tmp_path
