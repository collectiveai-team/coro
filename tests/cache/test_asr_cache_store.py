"""The ASR window cache store round-trips, survives a kill, and stays bounded.

The store is the only place cache state outlives a process, so its retention
rules are the difference between a useful cache and a volume that fills up. All
of it is asserted through the store's own surface — what goes in comes back,
what expires does not, what overflows is evicted oldest-first — rather than
through row layout, which is free to change.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from coro.cache.store import ASRCacheStore
from coro.core.models import TranscriptToken


def _tokens(text: str) -> list[TranscriptToken]:
    return [TranscriptToken(start=0.0, end=1.0, text=text, probability=0.5)]


def _store(tmp_path, *, max_bytes: int = 0, ttl_seconds: float = 0.0) -> ASRCacheStore:
    return ASRCacheStore(str(tmp_path / "cache"), max_bytes=max_bytes, ttl_seconds=ttl_seconds)


def test_a_stored_window_round_trips(tmp_path):
    with _store(tmp_path) as store:
        store.put("key-a", _tokens(" hola"))
        assert store.get("key-a") == _tokens(" hola")


def test_an_absent_window_is_a_miss(tmp_path):
    with _store(tmp_path) as store:
        assert store.get("never-written") is None


def test_a_none_probability_round_trips_as_none(tmp_path):
    """A backend that expresses no confidence must not gain a stand-in one."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" x", probability=None)]
    with _store(tmp_path) as store:
        store.put("key", tokens)
        assert store.get("key") == tokens


def test_each_window_is_committed_as_it_completes(tmp_path):
    """A process killed mid-run must lose at most the window in flight.

    Reading the database from a second connection that never saw the writes is
    the observable form of "committed": an uncommitted row would be invisible.
    """
    store = _store(tmp_path)
    store.put("key-a", _tokens(" one"))
    store.put("key-b", _tokens(" two"))

    observer = sqlite3.connect(store.path)
    try:
        (count,) = observer.execute("SELECT COUNT(*) FROM windows").fetchone()
    finally:
        observer.close()
        store.close()

    assert count == 2


def test_entries_survive_reopening_the_store(tmp_path):
    """The cache is only worth anything if it outlives the process that filled it."""
    with _store(tmp_path) as store:
        store.put("key-a", _tokens(" persisted"))

    with _store(tmp_path) as reopened:
        assert reopened.get("key-a") == _tokens(" persisted")


def test_an_expired_entry_is_a_miss(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, ttl_seconds=60.0) as store:
        store.put("key-a", _tokens(" stale"))
        clock["now"] += 61.0
        assert store.get("key-a") is None


def test_an_expired_entry_is_removed_on_lookup(tmp_path, monkeypatch):
    """Expiry is applied lazily, so lookup is also what reclaims the row."""
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, ttl_seconds=60.0) as store:
        store.put("key-a", _tokens(" stale"))
        clock["now"] += 61.0
        store.get("key-a")
        assert store.entry_count() == 0


def test_an_unexpired_entry_still_hits(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, ttl_seconds=60.0) as store:
        store.put("key-a", _tokens(" fresh"))
        clock["now"] += 59.0
        assert store.get("key-a") == _tokens(" fresh")


def test_a_zero_ttl_never_expires(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, ttl_seconds=0.0) as store:
        store.put("key-a", _tokens(" eternal"))
        clock["now"] += 10_000_000.0
        assert store.get("key-a") == _tokens(" eternal")


def test_purge_expired_reports_what_it_removed(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, ttl_seconds=60.0) as store:
        store.put("old", _tokens(" old"))
        clock["now"] += 61.0
        store.put("new", _tokens(" new"))
        assert store.purge_expired() == 1


def test_the_size_cap_bounds_the_store(tmp_path):
    """A cap that only warned would still let the cache fill the volume."""
    with _store(tmp_path, max_bytes=400) as store:
        for index in range(40):
            store.put(f"key-{index:02d}", _tokens(f" window {index}"))
        assert store.total_bytes() <= 400


def test_eviction_removes_the_least_recently_accessed_entry(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, max_bytes=180) as store:
        store.put("first", _tokens(" first"))
        clock["now"] += 1.0
        store.put("second", _tokens(" second"))
        clock["now"] += 1.0
        # Writing a third entry pushes the store over the cap; "first" is the
        # coldest and must be the one that goes.
        store.put("third", _tokens(" third"))

        assert store.get("first") is None
        assert store.get("third") == _tokens(" third")


def test_reading_an_entry_protects_it_from_eviction(tmp_path, monkeypatch):
    """An entry in active use must not be evicted from underneath its run."""
    clock = {"now": 1_000.0}
    monkeypatch.setattr("coro.cache.store.time.time", lambda: clock["now"])
    with _store(tmp_path, max_bytes=180) as store:
        store.put("first", _tokens(" first"))
        clock["now"] += 1.0
        store.put("second", _tokens(" second"))
        clock["now"] += 1.0
        # Touching "first" makes "second" the coldest entry instead.
        store.get("first")
        clock["now"] += 1.0
        store.put("third", _tokens(" third"))

        assert store.get("first") == _tokens(" first")
        assert store.get("second") is None


def test_a_zero_cap_disables_eviction(tmp_path):
    with _store(tmp_path, max_bytes=0) as store:
        for index in range(40):
            store.put(f"key-{index:02d}", _tokens(f" window {index}"))
        assert store.entry_count() == 40


def test_rewriting_a_key_replaces_rather_than_duplicates(tmp_path):
    with _store(tmp_path) as store:
        store.put("key-a", _tokens(" first"))
        store.put("key-a", _tokens(" second"))
        assert store.entry_count() == 1
        assert store.get("key-a") == _tokens(" second")


def test_concurrent_writers_all_land(tmp_path):
    """The store is shared across worker threads, so it must tolerate them."""
    with _store(tmp_path) as store:

        def write(index: int) -> None:
            for repeat in range(10):
                store.put(f"key-{index}-{repeat}", _tokens(f" w{index}-{repeat}"))

        threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert store.entry_count() == 80


def test_concurrent_readers_and_writers_agree(tmp_path):
    with _store(tmp_path) as store:
        store.put("shared", _tokens(" shared"))
        seen: list[list[TranscriptToken] | None] = []
        lock = threading.Lock()

        def read() -> None:
            value = store.get("shared")
            with lock:
                seen.append(value)

        def write(index: int) -> None:
            store.put(f"other-{index}", _tokens(f" other{index}"))

        threads = [threading.Thread(target=read) for _ in range(8)]
        threads += [threading.Thread(target=write, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen == [_tokens(" shared")] * 8


@pytest.mark.parametrize("tokens", [[], _tokens(" one")])
def test_an_empty_result_is_a_hit_not_a_miss(tmp_path, tokens):
    """Silence legitimately transcribes to nothing; that must not re-run the model."""
    with _store(tmp_path) as store:
        store.put("key", tokens)
        assert store.get("key") == tokens
