"""On-disk transcript spill store for flat-memory streaming.

During a long streaming transcription the finalized segments and raw words
would, if held in Python lists, grow O(audio length) and inflate host RSS.
This store spills them to a per-request SQLite database in WAL mode so the
process keeps only SQLite's bounded page cache resident, while the full
transcript remains queryable to assemble the final response.

The database MUST live on real disk: on this platform ``/tmp`` is tmpfs
(RAM-backed), so spilling there would not reduce RSS.  Callers pass an
explicit ``directory`` on persistent storage; the default falls back to the
system temp dir only for convenience in tests.

Schema:
- ``segments(idx, start, end, text, speaker, words_json)`` — one finalized,
  speaker-attributed segment per row; ``words_json`` holds that segment's
  interpolated word dicts (bounded by segment length).
- ``raw_words(idx, word, start, end, score)`` — one ASR token per row.

Rows are read back with a streaming cursor so iteration never materialises
the whole transcript in memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    idx INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT NOT NULL,
    words_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_words (
    idx INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    score REAL NOT NULL
);
"""


class TranscriptSpillStore:
    """Per-request SQLite WAL store for finalized segments and raw words."""

    def __init__(self, *, directory: str | None = None) -> None:
        """Open a fresh on-disk store.

        Args:
            directory: Persistent-storage directory for the database file.
                Defaults to the system temp dir (acceptable for tests only).

        """
        fd, path = tempfile.mkstemp(prefix="asr-transcript-", suffix=".sqlite3", dir=directory)
        # Close the descriptor; sqlite3 reopens the path by name.
        os.close(fd)
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Cap the page cache so resident memory stays bounded (~2 MB).
        self._conn.execute("PRAGMA cache_size=-2000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._segment_count = 0
        self._raw_word_count = 0

    @property
    def path(self) -> str:
        """Filesystem path of the backing database."""
        return self._path

    def append_segment(self, segment: dict) -> None:
        """Persist one finalized, speaker-attributed segment.

        Args:
            segment: Dict with ``start``, ``end``, ``text``, ``speaker`` and
                a ``words`` list of word dicts.

        """
        self._conn.execute(
            "INSERT INTO segments (idx, start, end, text, speaker, words_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._segment_count,
                float(segment["start"]),
                float(segment["end"]),
                str(segment["text"]),
                str(segment["speaker"]),
                json.dumps(segment.get("words", [])),
            ),
        )
        self._segment_count += 1
        self._conn.commit()

    def append_raw_words(self, words: list[dict]) -> None:
        """Persist a batch of raw ASR word dicts.

        Args:
            words: Dicts with ``word``, ``start``, ``end`` and ``score`` keys.

        """
        if not words:
            return
        rows = []
        for w in words:
            rows.append(
                (
                    self._raw_word_count,
                    str(w["word"]),
                    float(w["start"]),
                    float(w["end"]),
                    float(w["score"]),
                )
            )
            self._raw_word_count += 1
        self._conn.executemany(
            "INSERT INTO raw_words (idx, word, start, end, score) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    @property
    def segment_count(self) -> int:
        """Number of finalized segments persisted so far."""
        return self._segment_count

    @property
    def raw_word_count(self) -> int:
        """Number of raw words persisted so far."""
        return self._raw_word_count

    def iter_segments(self) -> Iterator[dict]:
        """Yield finalized segments in insertion order via a streaming cursor."""
        cursor = self._conn.execute(
            "SELECT start, end, text, speaker, words_json FROM segments ORDER BY idx"
        )
        for start, end, text, speaker, words_json in cursor:
            yield {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "words": json.loads(words_json),
            }

    def iter_raw_words(self) -> Iterator[dict]:
        """Yield raw words in insertion order via a streaming cursor."""
        cursor = self._conn.execute("SELECT word, start, end, score FROM raw_words ORDER BY idx")
        for word, start, end, score in cursor:
            yield {"word": word, "start": start, "end": end, "score": score}

    def close(self) -> None:
        """Close the connection and delete the database and its WAL sidecars."""
        with contextlib.suppress(Exception):
            self._conn.close()
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                Path(self._path + suffix).unlink()

    def __enter__(self) -> TranscriptSpillStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
