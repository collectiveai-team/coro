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
- ``segments(idx, start, end, text, tokens_json)`` — one finalized segment run
  per row; ``tokens_json`` holds that run's Project-Owned transcript tokens
  (bounded by run length), with their real timings and confidences. Tokens
  rather than response words are stored because speaker attribution — and the
  segment split that follows from it — happens at assembly, once the streaming
  diarizer has produced its complete timeline.
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
from dataclasses import asdict
from pathlib import Path

from coro.core.models import RawWord, TranscriptToken

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    idx INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    tokens_json TEXT NOT NULL
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

    def append_segment_tokens(
        self,
        tokens: list[TranscriptToken],
        *,
        start: float,
        end: float,
        text: str,
    ) -> None:
        """Persist one finalized segment run as its transcript tokens.

        Args:
            tokens: The run's Project-Owned transcript tokens, in order.
            start: Run start in seconds.
            end: Run end in seconds.
            text: The run's concatenated transcript text.

        """
        self._conn.execute(
            "INSERT INTO segments (idx, start, end, text, tokens_json) VALUES (?, ?, ?, ?, ?)",
            (
                self._segment_count,
                float(start),
                float(end),
                str(text),
                json.dumps([asdict(t) for t in tokens]),
            ),
        )
        self._segment_count += 1
        self._conn.commit()

    def append_raw_words(self, words: list[RawWord]) -> None:
        """Persist a batch of raw ASR words.

        Args:
            words: :class:`RawWord` items with ``word``, ``start``, ``end``
                and ``score``.

        """
        if not words:
            return
        rows = []
        for w in words:
            rows.append(
                (
                    self._raw_word_count,
                    str(w.word),
                    float(w.start),
                    float(w.end),
                    float(w.score),
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

    def iter_segment_tokens(self) -> Iterator[list[TranscriptToken]]:
        """Yield each finalized run's tokens in insertion order, streaming."""
        cursor = self._conn.execute("SELECT tokens_json FROM segments ORDER BY idx")
        for (tokens_json,) in cursor:
            yield [TranscriptToken(**t) for t in json.loads(tokens_json)]

    def iter_raw_words(self) -> Iterator[RawWord]:
        """Yield raw words in insertion order via a streaming cursor."""
        cursor = self._conn.execute("SELECT word, start, end, score FROM raw_words ORDER BY idx")
        for word, start, end, score in cursor:
            yield RawWord(word=word, start=start, end=end, score=score)

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
