"""On-disk transcript spill store for flat-memory streaming.

During a long streaming transcription the finalized segments and raw words
would, if held in Python lists, grow O(audio length) and inflate host RSS.
This store spills them to a per-request SQLite database in WAL mode so the
process keeps only SQLite's bounded page cache resident, while the full
transcript remains queryable to assemble the final response.

The database MUST live on real disk: ``/tmp`` is tmpfs (RAM-backed) on most
Linux distributions, so spilling there would not reduce RSS.  Callers pass a
``directory`` already resolved by :func:`coro.pipelines.spill.resolve_spill_dir`,
which rejects RAM-backed paths at startup; the ``None`` default falls back to
the system temp dir only for convenience in tests.

Writes are committed in batches rather than per append: one synchronous flush
per ASR window plus one per token batch dominated the store's cost while buying
nothing, since the database is per-request and deleted on close.  Uncommitted
pages stay bounded by the connection's capped page cache.

Segments are stored as *raw* spans, before speaker attribution, overlap
clamping and word interpolation.  Those steps depend on data that does not
exist yet when a segment finalizes — the complete speaker timeline and the next
segment's start — so applying them at append time made the streamed response
disagree with the batch one.  Assembly applies them in the batch builder's
order instead.

Schema:
- ``segments(idx, start, end, text, tokens_json)`` — one finalized, unattributed
  segment run per row; ``tokens_json`` holds that run's Project-Owned transcript
  tokens (bounded by run length), with their real timings and confidences.
  Tokens rather than response words are stored because per-word speaker
  attribution happens at assembly, once the streaming diarizer has produced its
  complete timeline.
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

# Rows buffered before a commit. Sized so a commit costs far less than the ASR
# window that produced the rows, while staying well inside the page cache.
COMMIT_ROW_INTERVAL = 512


class TranscriptSpillStore:
    """Per-request SQLite WAL store for finalized segments and raw words."""

    def __init__(self, *, directory: str | None = None) -> None:
        """Open a fresh on-disk store.

        Args:
            directory: Persistent-storage directory for the database file,
                created if missing. Defaults to the system temp dir
                (acceptable for tests only).

        """
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
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
        self._uncommitted_rows = 0

    @property
    def path(self) -> str:
        """Filesystem path of the backing database."""
        return self._path

    @property
    def uncommitted_rows(self) -> int:
        """Rows appended since the last commit."""
        return self._uncommitted_rows

    def _record_rows(self, rows: int) -> None:
        """Count appended rows, committing once the batch interval is reached."""
        self._uncommitted_rows += rows
        if self._uncommitted_rows >= COMMIT_ROW_INTERVAL:
            self.flush()

    def flush(self) -> None:
        """Commit any buffered appends."""
        if self._uncommitted_rows:
            self._conn.commit()
            self._uncommitted_rows = 0

    def append_segment_tokens(
        self,
        tokens: list[TranscriptToken],
        *,
        start: float,
        end: float,
        text: str,
    ) -> None:
        """Persist one finalized segment run as its transcript tokens.

        The run's speaker and words are not stored: both are derived at
        assembly, once the speaker timeline and the following run's start are
        known.

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
        self._record_rows(1)

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
        self._record_rows(len(rows))

    @property
    def segment_count(self) -> int:
        """Number of finalized segments persisted so far."""
        return self._segment_count

    @property
    def raw_word_count(self) -> int:
        """Number of raw words persisted so far."""
        return self._raw_word_count

    def iter_segment_tokens(self) -> Iterator[list[TranscriptToken]]:
        """Yield each finalized run's tokens in insertion order via a streaming cursor."""
        self.flush()
        cursor = self._conn.execute("SELECT tokens_json FROM segments ORDER BY idx")
        for (tokens_json,) in cursor:
            yield [TranscriptToken(**t) for t in json.loads(tokens_json)]

    def iter_raw_words(self) -> Iterator[RawWord]:
        """Yield raw words in insertion order via a streaming cursor."""
        self.flush()
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
