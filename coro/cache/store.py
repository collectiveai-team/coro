"""Persistent SQLite store for ASR window results.

One shared, long-lived database, distinct from the per-request Transcript Spill
Store and its delete-on-close lifecycle. Only the key and the resulting
transcript tokens are stored — never audio and never decoded PCM — which keeps
the store on the order of a few megabytes per hour of audio.

Design points that are load-bearing:

- **One commit per window.** Batching would lose every window in the batch when
  a long run is killed part-way, which is exactly the case the cache is supposed
  to survive. A window costs seconds of inference, so a commit per window is
  free by comparison.
- **Reads stream.** Row bodies are small, but iteration never materialises the
  whole table.
- **Retention has two bounds.** A size cap with least-recently-used eviction is
  the real protection against unbounded growth; a time-to-live applied lazily on
  lookup stops stale entries outliving the run that produced them. Access time is
  refreshed on read so an entry in active use is never evicted from underneath a
  run.
- **Eviction cannot corrupt a result.** Every entry is recomputable and none is
  load-bearing, so a full disk degrades performance rather than correctness.

The connection is shared across threads under a lock, because the ASR Adapter
decorator performs its reads and writes on worker threads to keep the event loop
free — a shared store with a busy timeout can block, unlike the uncontended
per-request spill store.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

from coro.core.models import TranscriptToken

logger = logging.getLogger(__name__)

DATABASE_FILENAME = "asr-windows.sqlite3"
"""Database file created inside the configured cache directory."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    key TEXT PRIMARY KEY,
    tokens_json TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at REAL NOT NULL,
    accessed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS windows_accessed_at ON windows (accessed_at);
"""

_BUSY_TIMEOUT_MS = 5000


class ASRCacheStore:
    """Shared on-disk store of ASR window results, with TTL and LRU retention."""

    def __init__(
        self,
        directory: str,
        *,
        max_bytes: int,
        ttl_seconds: float,
    ) -> None:
        """Open (creating if needed) the cache database under ``directory``.

        Args:
            directory: Cache directory, already resolved and validated. Created
                if missing.
            max_bytes: Size cap for stored rows. ``0`` disables the cap.
            ttl_seconds: Entry lifetime. ``0`` disables expiry.

        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._path = str(Path(directory) / DATABASE_FILENAME)
        self._max_bytes = max(0, int(max_bytes))
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA cache_size=-2000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> str:
        """Filesystem path of the backing database."""
        return self._path

    # Retention -------------------------------------------------------------
    def _expired_before(self, now: float) -> float:
        """Return the creation cutoff below which entries have expired."""
        return -1.0 if self._ttl_seconds == 0 else now - self._ttl_seconds

    def _evict_over_cap(self) -> int:
        """Delete least-recently-accessed rows until the size cap is met.

        Returns:
            The number of rows deleted.

        """
        if self._max_bytes == 0:
            return 0
        (total,) = self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM windows"
        ).fetchone()
        if total <= self._max_bytes:
            return 0

        deleted = 0
        cursor = self._conn.execute("SELECT key, size_bytes FROM windows ORDER BY accessed_at")
        doomed: list[str] = []
        for key, size_bytes in cursor:
            if total <= self._max_bytes:
                break
            doomed.append(key)
            total -= size_bytes
            deleted += 1
        cursor.close()
        if doomed:
            self._conn.executemany("DELETE FROM windows WHERE key = ?", ((k,) for k in doomed))
            logger.info("asr_cache evicted rows=%d remaining_bytes=%d", deleted, total)
        return deleted

    def purge_expired(self) -> int:
        """Delete every expired entry, returning how many were removed."""
        if self._ttl_seconds == 0:
            return 0
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM windows WHERE created_at < ?", (self._expired_before(time.time()),)
            )
            self._conn.commit()
            return cursor.rowcount

    # Lookup And Commit -----------------------------------------------------
    def get(self, key: str) -> list[TranscriptToken] | None:
        """Return the stored tokens for ``key``, or None on a miss.

        An expired entry is treated as a miss and removed. A hit refreshes the
        entry's access time so an entry in active use is not evicted from
        underneath the run reading it.

        Args:
            key: Window key from :func:`coro.cache.fingerprint.window_key`.

        Returns:
            The stored transcript tokens, or ``None``.

        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT tokens_json, created_at FROM windows WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            tokens_json, created_at = row
            if created_at < self._expired_before(now):
                self._conn.execute("DELETE FROM windows WHERE key = ?", (key,))
                self._conn.commit()
                return None
            self._conn.execute("UPDATE windows SET accessed_at = ? WHERE key = ?", (now, key))
            self._conn.commit()
        return [TranscriptToken(**token) for token in json.loads(tokens_json)]

    def put(self, key: str, tokens: list[TranscriptToken]) -> None:
        """Commit one window's result, then sweep the size cap.

        Sweeping on write rather than from a background task means a
        frequently-restarted process still enforces the cap; a process that only
        ever reads cannot grow the store anyway.

        Args:
            key: Window key from :func:`coro.cache.fingerprint.window_key`.
            tokens: The window's transcript tokens.

        """
        tokens_json = json.dumps([asdict(token) for token in tokens], separators=(",", ":"))
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO windows (key, tokens_json, size_bytes, created_at, accessed_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET tokens_json=excluded.tokens_json, "
                "size_bytes=excluded.size_bytes, created_at=excluded.created_at, "
                "accessed_at=excluded.accessed_at",
                (key, tokens_json, len(tokens_json) + len(key), now, now),
            )
            self._evict_over_cap()
            self._conn.commit()

    # Introspection ---------------------------------------------------------
    def entry_count(self) -> int:
        """Return how many entries the store currently holds."""
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM windows").fetchone()
        return int(count)

    def total_bytes(self) -> int:
        """Return the stored size, in bytes, of every entry."""
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM windows"
            ).fetchone()
        return int(total)

    def close(self) -> None:
        """Close the connection, leaving the database on disk."""
        with self._lock, contextlib.suppress(Exception):
            self._conn.close()

    def __enter__(self) -> ASRCacheStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
