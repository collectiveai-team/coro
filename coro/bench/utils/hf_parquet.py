"""Row-group-granular reader for Hugging Face auto-converted Parquet datasets.

Public speech corpora on the Hub are exposed as Parquet shards that embed the
audio bytes alongside the transcript columns. A benchmark **Workload Set** only
needs the first few dozen rows, so this module reads the Parquet footer over an
HTTP ``Range`` request and then downloads whole row groups on demand, stopping
as soon as the requested number of rows has been produced. Nothing is written to
disk here; callers decide what to materialise.

Reproducibility: shard URLs are resolved from the Hub's public Parquet index
endpoint, so a clean checkout with network access fetches exactly the same rows
in the same order.
"""

from __future__ import annotations

import io
import json
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

PARQUET_INDEX_URL = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"

_DEFAULT_TIMEOUT = 60
_FOOTER_READ_BYTES = 64 * 1024


def _require_pyarrow():
    """Import pyarrow.parquet, or fail with an actionable install hint."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise RuntimeError(
            "pyarrow is required to fetch Hugging Face Parquet corpora.\n"
            "Install the bench tooling with: uv sync --group bench"
        ) from exc
    return pq


def http_get(url: str, *, headers: dict[str, str] | None = None, timeout: int) -> bytes:
    """Fetch ``url`` and return the response body."""
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_size(url: str, *, timeout: int) -> int:
    """Return the byte length of ``url`` via a one-byte ranged request.

    A ranged GET is used rather than HEAD because the Hub redirects to a CDN
    that does not always answer HEAD with a usable ``Content-Length``.
    """
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        length = response.headers.get("Content-Length")
        if length is None:
            raise RuntimeError(f"Could not determine size of {url}")
        return int(length)


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP ``Range`` requests.

    Enough of the file protocol for ``pyarrow.parquet.ParquetFile`` to read a
    footer and individual row groups without downloading the whole shard.
    """

    def __init__(self, url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout
        self.size = http_size(url, timeout=timeout)
        self.bytes_fetched = 0
        self._pos = 0

    def readable(self) -> bool:
        """Return True; the file is always opened for reading."""
        return True

    def seekable(self) -> bool:
        """Return True; ranged requests give random access."""
        return True

    def tell(self) -> int:
        """Return the current read offset."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read offset and return the new absolute position."""
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes from the current offset."""
        if size is None or size < 0:
            size = self.size - self._pos
        end = min(self._pos + size, self.size) - 1
        if size == 0 or end < self._pos:
            return b""
        data = http_get(
            self.url,
            headers={"Range": f"bytes={self._pos}-{end}"},
            timeout=self.timeout,
        )
        self.bytes_fetched += len(data)
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@dataclass(frozen=True)
class FetchPlan:
    """How much of a shard set must be downloaded to satisfy a row limit."""

    shards: int
    row_groups: int
    rows: int
    download_bytes: int


def resolve_shard_urls(
    dataset: str,
    config: str,
    split: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> list[str]:
    """Return the ordered Parquet shard URLs for one dataset config/split."""
    url = PARQUET_INDEX_URL.format(dataset=dataset, config=config, split=split)
    payload = json.loads(http_get(url, timeout=timeout))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"No Parquet shards published for {dataset} {config}/{split}: {payload}")
    return [str(item) for item in payload]


def _row_group_bytes(metadata, index: int, columns: Sequence[str] | None) -> int:
    """Return the compressed size of the column chunks a read would fetch."""
    row_group = metadata.row_group(index)
    total = 0
    for column in range(row_group.num_columns):
        chunk = row_group.column(column)
        name = chunk.path_in_schema.split(".", 1)[0]
        if columns is None or name in columns:
            total += chunk.total_compressed_size
    return total


def plan_fetch(
    shard_urls: Sequence[str],
    *,
    limit: int,
    columns: Sequence[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> FetchPlan:
    """Estimate the download footprint of reading ``limit`` rows in order.

    Reads only Parquet footers (a few tens of KB per shard), so it is cheap
    enough to print before a large fetch.
    """
    pq = _require_pyarrow()

    remaining = limit
    shards = row_groups = rows = 0
    download_bytes = 0

    for url in shard_urls:
        if remaining <= 0:
            break
        handle = HttpRangeFile(url, timeout=timeout)
        metadata = pq.ParquetFile(handle).metadata
        shards += 1
        download_bytes += _FOOTER_READ_BYTES
        for index in range(metadata.num_row_groups):
            if remaining <= 0:
                break
            taken = min(remaining, metadata.row_group(index).num_rows)
            remaining -= taken
            rows += taken
            row_groups += 1
            download_bytes += _row_group_bytes(metadata, index, columns)

    return FetchPlan(shards=shards, row_groups=row_groups, rows=rows, download_bytes=download_bytes)


def iter_parquet_rows(
    shard_urls: Sequence[str],
    *,
    limit: int,
    columns: Sequence[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Iterator[dict[str, Any]]:
    """Yield up to ``limit`` rows in shard/row-group order as plain dicts.

    Row groups are downloaded one at a time and released before the next one is
    requested, so peak memory stays at roughly one row group.
    """
    pq = _require_pyarrow()

    remaining = limit
    for url in shard_urls:
        if remaining <= 0:
            return
        handle = HttpRangeFile(url, timeout=timeout)
        parquet_file = pq.ParquetFile(handle)
        for index in range(parquet_file.metadata.num_row_groups):
            if remaining <= 0:
                return
            table = parquet_file.read_row_group(index, columns=list(columns) if columns else None)
            for row in table.to_pylist():
                yield row
                remaining -= 1
                if remaining <= 0:
                    return
