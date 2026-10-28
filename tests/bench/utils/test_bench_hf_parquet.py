"""Tests for the row-group-granular Hugging Face Parquet reader."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from coro.bench.utils import hf_parquet


def _write_parquet(path: Path, rows: int, row_group_size: int) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "id": [f"row{i}" for i in range(rows)],
            "text": [f"texto {i}" for i in range(rows)],
            "audio": [{"bytes": bytes([i % 251]) * 8, "path": f"{i}.wav"} for i in range(rows)],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


class _LocalRangeFile(io.RawIOBase):
    """Stand-in for HttpRangeFile backed by a local file, counting reads."""

    reads: ClassVar[list[int]] = []

    def __init__(self, url: str, *, timeout: int = 60) -> None:
        self._data = Path(url).read_bytes()
        self.size = len(self._data)
        self._pos = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        self.bytes_fetched += len(chunk)
        _LocalRangeFile.reads.append(len(chunk))
        return chunk

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@pytest.fixture
def local_shard(tmp_path: Path, monkeypatch) -> str:
    path = tmp_path / "shard.parquet"
    _write_parquet(path, rows=30, row_group_size=10)
    _LocalRangeFile.reads = []
    monkeypatch.setattr(hf_parquet, "HttpRangeFile", _LocalRangeFile)
    return str(path)


class TestResolveShardUrls:
    def test_returns_index_payload(self, monkeypatch):
        monkeypatch.setattr(
            hf_parquet,
            "http_get",
            lambda url, timeout=60, headers=None: json.dumps(["a.parquet", "b.parquet"]).encode(),
        )

        urls = hf_parquet.resolve_shard_urls("ds", "cfg", "test")

        assert urls == ["a.parquet", "b.parquet"]

    def test_empty_index_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            hf_parquet,
            "http_get",
            lambda url, timeout=60, headers=None: b"[]",
        )

        with pytest.raises(RuntimeError, match="No Parquet shards"):
            hf_parquet.resolve_shard_urls("ds", "cfg", "test")


class TestIterParquetRows:
    def test_yields_rows_in_order_up_to_limit(self, local_shard: str):
        rows = list(hf_parquet.iter_parquet_rows([local_shard], limit=5, columns=["id", "text"]))

        assert [row["id"] for row in rows] == [f"row{i}" for i in range(5)]
        assert rows[0]["text"] == "texto 0"

    def test_stops_after_the_first_row_group(self, local_shard: str):
        pytest.importorskip("pyarrow")
        list(hf_parquet.iter_parquet_rows([local_shard], limit=3, columns=["id"]))
        fetched_for_three = sum(_LocalRangeFile.reads)

        _LocalRangeFile.reads = []
        list(hf_parquet.iter_parquet_rows([local_shard], limit=25, columns=["id"]))
        fetched_for_twentyfive = sum(_LocalRangeFile.reads)

        assert fetched_for_three < fetched_for_twentyfive

    def test_limit_beyond_available_rows_yields_everything(self, local_shard: str):
        rows = list(hf_parquet.iter_parquet_rows([local_shard], limit=999, columns=["id"]))

        assert len(rows) == 30

    def test_audio_struct_round_trips(self, local_shard: str):
        rows = list(hf_parquet.iter_parquet_rows([local_shard], limit=1, columns=["audio"]))

        assert rows[0]["audio"]["bytes"] == bytes([0]) * 8


class TestPlanFetch:
    def test_reports_row_groups_and_bytes(self, local_shard: str):
        plan = hf_parquet.plan_fetch([local_shard], limit=15, columns=["id", "text"])

        assert plan.shards == 1
        assert plan.rows == 15
        assert plan.row_groups == 2
        assert plan.download_bytes > 0

    def test_column_projection_shrinks_the_estimate(self, local_shard: str):
        narrow = hf_parquet.plan_fetch([local_shard], limit=30, columns=["id"])
        wide = hf_parquet.plan_fetch([local_shard], limit=30, columns=None)

        assert narrow.download_bytes < wide.download_bytes


class _RangeServer:
    """In-memory stand-in for ``http_get``/``http_size`` that records each Range header."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.ranges: list[str] = []

    def get(self, url, *, headers=None, timeout=60):
        requested = (headers or {})["Range"]
        self.ranges.append(requested)
        start, end = requested.removeprefix("bytes=").split("-")
        return self.payload[int(start) : int(end) + 1]

    def size(self, url, timeout=60):
        return len(self.payload)


class TestHttpRangeFile:
    def test_reads_are_issued_as_ranges(self, monkeypatch):
        server = _RangeServer(bytes(range(256)))
        monkeypatch.setattr(hf_parquet, "http_size", server.size)
        monkeypatch.setattr(hf_parquet, "http_get", server.get)

        handle = hf_parquet.HttpRangeFile("https://example.invalid/x.parquet")
        handle.seek(-16, io.SEEK_END)

        assert handle.read(16) == server.payload[240:]
        assert server.ranges == ["bytes=240-255"]
        assert handle.bytes_fetched == 16


class TestMissingPyarrow:
    def test_reading_rows_without_pyarrow_names_the_install_command(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

        with pytest.raises(RuntimeError, match="uv sync --group bench"):
            list(hf_parquet.iter_parquet_rows(["unused.parquet"], limit=1, columns=["id"]))
