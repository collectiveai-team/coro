"""Shared pytest fixtures, inherited by every directory below this one."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from support.corpus import SPANISH_CORPUS_ROWS, write_silent_wav
from support.factories import make_app, make_wav
from support.mounts import ROOT_MOUNT_ENTRY, FakeMounts, mountinfo_line

from coro import fsinfo
from coro.bench import spanish


@pytest.fixture
def minimal_wav() -> bytes:
    """A valid, silent, mono 16-bit WAV payload."""
    return make_wav()


@pytest.fixture
def build_app() -> Callable[..., Any]:
    """Factory building an app whose Singleton Runtime serves a pipeline."""
    return make_app


@pytest.fixture
def fake_spanish_corpus(monkeypatch):
    """Serve canned public-corpus rows and fake ffmpeg transcoding.

    Keeps Spanish Workload Set tests offline and free of an ffmpeg dependency
    while exercising the real materialisation, manifest and STM code paths.
    """
    monkeypatch.setattr(
        spanish,
        "resolve_shard_urls",
        lambda dataset, config, split: [f"https://example.invalid/{config}/{split}.parquet"],
    )

    def fake_iter(urls, *, limit, columns=None, timeout=60):
        key = "fleurs" if "es_419" in urls[0] else "mls"
        yield from SPANISH_CORPUS_ROWS[key][:limit]

    monkeypatch.setattr(spanish, "iter_parquet_rows", fake_iter)
    monkeypatch.setattr(
        spanish,
        "transcode_bytes_to_wav",
        lambda data, dst: write_silent_wav(dst),
    )


@pytest.fixture
def stub_server_handle() -> MagicMock:
    """Stand in for a Bench-Managed / Bench-Attached Server handle.

    Patch ``coro.bench.cli.build_server_handle`` with this so exercising
    ``coro.bench.cli.main`` never spawns a real server subprocess or blocks on
    ``/health`` polling.
    """
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.base_url = "http://127.0.0.1:9999"
    handle.server_pid = 4242
    return handle


@pytest.fixture
def fake_mounts(monkeypatch, tmp_path):
    """Install a synthetic mount table describing directories under tmp_path.

    Shared by the transcript spill store and the ASR window cache, which reject
    RAM-backed directories for the same reason and through the same probe.
    """

    def _install(**fs_type_by_name: str) -> FakeMounts:
        mounts = FakeMounts()
        lines = [ROOT_MOUNT_ENTRY]
        for index, (name, fs_type) in enumerate(fs_type_by_name.items(), start=26):
            directory = (tmp_path / name).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            mounts.directories[name] = directory
            lines.append(mountinfo_line(index, directory, fs_type))
        mountinfo = tmp_path / "mountinfo"
        mountinfo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(fsinfo, "_MOUNTINFO_PATH", mountinfo)
        return mounts

    return _install


@pytest.fixture
def undetectable_filesystem(monkeypatch, tmp_path):
    """Make every path's filesystem undeterminable, so none is rejected.

    Needed wherever a test writes into ``tmp_path``: on most Linux hosts that is
    itself tmpfs, which the real resolvers rightly refuse.
    """
    monkeypatch.setattr(fsinfo, "_MOUNTINFO_PATH", tmp_path / "absent-mountinfo")
