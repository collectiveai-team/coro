"""Fixtures shared by the bench tests: loopback stub servers and STM pairs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from support.bench_http import RecordingHandler, StubHandler, serve
from support.wder_stm import Line, write_stm


@pytest.fixture()
def stub_server() -> Iterator[str]:
    """Base URL of a stub server that answers with a canned transcript."""
    with serve(StubHandler) as url:
        yield url


@pytest.fixture()
def e2e_server() -> Iterator[str]:
    """Base URL of a stub server with diarized answers that records its uploads."""
    RecordingHandler.uploads = []
    with serve(RecordingHandler) as url:
        yield url


@pytest.fixture()
def stm_pair(tmp_path: Path) -> Callable[[list[Line], list[Line]], tuple[Path, Path]]:
    """Return a helper writing a reference/hypothesis STM pair into tmp_path."""

    def _make(ref_lines: list[Line], hyp_lines: list[Line]) -> tuple[Path, Path]:
        return (
            write_stm(tmp_path / "ref.stm", ref_lines),
            write_stm(tmp_path / "hyp.stm", hyp_lines),
        )

    return _make
