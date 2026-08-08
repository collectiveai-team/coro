"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


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
