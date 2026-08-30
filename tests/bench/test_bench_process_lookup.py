"""Server Process Tree root resolution for a Bench-Attached Server."""

from __future__ import annotations

import os

import pytest

from coro.bench.errors import ServerPidUnresolvedError
from coro.bench.process_lookup import (
    DEFAULT_SERVER_MATCH,
    ProcessEntry,
    list_processes,
    resolve_server_pid,
)


def _tree(*entries: tuple[int, int, str]) -> list[ProcessEntry]:
    return [ProcessEntry(pid=pid, ppid=ppid, cmdline=cmd) for pid, ppid, cmd in entries]


BENCH_CLIENT = (100, 1, "python -m coro.bench.cli all")


class TestResolveServerPid:
    def test_resolves_a_single_match(self):
        processes = _tree(BENCH_CLIENT, (200, 1, "coro --port 8123"))

        assert resolve_server_pid("coro", processes=processes, self_pid=100) == 200

    def test_excludes_the_bench_client_itself(self):
        """The default match 'coro' also appears in the bench client's own command line."""
        processes = _tree(BENCH_CLIENT, (200, 1, "coro --port 8123"))

        assert resolve_server_pid(DEFAULT_SERVER_MATCH, processes=processes, self_pid=100) == 200

    def test_excludes_the_bench_client_ancestors(self):
        processes = _tree((50, 1, "uv run coro-bench all"), (100, 50, "python -m coro.bench.cli"))

        with pytest.raises(ServerPidUnresolvedError):
            resolve_server_pid("coro", processes=processes, self_pid=100)

    def test_excludes_bench_client_descendants(self):
        processes = _tree(BENCH_CLIENT, (150, 100, "ffprobe coro-clip.wav"))

        with pytest.raises(ServerPidUnresolvedError):
            resolve_server_pid("coro", processes=processes, self_pid=100)

    def test_collapses_worker_processes_onto_their_root(self):
        """A server's workers belong to the same Server Process Tree."""
        processes = _tree(
            BENCH_CLIENT,
            (200, 1, "coro --port 8123"),
            (201, 200, "coro --port 8123"),
            (202, 201, "coro --port 8123"),
        )

        assert resolve_server_pid("coro", processes=processes, self_pid=100) == 200

    def test_raises_on_no_match(self):
        processes = _tree(BENCH_CLIENT)

        with pytest.raises(ServerPidUnresolvedError, match="matched no running process"):
            resolve_server_pid("nonexistent-server", processes=processes, self_pid=100)

    def test_raises_on_ambiguous_match(self):
        processes = _tree(
            BENCH_CLIENT,
            (200, 1, "coro --port 8123"),
            (300, 1, "coro --port 8124"),
        )

        with pytest.raises(ServerPidUnresolvedError, match="matched several") as exc_info:
            resolve_server_pid("coro", processes=processes, self_pid=100)

        assert exc_info.value.candidates == [200, 300]

    def test_error_message_offers_server_pid_as_the_escape_hatch(self):
        processes = _tree(BENCH_CLIENT)

        with pytest.raises(ServerPidUnresolvedError) as exc_info:
            resolve_server_pid("nope", processes=processes, self_pid=100)

        assert "--server-pid" in str(exc_info.value)

    def test_survives_a_parent_pid_cycle(self):
        """Corrupt /proc snapshots must not hang the resolver."""
        processes = _tree(BENCH_CLIENT, (200, 201, "coro a"), (201, 200, "coro b"))

        with pytest.raises(ServerPidUnresolvedError, match="matched several") as exc_info:
            resolve_server_pid("coro", processes=processes, self_pid=100)

        assert exc_info.value.candidates == [200, 201]


class TestListProcesses:
    def test_includes_the_current_process(self):
        pids = {entry.pid for entry in list_processes()}

        assert os.getpid() in pids

    def test_entries_have_non_empty_cmdlines(self):
        assert all(entry.cmdline for entry in list_processes())
