"""Tests for process-tree resource sampling in ``coro.bench.run``.

Covers the two /proc parsing properties the sampler depends on: discovering a
full process tree without forking, and parsing ``/proc/<pid>/stat`` correctly
when the process name contains spaces or parentheses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coro.bench.run import (
    _get_process_tree_pids,
    _read_child_pid_map,
    _read_proc_stat,
    sample_process_tree,
)

pytestmark = pytest.mark.skipif(
    not Path("/proc/self").is_dir(), reason="requires a Linux /proc filesystem"
)


def _pgrep_children(parent_pid: int) -> set[int]:
    """Reference implementation: the *direct* children of one PID, via ``pgrep -P``."""
    result = subprocess.run(
        ["pgrep", "-P", str(parent_pid)], capture_output=True, text=True, check=False
    )
    return {int(line) for line in result.stdout.strip().splitlines()}


def _pgrep_tree(root_pid: int) -> set[int]:
    """Reference implementation: recursive ``pgrep -P``, one fork per PID."""
    pids = {root_pid}
    for child in _pgrep_children(root_pid):
        pids |= _pgrep_tree(child)
    return pids


@pytest.fixture()
def nested_tree():
    """A real three-level tree: sh -> (sh -> sleep, sleep).

    The trailing ``; true`` stops the shell from ``exec``-ing into ``sleep``,
    which would collapse the grandchild the discovery walk must find.
    """
    proc = subprocess.Popen(["sh", "-c", "sh -c 'sleep 30; true' & sleep 30; true"])
    time.sleep(1.0)
    yield proc
    proc.kill()
    proc.wait()


class TestProcessTreeDiscovery:
    def test_matches_recursive_pgrep_on_a_real_tree(self, nested_tree):
        if shutil.which("pgrep") is None:
            pytest.skip("pgrep not available")
        assert _get_process_tree_pids(nested_tree.pid) == _pgrep_tree(nested_tree.pid)

    def test_finds_descendants_beyond_direct_children(self, nested_tree):
        """A grandchild must be included, not just the immediate children."""
        pids = _get_process_tree_pids(nested_tree.pid)
        children = _read_child_pid_map().children_of(nested_tree.pid)
        assert len(pids) > len(children) + 1

    def test_discovery_forks_no_subprocesses(self, nested_tree, monkeypatch):
        """The regression guard: this ran `pgrep` once per PID, per sample."""

        def fail(*args, **kwargs):
            raise AssertionError("process-tree discovery must not spawn a subprocess")

        monkeypatch.setattr(subprocess, "Popen", fail)
        monkeypatch.setattr(os, "fork", fail, raising=False)
        assert nested_tree.pid in _get_process_tree_pids(nested_tree.pid)

    def test_unknown_pid_yields_just_itself(self):
        assert _get_process_tree_pids(-1) == {-1}

    def test_sample_process_tree_reports_the_whole_tree(self, nested_tree):
        sample = sample_process_tree(nested_tree.pid)
        assert len(sample.pids) >= 3
        assert sample.thread_count >= 3
        assert sample.rss_kb > 0


class TestProcStatParsing:
    def test_handles_a_comm_containing_spaces_and_parentheses(self, tmp_path):
        """`/proc/<pid>/stat` splits wrongly unless sliced at the last ')'.

        A naive `.read().split()` shifts every field after `comm`, so a process
        named like this reported a garbage thread count.
        """
        weird = tmp_path / "we ird) x"
        shutil.copy(sys.executable, weird)
        proc = subprocess.Popen([str(weird), "-c", "import time; time.sleep(30)"])
        try:
            # Waiting on the kernel to publish a real PID in /proc; there is no
            # clock to freeze here, so the sleep is the synchronisation.
            time.sleep(1.0)  # falsegreen: ignore
            raw = Path(f"/proc/{proc.pid}/stat").read_text()
            assert " " in raw[raw.index("(") : raw.rindex(")")]

            stat = _read_proc_stat(proc.pid)
            assert stat.num_threads == 1
            assert stat.utime >= 0
            assert stat.stime >= 0
        finally:
            proc.kill()
            proc.wait()

    def test_reads_plausible_values_for_the_current_process(self):
        stat = _read_proc_stat(os.getpid())
        assert stat.num_threads >= 1
        assert stat.utime >= 0

    def test_missing_pid_returns_zeroed_stat(self):
        stat = _read_proc_stat(-1)
        assert stat.num_threads == 0
        assert stat.utime == 0


class TestChildPidMap:
    def test_maps_a_known_parent_to_its_child(self, nested_tree):
        """The /proc scan must agree with ``pgrep -P`` on the direct children."""
        children = _read_child_pid_map().children_of(nested_tree.pid)
        assert set(children) == _pgrep_children(nested_tree.pid)
        assert len(children) == len(set(children))

    def test_indexes_this_process_under_its_real_parent(self):
        """The scan must place the running test under its own parent PID."""
        assert os.getpid() in _read_child_pid_map().children_of(os.getppid())

    def test_every_key_and_pid_is_an_int(self):
        index = _read_child_pid_map()
        non_int_parents = [p for p in index.children if not isinstance(p, int)]
        non_int_children = [
            child
            for children in index.children.values()
            for child in children
            if not isinstance(child, int)
        ]
        assert non_int_parents == []
        assert non_int_children == []

    def test_unknown_parent_has_no_children(self):
        assert _read_child_pid_map().children_of(-1) == []
