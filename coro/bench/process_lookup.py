"""Server Process Tree root lookup for a Bench-Attached Server.

Resolves ``--server-match`` to exactly one root PID by scanning ``/proc``.
Sampling the wrong process yields resource numbers that describe nothing, so
this module never guesses: it resolves one candidate or raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coro.bench.errors import ServerPidUnresolvedError

DEFAULT_SERVER_MATCH = "coro"

_PROC_ROOT = Path("/proc")


@dataclass(frozen=True)
class ProcessEntry:
    """One running process: its PID, parent PID, and full command line."""

    pid: int
    ppid: int
    cmdline: str


def _read_ppid(pid: int) -> int:
    """Read ``PPid`` from ``/proc/<pid>/status`` (0 when unreadable).

    ``/proc/<pid>/status`` is used rather than ``stat`` because the latter's
    ``comm`` field may itself contain spaces and parentheses.
    """
    try:
        for line in (_PROC_ROOT / str(pid) / "status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return 0
    return 0


def list_processes() -> list[ProcessEntry]:
    """Snapshot every readable process with a non-empty command line."""
    entries: list[ProcessEntry] = []
    for child in _PROC_ROOT.iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        try:
            raw = (child / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not cmdline:
            continue
        entries.append(ProcessEntry(pid=pid, ppid=_read_ppid(pid), cmdline=cmdline))
    return entries


def _ancestors(pid: int, parents: dict[int, int]) -> set[int]:
    """Walk ``pid`` up to the root of its process tree, cycle-safe."""
    seen: set[int] = set()
    current = parents.get(pid, 0)
    while current and current not in seen:
        seen.add(current)
        current = parents.get(current, 0)
    return seen


def resolve_server_pid(
    match: str = DEFAULT_SERVER_MATCH,
    *,
    processes: list[ProcessEntry] | None = None,
    self_pid: int | None = None,
) -> int:
    """Resolve the root PID of the Server Process Tree matching ``match``.

    The bench client's own process tree is excluded, so a ``--server-match``
    substring that also appears in the bench command line (the default
    ``"coro"`` does) cannot resolve to the client itself. Candidates that
    descend from another candidate are collapsed into their root, since a
    server's workers belong to the same Server Process Tree.

    Raises:
        ServerPidUnresolvedError: when zero, or more than one unrelated
            process tree, matches.

    """
    import os

    if processes is None:
        processes = list_processes()
    if self_pid is None:
        self_pid = os.getpid()

    parents = {entry.pid: entry.ppid for entry in processes}
    own = {self_pid, *_ancestors(self_pid, parents)}

    candidates = [
        entry.pid
        for entry in processes
        if match in entry.cmdline
        and entry.pid not in own
        and self_pid not in _ancestors(entry.pid, parents)
    ]
    if not candidates:
        raise ServerPidUnresolvedError(match)

    candidate_set = set(candidates)
    roots = sorted(pid for pid in candidates if not (_ancestors(pid, parents) & candidate_set))
    if len(roots) == 1:
        return roots[0]
    # ``roots`` is empty only for a cyclic /proc snapshot; report every candidate then.
    raise ServerPidUnresolvedError(match, candidates=roots or sorted(candidate_set))
