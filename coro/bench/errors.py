"""Custom exceptions for the bench CLI."""

from __future__ import annotations


class ServerUnreachableError(RuntimeError):
    """Raised when the ASR server cannot be reached at the configured URL.

    Carries a user-facing message covering both server modes. The original
    exception (typically a ConnectionRefusedError or urllib URLError) is
    chained via __cause__.
    """

    def __init__(self, base_url: str, *, cause: BaseException | None = None) -> None:
        self.base_url = base_url
        message = (
            f"Could not reach the ASR server at {base_url}.\n"
            "\n"
            "Without --server-url the bench starts and manages its own server; a\n"
            "failure here means that server died or never became ready — check its\n"
            "stderr, and confirm `coro` is on PATH in this environment.\n"
            "\n"
            "With --server-url the bench attaches to a server you started; confirm\n"
            "it is listening at that URL, e.g.:\n"
            "  coro --port <PORT>\n"
            "with the matching CORO_* env vars for pipeline/model/diar, then re-run\n"
            "with --server-url http://127.0.0.1:<PORT>.\n"
        )
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class ServerPidUnresolvedError(RuntimeError):
    """Raised when the Server Process Tree root cannot be identified.

    Sampling an unrelated process silently produces resource numbers that
    describe nothing, so the bench refuses to guess: it either finds exactly
    one Bench-Attached Server candidate or asks for an explicit ``--server-pid``.
    """

    def __init__(self, match: str, *, candidates: list[int] | None = None) -> None:
        self.match = match
        self.candidates = candidates or []
        if self.candidates:
            detail = (
                f"--server-match {match!r} matched several unrelated processes: "
                f"{', '.join(str(pid) for pid in self.candidates)}."
            )
        else:
            detail = f"--server-match {match!r} matched no running process."
        message = (
            f"{detail}\n"
            "\n"
            "Resource samples must describe the Server Process Tree, so the bench\n"
            "will not fall back to an arbitrary process. Pass --server-pid <PID>\n"
            "with the root PID of the running server, or use a more specific\n"
            "--server-match, or drop --server-url so the bench manages the server\n"
            "itself and knows its PID.\n"
        )
        super().__init__(message)
