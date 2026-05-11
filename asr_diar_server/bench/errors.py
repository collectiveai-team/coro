"""Custom exceptions for the bench CLI."""

from __future__ import annotations


class ServerUnreachableError(RuntimeError):
    """Raised when the ASR server cannot be reached at the configured URL.

    Carries a user-facing message that explains how to start a server or
    point the bench at a running one. The original exception (typically a
    ConnectionRefusedError or urllib URLError) is chained via __cause__.
    """

    def __init__(self, base_url: str, *, cause: BaseException | None = None) -> None:
        self.base_url = base_url
        message = (
            f"Could not reach the ASR server at {base_url}.\n"
            "\n"
            "The bench CLI does not start a server on its own; it expects one\n"
            "to be already listening at --server-url (or 127.0.0.1:--server-port).\n"
            "\n"
            "To fix this, either:\n"
            "  1. Start the server in another terminal, e.g.:\n"
            "       asr-diar-server --port <PORT>\n"
            "     with the matching ASR_DIAR_* env vars for pipeline/model/diar,\n"
            "     then re-run the bench with --server-url http://127.0.0.1:<PORT>\n"
            "  2. Or pass --server-url pointing at an already-running server.\n"
        )
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause
