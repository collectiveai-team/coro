"""The house logger: structlog on top of stdlib logging, configured once.

Every module calls ``get_logger(__name__)``. Events reach stdlib logging with their
message and positional arguments unformatted, so ``caplog``, level overrides and the
handlers of uvicorn keep working. The root handler renders every record, from coro or
from a library, with structlog: a console line, or one JSON object per line when
``CORO_ENV``/``ENV`` is ``prod``.

Importing this module configures nothing. ``get_logger`` configures structlog, and the
first event installs the root handler unless the CLI (``configure_logging``) or another
owner such as pytest already installed one.
"""

from __future__ import annotations

import functools
import logging
import sys

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from coro.settings import get_log_settings


def level_number(level: str) -> int:
    """Return the stdlib number of a level name such as ``"info"``."""
    return logging.getLevelNamesMapping()[level.upper()]


def _formatter(*, prod: bool) -> structlog.stdlib.ProcessorFormatter:
    """Build the formatter of the root handler: JSON in production, a console otherwise."""
    renderers: list[Processor] = (
        [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        if prod
        else [structlog.dev.ConsoleRenderer()]
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.ExtraAdder(),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *renderers],
    )


def configure_logging(level: str | None = None) -> None:
    """Install the house handler on the root logger, as ``logging.basicConfig`` does.

    Does nothing when the root logger already has handlers.

    Args:
        level: Level name of the root logger; defaults to the one of ``LogSettings``.

    """
    settings = get_log_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_formatter(prod=settings.is_prod))
    logging.basicConfig(level=level_number(level or settings.log_level), handlers=[handler])


@functools.cache
def _configure_default() -> None:
    configure_logging()


def _configure_on_first_event(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    _configure_default()
    return event_dict


@functools.cache
def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            _configure_on_first_event,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.render_to_log_args_and_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return the logger of ``name``, usually ``__name__``, configuring structlog once."""
    _configure_structlog()
    return structlog.stdlib.get_logger(name)
