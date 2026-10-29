"""Server settings from command-line flags, which win over variables and defaults."""

from __future__ import annotations

from pydantic_settings import CliSettingsSource

from coro.settings.server import ServerSettings


def build_settings_from_cli(
    argv: list[str] | None = None, *, prog: str = "coro serve"
) -> ServerSettings:
    """Build ServerSettings honouring CLI flags, env vars, and defaults.

    Args:
        argv: Optional list of CLI arguments. Defaults to ``sys.argv[1:]``.
        prog: Program name shown in help output.

    Returns:
        A populated ``ServerSettings`` instance.

    """
    cli_source = CliSettingsSource(
        ServerSettings,
        cli_parse_args=argv if argv is not None else True,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_prog_name=prog,
        cli_use_class_docs_for_groups=True,
    )
    return ServerSettings(_cli_settings_source=cli_source)  # noqa: BTCONF003 -- flags are a source
