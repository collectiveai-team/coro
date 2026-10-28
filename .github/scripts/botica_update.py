#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Write the version of the mirror of botica in every other place of the project that names botica.

`prek update` moves the `rev` of `collectiveai-team/botica-pre-commit` and leaves the other versions
of botica as they are. This script reads the version of that `rev` in `prek.toml`, or in
`.pre-commit-config.yaml` when `prek.toml` has no mirror, and writes it in:

- each `botica@<version>` of `prek.toml`, of `.pre-commit-config.yaml` and of the `.yml` and `.yaml`
  files of `.github/workflows/`, out of the lines that are comments;
- the `rev` of the mirror in `.pre-commit-config.yaml`;
- the parameter `botica-version` of `[recipes.params]` in `botica.toml` and of
  `[tool.botica.recipes.params]` in `pyproject.toml`.

Run from the root of the project, it prints the version, then each file it changed. The workflow
`.github/workflows/botica-update.yml` runs it after `prek update`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MIRROR = "github.com/collectiveai-team/botica-pre-commit"
PREK = "prek.toml"
PRE_COMMIT = ".pre-commit-config.yaml"
WORKFLOWS = ".github/workflows"
CONFIGS = {"botica.toml": "", "pyproject.toml": "tool.botica."}

VERSION = re.compile(r"\d+(?:\.\d+)+[A-Za-z0-9.+!_-]*")
# The `@` of an action (`owner/botica@v1`) or of a longer name (`my-botica@1`) is not botica.
PIN = re.compile(r"(?<![A-Za-z0-9_./@-])botica@(?P<version>\d[A-Za-z0-9.+!_-]*)")
YAML_ENTRY = re.compile(
    r"^(?P<indent>\s*)(?P<item>-\s+)?(?P<key>[\w-]+):\s*(?P<quote>[\"']?)(?P<value>[^\"'\s#]+)(?P=quote)"
)
TOML_ENTRY = re.compile(
    r"^\s*(?:(?P<table>\[)|(?P<key>repo|rev)\s*=\s*(?P<quote>[\"'])(?P<value>[^\"']*)(?P=quote))"
)
TABLE = re.compile(r"^\s*\[\[?\s*(?P<name>[^\]]+?)\s*\]\]?\s*(?:#.*)?$")
PARAM = re.compile(
    r"(?P<head>(?:^\s*|[{,.]\s*)[\"']?botica-version[\"']?\s*=\s*)(?P<quote>[\"'])(?P<value>[^\"']*)(?P=quote)"
)


@dataclass(frozen=True)
class Alignment:
    """The version of the mirror, and the files that the alignment changed, from the root."""

    version: str
    changed: list[str]


def _is_mirror(url: str) -> bool:
    return url.rstrip("/").removesuffix(".git").lower().endswith(MIRROR)


def _version(rev: str, source: str) -> str:
    version = rev.removeprefix("v")
    if not VERSION.fullmatch(version):
        raise SystemExit(f"{source}: the `rev` of the mirror of botica, `{rev}`, is not a version")
    return version


def _lines(text: str) -> list[tuple[int, str]]:
    offsets = []
    offset = 0
    for line in text.splitlines(keepends=True):
        offsets.append((offset, line))
        offset += len(line)
    return offsets


def _entries(pattern: re.Pattern[str], text: str) -> list[tuple[int, re.Match[str]]]:
    """Return the offset and the match of each line of `text` out of comments that matches."""
    found = []
    for start, line in _lines(text):
        entry = None if line.lstrip().startswith("#") else pattern.match(line)
        if entry is not None:
            found.append((start, entry))
    return found


def pre_commit_rev(text: str) -> tuple[int, int] | None:
    """Return where the `rev` of the mirror is in the text of `.pre-commit-config.yaml`.

    Args:
        text: The text of `.pre-commit-config.yaml`.

    Returns:
        The start and the end of the value of the `rev`, without quotes, or None when no repo of
        the file is the mirror.

    """
    mirror_column: int | None = None
    for start, entry in _entries(YAML_ENTRY, text):
        column = len(entry["indent"])
        if entry["item"] and mirror_column is not None and column <= mirror_column:
            mirror_column = None
        if entry["key"] == "repo":
            item_column = column if entry["item"] else column - 2
            mirror_column = item_column if _is_mirror(entry["value"]) else None
        elif entry["key"] == "rev" and mirror_column is not None:
            return start + entry.start("value"), start + entry.end("value")
    return None


def mirror_version(root: Path) -> str:
    """Return the version that the `rev` of the mirror of botica names, without its `v`.

    Args:
        root: The root of the project.

    Returns:
        The version of the `rev` in `prek.toml`, or in `.pre-commit-config.yaml` when `prek.toml`
        has no mirror.

    Raises:
        SystemExit: Neither file has the mirror, or its `rev` is not a version.

    """
    prek = root / PREK
    rev = _prek_rev(prek.read_text(encoding="utf-8")) if prek.is_file() else None
    if rev is not None:
        return _version(rev, PREK)
    pre_commit = root / PRE_COMMIT
    if pre_commit.is_file():
        text = pre_commit.read_text(encoding="utf-8")
        span = pre_commit_rev(text)
        if span is not None:
            return _version(text[span[0] : span[1]], PRE_COMMIT)
    raise SystemExit(f"no `rev` of {MIRROR} in {PREK} or {PRE_COMMIT}")


def set_pins(text: str, version: str) -> str:
    """Return `text` with each `botica@<version>` out of the comment lines at `version`.

    Args:
        text: The text of a hook configuration or of a workflow.
        version: The version to write.

    Returns:
        The text with the pins replaced.

    """

    def replace(match: re.Match[str]) -> str:
        written = match["version"]
        tail = written[len(written.rstrip(".-")) :]
        return f"botica@{version}{tail}"

    return "".join(
        line if line.lstrip().startswith("#") else PIN.sub(replace, line)
        for _, line in _lines(text)
    )


def set_pre_commit_rev(text: str, version: str) -> str:
    """Return the text of `.pre-commit-config.yaml` with the `rev` of the mirror at `version`.

    Args:
        text: The text of `.pre-commit-config.yaml`.
        version: The version to write; the `rev` keeps its `v` if it has one.

    Returns:
        The text with the `rev` replaced, or as it is when no repo is the mirror.

    """
    span = pre_commit_rev(text)
    if span is None:
        return text
    start, end = span
    prefix = "v" if text[start:end].startswith("v") else ""
    return f"{text[:start]}{prefix}{version}{text[end:]}"


def set_param(text: str, version: str, prefix: str) -> str:
    """Return a configuration of botica with the parameter `botica-version` at `version`.

    Args:
        text: The text of `botica.toml` or of `pyproject.toml`.
        version: The version to write.
        prefix: The prefix of the tables of botica: empty in `botica.toml`, `tool.botica.` in
            `pyproject.toml`.

    Returns:
        The text with the parameter replaced in `[<prefix>recipes.params]`, or in a `params` of
        `[<prefix>recipes]`.

    """
    table = ""
    lines = []
    for _, line in _lines(text):
        header = TABLE.match(line)
        if header is not None:
            table = header["name"].replace(" ", "")
        in_params = table == f"{prefix}recipes.params" or (
            table == f"{prefix}recipes" and line.lstrip().startswith("params")
        )
        if header is None and in_params:
            lines.append(PARAM.sub(rf"\g<head>\g<quote>{version}\g<quote>", line))
        else:
            lines.append(line)
    return "".join(lines)


def _prek_rev(text: str) -> str | None:
    """Return the `rev` of the table `[[repos]]` of `prek.toml` whose `repo` is the mirror."""
    tables: list[dict[str, str]] = [{}]
    for _, entry in _entries(TOML_ENTRY, text):
        if entry["table"]:
            tables.append({})
        else:
            tables[-1][entry["key"]] = entry["value"]
    revs = (
        table["rev"] for table in tables if _is_mirror(table.get("repo", "")) and "rev" in table
    )
    return next(revs, None)


def _rewrite(root: Path, name: str, text: str) -> bool:
    path = root / name
    if path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _workflows(root: Path) -> list[str]:
    folder = root / WORKFLOWS
    if not folder.is_dir():
        return []
    return sorted(
        f"{WORKFLOWS}/{path.name}"
        for path in folder.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )


def align(root: Path) -> Alignment:
    """Write the version of the mirror of botica in the hooks, the workflows and the configuration.

    Args:
        root: The root of the project.

    Returns:
        The version, and the files that changed.

    """
    version = mirror_version(root)
    edits = {}
    for name in [PREK, PRE_COMMIT, *_workflows(root)]:
        if (root / name).is_file():
            edits[name] = set_pins((root / name).read_text(encoding="utf-8"), version)
    if PRE_COMMIT in edits:
        edits[PRE_COMMIT] = set_pre_commit_rev(edits[PRE_COMMIT], version)
    for name, prefix in CONFIGS.items():
        if (root / name).is_file():
            edits[name] = set_param((root / name).read_text(encoding="utf-8"), version, prefix)
    changed = [name for name, text in edits.items() if _rewrite(root, name, text)]
    return Alignment(version=version, changed=changed)


if __name__ == "__main__":
    alignment = align(Path.cwd())
    print(alignment.version)
    for changed_file in alignment.changed:
        print(changed_file)
