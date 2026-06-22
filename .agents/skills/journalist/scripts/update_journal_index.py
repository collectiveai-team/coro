#!/usr/bin/env python3
"""Rebuild .journals/index.md from journal entry frontmatter."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    path: Path
    created_at: str
    updated_at: str
    title: str
    topic: str
    brief: str


def parse_frontmatter(path: Path) -> tuple[dict[str, str], list[str]]:
    text = path.read_text(encoding="utf-8")
    warnings: list[str] = []
    if not text.startswith("---\n"):
        return {}, [f"{path}: missing frontmatter"]

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, [f"{path}: unterminated frontmatter"]

    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            warnings.append(f"{path}: ignored malformed frontmatter line: {line}")
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, warnings


def collect_entries(journal_root: Path) -> tuple[list[Entry], list[str]]:
    entries: list[Entry] = []
    warnings: list[str] = []
    required = ["created_at", "updated_at", "title", "topic", "brief"]

    for path in sorted(journal_root.glob("**/*.md")):
        if path == journal_root / "index.md":
            continue
        metadata, path_warnings = parse_frontmatter(path)
        warnings.extend(path_warnings)
        if not metadata:
            continue
        missing = [key for key in required if not metadata.get(key)]
        if missing:
            warnings.append(f"{path}: missing required metadata: {', '.join(missing)}")
            continue
        entries.append(
            Entry(
                path=path,
                created_at=metadata["created_at"],
                updated_at=metadata["updated_at"],
                title=metadata["title"],
                topic=metadata["topic"],
                brief=metadata["brief"],
            )
        )
    return entries, warnings


def topic_title(topic: str) -> str:
    return " ".join(word.capitalize() for word in re.split(r"[-_]+", topic) if word)


def build_index(journal_root: Path, entries: list[Entry]) -> str:
    lines = ["# Journal Index", "", "Generated from journal entry frontmatter.", ""]
    topics = sorted({entry.topic for entry in entries})
    if not topics:
        lines.extend(["No journal entries found.", ""])
        return "\n".join(lines)

    for topic in topics:
        lines.extend([f"## {topic_title(topic)}", ""])
        topic_entries = [entry for entry in entries if entry.topic == topic]
        topic_entries.sort(key=lambda entry: entry.created_at, reverse=True)
        for entry in topic_entries:
            rel_path = entry.path.relative_to(journal_root.parent).as_posix()
            date = entry.created_at[:10]
            lines.append(f"- **{date}** [{entry.title}]({rel_path})")
            lines.append(f"  - {entry.brief}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild .journals/index.md")
    parser.add_argument("repo_root", nargs="?", default=".", help="Repository root path")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    journal_root = repo_root / ".journals"
    journal_root.mkdir(parents=True, exist_ok=True)

    entries, warnings = collect_entries(journal_root)
    (journal_root / "index.md").write_text(build_index(journal_root, entries), encoding="utf-8")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
