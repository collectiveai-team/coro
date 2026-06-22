#!/usr/bin/env python3
"""Query .journals entries by topic, date, entry slug, or text."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    path: Path
    created_at: str
    title: str
    topic: str
    brief: str
    text: str


def parse_frontmatter(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not text.startswith("---\n"):
        return metadata
    end = text.find("\n---\n", 4)
    if end == -1:
        return metadata
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def collect_entries(journal_root: Path) -> list[Entry]:
    entries: list[Entry] = []
    for path in sorted(journal_root.glob("**/*.md")):
        if path == journal_root / "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        metadata = parse_frontmatter(text)
        if not metadata:
            continue
        entries.append(
            Entry(
                path=path,
                created_at=metadata.get("created_at", ""),
                title=metadata.get("title", ""),
                topic=metadata.get("topic", ""),
                brief=metadata.get("brief", ""),
                text=text,
            )
        )
    return entries


def contains(value: str, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in value.lower()


def matches(entry: Entry, args: argparse.Namespace) -> bool:
    if args.topic and not contains(entry.topic, args.topic):
        return False
    if args.date and not entry.created_at.startswith(args.date):
        return False
    if args.entry and args.entry.lower() not in entry.path.stem.lower():
        return False
    if args.text:
        haystack = "\n".join([entry.title, entry.topic, entry.brief, entry.text])
        if not contains(haystack, args.text):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Query .journals entries")
    parser.add_argument("repo_root", help="Repository root path")
    parser.add_argument("--topic", help="Match topic substring")
    parser.add_argument("--date", help="Match date prefix, for example 2026-06-13 or 2026-06")
    parser.add_argument("--entry", help="Match entry filename slug")
    parser.add_argument("--text", help="Search title, brief, topic, and body text")
    parser.add_argument("--full", action="store_true", help="Print full matching entries")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    journal_root = repo_root / ".journals"
    entries = [entry for entry in collect_entries(journal_root) if matches(entry, args)]
    entries.sort(key=lambda entry: entry.created_at, reverse=True)

    if not entries:
        print("No journal entries matched.")
        return 1

    for entry in entries:
        rel_path = entry.path.relative_to(repo_root).as_posix()
        if args.full:
            print(f"<!-- {rel_path} -->")
            print(entry.text.rstrip())
            print()
        else:
            print(f"- {entry.created_at[:10]} [{entry.topic}] {entry.title} - {rel_path}")
            if entry.brief:
                print(f"  {entry.brief}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
