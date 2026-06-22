---
name: journalist
description: "Write short daily journal entries for the current session under .journals, maintain a metadata-driven index, and query prior entries by topic, date, or entry slug. Use when the user asks to journal, log, recap, record session notes, capture research notes, or search the project journal."
---

# Journalist

Capture a concise, useful journal entry about the current session. The journal is local/private by default and lives under `.journals/`.

## Layout

Use this repo-local structure:

```text
.journals/
  index.md
  <date>/
    <date>_<topic-slug>_<title-slug>/
      assets/
      <date>_<topic-slug>_<title-slug>.md
```

Example: `.journals/2026-06-13/2026-06-13_repo-tooling_setup-repo-skill/2026-06-13_repo-tooling_setup-repo-skill.md`.

## Entry Rules

- Do not record secrets, credentials, tokens, private keys, or sensitive personal data.
- Keep entries short, but preserve important context, decisions, evidence, links, and follow-ups.
- Support development journals, research journals, debugging notes, and design notes; do not force every entry into a development-only template.
- Use assets only when they materially help. Store them under the entry's `assets/` directory and link with relative paths.
- If an entry for the same date, topic, and title already exists, update it in place and refresh `updated_at`; otherwise create a new entry.

## Topic Selection

1. Read `.journals/index.md` if present, then scan journal frontmatter if needed.
2. Reuse an existing kebab-case topic when it fits the session.
3. Create a new kebab-case topic when no existing topic fits.
4. Ask the user only when multiple existing topics are plausible and the choice matters.

## Entry Template

Use YAML frontmatter exactly like this:

```markdown
---
created_at: <ISO-8601 timestamp>
updated_at: <ISO-8601 timestamp>
title: <short title>
topic: <kebab-case topic>
brief: <2-3 sentence summary>
---

## Notes

<Short narrative, bullets, research observations, decisions, links, or embedded tables/plots as appropriate.>

## Follow-ups

- <Open question or next action, if any.>
```

Adjust body sections to fit the session. Research entries may include tables, plots, quotes, methods, or findings inline instead of `Notes`/`Follow-ups` when that reads better.

## Index And Query Scripts

Use this skill's bundled scripts; do not copy them into `.journals/`:

- `scripts/update_journal_index.py <repo-root>` recompiles `.journals/index.md` from entry frontmatter.
- `scripts/query_journals.py <repo-root> [--topic TOPIC] [--date DATE] [--entry ENTRY] [--text TEXT] [--full]` searches entries.

After writing or updating an entry, always run `update_journal_index.py`.

## Index Format

The generated index is human-oriented, grouped by topic, and sorted by topic then newest entry first. It includes each entry's date, title, brief, and relative path.

Example:

```markdown
# Journal Index

Generated from journal entry frontmatter.

## Repo Tooling

- **2026-06-13** [Setup Repo Skill](.journals/2026-06-13/2026-06-13_repo-tooling_setup-repo-skill/2026-06-13_repo-tooling_setup-repo-skill.md)
  - Updated repo setup defaults and non-destructive editing rules.
```

## Setup Note

If `.journals/` is not gitignored, add `.journals/` to `.gitignore` without removing or rewriting existing ignore rules.
