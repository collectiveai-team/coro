## Directive

Each ADR under `docs/adr` is named `NNNN-lower-kebab-case.md`, no two ADRs share a number, and a
retired number (`0008`) does not return. A break is an error, reported on the ADR file that causes
it. The rule also fails when `docs/adr` holds 5 ADRs or fewer, since a check that reads nothing
proves nothing.

## Why

`CONTEXT.md`, docstrings, other ADRs and commit messages cite an ADR by its number alone, so a
duplicate makes every one of those citations ambiguous. It has happened six times: `0002` shipped
duplicated, `0009` was claimed by two pull requests at once, `0008` was renumbered, contested and
retired, and a later pull request claimed `0010` after it was taken. Each was found by a reader and
cost a review cycle and a pass over every reference.

## Do

```text
docs/adr/0024-per-client-rate-limits.md
docs/adr/0025-next-decision.md
```

## Don't

```text
docs/adr/0024-per-client-rate-limits.md
docs/adr/0024-another-decision.md   # 0024 claimed twice
docs/adr/0008-new-decision.md       # 0008 is retired
docs/adr/0025_Next_Decision.md      # the number cannot be read back
```

## Suppression

None: renumber the ADR that merged later and repoint its references. A number is retired in
`RETIRED_NUMBERS` of `adr-numbering.py`.
