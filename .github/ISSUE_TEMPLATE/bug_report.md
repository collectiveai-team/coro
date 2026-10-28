---
name: Bug report
about: Something behaves, renders, or resolves incorrectly
title: ''
labels: bug
assignees: ''
---

<!-- Report evidence, not impressions. Delete sections that do not apply, but keep
"Measured evidence" and "Steps to reproduce". Redact secrets and personal data. -->

## Summary

One or two sentences. What is wrong, and where.

## Where it happens

| Field | Value |
| --- | --- |
| Entry point | Route, command, API endpoint, or module |
| Environment | OS, runtime/browser version; viewport and DPR for UI bugs |
| Scope | Affected configurations; configurations that work |
| Commit / version | Commit, release, or deploy date |

## Measured evidence

Actual output, errors, timings, counts, or computed styles. State the instrument
and the standard or threshold being violated. Screenshots can support the evidence.
For contrast reports, include resolved foreground/background and the applicable threshold.

## Steps to reproduce

1.
2.
3.

### Verification snippet

A minimal command, test, script, or DevTools snippet a reviewer can run to confirm
the bug and later verify the fix. Include prerequisites and expected output.

## Expected vs actual

- **Expected:**
- **Actual:**

## Source

`path/to/file:123` and the mechanism, if known. Say when the cause is unknown.

## Impact

Who is affected and how badly. Name the user-facing task or workflow.

- [ ] **critical**: a primary workflow is broken, or data is lost/exposed
- [ ] **major**: measurably harms task completion, reading, or performance
- [ ] **minor**: polish

## Not in scope for this issue

Explicit exclusions; link related issues instead of expanding this one.

## Ruled out

What you investigated and found correct, so the next person need not re-derive it.
