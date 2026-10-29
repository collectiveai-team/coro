---
name: Bug report
about: Something behaves or resolves incorrectly
title: ''
labels: bug
assignees: ''
---

<!-- Delete the sections that do not apply, but keep "Measured evidence" and
"Steps to reproduce". Redact secrets and personal data. -->

## Summary

What is wrong, and where, in one or two sentences.

## Where it happens

| Field | Value |
| --- | --- |
| Entry point | Route, command, API endpoint or module |
| Platform | OS, Python version, CPU or GPU |
| Scope | Configurations that fail; configurations that work |
| Version | coro version (`coro --version`) or commit |

## Measured evidence

The output, errors, timings or counts you saw, and the threshold they break.

## Steps to reproduce

1.
2.
3.

### Verification snippet

A command or test a reviewer can run to see the bug and, later, check the fix.
Include what it needs and what it prints.

## Expected and actual

- Expected:
- Actual:

## Source

`path/to/file:123` and the mechanism, if known. Say so when the cause is unknown.

## Impact

Who is affected, in which task.

- [ ] critical: a main workflow is broken, or data is lost or exposed
- [ ] major: a task gets measurably slower, harder or wrong
- [ ] minor: polish

## Out of scope

What this issue leaves out, with links to the issues that cover it.

## Ruled out

What you checked and found correct.
