## Directive

No tracked file references `realign_speaker_flicker` except `coro/core/realignment.py`, its tests
`tests/core/test_core_realignment.py`, ADR 0014 and the folder of this rule; and those module and
tests stay in the tree. A break is an error. The rule reads the git index, tracked and staged
files with the suffixes `.py`, `.md`, `.toml`, `.yml`, `.yaml`, `.sh`, `.cfg` and `.txt`, and
fails when it reads fewer than `min-files` (100) of them.

## Why

`realign_speaker_flicker` is the "sandwich rule": a word whose neighbours agree on a speaker is
relabelled to match them. It was measured on the 6-clip WDER workload, it did not pay, and ADR 0014
dropped it from response assembly. The module is kept as the record of that measurement (issue 17).
A reference elsewhere lets the rule creep back into assembly, and deleting the module or its tests
removes the evidence, so the settled question becomes arguable again.

## Do

```python
# coro/core/assembly.py
def assemble(words):
    return list(words)
```

## Don't

```python
# coro/core/assembly.py
from coro.core.realignment import realign_speaker_flicker
```

## Options

`min-files` (int, 100): the fewest files the rule must read before it trusts its own result.

## Suppression

None: the allowed references are in `ALLOWED_REFERENCES` of `flicker-rule-rejected.py`, and a
change to that list is a change to the decision of ADR 0014.
