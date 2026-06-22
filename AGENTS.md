# Code Quality Standards

Always apply these standards to all code you write.

## Reuse Before Creating

Before writing new code, analyze existing utilities, components, hooks, helpers and tests:

1. **Search first** — grep/glob for similar functionality before implementing
2. **Extend if close** — if something exists that's 80% of what you need, extend it
3. **Extract if duplicating** — if you're about to copy-paste, extract to shared module instead

## File Size & Organization

Keep files between **200-300 lines max**. If a file exceeds this:

1. **Split by responsibility** — one module = one concern
2. **Extract sub-components** — UI pieces that can stand alone should
3. **Separate logic from presentation** — hooks/utils in their own files
4. **Group by feature** — co-locate related files, not by type

Signs a file needs splitting:
- Multiple unrelated exports
- Scrolling to find what you need
- "Utils" file becoming a junk drawer
- Component doing data fetching + transformation + rendering

## Code Style

1. Prefer writing clear code and use inline comments sparingly
2. Document methods with block comments at the top of the method
3. Use Conventional Commit format

## Test To Verify Functionality

If you didn't test it, it doesn't work.

Verify written code by:
- Running unit tests
- Running end to end tests
- Checking for type errors
- Checking for lint errors
- Smoke testing and checking for runtime errors with Playwright
- Taking screenshots and verifying the UI is as expected

## Dev Workflow

Dependencies are managed with **uv**; the `cpu` and `cuda` extras are mutually
exclusive (pick one):

```sh
uv sync --extra cpu        # or: uv sync --extra cuda (GPU host)
```

Lint, format and type checks are centralized in `prek.toml` and run with
**prek** (a faster pre-commit). CI runs the exact same config, so local == CI:

```sh
uvx prek install           # install the git pre-commit hook (once)
uvx prek run --all-files   # ruff-format, ruff, pyrefly + hygiene hooks
uv run pytest              # tests (not part of prek)
```

## Releasing

Versioning is **tag-driven** via hatch-vcs — never edit a version string by
hand. To cut a release:

```sh
git tag v0.2.0 && git push origin v0.2.0
```

CI then builds the wheel (version derived from the tag), creates a GitHub
Release, publishes to PyPI as `coro-asr` (import stays `coro`), and pushes
`:0.2.0-cpu` / `:0.2.0-gpu` images to GHCR.

## Agent skills

### Issue tracker

Issues and PRDs live as markdown files under `.scratch/<feature-slug>/` (local markdown). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles map 1:1 to their default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Repo Workspace Defaults

- Use `.scratch/` for temporary plans, issue drafts, and disposable notes.
- Use `.tmp/` for generated local artifacts that should not be committed.
- Use `.worktrees/` for local Git worktrees when needed.
- Use `.journals/` for local/private session journals when using the `journalist` skill.
- Preserve unrelated user changes and avoid destructive Git commands unless explicitly requested.

### Resource Limits For Heavy Commands

Run computationally expensive commands (heavy builds, full test suites, data jobs, memory-hungry tooling) under `systemd-run` so a runaway process cannot exhaust the machine:

- Cap memory with `MemoryMax` (hard limit; process is killed if exceeded) and disable swap thrashing with `MemorySwapMax=0`.
- Cap CPU with `CPUQuota` (e.g. `400%` = up to 4 cores).
- Use a transient user scope (no root needed). If `systemd-run` is unavailable, fall back to `ulimit -v` for a memory cap.

Measure and log wall time and peak memory with `/usr/bin/time -v`, appending to `.tmp/logs/<command>/<date>_wall-stats.log`. Read the latest log before re-running to estimate limits instead of guessing:

```bash
cmd=build; ts=$(date +%Y-%m-%dT%H-%M-%S)
log=.tmp/logs/$cmd/${ts}_wall-stats.log; mkdir -p ".tmp/logs/$cmd"
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=400% \
  /usr/bin/time -v -o "$log" <command>
```

Run long commands in the background with a saved PID so they are not tied to the agent tool timeout, and delegate polling to a cheap monitor sub-agent (e.g. Haiku) that runs a `while kill -0 <pid>; do sleep <n>; done` loop and reports only the final exit status, output tail, and stats log path. This keeps the main session from being poisoned by repeated status checks.

```bash
( systemd-run --user --scope -p MemoryMax=4G -p CPUQuota=400% \
    /usr/bin/time -v -o "$log" <command> ) >".tmp/logs/$cmd/${ts}.out" 2>&1 &
echo $! > ".tmp/logs/$cmd/${ts}.pid"
```

Scripts we write ourselves must emit periodic progress to the output log (current step, counts, percent, or heartbeat with a timestamp) so the monitor can tell what they are doing. Flush each line (e.g. Python `print(..., flush=True)` or `PYTHONUNBUFFERED=1`) so progress appears live. The monitor always `tail`s the latest lines of the `.out` log — never reads the full log — to avoid poisoning the session with bulk output.

A consistently too-expensive step is a signal of poor code that needs optimization, not a reason to keep raising the limits — flag it.
