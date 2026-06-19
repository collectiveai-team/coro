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
