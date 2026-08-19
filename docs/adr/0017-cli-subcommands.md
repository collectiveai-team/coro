# CLI Subcommands, and Removing the Bare Invocation

The packaged `coro` entry point now takes a subcommand: `coro serve`, `coro run` or `coro bench`. **The bare invocation is removed.** `coro --port 8000`, which used to start the server, now fails and prints the three commands.

This is a **breaking change** to the packaged interface. It is worth it because `coro` previously had exactly one behaviour — start the server — and named it by naming nothing, which left no room for a second thing to do. Adding `coro run` under that shape would have meant either a magic positional argument (`coro file.wav` starts a transcription, `coro` starts a server) or a flag that changes what the program *is*. Both are worse than a rename that can be diagnosed in one line of error output.

## What each command is

`coro serve` is the previous behaviour, unchanged.

`coro run FILE` transcribes a local file. Getting a transcript otherwise means starting the server and uploading to it, which for a multi-gigabyte recording means pushing the whole thing through a socket before any work starts.

`coro bench` reaches the benchmark tooling. `coro-bench` remains a separate console script and keeps working — it appears throughout `docs/benchmark.md` and the README, and breaking it would buy nothing — so `coro bench` is a consistent alias rather than a replacement.

## `coro run` is in-process by default

The pipeline runs directly against the file: no upload, no server, no port. Attaching to an already-running server is available through `--server-url`, and it is **never auto-probed**. Probing was rejected outright: the same command must not behave differently depending on whether something happens to be listening on a port, which turns a reproducible command into a function of unrelated machine state. An attached run is also governed by the *server's* configuration rather than the flags typed locally, so a run that silently attached would silently ignore them — which is why both modes report which configuration actually produced the result.

Spawning a server in order to upload to it is explicitly not provided. Running in-process is better in every respect for a local file, so the only thing that option would add is a slower path with more failure modes.

`coro run` accepts the same pydantic-settings-derived flag surface as `coro serve`, so there is no second configuration vocabulary. It parses its own arguments first and hands everything else to the settings source, which still rejects genuinely unknown flags rather than ignoring them.

Both modes emit the same body, rendered by the same function the transcription endpoint renders through. Without that, `coro run` would be a fourth response shape alongside `json`, `verbose_json` and `diarized_json` that nothing documented and nothing tested; a test compares its output against the endpoint's for the same file. The default format is `diarized_json`, the richest project-native shape.

## Referencing a file without owning it

`AudioInput` gains a backing that references an existing file, with explicit ownership: cleanup unlinks only files the instance created. This is not a tidiness change. Uploads are spooled eagerly and both pipelines call `cleanup()` in a `finally`, so handing them an owning `AudioInput` over the user's input file would **delete that file** on every successful run.

## Lazy adapter construction

On this path the ASR Adapter is constructed on first inference rather than eagerly, so a fully-cached run never loads the model — which is what makes such a run genuinely fast rather than merely inference-free. Server startup keeps its eager behaviour so **Server Warmup** and **Warmup Readiness** semantics are unchanged.
