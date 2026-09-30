# Per-Client Rate Limits: Requests and Audio Minutes, Rejected with Retry-After

## Status

Accepted.

## Context

coro is mainly an offline transcription server; the Deepgram WebSocket exists
for platform compatibility, not for real-time work. With the live socket always
read and its backlog spilled to disk (ADR 0023), nothing stopped one client
from submitting hours of audio, over any entry point, and taking CPU and disk
from everyone else. The server has no client identity: it accepts Deepgram's
`Authorization` header but does not validate it, so the only key available
today is the client IP.

## Decision

Two per-client-IP token buckets, held in memory per server process
(`coro/api/rate_limit.py`):

- **Requests per minute** (`rate_limit_requests_per_minute`, default 60): a
  burst of that size, refilling evenly over a minute. Counted at the start of
  every REST, SSE and Deepgram REST request, and once per WebSocket connect.
- **Audio minutes per hour** (`rate_limit_audio_minutes_per_hour`, default 0,
  off): a budget in seconds of audio, refilling evenly over an hour.
  - Uploads are measured with `ffprobe` **before** the pipeline runs, so a
    rejected upload costs no CPU. The container's declared duration is used;
    a container without one (a browser `MediaRecorder` WebM) falls back to the
    end of the last audio packet, which demuxes without decoding. An upload
    `ffprobe` cannot measure is admitted uncharged; the decoder then rejects
    it if it is not audio.
  - A WebSocket stream is charged as audio arrives, since its length is not
    known up front. One that runs out gets an `Error` frame and close code
    1008, and its session is abandoned like a disconnected client's.
  - An upload longer than the whole budget is admitted once when the budget
    is full and leaves it in debt, so it is not refused forever.

Over a limit, the request is **rejected, not queued**: HTTP 429 with a
`Retry-After` hint, in the route's vendor shape (OpenAI `rate_limit_exceeded`,
Deepgram `TOO_MANY_REQUESTS`). A WebSocket over the request limit, or with no
audio left, is denied the upgrade with a plain HTTP 429 before `accept`. Work
already admitted is never cut, except a stream whose audio quota runs out.

Both are server configuration, with 0 disabling each. Enabling the audio limit
without `ffprobe` on `PATH` fails startup rather than silently not enforcing
it. The shipped image has `ffprobe` from its `ffmpeg` package.

## Consequences

- One budget per client across all entry points and both pipelines.
- The key is the address uvicorn reports. Behind a reverse proxy, uvicorn
  trusts `X-Forwarded-For` only from `FORWARDED_ALLOW_IPS` (default
  `127.0.0.1`); a proxy elsewhere must be listed there or every client shares
  the proxy's budget.
- Counters are per process: several workers or replicas each count
  separately. A shared store (e.g. Redis) would be needed to enforce one
  budget across them.
- With the audio limit enabled, every upload costs one or two `ffprobe`
  processes; a Deepgram REST body is spooled to a temp file for them.
- Buckets of idle clients are dropped once they have refilled, so memory is
  bounded by the clients active within one refill period.
- The Deepgram `err_code` for a 429 is coro's choice; Deepgram's own value for
  it was not verified.

## Alternatives Considered

- **Queue over-limit work and run it later.** Rejected: delaying work is
  what sharing the CPU already does, and a queued excess still costs disk and
  CPU later, so the limit would protect nobody.
- **Charge uploads after processing.** Rejected: the audio is only measured
  once the CPU has been spent, so one oversized upload always gets through.
- **Charge uploads while decoding and abort when the budget runs out.**
  Rejected: it cuts requests half-way and discards the CPU already spent.
- **Key by API key.** Deferred: it needs authentication, which coro does not
  have yet.
- **Throttle a stream's processing to 1.25x real time, as Deepgram describes.**
  Rejected: it does not help an overloaded server, where processing is
  already below real time, and adds artificial latency to offline use.
