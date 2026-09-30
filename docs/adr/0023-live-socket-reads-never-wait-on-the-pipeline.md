# The Live Socket Is Always Read; a Gone Client Abandons Its Backlog

## Status

Accepted. Amends ADR 0015 ("Streaming: a WebSocket, and it is genuinely
live"): the bounded live queue is replaced.

## Context

`WebSocket /v1/listen` fed audio into a `LiveAudioSource` bounded at 64
chunks, so a client sending faster than the pipeline consumed would get
backpressure. Under load the pipeline runs slower than real time (on CPU, the
`low` Sortformer tier processed 66 s of audio in ~308 s), and the bound turned
into a dropped connection:

1. The queue fills and the socket handler blocks in `push`, calling
   `websocket.receive()` only once per consumed chunk.
2. uvicorn's WebSocket protocol (`websockets-sansio`, the `auto` choice)
   pauses reading the TCP stream after every message until the app receives
   it.
3. Ping and pong frames share that stream with audio. uvicorn's own keepalive
   ping (20 s interval, 20 s timeout) is answered, but the client's pong sits
   behind unread audio.
4. uvicorn closes with `1011 keepalive ping timeout`.

Reproduced with the real handler on real uvicorn and a diarizer 4.7x slower
than real time: the server closed at 40 s when the client sent everything at
once, and at 80 s when it sent at real time. At 1.2x a 480 s real-time stream
closed at 380 s. Any stream long enough dies once the server falls behind.

Separately, a disconnect did not stop the session: the handler drained the
whole backlog and ran `finalize` for a client that was gone, which took about 5
minutes of CPU from the other connections in the load run.

Deepgram documents no lag error and no dropping. It absorbs the delay: "if
you send a large buffer of audio, the stream may wind up being significantly
delayed". Its only timeouts (`NET-0000`, `NET-0001`, `NET-0002`) concern an
idle socket, not a slow server.

## Decision

- `LiveAudioSource` is unbounded. The socket handler reads every frame as it
  arrives and never waits on the pipeline, so keepalive pings, `KeepAlive` and
  `CloseStream` are always processed, however far behind the pipeline is.
- The backlog is bounded in memory only: past 1 MiB per connection (about
  33 s of audio, more than one ASR window) chunks are appended to an
  anonymous temp file and read back in order. The file lives in the transcript
  spill directory, which is already resolved to real disk, and is unlinked on
  creation, so it is reclaimed even if the process dies. It is truncated
  whenever the consumer catches up.
- After `CloseStream` the handler keeps reading, ignoring further frames, so
  pings are still answered while the backlog drains.
- If the client disconnects before its results are delivered, the session is
  cancelled: no further chunk is ingested and `finalize` does not run.

## Consequences

- A lagging stream is late, not dropped, matching Deepgram.
- Host RAM stays flat per connection; the backlog grows on disk instead, at
  115 MB per hour of canonical PCM. Nothing caps the disk yet: a client
  sending hours of audio at once fills it at that rate. Controlling abusive
  clients is a separate decision.
- Spill writes and reads are small synchronous page-cache operations on the
  event loop, one per client frame.
- A diarizer chunk already running on a worker thread when the client leaves
  still finishes; only the chunks after it are skipped.

## Alternatives Considered

- **Close with an explicit error when the lag passes a threshold.** Rejected:
  Deepgram does not do it, and the stream is lost anyway.
- **Throttle reads to Deepgram's 1.25x real-time ingest rate.** Rejected: a
  client sending a recorded file at once still has its pong behind the
  unread audio, which is the original failure.
- **Drop diarization for a lagging stream.** Rejected: speaker attribution is
  the value of the Streaming Pipeline, and a Sortformer state cannot switch
  tiers mid-stream.
