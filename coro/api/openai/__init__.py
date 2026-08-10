"""OpenAI-compatible API surface.

Owns everything OpenAI defines: the ``POST /v1/audio/transcriptions`` route,
its response schemas, its SSE framing, and the OpenAI-style error body. The
route path is OpenAI's and must not change — an OpenAI SDK client points here
unmodified.
"""
