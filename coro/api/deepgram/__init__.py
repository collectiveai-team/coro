"""Deepgram-compatible API surface.

Owns everything Deepgram defines: ``POST /v1/listen`` (pre-recorded) and
``WebSocket /v1/listen`` (live), their response and error schemas, and their
parameter defaults. The route paths are Deepgram's and must not change — a
Deepgram SDK client points here unmodified.

Note the ``v1`` in the path is *Deepgram's* version, unrelated to the ``v1`` in
the OpenAI path. That collision is why this tree is grouped by provider rather
than by version.
"""
