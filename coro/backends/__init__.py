"""ML Model Integrations — external ASR and diarization backends.

Organized with a Capability-First Backend Layout (see ADR 0007): ASR
adapters live under ``coro.backends.asr`` and diarization adapters under
``coro.backends.diarization``, each capability owning a Backend Adapter
Factory. Each adapter module adapts one external library and converts its
native types into Project-Owned Transcript Model types at adapter edges.
"""
