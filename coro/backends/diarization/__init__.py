"""Diarization Backend Providers — speaker-timeline ML Model Integrations.

Each provider adapts one external diarization library into a Diarization
Adapter that produces a Project-Owned speaker timeline. Shared native-output
normalization lives in ``segments.py`` so every adapter emits an identical
SpeakerSegment shape; the Diarization Backend Adapter Factory in
``factory.py`` builds the configured adapter at startup.
"""
