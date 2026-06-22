"""NeMo diarization provider — batch and streaming Sortformer.

NeMo supplies diarization in two forms: batch Sortformer (``diarization.py``)
and streaming Sortformer (``streaming.py``). Both emit Project-Owned
SpeakerSegment timelines via the shared ``coro.backends.diarization.segments``
normalization.
"""
