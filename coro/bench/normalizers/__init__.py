"""Text normalizers used by the Quality Benchmark's Normalized Metric Lane.

The Basic Text Normalizer is vendored from OpenAI Whisper (MIT) rather than
pulled in as a dependency; see ``basic.py`` and ADR 0011.
"""

from coro.bench.normalizers.basic import BasicTextNormalizer

__all__ = ["BasicTextNormalizer"]
