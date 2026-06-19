"""Bench tooling data models.

Shared/boundary dataclasses for the benchmark tooling, grouped by concern.
Bench is internal tooling rather than a domain boundary, so these models live
under ``bench/`` instead of ``core/`` (see ADR 0006). Truly module-private
one-offs (``/proc`` parsing structs in ``bench.run``, ``GpuDevice`` in
``bench.gpu``) intentionally stay in-file.
"""
