# MeetEval for Quality Scoring

The **Quality Benchmark** uses MeetEval as the sole quality scoring engine, replacing the legacy single-WER and pyannote.metrics DER columns that were embedded in the **Resource CSV**.

## Context

The legacy bench scored a single normalized WER and a single pyannote.metrics DER per repetition, stored as Backfilled Quality Columns directly in the **Resource CSV**. This approach could not answer the questions that matter for diarized transcription:

- Which speaker-attribution assumption (single-speaker, concatenated, oracle, or difference-of-inferred) best characterises the error?
- Do our DER numbers match modern published AMI numbers, or are they inflated by a `collar=0.25` policy that hides boundary errors?
- Which error type (insertion, deletion, substitution, false alarm, missed detection, speaker confusion) dominates a regression?

A single WER and a single DER cannot answer any of these. The **MeetEval Metric Set** can.

## Decision

Use MeetEval for the entire **MeetEval Metric Set** reported per **Workload Item**:

- `meeteval.wer.cpwer` — concatenated-minimum-permutation WER
- `meeteval.wer.greedy_orcwer` — oracle-combination WER (greedy)
- `meeteval.wer.greedy_dicpwer` — difference-of-inferred concatenated-permutation WER (greedy)
- `meeteval.der.md_eval_22` — diarization error rate with default `collar=0.0, regions='all'`

Each WER variant is computed twice: once on the raw text and once on punctuation-stripped, whitespace-collapsed text, so a score change caused by punctuation is distinguishable from a score change caused by recognition.

`meeteval.wer.siwer` is **not** part of the set. SISO-WER requires unique `(session, speaker)` pairs, which a multi-speaker meeting recording does not satisfy; on AMI it does not compute at all. A single-speaker WER on multi-speaker audio would also be a meaningless headline.

All four metrics share one library, one set of speaker-permutation conventions, and one DER scoring policy. Each metric reports full error breakdowns (errors, length, insertions, deletions, substitutions for WER variants; false alarm, missed detection, speaker error, total speech for DER).

Diarization-only **Reference STM** files (speaker turns, no transcript) are scored for DER only; the WER half is skipped rather than scored against a sentinel transcript.

Quality results live in `quality/<item>.json` and `quality/summary.json` artifact families. The **Resource CSV** no longer carries `wer`, `der`, `der_collar_s`, `der_skip_overlap`, or `wer_normalization` columns.

The **Reference STM** replaces the Reference RTTM as the canonical format for both transcript and diarization quality scoring. The **Hypothesis STM** is produced from the server response via `hyp_segments_to_stm`, with speaker labels passed through unchanged.

`meeteval` and `rich` are declared in the internal `bench` PEP 735 dependency group in `pyproject.toml` so the runtime server install stays lean. It is a dependency group rather than an optional extra deliberately: the bench is internal tooling, not a supported install surface, so it is installed with `uv sync --group bench` and is not reachable as `coro-asr[bench]`.

## Consequences

- All four metrics share one library with consistent speaker-permutation conventions and one DER scoring policy.
- DER defaults (`collar=0.0, regions='all'`) diverge intentionally from the legacy `collar=0.25, skip_overlap=False` policy to align with modern published AMI numbers. Users who need legacy numbers can pass `--der-collar 0.25 --der-regions nooverlap`.
- The **Resource CSV** no longer carries `wer`, `der`, `der_collar_s`, `der_skip_overlap`, `wer_normalization`. Quality lives exclusively in the **Quality Benchmark** artifact family.
- The standalone Reference RTTM concept is replaced by **Reference STM** — one format for both transcript and diarization scoring.
- Per-item MeetEval failures are isolated: a failing item records `{"error": {...}, "metrics": null}` and the run continues across the rest of the **Workload Set**.
- Run-level quality summary is produced by `meeteval.wer.combine_error_rates`, giving a length-weighted combined score across the **Workload Set**.

## Alternatives Considered

- **pyannote.metrics DER alongside MeetEval DER**: rejected — two DER implementations with different policies invites cherry-picking and creates an inconsistency between the WER and DER halves of the **MeetEval Metric Set**.
- **Default `collar=0.25`**: rejected — silently hides boundary errors and mismatches modern published AMI numbers without any footnote visible in the report.
- **Flat scalars only (no error breakdowns)**: rejected — insertions, deletions, substitutions, false alarm, missed detection, and speaker error are the diagnostic signal when a metric regresses; headline scalars alone cannot guide debugging.
- **Including `siwer` in the metric set**: rejected — SISO-WER's unique-`(session, speaker)` precondition does not hold for meeting recordings, so it either fails or reports a number that answers no question the other variants do not already answer better.
