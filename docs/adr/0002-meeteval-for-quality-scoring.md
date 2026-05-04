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

- `meeteval.wer.siwer` — single-speaker WER (no permutation)
- `meeteval.wer.cpwer` — concatenated-minimum-permutation WER
- `meeteval.wer.greedy_orcwer` — oracle-combination WER (greedy)
- `meeteval.wer.greedy_dicpwer` — difference-of-inferred concatenated-permutation WER (greedy)
- `meeteval.der.md_eval_22` — diarization error rate with default `collar=0.0, regions='all'`

All five metrics share one library, one set of speaker-permutation conventions, and one DER scoring policy. Each metric reports full error breakdowns (errors, length, insertions, deletions, substitutions for WER variants; false alarm, missed detection, speaker error, total speech for DER).

Quality results live in `quality/<item>.json` and `quality/summary.json` artifact families. The **Resource CSV** no longer carries `wer`, `der`, `der_collar_s`, `der_skip_overlap`, or `wer_normalization` columns.

The **Reference STM** replaces the Reference RTTM as the canonical format for both transcript and diarization quality scoring. The **Hypothesis STM** is produced from the server response via `hyp_segments_to_stm`, with speaker labels passed through unchanged.

`meeteval` and `rich` are declared as a `[bench]` optional extra in `pyproject.toml` so the runtime server install stays lean.

## Consequences

- All five metrics share one library with consistent speaker-permutation conventions and one DER scoring policy.
- DER defaults (`collar=0.0, regions='all'`) diverge intentionally from the legacy `collar=0.25, skip_overlap=False` policy to align with modern published AMI numbers. Users who need legacy numbers can pass `--der-collar 0.25 --der-regions nooverlap`.
- The **Resource CSV** no longer carries `wer`, `der`, `der_collar_s`, `der_skip_overlap`, `wer_normalization`. Quality lives exclusively in the **Quality Benchmark** artifact family.
- The standalone Reference RTTM concept is replaced by **Reference STM** — one format for both transcript and diarization scoring.
- Per-item MeetEval failures are isolated: a failing item records `{"error": {...}, "metrics": null}` and the run continues across the rest of the **Workload Set**.
- Run-level quality summary is produced by `meeteval.wer.combine_error_rates`, giving a length-weighted combined score across the **Workload Set**.

## Alternatives Considered

- **pyannote.metrics DER alongside MeetEval DER**: rejected — two DER implementations with different policies invites cherry-picking and creates an inconsistency between the WER and DER halves of the **MeetEval Metric Set**.
- **Default `collar=0.25`**: rejected — silently hides boundary errors and mismatches modern published AMI numbers without any footnote visible in the report.
- **Flat scalars only (no error breakdowns)**: rejected — insertions, deletions, substitutions, false alarm, missed detection, and speaker error are the diagnostic signal when a metric regresses; headline scalars alone cannot guide debugging.
