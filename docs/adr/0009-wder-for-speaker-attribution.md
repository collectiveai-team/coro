# WDER for Speaker Attribution

The **Quality Benchmark** adds **WDER** (Word Diarization Error Rate) as the primary
metric for per-word speaker attribution, and demotes `cpWER − DI-cpWER` from KPI to
secondary metric. No existing metric is removed.

## Context

The **MeetEval Metric Set** had no metric that could see a per-word speaker change.
Issue `09` measured three radically different word-level attribution rules — split-only,
punctuation-majority realignment, and sandwich-only flicker correction — against
`cpWER − DI-cpWER`. It returned +0.0065, +0.0058 and +0.0047 with a per-clip sign test
of p = 0.688 in every arm. A metric that cannot distinguish three rules which provably
change different things is not measuring what those rules change.

Two structural reasons, both confirmed by re-scoring:

- **cpWER is dominated by hypothesis stream count.** The `-1` unknown-speaker sentinel
  adds one hypothesis stream that the optimal permutation cannot match, so every word
  carrying it becomes an insertion, and the permutation over every *other* stream is
  perturbed as well. `hyp_speakers` was +1 on 6 of 6 clips in all three branch arms,
  identical before and after every attribution rule. The KPI was reading the sentinel,
  not the attribution.
- **cpWER is diluted by the ASR error floor.** At cpWER ≈ 0.68, attribution changes of a
  few hundred words move the headline by less than the run-to-run noise.

DER is not a substitute: it is computed on a timeline and is structurally blind to
transcript segmentation, so it can neither validate nor refute a word-label change.
Using it for one was the category error issue `09` made.

## Decision

Add WDER (Shafey, Soltau & Shafran 2019; the metric DiarizationLM, arXiv:2401.03506,
uses for this exact reconciliation step):

```
WDER = (S_IS + C_IS) / (S + C)
```

`S` is ASR substitutions, `C` correct ASR words, and `_IS` counts those carrying an
incorrect speaker after the optimal hypothesis→reference speaker mapping. Insertions and
deletions are excluded from the denominator.

Report **three** numbers per item and for the combined summary — the decomposition is
the point:

- `wder` — all scored words, `-1` counted as an error. The honest external number.
- `wder_claimed` — precision over words where a real speaker was committed to.
- `abstention_rate` — share of scored words left unknown. Coverage.

They satisfy an exact identity, `wder = wder_claimed × (1 − abstention_rate) +
abstention_rate`, which turns "abstaining costs us the KPI" from a contradiction between
two acceptance criteria into an explicit precision/coverage trade a human can decide.

Two implementation choices:

1. **The speaker mapping is taken, not recomputed.** `meeteval.wer.wer.cp.CPErrorRate`
   carries an `assignment` field with the optimal hypothesis→reference permutation. The
   benchmark already calls `cpwer`; `score_item` now keeps the per-session result dict
   instead of immediately combining it, and hands the assignment to the WDER scorer.
2. **The word alignment reuses meeteval rather than adding a dependency.** meeteval 0.4.3
   ships no WDER and no public word-alignment API, but
   `meeteval.wer.wer.time_constrained.align` becomes a plain Levenshtein alignment when
   driven with a collar wider than the recording — the same reuse `meeteval.viz` performs
   for its `'levenshtein'` alignment type. `kaldialign` and a hand-written backtrace were
   both considered and rejected: the first adds a dependency for something already
   vendored, the second duplicates a well-tested Cython implementation.

The alignment is over a **single time-ordered word stream per side**, not per-speaker
streams. This is required, not incidental: a word given to the wrong speaker must remain
a substitution or a correct word so it can be counted as a speaker error. The per-stream
alignment cpWER performs turns it into an insertion plus a deletion, which WDER excludes
by definition, and the metric would read zero for every hypothesis.

## Consequences

- **`cpWER − DI-cpWER` is demoted to a secondary metric.** It is retained and still
  reported. It stays useful as a *diagnostic of stream-count damage* — which is what it
  actually measures — but it is no longer the acceptance criterion for word-level
  attribution work. The justification is empirical: across four arms on 6 × 600 s of AMI
  it moved +0.0065 / +0.0058 / +0.0047 at p = 0.688, while `wder_claimed` over the same
  artifacts moved 0.1795 → 0.1537 / 0.1601 / 0.1536, better on 5 of 6 clips in every arm.
- **cpWER itself is not demoted and not removed.** It remains the comparability anchor to
  published work; CHiME reports it. ORC-WER remains the speaker-agnostic control that
  says whether a change touched the text at all. DER remains the correct metric for
  *timeline* changes — WDER is for word labels, DER is for the diarizer.
- **WDER is blind to segmentation by construction.** A change that only re-chunks segments
  without moving a word label moves it by exactly zero. That is the desired property, but
  it means flicker rate, segment count and one-word-segment count remain necessary as
  separate readability diagnostics; WDER cannot express them.
- **WDER does not make the `-1` problem disappear.** An abstained word is still an error
  in `wder`. What changes is that the cost is one error per word instead of cpWER's
  super-linear penalty, and that `wder_claimed` isolates the part the response layer
  controls.
- **A consistent relabelling of speakers scores 0, not 1.** Speaker names are arbitrary
  and the cpWER assignment quotients them out; only attribution that no global permutation
  can repair is an error. WDER = 1 is therefore reachable only when every scored word sits
  on a stream the assignment cannot match — the all-`-1` case, or a surplus stream.
- **Cost is negligible.** Re-scoring four arms × six 600 s clips took 10 s wall and
  110 MB RSS on CPU, so WDER adds no meaningful time to a quality run.
- `is_diarization_only_stm` references (e.g. VoxConverse) get no WDER, consistent with the
  other WER metrics: there is no transcript to align.

## Alternatives Considered

- **Fixing cpWER by dropping `-1` lines from the Hypothesis STM**: rejected — it makes
  abstention free rather than measured, and hides the coverage half of the trade that
  `abstention_rate` exists to expose.
- **Adding `kaldialign`**: rejected — meeteval already vendors a tested Cython Levenshtein
  and exposes it through `align`; a second alignment implementation would need its own
  tie-breaking and tokenisation conventions kept in sync.
- **Recomputing the speaker permutation inside the WDER scorer** (as DiarizationLM's
  reference implementation does, via a per-alignment cost matrix): rejected — it would
  give WDER a different permutation from cpWER on the same artifacts, making the two
  metrics silently incomparable. A test asserts the supplied assignment is used verbatim.
- **Replacing DER with WDER**: rejected — they measure different axes. See the
  consequences above.
