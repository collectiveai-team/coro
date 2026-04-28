export const DEFAULT_THRESHOLDS = {
  minIoU: 0.3,
  longSegmentSeconds: 60,
  contextSeconds: 1,
  highWer: 0.6,
};

const DEEPGRAM_SEGMENT_GAP_SECONDS = 1;

function deepgramUtteranceRows(input) {
  if (!Array.isArray(input?.results?.utterances)) return [];
  return input.results.utterances.map((utterance) => ({
    start: utterance?.start,
    end: utterance?.end,
    text: utterance?.transcript ?? utterance?.text,
    speaker: utterance?.speaker,
  }));
}

function deepgramWordRows(input) {
  const rows = [];
  const channels = input?.results?.channels;
  if (!Array.isArray(channels)) return rows;

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const alternative = channels[channelIndex]?.alternatives?.[0];
    if (!Array.isArray(alternative?.words)) continue;

    let current = null;
    for (const word of alternative.words) {
      const start = Number(word?.start);
      const end = Number(word?.end);
      const text = String(word?.punctuated_word ?? word?.word ?? "").trim();
      const speaker = word?.speaker == null ? `channel-${channelIndex}` : String(word.speaker);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || !text) continue;

      const shouldStartSegment = !current || current.speaker !== speaker || start - current.end > DEEPGRAM_SEGMENT_GAP_SECONDS;
      if (shouldStartSegment) {
        if (current) rows.push(current);
        current = { start, end, text, speaker };
      } else {
        current.end = end;
        current.text = `${current.text} ${text}`;
      }
    }
    if (current) rows.push(current);
  }

  return rows;
}

function deepgramRows(input) {
  const utteranceRows = deepgramUtteranceRows(input);
  if (utteranceRows.length > 0) return utteranceRows;
  return deepgramWordRows(input);
}

function asRows(input) {
  if (Array.isArray(input)) return input;
  if (input && Array.isArray(input.lines)) return input.lines;
  if (input && Array.isArray(input.segments)) return input.segments;
  const rows = deepgramRows(input);
  if (rows.length > 0) return rows;
  throw new Error("Unsupported JSON shape: expected lines[], segments[], Deepgram results, or an array of segments");
}

export function normalizeInput(input, source) {
  const rows = asRows(input);
  const segments = [];
  const warnings = [];

  rows.forEach((row, originalIndex) => {
    const start = Number(row?.start);
    const end = Number(row?.end);
    const text = String(row?.text ?? "").trim();
    const speaker = row?.speaker == null ? "unknown" : String(row.speaker);

    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      warnings.push({ source, originalIndex, reason: "start and end must be numeric" });
      return;
    }
    if (end <= start) {
      warnings.push({ source, originalIndex, reason: "end must be greater than start" });
      return;
    }
    if (!text) {
      warnings.push({ source, originalIndex, reason: "text must be non-empty" });
      return;
    }

    segments.push({ source, originalIndex, start, end, text, speaker });
  });

  return { segments, warnings };
}

export function mergeIntervals(intervals) {
  const sorted = intervals
    .map(({ start, end }) => ({ start, end }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  for (const interval of sorted) {
    const last = merged.at(-1);
    if (!last || interval.start > last.end) {
      merged.push({ ...interval });
    } else {
      last.end = Math.max(last.end, interval.end);
    }
  }
  return merged;
}

export function summarizeCoverage(intervals) {
  return intervals.reduce((sum, interval) => sum + Math.max(0, interval.end - interval.start), 0);
}

export function overlapSeconds(a, b) {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

export function findGaps(baseIntervals, coverIntervals) {
  const base = mergeIntervals(baseIntervals);
  const cover = mergeIntervals(coverIntervals);
  const gaps = [];

  for (const interval of base) {
    let cursor = interval.start;
    for (const blocker of cover) {
      if (blocker.end <= cursor) continue;
      if (blocker.start >= interval.end) break;
      if (blocker.start > cursor) gaps.push({ start: cursor, end: Math.min(blocker.start, interval.end) });
      cursor = Math.max(cursor, blocker.end);
      if (cursor >= interval.end) break;
    }
    if (cursor < interval.end) gaps.push({ start: cursor, end: interval.end });
  }

  return gaps.filter((gap) => gap.end > gap.start);
}

export function findSuspiciousSegments(segments, longSegmentSeconds = DEFAULT_THRESHOLDS.longSegmentSeconds) {
  return segments
    .map((segment) => ({ ...segment, duration: segment.end - segment.start }))
    .filter((segment) => segment.duration >= longSegmentSeconds);
}

export function iou(a, b) {
  const overlap = overlapSeconds(a, b);
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union > 0 ? overlap / union : 0;
}

export function alignSegments(aSegments, bSegments, minIoU = DEFAULT_THRESHOLDS.minIoU) {
  const matches = [];
  const usedB = new Set();

  for (const segmentA of aSegments) {
    let best = null;
    for (const segmentB of bSegments) {
      if (usedB.has(segmentB.originalIndex)) continue;
      const score = iou(segmentA, segmentB);
      if (!best || score > best.iou) best = { a: segmentA, b: segmentB, iou: score };
    }
    if (best && best.iou >= minIoU) {
      usedB.add(best.b.originalIndex);
      matches.push(best);
    }
  }

  const matchedA = new Set(matches.map((match) => match.a.originalIndex));
  return {
    matches,
    unmatchedA: aSegments.filter((segment) => !matchedA.has(segment.originalIndex)),
    unmatchedB: bSegments.filter((segment) => !usedB.has(segment.originalIndex)),
  };
}

export function tokenize(text) {
  return String(text)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .match(/[\p{L}\p{N}]+/gu) || [];
}

function editDistance(a, b) {
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 1; i <= a.length; i += 1) {
    const current = [i];
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
    previous.splice(0, previous.length, ...current);
  }
  return previous[b.length];
}

export function compareText(referenceText, candidateText) {
  const reference = tokenize(referenceText);
  const candidate = tokenize(candidateText);
  const distance = editDistance(reference, candidate);
  return {
    distance,
    referenceTokens: reference.length,
    candidateTokens: candidate.length,
    wer: reference.length ? distance / reference.length : candidate.length ? 1 : 0,
  };
}

export function buildSpeakerMapping(aSegments, bSegments) {
  const matrix = new Map();
  for (const b of bSegments) {
    if (!matrix.has(b.speaker)) matrix.set(b.speaker, new Map());
    for (const a of aSegments) {
      const overlap = overlapSeconds(a, b);
      if (overlap <= 0) continue;
      const row = matrix.get(b.speaker);
      row.set(a.speaker, (row.get(a.speaker) || 0) + overlap);
    }
  }

  const mapping = new Map();
  const warnings = [];
  for (const [speakerB, overlaps] of matrix.entries()) {
    const ranked = [...overlaps.entries()].sort((left, right) => right[1] - left[1]);
    if (ranked.length > 0) mapping.set(speakerB, ranked[0][0]);
    if (ranked.length > 1 && ranked[1][1] / ranked[0][1] > 0.35) {
      warnings.push({ type: "speaker-merge-risk", speakerB, candidates: ranked.slice(0, 3) });
    }
  }

  const reverse = new Map();
  for (const [speakerB, speakerA] of mapping.entries()) {
    if (!reverse.has(speakerA)) reverse.set(speakerA, []);
    reverse.get(speakerA).push(speakerB);
  }
  for (const [speakerA, speakersB] of reverse.entries()) {
    if (speakersB.length > 1) warnings.push({ type: "speaker-fragmentation-risk", speakerA, speakersB });
  }

  return { matrix, mapping, warnings };
}

export function compareTranscriptions(inputA, inputB, options = {}) {
  const thresholds = { ...DEFAULT_THRESHOLDS, ...(options.thresholds || {}) };
  const normalizedA = normalizeInput(inputA, "A");
  const normalizedB = normalizeInput(inputB, "B");
  const segmentsA = normalizedA.segments;
  const segmentsB = normalizedB.segments;
  if (segmentsA.length === 0 || segmentsB.length === 0) throw new Error("Both files must contain at least one valid segment");

  const unionA = mergeIntervals(segmentsA);
  const unionB = mergeIntervals(segmentsB);
  const alignment = alignSegments(segmentsA, segmentsB, thresholds.minIoU);
  const speaker = buildSpeakerMapping(segmentsA, segmentsB);
  const matched = alignment.matches.map((match) => {
    const text = compareText(match.a.text, match.b.text);
    const mappedSpeaker = speaker.mapping.get(match.b.speaker) || "unknown";
    return { ...match, text, mappedSpeaker, speakerMismatch: mappedSpeaker !== match.a.speaker };
  });

  const timelineStart = Math.min(...segmentsA.map((s) => s.start), ...segmentsB.map((s) => s.start));
  const timelineEnd = Math.max(...segmentsA.map((s) => s.end), ...segmentsB.map((s) => s.end));
  const timelineSeconds = Math.max(0, timelineEnd - timelineStart);
  const coverageASeconds = summarizeCoverage(unionA);
  const coverageBSeconds = summarizeCoverage(unionB);

  return {
    labels: { a: options.labelA || "A", b: options.labelB || "B" },
    thresholds,
    warnings: [...normalizedA.warnings, ...normalizedB.warnings, ...speaker.warnings],
    segments: { a: segmentsA, b: segmentsB },
    timeline: { start: timelineStart, end: timelineEnd, seconds: timelineSeconds },
    coverage: {
      unionA,
      unionB,
      coverageASeconds,
      coverageBSeconds,
      coverageAPercent: timelineSeconds ? coverageASeconds / timelineSeconds : 0,
      coverageBPercent: timelineSeconds ? coverageBSeconds / timelineSeconds : 0,
      missingFromB: findGaps(unionA, unionB),
      missingFromA: findGaps(unionB, unionA),
      suspiciousA: findSuspiciousSegments(segmentsA, thresholds.longSegmentSeconds),
      suspiciousB: findSuspiciousSegments(segmentsB, thresholds.longSegmentSeconds),
    },
    alignment: { ...alignment, matches: matched },
    speaker,
  };
}

export function activeSegmentsAt(segments, timeSeconds) {
  return segments.filter((segment) => segment.start <= timeSeconds && timeSeconds <= segment.end);
}
