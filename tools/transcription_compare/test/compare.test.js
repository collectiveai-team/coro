import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  activeSegmentsAt,
  alignSegments,
  buildSpeakerMapping,
  compareTranscriptions,
  compareText,
  findGaps,
  findSuspiciousSegments,
  iou,
  mergeIntervals,
  normalizeInput,
  overlapSeconds,
  summarizeCoverage,
  tokenize,
} from "../src/compare.js";

test("normalizes custom-server lines format", () => {
  const result = normalizeInput({ lines: [{ start: 1, end: 2, text: " hola ", speaker: 3 }] }, "A");
  assert.deepEqual(result.segments, [{ source: "A", originalIndex: 0, start: 1, end: 2, text: "hola", speaker: "3" }]);
  assert.deepEqual(result.warnings, []);
});

test("normalizes OpenAI segments format", () => {
  const result = normalizeInput({ segments: [{ start: 0, end: 1.5, text: "hello", speaker: "S1" }] }, "B");
  assert.equal(result.segments[0].source, "B");
  assert.equal(result.segments[0].speaker, "S1");
});

test("normalizes direct array format and defaults missing speaker", () => {
  const result = normalizeInput([{ start: 0, end: 1, text: "text" }], "A");
  assert.equal(result.segments[0].speaker, "unknown");
});

test("normalizes Deepgram words format", async () => {
  const deepgram = JSON.parse(await readFile(new URL("./fixtures/deepgram.json", import.meta.url), "utf8"));
  const result = normalizeInput(deepgram, "B");

  assert.deepEqual(result.segments, [
    { source: "B", originalIndex: 0, start: 0.1, end: 0.8, text: "hola mundo", speaker: "0" },
    { source: "B", originalIndex: 1, start: 2.2, end: 3.0, text: "buenos dias", speaker: "1" },
  ]);
  assert.deepEqual(result.warnings, []);
});

test("reports invalid rows without throwing", () => {
  const result = normalizeInput({ lines: [{ start: 2, end: 1, text: "bad" }, { start: 0, end: 1, text: "" }] }, "A");
  assert.deepEqual(result.segments, []);
  assert.equal(result.warnings.length, 2);
  assert.match(result.warnings[0].reason, /end must be greater/);
  assert.match(result.warnings[1].reason, /text must be non-empty/);
});

test("rejects unsupported JSON shape", () => {
  assert.throws(() => normalizeInput({ text: "only text" }, "A"), /Unsupported JSON shape/);
});

test("merges overlapping intervals and computes coverage", () => {
  const merged = mergeIntervals([{ start: 0, end: 2 }, { start: 1, end: 3 }, { start: 5, end: 6 }]);
  assert.deepEqual(merged, [{ start: 0, end: 3 }, { start: 5, end: 6 }]);
  assert.equal(summarizeCoverage(merged), 4);
});

test("finds intervals covered by A but not B", () => {
  const gaps = findGaps([{ start: 0, end: 10 }], [{ start: 0, end: 2 }, { start: 4, end: 7 }]);
  assert.deepEqual(gaps, [{ start: 2, end: 4 }, { start: 7, end: 10 }]);
});

test("computes overlap seconds", () => {
  assert.equal(overlapSeconds({ start: 1, end: 4 }, { start: 3, end: 6 }), 1);
  assert.equal(overlapSeconds({ start: 1, end: 2 }, { start: 3, end: 4 }), 0);
});

test("flags suspicious long segments", () => {
  const suspicious = findSuspiciousSegments([{ start: 0, end: 90, text: "long", source: "B", originalIndex: 0, speaker: "1" }], 60);
  assert.equal(suspicious.length, 1);
  assert.equal(suspicious[0].duration, 90);
});

test("computes temporal IoU", () => {
  assert.equal(iou({ start: 0, end: 4 }, { start: 2, end: 6 }), 2 / 6);
});

test("aligns segments by best IoU and marks unmatched", () => {
  const a = [{ source: "A", originalIndex: 0, start: 0, end: 4, text: "one two", speaker: "A" }];
  const b = [{ source: "B", originalIndex: 0, start: 1, end: 3, text: "one too", speaker: "1" }];
  const result = alignSegments(a, b, 0.3);
  assert.equal(result.matches.length, 1);
  assert.equal(result.unmatchedA.length, 0);
  assert.equal(result.unmatchedB.length, 0);
});

test("leaves low-overlap segments unmatched", () => {
  const a = [{ source: "A", originalIndex: 0, start: 0, end: 1, text: "a", speaker: "A" }];
  const b = [{ source: "B", originalIndex: 0, start: 10, end: 11, text: "b", speaker: "1" }];
  const result = alignSegments(a, b, 0.3);
  assert.equal(result.matches.length, 0);
  assert.equal(result.unmatchedA.length, 1);
  assert.equal(result.unmatchedB.length, 1);
});

test("tokenizes and computes WER-like divergence", () => {
  assert.deepEqual(tokenize("Hola, mundo!"), ["hola", "mundo"]);
  const result = compareText("one two three", "one too three");
  assert.equal(result.distance, 1);
  assert.equal(result.referenceTokens, 3);
  assert.equal(result.wer, 1 / 3);
});

test("builds speaker mapping from overlap duration", () => {
  const a = [{ source: "A", originalIndex: 0, start: 0, end: 5, text: "a", speaker: "HOST" }];
  const b = [{ source: "B", originalIndex: 0, start: 1, end: 4, text: "b", speaker: "1" }];
  const result = buildSpeakerMapping(a, b);
  assert.equal(result.mapping.get("1"), "HOST");
  assert.equal(result.matrix.get("1").get("HOST"), 3);
});

test("assembles a coverage-first report", () => {
  const report = compareTranscriptions(
    { lines: [{ start: 0, end: 10, text: "one two three", speaker: "A" }] },
    { segments: [{ start: 0, end: 2, text: "one two", speaker: "1" }] },
    { labelA: "truth", labelB: "candidate" },
  );
  assert.equal(report.labels.a, "truth");
  assert.equal(report.labels.b, "candidate");
  assert.equal(report.coverage.missingFromB.length, 1);
  assert.deepEqual(report.coverage.missingFromB[0], { start: 2, end: 10 });
  assert.equal(report.alignment.unmatchedA.length, 1);
});

test("finds active segments for realtime playback highlighting", () => {
  const segments = [
    { source: "A", originalIndex: 0, start: 0, end: 2, text: "first", speaker: "A" },
    { source: "A", originalIndex: 1, start: 2, end: 4, text: "second", speaker: "A" },
    { source: "A", originalIndex: 2, start: 5, end: 6, text: "third", speaker: "B" },
  ];
  assert.deepEqual(activeSegmentsAt(segments, 1.5).map((segment) => segment.originalIndex), [0]);
  assert.deepEqual(activeSegmentsAt(segments, 2).map((segment) => segment.originalIndex), [0, 1]);
  assert.deepEqual(activeSegmentsAt(segments, 4.5), []);
});

test("smoke compares OpenAI truth fixture against custom-server final fixture", async () => {
  const truth = JSON.parse(await readFile(new URL("./fixtures/openai-truth.json", import.meta.url), "utf8"));
  const candidate = JSON.parse(await readFile(new URL("./fixtures/custom-server-final.json", import.meta.url), "utf8"));
  const report = compareTranscriptions(truth, candidate, { labelA: "truth", labelB: "nemo-parakeet" });

  assert.equal(report.labels.a, "truth");
  assert.equal(report.labels.b, "nemo-parakeet");
  assert.ok(report.coverage.coverageASeconds > 0);
  assert.ok(report.coverage.coverageBSeconds > 0);
  assert.ok(report.coverage.suspiciousB.length > 0);
  assert.ok(report.coverage.missingFromB.length > 0 || report.alignment.unmatchedA.length > 0);
});
