import { activeSegmentsAt, compareTranscriptions } from "./compare.js";

const elements = {
  labelA: document.querySelector("#label-a"),
  labelB: document.querySelector("#label-b"),
  fileA: document.querySelector("#file-a"),
  fileB: document.querySelector("#file-b"),
  audioFile: document.querySelector("#audio-file"),
  runButton: document.querySelector("#run-button"),
  messages: document.querySelector("#messages"),
  summary: document.querySelector("#summary"),
  timeline: document.querySelector("#timeline"),
  sideBySide: document.querySelector("#side-by-side"),
  tables: document.querySelector("#tables"),
  audio: document.querySelector("#audio"),
};

let latestReport = null;
let audioObjectUrl = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
}

function formatSeconds(value) {
  return `${value.toFixed(2)}s`;
}

async function readJsonFile(input, label) {
  const file = input.files?.[0];
  if (!file) throw new Error(`Choose ${label} JSON first`);
  return JSON.parse(await file.text());
}

function renderMessages(report) {
  const warnings = report.warnings.map((warning) => `<div class="message warn">${escapeHtml(warning.reason || warning.type)} ${escapeHtml(warning.source || "")}</div>`);
  elements.messages.innerHTML = warnings.join("") || '<div class="message">Comparison completed.</div>';
}

function renderSummary(report) {
  const missingBSeconds = report.coverage.missingFromB.reduce((sum, gap) => sum + gap.end - gap.start, 0);
  const missingASeconds = report.coverage.missingFromA.reduce((sum, gap) => sum + gap.end - gap.start, 0);
  const meanWer = report.alignment.matches.length
    ? report.alignment.matches.reduce((sum, match) => sum + match.text.wer, 0) / report.alignment.matches.length
    : 0;
  const suspicious = report.coverage.suspiciousA.length + report.coverage.suspiciousB.length;

  elements.summary.innerHTML = [
    ["Coverage A", `${(report.coverage.coverageAPercent * 100).toFixed(1)}%`],
    ["Coverage B", `${(report.coverage.coverageBPercent * 100).toFixed(1)}%`],
    ["Missing from B", `${formatSeconds(missingBSeconds)} / ${report.coverage.missingFromB.length}`],
    ["Missing from A", `${formatSeconds(missingASeconds)} / ${report.coverage.missingFromA.length}`],
    ["Suspicious long", String(suspicious)],
    ["Mean WER", meanWer.toFixed(2)],
  ].map(([name, value]) => `<div class="metric"><span>${escapeHtml(name)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function setupAudio() {
  const file = elements.audioFile.files?.[0];
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = null;
  if (!file) {
    elements.audio.hidden = true;
    elements.audio.removeAttribute("src");
    return;
  }
  audioObjectUrl = URL.createObjectURL(file);
  elements.audio.src = audioObjectUrl;
  elements.audio.hidden = false;
}

function percent(report, value) {
  return report.timeline.seconds ? ((value - report.timeline.start) / report.timeline.seconds) * 100 : 0;
}

function playInterval(start, end) {
  if (!elements.audio.src) return;
  const context = latestReport?.thresholds.contextSeconds || 1;
  elements.audio.currentTime = Math.max(0, start - context);
  const stopAt = end + context;
  elements.audio.play();
  const stop = () => {
    if (elements.audio.currentTime >= stopAt) {
      elements.audio.pause();
      elements.audio.removeEventListener("timeupdate", stop);
    }
  };
  elements.audio.addEventListener("timeupdate", stop);
}

function renderBars(report, segments, className) {
  return segments.map((segment) => {
    const left = percent(report, segment.start);
    const width = Math.max(0.2, percent(report, segment.end) - left);
    const isLong = segment.end - segment.start >= report.thresholds.longSegmentSeconds;
    return `<button class="bar ${className} ${isLong ? "long" : ""}" style="left:${left}%;width:${width}%" title="${escapeHtml(segment.start.toFixed(2))}-${escapeHtml(segment.end.toFixed(2))}: ${escapeHtml(segment.text)}" data-start="${segment.start}" data-end="${segment.end}"></button>`;
  }).join("");
}

function renderMissingBars(report, intervals) {
  return intervals.map((interval) => {
    const left = percent(report, interval.start);
    const width = Math.max(0.2, percent(report, interval.end) - left);
    return `<button class="bar missing" style="left:${left}%;width:${width}%" title="Missing ${interval.start.toFixed(2)}-${interval.end.toFixed(2)}" data-start="${interval.start}" data-end="${interval.end}"></button>`;
  }).join("");
}

function attachPlayback(container) {
  container.querySelectorAll("[data-start][data-end]").forEach((node) => {
    node.addEventListener("click", () => playInterval(Number(node.dataset.start), Number(node.dataset.end)));
  });
}

function renderTranscriptColumn(label, side, segments) {
  return `
    <article class="transcript-column" data-side="${side}">
      <header class="transcript-header">
        <h3>${escapeHtml(label)}</h3>
        <span>${segments.length} segments</span>
      </header>
      <div class="segment-list">
        ${segments.map((segment) => `
          <button class="segment-card" data-side="${side}" data-index="${segment.originalIndex}" data-start="${segment.start}" data-end="${segment.end}">
            <span class="segment-meta">${escapeHtml(segment.start.toFixed(2))}-${escapeHtml(segment.end.toFixed(2))} · Speaker ${escapeHtml(segment.speaker)}</span>
            <span class="segment-text">${escapeHtml(segment.text)}</span>
          </button>
        `).join("")}
      </div>
    </article>
  `;
}

function updateRealtimeHighlights() {
  if (!latestReport) return;
  const time = elements.audio.currentTime;
  const activeA = new Set(activeSegmentsAt(latestReport.segments.a, time).map((segment) => String(segment.originalIndex)));
  const activeB = new Set(activeSegmentsAt(latestReport.segments.b, time).map((segment) => String(segment.originalIndex)));

  elements.sideBySide.querySelectorAll(".segment-card").forEach((node) => {
    const activeSet = node.dataset.side === "a" ? activeA : activeB;
    const isActive = activeSet.has(node.dataset.index);
    node.classList.toggle("active", isActive);
    if (isActive) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

function renderSideBySide(report) {
  elements.sideBySide.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Side-by-side Playback Review</h2>
        <p>Click a segment to play from its start. While audio plays, active segments are highlighted in both transcripts.</p>
      </div>
    </div>
    <div class="side-by-side-grid">
      ${renderTranscriptColumn(report.labels.a, "a", report.segments.a)}
      ${renderTranscriptColumn(report.labels.b, "b", report.segments.b)}
    </div>
  `;
  attachPlayback(elements.sideBySide);
  updateRealtimeHighlights();
}

function renderTimeline(report) {
  elements.timeline.innerHTML = `
    <h2>Coverage Timeline</h2>
    <div class="timeline-row"><strong>${escapeHtml(report.labels.a)}</strong><div class="track">${renderBars(report, report.segments.a, "a")}</div></div>
    <div class="timeline-row"><strong>${escapeHtml(report.labels.b)}</strong><div class="track">${renderBars(report, report.segments.b, "b")}${renderMissingBars(report, report.coverage.missingFromB)}</div></div>
  `;
  attachPlayback(elements.timeline);
}

function table(title, rows, columns) {
  const body = rows.length
    ? rows.map((row) => `<tr>${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}</tr>`).join("")
    : `<tr><td colspan="${columns.length}">No rows</td></tr>`;
  return `<section class="panel"><h2>${escapeHtml(title)}</h2><table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></section>`;
}

function renderTables(report) {
  const intervalColumns = [
    { label: "Start", render: (row) => escapeHtml(row.start.toFixed(2)) },
    { label: "End", render: (row) => escapeHtml(row.end.toFixed(2)) },
    { label: "Duration", render: (row) => escapeHtml((row.end - row.start).toFixed(2)) },
    { label: "Audio", render: (row) => `<button data-start="${row.start}" data-end="${row.end}">Play</button>` },
  ];
  const divergent = report.alignment.matches.filter((match) => match.text.wer >= report.thresholds.highWer);
  elements.tables.innerHTML = [
    table("Missing from B", report.coverage.missingFromB, intervalColumns),
    table("Missing from A", report.coverage.missingFromA, intervalColumns),
    table("Matched but divergent", divergent, [
      { label: "A time", render: (row) => `${escapeHtml(row.a.start.toFixed(2))}-${escapeHtml(row.a.end.toFixed(2))}` },
      { label: "B time", render: (row) => `${escapeHtml(row.b.start.toFixed(2))}-${escapeHtml(row.b.end.toFixed(2))}` },
      { label: "IoU", render: (row) => escapeHtml(row.iou.toFixed(2)) },
      { label: "WER", render: (row) => escapeHtml(row.text.wer.toFixed(2)) },
      { label: "A text", render: (row) => escapeHtml(row.a.text) },
      { label: "B text", render: (row) => escapeHtml(row.b.text) },
      { label: "Audio", render: (row) => `<button data-start="${Math.max(row.a.start, row.b.start)}" data-end="${Math.min(row.a.end, row.b.end)}">Play</button>` },
    ]),
    table("Speaker mismatches", report.alignment.matches.filter((match) => match.speakerMismatch), [
      { label: "Time", render: (row) => `${escapeHtml(row.a.start.toFixed(2))}-${escapeHtml(row.a.end.toFixed(2))}` },
      { label: "A speaker", render: (row) => escapeHtml(row.a.speaker) },
      { label: "B speaker", render: (row) => escapeHtml(row.b.speaker) },
      { label: "Mapped B speaker", render: (row) => escapeHtml(row.mappedSpeaker) },
    ]),
    table("Suspicious spans", [...report.coverage.suspiciousA, ...report.coverage.suspiciousB], [
      { label: "Source", render: (row) => escapeHtml(row.source) },
      { label: "Start", render: (row) => escapeHtml(row.start.toFixed(2)) },
      { label: "End", render: (row) => escapeHtml(row.end.toFixed(2)) },
      { label: "Duration", render: (row) => escapeHtml(row.duration.toFixed(2)) },
      { label: "Text", render: (row) => escapeHtml(row.text) },
    ]),
  ].join("");
  attachPlayback(elements.tables);
}

elements.runButton.addEventListener("click", async () => {
  try {
    elements.messages.innerHTML = '<div class="message">Reading files...</div>';
    const inputA = await readJsonFile(elements.fileA, "A");
    const inputB = await readJsonFile(elements.fileB, "B");
    latestReport = compareTranscriptions(inputA, inputB, { labelA: elements.labelA.value, labelB: elements.labelB.value });
    setupAudio();
    renderMessages(latestReport);
    renderSummary(latestReport);
    renderTimeline(latestReport);
    renderSideBySide(latestReport);
    renderTables(latestReport);
  } catch (error) {
    elements.messages.innerHTML = `<div class="message error">${escapeHtml(error.message)}</div>`;
  }
});

elements.audio.addEventListener("timeupdate", updateRealtimeHighlights);
elements.audio.addEventListener("seeked", updateRealtimeHighlights);
