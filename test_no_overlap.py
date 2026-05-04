#!/usr/bin/env python3
"""Test that http-final.json has no overlapping segments and consistent derived fields."""
import json
import sys

path = "outputs/audios/RNE14-agosto-13.mp3/whisper-medium/http-final.json"

try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"SKIP: {path} not found — run the server first, then re-run this test.")
    sys.exit(0)

issues = []

# --- Check 1: lines / segments non-overlapping ---
lines = data.get("lines", [])
segs = data.get("segments", lines)  # fall back to "lines" key if "segments" absent

if not segs:
    issues.append("No segments found in the JSON output.")
else:
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        a_end = a.get("end", 0.0)
        b_start = b.get("start", 0.0)
        if a_end > b_start + 1e-6:  # 1 ms tolerance for float rounding
            issues.append(
                f"Overlap at index {i}->{i+1}: "
                f"seg[{i}].end={a_end} > seg[{i+1}].start={b_start}  "
                f"(texts: {repr(a.get('text','')[:40])} / {repr(b.get('text','')[:40])})"
            )

# --- Check 2: no near-exact substring duplication between adjacent segments ---
for i in range(len(segs) - 1):
    a_text = (segs[i].get("text") or "").strip()[:50]
    b_text = (segs[i + 1].get("text") or "").strip()[:50]
    if a_text and b_text and len(a_text) > 10:
        if a_text in b_text or b_text in a_text:
            issues.append(
                f"Adjacent duplicate text at index {i}->{i+1}: "
                f"{repr(a_text)} / {repr(b_text)}"
            )

# --- Check 3: 'transcript' field exists and count matches segments ---
if "transcript" not in data:
    issues.append("Missing 'transcript' field in JSON output.")
else:
    transcript = data["transcript"]
    if len(transcript) != len(segs):
        issues.append(
            f"'transcript' count ({len(transcript)}) != segments count ({len(segs)})"
        )

# --- Check 4: 'diarization' field exists and count matches segments ---
if "diarization" not in data:
    issues.append("Missing 'diarization' field in JSON output.")
else:
    diarization = data["diarization"]
    if len(diarization) != len(segs):
        issues.append(
            f"'diarization' count ({len(diarization)}) != segments count ({len(segs)})"
        )

# --- Summary ---
print(f"Checked {len(segs)} segments in {path}")
if issues:
    print(f"\nFAIL — {len(issues)} issue(s) found:")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)
else:
    print("OK — no overlaps, no adjacent duplicates, transcript and diarization counts match.")
    sys.exit(0)
