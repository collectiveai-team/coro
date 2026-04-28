# Transcription Compare

Standalone local webapp for comparing two diarized transcription JSON files.

## Supported JSON Formats

- Custom-server final output with top-level `lines`.
- OpenAI verbose output with top-level `segments`.
- Deepgram output with `results.channels[].alternatives[].words` or `results.utterances`.
- Direct array of `{ "start": number, "end": number, "text": string, "speaker": string }`.

## Run

```sh
cd tools/transcription_compare
npm start
```

Open `http://localhost:5177`.

## Test

```sh
cd tools/transcription_compare
npm test
```

## Workflow

1. Upload JSON A and JSON B.
2. Optionally upload the source audio file.
3. Click `Run comparison`.
4. Start with the summary cards and `Missing from B` table to find likely skipped audio.
5. Use the timeline and evidence tables to inspect divergent text, speaker mismatches, and suspicious long spans.

## Notes

Speaker labels are arbitrary between ASR/diarization systems. The app maps B speakers to A speakers by temporal overlap before reporting speaker mismatches.
