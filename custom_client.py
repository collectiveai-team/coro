#!/usr/bin/env python3
"""Custom headless client for WhisperLiveKit.

Supports WebSocket (/asr) and OpenAI HTTP (/v1/audio/transcriptions) modes.
Optionally writes intermediate responses (JSONL), final response (JSON),
and concatenated plain-text transcription to output files.

Usage:
    # WebSocket mode (default):
    python custom_client.py audio.wav

    # OpenAI HTTP mode:
    python custom_client.py audio.wav --openai --url http://localhost:8000

    # All three output files:
    python custom_client.py audio.wav \\
        --intermediate-output intermediates.jsonl \\
        --final-output final.json \\
        --concat-output transcript.txt

    # HTTP mode + final + concat (intermediates not available in HTTP mode):
    python custom_client.py audio.wav --openai \\
        --final-output final.json \\
        --concat-output transcript.txt
"""

import argparse
import json
import logging
import struct
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="custom-client",
        description=(
            "Headless transcription client for WhisperLiveKit. "
            "Supports WebSocket and OpenAI HTTP modes with optional file outputs."
        ),
    )
    # Positional
    parser.add_argument("audio", help="Path to audio file (wav, mp3, flac, ...)")

    # Transport
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/asr",
        help=(
            "WebSocket endpoint URL (default: ws://localhost:8000/asr). "
            "In --openai mode, used as HTTP base URL; ws:// prefix is auto-converted "
            "to http:// when --openai is active."
        ),
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI HTTP mode (POST /v1/audio/transcriptions) instead of WebSocket.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use SSE streaming in --openai mode (requires --openai).",
    )

    # WS tuning
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time, 0 = fastest, default: 1.0)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Chunk duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for server after audio ends (default: 60)",
    )

    # Language
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Transcription language override (e.g. en, fr, auto)",
    )

    # Stdout output modes (inherited from test_client)
    parser.add_argument("--json", action="store_true", help="Output raw JSON responses to stdout")
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Use diff protocol (WebSocket mode only)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Print transcription updates as they arrive (WebSocket mode only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    # File outputs
    parser.add_argument(
        "--intermediate-output",
        metavar="FILE",
        help=(
            "Append each JSON response as a line (JSONL) to FILE as they arrive. "
            "WebSocket mode only; warns and skips in --openai mode."
        ),
    )
    parser.add_argument(
        "--final-output",
        metavar="FILE",
        help="Write the last response as compact JSON to FILE after session ends.",
    )
    parser.add_argument(
        "--concat-output",
        metavar="FILE",
        help=("Write the committed transcription text to FILE (no timestamps, no speakers). Works in both modes."),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Exit with error on incompatible flag combinations."""
    if args.stream and not args.openai:
        print("Error: --stream requires --openai.", file=sys.stderr)
        sys.exit(1)
    if args.openai and args.diff:
        print("Error: --openai and --diff are incompatible.", file=sys.stderr)
        sys.exit(1)
    if args.openai and not args.stream and args.live:
        print("Error: --openai and --live are incompatible (use --stream for live HTTP output).", file=sys.stderr)
        sys.exit(1)
    if args.openai and not args.stream and args.intermediate_output:
        print(
            "Warning: --intermediate-output is ignored in --openai (HTTP) mode without --stream.",
            file=sys.stderr,
        )


def resolve_url(args: argparse.Namespace) -> str:
    """Return the effective URL for the selected mode.

    In --openai mode, auto-convert ws://host/path to http://host.
    """
    url = args.url
    if args.openai:
        if url.startswith("ws://"):
            # Strip path and convert scheme: ws://localhost:8000/asr -> http://localhost:8000
            parsed = urlparse(url)
            url = f"http://{parsed.netloc}"
        elif url.startswith("wss://"):
            parsed = urlparse(url)
            url = f"https://{parsed.netloc}"
    return url


def write_outputs(
    result,  # TranscriptionResult
    args: argparse.Namespace,
) -> None:
    """Write file outputs after session ends."""
    if args.final_output:
        if result.responses:
            Path(args.final_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.final_output, "w") as f:
                f.write(json.dumps(result.responses[-1]) + "\n")
        else:
            logger.warning("No responses to write to --final-output")

    if args.concat_output:
        text = _complete_text(result)
        Path(args.concat_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.concat_output, "w") as f:
            f.write(text + "\n")


def _complete_text(result) -> str:
    """Return final text including pending diarization/transcription buffers."""
    if not result.responses:
        return ""
    for resp in reversed(result.responses):
        parts = [line["text"] for line in resp.get("lines", []) if line.get("text")]
        for key in ("buffer_diarization", "buffer_transcription"):
            value = (resp.get(key) or "").strip()
            if value and value not in parts:
                parts.append(value)
        if parts:
            return " ".join(parts)
    return ""


def _promote_pending_buffers(result) -> None:
    """Make final pending buffers visible to the inherited printer/json output."""
    if not result.responses:
        return
    resp = result.responses[-1]
    lines = resp.setdefault("lines", [])
    existing_text = {(line.get("text") or "").strip() for line in lines}
    joined_text = " ".join(text for text in existing_text if text).strip()
    start = lines[-1].get("end", "") if lines else ""
    for key in ("buffer_diarization", "buffer_transcription"):
        value = (resp.get(key) or "").strip()
        if not value or value in existing_text or value in joined_text:
            continue
        lines.append({"text": value, "start": start, "end": "", "speaker": "-1"})
        existing_text.add(value)
        resp[key] = ""


def _make_intermediate_callback(intermediate_output: Optional[str]):
    """Return an on_response callback that appends JSONL to a file, or None.

    The file is opened once (append mode) and kept open for the session lifetime.
    The caller is responsible for not using the callback after the session ends.
    """
    if not intermediate_output:
        return None

    Path(intermediate_output).parent.mkdir(parents=True, exist_ok=True)
    fh = open(intermediate_output, "a")  # noqa: SIM115 — intentionally held open

    def callback(data: dict) -> None:
        fh.write(json.dumps(data) + "\n")
        fh.flush()

    return callback


async def run_ws(args: argparse.Namespace, url: str):
    from whisperlivekit.test_client import transcribe_audio

    on_response = _make_intermediate_callback(args.intermediate_output)

    if args.live:

        def live_callback(data: dict) -> None:
            lines = data.get("lines", [])
            buf = data.get("buffer_transcription", "")
            diarization_buf = data.get("buffer_diarization", "")
            parts = [ln["text"] for ln in lines if ln.get("text")]
            if diarization_buf:
                parts.append(f"[{diarization_buf}]")
            if buf:
                parts.append(f"[{buf}]")
            if parts:
                print("\r" + " ".join(parts), end="", flush=True)

        # Chain: live_callback runs first, then intermediate
        original_on_response = on_response

        def combined_callback(data: dict) -> None:
            live_callback(data)
            if original_on_response:
                original_on_response(data)

        on_response = combined_callback

    # Build URL query params
    params = {}
    if args.language:
        params["language"] = args.language
    if args.diff:
        params["mode"] = "diff"
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode(params)}"

    result = await transcribe_audio(
        audio_path=str(args.audio),
        url=url,
        chunk_duration=args.chunk_duration,
        speed=args.speed,
        timeout=args.timeout,
        on_response=on_response,
        mode="diff" if args.diff else "full",
    )
    return result


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw s16le mono PCM bytes in a minimal WAV header."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


async def run_http(args: argparse.Namespace, base_url: str):
    try:
        import httpx
    except ImportError:
        print(
            "Error: httpx is required for --openai mode.\nInstall it with: pip install httpx",
            file=sys.stderr,
        )
        sys.exit(1)

    from whisperlivekit.test_client import BYTES_PER_SAMPLE, SAMPLE_RATE, TranscriptionResult, load_audio_pcm

    pcm_data = load_audio_pcm(str(args.audio))
    audio_duration = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    wav_bytes = _pcm_to_wav(pcm_data, sample_rate=SAMPLE_RATE)

    data = {"response_format": "verbose_json"}
    if args.language:
        data["language"] = args.language

    async with httpx.AsyncClient(timeout=args.timeout + audio_duration + 30) as client:
        response = await client.post(
            f"{base_url}/v1/audio/transcriptions",
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        if response.status_code != 200:
            print(
                f"Error: server returned {response.status_code}\n{response.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        body = response.json()

    segments = body.get("segments", [])
    lines = [
        {
            "text": seg.get("text", "").strip(),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": seg.get("speaker", ""),
        }
        for seg in segments
        if seg.get("text", "").strip()
    ]

    result = TranscriptionResult()
    result.audio_duration = audio_duration
    result.responses.append({"lines": lines, "buffer_transcription": ""})
    return result


async def run_http_stream(args: argparse.Namespace, base_url: str):
    try:
        import httpx
    except ImportError:
        print(
            "Error: httpx is required for --openai mode.\nInstall it with: pip install httpx",
            file=sys.stderr,
        )
        sys.exit(1)

    from whisperlivekit.test_client import BYTES_PER_SAMPLE, SAMPLE_RATE, TranscriptionResult, load_audio_pcm

    pcm_data = load_audio_pcm(str(args.audio))
    audio_duration = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    wav_bytes = _pcm_to_wav(pcm_data, sample_rate=SAMPLE_RATE)

    data = {"stream": "true"}
    if args.language:
        data["language"] = args.language

    intermediate_fh = None
    if args.intermediate_output:
        Path(args.intermediate_output).parent.mkdir(parents=True, exist_ok=True)
        intermediate_fh = open(args.intermediate_output, "a")  # noqa: SIM115

    accumulated_text = ""
    final_whisperx = None

    async with httpx.AsyncClient(timeout=args.timeout + audio_duration + 30) as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/audio/transcriptions",
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(
                    f"Error: server returned {response.status_code}\n{body.decode()}",
                    file=sys.stderr,
                )
                sys.exit(1)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "transcript.text.delta":
                    delta = event.get("delta", "")
                    accumulated_text += delta
                    if args.live:
                        print(delta, end="", flush=True)
                    if intermediate_fh:
                        intermediate_fh.write(json.dumps(event) + "\n")
                        intermediate_fh.flush()

                elif event_type == "transcript.text.done":
                    final_whisperx = event.get("text")
                    if intermediate_fh:
                        intermediate_fh.write(json.dumps(event) + "\n")
                        intermediate_fh.flush()

                elif event_type == "error":
                    print(f"\nError from server: {event.get('error', 'unknown')}", file=sys.stderr)

    if intermediate_fh:
        intermediate_fh.close()

    if args.live:
        print()

    result = TranscriptionResult()
    result.audio_duration = audio_duration

    if final_whisperx:
        try:
            body = json.loads(final_whisperx) if isinstance(final_whisperx, str) else final_whisperx
        except json.JSONDecodeError:
            body = {}
        segments = body.get("segments", [])
        lines = [
            {
                "text": seg.get("text", "").strip(),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg.get("speaker", ""),
            }
            for seg in segments
            if seg.get("text", "").strip()
        ]
        result.responses.append({"lines": lines, "buffer_transcription": ""})
    elif accumulated_text.strip():
        result.responses.append({"lines": [{"text": accumulated_text.strip()}], "buffer_transcription": ""})
    else:
        result.responses.append({"lines": [], "buffer_transcription": ""})

    return result


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    validate_args(args)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    url = resolve_url(args)

    import asyncio

    if args.openai:
        if args.stream:
            result = asyncio.run(run_http_stream(args, url))
        else:
            result = asyncio.run(run_http(args, url))
    else:
        result = asyncio.run(run_ws(args, url))

    if args.live:
        print()  # newline after live output

    from whisperlivekit.test_client import _print_result

    _promote_pending_buffers(result)
    _print_result(result, output_json=args.json)

    write_outputs(result, args)


if __name__ == "__main__":
    main()
