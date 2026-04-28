#!/usr/bin/env python3
"""
Start custom_server.py, wait for it to be healthy, run custom_client.py, then shut down.

Usage:
    uv run --no-sync python run_transcription.py [AUDIO] [extra server args...]

Defaults:
    AUDIO  = ./audios/RNE14-agosto-13.mp3
    MODEL  = medium  (set via --model, passed as extra server arg)

Extra args after AUDIO are forwarded to the server command.
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_AUDIO = "./audios/RNE14-agosto-13.mp3"
DEFAULT_MODEL = "medium"
SERVER_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
HEALTH_TIMEOUT = 120  # seconds
HEALTH_INTERVAL = 1   # seconds


def wait_for_server(timeout: int = HEALTH_TIMEOUT) -> bool:
    """Poll /health until it returns 200 or timeout is reached."""
    deadline = time.monotonic() + timeout
    print(f"Waiting for server at {HEALTH_ENDPOINT} ...", flush=True)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_ENDPOINT, timeout=2) as resp:
                if resp.status == 200:
                    print("Server is healthy.", flush=True)
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_INTERVAL)
    return False


def build_output_paths(audio: str, model: str, sse: bool = False) -> tuple[str, str, str | None]:
    """Return (final_json_path, transcript_txt_path, intermediates_jsonl_path) under outputs/.

    intermediates_jsonl_path is only set when sse=True.
    """
    # Strip leading ./ for cleaner paths
    audio_key = audio.lstrip("./")
    base = os.path.join("outputs", audio_key, model)
    os.makedirs(base, exist_ok=True)
    prefix = "sse" if sse else "http"
    intermediates = os.path.join(base, "sse-intermediates.jsonl") if sse else None
    return (
        os.path.join(base, f"{prefix}-final.json"),
        os.path.join(base, f"{prefix}-transcript.txt"),
        intermediates,
    )


def extract_model(extra_args: list[str]) -> str:
    """Extract --model value from extra args, or return DEFAULT_MODEL."""
    for i, arg in enumerate(extra_args):
        if arg == "--model" and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch server + client for audio transcription.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Path to audio file (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        default=False,
        help="Use SSE streaming endpoint instead of plain HTTP batch; saves intermediate responses to sse-intermediates.jsonl",
    )
    # Collect remaining args to forward to the server
    args, extra_server_args = parser.parse_known_args()

    audio = args.audio
    use_sse = args.sse
    model = extract_model(extra_server_args)

    # Build server command
    server_cmd = [
        "uv", "run", "--no-sync", "python", "custom_server.py",
        "--pcm-input",
        "--model", model,
        "--diarization",
        "--language", "es",
    ] + extra_server_args

    # Build client command
    final_json, transcript_txt, intermediates = build_output_paths(audio, model, sse=use_sse)
    client_cmd = [
        "uv", "run", "--no-sync", "python", "custom_client.py", audio,
        "--language", "es",
        "--openai",
        "--url", SERVER_URL,
        "--final-output", final_json,
        "--concat-output", transcript_txt,
    ]
    if use_sse:
        client_cmd += ["--stream", "--intermediate-output", intermediates]

    print("Server command:", " ".join(server_cmd), flush=True)
    print("Client command:", " ".join(client_cmd), flush=True)
    print(flush=True)

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    client_returncode = 1
    try:
        if not wait_for_server():
            print(
                f"ERROR: Server did not become healthy within {HEALTH_TIMEOUT}s.",
                file=sys.stderr,
            )
            return 1

        print("\nStarting client ...\n", flush=True)
        client_proc = subprocess.run(client_cmd, check=False)
        client_returncode = client_proc.returncode
        if client_returncode != 0:
            print(
                f"Client exited with code {client_returncode}.", file=sys.stderr
            )
    finally:
        print("\nShutting down server ...", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server did not stop gracefully; killing.", file=sys.stderr)
            server_proc.kill()
            server_proc.wait()
        print("Done.", flush=True)

    return client_returncode


if __name__ == "__main__":
    sys.exit(main())
