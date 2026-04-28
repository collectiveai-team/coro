import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from whisperlivekit import (
    AudioProcessor,
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.qwen3_asr").setLevel(logging.DEBUG)

config = parse_args()
transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(config=config)

    # Patch FasterWhisper transcribe_kargs with anti-repetition parameters.
    # transcribe_kargs is spread into model.transcribe() via **self.transcribe_kargs,
    # so any faster-whisper transcribe() kwarg can be injected here.
    asr = getattr(transcription_engine, "asr", None)
    if asr is not None and hasattr(asr, "transcribe_kargs"):
        asr.transcribe_kargs.update({
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 1.8,
            "no_speech_threshold": 0.45,
            "logprob_threshold": -0.8,
        })
        logger.info("Applied anti-repetition transcribe_kargs to FasterWhisperASR: %s", asr.transcribe_kargs)

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/health")
async def health():
    """Health check endpoint."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else None
    return JSONResponse(
        {
            "status": "ok",
            "backend": backend,
            "ready": transcription_engine is not None,
        }
    )


async def handle_websocket_results(websocket, results_generator, diff_tracker=None):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            if diff_tracker is not None:
                await websocket.send_json(diff_tracker.to_message(response))
            else:
                await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine

    # Read per-session options from query parameters
    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=session_language,
    )
    await websocket.accept()
    logger.info(
        "WebSocket connection opened.%s",
        f" language={session_language}" if session_language else "",
    )
    diff_tracker = None
    if mode == "diff":
        from whisperlivekit.diff_protocol import DiffTracker

        diff_tracker = DiffTracker()
        logger.info("Client requested diff mode")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(config.pcm_input), "mode": mode})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, diff_tracker))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


# ---------------------------------------------------------------------------
# Deepgram-compatible WebSocket API  (/v1/listen)
# ---------------------------------------------------------------------------


@app.websocket("/v1/listen")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """Deepgram-compatible live transcription WebSocket."""
    global transcription_engine
    from whisperlivekit.deepgram_compat import handle_deepgram_websocket

    await handle_deepgram_websocket(websocket, transcription_engine, config)


# ---------------------------------------------------------------------------
# WhisperX REST API  (/v1/audio/transcriptions)
# ---------------------------------------------------------------------------


async def _convert_to_pcm(audio_bytes: bytes) -> bytes:
    """Convert any audio format to PCM s16le mono 16kHz using ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "error",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)
    if proc.returncode != 0:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Audio conversion failed: {stderr.decode().strip()}",
        )
    return stdout


def _parse_time_str(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _segment_end_seconds(seg: dict) -> float:
    end = seg.get("end", 0.0)
    if isinstance(end, str):
        return _parse_time_str(end) if end else 0.0
    return float(end or 0.0)


def _append_pending_text_segment(result: dict, pending_text: str, duration: float) -> None:
    """Preserve text that is recognized but not yet speaker-attributed."""
    pending_text = (pending_text or "").strip()
    if not pending_text:
        return

    segments = result["segments"]
    existing_text = " ".join((seg.get("text") or "").strip() for seg in segments).strip()
    if pending_text == existing_text or pending_text in existing_text:
        return

    start = max((_segment_end_seconds(seg) for seg in segments), default=0.0)
    end = max(start, duration)
    raw_words = pending_text.split()
    words = []
    if raw_words:
        word_duration = (end - start) / len(raw_words) if end > start else 0.0
        for j, word in enumerate(raw_words):
            word_entry = {
                "word": word,
                "start": round(start + j * word_duration, 2),
                "end": round(start + (j + 1) * word_duration, 2),
                "score": 1.0,
                "speaker": "-1",
            }
            words.append(word_entry)
            result["word_segments"].append(word_entry.copy())

    segments.append(
        {
            "start": round(start, 2),
            "end": round(end, 2),
            "text": pending_text,
            "words": words,
            "speaker": "-1",
        }
    )


def _to_whisperx(front_data, duration: float = 0.0) -> dict:
    """Convert FrontData to WhisperX JSON schema.

    Returns:
        {
          "segments": [{"start", "end", "text", "speaker", "words": [...]}],
          "word_segments": [{"word", "start", "end", "score", "speaker"}]
        }
    Speaker -2 == silence; those segments are excluded.
    Speaker -1 == diarization not yet assigned; emitted as -1.
    Word timestamps are linearly interpolated within each segment.
    score is always 1.0 (no per-token confidence available).

    Reads from front_data.lines (raw Segment objects) rather than to_dict() to
    preserve speaker=-1 (unassigned), which to_dict() would otherwise mangle to 1.
    """
    segments = []
    word_segments = []

    for seg_obj in getattr(front_data, "lines", []):
        speaker = str(int(getattr(seg_obj, "speaker", -1)))
        # -2 is the silence sentinel; skip it
        if speaker == "-2":
            continue
        text = getattr(seg_obj, "text", None) or ""
        if not text:
            continue

        start = getattr(seg_obj, "start", 0.0)
        end = getattr(seg_obj, "end", 0.0)

        # Build word list with linearly interpolated timestamps
        raw_words = text.split()
        words = []
        if raw_words:
            word_duration = (end - start) / len(raw_words)
            for j, w in enumerate(raw_words):
                word_entry: dict = {
                    "word": w,
                    "start": round(start + j * word_duration, 2),
                    "end": round(start + (j + 1) * word_duration, 2),
                    "score": 1.0,
                    "speaker": speaker,
                }
                words.append(word_entry)
                word_segments.append(word_entry.copy())  # copy avoids aliasing with words list

        seg: dict = {
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
            "words": words,
            "speaker": speaker,
        }
        segments.append(seg)

    result = {"segments": segments, "word_segments": word_segments}
    _append_pending_text_segment(result, getattr(front_data, "buffer_diarization", ""), duration)
    _append_pending_text_segment(result, getattr(front_data, "buffer_transcription", ""), duration)
    return result


def _deduplicate_segments(segments: list) -> list:
    """Remove near-duplicate segments that share text and overlap in time.

    The pipeline refines timestamps across snapshots, so the same logical
    segment can appear with slightly different start/end values. This function
    merges segments that share the same text content by keeping the one with
    the longest duration (most refined timestamps).
    """
    if not segments:
        return segments

    deduped = []
    for seg in segments:
        text = getattr(seg, "text", "").strip()
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))
        speaker = getattr(seg, "speaker", None)

        merged = False
        for existing in deduped:
            e_text = getattr(existing, "text", "").strip()
            e_start = float(getattr(existing, "start", 0.0))
            e_end = float(getattr(existing, "end", 0.0))
            e_speaker = getattr(existing, "speaker", None)

            if speaker != e_speaker:
                continue

            overlap = min(end, e_end) - max(start, e_start)
            shorter = min(end - start, e_end - e_start)
            if shorter <= 0:
                continue

            if overlap / shorter > 0.5 and (text == e_text or text in e_text or e_text in text):
                if end - start > e_end - e_start:
                    deduped.remove(existing)
                    deduped.append(seg)
                merged = True
                break

        if not merged:
            deduped.append(seg)

    return sorted(deduped, key=lambda s: float(getattr(s, "start", 0.0)))


def _extract_text_from_frontdata(front_data) -> str:
    parts = [
        getattr(seg, "text", "")
        for seg in getattr(front_data, "lines", [])
        if getattr(seg, "text", None) and getattr(seg, "speaker", None) != -2
    ]
    diarization_buffer = getattr(front_data, "buffer_diarization", "") or ""
    buffer = getattr(front_data, "buffer_transcription", "") or ""
    if diarization_buffer:
        parts.append(diarization_buffer)
    if buffer:
        parts.append(buffer)
    return " ".join(part.strip() for part in parts if part.strip())


async def _stream_transcription(results_generator, processor, response_format: str, language, duration: float):
    import json

    sent_text = ""
    front_data = None
    accumulated_segments: dict[tuple, object] = {}

    try:
        async for front_data in results_generator:
            for seg in getattr(front_data, "lines", []):
                if getattr(seg, "speaker", None) == -2:
                    continue
                if not getattr(seg, "text", None):
                    continue
                key = (
                    round(float(getattr(seg, "start", 0.0)), 2),
                    round(float(getattr(seg, "end", 0.0)), 2),
                    str(getattr(seg, "speaker", "")),
                )
                accumulated_segments[key] = seg

            current_text = _extract_text_from_frontdata(front_data)
            if len(current_text) > len(sent_text):
                delta = current_text[len(sent_text):]
                sent_text = current_text
                event_data = json.dumps({"type": "transcript.text.delta", "delta": delta})
                yield f"data: {event_data}\n\n"
    except Exception as e:
        error_data = json.dumps({"type": "error", "error": str(e)})
        yield f"data: {error_data}\n\n"
        return
    finally:
        await processor.cleanup()

    if front_data is not None and accumulated_segments:
        front_data.lines = _deduplicate_segments(list(accumulated_segments.values()))

    if front_data is not None:
        result = _to_whisperx(front_data, duration=duration)
        done_data = json.dumps({"type": "transcript.text.done", "text": json.dumps(result)})
        yield f"data: {done_data}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(
        default="",
        description="Model parameter is accepted but ignored (server uses configured backend)",
    ),
    language: str | None = Form(default=None, description="Optional language hint (e.g. 'en', 'es', 'fr')."),
    prompt: str = Form(
        default="",
        description="Optional prompt text to guide transcription (currently ignored).",
    ),
    stream: bool = Form(default=False, description="If true, return SSE streaming results."),
):
    """WhisperX-format audio transcription endpoint.

    Always returns WhisperX JSON schema:
      {"segments": [...], "word_segments": [...]}

    When stream=True, returns SSE events with partial transcription deltas.

    The `model` and `prompt` parameters are accepted but ignored.
    """
    global transcription_engine

    audio_bytes = await file.read()
    if not audio_bytes:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Empty audio file")

    pcm_data = await _convert_to_pcm(audio_bytes)
    duration = len(pcm_data) / (16000 * 2)

    processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
    )
    processor.is_pcm_input = True

    results_gen = await processor.create_tasks()

    if stream:
        async def feed_audio():
            chunk_size = 16000 * 2
            for i in range(0, len(pcm_data), chunk_size):
                await processor.process_audio(pcm_data[i : i + chunk_size])
            await processor.process_audio(b"")

        asyncio.create_task(feed_audio())

        return StreamingResponse(
            _stream_transcription(results_gen, processor, "json", language, duration),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    final_result = None
    accumulated_segments: dict[tuple, object] = {}

    async def collect():
        nonlocal final_result
        async for result in results_gen:
            for seg in getattr(result, "lines", []):
                # Skip silence sentinels (speaker == -2); they have no text.
                if getattr(seg, "speaker", None) == -2:
                    continue
                if not getattr(seg, "text", None):
                    continue
                key = (
                    round(float(getattr(seg, "start", 0.0)), 2),
                    round(float(getattr(seg, "end", 0.0)), 2),
                    str(getattr(seg, "speaker", "")),
                )
                # Last write wins so refinements (text/speaker updates) are kept.
                accumulated_segments[key] = seg
            final_result = result

    collect_task = asyncio.create_task(collect())

    # Feed audio in chunks (1 second each)
    chunk_size = 16000 * 2  # 1 second of PCM
    for i in range(0, len(pcm_data), chunk_size):
        await processor.process_audio(pcm_data[i : i + chunk_size])

    # Signal end of audio
    await processor.process_audio(b"")

    # Wait for pipeline to finish
    try:
        await asyncio.wait_for(collect_task, timeout=120.0)
    except asyncio.TimeoutError:
        logger.warning("Transcription timed out after 120s")
    finally:
        await processor.cleanup()

    if final_result is None:
        return JSONResponse({"segments": [], "word_segments": []})

    # Replace the (pruned) last snapshot's lines with the full accumulated set
    # so _to_whisperx sees every segment from the entire audio.
    if accumulated_segments:
        final_result.lines = _deduplicate_segments(list(accumulated_segments.values()))

    return JSONResponse(_to_whisperx(final_result, duration=duration))


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else "whisper"
    model_size = getattr(transcription_engine.config, "model_size", "base") if transcription_engine else "base"
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": f"{backend}/{model_size}" if backend != "whisper" else f"whisper-{model_size}",
                    "object": "model",
                    "owned_by": "whisperlivekit",
                }
            ],
        }
    )


def main():
    """Entry point for the CLI command."""
    import uvicorn

    from whisperlivekit.cli import print_banner

    ssl = bool(config.ssl_certfile and config.ssl_keyfile)
    print_banner(config, config.host, config.port, ssl=ssl)

    uvicorn_kwargs = {
        "app": "custom_server:app",
        "host": config.host,
        "port": config.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs: dict[str, str] = {}
    if config.ssl_certfile or config.ssl_keyfile:
        if not (config.ssl_certfile and config.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if config.forwarded_allow_ips:
        uvicorn_kwargs = {
            **uvicorn_kwargs,
            "forwarded_allow_ips": config.forwarded_allow_ips,
        }

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
