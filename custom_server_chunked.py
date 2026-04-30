import asyncio
import logging
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from whisperlivekit import (
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)
from whisperlivekit.timed_objects import ASRToken, Segment
from whisperlivekit.tokens_alignment import PuncSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.qwen3_asr").setLevel(logging.DEBUG)

config = parse_args()
transcription_engine = None
# Serialises concurrent batch transcriptions that mutate asr.original_language.
# A single GPU makes true parallelism impossible anyway; this just makes it safe.
_asr_lock = threading.Lock()


@dataclass
class _ChunkProgress:
    processed_seconds: float
    stage: str


@dataclass
class _ChunkResult:
    new_tokens: list
    new_diarization: list
    progress: _ChunkProgress


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(config=config)

    # Patch FasterWhisper transcribe_kargs with anti-repetition parameters.
    # transcribe_kargs is spread into model.transcribe() via **self.transcribe_kargs,
    # so any faster-whisper transcribe() kwarg can be injected here.
    asr = getattr(transcription_engine, "asr", None)
    if asr is not None and hasattr(asr, "transcribe_kargs"):
        # asr.transcribe_kargs.update({
        #     "condition_on_previous_text": False,
        #     "compression_ratio_threshold": 1.8,
        #     "no_speech_threshold": 0.45,
        #     "logprob_threshold": -0.8,
        # })
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


# ---------------------------------------------------------------------------
# WhisperX REST API  (/v1/audio/transcriptions)
# ---------------------------------------------------------------------------


async def _spool_upload_to_tempfile(file: UploadFile) -> str:
    fd, path = tempfile.mkstemp(prefix="aymurai-upload-", suffix=".audio")
    wrote_data = False
    try:
        with os.fdopen(fd, "wb") as temp_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                wrote_data = True
                temp_file.write(chunk)
        if not wrote_data:
            raise HTTPException(400, "Empty audio file")
        return path
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def _iter_aligned_pcm_chunks(byte_chunks, target_bytes: int):
    pending = b""
    target_bytes = max(2, target_bytes - (target_bytes % 2))
    for chunk in byte_chunks:
        if not chunk:
            continue
        pending += chunk
        while len(pending) >= target_bytes:
            yield pending[:target_bytes]
            pending = pending[target_bytes:]
        if len(pending) % 2:
            continue
        if len(pending) == target_bytes:
            yield pending
            pending = b""
    if len(pending) >= 2:
        yield pending[:len(pending) - (len(pending) % 2)]


async def _stream_pcm_chunks_from_file(path: str, chunk_seconds: float = 1.0):
    target_bytes = max(2, int(16000 * 2 * chunk_seconds))
    target_bytes -= target_bytes % 2
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        path,
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pending = b""
    stderr_chunks = []

    async def drain_stderr():
        if not proc.stderr:
            return
        total = 0
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            if total < 65536:
                stderr_chunks.append(chunk[:65536 - total])
                total += len(stderr_chunks[-1])

    stderr_task = asyncio.create_task(drain_stderr())
    try:
        while True:
            chunk = await proc.stdout.read(target_bytes) if proc.stdout else b""
            if not chunk:
                break
            pending += chunk
            aligned_len = len(pending) - (len(pending) % 2)
            while aligned_len >= target_bytes:
                yield pending[:target_bytes]
                pending = pending[target_bytes:]
                aligned_len = len(pending) - (len(pending) % 2)

        returncode = await proc.wait()
        await stderr_task
        if returncode != 0:
            stderr = b"".join(stderr_chunks)
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {stderr.decode().strip()}")
        if len(pending) >= 2:
            yield pending[:len(pending) - (len(pending) % 2)]
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass


# Legacy full-memory helper retained temporarily for comparison tests. Uploaded-file endpoints should use the disk-backed pipeline instead.
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


def _iter_batch_transcribe_chunks(pcm_bytes: bytes, language: str | None = None):
    """Yield accepted ASRToken batches for each direct faster-whisper chunk."""
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 30
    OVERLAP_SECONDS = 2
    CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE
    OVERLAP_SAMPLES = OVERLAP_SECONDS * SAMPLE_RATE

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(audio)

    asr = transcription_engine.asr
    fw_model = getattr(asr, "fw_encoder", None) or getattr(asr, "model", None)
    if fw_model is None or not hasattr(fw_model, "transcribe"):
        raise RuntimeError(
            f"Cannot find a faster-whisper WhisperModel on transcription_engine.asr "
            f"({type(asr).__name__}). Batch transcription requires faster-whisper."
        )

    lan = language or getattr(asr, "lan", None) or getattr(asr, "original_language", None)
    all_tokens = []
    prev_end_time = 0.0  # wall-clock end of last accepted token from previous chunk
    init_prompt = ""

    chunk_start_sample = 0
    while chunk_start_sample < total_samples:
        chunk_end_sample = min(chunk_start_sample + CHUNK_SAMPLES, total_samples)
        chunk_audio = audio[chunk_start_sample:chunk_end_sample]
        offset_seconds = chunk_start_sample / SAMPLE_RATE

        logger.debug(
            "Batch transcribing chunk %.1f-%.1fs (%d samples)",
            offset_seconds,
            chunk_end_sample / SAMPLE_RATE,
            len(chunk_audio),
        )

        # Call faster-whisper directly — no local-agreement, no buffer resets.
        # Lock so concurrent requests don't interfere with shared model state.
        with _asr_lock:
            raw_segs, _ = fw_model.transcribe(
                chunk_audio,
                language=lan if lan and lan != "auto" else None,
                initial_prompt=init_prompt,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True,
            )
            raw_segs = list(raw_segs)  # consume the generator inside the lock

        # Convert faster-whisper Segment.words → ASRToken, filter no_speech
        tokens = []
        for seg in raw_segs:
            if getattr(seg, "no_speech_prob", 0.0) > 0.9:
                continue
            for word in getattr(seg, "words", []):
                tokens.append(ASRToken(word.start, word.end, word.word, probability=word.probability))

        # Adjust timestamps to global audio time and build ASRToken objects
        chunk_tokens = []
        for t in tokens:
            adjusted = ASRToken(
                start=round(t.start + offset_seconds, 3),
                end=round(t.end + offset_seconds, 3),
                text=t.text,
                probability=t.probability,
            )
            chunk_tokens.append(adjusted)

        # Drop tokens from the overlap region already covered by previous chunk.
        # Use half the overlap as dedup boundary to avoid losing words at boundaries.
        dedup_boundary = prev_end_time - (OVERLAP_SECONDS * 0.5) if all_tokens else 0.0
        new_tokens = [t for t in chunk_tokens if t.start >= dedup_boundary]

        if new_tokens:
            all_tokens.extend(new_tokens)
            prev_end_time = new_tokens[-1].end

        # Build init_prompt from last ~200 chars of committed text for context
        if new_tokens:
            recent_text = "".join(t.text for t in all_tokens[-50:])
            init_prompt = recent_text[-200:] if len(recent_text) > 200 else recent_text

        yield new_tokens

        # Advance by chunk size minus overlap so next chunk re-covers the tail
        chunk_start_sample += CHUNK_SAMPLES - OVERLAP_SAMPLES


# Legacy full-memory helper retained temporarily for comparison tests. Uploaded-file endpoints should use the disk-backed pipeline instead.
def _batch_transcribe(pcm_bytes: bytes, language: str | None = None) -> list:
    """Transcribe the full audio by calling the ASR model directly in 30s chunks.

    Bypasses the legacy streaming pipeline (which silently discards audio
    during buffer resets at online_asr.py:244-252) to achieve full audio coverage.
    """
    all_tokens = []
    for new_tokens in _iter_batch_transcribe_chunks(pcm_bytes, language):
        all_tokens.extend(new_tokens)

    # Sort by start time (chunk boundaries can produce minor ordering jitter)
    all_tokens.sort(key=lambda t: t.start)
    logger.info(
        "Batch transcription complete: %d tokens across %.1fs",
        len(all_tokens),
        len(pcm_bytes) / (16000 * 2),
    )
    return all_tokens


def _speaker_to_one_indexed(speaker) -> int:
    if isinstance(speaker, (int, np.integer)):
        return int(speaker) + 1
    match = re.search(r"\d+", str(speaker))
    if match:
        return int(match.group(0)) + 1
    return 1


class _StreamingDiarizationFeeder:
    def __init__(self):
        self._online_diarization = None
        self._seen = set()
        self._enabled = bool(getattr(getattr(transcription_engine, "config", None), "diarization", False))
        diarization_model = getattr(transcription_engine, "diarization_model", None)
        if not self._enabled or diarization_model is None:
            return

        from whisperlivekit.core import online_diarization_factory

        args = getattr(transcription_engine, "args", None) or getattr(transcription_engine, "config", None)
        self._online_diarization = online_diarization_factory(args, diarization_model)

    async def feed(self, pcm_chunk: bytes) -> list:
        if self._online_diarization is None or not pcm_chunk:
            return []
        audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._online_diarization.insert_audio_chunk(audio)
        return self._segments_to_timeline(await self._online_diarization.diarize())

    async def flush(self) -> list:
        if self._online_diarization is None:
            return []
        self._online_diarization.insert_audio_chunk(np.zeros(16000, dtype=np.float32))
        return self._segments_to_timeline(await self._online_diarization.diarize())

    def close(self) -> None:
        if self._online_diarization is None:
            return
        close = getattr(self._online_diarization, "close", None)
        if close:
            close()

    def _segments_to_timeline(self, segments) -> list:
        timeline = []
        for segment in segments or []:
            start = max(0.0, float(getattr(segment, "start", 0.0) or 0.0))
            end = max(start, float(getattr(segment, "end", 0.0) or 0.0))
            if end <= start:
                continue
            item = {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": _speaker_to_one_indexed(getattr(segment, "speaker", 0)),
            }
            key = (item["start"], item["end"], item["speaker"])
            if key in self._seen:
                continue
            self._seen.add(key)
            timeline.append(item)
        return timeline


async def _accept_asr_pcm(pcm_bytes: bytes, language, offset_seconds: float, init_prompt: str):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    asr = transcription_engine.asr
    fw_model = getattr(asr, "fw_encoder", None) or getattr(asr, "model", None)
    if fw_model is None or not hasattr(fw_model, "transcribe"):
        raise RuntimeError(
            f"Cannot find a faster-whisper WhisperModel on transcription_engine.asr ({type(asr).__name__})."
        )
    lan = language or getattr(asr, "lan", None) or getattr(asr, "original_language", None)

    def transcribe_blocking():
        with _asr_lock:
            raw_segs, _ = fw_model.transcribe(
                audio,
                language=lan if lan and lan != "auto" else None,
                initial_prompt=init_prompt,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True,
            )
            return list(raw_segs)

    raw_segs = await asyncio.to_thread(transcribe_blocking)

    tokens = []
    for seg in raw_segs:
        if getattr(seg, "no_speech_prob", 0.0) > 0.9:
            continue
        for word in getattr(seg, "words", []):
            tokens.append(
                ASRToken(
                    start=round(float(word.start) + offset_seconds, 3),
                    end=round(float(word.end) + offset_seconds, 3),
                    text=word.word,
                    probability=getattr(word, "probability", None),
                )
            )
    return tokens


async def _iter_disk_backed_transcription(path: str, language, initial_prompt: str | None = None):
    sample_rate = 16000
    bytes_per_second = sample_rate * 2
    window_bytes = 30 * bytes_per_second
    overlap_bytes = 2 * bytes_per_second
    window_seconds = window_bytes / bytes_per_second
    overlap_seconds = overlap_bytes / bytes_per_second
    asr_buffer = b""
    asr_offset = 0.0
    accepted_until = 0.0
    processed_bytes = 0
    all_tokens = []
    init_prompt = initial_prompt or ""
    pending_diarization = []
    diarization = _StreamingDiarizationFeeder()

    def is_overlap_duplicate(token) -> bool:
        text = (getattr(token, "text", "") or "").strip().lower()
        if not text:
            return False
        start = float(getattr(token, "start", 0.0) or 0.0)
        end = float(getattr(token, "end", 0.0) or 0.0)
        for accepted in all_tokens[-100:]:
            accepted_text = (getattr(accepted, "text", "") or "").strip().lower()
            if accepted_text != text:
                continue
            accepted_start = float(getattr(accepted, "start", 0.0) or 0.0)
            accepted_end = float(getattr(accepted, "end", 0.0) or 0.0)
            if abs(start - accepted_start) <= 0.25 and abs(end - accepted_end) <= 0.25:
                return True
        return False

    async def transcribe_window(window: bytes, offset: float, stage: str, new_diarization: list):
        nonlocal init_prompt, accepted_until
        chunk_tokens = await _accept_asr_pcm(window, language, offset, init_prompt)
        overlap_end = offset + overlap_seconds
        new_tokens = []
        for token in chunk_tokens:
            in_reprocessed_overlap = accepted_until != 0.0 and float(getattr(token, "start", 0.0) or 0.0) < overlap_end
            if in_reprocessed_overlap and is_overlap_duplicate(token):
                continue
            new_tokens.append(token)
        if new_tokens:
            all_tokens.extend(new_tokens)
            recent_text = "".join(getattr(token, "text", "") for token in all_tokens[-50:])
            init_prompt = recent_text[-200:]
        accepted_until = offset + max(0.0, (len(window) / bytes_per_second) - overlap_seconds)
        return _ChunkResult(
            new_tokens=new_tokens,
            new_diarization=new_diarization,
            progress=_ChunkProgress(processed_seconds=round(processed_bytes / bytes_per_second, 3), stage=stage),
        )

    try:
        async for pcm_chunk in _stream_pcm_chunks_from_file(path, chunk_seconds=1.0):
            processed_bytes += len(pcm_chunk)
            new_diarization = await diarization.feed(pcm_chunk)
            pending_diarization.extend(new_diarization)
            asr_buffer += pcm_chunk
            while len(asr_buffer) >= window_bytes:
                result = await transcribe_window(asr_buffer[:window_bytes], asr_offset, "transcribing", pending_diarization)
                yield result
                pending_diarization = []
                keep = asr_buffer[window_bytes - overlap_bytes:]
                asr_offset += window_seconds - overlap_seconds
                asr_buffer = keep

        flushed_diarization = await diarization.flush()
        pending_diarization.extend(flushed_diarization)
        if asr_buffer:
            yield await transcribe_window(asr_buffer, asr_offset, "transcribing", pending_diarization)
        elif pending_diarization:
            yield _ChunkResult(
                new_tokens=[],
                new_diarization=pending_diarization,
                progress=_ChunkProgress(processed_seconds=round(processed_bytes / bytes_per_second, 3), stage="diarizing"),
            )
    finally:
        diarization.close()


async def _collect_disk_backed_transcription(path: str, language, initial_prompt: str | None = None):
    all_tokens = []
    all_diarization = []
    duration = 0.0
    async for chunk in _iter_disk_backed_transcription(path, language, initial_prompt=initial_prompt):
        all_tokens.extend(chunk.new_tokens)
        all_diarization.extend(chunk.new_diarization)
        duration = max(duration, chunk.progress.processed_seconds)
    all_tokens.sort(key=lambda token: float(getattr(token, "start", 0.0) or 0.0))
    all_diarization.sort(key=lambda item: float(item.get("start", 0.0) or 0.0))
    return all_tokens, all_diarization, duration


# Legacy full-memory helper retained temporarily for comparison tests. Uploaded-file endpoints should use the disk-backed pipeline instead.
async def _batch_diarize(pcm_bytes: bytes) -> list:
    """Run the configured diarization backend over full batch PCM audio."""
    if not getattr(getattr(transcription_engine, "config", None), "diarization", False):
        return []
    diarization_model = getattr(transcription_engine, "diarization_model", None)
    if diarization_model is None:
        return []

    from whisperlivekit.core import online_diarization_factory

    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 1.0
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    online_diarization = online_diarization_factory(transcription_engine.args, diarization_model)
    raw_segments = []

    try:
        for start in range(0, len(audio), CHUNK_SAMPLES):
            online_diarization.insert_audio_chunk(audio[start:start + CHUNK_SAMPLES])
            new_segments = await online_diarization.diarize()
            if new_segments:
                raw_segments.extend(new_segments)

        # Flush streaming backends that only emit after a full internal chunk.
        online_diarization.insert_audio_chunk(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        new_segments = await online_diarization.diarize()
        if new_segments:
            raw_segments.extend(new_segments)
    finally:
        close = getattr(online_diarization, "close", None)
        if close:
            close()

    duration = len(audio) / SAMPLE_RATE
    timeline = []
    seen = set()
    for segment in raw_segments:
        start = max(0.0, float(getattr(segment, "start", 0.0) or 0.0))
        end = min(duration, float(getattr(segment, "end", 0.0) or 0.0))
        if end <= start:
            continue
        item = {
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": _speaker_to_one_indexed(getattr(segment, "speaker", 0)),
        }
        key = (item["start"], item["end"], item["speaker"])
        if key not in seen:
            timeline.append(item)
            seen.add(key)

    timeline.sort(key=lambda item: item["start"])
    logger.info("Batch diarization complete: %d speaker segments", len(timeline))
    return timeline


def _build_segments_from_tokens(all_tokens: list, all_diarization: list) -> list:
    """Build non-overlapping Segment objects from the full committed token stream.

    Args:
        all_tokens: processor.state.tokens — the complete, never-pruned list of
            ASRToken and Silence objects committed during the session.
        all_diarization: accumulated diarization records from the collect() loop,
            each a dict {"start": float, "end": float, "speaker": any}.
            This must be accumulated externally because all_diarization_segments
            inside TokensAlignment is pruned to a 300s window.

    Returns:
        List of Segment objects with non-overlapping time ranges, suitable for
        assigning to front_data.lines and passing to _to_whisperx().
    """
    # Step A — Group tokens into punctuation-boundary PuncSegments.
    # After creating each segment, clamp its start/end to the true min/max
    # timestamps across all tokens in the chunk.  PuncSegment.from_tokens()
    # uses tokens[0].start and tokens[-1].end, but adjacent-chunk interleaving
    # can make those boundary tokens non-chronological.
    def _make_punc_seg(chunk):
        ps = PuncSegment.from_tokens(tokens=chunk)
        if ps is None:
            return None
        times = [t for t in chunk
                 if not getattr(t, 'is_silence', lambda: False)()
                 and getattr(t, 'start', None) is not None
                 and getattr(t, 'end', None) is not None]
        if times:
            ps.start = min(t.start for t in times)
            ps.end   = max(t.end   for t in times)
        return ps

    punc_segments = []
    seg_start = 0
    for i, token in enumerate(all_tokens):
        is_sil = getattr(token, 'is_silence', lambda: False)()
        if is_sil:
            chunk = all_tokens[seg_start:i]
            if chunk:
                ps = _make_punc_seg(chunk)
                if ps:
                    punc_segments.append(ps)
            sil_ps = PuncSegment.from_tokens(tokens=[token], is_silence=True)
            if sil_ps is not None:
                punc_segments.append(sil_ps)
            seg_start = i + 1
        elif getattr(token, 'has_punctuation', lambda: False)():
            chunk = all_tokens[seg_start:i + 1]
            ps = _make_punc_seg(chunk)
            if ps:
                punc_segments.append(ps)
            seg_start = i + 1

    tail = all_tokens[seg_start:]
    if tail:
        ps = _make_punc_seg(tail)
        if ps:
            punc_segments.append(ps)

    # Step B — Build merged diarization timeline from all_diarization
    merged_diar = []
    for d in sorted(all_diarization, key=lambda x: x['start']):
        if merged_diar and d['speaker'] == merged_diar[-1]['speaker']:
            merged_diar[-1]['end'] = max(merged_diar[-1]['end'], d['end'])
        else:
            merged_diar.append(dict(d))

    # Step C — Assign speaker to each non-silence PuncSegment
    for ps in punc_segments:
        if ps.is_silence():
            continue
        if not merged_diar:
            # No diarization data at all: attribute to speaker 1 (matches get_lines_diarization default).
            ps.speaker = 1
            continue
        if ps.start >= merged_diar[-1]['end']:
            # Segment is beyond the last diarization entry: mark as unattributed.
            ps.speaker = -1
            continue
        max_overlap = 0.0
        best_speaker = 1
        for d in merged_diar:
            overlap = max(0.0, min(ps.end, d['end']) - max(ps.start, d['start']))
            if overlap > max_overlap:
                max_overlap = overlap
                # Speaker values in all_diarization are already 1-indexed (get_lines_diarization adds +1).
                best_speaker = d['speaker']
        ps.speaker = best_speaker

    # Step D — Merge adjacent same-speaker PuncSegments into final Segment objects
    segments = []
    for ps in punc_segments:
        if ps.is_silence():
            continue
        if not ps.text or not ps.text.strip():
            continue
        # Guard against inverted timestamps from the ASR model (known to occur
        # occasionally with faster-whisper word_timestamps=True).
        seg_start = float(ps.start or 0.0)
        seg_end = float(ps.end or 0.0)
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start
        if segments and segments[-1].speaker == ps.speaker:
            # PuncSegment.text already carries leading whitespace from the ASR tokenizer.
            segments[-1].text = (segments[-1].text or '') + (ps.text or '')
            segments[-1].end = max(segments[-1].end, seg_end)
        else:
            segments.append(Segment(
                start=seg_start,
                end=seg_end,
                text=ps.text,
                speaker=ps.speaker,
            ))

    # Sort by start time — inverted-timestamp swaps above can disturb order.
    segments.sort(key=lambda s: float(s.start or 0.0))
    return segments


def _clamp_segment_overlaps(segments: list) -> list:
    """Clamp adjacent batch segments so output ranges stay non-overlapping."""
    ordered = sorted(segments, key=lambda s: float(getattr(s, "start", 0.0) or 0.0))
    for i in range(len(ordered) - 1):
        current = ordered[i]
        next_segment = ordered[i + 1]
        current_end = float(getattr(current, "end", 0.0) or 0.0)
        next_start = float(getattr(next_segment, "start", 0.0) or 0.0)
        if current_end > next_start:
            current.end = max(float(getattr(current, "start", 0.0) or 0.0), next_start)
    return ordered


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

    # Derived convenience fields built from the final segments list.
    result["transcript"] = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result["segments"]
    ]
    result["diarization"] = [
        {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]}
        for seg in result["segments"]
    ]

    return result


def _build_whisperx_response(all_tokens, all_diarization, duration: float) -> dict:
    diarization = sorted((dict(item) for item in all_diarization), key=lambda item: float(item.get("start", 0.0) or 0.0))
    if not all_tokens:
        return {"segments": [], "word_segments": [], "transcript": [], "diarization": diarization, "raw_words": []}
    token_segments = _build_segments_from_tokens(all_tokens, all_diarization)
    token_segments = _clamp_segment_overlaps(token_segments)

    class _FrontData:
        lines = token_segments
        buffer_transcription = ""
        buffer_diarization = ""

    result = _to_whisperx(_FrontData(), duration=duration)
    if diarization:
        result["diarization"] = diarization
    result["raw_words"] = [
        {
            "word": getattr(token, "text", ""),
            "start": round(float(getattr(token, "start", 0.0)), 3),
            "end": round(float(getattr(token, "end", 0.0)), 3),
            "probability": getattr(token, "probability", None),
        }
        for token in all_tokens
        if getattr(token, "text", "").strip()
    ]
    return result


def _speakers_compatible(s1, s2) -> bool:
    """Return True if two speaker labels can belong to the same segment.

    Speaker -1 means "unknown / not yet diarized" and is treated as
    compatible with any speaker so that refinements across snapshots
    (where diarization may flip from -1 to a real label) still merge.
    """
    if s1 == s2:
        return True
    # -1 (int) and "-1" (str) are both used as the undifferentiated-speaker sentinel
    unknown = {-1, "-1"}
    return s1 in unknown or s2 in unknown


def _deduplicate_segments(segments: list) -> list:
    """Remove near-duplicate segments that share text and overlap in time.

    The pipeline refines timestamps across snapshots, so the same logical
    segment can appear with slightly different start/end values *and* with
    different speaker labels (e.g. -1 → 1 as diarization resolves).
    This function merges such duplicates, keeping the version whose text
    is a superset of the other's (i.e. the more complete snapshot) and
    preferring the longer duration when both texts are equal.
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
        for i, existing in enumerate(deduped):
            e_text = getattr(existing, "text", "").strip()
            e_start = float(getattr(existing, "start", 0.0))
            e_end = float(getattr(existing, "end", 0.0))
            e_speaker = getattr(existing, "speaker", None)

            # Speaker must be compatible (same label or one of them is -1)
            if not _speakers_compatible(speaker, e_speaker):
                continue

            overlap = min(end, e_end) - max(start, e_start)
            shorter = min(end - start, e_end - e_start)
            if shorter <= 0 or overlap <= 0:
                continue

            if overlap / shorter > 0.5 and (text == e_text or text in e_text or e_text in text):
                # Keep the segment whose text is the superset (most complete).
                # If both texts are equal, keep the longer duration (more refined).
                text_is_superset = e_text in text and text != e_text
                duration_is_longer = (end - start) > (e_end - e_start)
                if text_is_superset or (text == e_text and duration_is_longer):
                    deduped[i] = seg
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


# Legacy full-memory helper retained temporarily for comparison tests. Uploaded-file endpoints should use the disk-backed pipeline instead.
async def _stream_batch_transcription(pcm_data: bytes, language, duration: float):
    import json

    all_tokens = []
    sent_text = ""

    try:
        for new_tokens in _iter_batch_transcribe_chunks(pcm_data, language):
            all_tokens.extend(new_tokens)
            current_text = "".join(getattr(token, "text", "") for token in all_tokens)
            if len(current_text) > len(sent_text):
                delta = current_text[len(sent_text):]
                sent_text = current_text
                event_data = json.dumps({"type": "transcript.text.delta", "delta": delta})
                yield f"data: {event_data}\n\n"
            await asyncio.sleep(0)

        if not all_tokens:
            result = {
                "segments": [],
                "word_segments": [],
                "transcript": [],
                "diarization": [],
                "raw_words": [],
            }
        else:
            all_tokens.sort(key=lambda t: t.start)
            all_diarization = await _batch_diarize(pcm_data)
            token_segments = _build_segments_from_tokens(all_tokens, all_diarization)
            token_segments = _clamp_segment_overlaps(token_segments)

            # Duck-type a minimal object for _to_whisperx — it reads attributes via getattr.
            class _FrontData:
                lines = token_segments
                buffer_transcription = ""
                buffer_diarization = ""

            result = _to_whisperx(_FrontData(), duration=duration)
            result["raw_words"] = [
                {
                    "word": getattr(t, "text", ""),
                    "start": round(float(getattr(t, "start", 0.0)), 3),
                    "end": round(float(getattr(t, "end", 0.0)), 3),
                    "score": float(getattr(t, "probability", 1.0) or 1.0),
                }
                for t in all_tokens
            ]

        done_data = json.dumps({"type": "transcript.text.done", "text": json.dumps(result)})
        yield f"data: {done_data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_data = json.dumps({"type": "error", "error": str(e)})
        yield f"data: {error_data}\n\n"


async def _stream_disk_backed_transcription(path: str, language, initial_prompt: str | None = None):
    import json

    all_tokens = []
    all_diarization = []
    sent_text = ""
    duration = 0.0
    try:
        async for chunk in _iter_disk_backed_transcription(path, language, initial_prompt=initial_prompt):
            all_tokens.extend(chunk.new_tokens)
            all_diarization.extend(chunk.new_diarization)
            duration = max(duration, chunk.progress.processed_seconds)
            progress_data = json.dumps(
                {
                    "type": "transcript.progress",
                    "processed_seconds": chunk.progress.processed_seconds,
                    "stage": chunk.progress.stage,
                }
            )
            yield f"data: {progress_data}\n\n"

            current_text = "".join(getattr(token, "text", "") for token in all_tokens)
            if len(current_text) > len(sent_text):
                delta = current_text[len(sent_text):]
                sent_text = current_text
                delta_data = json.dumps({"type": "transcript.text.delta", "delta": delta})
                yield f"data: {delta_data}\n\n"
            await asyncio.sleep(0)

        response = _build_whisperx_response(all_tokens, all_diarization, duration)
        done_data = json.dumps({"type": "transcript.text.done", "text": json.dumps(response)})
        yield f"data: {done_data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_data = json.dumps({"type": "error", "error": str(e)})
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


async def _collect_sse_events_for_test(path: str, language, initial_prompt: str | None = None):
    return [event async for event in _stream_disk_backed_transcription(path, language, initial_prompt=initial_prompt)]


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
        description="Optional prompt text to guide transcription.",
    ),
    stream: bool = Form(default=False, description="If true, return SSE streaming results."),
):
    """WhisperX-format audio transcription endpoint.

    Always returns WhisperX JSON schema:
      {"segments": [...], "word_segments": [...]}

    When stream=True, returns SSE events with partial transcription deltas.

    The `model` parameter is accepted but ignored.
    """
    global transcription_engine

    source_path = await _spool_upload_to_tempfile(file)

    if stream:
        return StreamingResponse(
            _stream_disk_backed_transcription(source_path, language, initial_prompt=prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        all_tokens, all_diarization, duration = await _collect_disk_backed_transcription(source_path, language, initial_prompt=prompt)
        return JSONResponse(_build_whisperx_response(all_tokens, all_diarization, duration))
    finally:
        try:
            os.unlink(source_path)
        except FileNotFoundError:
            pass


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
