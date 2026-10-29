"""StreamingDiarizer construction and chunked ingest, over an in-memory Sortformer.

The fakes come from ``support.fake_sortformer`` through this directory's
``conftest.py``; they raise ``ContractError`` on any call NeMo would reject.
Finalize and post-processing live in ``test_streaming_diarizer_finalize.py``.
The tests read the public ``processed_chunks`` that the Streaming Pipeline
reports, the result of ``finalize``, and the predictions the post-processor
receives.

With ``CHUNK_LEN`` 6 and one right-context frame, an ingested chunk is 0.56 s
of audio, of which 0.48 s is its own and 0.08 s is shared with the next
chunk; the model predicts 6 frames for it. The finalize flush predicts one
frame per 0.08 s left in the buffer.
"""

from __future__ import annotations

import struct

from support.fake_sortformer import FakeMelPreprocessor, FakeStreamingSortformer, FixedPostProcessor

from coro.backends.diarization.nemo.streaming import StreamingDiarizer

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
CHUNK_RIGHT_CONTEXT = 1
CHUNK_AUDIO_SECONDS = CHUNK_LEN * SUBSAMPLING_FACTOR * 0.01
CHUNK_AUDIO_BYTES = int(CHUNK_AUDIO_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
RIGHT_CONTEXT_SECONDS = CHUNK_RIGHT_CONTEXT * SUBSAMPLING_FACTOR * 0.01
RIGHT_CONTEXT_BYTES = int(RIGHT_CONTEXT_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
MIN_CHUNK_BYTES = CHUNK_AUDIO_BYTES + RIGHT_CONTEXT_BYTES


def pcm_of_bytes(n_bytes: int, *, sample: int = 1000) -> bytes:
    n_samples = n_bytes // BYTES_PER_SAMPLE
    return struct.pack(f"<{n_samples}h", *([sample] * n_samples))


def make_diarizer(model, *, post_processor=None) -> StreamingDiarizer:
    return StreamingDiarizer(
        model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=FakeMelPreprocessor(),
        post_processor=post_processor or FixedPostProcessor(segments=[]),
    )


def predicted_frames(diarizer, post_processor) -> int:
    diarizer.finalize()
    (predictions,) = post_processor.received
    return predictions.shape[1]


def test_ingest_processes_one_chunk(diarizer):
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES))

    assert diarizer.processed_chunks == 1


def test_ingest_buffers_until_chunk_ready(diarizer):
    part_pcm = pcm_of_bytes(MIN_CHUNK_BYTES // 2)

    diarizer.ingest_pcm_chunk(part_pcm)
    assert diarizer.processed_chunks == 0

    diarizer.ingest_pcm_chunk(part_pcm)
    assert diarizer.processed_chunks == 1


def test_ingest_processes_multiple_chunks(diarizer):
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES * 2))

    assert diarizer.processed_chunks == 2


def test_the_right_context_of_a_chunk_is_kept_for_the_flush(diarizer, post_processor):
    """The 0.08 s overlap stays buffered after the chunk and the flush predicts it."""
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES))

    assert predicted_frames(diarizer, post_processor) == CHUNK_LEN + 1


def test_every_frame_is_predicted_once(diarizer, post_processor):
    """Two chunks and a 0.16 s flush give 6 + 6 + 2 frames.

    A chunk that passed its right context as plain audio (``right_offset`` 0)
    would predict 7 frames; a flush that claimed right context would drop
    one; history passed back as ``total_preds`` or stacked features would
    repeat frames.
    """
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES * 2))

    assert predicted_frames(diarizer, post_processor) == CHUNK_LEN * 2 + 2


def test_predictions_reach_post_processing_on_the_cpu(diarizer, post_processor):
    """Only the current prediction chunk lives on the model device."""
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES * 2))
    diarizer.finalize()

    assert post_processor.received[0].device.type == "cpu"


def test_full_scale_audio_streams_within_the_nemo_contract(diarizer):
    """The fakes check, on every call, what earlier warmup bugs broke: keyword
    arguments to the preprocessor, a float signal in [-1, 1] with its length on
    the same device, autograd off (eval mode alone keeps activation memory),
    time-first features, the state of the previous step, an empty
    ``total_preds``, and one synchronous streaming state on the model device.
    """
    diarizer.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES * 3, sample=-32768))
    diarizer.finalize()

    assert diarizer.processed_chunks == 4


def test_two_instances_independent():
    first = make_diarizer(FakeStreamingSortformer())
    second = make_diarizer(FakeStreamingSortformer())

    first.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES))
    second.ingest_pcm_chunk(pcm_of_bytes(MIN_CHUNK_BYTES // 2))

    assert (first.processed_chunks, second.processed_chunks) == (1, 0)


def test_right_context_overlap_is_counted_once_in_the_duration():
    """The overlap of each chunk is audio of the next one, not extra audio: a
    segment past the end is clipped to the duration of the input exactly."""
    diarizer = make_diarizer(
        FakeStreamingSortformer(), post_processor=FixedPostProcessor(segments=[(0.0, 100.0, 0)])
    )
    pcm = pcm_of_bytes(MIN_CHUNK_BYTES * 2)

    diarizer.ingest_pcm_chunk(pcm)
    (segment,) = diarizer.finalize()

    assert segment.end == round(len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE), 3)
