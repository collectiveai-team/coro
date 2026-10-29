"""A ``StreamingDiarizer`` over the in-memory Sortformer of ``support.fake_sortformer``.

No real NeMo model is loaded. The fakes raise ``ContractError`` when the
diarizer calls them in a way NeMo rejects. ``CHUNK_LEN`` and ``SUBSAMPLING_FACTOR`` match
the constants of the test modules that use these fixtures.
"""

from __future__ import annotations

import pytest
from support.fake_sortformer import (
    FakeMelPreprocessor,
    FakeStreamingSortformer,
    FixedPostProcessor,
)

from coro.backends.diarization.nemo.streaming import StreamingDiarizer

CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8


@pytest.fixture()
def sortformer():
    return FakeStreamingSortformer(subsampling_factor=SUBSAMPLING_FACTOR)


@pytest.fixture()
def mel():
    return FakeMelPreprocessor()


@pytest.fixture()
def post_processor():
    return FixedPostProcessor(segments=[(0.0, 1.0, 0), (1.0, 2.0, 1)])


@pytest.fixture()
def diarizer(sortformer, mel, post_processor):
    return StreamingDiarizer(
        sortformer,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mel,
        post_processor=post_processor,
    )
