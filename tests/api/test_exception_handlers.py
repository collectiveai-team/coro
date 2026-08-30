"""Transcription Exception Handler behavior."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api.exceptions import TranscriptionValidationError
from coro.app import create_app
from coro.settings import ServerSettings


@pytest.mark.asyncio
async def test_transcription_exception_handler_returns_openai_style_error():
    app = create_app(ServerSettings(_env_file=None))

    @app.get("/raises-transcription-validation")
    async def raises_transcription_validation():
        raise TranscriptionValidationError("bad file", param="file")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/raises-transcription-validation")

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "bad file",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }
    }
