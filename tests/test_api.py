"""Tests for the Groq API client and model registry."""

import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from custom_components.groq_cloud_conversation.api import (
    GroqApiClient,
    GroqAuthenticationError,
    GroqConnectionError,
    GroqSpeechRequest,
    GroqTranscriptionRequest,
)
from custom_components.groq_cloud_conversation.model_registry import (
    GroqCapability,
    GroqModelInfo,
    GroqModelRegistry,
)

Handler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def _client(handler: Handler) -> tuple[GroqApiClient, httpx.AsyncClient]:
    """Create a Groq client backed by an HTTPX mock transport."""
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    )
    return (
        GroqApiClient(
            api_key="groq-key",
            base_url="https://api.groq.test/openai/v1",
            http_client=http_client,
        ),
        http_client,
    )


async def test_api_client_lists_models_and_sends_auth_header() -> None:
    """Test model discovery uses Groq's OpenAI-compatible models endpoint."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a fake Groq model list."""
        assert request.url.path == "/openai/v1/models"
        assert request.headers["authorization"] == "Bearer groq-key"
        return httpx.Response(
            200,
            json={"data": [{"id": "llama-3.3-70b-versatile"}]},
        )

    client, http_client = _client(handler)
    try:
        models = await client.async_list_models()
    finally:
        await http_client.aclose()

    assert models == [GroqModelInfo.from_api({"id": "llama-3.3-70b-versatile"})]


async def test_api_client_maps_authentication_errors() -> None:
    """Test Groq authentication failures get a typed error."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        """Return a fake authentication failure."""
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client, http_client = _client(handler)
    try:
        with pytest.raises(GroqAuthenticationError, match="bad key"):
            await client.async_list_models()
    finally:
        await http_client.aclose()


async def test_api_client_maps_transport_errors() -> None:
    """Test HTTP transport failures get a typed connection error."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        """Raise a fake protocol failure."""
        raise httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(GroqConnectionError, match="Cannot connect to Groq"):
            await client.async_list_models()
    finally:
        await http_client.aclose()


async def test_api_client_parses_chat_completion_stream() -> None:
    """Test streaming Chat Completions SSE chunks are parsed as dictionaries."""
    chunk = {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
                "index": 0,
            }
        ]
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a fake streaming response."""
        assert request.url.path == "/openai/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
        )

    client, http_client = _client(handler)
    try:
        chunks = [
            event
            async for event in client.async_stream_chat_completion(
                {"model": "llama-3.3-70b-versatile", "stream": True}
            )
        ]
    finally:
        await http_client.aclose()

    assert chunks == [chunk]


async def test_api_client_sends_audio_transcription_multipart() -> None:
    """Test audio transcription sends a multipart Groq request."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a fake transcription response."""
        assert request.url.path == "/openai/v1/audio/transcriptions"
        assert "multipart/form-data" in request.headers["content-type"]
        assert b'name="model"\r\n\r\nwhisper-large-v3' in request.content
        assert b'name="file"; filename="a.wav"' in request.content
        return httpx.Response(200, json={"text": "Licht an"})

    client, http_client = _client(handler)
    try:
        text = await client.async_transcribe_audio(
            GroqTranscriptionRequest(
                audio=b"wav",
                filename="a.wav",
                language="de",
                model="whisper-large-v3",
                prompt="Smart home",
            )
        )
    finally:
        await http_client.aclose()

    assert text == "Licht an"


async def test_api_client_sends_speech_json_and_returns_audio() -> None:
    """Test text-to-speech sends a JSON Groq speech request."""

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a fake speech response."""
        assert request.url.path == "/openai/v1/audio/speech"
        assert request.headers["content-type"] == "application/json"
        body = json.loads(request.content)
        assert body == {
            "input": "Hallo",
            "model": "canopylabs/orpheus-v1-english",
            "response_format": "wav",
            "voice": "troy",
        }
        return httpx.Response(200, content=b"RIFFwav")

    client, http_client = _client(handler)
    try:
        audio = await client.async_generate_speech(
            GroqSpeechRequest(
                model="canopylabs/orpheus-v1-english",
                text="Hallo",
                voice="troy",
            )
        )
    finally:
        await http_client.aclose()

    assert audio == b"RIFFwav"


def test_model_registry_infers_and_merges_capabilities() -> None:
    """Test live model metadata is merged with built-in capability data."""
    registry = GroqModelRegistry(
        [
            GroqModelInfo.from_api(
                {
                    "id": "custom/vision-model",
                    "capabilities": ["vision", "tool_calling"],
                }
            )
        ]
    )

    assert registry.supports("openai/gpt-oss-20b", GroqCapability.STRUCTURED_OUTPUTS)
    assert registry.supports("whisper-large-v3-turbo", GroqCapability.SPEECH_TO_TEXT)
    assert registry.supports(
        "canopylabs/orpheus-v1-english",
        GroqCapability.TEXT_TO_SPEECH,
    )
    assert registry.supports("custom/vision-model", GroqCapability.VISION)
    assert registry.supports("custom/vision-model", GroqCapability.TOOL_CALLING)
