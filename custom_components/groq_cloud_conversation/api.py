"""Groq OpenAI-compatible API client."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx

from .const import GROQ_BASE_URL
from .model_registry import GroqModelInfo

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
AUDIO_TRANSCRIPTIONS_PATH = "/audio/transcriptions"
AUDIO_SPEECH_PATH = "/audio/speech"
DEFAULT_TIMEOUT = 60.0
HTTP_ERROR_STATUS = 400


@dataclass(frozen=True, slots=True)
class GroqTranscriptionRequest:
    """Audio transcription request data."""

    audio: bytes
    filename: str
    language: str
    model: str
    prompt: str


@dataclass(frozen=True, slots=True)
class GroqSpeechRequest:
    """Text-to-speech request data."""

    model: str
    text: str
    voice: str
    response_format: str = "wav"


class GroqApiError(Exception):
    """Base error for Groq API failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize the Groq API error."""
        super().__init__(message)
        self.status_code = status_code


class GroqAuthenticationError(GroqApiError):
    """Authentication failed against Groq."""


class GroqConnectionError(GroqApiError):
    """Groq could not be reached."""


class GroqRateLimitError(GroqApiError):
    """Groq rejected a request because of rate limiting or quota."""


class GroqApiClient:
    """Small async client for Groq's OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = GROQ_BASE_URL,
    ) -> None:
        """Initialize the Groq API client."""
        self._api_key = api_key
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    async def async_list_models(
        self,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> list[GroqModelInfo]:
        """Return models visible to the configured Groq API key."""
        payload = await self._async_request_json(
            "GET",
            MODELS_PATH,
            request_timeout=request_timeout,
        )
        models = payload.get("data")
        if not isinstance(models, list):
            raise GroqApiError("Groq models response did not include a data list")

        return [
            GroqModelInfo.from_api(model)
            for model in models
            if isinstance(model, dict) and model.get("id")
        ]

    async def async_chat_completion(
        self,
        payload: Mapping[str, Any],
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Create a non-streaming Chat Completions response."""
        return await self._async_request_json(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json_payload=dict(payload),
            request_timeout=request_timeout,
        )

    async def async_stream_chat_completion(
        self,
        payload: Mapping[str, Any],
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Create a streaming Chat Completions response."""
        try:
            async with self._http_client.stream(
                "POST",
                self._url(CHAT_COMPLETIONS_PATH),
                headers=self._headers(),
                json=dict(payload),
                timeout=request_timeout,
            ) as response:
                await self._async_raise_for_response(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as err:
                        raise GroqApiError(
                            "Groq returned malformed streaming data"
                        ) from err
                    if isinstance(chunk, dict):
                        yield chunk
        except (httpx.TimeoutException, httpx.NetworkError) as err:
            raise GroqConnectionError("Cannot connect to Groq") from err

    async def async_transcribe_audio(
        self,
        request: GroqTranscriptionRequest,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> str | None:
        """Transcribe audio through Groq's OpenAI-compatible endpoint."""
        try:
            response = await self._http_client.post(
                self._url(AUDIO_TRANSCRIPTIONS_PATH),
                headers=self._headers(),
                data={
                    "language": request.language,
                    "model": request.model,
                    "prompt": request.prompt,
                    "response_format": "json",
                },
                files={
                    "file": (
                        request.filename,
                        request.audio,
                        "application/octet-stream",
                    )
                },
                timeout=request_timeout,
            )
            await self._async_raise_for_response(response)
        except (httpx.TimeoutException, httpx.NetworkError) as err:
            raise GroqConnectionError("Cannot connect to Groq") from err

        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) else None

    async def async_generate_speech(
        self,
        request: GroqSpeechRequest,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> bytes:
        """Generate speech audio through Groq's OpenAI-compatible endpoint."""
        try:
            response = await self._http_client.post(
                self._url(AUDIO_SPEECH_PATH),
                headers={
                    **self._headers(),
                    "Content-Type": "application/json",
                },
                json={
                    "input": request.text,
                    "model": request.model,
                    "response_format": request.response_format,
                    "voice": request.voice,
                },
                timeout=request_timeout,
            )
            await self._async_raise_for_response(response)
        except (httpx.TimeoutException, httpx.NetworkError) as err:
            raise GroqConnectionError("Cannot connect to Groq") from err

        return response.content

    async def _async_request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Send a JSON request and return a JSON object response."""
        try:
            response = await self._http_client.request(
                method,
                self._url(path),
                headers=self._headers(),
                json=dict(json_payload) if json_payload is not None else None,
                timeout=request_timeout,
            )
            await self._async_raise_for_response(response)
        except (httpx.TimeoutException, httpx.NetworkError) as err:
            raise GroqConnectionError("Cannot connect to Groq") from err

        try:
            payload = response.json()
        except json.JSONDecodeError as err:
            raise GroqApiError("Groq returned malformed JSON") from err
        if not isinstance(payload, dict):
            raise GroqApiError("Groq returned a non-object JSON response")
        return payload

    async def _async_raise_for_response(self, response: httpx.Response) -> None:
        """Raise a typed Groq error for unsuccessful HTTP responses."""
        if response.status_code < HTTP_ERROR_STATUS:
            return

        message = await _async_error_message(response)
        if response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            raise GroqAuthenticationError(message, response.status_code)
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            raise GroqRateLimitError(message, response.status_code)
        raise GroqApiError(message, response.status_code)

    def _headers(self) -> dict[str, str]:
        """Return standard Groq request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "home-assistant-groq-cloud-conversation",
        }

    def _url(self, path: str) -> str:
        """Return an absolute Groq API URL."""
        return f"{self._base_url}{path}"


async def _async_error_message(response: httpx.Response) -> str:
    """Return a safe message for a failed Groq response."""
    body = await response.aread()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        text = body.decode(errors="replace").strip()
        if text:
            return text
        return f"Groq returned HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error_message = error.get("message")
            if isinstance(error_message, str):
                return error_message
        message = payload.get("message")
        if isinstance(message, str):
            return message

    return f"Groq returned HTTP {response.status_code}"
