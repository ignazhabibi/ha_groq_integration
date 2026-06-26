"""Tests for the Groq Cloud speech-to-text entity."""

import io
import wave
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_PROMPT
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.api import (
    GroqApiError,
    GroqAuthenticationError,
)
from custom_components.groq_cloud_conversation.const import (
    CONF_STT_MODEL,
    DOMAIN,
    RECOMMENDED_STT_MODEL,
)
from custom_components.groq_cloud_conversation.model_registry import GroqModelRegistry
from custom_components.groq_cloud_conversation.runtime import GroqCloudRuntimeData
from custom_components.groq_cloud_conversation.stt import (
    GroqCloudSTTEntity,
    async_setup_entry,
)


async def _audio_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    """Yield fake Home Assistant speech audio chunks."""
    for chunk in chunks:
        yield chunk


def _metadata(
    audio_format: stt.AudioFormats = stt.AudioFormats.WAV,
    codec: stt.AudioCodecs = stt.AudioCodecs.PCM,
) -> stt.SpeechMetadata:
    """Create speech metadata for STT tests."""
    return stt.SpeechMetadata(
        language="de-DE",
        format=audio_format,
        codec=codec,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


def _subentry(data: dict[str, Any] | None = None) -> ConfigSubentry:
    """Create a speech-to-text config subentry."""
    return ConfigSubentry(
        data=MappingProxyType(data or {}),
        subentry_id="stt-subentry",
        subentry_type="stt",
        title="Groq Cloud STT",
        unique_id=None,
    )


def _entity(
    client: MagicMock,
    data: dict[str, Any] | None = None,
) -> GroqCloudSTTEntity:
    """Create a speech-to-text entity with fake Groq runtime data."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    )
    entry.runtime_data = GroqCloudRuntimeData(
        client=client,
        model_registry=GroqModelRegistry(),
    )
    return GroqCloudSTTEntity(cast("GroqCloudConfigEntry", entry), _subentry(data))


async def test_setup_entry_adds_entities_for_stt_subentries(
    hass: HomeAssistant,
) -> None:
    """Test setup adds one STT entity for each STT subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        subentries_data=[
            {
                "data": {},
                "subentry_type": "conversation",
                "title": "Groq Conversation",
                "unique_id": None,
            },
            {
                "data": {},
                "subentry_id": "stt-subentry",
                "subentry_type": "stt",
                "title": "Groq STT",
                "unique_id": None,
            },
        ],
        title="Groq Cloud",
    )
    entry.runtime_data = MagicMock()
    add_entities = MagicMock()

    await async_setup_entry(hass, cast("GroqCloudConfigEntry", entry), add_entities)

    add_entities.assert_called_once()
    entities = add_entities.call_args.args[0]
    assert len(entities) == 1
    assert isinstance(entities[0], GroqCloudSTTEntity)
    assert add_entities.call_args.kwargs == {"config_subentry_id": "stt-subentry"}


def test_stt_entity_reports_supported_metadata() -> None:
    """Test the entity advertises supported Home Assistant STT metadata."""
    entity = _entity(MagicMock())

    assert "de-DE" in entity.supported_languages
    assert "en-US" in entity.supported_languages
    assert entity.supported_formats == [stt.AudioFormats.WAV, stt.AudioFormats.OGG]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]
    assert stt.AudioSampleRates.SAMPLERATE_16000 in entity.supported_sample_rates
    assert entity.supported_channels == [
        stt.AudioChannels.CHANNEL_MONO,
        stt.AudioChannels.CHANNEL_STEREO,
    ]


async def test_process_wav_stream_adds_header_and_transcribes() -> None:
    """Test WAV metadata produces a WAV upload for Groq transcription."""
    client = MagicMock()
    client.async_transcribe_audio = AsyncMock(return_value="Licht an")
    entity = _entity(
        client,
        {
            CONF_PROMPT: "Smart home context",
            CONF_STT_MODEL: "whisper-large-v3",
        },
    )

    result = await entity.async_process_audio_stream(
        _metadata(),
        _audio_stream(b"\x01\x00", b"\x02\x00"),
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Licht an"
    client.async_transcribe_audio.assert_awaited_once()
    request = client.async_transcribe_audio.call_args.args[0]
    assert request.model == "whisper-large-v3"
    assert request.language == "de"
    assert request.prompt == "Smart home context"
    assert request.filename == "a.wav"

    with wave.open(io.BytesIO(request.audio), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(2) == b"\x01\x00\x02\x00"


async def test_process_ogg_stream_sends_raw_audio() -> None:
    """Test OGG metadata sends the input bytes without WAV wrapping."""
    client = MagicMock()
    client.async_transcribe_audio = AsyncMock(return_value="Hallo")
    entity = _entity(client)

    result = await entity.async_process_audio_stream(
        _metadata(stt.AudioFormats.OGG, stt.AudioCodecs.OPUS),
        _audio_stream(b"ogg", b"data"),
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    request = client.async_transcribe_audio.call_args.args[0]
    assert request.filename == "a.ogg"
    assert request.audio == b"oggdata"
    assert request.model == RECOMMENDED_STT_MODEL


async def test_process_stream_returns_error_for_empty_transcription() -> None:
    """Test an empty Groq transcription is returned as an STT error."""
    client = MagicMock()
    client.async_transcribe_audio = AsyncMock(return_value="")
    entity = _entity(client)

    result = await entity.async_process_audio_stream(
        _metadata(),
        _audio_stream(b"\x00\x00"),
    )

    assert result == stt.SpeechResult(None, stt.SpeechResultState.ERROR)


async def test_process_stream_returns_error_for_groq_failure() -> None:
    """Test non-auth Groq transcription failures become STT errors."""
    client = MagicMock()
    client.async_transcribe_audio = AsyncMock(side_effect=GroqApiError("boom"))
    entity = _entity(client)

    result = await entity.async_process_audio_stream(
        _metadata(),
        _audio_stream(b"\x00\x00"),
    )

    assert result == stt.SpeechResult(None, stt.SpeechResultState.ERROR)


async def test_process_stream_starts_reauth_for_authentication_error(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """Test authentication failures start reauth and return an STT error."""
    client = MagicMock()
    client.async_transcribe_audio = AsyncMock(
        side_effect=GroqAuthenticationError("Invalid API key")
    )
    entity = _entity(client)
    entity.hass = hass
    start_reauth = MagicMock()
    monkeypatch.setattr(entity.entry, "async_start_reauth", start_reauth)

    result = await entity.async_process_audio_stream(
        _metadata(),
        _audio_stream(b"\x00\x00"),
    )

    assert result == stt.SpeechResult(None, stt.SpeechResultState.ERROR)
    start_reauth.assert_called_once_with(hass)
