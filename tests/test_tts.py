"""Tests for the Groq Cloud text-to-speech entity."""

from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import tts as ha_tts
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.api import (
    GroqApiError,
    GroqAuthenticationError,
)
from custom_components.groq_cloud_conversation.const import (
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DOMAIN,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
)
from custom_components.groq_cloud_conversation.model_registry import GroqModelRegistry
from custom_components.groq_cloud_conversation.runtime import GroqCloudRuntimeData
from custom_components.groq_cloud_conversation.tts import (
    GroqCloudTTSEntity,
    async_setup_entry,
)


def _subentry(data: dict[str, Any] | None = None) -> ConfigSubentry:
    """Create a text-to-speech config subentry."""
    return ConfigSubentry(
        data=MappingProxyType(data or {}),
        subentry_id="tts-subentry",
        subentry_type="tts",
        title="Groq Cloud TTS",
        unique_id=None,
    )


def _entity(
    client: MagicMock,
    data: dict[str, Any] | None = None,
) -> GroqCloudTTSEntity:
    """Create a text-to-speech entity with fake Groq runtime data."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    )
    entry.runtime_data = GroqCloudRuntimeData(
        client=client,
        model_registry=GroqModelRegistry(),
    )
    return GroqCloudTTSEntity(cast("GroqCloudConfigEntry", entry), _subentry(data))


async def test_setup_entry_adds_entities_for_tts_subentries(
    hass: HomeAssistant,
) -> None:
    """Test setup adds one TTS entity for each TTS subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        subentries_data=[
            {
                "data": {},
                "subentry_type": "stt",
                "title": "Groq STT",
                "unique_id": None,
            },
            {
                "data": {},
                "subentry_id": "tts-subentry",
                "subentry_type": "tts",
                "title": "Groq TTS",
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
    assert isinstance(entities[0], GroqCloudTTSEntity)
    assert add_entities.call_args.kwargs == {"config_subentry_id": "tts-subentry"}


def test_tts_entity_reports_supported_english_voice_metadata() -> None:
    """Test the entity advertises English Groq TTS metadata."""
    entity = _entity(MagicMock())

    assert entity.default_language == "en-US"
    assert entity.supported_languages == ["en-US"]
    assert entity.supported_options == [ha_tts.ATTR_VOICE]
    assert entity.default_options == {ha_tts.ATTR_VOICE: RECOMMENDED_TTS_VOICE}
    assert entity.async_get_supported_voices("en-US") == [
        ha_tts.Voice("autumn", "Autumn"),
        ha_tts.Voice("diana", "Diana"),
        ha_tts.Voice("hannah", "Hannah"),
        ha_tts.Voice("austin", "Austin"),
        ha_tts.Voice("daniel", "Daniel"),
        ha_tts.Voice("troy", "Troy"),
    ]
    assert entity.async_get_supported_voices("ar-SA") is None


def test_tts_entity_reports_supported_arabic_voice_metadata() -> None:
    """Test the entity advertises Arabic Groq TTS metadata."""
    entity = _entity(
        MagicMock(),
        {
            CONF_TTS_MODEL: "canopylabs/orpheus-arabic-saudi",
            CONF_TTS_VOICE: "noura",
        },
    )

    assert entity.default_language == "ar-SA"
    assert entity.supported_languages == ["ar-SA"]
    assert entity.default_options == {ha_tts.ATTR_VOICE: "noura"}
    assert entity.async_get_supported_voices("ar-SA") == [
        ha_tts.Voice("abdullah", "Abdullah"),
        ha_tts.Voice("fahad", "Fahad"),
        ha_tts.Voice("sultan", "Sultan"),
        ha_tts.Voice("lulwa", "Lulwa"),
        ha_tts.Voice("noura", "Noura"),
        ha_tts.Voice("aisha", "Aisha"),
    ]


async def test_tts_entity_generates_wav_audio() -> None:
    """Test TTS generation calls Groq speech with configured options."""
    client = MagicMock()
    client.async_generate_speech = AsyncMock(return_value=b"RIFFwav")
    entity = _entity(client)

    extension, audio = await entity.async_get_tts_audio("Hallo", "en-US", {})

    assert extension == "wav"
    assert audio == b"RIFFwav"
    client.async_generate_speech.assert_awaited_once()
    request = client.async_generate_speech.call_args.args[0]
    assert request.model == RECOMMENDED_TTS_MODEL
    assert request.voice == RECOMMENDED_TTS_VOICE
    assert request.text == "Hallo"
    assert request.response_format == "wav"


async def test_tts_entity_accepts_voice_override() -> None:
    """Test TTS generation accepts a Home Assistant voice option."""
    client = MagicMock()
    client.async_generate_speech = AsyncMock(return_value=b"RIFFwav")
    entity = _entity(client)

    await entity.async_get_tts_audio("Hallo", "en-US", {ha_tts.ATTR_VOICE: "hannah"})

    request = client.async_generate_speech.call_args.args[0]
    assert request.voice == "hannah"


@pytest.mark.parametrize(
    ("message", "language", "options", "error"),
    [
        ("Hallo", "ar-SA", {}, "Unsupported Groq TTS language"),
        ("Hallo", "en-US", {ha_tts.ATTR_VOICE: "noura"}, "Unsupported Groq TTS voice"),
        ("x" * 201, "en-US", {}, "200 characters or fewer"),
    ],
)
async def test_tts_entity_rejects_invalid_requests(
    message: str,
    language: str,
    options: dict[str, Any],
    error: str,
) -> None:
    """Test invalid TTS requests fail before calling Groq."""
    client = MagicMock()
    client.async_generate_speech = AsyncMock()
    entity = _entity(client)

    with pytest.raises(HomeAssistantError, match=error):
        await entity.async_get_tts_audio(message, language, options)

    client.async_generate_speech.assert_not_awaited()


async def test_tts_entity_maps_groq_failure() -> None:
    """Test non-auth Groq TTS failures become Home Assistant errors."""
    client = MagicMock()
    client.async_generate_speech = AsyncMock(side_effect=GroqApiError("boom"))
    entity = _entity(client)

    with pytest.raises(HomeAssistantError, match="Error during Groq TTS"):
        await entity.async_get_tts_audio("Hallo", "en-US", {})


async def test_tts_entity_starts_reauth_for_authentication_error(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """Test authentication failures start reauth for TTS."""
    client = MagicMock()
    client.async_generate_speech = AsyncMock(
        side_effect=GroqAuthenticationError("Invalid API key")
    )
    entity = _entity(client)
    entity.hass = hass
    start_reauth = MagicMock()
    monkeypatch.setattr(entity.entry, "async_start_reauth", start_reauth)

    with pytest.raises(HomeAssistantError, match="Authentication error with Groq"):
        await entity.async_get_tts_audio("Hallo", "en-US", {})

    start_reauth.assert_called_once_with(hass)
