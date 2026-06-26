"""Text-to-speech platform for Groq Cloud Conversation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from homeassistant.components import tts as ha_tts
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import GroqApiError, GroqAuthenticationError, GroqSpeechRequest
from .const import (
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DOMAIN,
    GROQ_TTS_VOICES,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
)

if TYPE_CHECKING:
    from . import GroqCloudConfigEntry

_LOGGER = logging.getLogger(__name__)

GROQ_TTS_MAX_INPUT_LENGTH: Final = 200
TTS_MODEL_LANGUAGES: Final[dict[str, str]] = {
    "canopylabs/orpheus-v1-english": "en-US",
    "canopylabs/orpheus-arabic-saudi": "ar-SA",
}


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: GroqCloudConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud text-to-speech entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        async_add_entities(
            [GroqCloudTTSEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GroqCloudTTSEntity(ha_tts.TextToSpeechEntity):
    """Groq Cloud text-to-speech entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self,
        entry: GroqCloudConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the Groq Cloud text-to-speech entity."""
        self.entry = entry
        self.subentry = subentry
        model = self._model
        voice = self._voice
        self._attr_unique_id = subentry.subentry_id
        self._attr_default_language = _language_for_model(model)
        self._attr_supported_languages = [self._attr_default_language]
        self._attr_supported_options = [ha_tts.ATTR_VOICE]
        self._attr_default_options = {ha_tts.ATTR_VOICE: voice}
        self._attr_device_info = dr.DeviceInfo(
            entry_type=dr.DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, subentry.subentry_id)},
            manufacturer="Groq",
            model=model,
            name=subentry.title,
        )

    @property
    def _model(self) -> str:
        """Return the configured Groq TTS model."""
        model = self.subentry.data.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
        return str(model)

    @property
    def _voice(self) -> str:
        """Return the configured Groq TTS voice."""
        voice = self.subentry.data.get(CONF_TTS_VOICE, _default_voice(self._model))
        return str(voice)

    @callback
    def async_get_supported_voices(self, language: str) -> list[ha_tts.Voice] | None:
        """Return supported Groq voices for a Home Assistant language."""
        if language not in self.supported_languages:
            return None

        return [
            ha_tts.Voice(voice_id=voice_id, name=voice_name)
            for voice_id, voice_name in GROQ_TTS_VOICES.get(self._model, {}).items()
        ]

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> ha_tts.TtsAudioType:
        """Generate speech audio through Groq Cloud text-to-speech."""
        if language not in self.supported_languages:
            raise HomeAssistantError(f"Unsupported Groq TTS language: {language}")
        if len(message) > GROQ_TTS_MAX_INPUT_LENGTH:
            raise HomeAssistantError("Groq TTS input must be 200 characters or fewer")

        voice = options.get(ha_tts.ATTR_VOICE, self._voice)
        model_voices = GROQ_TTS_VOICES.get(self._model, {})
        if voice not in model_voices:
            raise HomeAssistantError(f"Unsupported Groq TTS voice: {voice}")

        try:
            audio = await self.entry.runtime_data.client.async_generate_speech(
                GroqSpeechRequest(
                    model=self._model,
                    text=message,
                    voice=str(voice),
                )
            )
        except GroqAuthenticationError as err:
            self.entry.async_start_reauth(self.hass)
            _LOGGER.exception("Authentication error during Groq TTS")
            raise HomeAssistantError("Authentication error with Groq") from err
        except GroqApiError as err:
            _LOGGER.exception("Error during Groq TTS")
            raise HomeAssistantError("Error during Groq TTS") from err

        if not audio:
            raise HomeAssistantError("Groq returned no TTS audio")
        return ("wav", audio)


def _language_for_model(model_id: str) -> str:
    """Return the Home Assistant language tag for a Groq TTS model."""
    return TTS_MODEL_LANGUAGES.get(model_id, "en-US")


def _default_voice(model_id: str) -> str:
    """Return the default Groq voice for a TTS model."""
    if model_id == RECOMMENDED_TTS_MODEL:
        return RECOMMENDED_TTS_VOICE
    voices = GROQ_TTS_VOICES.get(model_id)
    if voices:
        return next(iter(voices))
    return RECOMMENDED_TTS_VOICE
