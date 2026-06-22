"""Speech-to-text platform for Groq Cloud Conversation."""

import io
import logging
import wave
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Final, cast

import openai
from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from openai.types.audio import Transcription

from .const import (
    CONF_STT_MODEL,
    DEFAULT_STT_PROMPT,
    DOMAIN,
    RECOMMENDED_STT_MODEL,
)

if TYPE_CHECKING:
    from . import GroqCloudConfigEntry

_LOGGER = logging.getLogger(__name__)

SUPPORTED_STT_LANGUAGES: Final[tuple[str, ...]] = (
    "af-ZA",
    "ar-SA",
    "hy-AM",
    "az-AZ",
    "be-BY",
    "bs-BA",
    "bg-BG",
    "ca-ES",
    "zh-CN",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-NL",
    "en-US",
    "et-EE",
    "fi-FI",
    "fr-FR",
    "gl-ES",
    "de-DE",
    "el-GR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "is-IS",
    "id-ID",
    "it-IT",
    "ja-JP",
    "kn-IN",
    "kk-KZ",
    "ko-KR",
    "lv-LV",
    "lt-LT",
    "mk-MK",
    "ms-MY",
    "mr-IN",
    "mi-NZ",
    "ne-NP",
    "no-NO",
    "fa-IR",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sr-RS",
    "sk-SK",
    "sl-SI",
    "es-ES",
    "sw-KE",
    "sv-SE",
    "fil-PH",
    "ta-IN",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "ur-PK",
    "vi-VN",
    "cy-GB",
)


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: GroqCloudConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud speech-to-text entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [GroqCloudSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GroqCloudSTTEntity(stt.SpeechToTextEntity):
    """Groq Cloud speech-to-text entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self,
        entry: GroqCloudConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the Groq Cloud speech-to-text entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            entry_type=dr.DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, subentry.subentry_id)},
            manufacturer="Groq",
            model=subentry.data.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
            name=subentry.title,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return list(SUPPORTED_STT_LANGUAGES)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV, stt.AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return [
            stt.AudioBitRates.BITRATE_8,
            stt.AudioBitRates.BITRATE_16,
            stt.AudioBitRates.BITRATE_24,
            stt.AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [
            stt.AudioSampleRates.SAMPLERATE_8000,
            stt.AudioSampleRates.SAMPLERATE_11000,
            stt.AudioSampleRates.SAMPLERATE_16000,
            stt.AudioSampleRates.SAMPLERATE_18900,
            stt.AudioSampleRates.SAMPLERATE_22000,
            stt.AudioSampleRates.SAMPLERATE_32000,
            stt.AudioSampleRates.SAMPLERATE_37800,
            stt.AudioSampleRates.SAMPLERATE_44100,
            stt.AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO, stt.AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
        self,
        metadata: stt.SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> stt.SpeechResult:
        """Process an audio stream through Groq Cloud speech-to-text."""
        audio_data = await _async_read_audio_stream(stream)
        if metadata.format == stt.AudioFormats.WAV:
            audio_data = _add_wav_header(metadata, audio_data)

        client = self.entry.runtime_data
        options = self.subentry.data

        try:
            response = cast(
                "Transcription",
                await client.audio.transcriptions.create(
                    file=(f"a.{metadata.format.value}", audio_data),
                    language=metadata.language.split("-")[0],
                    model=options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
                    prompt=options.get(CONF_PROMPT, DEFAULT_STT_PROMPT),
                    response_format="json",
                ),
            )
        except openai.AuthenticationError:
            self.entry.async_start_reauth(self.hass)
            _LOGGER.exception("Authentication error during Groq STT")
        except openai.OpenAIError:
            _LOGGER.exception("Error during Groq STT")
        else:
            if response.text:
                return stt.SpeechResult(
                    response.text,
                    stt.SpeechResultState.SUCCESS,
                )

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)


async def _async_read_audio_stream(stream: AsyncIterable[bytes]) -> bytes:
    """Read a Home Assistant speech stream into bytes for Groq upload."""
    audio_bytes = bytearray()
    async for chunk in stream:
        audio_bytes.extend(chunk)
    return bytes(audio_bytes)


def _add_wav_header(metadata: stt.SpeechMetadata, audio_data: bytes) -> bytes:
    """Add the WAV header expected by the OpenAI-compatible transcription API."""
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(metadata.channel.value)
        wav_file.setsampwidth(metadata.bit_rate.value // 8)
        wav_file.setframerate(metadata.sample_rate.value)
        wav_file.writeframes(audio_data)

    return wav_buffer.getvalue()
