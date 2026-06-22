"""The Groq Cloud Conversation integration."""

import openai
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, GROQ_BASE_URL

PLATFORMS: tuple[Platform, ...] = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.STT,
)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type GroqCloudConfigEntry = ConfigEntry[openai.AsyncClient]


async def async_setup(_hass: HomeAssistant, _config: ConfigType) -> bool:
    """Set up Groq Cloud Conversation."""
    return True


async def async_setup_entry(
    hass: HomeAssistant,
    entry: GroqCloudConfigEntry,
) -> bool:
    """Set up Groq Cloud Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=GROQ_BASE_URL,
        http_client=get_async_client(hass),
    )

    try:
        await client.models.list(timeout=10.0)
    except openai.AuthenticationError as err:
        raise ConfigEntryAuthFailed(err) from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: GroqCloudConfigEntry,
) -> bool:
    """Unload Groq Cloud Conversation."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(
    hass: HomeAssistant,
    entry: GroqCloudConfigEntry,
) -> None:
    """Reload the integration when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
