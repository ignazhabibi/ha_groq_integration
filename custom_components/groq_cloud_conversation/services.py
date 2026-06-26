"""Home Assistant actions for the Groq Cloud Conversation integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import ATTR_CONFIG_ENTRY_ID, ATTR_MODEL, CONF_PROMPT
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .api import GroqApiError, GroqAuthenticationError, GroqRateLimitError
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from . import GroqCloudConfigEntry

SERVICE_GENERATE_TEXT = "generate_text"

ATTR_FINISH_REASON = "finish_reason"
ATTR_SYSTEM_PROMPT = "system_prompt"
ATTR_TEXT = "text"
ATTR_USAGE = "usage"

SERVICE_GENERATE_TEXT_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Required(CONF_PROMPT): cv.string,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Optional(ATTR_MODEL): cv.string,
        vol.Optional(CONF_MAX_TOKENS): vol.All(vol.Coerce(int), vol.Range(min=1)),
        vol.Optional(CONF_TEMPERATURE): vol.All(
            vol.Coerce(float),
            vol.Range(min=0, max=2),
        ),
        vol.Optional(CONF_TOP_P): vol.All(
            vol.Coerce(float),
            vol.Range(min=0, max=1),
        ),
    }
)


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register Groq Cloud actions."""
    if hass.services.has_service(DOMAIN, SERVICE_GENERATE_TEXT):
        return

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_TEXT,
        async_handle_generate_text,
        schema=SERVICE_GENERATE_TEXT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


async def async_handle_generate_text(call: ServiceCall) -> ServiceResponse:
    """Generate text through Groq Chat Completions."""
    entry = _get_service_entry(call)
    options = _default_chat_options(entry)

    messages: list[dict[str, str]] = []
    if system_prompt := call.data.get(ATTR_SYSTEM_PROMPT):
        messages.append({"content": str(system_prompt), "role": "system"})
    messages.append({"content": str(call.data[CONF_PROMPT]), "role": "user"})

    payload: dict[str, Any] = {
        "max_completion_tokens": call.data.get(
            CONF_MAX_TOKENS,
            options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        ),
        "messages": messages,
        "model": call.data.get(
            ATTR_MODEL,
            options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
        ),
        "stream": False,
        "temperature": call.data.get(
            CONF_TEMPERATURE,
            options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
        ),
        "top_p": call.data.get(
            CONF_TOP_P,
            options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
        ),
        "user": call.context.user_id or call.context.id,
    }

    try:
        completion = await entry.runtime_data.client.async_chat_completion(payload)
    except GroqAuthenticationError as err:
        entry.async_start_reauth(call.hass)
        raise HomeAssistantError("Authentication error with Groq") from err
    except GroqRateLimitError as err:
        LOGGER.error("Rate limited by Groq: %s", err)
        raise HomeAssistantError("Rate limited or insufficient funds") from err
    except GroqApiError as err:
        LOGGER.error("Error talking to Groq: %s", err)
        raise HomeAssistantError("Error talking to Groq") from err

    return _format_generate_text_response(completion)


def _get_service_entry(call: ServiceCall) -> GroqCloudConfigEntry:
    """Return the loaded Groq config entry for a service call."""
    entry_id = call.data.get(ATTR_CONFIG_ENTRY_ID)
    if entry_id is not None:
        entry = call.hass.config_entries.async_get_entry(str(entry_id))
        if entry is None or entry.domain != DOMAIN:
            raise HomeAssistantError("Groq Cloud config entry not found")
        if entry.state is not ConfigEntryState.LOADED:
            raise HomeAssistantError("Groq Cloud config entry is not loaded")
        return cast("GroqCloudConfigEntry", entry)

    loaded_entries = [
        entry
        for entry in call.hass.config_entries.async_entries(DOMAIN)
        if entry.state is ConfigEntryState.LOADED
    ]
    if not loaded_entries:
        raise HomeAssistantError("No loaded Groq Cloud config entry found")
    if len(loaded_entries) > 1:
        raise HomeAssistantError(
            "config_entry_id is required for multiple Groq entries"
        )
    return cast("GroqCloudConfigEntry", loaded_entries[0])


def _default_chat_options(entry: GroqCloudConfigEntry) -> dict[str, Any]:
    """Return default text-generation options for the service entry."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "conversation":
            return dict(subentry.data)
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "ai_task_data":
            return dict(subentry.data)
    return {}


def _format_generate_text_response(completion: dict[str, Any]) -> dict[str, Any]:
    """Return Home Assistant service response data for a Chat Completion."""
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HomeAssistantError("Groq returned no choices")

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise HomeAssistantError("Groq returned no text")

    response: dict[str, Any] = {ATTR_TEXT: text}
    if isinstance(completion.get("model"), str):
        response[ATTR_MODEL] = completion["model"]
    if isinstance(first_choice, dict) and isinstance(
        first_choice.get("finish_reason"),
        str,
    ):
        response[ATTR_FINISH_REASON] = first_choice["finish_reason"]
    if isinstance(completion.get("usage"), dict):
        response[ATTR_USAGE] = completion["usage"]
    return response
