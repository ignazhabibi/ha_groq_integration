"""Tests for Groq Cloud Conversation actions."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_MODEL,
    CONF_API_KEY,
    CONF_PROMPT,
)
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.service import async_get_all_descriptions
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import async_setup
from custom_components.groq_cloud_conversation.api import GroqAuthenticationError
from custom_components.groq_cloud_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
)
from custom_components.groq_cloud_conversation.model_registry import GroqModelRegistry
from custom_components.groq_cloud_conversation.runtime import GroqCloudRuntimeData
from custom_components.groq_cloud_conversation.services import (
    ATTR_FINISH_REASON,
    ATTR_SYSTEM_PROMPT,
    ATTR_TEXT,
    ATTR_USAGE,
    SERVICE_GENERATE_TEXT,
    async_handle_generate_text,
)


def _completion(text: str = "Done") -> dict[str, object]:
    """Return a fake Chat Completion response."""
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": text,
                    "role": "assistant",
                },
            }
        ],
        "model": "llama-3.3-70b-versatile",
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens": 5,
            "total_tokens": 7,
        },
    }


def _entry(client: MagicMock, entry_id: str = "groq-entry") -> MockConfigEntry:
    """Create a loaded Groq config entry for service tests."""
    entry = MockConfigEntry(
        data={CONF_API_KEY: "groq-key"},
        domain=DOMAIN,
        entry_id=entry_id,
        state=ConfigEntryState.LOADED,
        subentries_data=[
            {
                "data": {
                    CONF_CHAT_MODEL: "configured-model",
                    CONF_MAX_TOKENS: 321,
                    CONF_TEMPERATURE: 0.2,
                    CONF_TOP_P: 0.8,
                },
                "subentry_id": "conversation-subentry",
                "subentry_type": "conversation",
                "title": "Groq Conversation",
                "unique_id": None,
            }
        ],
        title="Groq Cloud",
    )
    entry.runtime_data = GroqCloudRuntimeData(
        client=client,
        model_registry=GroqModelRegistry(),
    )
    return entry


async def test_setup_registers_generate_text_action(hass: HomeAssistant) -> None:
    """Test setup registers the Groq generate text action."""
    assert await async_setup(hass, {})

    assert hass.services.has_service(DOMAIN, SERVICE_GENERATE_TEXT)
    assert (
        hass.services.supports_response(DOMAIN, SERVICE_GENERATE_TEXT)
        is SupportsResponse.ONLY
    )


async def test_generate_text_action_description_loads(hass: HomeAssistant) -> None:
    """Test Home Assistant can load the generate text action description."""
    assert await async_setup(hass, {})

    descriptions = await async_get_all_descriptions(hass)

    description = descriptions[DOMAIN][SERVICE_GENERATE_TEXT]
    assert description["fields"][CONF_PROMPT]["required"] is True
    assert description["fields"][ATTR_CONFIG_ENTRY_ID]["selector"] == {
        "config_entry": {"integration": DOMAIN}
    }
    assert description["response"] == {"optional": False}


async def test_generate_text_action_calls_chat_completions(
    hass: HomeAssistant,
) -> None:
    """Test the generate text action returns Chat Completions response data."""
    client = MagicMock()
    client.async_chat_completion = AsyncMock(return_value=_completion("Hello"))
    entry = _entry(client)
    entry.add_to_hass(hass)
    assert await async_setup(hass, {})

    response = await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_TEXT,
        {
            CONF_PROMPT: "Write a notification",
            ATTR_SYSTEM_PROMPT: "Be concise",
        },
        blocking=True,
        return_response=True,
    )

    assert response == {
        ATTR_FINISH_REASON: "stop",
        ATTR_MODEL: "llama-3.3-70b-versatile",
        ATTR_TEXT: "Hello",
        ATTR_USAGE: {
            "completion_tokens": 2,
            "prompt_tokens": 5,
            "total_tokens": 7,
        },
    }
    client.async_chat_completion.assert_awaited_once()
    payload = client.async_chat_completion.call_args.args[0]
    assert payload["model"] == "configured-model"
    assert payload["max_completion_tokens"] == 321
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["messages"] == [
        {"content": "Be concise", "role": "system"},
        {"content": "Write a notification", "role": "user"},
    ]
    assert payload["stream"] is False


async def test_generate_text_action_accepts_overrides(
    hass: HomeAssistant,
) -> None:
    """Test the generate text action accepts explicit model options."""
    client = MagicMock()
    client.async_chat_completion = AsyncMock(return_value=_completion())
    entry = _entry(client)
    entry.add_to_hass(hass)
    assert await async_setup(hass, {})

    await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_TEXT,
        {
            ATTR_CONFIG_ENTRY_ID: entry.entry_id,
            ATTR_MODEL: "override-model",
            CONF_MAX_TOKENS: 123,
            CONF_PROMPT: "Write a title",
            CONF_TEMPERATURE: 0.4,
            CONF_TOP_P: 0.9,
        },
        blocking=True,
        return_response=True,
    )

    payload = client.async_chat_completion.call_args.args[0]
    assert payload["model"] == "override-model"
    assert payload["max_completion_tokens"] == 123
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.9


async def test_generate_text_action_requires_config_entry_when_ambiguous(
    hass: HomeAssistant,
) -> None:
    """Test multiple loaded entries require an explicit config entry id."""
    _entry(MagicMock(), "first").add_to_hass(hass)
    _entry(MagicMock(), "second").add_to_hass(hass)
    assert await async_setup(hass, {})

    with pytest.raises(HomeAssistantError, match="config_entry_id is required"):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_GENERATE_TEXT,
            {CONF_PROMPT: "Write a notification"},
            blocking=True,
            return_response=True,
        )


async def test_generate_text_action_starts_reauth_on_authentication_error(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test authentication errors start reauthentication."""
    client = MagicMock()
    client.async_chat_completion = AsyncMock(
        side_effect=GroqAuthenticationError("invalid key")
    )
    entry = _entry(client)
    entry.add_to_hass(hass)
    start_reauth = MagicMock()
    monkeypatch.setattr(entry, "async_start_reauth", start_reauth)

    call = MagicMock()
    call.data = {CONF_PROMPT: "Write a notification"}
    call.hass = hass
    call.context.user_id = None
    call.context.id = "context-id"

    with pytest.raises(HomeAssistantError, match="Authentication error with Groq"):
        await async_handle_generate_text(call)

    start_reauth.assert_called_once_with(hass)
