"""Tests for Groq Cloud diagnostics."""

from typing import Any, cast
from unittest.mock import MagicMock

from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import CONF_API_KEY, CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.const import (
    CONF_CHAT_MODEL,
    DOMAIN,
)
from custom_components.groq_cloud_conversation.diagnostics import (
    async_get_config_entry_diagnostics,
    async_get_device_diagnostics,
)
from custom_components.groq_cloud_conversation.model_registry import (
    GroqModelInfo,
    GroqModelRegistry,
)
from custom_components.groq_cloud_conversation.runtime import GroqCloudRuntimeData


def _entry() -> GroqCloudConfigEntry:
    """Create a Groq config entry with sensitive diagnostic values."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-secret"},
        subentries_data=[
            {
                "data": {
                    CONF_CHAT_MODEL: "llama-3.3-70b-versatile",
                    CONF_PROMPT: "private home instructions",
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
        client=MagicMock(),
        model_registry=GroqModelRegistry(
            [GroqModelInfo.from_api({"id": "custom/live-chat-model"})]
        ),
    )
    return cast("GroqCloudConfigEntry", entry)


async def test_config_entry_diagnostics_redacts_secrets_and_prompts(
    hass: HomeAssistant,
) -> None:
    """Test diagnostics redact API keys and configured prompts."""
    diagnostics = await async_get_config_entry_diagnostics(hass, _entry())

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED
    assert diagnostics["subentries"][0]["data"][CONF_PROMPT] == REDACTED
    assert diagnostics["subentries"][0]["data"][CONF_CHAT_MODEL] == (
        "llama-3.3-70b-versatile"
    )
    assert diagnostics["runtime"]["loaded"] is True
    assert diagnostics["runtime"]["model_count"] >= 1
    assert "custom/live-chat-model" in {
        model["id"] for model in diagnostics["runtime"]["models"]
    }

    diagnostic_text = str(diagnostics)
    assert "groq-secret" not in diagnostic_text
    assert "private home instructions" not in diagnostic_text


async def test_device_diagnostics_adds_matching_subentry(
    hass: HomeAssistant,
) -> None:
    """Test device diagnostics identify the subentry behind a service device."""
    entry = _entry()
    identifiers: set[tuple[str, str]] = {(DOMAIN, "conversation-subentry")}
    device = dr.DeviceEntry(
        identifiers=cast("Any", identifiers),
        name="Groq Conversation",
    )

    diagnostics = await async_get_device_diagnostics(hass, entry, device)

    assert diagnostics["device"]["name"] == "Groq Conversation"
    assert diagnostics["device_subentry"]["subentry_id"] == "conversation-subentry"
    assert diagnostics["device_subentry"]["data"][CONF_PROMPT] == REDACTED
