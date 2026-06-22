"""Tests for the Groq Cloud conversation entity."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast
from unittest.mock import MagicMock

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.const import (
    DOMAIN,
    RECOMMENDED_CONVERSATION_OPTIONS,
)
from custom_components.groq_cloud_conversation.conversation import (
    GroqCloudConversationEntity,
)

type ConversationOptionValue = bool | list[str] | str


def _make_entity(
    data: Mapping[str, ConversationOptionValue] | None = None,
) -> GroqCloudConversationEntity:
    """Create a Groq conversation entity for tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    )
    entry.runtime_data = MagicMock()
    options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
    if data:
        options.update(data)
    subentry = ConfigSubentry(
        data=MappingProxyType(options),
        subentry_id="conversation-subentry",
        subentry_type="conversation",
        title="Groq Conversation",
        unique_id=None,
    )
    entity = GroqCloudConversationEntity(cast("GroqCloudConfigEntry", entry), subentry)
    entity.entity_id = "conversation.groq_cloud"
    return entity


def test_control_feature_enabled_when_llm_api_selected() -> None:
    """Test selecting an HA LLM API enables conversation control support."""
    entity = _make_entity({CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    assert entity.supported_features is not None
    assert entity.supported_features & conversation.ConversationEntityFeature.CONTROL


async def test_llm_provisioning_error_returns_conversation_result(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test LLM provisioning errors are returned as conversation results."""
    entity = _make_entity()
    user_input = conversation.ConversationInput(
        text="Turn on the lights",
        context=Context(),
        conversation_id="conversation-id",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.groq_cloud",
    )
    chat_log = conversation.ChatLog(hass, "conversation-id")
    response = intent.IntentResponse(language="en")
    response.async_set_speech("Unable to prepare tools.")
    converse_error = conversation.ConverseError(
        "Unable to prepare tools.",
        "conversation-id",
        response,
    )

    async def fake_provide_llm_data(*_: object) -> None:
        """Raise a fake LLM provisioning error."""
        raise converse_error

    monkeypatch.setattr(
        chat_log,
        "async_provide_llm_data",
        fake_provide_llm_data,
    )

    result = await entity._async_handle_message(user_input, chat_log)

    assert result.conversation_id == "conversation-id"
    assert result.response.speech["plain"]["speech"] == "Unable to prepare tools."
