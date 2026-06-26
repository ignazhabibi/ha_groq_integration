"""Tests for the Groq Cloud AI task entity."""

from types import MappingProxyType
from typing import cast
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.ai_task import GroqCloudTaskEntity
from custom_components.groq_cloud_conversation.const import (
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
)


def _make_entity() -> GroqCloudTaskEntity:
    """Create a Groq AI task entity for tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    )
    entry.runtime_data = MagicMock()
    subentry = ConfigSubentry(
        data=MappingProxyType(RECOMMENDED_AI_TASK_OPTIONS.copy()),
        subentry_id="ai-task-subentry",
        subentry_type="ai_task_data",
        title="Groq AI Task",
        unique_id=None,
    )
    entity = GroqCloudTaskEntity(cast("GroqCloudConfigEntry", entry), subentry)
    entity.entity_id = "ai_task.groq_cloud"
    return entity


def test_ai_task_entity_supports_attachments() -> None:
    """Test the AI task entity advertises image attachment support."""
    entity = _make_entity()

    assert entity.supported_features & ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS


async def _generate_with_response(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    task: ai_task.GenDataTask,
    response: str,
) -> ai_task.GenDataTaskResult:
    """Run an AI task while faking the Groq assistant response."""
    entity = _make_entity()
    chat_log = conversation.ChatLog(hass, "conversation-id")
    chat_log.async_add_user_content(conversation.UserContent(task.instructions))

    async def fake_handle_chat_log(
        chat_log: conversation.ChatLog,
        **_: object,
    ) -> None:
        """Append a fake assistant response."""
        chat_log.content.append(
            conversation.AssistantContent(
                agent_id=entity.entity_id,
                content=response,
            )
        )

    monkeypatch.setattr(
        entity,
        "_async_handle_chat_log",
        fake_handle_chat_log,
    )
    return await entity._async_generate_data(task, chat_log)


async def test_generate_data_returns_plain_text(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test AI task plain text generation returns text as-is."""
    task = ai_task.GenDataTask(
        name="Summarize",
        instructions="Summarize this text.",
    )

    result = await _generate_with_response(hass, monkeypatch, task, "A short summary.")

    assert result.conversation_id == "conversation-id"
    assert result.data == "A short summary."


async def test_generate_data_returns_structured_json(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test AI task structured generation parses JSON responses."""
    task = ai_task.GenDataTask(
        name="Extract",
        instructions="Extract data.",
        structure=vol.Schema({vol.Required("value"): str}),
    )

    result = await _generate_with_response(hass, monkeypatch, task, '{"value": "ok"}')

    assert result.data == {"value": "ok"}


async def test_generate_data_rejects_malformed_json(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test malformed structured responses fail clearly."""
    task = ai_task.GenDataTask(
        name="Extract",
        instructions="Extract data.",
        structure=vol.Schema({vol.Required("value"): str}),
    )

    with pytest.raises(HomeAssistantError, match="structured response"):
        await _generate_with_response(hass, monkeypatch, task, "{not-json")
