"""Tests for the Groq Cloud conversation entity."""

from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import TypeVar, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.const import (
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
)
from custom_components.groq_cloud_conversation.conversation import (
    GroqCloudConversationEntity,
)

type ConversationOptionValue = bool | list[str] | str
StreamEventT = TypeVar("StreamEventT")


class FakeStream[StreamEventT]:
    """Async iterator for fake streaming events."""

    def __init__(self, events: list[StreamEventT]) -> None:
        """Initialize the fake stream."""
        self._events = iter(events)

    def __aiter__(self) -> AsyncIterator[StreamEventT]:
        """Return the stream iterator."""
        return self

    async def __anext__(self) -> StreamEventT:
        """Return the next fake stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


def _chat_message_chunks(text: str) -> list[ChatCompletionChunk]:
    """Return fake Chat Completions chunks for an assistant text response."""
    return [
        ChatCompletionChunk.model_construct(
            id="chatcmpl_1",
            choices=[
                Choice.model_construct(
                    delta=ChoiceDelta.model_construct(
                        content=text,
                        role="assistant",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=0,
            model=RECOMMENDED_CHAT_MODEL,
            object="chat.completion.chunk",
        )
    ]


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


async def test_handle_message_returns_successful_conversation_result(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a successful conversation run returns the streamed assistant speech."""
    entity = _make_entity()
    client = cast("MagicMock", entity.entry.runtime_data)
    client.chat.completions.create = AsyncMock(
        return_value=FakeStream(_chat_message_chunks("The lights are on."))
    )
    user_input = conversation.ConversationInput(
        text="Turn on the lights",
        context=Context(),
        conversation_id="conversation-id",
        device_id="voice-pe",
        satellite_id=None,
        language="en",
        agent_id="conversation.groq_cloud",
        extra_system_prompt="Keep it short.",
    )
    chat_log = conversation.ChatLog(hass, "conversation-id")
    chat_log.async_add_user_content(conversation.UserContent(user_input.text))
    provided_llm_data: list[tuple[llm.LLMContext, object, object, object]] = []

    async def fake_provide_llm_data(
        llm_context: llm.LLMContext,
        user_llm_hass_api: object = None,
        user_llm_prompt: object = None,
        user_extra_system_prompt: object = None,
    ) -> None:
        """Record the LLM data requested by the conversation entity."""
        chat_log.llm_input_provided_index = len(chat_log.content)
        provided_llm_data.append(
            (
                llm_context,
                user_llm_hass_api,
                user_llm_prompt,
                user_extra_system_prompt,
            )
        )

    monkeypatch.setattr(
        chat_log,
        "async_provide_llm_data",
        fake_provide_llm_data,
    )

    result = await entity._async_handle_message(user_input, chat_log)

    assert result.conversation_id == "conversation-id"
    assert result.response.speech["plain"]["speech"] == "The lights are on."
    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.call_args.kwargs
    assert request["stream"] is True
    assert any(
        message["role"] == "user" and message["content"] == "Turn on the lights"
        for message in request["messages"]
    )
    assert len(provided_llm_data) == 1
    llm_context, llm_hass_api, llm_prompt, extra_system_prompt = provided_llm_data[0]
    assert llm_context.platform == DOMAIN
    assert llm_context.context is user_input.context
    assert llm_context.device_id == "voice-pe"
    assert llm_hass_api == RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API]
    assert llm_prompt == RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT]
    assert extra_system_prompt == "Keep it short."
