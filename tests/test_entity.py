"""Tests for the shared Groq Cloud LLM entity adapter."""

from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import GroqCloudConfigEntry
from custom_components.groq_cloud_conversation.const import (
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from custom_components.groq_cloud_conversation.entity import (
    GroqCloudBaseLLMEntity,
    _convert_content_to_param,
)


class FakeStream:
    """Async iterator for fake Responses API streaming events."""

    def __init__(self, events: list[ResponseStreamEvent]) -> None:
        """Initialize the fake stream."""
        self._events = iter(events)

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        """Return the stream iterator."""
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        """Return the next fake stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class DummyLLMEntity(GroqCloudBaseLLMEntity):
    """Concrete entity for testing the shared base class."""


class FakeAPI(llm.API):
    """Minimal LLM API for tool-call tests."""

    async def async_get_api_instance(
        self,
        llm_context: llm.LLMContext,
    ) -> llm.APIInstance:
        """Return an empty API instance."""
        return llm.APIInstance(
            api=self,
            api_prompt="",
            llm_context=llm_context,
            tools=[],
        )


class EchoTool(llm.Tool):
    """Tool that returns the provided value."""

    name = "Echo"
    description = "Echo a value."
    parameters = vol.Schema({vol.Required("value"): str})

    async def async_call(
        self,
        _hass: HomeAssistant,
        tool_input: llm.ToolInput,
        _llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Return the requested value."""
        return {"value": tool_input.tool_args["value"]}


def _message_events(text: str) -> list[ResponseStreamEvent]:
    """Return fake stream events for an assistant text response."""
    return [
        cast(
            "ResponseStreamEvent",
            ResponseOutputItemAddedEvent.model_construct(
                item=ResponseOutputMessage.model_construct(
                    id="msg_1",
                    content=[],
                    role="assistant",
                    status="in_progress",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
                type="response.output_item.added",
            ),
        ),
        cast(
            "ResponseStreamEvent",
            ResponseTextDeltaEvent.model_construct(
                content_index=0,
                delta=text,
                item_id="msg_1",
                logprobs=[],
                output_index=0,
                sequence_number=1,
                type="response.output_text.delta",
            ),
        ),
    ]


def _tool_call_events(
    arguments: str = '{"value": "lamp"}',
) -> list[ResponseStreamEvent]:
    """Return fake stream events for a function tool call."""
    return [
        cast(
            "ResponseStreamEvent",
            ResponseOutputItemAddedEvent.model_construct(
                item=ResponseFunctionToolCall.model_construct(
                    arguments="",
                    call_id="call_1",
                    id="fc_1",
                    name="Echo",
                    status="in_progress",
                    type="function_call",
                ),
                output_index=0,
                sequence_number=0,
                type="response.output_item.added",
            ),
        ),
        cast(
            "ResponseStreamEvent",
            ResponseFunctionCallArgumentsDeltaEvent.model_construct(
                delta=arguments,
                item_id="fc_1",
                output_index=0,
                sequence_number=1,
                type="response.function_call_arguments.delta",
            ),
        ),
        cast(
            "ResponseStreamEvent",
            ResponseFunctionCallArgumentsDoneEvent.model_construct(
                arguments=arguments,
                item_id="fc_1",
                name="Echo",
                output_index=0,
                sequence_number=2,
                type="response.function_call_arguments.done",
            ),
        ),
    ]


def _make_subentry(
    data: dict[str, Any] | None = None,
    subentry_type: str = "conversation",
) -> ConfigSubentry:
    """Create a config subentry for entity tests."""
    options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
    if data:
        options.update(data)
    return ConfigSubentry(
        data=MappingProxyType(options),
        subentry_id="subentry-id",
        subentry_type=subentry_type,
        title="Groq Cloud",
        unique_id=None,
    )


def _make_entity(client: MagicMock, subentry: ConfigSubentry) -> DummyLLMEntity:
    """Create a test entity with fake Groq runtime data."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, title="Groq Cloud")
    entry.runtime_data = client
    entity = DummyLLMEntity(cast("GroqCloudConfigEntry", entry), subentry)
    entity.entity_id = "conversation.groq_cloud"
    return entity


def _make_chat_log(hass: HomeAssistant) -> conversation.ChatLog:
    """Create a chat log with a single user message."""
    chat_log = conversation.ChatLog(hass, "conversation-id")
    chat_log.async_add_user_content(conversation.UserContent("Hello"))
    return chat_log


def test_convert_content_to_responses_input() -> None:
    """Test Home Assistant chat content is converted to Responses input."""
    tool_input = llm.ToolInput(
        id="call_1",
        tool_args={"value": "lamp"},
        tool_name="Echo",
    )
    content: list[conversation.Content] = [
        conversation.SystemContent("Be concise."),
        conversation.UserContent("Turn on the lamp."),
        conversation.AssistantContent(
            agent_id="conversation.groq_cloud",
            tool_calls=[tool_input],
        ),
        conversation.ToolResultContent(
            agent_id="conversation.groq_cloud",
            tool_call_id="call_1",
            tool_name="Echo",
            tool_result={"value": "lamp"},
        ),
    ]

    messages = _convert_content_to_param(content)
    message_dicts = cast("list[dict[str, Any]]", messages)

    assert message_dicts[0]["role"] == "system"
    assert message_dicts[1]["role"] == "user"
    assert message_dicts[2]["type"] == "function_call"
    assert message_dicts[3]["type"] == "function_call_output"


async def test_handle_chat_log_streams_text(hass: HomeAssistant) -> None:
    """Test streamed text deltas are added to the Home Assistant chat log."""
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=FakeStream(_message_events("Hi")))
    entity = _make_entity(client, _make_subentry())
    chat_log = _make_chat_log(hass)

    await entity._async_handle_chat_log(chat_log)

    assert isinstance(chat_log.content[-1], conversation.AssistantContent)
    assert chat_log.content[-1].content == "Hi"
    client.responses.create.assert_awaited_once()
    request = client.responses.create.call_args.kwargs
    assert request["model"] == RECOMMENDED_CHAT_MODEL
    assert request["max_output_tokens"] == RECOMMENDED_MAX_TOKENS
    assert request["temperature"] == RECOMMENDED_TEMPERATURE
    assert request["top_p"] == RECOMMENDED_TOP_P


async def test_handle_chat_log_runs_ha_tool_round_trip(
    hass: HomeAssistant,
) -> None:
    """Test Groq function calls are executed through a Home Assistant LLM API."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[
            FakeStream(_tool_call_events()),
            FakeStream(_message_events("Done")),
        ]
    )
    entity = _make_entity(client, _make_subentry())
    chat_log = _make_chat_log(hass)
    llm_context = llm.LLMContext(
        platform="conversation",
        context=None,
        language="en",
        assistant=DOMAIN,
        device_id=None,
    )
    chat_log.llm_api = llm.APIInstance(
        api=FakeAPI(hass=hass, id="fake", name="Fake"),
        api_prompt="",
        llm_context=llm_context,
        tools=[EchoTool()],
    )

    await entity._async_handle_chat_log(chat_log)

    assert client.responses.create.await_count == 2
    first_request = client.responses.create.await_args_list[0].kwargs
    second_request = client.responses.create.await_args_list[1].kwargs
    assert first_request["tools"][0]["name"] == "Echo"
    assert any(
        message["type"] == "function_call_output" and message["call_id"] == "call_1"
        for message in cast("ResponseInputParam", second_request["input"])
    )
    assert isinstance(chat_log.content[-1], conversation.AssistantContent)
    assert chat_log.content[-1].content == "Done"


async def test_handle_chat_log_enforces_tool_iteration_cap(
    hass: HomeAssistant,
) -> None:
    """Test tool loops fail after the configured iteration cap."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[
            FakeStream(_tool_call_events()),
            FakeStream(_tool_call_events('{"value": "second"}')),
        ]
    )
    entity = _make_entity(client, _make_subentry())
    chat_log = _make_chat_log(hass)
    llm_context = llm.LLMContext(
        platform="conversation",
        context=None,
        language="en",
        assistant=DOMAIN,
        device_id=None,
    )
    chat_log.llm_api = llm.APIInstance(
        api=FakeAPI(hass=hass, id="fake", name="Fake"),
        api_prompt="",
        llm_context=llm_context,
        tools=[EchoTool()],
    )

    with pytest.raises(HomeAssistantError, match="tool call limit"):
        await entity._async_handle_chat_log(chat_log, max_iterations=2)

    assert client.responses.create.await_count == 2
