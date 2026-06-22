"""Tests for the shared Groq Cloud LLM entity adapter."""

from collections.abc import AsyncIterator
from http import HTTPStatus
from types import MappingProxyType
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
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
    _convert_content_to_chat_completion_param,
)

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


def _chat_message_chunks(text: str) -> list[ChatCompletionChunk]:
    """Return fake Chat Completions chunks for an assistant text response."""
    return [
        ChatCompletionChunk.model_construct(
            id="chatcmpl_1",
            choices=[
                ChunkChoice.model_construct(
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


def _chat_tool_call_chunks(
    arguments: str = '{"value": "lamp"}',
) -> list[ChatCompletionChunk]:
    """Return fake Chat Completions chunks for a function tool call."""
    return [
        ChatCompletionChunk.model_construct(
            id="chatcmpl_1",
            choices=[
                ChunkChoice.model_construct(
                    delta=ChoiceDelta.model_construct(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall.model_construct(
                                function=ChoiceDeltaToolCallFunction.model_construct(
                                    arguments="",
                                    name="Echo",
                                ),
                                id="call_1",
                                index=0,
                                type="function",
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=0,
            model=RECOMMENDED_CHAT_MODEL,
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk.model_construct(
            id="chatcmpl_1",
            choices=[
                ChunkChoice.model_construct(
                    delta=ChoiceDelta.model_construct(
                        tool_calls=[
                            ChoiceDeltaToolCall.model_construct(
                                function=ChoiceDeltaToolCallFunction.model_construct(
                                    arguments=arguments,
                                ),
                                index=0,
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                    index=0,
                )
            ],
            created=0,
            model=RECOMMENDED_CHAT_MODEL,
            object="chat.completion.chunk",
        ),
    ]


def _chat_completion(text: str) -> ChatCompletion:
    """Return a fake non-streaming Chat Completion response."""
    return ChatCompletion.model_construct(
        id="chatcmpl_1",
        choices=[
            ChatChoice.model_construct(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage.model_construct(
                    content=text,
                    role="assistant",
                ),
            )
        ],
        created=0,
        model=RECOMMENDED_CHAT_MODEL,
        object="chat.completion",
        usage=CompletionUsage.model_construct(
            completion_tokens=3,
            prompt_tokens=7,
            total_tokens=10,
        ),
    )


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


def _make_openai_response(status_code: HTTPStatus) -> httpx.Response:
    """Create an HTTPX response for OpenAI status errors."""
    return httpx.Response(
        status_code,
        request=httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
    )


async def _handle_chat_log_for_error_case(
    entity: DummyLLMEntity,
    chat_log: conversation.ChatLog,
    is_structured: bool,
) -> None:
    """Handle a chat log through the requested Groq response path."""
    if is_structured:
        await entity._async_handle_chat_log(
            chat_log,
            structure=vol.Schema({vol.Required("value"): str}),
            structure_name="Extract data",
        )
        return

    await entity._async_handle_chat_log(chat_log)


def test_convert_content_to_chat_completion_input() -> None:
    """Test Home Assistant chat content is converted to Chat Completions input."""
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

    messages = _convert_content_to_chat_completion_param(content)
    message_dicts = cast("list[dict[str, Any]]", messages)

    assert message_dicts[0]["role"] == "system"
    assert message_dicts[1]["role"] == "user"
    assert message_dicts[2]["role"] == "assistant"
    assert message_dicts[2]["tool_calls"][0]["id"] == "call_1"
    assert message_dicts[3]["role"] == "tool"
    assert message_dicts[3]["tool_call_id"] == "call_1"


async def test_handle_chat_log_streams_text(hass: HomeAssistant) -> None:
    """Test streamed text deltas are added to the Home Assistant chat log."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=FakeStream(_chat_message_chunks("Hi"))
    )
    entity = _make_entity(client, _make_subentry())
    chat_log = _make_chat_log(hass)

    await entity._async_handle_chat_log(chat_log)

    assert isinstance(chat_log.content[-1], conversation.AssistantContent)
    assert chat_log.content[-1].content == "Hi"
    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == RECOMMENDED_CHAT_MODEL
    assert request["max_completion_tokens"] == RECOMMENDED_MAX_TOKENS
    assert request["temperature"] == RECOMMENDED_TEMPERATURE
    assert request["top_p"] == RECOMMENDED_TOP_P
    assert "tools" not in request


async def test_handle_chat_log_runs_ha_tool_round_trip(
    hass: HomeAssistant,
) -> None:
    """Test Groq function calls are executed through a Home Assistant LLM API."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            FakeStream(_chat_tool_call_chunks()),
            FakeStream(_chat_message_chunks("Done")),
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

    assert client.chat.completions.create.await_count == 2
    first_request = client.chat.completions.create.await_args_list[0].kwargs
    second_request = client.chat.completions.create.await_args_list[1].kwargs
    assert first_request["tools"][0]["function"]["name"] == "Echo"
    assert any(
        message["role"] == "tool" and message["tool_call_id"] == "call_1"
        for message in cast("list[dict[str, Any]]", second_request["messages"])
    )
    assert isinstance(chat_log.content[-1], conversation.AssistantContent)
    assert chat_log.content[-1].content == "Done"


async def test_handle_chat_log_uses_chat_completions_structured_output(
    hass: HomeAssistant,
) -> None:
    """Test structured responses use non-streaming Chat Completions."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_chat_completion('{"value": "ok"}')
    )
    entity = _make_entity(client, _make_subentry())
    chat_log = _make_chat_log(hass)

    await entity._async_handle_chat_log(
        chat_log,
        structure=vol.Schema({vol.Required("value"): str}),
        structure_name="Extract data",
    )

    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.call_args.kwargs
    assert request["stream"] is False
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["name"] == "extract_data"
    assert request["response_format"]["json_schema"]["strict"] is False
    assert "tools" not in request
    assert isinstance(chat_log.content[-1], conversation.AssistantContent)
    assert chat_log.content[-1].content == '{"value": "ok"}'


async def test_handle_chat_log_enforces_tool_iteration_cap(
    hass: HomeAssistant,
) -> None:
    """Test tool loops fail after the configured iteration cap."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            FakeStream(_chat_tool_call_chunks()),
            FakeStream(_chat_tool_call_chunks('{"value": "second"}')),
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

    assert client.chat.completions.create.await_count == 2


@pytest.mark.parametrize("is_structured", [False, True])
@pytest.mark.parametrize(
    "error_case",
    [
        (
            openai.RateLimitError(
                "Rate limited",
                response=_make_openai_response(HTTPStatus.TOO_MANY_REQUESTS),
                body=None,
            ),
            "Rate limited or insufficient funds",
            False,
        ),
        (
            openai.AuthenticationError(
                "Invalid API key",
                response=_make_openai_response(HTTPStatus.UNAUTHORIZED),
                body=None,
            ),
            "Authentication error with Groq",
            True,
        ),
    ],
)
async def test_handle_chat_log_maps_groq_status_errors(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    is_structured: bool,
    error_case: tuple[openai.APIStatusError, str, bool],
) -> None:
    """Test Groq status errors are exposed as Home Assistant errors."""
    exception, message, should_start_reauth = error_case
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exception)
    entity = _make_entity(client, _make_subentry())
    entity.hass = hass
    chat_log = _make_chat_log(hass)
    start_reauth = MagicMock()
    monkeypatch.setattr(entity.entry, "async_start_reauth", start_reauth)

    with pytest.raises(HomeAssistantError, match=message):
        await _handle_chat_log_for_error_case(entity, chat_log, is_structured)

    if should_start_reauth:
        start_reauth.assert_called_once_with(hass)
    else:
        start_reauth.assert_not_called()
