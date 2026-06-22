"""Shared LLM entity support for Groq Cloud Conversation."""

import json
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import openai
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
from homeassistant.util import slugify
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ToolParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_input_param import FunctionCallOutput
from voluptuous_openapi import convert

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

JsonSchema = dict[str, Any]

MAX_TOOL_ITERATIONS = 10


def _format_tool_parameters(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> JsonSchema:
    """Format a Home Assistant tool parameter schema for Groq."""
    unsupported_keys = {"allOf", "anyOf", "enum", "not", "oneOf"}
    schema = convert(tool.parameters, custom_serializer=custom_serializer)
    if unsupported_keys.intersection(schema):
        schema = {
            key: value for key, value in schema.items() if key not in unsupported_keys
        }
    return cast("JsonSchema", schema)


def _adjust_schema(schema: JsonSchema) -> None:
    """Adjust structured output schemas to the Responses API shape."""
    schema_type = schema.get("type")
    if schema_type == "object":
        schema.setdefault("strict", True)
        schema.setdefault("additionalProperties", False)
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return

        required = schema.setdefault("required", [])
        if not isinstance(required, list):
            schema["required"] = required = []

        for prop, prop_info in properties.items():
            if not isinstance(prop_info, dict):
                continue
            _adjust_schema(prop_info)
            if prop not in required:
                if "type" in prop_info:
                    prop_info["type"] = [prop_info["type"], "null"]
                required.append(prop)
        return

    if schema_type == "array" and isinstance(schema.get("items"), dict):
        _adjust_schema(schema["items"])


def _format_structured_output(
    schema: vol.Schema, llm_api: llm.APIInstance | None
) -> JsonSchema:
    """Format a Home Assistant schema for Groq structured responses."""
    result = cast(
        "JsonSchema",
        convert(
            schema,
            custom_serializer=(
                llm_api.custom_serializer if llm_api else llm.selector_serializer
            ),
        ),
    )
    _adjust_schema(result)
    return result


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format a Home Assistant LLM tool for the Responses API."""
    return FunctionToolParam(
        description=tool.description,
        name=tool.name,
        parameters=_format_tool_parameters(tool, custom_serializer),
        strict=False,
        type="function",
    )


def _format_chat_completion_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format a Home Assistant LLM tool for Chat Completions."""
    return ChatCompletionToolParam(
        function={
            "description": tool.description or "",
            "name": tool.name,
            "parameters": _format_tool_parameters(tool, custom_serializer),
            "strict": False,
        },
        type="function",
    )


def _convert_content_to_param(
    chat_content: Iterable[conversation.Content],
) -> ResponseInputParam:
    """Convert Home Assistant chat content to Responses API input."""
    messages: ResponseInputParam = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                FunctionCallOutput(
                    call_id=content.tool_call_id,
                    output=json_dumps(content.tool_result),
                    type="function_call_output",
                )
            )
            continue

        if content.content:
            role: Literal["user", "assistant", "system", "developer"] = content.role
            messages.append(
                EasyInputMessageParam(
                    content=content.content,
                    role=role,
                    type="message",
                )
            )

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            for tool_call in content.tool_calls:
                messages.append(
                    ResponseFunctionToolCallParam(
                        arguments=json_dumps(tool_call.tool_args),
                        call_id=tool_call.id,
                        name=tool_call.tool_name,
                        type="function_call",
                    )
                )

    return messages


def _convert_content_to_chat_completion_param(
    chat_content: Iterable[conversation.Content],
) -> list[ChatCompletionMessageParam]:
    """Convert Home Assistant chat content to Chat Completions messages."""
    messages: list[ChatCompletionMessageParam] = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                cast(
                    "ChatCompletionMessageParam",
                    {
                        "content": json_dumps(content.tool_result),
                        "role": "tool",
                        "tool_call_id": content.tool_call_id,
                    },
                )
            )
            continue

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            messages.append(
                cast(
                    "ChatCompletionMessageParam",
                    {
                        "content": content.content or None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": Function(
                                    arguments=json_dumps(tool_call.tool_args),
                                    name=tool_call.tool_name,
                                ),
                                "id": tool_call.id,
                                "type": "function",
                            }
                            for tool_call in content.tool_calls
                        ],
                    },
                )
            )
            continue

        if content.content:
            messages.append(
                cast(
                    "ChatCompletionMessageParam",
                    {
                        "content": content.content,
                        "role": content.role,
                    },
                )
            )

    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Responses API stream into Home Assistant chat deltas."""
    current_tool_call: ResponseFunctionToolCall | None = None
    last_role: Literal["assistant"] | None = None

    async for event in stream:
        LOGGER.debug("Received Groq response event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                yield {"role": "assistant"}
                current_tool_call = event.item
                last_role = "assistant"
            elif (
                isinstance(event.item, ResponseOutputMessage)
                or last_role != "assistant"
            ):
                yield {"role": "assistant"}
                last_role = "assistant"
        elif isinstance(event, ResponseTextDeltaEvent):
            if event.delta:
                yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            if current_tool_call is not None:
                current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            if current_tool_call is None:
                continue
            current_tool_call.status = "completed"
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call.call_id,
                        tool_args=json.loads(current_tool_call.arguments),
                        tool_name=current_tool_call.name,
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, ResponseIncompleteEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if event.response.incomplete_details is not None:
                reason = event.response.incomplete_details.reason or reason
            if reason == "max_output_tokens":
                reason = "max output tokens reached"
            raise HomeAssistantError(f"Groq response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if event.response.error is not None:
                reason = event.response.error.message
            raise HomeAssistantError(f"Groq response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"Groq response error: {event.message}")


async def _transform_chat_completion_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Chat Completions stream into Home Assistant chat deltas."""
    tool_calls: dict[int, dict[str, str]] = {}
    last_role: Literal["assistant"] | None = None

    async for chunk in stream:
        LOGGER.debug("Received Groq chat completion chunk: %s", chunk)

        if chunk.usage is not None:
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                }
            )

        for choice in chunk.choices:
            delta = choice.delta
            if delta.role == "assistant" and last_role != "assistant":
                yield {"role": "assistant"}
                last_role = "assistant"

            if delta.content:
                if last_role != "assistant":
                    yield {"role": "assistant"}
                    last_role = "assistant"
                yield {"content": delta.content}

            if delta.tool_calls:
                if last_role != "assistant":
                    yield {"role": "assistant"}
                    last_role = "assistant"
                for tool_call_delta in delta.tool_calls:
                    tool_call = tool_calls.setdefault(
                        tool_call_delta.index,
                        {"arguments": "", "id": "", "name": ""},
                    )
                    if tool_call_delta.id:
                        tool_call["id"] = tool_call_delta.id
                    if tool_call_delta.function is None:
                        continue
                    if tool_call_delta.function.name:
                        tool_call["name"] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_call["arguments"] += tool_call_delta.function.arguments

            if choice.finish_reason == "tool_calls" and tool_calls:
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_args=json.loads(tool_call["arguments"] or "{}"),
                            tool_name=tool_call["name"],
                        )
                        for _, tool_call in sorted(tool_calls.items())
                    ]
                }
                tool_calls.clear()


class GroqCloudBaseLLMEntity(Entity):
    """Base entity for Groq Cloud conversation and AI task entities."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: GroqCloudConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the Groq Cloud base entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            entry_type=dr.DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, subentry.subentry_id)},
            manufacturer="Groq",
            model=subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            name=subentry.title,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate a Groq response for the given Home Assistant chat log."""
        if chat_log.llm_api and chat_log.llm_api.tools and not structure:
            await self._async_handle_chat_completion_chat_log(
                chat_log,
                max_iterations,
            )
            return

        options = self.subentry.data
        messages = _convert_content_to_param(chat_log.content)

        model_args = ResponseCreateParamsStreaming(
            input=messages,
            max_output_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            stream=True,
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            user=chat_log.conversation_id,
        )

        tools: list[ToolParam] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        if tools:
            model_args["tools"] = tools

        if structure and structure_name:
            model_args["text"] = {
                "format": {
                    "name": slugify(structure_name),
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                    "type": "json_schema",
                },
            }

        client = self.entry.runtime_data

        for _iteration in range(max_iterations):
            try:
                stream = await client.responses.create(**model_args)
                content_stream = chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_stream(chat_log, stream),
                )
                messages.extend(
                    _convert_content_to_param(
                        [content async for content in content_stream]
                    )
                )
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Authentication error with Groq") from err
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by Groq: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Groq: %s", err)
                raise HomeAssistantError("Error talking to Groq") from err

            if not chat_log.unresponded_tool_results:
                break
        else:
            raise HomeAssistantError("Groq tool call limit reached")

    async def _async_handle_chat_completion_chat_log(
        self,
        chat_log: conversation.ChatLog,
        max_iterations: int,
    ) -> None:
        """Generate a Groq response through Chat Completions for tool calls."""
        llm_api = chat_log.llm_api
        if llm_api is None:
            raise HomeAssistantError("Home Assistant LLM API is not available")

        options = self.subentry.data
        messages = _convert_content_to_chat_completion_param(chat_log.content)
        tools = [
            _format_chat_completion_tool(tool, llm_api.custom_serializer)
            for tool in llm_api.tools
        ]

        model_args = CompletionCreateParamsStreaming(
            max_completion_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            messages=messages,
            model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            stream=True,
            stream_options={"include_usage": True},
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            tool_choice="auto",
            tools=tools,
            top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            user=chat_log.conversation_id,
        )

        client = self.entry.runtime_data

        for _iteration in range(max_iterations):
            try:
                stream = await client.chat.completions.create(**model_args)
                content_stream = chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_chat_completion_stream(chat_log, stream),
                )
                messages.extend(
                    _convert_content_to_chat_completion_param(
                        [content async for content in content_stream]
                    )
                )
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Authentication error with Groq") from err
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by Groq: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Groq: %s", err)
                raise HomeAssistantError("Error talking to Groq") from err

            if not chat_log.unresponded_tool_results:
                break
        else:
            raise HomeAssistantError("Groq tool call limit reached")
