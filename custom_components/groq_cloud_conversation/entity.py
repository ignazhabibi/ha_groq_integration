"""Shared LLM entity support for Groq Cloud Conversation."""

import base64
import json
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Literal, cast

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
from homeassistant.util import slugify
from voluptuous_openapi import convert

from .api import GroqApiError, GroqAuthenticationError, GroqRateLimitError
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VISION_MODEL,
    DOMAIN,
    GROQ_STRUCTURED_OUTPUT_MODEL_IDS,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STRUCTURED_OUTPUT_MODEL,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
)
from .model_registry import GroqCapability

if TYPE_CHECKING:
    from . import GroqCloudConfigEntry

JsonSchema = dict[str, Any]
ChatCompletionMessageParam = dict[str, Any]
ChatCompletionToolParam = dict[str, Any]

MAX_TOOL_ITERATIONS = 10
MAX_VISION_IMAGES = 5
MAX_BASE64_IMAGE_SIZE = 4 * 1024 * 1024


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
    """Adjust structured output schemas to the Chat Completions shape."""
    schema_type = schema.get("type")
    if schema_type == "object":
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


def _format_chat_completion_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format a Home Assistant LLM tool for Chat Completions."""
    return {
        "function": {
            "description": tool.description or "",
            "name": tool.name,
            "parameters": _format_tool_parameters(tool, custom_serializer),
            "strict": False,
        },
        "type": "function",
    }


def _decode_tool_call_arguments(tool_call: Mapping[str, str]) -> dict[str, Any]:
    """Decode a Groq tool call argument payload."""
    try:
        arguments = json.loads(tool_call["arguments"] or "{}")
    except JSONDecodeError as err:
        raise HomeAssistantError("Groq returned malformed tool call arguments") from err

    if not isinstance(arguments, dict):
        raise HomeAssistantError("Groq returned non-object tool call arguments")

    return cast("dict[str, Any]", arguments)


def _convert_content_to_chat_completion_param(
    chat_content: Iterable[conversation.Content],
) -> list[ChatCompletionMessageParam]:
    """Convert Home Assistant chat content to Chat Completions messages."""
    messages: list[ChatCompletionMessageParam] = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "content": json_dumps(content.tool_result),
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                }
            )
            continue

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            messages.append(
                {
                    "content": content.content or None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json_dumps(tool_call.tool_args),
                                "name": tool_call.tool_name,
                            },
                            "id": tool_call.id,
                            "type": "function",
                        }
                        for tool_call in content.tool_calls
                    ],
                }
            )
            continue

        if content.content:
            messages.append(
                {
                    "content": content.content,
                    "role": content.role,
                }
            )

    return messages


async def _async_convert_content_to_chat_completion_param(
    hass: HomeAssistant,
    chat_content: Iterable[conversation.Content],
) -> list[ChatCompletionMessageParam]:
    """Convert Home Assistant chat content, including attachments, to messages."""
    messages: list[ChatCompletionMessageParam] = []

    for content in chat_content:
        if isinstance(content, conversation.UserContent) and content.attachments:
            messages.append(
                {
                    "content": await _async_format_user_content_with_attachments(
                        hass,
                        content,
                    ),
                    "role": "user",
                }
            )
            continue

        messages.extend(_convert_content_to_chat_completion_param([content]))

    return messages


async def _async_format_user_content_with_attachments(
    hass: HomeAssistant,
    content: conversation.UserContent,
) -> list[dict[str, Any]]:
    """Format user text and image attachments for Groq vision models."""
    attachments = content.attachments or []
    if len(attachments) > MAX_VISION_IMAGES:
        raise HomeAssistantError("Groq vision supports at most 5 images per request")

    message_content: list[dict[str, Any]] = []
    if content.content:
        message_content.append({"text": content.content, "type": "text"})

    for attachment in attachments:
        if not attachment.mime_type.startswith("image/"):
            raise HomeAssistantError("Groq vision attachments must be images")
        image_url = await hass.async_add_executor_job(
            _attachment_to_data_url,
            attachment,
        )
        message_content.append(
            {
                "image_url": {"url": image_url},
                "type": "image_url",
            }
        )

    return message_content


def _attachment_to_data_url(attachment: conversation.Attachment) -> str:
    """Read an image attachment and return a base64 data URL."""
    encoded = base64.b64encode(attachment.path.read_bytes()).decode()
    if len(encoded) > MAX_BASE64_IMAGE_SIZE:
        raise HomeAssistantError("Groq vision image attachments must be under 4 MB")
    return f"data:{attachment.mime_type};base64,{encoded}"


def _chat_content_has_attachments(
    chat_content: Iterable[conversation.Content],
) -> bool:
    """Return whether chat content includes user attachments."""
    return any(
        isinstance(content, conversation.UserContent) and bool(content.attachments)
        for content in chat_content
    )


async def _transform_chat_completion_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncGenerator[dict[str, Any]],
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Chat Completions stream into Home Assistant chat deltas."""
    tool_calls: dict[int, dict[str, str]] = {}
    last_role: Literal["assistant"] | None = None

    async for chunk in stream:
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            continue

        LOGGER.debug(
            "Received Groq chat completion chunk with %d choices",
            len(choices),
        )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    }
                }
            )

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("role") == "assistant" and last_role != "assistant":
                yield {"role": "assistant"}
                last_role = "assistant"

            if content := delta.get("content"):
                if last_role != "assistant":
                    yield {"role": "assistant"}
                    last_role = "assistant"
                yield {"content": content}

            if tool_call_deltas := delta.get("tool_calls"):
                if last_role != "assistant":
                    yield {"role": "assistant"}
                    last_role = "assistant"
                for tool_call_delta in tool_call_deltas:
                    if not isinstance(tool_call_delta, dict):
                        continue
                    tool_call = tool_calls.setdefault(
                        int(tool_call_delta["index"]),
                        {"arguments": "", "id": "", "name": ""},
                    )
                    if tool_call_id := tool_call_delta.get("id"):
                        tool_call["id"] = str(tool_call_id)
                    function_delta = tool_call_delta.get("function")
                    if not isinstance(function_delta, dict):
                        continue
                    if function_name := function_delta.get("name"):
                        tool_call["name"] = str(function_name)
                    if function_arguments := function_delta.get("arguments"):
                        tool_call["arguments"] += str(function_arguments)

            if choice.get("finish_reason") == "tool_calls" and tool_calls:
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_args=_decode_tool_call_arguments(tool_call),
                            tool_name=tool_call["name"],
                        )
                        for _, tool_call in sorted(tool_calls.items())
                    ]
                }
                tool_calls.clear()


def _add_chat_completion_usage_to_trace(
    chat_log: conversation.ChatLog,
    completion: dict[str, Any],
) -> None:
    """Add Chat Completions token usage to the Home Assistant trace."""
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return

    chat_log.async_trace(
        {
            "stats": {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }
        }
    )


def _model_for_structured_output(model: str) -> str:
    """Return a model compatible with Groq Structured Outputs."""
    if model in GROQ_STRUCTURED_OUTPUT_MODEL_IDS:
        return model
    return RECOMMENDED_STRUCTURED_OUTPUT_MODEL


def _assistant_content_from_completion(
    entity_id: str,
    completion: dict[str, Any],
) -> conversation.AssistantContent:
    """Return assistant content from a non-streaming Chat Completion."""
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HomeAssistantError("Groq returned no choices")
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return conversation.AssistantContent(
        agent_id=entity_id,
        content=content if isinstance(content, str) else None,
    )


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
        has_attachments = _chat_content_has_attachments(chat_log.content)
        if has_attachments and structure:
            raise HomeAssistantError(
                "Groq structured output with image attachments is not supported"
            )
        if has_attachments:
            await self._async_handle_vision_chat_log(chat_log)
            return

        if structure and structure_name:
            await self._async_handle_structured_chat_log(
                chat_log,
                structure_name,
                structure,
            )
            return

        options = self.subentry.data
        messages = await _async_convert_content_to_chat_completion_param(
            self.hass,
            chat_log.content,
        )

        model_args: dict[str, Any] = {
            "max_completion_tokens": options.get(
                CONF_MAX_TOKENS,
                RECOMMENDED_MAX_TOKENS,
            ),
            "messages": messages,
            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "user": chat_log.conversation_id,
        }

        if chat_log.llm_api:
            tools = [
                _format_chat_completion_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
            if tools:
                model_args["tool_choice"] = "auto"
                model_args["tools"] = tools

        client = self.entry.runtime_data.client

        for _iteration in range(max_iterations):
            try:
                stream = client.async_stream_chat_completion(model_args)
                content_stream = chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_chat_completion_stream(chat_log, stream),
                )
                messages.extend(
                    _convert_content_to_chat_completion_param(
                        [content async for content in content_stream]
                    )
                )
            except GroqAuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Authentication error with Groq") from err
            except GroqRateLimitError as err:
                LOGGER.error("Rate limited by Groq: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except GroqApiError as err:
                LOGGER.error("Error talking to Groq: %s", err)
                raise HomeAssistantError("Error talking to Groq") from err

            if not chat_log.unresponded_tool_results:
                break
        else:
            raise HomeAssistantError("Groq tool call limit reached")

    async def _async_handle_vision_chat_log(
        self,
        chat_log: conversation.ChatLog,
    ) -> None:
        """Generate a Groq vision response for chat content with images."""
        options = self.subentry.data
        model = str(options.get(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL))
        if not self.entry.runtime_data.model_registry.supports(
            model,
            GroqCapability.VISION,
        ):
            model = RECOMMENDED_VISION_MODEL

        model_args: dict[str, Any] = {
            "max_completion_tokens": options.get(
                CONF_MAX_TOKENS,
                RECOMMENDED_MAX_TOKENS,
            ),
            "messages": await _async_convert_content_to_chat_completion_param(
                self.hass,
                chat_log.content,
            ),
            "model": model,
            "stream": False,
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "user": chat_log.conversation_id,
        }

        client = self.entry.runtime_data.client

        try:
            completion = await client.async_chat_completion(model_args)
        except GroqAuthenticationError as err:
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError("Authentication error with Groq") from err
        except GroqRateLimitError as err:
            LOGGER.error("Rate limited by Groq: %s", err)
            raise HomeAssistantError("Rate limited or insufficient funds") from err
        except GroqApiError as err:
            LOGGER.error("Error talking to Groq: %s", err)
            raise HomeAssistantError("Error talking to Groq") from err

        _add_chat_completion_usage_to_trace(chat_log, completion)
        chat_log.async_add_assistant_content_without_tools(
            _assistant_content_from_completion(self.entity_id, completion)
        )

    async def _async_handle_structured_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str,
        structure: vol.Schema,
    ) -> None:
        """Generate a structured Groq response through Chat Completions."""
        options = self.subentry.data
        model_args: dict[str, Any] = {
            "max_completion_tokens": options.get(
                CONF_MAX_TOKENS,
                RECOMMENDED_MAX_TOKENS,
            ),
            "messages": await _async_convert_content_to_chat_completion_param(
                self.hass,
                chat_log.content,
            ),
            "model": _model_for_structured_output(
                options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            ),
            "response_format": {
                "json_schema": {
                    "name": slugify(structure_name),
                    "schema": _format_structured_output(
                        structure,
                        chat_log.llm_api,
                    ),
                    "strict": True,
                },
                "type": "json_schema",
            },
            "stream": False,
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "user": chat_log.conversation_id,
        }

        client = self.entry.runtime_data.client

        try:
            completion = await client.async_chat_completion(model_args)
        except GroqAuthenticationError as err:
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError("Authentication error with Groq") from err
        except GroqRateLimitError as err:
            LOGGER.error("Rate limited by Groq: %s", err)
            raise HomeAssistantError("Rate limited or insufficient funds") from err
        except GroqApiError as err:
            LOGGER.error("Error talking to Groq: %s", err)
            raise HomeAssistantError("Error talking to Groq") from err

        _add_chat_completion_usage_to_trace(chat_log, completion)
        chat_log.async_add_assistant_content_without_tools(
            _assistant_content_from_completion(self.entity_id, completion)
        )
