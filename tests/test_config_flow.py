"""Tests for the Groq Cloud Conversation config flow."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
import voluptuous as vol
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import SelectSelector
from openai.types import Model
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation.config_flow import (
    _model_selector_options,
    validate_input,
)
from custom_components.groq_cloud_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_RECOMMENDED,
    CONF_STT_MODEL,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DOMAIN,
    RECOMMENDED_STT_MODEL,
)


def _auth_error() -> openai.AuthenticationError:
    """Return a reusable OpenAI authentication error."""
    request = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("invalid key", response=response, body=None)


def _connection_error() -> openai.APIConnectionError:
    """Return a reusable OpenAI connection error."""
    request = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    return openai.APIConnectionError(request=request)


def _model(model_id: str) -> Model:
    """Return an OpenAI model object for flow tests."""
    return Model.model_construct(
        id=model_id,
        created=0,
        object="model",
        owned_by="groq",
    )


def _get_schema_field(schema: vol.Schema, field_name: str) -> SelectSelector:
    """Return a validator from a Home Assistant flow schema."""
    for key, validator in schema.schema.items():
        if key.schema == field_name:
            return cast("SelectSelector", validator)
    raise AssertionError(f"Missing schema field: {field_name}")


async def test_user_flow_creates_default_subentries(hass: HomeAssistant) -> None:
    """Test a successful setup stores the API key and creates subentries."""
    with patch(
        "custom_components.groq_cloud_conversation.config_flow.validate_input",
        AsyncMock(return_value=None),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": "user"},
            data={CONF_API_KEY: "groq-key"},
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Groq Cloud"
    assert result["data"] == {CONF_API_KEY: "groq-key"}

    entry = result["result"]
    subentries = list(entry.subentries.values())
    assert {subentry.subentry_type for subentry in subentries} == {
        "ai_task_data",
        "conversation",
        "stt",
    }
    assert {subentry.title for subentry in subentries} == {
        DEFAULT_AI_TASK_NAME,
        DEFAULT_CONVERSATION_NAME,
        DEFAULT_STT_NAME,
    }
    assert all(subentry.data[CONF_RECOMMENDED] for subentry in subentries)


async def test_user_flow_aborts_duplicate_api_key(hass: HomeAssistant) -> None:
    """Test duplicate API keys cannot be configured twice."""
    MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    ).add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data={CONF_API_KEY: "groq-key"},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


@pytest.mark.parametrize(
    ("error_factory", "expected_error"),
    [
        (_auth_error, "invalid_auth"),
        (_connection_error, "cannot_connect"),
        (lambda: openai.OpenAIError("boom"), "unknown"),
    ],
)
async def test_user_flow_maps_validation_errors(
    hass: HomeAssistant,
    error_factory: Callable[[], Exception],
    expected_error: str,
) -> None:
    """Test Groq validation errors are shown in the setup form."""
    with patch(
        "custom_components.groq_cloud_conversation.config_flow.validate_input",
        AsyncMock(side_effect=error_factory()),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": "user"},
            data={CONF_API_KEY: "groq-key"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": expected_error}


async def test_reauth_updates_api_key(hass: HomeAssistant) -> None:
    """Test reauthentication replaces the stored Groq API key."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "old-key"},
        title="Groq Cloud",
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.groq_cloud_conversation.config_flow.validate_input",
        AsyncMock(return_value=None),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": SOURCE_REAUTH, "entry_id": entry.entry_id},
        )
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "reauth_confirm"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={CONF_API_KEY: "new-key"},
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data[CONF_API_KEY] == "new-key"


async def test_validate_input_uses_groq_base_url_and_timeout(
    hass: HomeAssistant,
) -> None:
    """Test validation creates a Groq OpenAI-compatible client."""
    models = AsyncMock()
    client = AsyncMock()
    client.models.list = models

    with patch(
        "custom_components.groq_cloud_conversation.config_flow.openai.AsyncOpenAI",
        return_value=client,
    ) as mock_client:
        await validate_input(hass, {CONF_API_KEY: "groq-key"})

    assert mock_client.call_args.kwargs["api_key"] == "groq-key"
    assert str(mock_client.call_args.kwargs["base_url"]) == (
        "https://api.groq.com/openai/v1"
    )
    models.assert_awaited_once_with(timeout=10.0)


def test_model_selector_options_label_and_filter_models() -> None:
    """Test model options distinguish known groups and filter incompatible IDs."""
    options = _model_selector_options(
        [
            "custom/live-chat-model",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "whisper-large-v3",
        ]
    )

    assert [option["value"] for option in options] == [
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "custom/live-chat-model",
    ]
    assert [option["label"] for option in options] == [
        "Production - Llama 3.3 70B (llama-3.3-70b-versatile)",
        "Preview - Qwen3-32B (qwen/qwen3-32b)",
        "Available - custom/live-chat-model",
    ]


async def test_subentry_advanced_step_uses_live_model_dropdown(
    hass: HomeAssistant,
) -> None:
    """Test advanced subentry options show live Groq models in a dropdown."""
    client = MagicMock()
    client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                _model("custom/live-chat-model"),
                _model("llama-3.1-8b-instant"),
                _model("meta-llama/llama-4-scout-17b-16e-instruct"),
                _model("whisper-large-v3"),
            ]
        )
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = client
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "conversation"),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq Conversation", CONF_RECOMMENDED: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"
    client.models.list.assert_awaited_once_with(timeout=10.0)

    data_schema = result["data_schema"]
    assert data_schema is not None
    selector = _get_schema_field(data_schema, CONF_CHAT_MODEL)
    assert selector.serialize()["selector"]["select"] == {
        "options": [
            {
                "label": "Production - Llama 3.1 8B (llama-3.1-8b-instant)",
                "value": "llama-3.1-8b-instant",
            },
            {
                "label": (
                    "Preview - Llama 4 Scout 17B 16E "
                    "(meta-llama/llama-4-scout-17b-16e-instruct)"
                ),
                "value": "meta-llama/llama-4-scout-17b-16e-instruct",
            },
            {
                "label": "Available - custom/live-chat-model",
                "value": "custom/live-chat-model",
            },
        ],
        "mode": "dropdown",
        "sort": False,
        "custom_value": False,
        "multiple": False,
    }
    object.__setattr__(entry, "state", ConfigEntryState.NOT_LOADED)


async def test_stt_subentry_advanced_step_uses_whisper_model_dropdown(
    hass: HomeAssistant,
) -> None:
    """Test advanced STT options show the documented Groq Whisper models."""
    client = MagicMock()
    client.models.list = AsyncMock()
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = client
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "stt"),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq STT", CONF_RECOMMENDED: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"
    client.models.list.assert_not_awaited()

    data_schema = result["data_schema"]
    assert data_schema is not None
    selector = _get_schema_field(data_schema, CONF_STT_MODEL)
    assert selector.serialize()["selector"]["select"] == {
        "options": [
            {
                "label": "Whisper Large v3 Turbo (whisper-large-v3-turbo)",
                "value": RECOMMENDED_STT_MODEL,
            },
            {
                "label": "Whisper Large v3 (whisper-large-v3)",
                "value": "whisper-large-v3",
            },
        ],
        "mode": "dropdown",
        "sort": False,
        "custom_value": False,
        "multiple": False,
    }
    object.__setattr__(entry, "state", ConfigEntryState.NOT_LOADED)


async def test_stt_subentry_recommended_step_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test the STT subentry flow can create a recommended subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = MagicMock()
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "stt"),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq STT", CONF_RECOMMENDED: True},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Groq STT"
    assert result["data"] == {CONF_RECOMMENDED: True}
    object.__setattr__(entry, "state", ConfigEntryState.NOT_LOADED)
