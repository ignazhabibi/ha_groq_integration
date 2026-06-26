"""Tests for the Groq Cloud Conversation config flow."""

from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import SelectSelector
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation.api import (
    GroqApiError,
    GroqAuthenticationError,
    GroqConnectionError,
)
from custom_components.groq_cloud_conversation.config_flow import (
    _model_selector_options,
    validate_input,
)
from custom_components.groq_cloud_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_RECOMMENDED,
    CONF_STT_MODEL,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    CONF_VISION_MODEL,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_STRUCTURED_OUTPUT_MODEL,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
    RECOMMENDED_VISION_MODEL,
)
from custom_components.groq_cloud_conversation.model_registry import (
    GroqModelInfo,
    GroqModelRegistry,
)
from custom_components.groq_cloud_conversation.runtime import GroqCloudRuntimeData


def _model(model_id: str) -> GroqModelInfo:
    """Return a Groq model object for flow tests."""
    return GroqModelInfo.from_api({"id": model_id})


def _runtime(client: MagicMock) -> GroqCloudRuntimeData:
    """Return runtime data for flow tests."""
    return GroqCloudRuntimeData(
        client=client,
        model_registry=GroqModelRegistry(),
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
        "tts",
    }
    assert {subentry.title for subentry in subentries} == {
        DEFAULT_AI_TASK_NAME,
        DEFAULT_CONVERSATION_NAME,
        DEFAULT_STT_NAME,
        DEFAULT_TTS_NAME,
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
        (lambda: GroqAuthenticationError("invalid key"), "invalid_auth"),
        (lambda: GroqConnectionError("cannot connect"), "cannot_connect"),
        (lambda: GroqApiError("boom"), "unknown"),
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
    """Test validation creates a Groq API client."""
    client = AsyncMock()
    client.async_list_models = AsyncMock(return_value=[])

    with patch(
        "custom_components.groq_cloud_conversation.config_flow.GroqApiClient",
        return_value=client,
    ) as mock_client:
        await validate_input(hass, {CONF_API_KEY: "groq-key"})

    assert mock_client.call_args.kwargs["api_key"] == "groq-key"
    client.async_list_models.assert_awaited_once_with(request_timeout=10.0)


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
    client.async_list_models = AsyncMock(
        return_value=[
            _model("custom/live-chat-model"),
            _model("llama-3.1-8b-instant"),
            _model("meta-llama/llama-4-scout-17b-16e-instruct"),
            _model("whisper-large-v3"),
        ]
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = _runtime(client)
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
    client.async_list_models.assert_awaited_once_with(request_timeout=10.0)

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
                "label": "Production - Llama 3.3 70B (llama-3.3-70b-versatile)",
                "value": "llama-3.3-70b-versatile",
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


async def test_ai_task_subentry_advanced_step_filters_structured_output_models(
    hass: HomeAssistant,
) -> None:
    """Test AI task model options only show Structured Outputs models."""
    client = MagicMock()
    client.async_list_models = AsyncMock(
        return_value=[
            _model("custom/live-chat-model"),
            _model("meta-llama/llama-4-scout-17b-16e-instruct"),
            _model("llama-3.3-70b-versatile"),
            _model("openai/gpt-oss-120b"),
            _model("openai/gpt-oss-20b"),
            _model("qwen/qwen3.6-27b"),
        ]
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = _runtime(client)
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "ai_task_data"),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq AI Task", CONF_RECOMMENDED: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"
    data_schema = result["data_schema"]
    assert data_schema is not None
    selector = _get_schema_field(data_schema, CONF_CHAT_MODEL)
    assert selector.serialize()["selector"]["select"] == {
        "options": [
            {
                "label": "Production - OpenAI GPT-OSS 120B (openai/gpt-oss-120b)",
                "value": "openai/gpt-oss-120b",
            },
            {
                "label": "Production - OpenAI GPT-OSS 20B (openai/gpt-oss-20b)",
                "value": RECOMMENDED_STRUCTURED_OUTPUT_MODEL,
            },
        ],
        "mode": "dropdown",
        "sort": False,
        "custom_value": False,
        "multiple": False,
    }
    vision_selector = _get_schema_field(data_schema, CONF_VISION_MODEL)
    assert vision_selector.serialize()["selector"]["select"] == {
        "options": [
            {
                "label": (
                    "Preview - Llama 4 Scout 17B 16E "
                    "(meta-llama/llama-4-scout-17b-16e-instruct)"
                ),
                "value": RECOMMENDED_VISION_MODEL,
            },
            {
                "label": "Preview - Qwen/Qwen3.6-27B (qwen/qwen3.6-27b)",
                "value": "qwen/qwen3.6-27b",
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
    client.async_list_models = AsyncMock()
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = _runtime(client)
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
    client.async_list_models.assert_not_awaited()

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


async def test_tts_subentry_advanced_step_uses_orpheus_dropdowns(
    hass: HomeAssistant,
) -> None:
    """Test advanced TTS options show the documented Groq Orpheus models."""
    client = MagicMock()
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = _runtime(client)
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "tts"),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq TTS", CONF_RECOMMENDED: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"
    client.async_list_models.assert_not_called()

    data_schema = result["data_schema"]
    assert data_schema is not None
    model_selector = _get_schema_field(data_schema, CONF_TTS_MODEL)
    assert model_selector.serialize()["selector"]["select"] == {
        "options": [
            {
                "label": ("Orpheus English (canopylabs/orpheus-v1-english)"),
                "value": RECOMMENDED_TTS_MODEL,
            },
            {
                "label": ("Orpheus Arabic Saudi (canopylabs/orpheus-arabic-saudi)"),
                "value": "canopylabs/orpheus-arabic-saudi",
            },
        ],
        "mode": "dropdown",
        "sort": False,
        "custom_value": False,
        "multiple": False,
    }

    voice_selector = _get_schema_field(data_schema, CONF_TTS_VOICE)
    voice_options = voice_selector.serialize()["selector"]["select"]["options"]
    assert voice_options[0] == {
        "label": "Orpheus English - Troy (troy)",
        "value": RECOMMENDED_TTS_VOICE,
    }
    assert {
        "label": "Orpheus Arabic Saudi - Noura (noura)",
        "value": "noura",
    } in voice_options
    object.__setattr__(entry, "state", ConfigEntryState.NOT_LOADED)


async def test_tts_subentry_advanced_step_rejects_voice_model_mismatch(
    hass: HomeAssistant,
) -> None:
    """Test TTS flow rejects a voice that belongs to another model."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        state=ConfigEntryState.LOADED,
        subentries_data=[],
        title="Groq Cloud",
    )
    entry.runtime_data = _runtime(MagicMock())
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "tts"),
        context={"source": "user"},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Groq TTS", CONF_RECOMMENDED: False},
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_TTS_MODEL: "canopylabs/orpheus-arabic-saudi",
            CONF_TTS_VOICE: "troy",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "unsupported_voice"}
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
    entry.runtime_data = _runtime(MagicMock())
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
