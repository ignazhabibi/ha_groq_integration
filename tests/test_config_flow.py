"""Tests for the Groq Cloud Conversation config flow."""

from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from homeassistant.config_entries import SOURCE_REAUTH
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation.config_flow import validate_input
from custom_components.groq_cloud_conversation.const import (
    CONF_RECOMMENDED,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
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
    }
    assert {subentry.title for subentry in subentries} == {
        DEFAULT_AI_TASK_NAME,
        DEFAULT_CONVERSATION_NAME,
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
