"""Tests for Groq Cloud Conversation integration setup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq_cloud_conversation import (
    PLATFORMS,
    async_setup_entry,
    async_unload_entry,
    async_update_options,
)
from custom_components.groq_cloud_conversation.api import (
    GroqApiError,
    GroqAuthenticationError,
)
from custom_components.groq_cloud_conversation.const import DOMAIN


def _make_entry() -> MockConfigEntry:
    """Create a Groq config entry for setup tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "groq-key"},
        title="Groq Cloud",
    )


async def test_setup_entry_creates_client_and_forwards_platforms(
    hass: HomeAssistant,
) -> None:
    """Test setup validates the Groq client and forwards platforms."""
    entry = _make_entry()
    entry.add_to_hass(hass)
    client = MagicMock()
    client.async_list_models = AsyncMock(return_value=[])

    with (
        patch(
            "custom_components.groq_cloud_conversation.GroqApiClient",
            return_value=client,
        ) as mock_client,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(return_value=True),
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.client is client
    assert mock_client.call_args.kwargs["api_key"] == "groq-key"
    client.async_list_models.assert_awaited_once_with(request_timeout=10.0)
    forward_setups.assert_awaited_once_with(entry, PLATFORMS)


@pytest.mark.parametrize(
    ("setup_error", "expected_exception"),
    [
        (GroqAuthenticationError("invalid key"), ConfigEntryAuthFailed),
        (GroqApiError("boom"), ConfigEntryNotReady),
    ],
)
async def test_setup_entry_maps_validation_errors(
    hass: HomeAssistant,
    setup_error: GroqApiError,
    expected_exception: type[Exception],
) -> None:
    """Test setup maps Groq validation errors to config entry errors."""
    entry = _make_entry()
    entry.add_to_hass(hass)
    client = MagicMock()
    client.async_list_models = AsyncMock(side_effect=setup_error)

    with (
        patch(
            "custom_components.groq_cloud_conversation.GroqApiClient",
            return_value=client,
        ),
        pytest.raises(expected_exception),
    ):
        await async_setup_entry(hass, entry)


async def test_unload_entry_unloads_platforms(hass: HomeAssistant) -> None:
    """Test unloading delegates to Home Assistant platform unloading."""
    entry = _make_entry()
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        AsyncMock(return_value=True),
    ) as unload_platforms:
        assert await async_unload_entry(hass, entry)

    unload_platforms.assert_awaited_once_with(entry, PLATFORMS)


async def test_options_update_reloads_entry(hass: HomeAssistant) -> None:
    """Test options updates reload the config entry."""
    entry = _make_entry()
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_reload",
        AsyncMock(return_value=True),
    ) as reload_entry:
        await async_update_options(hass, entry)

    reload_entry.assert_awaited_once_with(entry.entry_id)


def test_platforms_include_stt() -> None:
    """Test setup forwards all supported platforms."""
    assert PLATFORMS == (
        Platform.AI_TASK,
        Platform.CONVERSATION,
        Platform.STT,
        Platform.TTS,
    )
