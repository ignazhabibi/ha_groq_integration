"""Shared pytest fixtures for the Groq Cloud Conversation integration."""

from typing import Any, cast

import pytest
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.core import HomeAssistant

pytest_plugins = "pytest_homeassistant_custom_component"


class FakeExposedEntities:
    """Minimal exposed entity registry for conversation setup tests."""

    def async_should_expose(self, _assistant: str, _entity_id: str) -> bool:
        """Return whether an entity should be exposed to an assistant."""
        return False


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> None:
    """Enable Home Assistant to load custom integrations from the workspace."""
    _ = enable_custom_integrations


@pytest.fixture(autouse=True)
def mock_exposed_entities(hass: HomeAssistant) -> None:
    """Provide exposed entity data expected by the conversation dependency."""
    hass_data = cast("dict[Any, Any]", hass.data)
    if DATA_EXPOSED_ENTITIES not in hass_data:
        hass_data[DATA_EXPOSED_ENTITIES] = FakeExposedEntities()
