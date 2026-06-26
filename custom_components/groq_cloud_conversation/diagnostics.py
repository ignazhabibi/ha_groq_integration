"""Diagnostics support for the Groq Cloud Conversation integration."""

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntry

from . import GroqCloudConfigEntry
from .const import DOMAIN

TO_REDACT = {
    CONF_API_KEY,
    CONF_PROMPT,
    "api_key",
    "extra_system_prompt",
    "prompt",
    "system_prompt",
}


async def async_get_config_entry_diagnostics(
    _hass: HomeAssistant,
    entry: GroqCloudConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a Groq Cloud config entry."""
    return {
        "entry": async_redact_data(entry.as_dict(), TO_REDACT),
        "subentries": [
            _subentry_diagnostics(subentry) for subentry in entry.subentries.values()
        ],
        "runtime": _runtime_diagnostics(entry),
    }


async def async_get_device_diagnostics(
    hass: HomeAssistant,
    entry: GroqCloudConfigEntry,
    device: DeviceEntry,
) -> dict[str, Any]:
    """Return diagnostics for a Groq Cloud service device."""
    subentry = _device_subentry(entry, device)
    return {
        **await async_get_config_entry_diagnostics(hass, entry),
        "device": device.dict_repr,
        "device_subentry": _subentry_diagnostics(subentry) if subentry else None,
    }


def _subentry_diagnostics(subentry: ConfigSubentry) -> dict[str, Any]:
    """Return redacted diagnostics for a config subentry."""
    return async_redact_data(
        {
            "data": dict(subentry.data),
            "subentry_id": subentry.subentry_id,
            "subentry_type": subentry.subentry_type,
            "title": subentry.title,
            "unique_id": subentry.unique_id,
        },
        TO_REDACT,
    )


def _runtime_diagnostics(entry: GroqCloudConfigEntry) -> dict[str, Any]:
    """Return non-secret runtime diagnostics."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        return {"loaded": False}

    models = [model.as_dict() for model in runtime_data.model_registry.models()]
    return {
        "loaded": True,
        "model_count": len(models),
        "models": models,
    }


def _device_subentry(
    entry: GroqCloudConfigEntry,
    device: DeviceEntry,
) -> ConfigSubentry | None:
    """Return the Groq subentry represented by a diagnostics device."""
    subentry_ids = {
        identifier for domain, identifier in device.identifiers if domain == DOMAIN
    }
    for subentry_id in subentry_ids:
        if subentry := entry.subentries.get(subentry_id):
            return subentry
    return None
