"""AI task platform for Groq Cloud Conversation."""

import logging
from json import JSONDecodeError
from typing import TYPE_CHECKING

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .entity import GroqCloudBaseLLMEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import GroqCloudConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: GroqCloudConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud AI task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [GroqCloudTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GroqCloudTaskEntity(
    ai_task.AITaskEntity,
    GroqCloudBaseLLMEntity,
):
    """Groq Cloud AI task entity."""

    def __init__(
        self,
        entry: GroqCloudConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the Groq Cloud AI task entity."""
        super().__init__(entry, subentry)
        self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a Groq Cloud generate data task."""
        await self._async_handle_chat_log(
            chat_log,
            structure=task.structure,
            structure_name=task.name,
            max_iterations=1000,
        )

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            message = "Last content in chat log is not an AssistantContent"
            raise HomeAssistantError(
                message,
            )

        text = chat_log.content[-1].content or ""
        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.warning("Failed to parse Groq structured response: %s", err)
            message = "Error with Groq structured response"
            raise HomeAssistantError(message) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
