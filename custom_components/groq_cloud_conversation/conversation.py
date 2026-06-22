"""Conversation platform for Groq Cloud Conversation."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import GroqCloudConfigEntry
from .const import DOMAIN
from .entity import GroqCloudBaseLLMEntity


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: GroqCloudConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [GroqCloudConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GroqCloudConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,  # type: ignore[name-defined,misc]
    GroqCloudBaseLLMEntity,
):
    """Groq Cloud conversation agent."""

    _attr_supports_streaming = True

    def __init__(
        self,
        entry: GroqCloudConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the Groq Cloud conversation entity."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return the supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register this entity as the conversation agent for the entry."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister this entity as the conversation agent for the entry."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input with Groq Cloud."""
        options = self.subentry.data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
