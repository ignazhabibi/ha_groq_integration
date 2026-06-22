"""Constants for the Groq Cloud Conversation integration."""

import logging
from typing import Final

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN: Final = "groq_cloud_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

GROQ_BASE_URL: Final = "https://api.groq.com/openai/v1"

DEFAULT_AI_TASK_NAME: Final = "Groq Cloud AI Task"
DEFAULT_CONVERSATION_NAME: Final = "Groq Cloud Conversation"
DEFAULT_NAME: Final = "Groq Cloud Conversation"

CONF_CHAT_MODEL: Final = "chat_model"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_RECOMMENDED: Final = "recommended"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"

RECOMMENDED_CHAT_MODEL: Final = "meta-llama/llama-4-scout-17b-16e-instruct"
RECOMMENDED_MAX_TOKENS: Final = 1024
RECOMMENDED_TEMPERATURE: Final = 0.7
RECOMMENDED_TOP_P: Final = 1.0

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
}

RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_RECOMMENDED: True,
}

