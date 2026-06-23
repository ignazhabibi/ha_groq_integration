"""Constants for the Groq Cloud Conversation integration."""

import logging
from collections.abc import Mapping
from typing import Final

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN: Final = "groq_cloud_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

GROQ_BASE_URL: Final = "https://api.groq.com/openai/v1"

DEFAULT_AI_TASK_NAME: Final = "Groq Cloud AI Task"
DEFAULT_CONVERSATION_NAME: Final = "Groq Cloud Conversation"
DEFAULT_NAME: Final = "Groq Cloud Conversation"
DEFAULT_STT_NAME: Final = "Groq Cloud STT"

CONF_CHAT_MODEL: Final = "chat_model"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_RECOMMENDED: Final = "recommended"
CONF_STT_MODEL: Final = "stt_model"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"

RECOMMENDED_CHAT_MODEL: Final = "llama-3.3-70b-versatile"
RECOMMENDED_MAX_TOKENS: Final = 1024
RECOMMENDED_STRUCTURED_OUTPUT_MODEL: Final = "openai/gpt-oss-20b"
RECOMMENDED_STT_MODEL: Final = "whisper-large-v3-turbo"
RECOMMENDED_TEMPERATURE: Final = 0.7
RECOMMENDED_TOP_P: Final = 1.0

GROQ_PRODUCTION_CHAT_MODELS: Final[Mapping[str, str]] = {
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
    "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B",
    "groq/compound": "Groq Compound",
    "groq/compound-mini": "Groq Compound Mini",
}
GROQ_PREVIEW_CHAT_MODELS: Final[Mapping[str, str]] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B 16E",
    "qwen/qwen3-32b": "Qwen3-32B",
    "qwen/qwen3.6-27b": "Qwen/Qwen3.6-27B",
}
GROQ_UNSUPPORTED_CHAT_MODEL_IDS: Final[frozenset[str]] = frozenset(
    {
        "canopylabs/orpheus-arabic-saudi",
        "canopylabs/orpheus-v1-english",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-safeguard-20b",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    }
)
GROQ_STRUCTURED_OUTPUT_MODEL_IDS: Final[frozenset[str]] = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
)
GROQ_STT_MODELS: Final[Mapping[str, str]] = {
    "whisper-large-v3-turbo": "Whisper Large v3 Turbo",
    "whisper-large-v3": "Whisper Large v3",
}
DEFAULT_STT_PROMPT: Final = (
    "The following conversation is a smart home user talking to Home Assistant."
)

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_CHAT_MODEL: RECOMMENDED_STRUCTURED_OUTPUT_MODEL,
    CONF_RECOMMENDED: True,
}

RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_RECOMMENDED: True,
}

RECOMMENDED_STT_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
}
