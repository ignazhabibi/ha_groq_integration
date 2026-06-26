"""Config flow for the Groq Cloud Conversation integration."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME, CONF_PROMPT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from . import GroqCloudConfigEntry
from .api import (
    GroqApiClient,
    GroqApiError,
    GroqAuthenticationError,
    GroqConnectionError,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_RECOMMENDED,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    CONF_VISION_MODEL,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_STT_PROMPT,
    DEFAULT_TTS_NAME,
    DOMAIN,
    GROQ_PREVIEW_CHAT_MODELS,
    GROQ_PRODUCTION_CHAT_MODELS,
    GROQ_STRUCTURED_OUTPUT_MODEL_IDS,
    GROQ_STT_MODELS,
    GROQ_TTS_MODELS,
    GROQ_TTS_VOICES,
    GROQ_UNSUPPORTED_CHAT_MODEL_IDS,
    GROQ_VISION_MODEL_IDS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STRUCTURED_OUTPUT_MODEL,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_OPTIONS,
    RECOMMENDED_TTS_VOICE,
    RECOMMENDED_VISION_MODEL,
)
from .model_registry import GroqCapability

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})
MODEL_FETCH_TIMEOUT = 10.0


async def validate_input(hass: HomeAssistant, data: dict[str, str]) -> None:
    """Validate that the provided Groq API key can reach the API."""
    client = GroqApiClient(
        api_key=data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )
    await client.async_list_models(request_timeout=MODEL_FETCH_TIMEOUT)


def _known_chat_model_ids() -> tuple[str, ...]:
    """Return the fallback Chat Completions model IDs in display order."""
    return tuple(GROQ_PRODUCTION_CHAT_MODELS) + tuple(GROQ_PREVIEW_CHAT_MODELS)


def _model_sort_key(model_id: str) -> tuple[int, int, str]:
    """Return a stable sort key that keeps Production before Preview models."""
    known_model_ids = _known_chat_model_ids()
    if model_id in GROQ_PRODUCTION_CHAT_MODELS:
        return (0, known_model_ids.index(model_id), model_id)
    if model_id in GROQ_PREVIEW_CHAT_MODELS:
        return (1, known_model_ids.index(model_id), model_id)
    if model_id in GROQ_UNSUPPORTED_CHAT_MODEL_IDS:
        return (3, len(known_model_ids), model_id)
    return (2, len(known_model_ids), model_id)


def _model_label(model_id: str) -> str:
    """Return the user-facing label for a Groq model selector option."""
    if model_name := GROQ_PRODUCTION_CHAT_MODELS.get(model_id):
        return f"Production - {model_name} ({model_id})"
    if model_name := GROQ_PREVIEW_CHAT_MODELS.get(model_id):
        return f"Preview - {model_name} ({model_id})"
    if model_id in GROQ_UNSUPPORTED_CHAT_MODEL_IDS:
        return f"Unsupported for chat - {model_id}"
    return f"Available - {model_id}"


def _model_selector_options(
    model_ids: list[str],
    selected_model: str | None = None,
    allowed_model_ids: frozenset[str] | None = None,
) -> list[SelectOptionDict]:
    """Build selector options for Groq Chat Completions models."""
    available_model_ids = {
        model_id
        for model_id in model_ids
        if model_id and model_id not in GROQ_UNSUPPORTED_CHAT_MODEL_IDS
        if allowed_model_ids is None or model_id in allowed_model_ids
    }
    if selected_model and (
        allowed_model_ids is None or selected_model in allowed_model_ids
    ):
        available_model_ids.add(selected_model)

    return [
        SelectOptionDict(label=_model_label(model_id), value=model_id)
        for model_id in sorted(available_model_ids, key=_model_sort_key)
    ]


def _stt_model_label(model_id: str) -> str:
    """Return the user-facing label for a Groq STT model selector option."""
    if model_name := GROQ_STT_MODELS.get(model_id):
        return f"{model_name} ({model_id})"
    return model_id


def _stt_model_selector_options(
    selected_model: str | None = None,
) -> list[SelectOptionDict]:
    """Build selector options for Groq speech-to-text models."""
    model_ids = set(GROQ_STT_MODELS)
    if selected_model:
        model_ids.add(selected_model)

    return [
        SelectOptionDict(label=_stt_model_label(model_id), value=model_id)
        for model_id in sorted(
            model_ids,
            key=lambda model_id: (
                model_id != RECOMMENDED_STT_MODEL,
                model_id,
            ),
        )
    ]


def _tts_model_label(model_id: str) -> str:
    """Return the user-facing label for a Groq TTS model selector option."""
    if model_name := GROQ_TTS_MODELS.get(model_id):
        return f"{model_name} ({model_id})"
    return model_id


def _tts_model_selector_options(
    selected_model: str | None = None,
) -> list[SelectOptionDict]:
    """Build selector options for Groq text-to-speech models."""
    model_ids = set(GROQ_TTS_MODELS)
    if selected_model:
        model_ids.add(selected_model)

    return [
        SelectOptionDict(label=_tts_model_label(model_id), value=model_id)
        for model_id in sorted(
            model_ids,
            key=lambda model_id: (
                model_id != RECOMMENDED_TTS_MODEL,
                model_id,
            ),
        )
    ]


def _tts_voice_label(model_id: str, voice_id: str) -> str:
    """Return the user-facing label for a Groq TTS voice selector option."""
    model_label = GROQ_TTS_MODELS.get(model_id, model_id)
    voice_label = GROQ_TTS_VOICES.get(model_id, {}).get(voice_id, voice_id)
    return f"{model_label} - {voice_label} ({voice_id})"


def _tts_voice_selector_options(
    selected_voice: str | None = None,
) -> list[SelectOptionDict]:
    """Build selector options for Groq text-to-speech voices."""
    voice_options = [
        SelectOptionDict(
            label=_tts_voice_label(model_id, voice_id),
            value=voice_id,
        )
        for model_id, voices in GROQ_TTS_VOICES.items()
        for voice_id in voices
    ]

    if selected_voice and selected_voice not in {
        option["value"] for option in voice_options
    }:
        voice_options.append(
            SelectOptionDict(label=selected_voice, value=selected_voice)
        )

    return sorted(
        voice_options,
        key=lambda option: (
            option["value"] != RECOMMENDED_TTS_VOICE,
            option["label"],
        ),
    )


def _tts_voice_model(voice_id: str) -> str | None:
    """Return the Groq TTS model that supports a voice."""
    for model_id, voices in GROQ_TTS_VOICES.items():
        if voice_id in voices:
            return model_id
    return None


def _default_tts_voice(model_id: str) -> str:
    """Return the default voice for a Groq TTS model."""
    if model_id == RECOMMENDED_TTS_MODEL:
        return RECOMMENDED_TTS_VOICE
    voices = GROQ_TTS_VOICES.get(model_id)
    if voices:
        return next(iter(voices))
    return RECOMMENDED_TTS_VOICE


async def _async_get_model_selector_options(
    entry: GroqCloudConfigEntry,
    selected_model: str | None = None,
    allowed_model_ids: frozenset[str] | None = None,
) -> list[SelectOptionDict]:
    """Fetch the live Groq model list and return dropdown options."""
    try:
        models = await entry.runtime_data.client.async_list_models(
            request_timeout=MODEL_FETCH_TIMEOUT
        )
    except GroqApiError:
        _LOGGER.warning(
            "Could not fetch Groq models; using documented fallback models",
            exc_info=True,
        )
        return _model_selector_options(
            list(_known_chat_model_ids()),
            selected_model,
            allowed_model_ids,
        )

    model_options = _model_selector_options(
        [model.id for model in models],
        selected_model,
        allowed_model_ids,
    )
    entry.runtime_data.model_registry.update(models)
    if model_options:
        return model_options

    return _model_selector_options(
        list(_known_chat_model_ids()),
        selected_model,
        allowed_model_ids,
    )


def _default_chat_model_for_subentry(subentry_type: str) -> str:
    """Return the default Chat Completions model for a subentry type."""
    if subentry_type == "ai_task_data":
        return RECOMMENDED_STRUCTURED_OUTPUT_MODEL
    return RECOMMENDED_CHAT_MODEL


class GroqCloudConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Groq Cloud Conversation."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self,
        user_input: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial config flow step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._async_abort_entries_match(user_input)
            try:
                await validate_input(self.hass, user_input)
            except GroqConnectionError:
                errors["base"] = "cannot_connect"
            except GroqAuthenticationError:
                errors["base"] = "invalid_auth"
            except GroqApiError:
                _LOGGER.exception("Unexpected Groq API error")
                errors["base"] = "unknown"
            else:
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(),
                        data_updates=user_input,
                    )
                return self.async_create_entry(
                    title="Groq Cloud",
                    data=user_input,
                    subentries=[
                        {
                            "data": RECOMMENDED_CONVERSATION_OPTIONS,
                            "subentry_type": "conversation",
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                        {
                            "data": RECOMMENDED_AI_TASK_OPTIONS,
                            "subentry_type": "ai_task_data",
                            "title": DEFAULT_AI_TASK_NAME,
                            "unique_id": None,
                        },
                        {
                            "data": RECOMMENDED_STT_OPTIONS,
                            "subentry_type": "stt",
                            "title": DEFAULT_STT_NAME,
                            "unique_id": None,
                        },
                        {
                            "data": RECOMMENDED_TTS_OPTIONS,
                            "subentry_type": "tts",
                            "title": DEFAULT_TTS_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA,
                user_input,
            ),
            errors=errors,
            description_placeholders={
                "instructions_url": "https://console.groq.com/keys",
            },
        )

    async def async_step_reauth(
        self,
        _entry_data: Mapping[str, Any],
    ) -> ConfigFlowResult:
        """Perform reauth after an authentication failure."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self,
        user_input: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Ask for an updated Groq API key."""
        if user_input is None:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        return await self.async_step_user(user_input)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls,
        _config_entry: ConfigEntry,
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentries supported by this integration."""
        return {
            "ai_task_data": GroqCloudSubentryFlowHandler,
            "conversation": GroqCloudSubentryFlowHandler,
            "stt": GroqCloudSubentryFlowHandler,
            "tts": GroqCloudSubentryFlowHandler,
        }


class GroqCloudSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing Groq Cloud subentries."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return whether this flow creates a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self,
        _user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Add a new subentry."""
        if self._subentry_type == "ai_task_data":
            self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
        elif self._subentry_type == "stt":
            self.options = RECOMMENDED_STT_OPTIONS.copy()
        elif self._subentry_type == "tts":
            self.options = RECOMMENDED_TTS_OPTIONS.copy()
        else:
            self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self,
        _user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Reconfigure an existing subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Manage basic subentry options."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options
        step_schema: VolDictType = {}

        if self._is_new:
            default_name = {
                "ai_task_data": DEFAULT_AI_TASK_NAME,
                "conversation": DEFAULT_CONVERSATION_NAME,
                "stt": DEFAULT_STT_NAME,
                "tts": DEFAULT_TTS_NAME,
            }[self._subentry_type]
            step_schema[vol.Required(CONF_NAME, default=default_name)] = str

        if self._subentry_type == "conversation":
            hass_apis: list[SelectOptionDict] = [
                SelectOptionDict(label=api.name, value=api.id)
                for api in llm.async_get_apis(self.hass)
            ]
            if suggested_llm_apis := options.get(CONF_LLM_HASS_API):
                if isinstance(suggested_llm_apis, str):
                    suggested_llm_apis = [suggested_llm_apis]
                valid_apis = {api.id for api in llm.async_get_apis(self.hass)}
                options[CONF_LLM_HASS_API] = [
                    api for api in suggested_llm_apis if api in valid_apis
                ]
            step_schema.update(
                {
                    vol.Optional(
                        CONF_PROMPT,
                        description={
                            "suggested_value": options.get(
                                CONF_PROMPT,
                                llm.DEFAULT_INSTRUCTIONS_PROMPT,
                            ),
                        },
                    ): TemplateSelector(),
                    vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True),
                    ),
                }
            )

        step_schema[
            vol.Required(CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True))
        ] = bool

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            if user_input[CONF_RECOMMENDED]:
                if self._is_new:
                    return self.async_create_entry(
                        title=user_input.pop(CONF_NAME),
                        data=user_input,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )
            options.update(user_input)
            if CONF_LLM_HASS_API in options and CONF_LLM_HASS_API not in user_input:
                options.pop(CONF_LLM_HASS_API)
            return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema),
                options,
            ),
        )

    async def async_step_advanced(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Manage advanced Groq model settings."""
        if self._subentry_type == "stt":
            return await self.async_step_stt_advanced(user_input)
        if self._subentry_type == "tts":
            return await self.async_step_tts_advanced(user_input)

        options = self.options
        default_chat_model = _default_chat_model_for_subentry(self._subentry_type)
        selected_chat_model = options.get(CONF_CHAT_MODEL, default_chat_model)
        selected_vision_model = options.get(
            CONF_VISION_MODEL,
            RECOMMENDED_VISION_MODEL,
        )
        allowed_model_ids = (
            GROQ_STRUCTURED_OUTPUT_MODEL_IDS
            if self._subentry_type == "ai_task_data"
            else None
        )
        if (
            allowed_model_ids is not None
            and selected_chat_model not in allowed_model_ids
        ):
            selected_chat_model = default_chat_model
        if (
            self._subentry_type == "ai_task_data"
            and selected_vision_model not in GROQ_VISION_MODEL_IDS
        ):
            selected_vision_model = RECOMMENDED_VISION_MODEL

        model_options = await _async_get_model_selector_options(
            cast("GroqCloudConfigEntry", self._get_entry()),
            selected_chat_model,
            allowed_model_ids,
        )
        step_schema: VolDictType = {
            vol.Optional(
                CONF_CHAT_MODEL,
                default=selected_chat_model,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_MAX_TOKENS, default=RECOMMENDED_MAX_TOKENS): int,
            vol.Optional(CONF_TEMPERATURE, default=RECOMMENDED_TEMPERATURE): (
                NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))
            ),
            vol.Optional(CONF_TOP_P, default=RECOMMENDED_TOP_P): NumberSelector(
                NumberSelectorConfig(min=0, max=1, step=0.05),
            ),
        }
        if self._subentry_type == "ai_task_data":
            vision_model_options = _model_selector_options(
                cast(
                    "GroqCloudConfigEntry", self._get_entry()
                ).runtime_data.model_registry.model_ids_for_capability(
                    GroqCapability.VISION
                ),
                selected_vision_model,
                GROQ_VISION_MODEL_IDS,
            )
            if not vision_model_options:
                vision_model_options = _model_selector_options(
                    list(_known_chat_model_ids()),
                    selected_vision_model,
                    GROQ_VISION_MODEL_IDS,
                )
            step_schema[
                vol.Optional(
                    CONF_VISION_MODEL,
                    default=selected_vision_model,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=vision_model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        if user_input is not None:
            options.update(user_input)
            if self._is_new:
                return self.async_create_entry(
                    title=options.pop(CONF_NAME),
                    data=options,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema),
                options,
            ),
        )

    async def async_step_stt_advanced(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Manage advanced Groq speech-to-text settings."""
        options = self.options
        step_schema: VolDictType = {
            vol.Optional(
                CONF_STT_MODEL,
                default=options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=_stt_model_selector_options(
                        options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
                    ),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": options.get(CONF_PROMPT, DEFAULT_STT_PROMPT),
                },
            ): TemplateSelector(),
        }

        if user_input is not None:
            options.update(user_input)
            if self._is_new:
                return self.async_create_entry(
                    title=options.pop(CONF_NAME),
                    data=options,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema),
                options,
            ),
        )

    async def async_step_tts_advanced(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Manage advanced Groq text-to-speech settings."""
        options = self.options
        selected_model = options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
        selected_voice = options.get(
            CONF_TTS_VOICE,
            _default_tts_voice(selected_model),
        )
        errors: dict[str, str] = {}

        if user_input is not None:
            selected_model = user_input.get(CONF_TTS_MODEL, selected_model)
            selected_voice = user_input.get(
                CONF_TTS_VOICE,
                _default_tts_voice(selected_model),
            )
            if _tts_voice_model(selected_voice) != selected_model:
                errors["base"] = "unsupported_voice"
            else:
                options.update(user_input)
                options[CONF_TTS_MODEL] = selected_model
                options[CONF_TTS_VOICE] = selected_voice
                if self._is_new:
                    return self.async_create_entry(
                        title=options.pop(CONF_NAME),
                        data=options,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=options,
                )

        step_schema: VolDictType = {
            vol.Optional(
                CONF_TTS_MODEL,
                default=selected_model,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=_tts_model_selector_options(selected_model),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_VOICE,
                default=selected_voice,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=_tts_voice_selector_options(selected_voice),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema),
                options,
            ),
            errors=errors,
        )
