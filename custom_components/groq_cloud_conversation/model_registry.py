"""Model capability registry for Groq Cloud."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .const import (
    GROQ_PREVIEW_CHAT_MODELS,
    GROQ_PRODUCTION_CHAT_MODELS,
    GROQ_STRUCTURED_OUTPUT_MODEL_IDS,
    GROQ_STT_MODELS,
    GROQ_TTS_MODELS,
    GROQ_VISION_MODEL_IDS,
)


class GroqCapability(StrEnum):
    """Capabilities used to route Groq models to Home Assistant surfaces."""

    TEXT = "text"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUTS = "structured_outputs"
    SPEECH_TO_TEXT = "speech_to_text"
    VISION = "vision"
    TEXT_TO_SPEECH = "text_to_speech"


@dataclass(frozen=True, slots=True)
class GroqModelInfo:
    """Groq model metadata relevant to this integration."""

    id: str
    capabilities: frozenset[GroqCapability] = frozenset()

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> GroqModelInfo:
        """Create model metadata from a Groq `/models` item."""
        model_id = str(data["id"])
        return cls(model_id, infer_capabilities(model_id, data))

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable model metadata."""
        return {
            "id": self.id,
            "capabilities": sorted(str(capability) for capability in self.capabilities),
        }


BUILT_IN_MODELS: dict[str, GroqModelInfo] = {
    **{
        model_id: GroqModelInfo(
            model_id,
            frozenset({GroqCapability.TEXT, GroqCapability.TOOL_CALLING}),
        )
        for model_id in GROQ_PRODUCTION_CHAT_MODELS
        if model_id not in {"groq/compound", "groq/compound-mini"}
    },
    **{
        model_id: GroqModelInfo(
            model_id,
            frozenset({GroqCapability.TEXT, GroqCapability.TOOL_CALLING}),
        )
        for model_id in GROQ_PREVIEW_CHAT_MODELS
    },
    **{
        model_id: GroqModelInfo(
            model_id,
            frozenset({GroqCapability.SPEECH_TO_TEXT}),
        )
        for model_id in GROQ_STT_MODELS
    },
    **{
        model_id: GroqModelInfo(
            model_id,
            frozenset({GroqCapability.TEXT_TO_SPEECH}),
        )
        for model_id in GROQ_TTS_MODELS
    },
    "groq/compound": GroqModelInfo(
        "groq/compound",
        frozenset({GroqCapability.TEXT}),
    ),
    "groq/compound-mini": GroqModelInfo(
        "groq/compound-mini",
        frozenset({GroqCapability.TEXT}),
    ),
}

for _model_id in GROQ_STRUCTURED_OUTPUT_MODEL_IDS:
    _model = BUILT_IN_MODELS[_model_id]
    BUILT_IN_MODELS[_model_id] = GroqModelInfo(
        _model.id,
        _model.capabilities | {GroqCapability.STRUCTURED_OUTPUTS},
    )

for _model_id in GROQ_VISION_MODEL_IDS:
    _model = BUILT_IN_MODELS[_model_id]
    BUILT_IN_MODELS[_model_id] = GroqModelInfo(
        _model.id,
        _model.capabilities | {GroqCapability.VISION},
    )


class GroqModelRegistry:
    """Registry of known and discovered Groq model capabilities."""

    def __init__(self, models: list[GroqModelInfo] | None = None) -> None:
        """Initialize the registry with built-ins and optional live models."""
        self._models = dict(BUILT_IN_MODELS)
        if models:
            self.update(models)

    def update(self, models: list[GroqModelInfo]) -> None:
        """Merge live model metadata into the registry."""
        for model in models:
            built_in = self._models.get(model.id)
            capabilities = model.capabilities
            if built_in:
                capabilities = built_in.capabilities | capabilities
            self._models[model.id] = GroqModelInfo(model.id, capabilities)

    def model_ids(self) -> list[str]:
        """Return all known model IDs."""
        return sorted(self._models)

    def models(self) -> list[GroqModelInfo]:
        """Return all known model metadata."""
        return sorted(self._models.values(), key=lambda model: model.id)

    def model_ids_for_capability(self, capability: GroqCapability) -> list[str]:
        """Return model IDs known to support a capability."""
        return sorted(
            model.id
            for model in self._models.values()
            if capability in model.capabilities
        )

    def supports(self, model_id: str, capability: GroqCapability) -> bool:
        """Return whether a model is known to support a capability."""
        model = self._models.get(model_id)
        return model is not None and capability in model.capabilities


def infer_capabilities(
    model_id: str,
    metadata: dict[str, Any] | None = None,
) -> frozenset[GroqCapability]:
    """Infer capabilities from built-ins, model IDs, and API metadata."""
    capabilities = set(
        BUILT_IN_MODELS.get(model_id, GroqModelInfo(model_id)).capabilities
    )
    normalized = model_id.lower()

    if normalized.startswith("whisper"):
        capabilities.add(GroqCapability.SPEECH_TO_TEXT)
    elif normalized.startswith("canopylabs/orpheus"):
        capabilities.add(GroqCapability.TEXT_TO_SPEECH)
    elif "vision" in normalized or "llama-4" in normalized or "qwen3.6" in normalized:
        capabilities.update(
            {
                GroqCapability.TEXT,
                GroqCapability.TOOL_CALLING,
                GroqCapability.VISION,
            }
        )
    elif not capabilities:
        capabilities.add(GroqCapability.TEXT)

    if model_id in GROQ_STRUCTURED_OUTPUT_MODEL_IDS:
        capabilities.add(GroqCapability.STRUCTURED_OUTPUTS)

    for token in _metadata_tokens(metadata or {}):
        if token in {"tool_calling", "tools", "function_calling"}:
            capabilities.add(GroqCapability.TOOL_CALLING)
        elif token in {"structured_outputs", "structured_output", "json_schema"}:
            capabilities.add(GroqCapability.STRUCTURED_OUTPUTS)
        elif token in {"vision", "image", "image_input", "multimodal"}:
            capabilities.add(GroqCapability.VISION)
        elif token in {"speech_to_text", "transcription", "audio_transcription"}:
            capabilities.add(GroqCapability.SPEECH_TO_TEXT)
        elif token in {"text_to_speech", "speech", "tts"}:
            capabilities.add(GroqCapability.TEXT_TO_SPEECH)

    return frozenset(capabilities)


def _metadata_tokens(metadata: dict[str, Any]) -> set[str]:
    """Return normalized string tokens from shallow model metadata."""
    tokens: set[str] = set()
    for key, value in metadata.items():
        tokens.add(_normalize_token(str(key)))
        if isinstance(value, str):
            tokens.add(_normalize_token(value))
        elif isinstance(value, list | tuple | set):
            tokens.update(_normalize_token(str(item)) for item in value)
        elif isinstance(value, dict):
            tokens.update(_metadata_tokens(value))
    return {token for token in tokens if token}


def _normalize_token(value: str) -> str:
    """Normalize model metadata tokens for capability matching."""
    return (
        value.lower().replace("-", "_").replace("/", "_").replace(".", "_").strip("_")
    )
