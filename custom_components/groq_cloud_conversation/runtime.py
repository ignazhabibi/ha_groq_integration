"""Runtime data for the Groq Cloud Conversation integration."""

from dataclasses import dataclass

from .api import GroqApiClient
from .model_registry import GroqModelRegistry


@dataclass(slots=True)
class GroqCloudRuntimeData:
    """Objects shared by Groq config entry platforms."""

    client: GroqApiClient
    model_registry: GroqModelRegistry
