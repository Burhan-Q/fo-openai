"""fo-openai: OpenAI-powered image labeling via LiteLLM for FiftyOne."""

from __future__ import annotations

from typing import Any

from .operators import CheckOpenAIStatus, OpenAIInference


def register(plugin: Any) -> None:
    """Register all operators with the FiftyOne plugin system."""
    plugin.register(OpenAIInference)
    plugin.register(CheckOpenAIStatus)
