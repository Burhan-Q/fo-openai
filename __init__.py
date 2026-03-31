"""fo-openai: OpenAI-powered image labeling via LiteLLM for FiftyOne."""

from .operators import CheckOpenAIStatus, OpenAIInference


def register(plugin):
    plugin.register(OpenAIInference)
    plugin.register(CheckOpenAIStatus)
