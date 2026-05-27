"""
Factory for creating the right LLM client from a provider name string.

Supported providers
-------------------
  anthropic   Claude models  (ANTHROPIC_API_KEY)
  gemini      Google Gemini  (GEMINI_API_KEY)

Usage::

    client = create_llm_client(provider="gemini", api_key="...", model="gemini-2.5-flash")
"""

from __future__ import annotations

from rover_agent.llm.base import NavigationLLMClient
from rover_agent.config import CONFIG

# Map provider name → (client class module path, default model)
_PROVIDERS: dict[str, tuple[str, str]] = {
    "anthropic": ("rover_agent.llm.anthropic_client", "AnthropicNavigationClient"),
    "gemini":    ("rover_agent.llm.gemini_client",    "GeminiNavigationClient"),
}

SUPPORTED_PROVIDERS = list(_PROVIDERS.keys())


def create_llm_client(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = CONFIG.LLM_MAX_TOKENS,
    image_width: int = CONFIG.IMAGE_WIDTH,
    image_height: int = CONFIG.IMAGE_HEIGHT,
) -> NavigationLLMClient:
    """
    Instantiate the LLM client for *provider*.

    Args:
        provider:     One of "anthropic" or "gemini".
        api_key:      API key override. If None, the SDK reads from the
                      environment (GEMINI_API_KEY or ANTHROPIC_API_KEY).
        model:        Model name override; uses the provider's default if None.
        max_tokens:   Maximum tokens in the LLM response.
        image_width:  Camera image width (used to build the system prompt).
        image_height: Camera image height.

    Raises:
        ValueError: Unknown provider name.
        ImportError: Provider SDK not installed.
    """
    provider = provider.lower()
    if provider not in _PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider {provider!r}. Supported: {supported}")

    module_path, class_name = _PROVIDERS[provider]

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    kwargs: dict = {
        "max_tokens": max_tokens,
        "image_width": image_width,
        "image_height": image_height,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if model is not None:
        kwargs["model"] = model

    return cls(**kwargs)
