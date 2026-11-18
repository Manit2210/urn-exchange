"""Provider-agnostic chat-with-tools interface. SDKs are imported lazily inside each provider."""

from __future__ import annotations

from typing import Any

from urn_exchange.providers.base import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCallOut,
    Usage,
    classify_error,
    request_hash,
    retry_after_seconds,
)
from urn_exchange.providers.fake import FakeProvider, idle_policy

__all__ = [
    "Provider", "ProviderError", "ProviderResponse", "ToolCallOut", "Usage", "classify_error", "request_hash",
    "retry_after_seconds", "FakeProvider", "make_provider", "PROVIDER_KINDS",
]

# openai_compatible: OpenRouter (default base_url) or any OpenAI-compatible gateway. The v1 route.
# gemini_api: Gemini Developer API direct. openai_responses: OpenAI direct. anthropic: Anthropic direct.
PROVIDER_KINDS = ("openai_compatible", "openai", "openai_responses", "gemini_api", "anthropic", "fake")


def make_provider(kind: str, model: str, **kw: Any) -> Provider:
    if kind == "openai_compatible":
        from urn_exchange.providers.openai_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(model, **kw)
    if kind == "openai":
        from urn_exchange.providers.openai_provider import OpenAIProvider
        kw.pop("routing", None)
        return OpenAIProvider(model, **kw)
    if kind == "openai_responses":
        from urn_exchange.providers.openai_provider import OpenAIResponsesProvider
        kw.pop("routing", None)
        return OpenAIResponsesProvider(model, **kw)
    if kind == "gemini_api":
        from urn_exchange.providers.gemini_provider import GeminiAPIProvider
        return GeminiAPIProvider(model, **kw)
    if kind == "anthropic":
        from urn_exchange.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model, **kw)
    if kind == "fake":
        for k in ("api_key_env", "base_url", "routing"):
            kw.pop(k, None)
        return FakeProvider(model, policy=idle_policy, **kw)
    raise ValueError(f"unknown provider kind {kind!r}; one of {PROVIDER_KINDS}")
