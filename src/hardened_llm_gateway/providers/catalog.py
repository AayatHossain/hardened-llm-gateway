"""Known providers and how to build them from settings.

A provider is enabled only when its API key is set.
"""

from dataclasses import dataclass

import httpx
from pydantic import SecretStr

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.providers.base import Provider
from hardened_llm_gateway.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    key_setting: str  # the Settings field holding this provider's API key


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec("groq", "https://api.groq.com/openai/v1", "groq_api_key"),
    ProviderSpec(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key"
    ),
    ProviderSpec("nvidia", "https://integrate.api.nvidia.com/v1", "nvidia_api_key"),
    ProviderSpec("openrouter", "https://openrouter.ai/api/v1", "openrouter_api_key"),
    ProviderSpec("cohere", "https://api.cohere.ai/compatibility/v1", "cohere_api_key"),
    ProviderSpec("zai", "https://api.z.ai/api/paas/v4", "zai_api_key"),
)


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[str, Provider]:
    """Create one adapter per provider whose key is set, all sharing one HTTP client."""
    providers: dict[str, Provider] = {}
    for spec in PROVIDER_SPECS:
        key = getattr(settings, spec.key_setting)
        if isinstance(key, SecretStr) and key.get_secret_value():
            providers[spec.name] = OpenAICompatibleProvider(
                spec.name,
                spec.base_url,
                key,
                client,
                timeout_seconds=settings.request_timeout_seconds,
            )
    return providers
