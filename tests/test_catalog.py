import httpx
import pytest
from pydantic import SecretStr

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.providers.catalog import PROVIDER_SPECS, build_providers

pytestmark = pytest.mark.anyio


async def test_only_providers_with_keys_are_built() -> None:
    settings = Settings(groq_api_key=SecretStr("g-key"), gemini_api_key=SecretStr("m-key"))
    async with httpx.AsyncClient() as client:
        providers = build_providers(settings, client)
    assert sorted(providers) == ["gemini", "groq"]
    assert providers["groq"].name == "groq"


async def test_cohere_and_zai_are_built_from_their_keys() -> None:
    settings = Settings(cohere_api_key=SecretStr("c-key"), zai_api_key=SecretStr("z-key"))
    async with httpx.AsyncClient() as client:
        providers = build_providers(settings, client)
    assert sorted(providers) == ["cohere", "zai"]
    assert "api.cohere.ai/compatibility/v1/chat/completions" in repr(providers["cohere"])
    assert "api.z.ai/api/paas/v4/chat/completions" in repr(providers["zai"])


async def test_empty_key_counts_as_missing() -> None:
    settings = Settings(groq_api_key=SecretStr(""))
    async with httpx.AsyncClient() as client:
        assert build_providers(settings, client) == {}


def test_every_spec_points_to_a_real_settings_field() -> None:
    for spec in PROVIDER_SPECS:
        assert spec.key_setting in Settings.model_fields
        assert spec.base_url.startswith("https://")
