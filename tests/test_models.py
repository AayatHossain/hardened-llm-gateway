"""Tests for the cached model directory, with a fake clock and fake providers."""

from collections.abc import AsyncGenerator

import pytest

from hardened_llm_gateway.models import ModelDirectory
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
)

pytestmark = pytest.mark.anyio


class ListingProvider:
    def __init__(self, name: str, ids: list[str]) -> None:
        self._name = name
        self.ids = ids
        self.fail = False
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def list_models(self) -> list[str]:
        self.calls += 1
        if self.fail:
            raise ProviderError(self._name, "HTTP 503", status_code=503, retryable=True)
        return list(self.ids)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        raise NotImplementedError
        yield  # makes this an async generator


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_lists_every_provider_with_prefix() -> None:
    providers: dict[str, Provider] = {
        "groq": ListingProvider("groq", ["openai/gpt-oss-20b"]),
        "gemini": ListingProvider("gemini", ["gemini-2.5-flash", "gemini-2.5-pro"]),
    }
    ids = await ModelDirectory().model_ids(providers)
    assert ids == [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
        "groq/openai/gpt-oss-20b",
    ]


async def test_lists_are_cached_until_they_expire() -> None:
    clock = FakeClock()
    groq = ListingProvider("groq", ["m1"])
    directory = ModelDirectory(ttl_seconds=300, clock=clock)

    await directory.model_ids({"groq": groq})
    clock.now = 299
    await directory.model_ids({"groq": groq})
    assert groq.calls == 1

    groq.ids = ["m1", "m2"]
    clock.now = 300
    assert await directory.model_ids({"groq": groq}) == ["groq/m1", "groq/m2"]
    assert groq.calls == 2


async def test_failing_provider_uses_its_last_list_or_is_left_out() -> None:
    clock = FakeClock()
    groq = ListingProvider("groq", ["m1"])
    gemini = ListingProvider("gemini", ["g1"])
    gemini.fail = True
    directory = ModelDirectory(ttl_seconds=300, clock=clock)

    assert await directory.model_ids({"groq": groq, "gemini": gemini}) == ["groq/m1"]

    groq.fail = True
    clock.now = 1000  # cache expired, groq now failing: its last list is still served
    assert await directory.model_ids({"groq": groq}) == ["groq/m1"]
