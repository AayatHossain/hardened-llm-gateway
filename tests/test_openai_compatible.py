"""Tests for the OpenAI-compatible adapter, against a fake HTTP server (no network, no keys)."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from hardened_llm_gateway.providers import (
    ChatMessage,
    ChatRequest,
    OpenAICompatibleProvider,
    ProviderError,
)

pytestmark = pytest.mark.anyio

API_KEY = "test-key-do-not-leak"
BASE_URL = "https://api.example.com/openai/v1"

Handler = Callable[[httpx.Request], httpx.Response]


def make_provider(handler: Handler) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("fake", BASE_URL, SecretStr(API_KEY), client)


def make_request(**overrides: Any) -> ChatRequest:
    fields: dict[str, Any] = {
        "model": "some-model",
        "messages": [ChatMessage(role="user", content="hi")],
    }
    fields.update(overrides)
    return ChatRequest(**fields)


def completion_body(content: str | None = "Hello!") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "some-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "queue_time": 0.1},
    }


async def test_successful_reply() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_body())

    reply = await make_provider(handler).chat(make_request(temperature=0.2))

    assert reply.content == "Hello!"
    assert reply.provider == "fake"
    assert reply.model == "some-model"
    assert reply.finish_reason == "stop"
    assert reply.usage.total_tokens == 7
    assert reply.latency_ms >= 0

    sent = seen[0]
    assert str(sent.url) == f"{BASE_URL}/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    body = json.loads(sent.content)
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.2
    assert "max_tokens" not in body  # unset options are not sent


async def test_null_content_becomes_empty_string() -> None:
    reply = await make_provider(
        lambda _: httpx.Response(200, json=completion_body(content=None))
    ).chat(make_request())
    assert reply.content == ""


async def test_rate_limit_is_retryable_with_retry_after() -> None:
    provider = make_provider(
        lambda _: httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
    )
    with pytest.raises(ProviderError) as info:
        await provider.chat(make_request())
    assert info.value.status_code == 429
    assert info.value.retryable is True
    assert info.value.retry_after_seconds == 7.0


async def test_server_error_is_retryable() -> None:
    provider = make_provider(lambda _: httpx.Response(503, text="overloaded"))
    with pytest.raises(ProviderError) as info:
        await provider.chat(make_request())
    assert info.value.status_code == 503
    assert info.value.retryable is True


async def test_bad_request_is_not_retryable() -> None:
    provider = make_provider(lambda _: httpx.Response(400, json={"error": "unknown model"}))
    with pytest.raises(ProviderError) as info:
        await provider.chat(make_request())
    assert info.value.status_code == 400
    assert info.value.retryable is False
    assert "unknown model" in str(info.value)


async def test_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderError) as info:
        await make_provider(handler).chat(make_request())
    assert info.value.retryable is True
    assert info.value.status_code is None


async def test_network_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ProviderError) as info:
        await make_provider(handler).chat(make_request())
    assert info.value.retryable is True


async def test_reply_in_wrong_format_is_an_error() -> None:
    provider = make_provider(lambda _: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError) as info:
        await provider.chat(make_request())
    assert info.value.retryable is False


async def test_api_key_never_appears_in_errors_or_repr() -> None:
    provider = make_provider(lambda _: httpx.Response(401, json={"error": "invalid key"}))
    with pytest.raises(ProviderError) as info:
        await provider.chat(make_request())
    assert API_KEY not in str(info.value)
    assert API_KEY not in repr(info.value)
    assert info.value.__cause__ is None
    assert API_KEY not in repr(provider)


def test_request_needs_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="some-model", messages=[])
