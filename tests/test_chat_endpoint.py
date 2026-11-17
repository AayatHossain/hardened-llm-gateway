"""Tests for the /v1 endpoints, with a fake provider (no network, no keys).

The last section drives the gateway with the official OpenAI Python SDK.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from hardened_llm_gateway.app import create_app
from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.keys import hash_key
from hardened_llm_gateway.pricing import ModelPrice, PriceTable
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeProvider:
    """Answers "Hello!" (streamed as "Hel" + "lo!"), or raises `error`.

    When streaming, the error is raised after `fail_after_chunks` chunks.
    """

    def __init__(
        self,
        error: ProviderError | None = None,
        fail_after_chunks: int = 0,
        name: str = "fake",
    ) -> None:
        self.error = error
        self.fail_after_chunks = fail_after_chunks
        self._name = name
        self.requests: list[ChatRequest] = []
        self.model_list_calls = 0

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        for index, piece in enumerate(["Hel", "lo!"]):
            if self.error is not None and index == self.fail_after_chunks:
                raise self.error
            yield ChatChunk(id="abc", provider=self._name, model=request.model, delta=piece)
        yield ChatChunk(
            id="abc",
            provider=self._name,
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    @property
    def name(self) -> str:
        return self._name

    async def list_models(self) -> list[str]:
        self.model_list_calls += 1
        if self.error is not None:
            raise self.error
        return ["m1", "m2"]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ChatResponse(
            id="abc",
            provider=self._name,
            model=request.model,
            content="Hello!",
            finish_reason="stop",
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            latency_ms=12.3,
        )


KEY = "gw-test-key"
# fake/m1 is priced (1 USD in, 2 USD out per 1M tokens); backup/m2 is not.
PRICES = PriceTable({"fake/m1": ModelPrice(input=1.0, output=2.0)})
BASE_SETTINGS: dict[str, Any] = {
    "clients": {"tests": {"key_sha256": hash_key(KEY)}},
    # No real waiting in tests: zero backoff, and any Retry-After is "too long" (use a fallback).
    "retry_backoff_seconds": 0,
    "max_retry_wait_seconds": 0,
}


def post(
    provider: FakeProvider,
    body: dict[str, Any],
    *others: FakeProvider,
    key: str | None = KEY,
    **settings: Any,
) -> Any:
    providers = {p.name: p for p in (provider, *others)}
    app = create_app(
        settings=Settings(**{**BASE_SETTINGS, **settings}), providers=providers, prices=PRICES
    )
    headers = {"authorization": f"Bearer {key}"} if key is not None else {}
    with TestClient(app) as client:
        return client.post("/v1/chat/completions", json=body, headers=headers)


# --- client keys -------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "message"), [(None, "missing API key"), ("gw-wrong", "invalid API key")]
)
def test_request_without_a_valid_key_is_rejected(key: str | None, message: str) -> None:
    provider = FakeProvider()
    response = post(provider, {"model": "fake/m1", "messages": MESSAGES}, key=key)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    error = response.json()["error"]
    assert error["code"] == "invalid_api_key"
    assert message in error["message"]
    assert provider.requests == []  # rejected before any provider call


def test_no_configured_clients_rejects_every_key() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": MESSAGES}, clients={})
    assert response.status_code == 401


def test_health_needs_no_key() -> None:
    with TestClient(create_app(settings=Settings())) as client:
        assert client.get("/health").status_code == 200


# --- model list --------------------------------------------------------------


def get_models(*providers: FakeProvider, key: str | None = KEY) -> Any:
    app = create_app(
        settings=Settings(**BASE_SETTINGS), providers={p.name: p for p in providers}, prices=PRICES
    )
    headers = {"authorization": f"Bearer {key}"} if key is not None else {}
    with TestClient(app) as client:
        return client.get("/v1/models", headers=headers)


def test_models_lists_every_enabled_provider() -> None:
    response = get_models(FakeProvider(), FakeProvider(name="backup"))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["backup/m1", "backup/m2", "fake/m1", "fake/m2"]
    assert body["data"][0] == {
        "id": "backup/m1",
        "object": "model",
        "created": 0,
        "owned_by": "backup",
    }
    assert response.headers["x-ratelimit-remaining-requests"] == "59"


def test_models_skips_a_failing_provider() -> None:
    response = get_models(FakeProvider(UNAVAILABLE), FakeProvider(name="backup"))
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["backup/m1", "backup/m2"]


def test_models_needs_a_key() -> None:
    provider = FakeProvider()
    assert get_models(provider, key=None).status_code == 401
    assert provider.model_list_calls == 0


# --- per-client rate limits --------------------------------------------------


def test_client_over_its_limit_gets_429_before_any_provider_call() -> None:
    other_key = "gw-other-key"
    clients = {
        "tests": {"key_sha256": hash_key(KEY), "requests_per_minute": 2},
        "other": {"key_sha256": hash_key(other_key), "requests_per_minute": 2},
    }
    provider = FakeProvider()
    app = create_app(
        settings=Settings(**{**BASE_SETTINGS, "clients": clients}),
        providers={"fake": provider},
    )
    body = {"model": "fake/m1", "messages": MESSAGES}

    def send(key: str) -> Any:
        return client.post(
            "/v1/chat/completions", json=body, headers={"authorization": f"Bearer {key}"}
        )

    with TestClient(app) as client:
        first, second, third = send(KEY), send(KEY), send(KEY)
        other = send(other_key)

    assert [r.status_code for r in (first, second)] == [200, 200]
    assert first.headers["x-ratelimit-limit-requests"] == "2"
    assert first.headers["x-ratelimit-remaining-requests"] == "1"

    assert third.status_code == 429
    assert third.json()["error"]["code"] == "client_rate_limit_exceeded"
    assert third.headers["x-ratelimit-remaining-requests"] == "0"
    assert 1 <= int(third.headers["retry-after"]) <= 30  # 2 per minute: 1 back every 30 s
    assert len(provider.requests) == 3  # 2 for "tests", 1 for "other"; the 429 never got through

    assert other.status_code == 200  # each client has its own limit


def test_streamed_reply_has_rate_limit_headers() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": MESSAGES, "stream": True})
    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit-requests"] == "60"
    assert response.headers["x-ratelimit-remaining-requests"] == "59"


# --- chat --------------------------------------------------------------------


def test_successful_chat() -> None:
    provider = FakeProvider()
    response = post(provider, {"model": "fake/m1", "messages": MESSAGES, "temperature": 0.5})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "abc"
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake/m1"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 5
    assert response.headers["x-gateway-provider"] == "fake"
    assert response.headers["x-gateway-latency-ms"] == "12"

    sent = provider.requests[0]
    assert sent.model == "m1"
    assert sent.temperature == 0.5


def test_model_name_may_contain_slashes() -> None:
    provider = FakeProvider()
    post(provider, {"model": "fake/openai/gpt-oss-20b", "messages": MESSAGES})
    assert provider.requests[0].model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("model", ["m1", "fake/"])
def test_model_without_provider_prefix_is_rejected(model: str) -> None:
    response = post(FakeProvider(), {"model": model, "messages": MESSAGES})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_provider_lists_enabled_ones() -> None:
    response = post(FakeProvider(), {"model": "nope/m1", "messages": MESSAGES})
    assert response.status_code == 400
    assert "enabled: fake" in response.json()["error"]["message"]


def stream_events(response: Any) -> list[Any]:
    """Parse a server-sent-events body: JSON events as dicts, "[DONE]" as a string."""
    events: list[Any] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_streaming_chat() -> None:
    provider = FakeProvider()
    response = post(provider, {"model": "fake/m1", "messages": MESSAGES, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-gateway-provider"] == "fake"
    assert "x-gateway-first-chunk-ms" in response.headers

    events = stream_events(response)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == "fake/m1" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "Hello!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 5
    assert provider.requests[0].model == "m1"


def test_stream_error_before_first_chunk_is_a_normal_http_error() -> None:
    error = ProviderError(
        "fake", "HTTP 429", status_code=429, retryable=True, retry_after_seconds=2
    )
    response = post(
        FakeProvider(error, fail_after_chunks=0),
        {"model": "fake/m1", "messages": MESSAGES, "stream": True},
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"


def test_stream_error_mid_reply_becomes_an_error_event() -> None:
    error = ProviderError("fake", "network error: ReadError", retryable=True)
    response = post(
        FakeProvider(error, fail_after_chunks=1),
        {"model": "fake/m1", "messages": MESSAGES, "stream": True},
    )
    assert response.status_code == 200  # headers were already sent
    events = stream_events(response)
    assert events[0]["choices"][0]["delta"]["content"] == "Hel"
    assert events[-1] == {"error": {"message": "provider 'fake' failed", "type": "provider_error"}}
    assert "[DONE]" not in events


def test_empty_messages_fail_validation() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": []})
    assert response.status_code == 422


def test_provider_rate_limit_becomes_429_with_retry_after() -> None:
    error = ProviderError(
        "fake", "HTTP 429", status_code=429, retryable=True, retry_after_seconds=6.2
    )
    response = post(FakeProvider(error), {"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
    assert response.json()["error"]["type"] == "rate_limit_error"


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ProviderTimeoutError("fake"), 504),
        (ProviderError("fake", "HTTP 400: unknown model", status_code=400), 400),
        (ProviderError("fake", "HTTP 500: boom", status_code=500, retryable=True), 502),
        (ProviderError("fake", "network error: ConnectError", retryable=True), 502),
    ],
)
def test_provider_errors_map_to_http_status(error: ProviderError, expected_status: int) -> None:
    response = post(FakeProvider(error), {"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == expected_status


def test_rejected_request_shows_only_the_provider_reason() -> None:
    error = ProviderError(
        "fake",
        'HTTP 404: {"error": {"message": "The model `m1` does not exist"}}',
        status_code=404,
        reason="The model `m1` does not exist",
    )
    response = post(FakeProvider(error), {"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "provider 'fake' rejected the request: The model `m1` does not exist"
    )


def test_provider_auth_failure_details_are_not_shown_to_client() -> None:
    error = ProviderError("fake", "HTTP 401: Invalid API Key", status_code=401)
    response = post(FakeProvider(error), {"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == 502
    assert "Invalid API Key" not in response.text


# --- retries and fallback ----------------------------------------------------

UNAVAILABLE = ProviderError("fake", "HTTP 503", status_code=503, retryable=True)
FALLBACK: dict[str, Any] = {"fallbacks": {"fake/m1": ["backup/m2"]}}


def test_decision_headers_on_first_try() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": MESSAGES})
    assert response.headers["x-gateway-target"] == "fake/m1"
    assert response.headers["x-gateway-attempts"] == "1"


def test_retryable_failure_is_retried_on_the_same_target() -> None:
    primary = FakeProvider(UNAVAILABLE)
    response = post(primary, {"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == 502  # still failing after the retry
    assert len(primary.requests) == 2  # 1 try + 1 retry (max_retries defaults to 1)


def test_falls_back_to_the_configured_backup() -> None:
    primary = FakeProvider(UNAVAILABLE)
    backup = FakeProvider(name="backup")
    response = post(primary, {"model": "fake/m1", "messages": MESSAGES}, backup, **FALLBACK)

    assert response.status_code == 200
    assert response.json()["model"] == "backup/m2"
    assert response.headers["x-gateway-provider"] == "backup"
    assert response.headers["x-gateway-target"] == "backup/m2"
    assert response.headers["x-gateway-attempts"] == "3"
    assert backup.requests[0].model == "m2"


def test_invalid_request_does_not_fall_back() -> None:
    error = ProviderError("fake", "HTTP 400: bad temperature", status_code=400)
    backup = FakeProvider(name="backup")
    response = post(
        FakeProvider(error), {"model": "fake/m1", "messages": MESSAGES}, backup, **FALLBACK
    )
    assert response.status_code == 400
    assert backup.requests == []


def test_stream_falls_back_before_the_first_chunk() -> None:
    primary = FakeProvider(UNAVAILABLE, fail_after_chunks=0)
    backup = FakeProvider(name="backup")
    response = post(
        primary,
        {"model": "fake/m1", "messages": MESSAGES, "stream": True},
        backup,
        **FALLBACK,
    )
    assert response.status_code == 200
    assert response.headers["x-gateway-target"] == "backup/m2"
    events = stream_events(response)
    assert events[-1] == "[DONE]"
    assert all(e["model"] == "backup/m2" for e in events[:-1])
    assert "".join(e["choices"][0]["delta"]["content"] for e in events[:-1]) == "Hello!"


# --- the official OpenAI SDK, pointed at the gateway --------------------------


@asynccontextmanager
async def sdk(
    *providers: FakeProvider, key: str = KEY, **settings: Any
) -> AsyncIterator[openai.AsyncOpenAI]:
    """An OpenAI SDK client whose only changes are base_url and api_key.

    It reaches the app in memory through an ASGI transport, so no server or network
    is needed. SDK retries are off so each test sees exactly one gateway reply.
    """
    app = create_app(
        settings=Settings(**{**BASE_SETTINGS, **settings}),
        providers={p.name: p for p in providers},
        prices=PRICES,
    )
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as http,
    ):
        yield openai.AsyncOpenAI(
            api_key=key, base_url="http://gateway/v1", http_client=http, max_retries=0
        )


@pytest.mark.anyio
async def test_sdk_chat() -> None:
    async with sdk(FakeProvider()) as client:
        reply = await client.chat.completions.create(
            model="fake/m1", messages=[{"role": "user", "content": "hi"}]
        )
    assert reply.choices[0].message.content == "Hello!"
    assert reply.model == "fake/m1"
    assert reply.usage is not None
    assert reply.usage.total_tokens == 5


@pytest.mark.anyio
async def test_sdk_streaming() -> None:
    async with sdk(FakeProvider()) as client:
        stream = await client.chat.completions.create(
            model="fake/m1", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        pieces = [c.choices[0].delta.content or "" async for c in stream if c.choices]
    assert "".join(pieces) == "Hello!"


@pytest.mark.anyio
async def test_sdk_reads_gateway_headers() -> None:
    async with sdk(FakeProvider()) as client:
        raw = await client.chat.completions.with_raw_response.create(
            model="fake/m1", messages=[{"role": "user", "content": "hi"}]
        )
    assert raw.headers["x-gateway-target"] == "fake/m1"
    assert raw.headers["x-ratelimit-remaining-requests"] == "59"


@pytest.mark.anyio
async def test_sdk_lists_models() -> None:
    async with sdk(FakeProvider()) as client:
        ids = [model.id async for model in client.models.list()]
    assert ids == ["fake/m1", "fake/m2"]


@pytest.mark.anyio
async def test_sdk_raises_its_usual_exceptions() -> None:
    message: Any = [{"role": "user", "content": "hi"}]

    async with sdk(FakeProvider(), key="gw-wrong") as client:
        with pytest.raises(openai.AuthenticationError):
            await client.chat.completions.create(model="fake/m1", messages=message)

    async with sdk(FakeProvider()) as client:
        with pytest.raises(openai.BadRequestError, match="provider"):
            await client.chat.completions.create(model="no-prefix", messages=message)

    limited = {"tests": {"key_sha256": hash_key(KEY), "requests_per_minute": 1}}
    async with sdk(FakeProvider(), clients=limited) as client:
        await client.chat.completions.create(model="fake/m1", messages=message)
        with pytest.raises(openai.RateLimitError):
            await client.chat.completions.create(model="fake/m1", messages=message)


# --- cost ---------------------------------------------------------------------

# FakeProvider reports 3 prompt and 2 completion tokens: 3 x 1 + 2 x 2 = 7 USD per 1M.


def test_reply_cost_is_in_a_header() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": MESSAGES})
    assert response.headers["x-gateway-cost-usd"] == "0.000007"


def test_unpriced_model_cost_is_unknown_not_zero() -> None:
    primary = FakeProvider(UNAVAILABLE)
    backup = FakeProvider(name="backup")
    response = post(primary, {"model": "fake/m1", "messages": MESSAGES}, backup, **FALLBACK)
    assert response.headers["x-gateway-target"] == "backup/m2"
    assert response.headers["x-gateway-cost-usd"] == "unknown"


def test_streamed_cost_comes_with_the_usage_event() -> None:
    response = post(FakeProvider(), {"model": "fake/m1", "messages": MESSAGES, "stream": True})
    chunks = stream_events(response)[:-1]
    assert "x_gateway" not in chunks[0]
    assert chunks[-1]["usage"]["total_tokens"] == 5
    assert chunks[-1]["x_gateway"]["cost_usd"] == pytest.approx(0.000007)


@pytest.mark.anyio
async def test_sdk_accepts_the_extra_cost_field() -> None:
    async with sdk(FakeProvider()) as client:
        stream = await client.chat.completions.create(
            model="fake/m1", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        chunks = [chunk async for chunk in stream]
    extra = chunks[-1].model_extra or {}
    assert extra["x_gateway"]["cost_usd"] == pytest.approx(0.000007)
