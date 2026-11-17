"""Tests for POST /v1/chat/completions, with a fake provider (no network, no keys)."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hardened_llm_gateway.app import create_app
from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.keys import hash_key
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
BASE_SETTINGS = {
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
    app = create_app(settings=Settings(**{**BASE_SETTINGS, **settings}), providers=providers)
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
