"""Tests for POST /v1/chat/completions, with a fake provider (no network, no keys)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from hardened_llm_gateway.app import create_app
from hardened_llm_gateway.providers import (
    ChatRequest,
    ChatResponse,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeProvider:
    def __init__(self, error: ProviderError | None = None) -> None:
        self.error = error
        self.requests: list[ChatRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ChatResponse(
            id="abc",
            provider="fake",
            model=request.model,
            content="Hello!",
            finish_reason="stop",
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            latency_ms=12.3,
        )


def post(provider: FakeProvider, body: dict[str, Any]) -> Any:
    with TestClient(create_app(providers={"fake": provider})) as client:
        return client.post("/v1/chat/completions", json=body)


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


def test_streaming_is_rejected_for_now() -> None:
    provider = FakeProvider()
    response = post(provider, {"model": "fake/m1", "messages": MESSAGES, "stream": True})
    assert response.status_code == 400
    assert provider.requests == []


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
