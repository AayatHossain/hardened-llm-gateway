"""Tests for the OpenAI-compatible adapter, against a fake HTTP server (no network, no keys)."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderTimeoutError,
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


GROQ_404 = {"error": {"message": "The model `x` does not exist", "type": "invalid_request_error"}}
GEMINI_404 = [{"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}}]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (GROQ_404, "The model `x` does not exist"),
        (GEMINI_404, "models/x is not found"),
        ({"error": "plain string"}, None),
        ("not json", None),
    ],
    ids=["groq", "gemini", "other-json", "text"],
)
async def test_error_reason_is_taken_from_the_provider_body(body: Any, reason: str | None) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(404, text=body)
        return httpx.Response(404, json=body)

    with pytest.raises(ProviderError) as info:
        await make_provider(handler).chat(make_request())
    assert info.value.reason == reason
    assert str(info.value).startswith("fake: HTTP 404: ")


async def test_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderTimeoutError) as info:
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


# --- streaming -------------------------------------------------------------


def sse(*events: dict[str, Any] | str) -> bytes:
    """Build a server-sent-events body.

    Dicts become JSON data events, strings starting with ":" are comment lines,
    other strings are sent as raw data (e.g. "[DONE]").
    """
    lines = []
    for event in events:
        if isinstance(event, str) and event.startswith(":"):
            lines.append(f"{event}\n\n")
            continue
        data = json.dumps(event) if isinstance(event, dict) else event
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


def chunk(content: str | None = None, finish: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "some-model",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
        **extra,
    }


async def collect(provider: OpenAICompatibleProvider) -> list[ChatChunk]:
    return [piece async for piece in provider.stream(make_request())]


async def test_stream_yields_pieces_in_order() -> None:
    seen: list[httpx.Request] = []
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    body = sse(
        chunk("Hel"),
        ": keep-alive",
        chunk("lo!"),
        chunk(finish="stop"),
        {"id": "chatcmpl-1", "model": "some-model", "choices": [], "usage": usage},
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    pieces = await collect(make_provider(handler))

    assert "".join(p.delta for p in pieces) == "Hello!"
    assert pieces[2].finish_reason == "stop"
    assert pieces[-1].usage is not None
    assert pieces[-1].usage.total_tokens == 7
    assert all(p.provider == "fake" for p in pieces)
    sent = json.loads(seen[0].content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


async def test_stream_reads_groq_usage() -> None:
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    body = sse(chunk("hi", finish="stop", x_groq={"usage": usage}), "[DONE]")
    pieces = await collect(make_provider(lambda _: httpx.Response(200, content=body)))
    assert pieces[0].usage is not None
    assert pieces[0].usage.total_tokens == 2


async def test_stream_http_error_is_raised_before_any_piece() -> None:
    provider = make_provider(
        lambda _: httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"})
    )
    with pytest.raises(ProviderError) as info:
        await collect(provider)
    assert info.value.status_code == 429
    assert info.value.retryable is True
    assert info.value.retry_after_seconds == 3.0


async def test_stream_bad_event_is_an_error() -> None:
    body = sse(chunk("ok"), "{not json")
    with pytest.raises(ProviderError) as info:
        await collect(make_provider(lambda _: httpx.Response(200, content=body)))
    assert info.value.retryable is False


async def test_stream_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    with pytest.raises(ProviderTimeoutError):
        await collect(make_provider(handler))


async def test_list_models_keeps_chat_models_with_clean_ids() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "openai/gpt-oss-20b", "output_modalities": ["text"]},
                    {"id": "whisper-large-v3", "output_modalities": ["transcription"]},
                    {"id": "models/gemini-2.5-flash", "owned_by": "google"},
                    {"id": "gemini-2.5-flash"},
                ],
            },
        )

    ids = await make_provider(handler).list_models()
    assert ids == ["gemini-2.5-flash", "openai/gpt-oss-20b"]
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"{BASE_URL}/models"
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"


async def test_list_models_errors() -> None:
    provider = make_provider(lambda _: httpx.Response(401, json=GROQ_404))
    with pytest.raises(ProviderError) as info:
        await provider.list_models()
    assert info.value.status_code == 401

    provider = make_provider(lambda _: httpx.Response(200, json={"models": []}))
    with pytest.raises(ProviderError, match="model list"):
        await provider.list_models()
