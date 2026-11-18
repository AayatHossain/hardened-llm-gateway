"""Tests for OpenTelemetry tracing: which spans a request makes and what they carry."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from test_chat_endpoint import (
    BASE_SETTINGS,
    FALLBACK,
    KEY,
    MESSAGES,
    PRICES,
    UNAVAILABLE,
    FakeProvider,
)

from hardened_llm_gateway.app import create_app
from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.providers import ProviderError
from hardened_llm_gateway.tracing import build_tracer_provider


def traced_post(
    body: dict[str, Any],
    *providers: FakeProvider,
    key: str = KEY,
    path: str = "/v1/chat/completions",
    **settings: Any,
) -> tuple[Any, list[ReadableSpan]]:
    """Send 1 request; return the response and the finished spans, in finishing order."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(
        settings=Settings(**{**BASE_SETTINGS, **settings}),
        providers={p.name: p for p in providers or (FakeProvider(),)},
        prices=PRICES,
        tracer_provider=tracer_provider,
    )
    with TestClient(app) as client:
        headers = {"authorization": f"Bearer {key}"}
        if path == "/v1/chat/completions":
            response = client.post(path, json=body, headers=headers)
        else:
            response = client.get(path, headers=headers)
    return response, list(exporter.get_finished_spans())


def by_name(spans: list[ReadableSpan]) -> dict[str, list[ReadableSpan]]:
    named: dict[str, list[ReadableSpan]] = {}
    for span in spans:
        named.setdefault(span.name, []).append(span)
    return named


def root_of(spans: list[ReadableSpan]) -> ReadableSpan:
    (root,) = [s for s in spans if s.parent is None]
    return root


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def test_successful_chat_makes_1_trace_with_a_span_per_step() -> None:
    response, spans = traced_post({"model": "fake/m1", "messages": MESSAGES})
    assert response.status_code == 200

    root = root_of(spans)
    assert root.name == "POST /v1/chat/completions"
    assert root.kind == SpanKind.SERVER
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    assert sorted(s.name for s in spans if s is not root) == ["chat fake/m1", "rate limit"]
    assert all(s.parent is not None and s.parent.span_id == root.context.span_id
               for s in spans if s is not root)  # fmt: skip

    assert attrs(root) | {
        "http.response.status_code": 200,
        "gateway.client": "tests",
        "gateway.request_id": response.headers["x-gateway-request-id"],
        "gen_ai.request.model": "fake/m1",
        "gen_ai.provider.name": "fake",
        "gateway.target": "fake/m1",
        "gen_ai.response.model": "fake/m1",
        "gen_ai.usage.input_tokens": 3,
        "gen_ai.usage.output_tokens": 2,
        "gateway.cost_usd": pytest.approx(0.000007),
        "gateway.attempts": 1,
        "gateway.stream": False,
    } == attrs(root)
    assert root.status.status_code == StatusCode.UNSET

    (call,) = by_name(spans)["chat fake/m1"]
    assert call.kind == SpanKind.CLIENT
    assert attrs(call) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "m1",
        "gen_ai.response.model": "m1",
        "gen_ai.usage.input_tokens": 3,
        "gen_ai.usage.output_tokens": 2,
        "gateway.target": "fake/m1",
        "gateway.attempt": 1,
    }

    (limit,) = by_name(spans)["rate limit"]
    assert attrs(limit) == {
        "gateway.rate_limit.limit": 60,
        "gateway.rate_limit.remaining": 59,
        "gateway.rate_limit.allowed": True,
    }


def test_retries_and_fallback_show_as_separate_spans() -> None:
    primary = FakeProvider(UNAVAILABLE)
    backup = FakeProvider(name="backup")
    response, spans = traced_post(
        {"model": "fake/m1", "messages": MESSAGES}, primary, backup, **FALLBACK
    )
    assert response.status_code == 200

    root = root_of(spans)
    steps = sorted((s for s in spans if s is not root), key=lambda s: s.start_time or 0)
    assert [s.name for s in steps] == [
        "rate limit",
        "chat fake/m1",
        "retry wait",
        "chat fake/m1",
        "chat backup/m2",
    ]
    first, _, second, backup_call = steps[1:]
    assert attrs(first)["error.type"] == "503"
    assert first.status.status_code == StatusCode.ERROR
    assert (attrs(first)["gateway.attempt"], attrs(second)["gateway.attempt"]) == (1, 2)
    assert attrs(backup_call)["gateway.attempt"] == 3
    assert backup_call.status.status_code == StatusCode.UNSET
    assert attrs(root)["gen_ai.response.model"] == "backup/m2"
    assert attrs(root)["gateway.attempts"] == 3


def test_provider_failure_marks_the_request_as_failed() -> None:
    response, spans = traced_post({"model": "fake/m1", "messages": MESSAGES},
                                  FakeProvider(UNAVAILABLE), max_retries=0)  # fmt: skip
    assert response.status_code == 502
    root = root_of(spans)
    assert root.status.status_code == StatusCode.ERROR
    assert attrs(root)["http.response.status_code"] == 502
    assert attrs(root)["error.type"] == "provider_error"
    assert attrs(root)["gateway.target"] == "fake/m1"
    assert "gen_ai.response.model" not in attrs(root)  # nothing answered


def test_bad_request_is_noted_but_not_an_error() -> None:
    error = ProviderError("fake", "HTTP 404", status_code=404, reason="no such model")
    response, spans = traced_post({"model": "fake/nope", "messages": MESSAGES},
                                  FakeProvider(error))  # fmt: skip
    assert response.status_code == 400
    root = root_of(spans)
    assert attrs(root)["error.type"] == "invalid_request_error"
    assert root.status.status_code == StatusCode.UNSET


def test_exception_text_is_not_put_on_spans() -> None:
    error = ProviderError("fake", "HTTP 503: echo of the user's prompt", status_code=503)
    _, spans = traced_post({"model": "fake/m1", "messages": MESSAGES},
                           FakeProvider(error), max_retries=0)  # fmt: skip
    for span in spans:
        assert span.events == ()  # no recorded exception events
        assert "prompt" not in str(attrs(span))


def test_streamed_chat_is_traced_until_the_last_chunk() -> None:
    response, spans = traced_post({"model": "fake/m1", "messages": MESSAGES, "stream": True})
    assert response.status_code == 200
    root = root_of(spans)
    (call,) = by_name(spans)["chat fake/m1"]
    # The attempt span covers waiting for the first chunk; the root covers the whole stream.
    assert root.end_time is not None and call.end_time is not None
    assert root.end_time >= call.end_time
    assert attrs(root)["gateway.stream"] is True
    assert attrs(root)["gen_ai.usage.output_tokens"] == 2
    assert attrs(root)["gateway.first_chunk_ms"] >= 0


def test_stream_failing_mid_reply_marks_the_request_as_failed() -> None:
    error = ProviderError("fake", "network error", retryable=True)
    _, spans = traced_post({"model": "fake/m1", "messages": MESSAGES, "stream": True},
                           FakeProvider(error, fail_after_chunks=1))  # fmt: skip
    root = root_of(spans)
    assert attrs(root)["http.response.status_code"] == 200
    assert attrs(root)["error.type"] == "provider_error"
    assert root.status.status_code == StatusCode.ERROR


def test_rejected_key_still_leaves_a_trace() -> None:
    response, spans = traced_post({"model": "fake/m1", "messages": MESSAGES}, key="gw-wrong")
    assert response.status_code == 401
    (root,) = spans
    assert attrs(root)["http.response.status_code"] == 401
    assert "gateway.client" not in attrs(root)


def test_model_list_is_traced() -> None:
    response, spans = traced_post({}, path="/v1/models")
    assert response.status_code == 200
    assert root_of(spans).name == "GET /v1/models"


def test_health_is_not_traced() -> None:
    _, spans = traced_post({}, path="/health")
    assert spans == []


def test_tracing_is_off_by_default() -> None:
    assert Settings().tracing == "off"
    assert build_tracer_provider("off") is None
    provider = build_tracer_provider("console")
    assert provider is not None
    assert provider.resource.attributes["service.name"] == "hardened-llm-gateway"
    provider.shutdown()
