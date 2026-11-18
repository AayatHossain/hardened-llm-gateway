"""OpenTelemetry tracing: 1 trace per /v1 request, with a span for each step inside it.

A chat request's trace looks like this:

    POST /v1/chat/completions          the whole request, as the client saw it
    ├── rate limit                     key check and the client's per-minute limit
    ├── chat groq/openai/gpt-oss-20b   attempt 1 (for a stream: until the first chunk)
    ├── retry wait                     the backoff before attempt 2
    ├── chat groq/openai/gpt-oss-20b   attempt 2
    └── chat gemini/gemini-2.5-flash   a fallback target

Model calls use the OpenTelemetry GenAI names (gen_ai.provider.name,
gen_ai.usage.input_tokens, ...) so tools such as Langfuse can read them. Prompt and
reply text are never put on spans.

GATEWAY_TRACING picks where spans go: "off" (default) or "console" (printed as JSON).
"""

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import NoOpTracerProvider, SpanKind, Status, StatusCode, Tracer
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hardened_llm_gateway import __version__

logger = logging.getLogger(__name__)

TRACER_NAME = "hardened_llm_gateway"


def build_tracer_provider(mode: str) -> TracerProvider | None:
    """A tracer provider for GATEWAY_TRACING, or None when tracing is off."""
    if mode == "off":
        return None
    resource = Resource.create(
        {"service.name": "hardened-llm-gateway", "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    if mode == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:  # config.py only allows the modes above
        raise ValueError(f"unknown tracing mode {mode!r}")
    return provider


def get_tracer(provider: TracerProvider | None) -> Tracer:
    """The gateway's tracer. With no provider it makes spans that record nothing."""
    return trace.get_tracer(TRACER_NAME, __version__, provider or NoOpTracerProvider())


class TracingMiddleware:
    """Opens the root span of every /v1 request, so every later span is its child.

    A plain ASGI middleware: a streamed reply is sent inside this call, so the span
    ends only when the last chunk has gone out.
    """

    def __init__(self, app: ASGIApp, tracer: Tracer) -> None:
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        method, path = scope["method"], scope["path"]
        with self.tracer.start_as_current_span(
            f"{method} {path}",
            kind=SpanKind.SERVER,
            attributes={"http.request.method": method, "url.path": path},
        ) as span:

            async def send_and_note_status(message: Message) -> None:
                if message["type"] == "http.response.start":
                    status = message["status"]
                    span.set_attribute("http.response.status_code", status)
                    if status >= 500:
                        span.set_status(Status(StatusCode.ERROR))
                await send(message)

            await self.app(scope, receive, send_and_note_status)
