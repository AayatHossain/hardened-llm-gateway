"""OpenTelemetry tracing: 1 trace per /v1 request, with a span for each step inside it.

A chat request's trace looks like this:

    POST /v1/chat/completions          the whole request, as the client saw it
    ├── rate limit                     the client's per-minute limit
    ├── chat groq/openai/gpt-oss-20b   attempt 1
    ├── retry wait                     the backoff before attempt 2
    ├── chat groq/openai/gpt-oss-20b   attempt 2
    └── chat gemini/gemini-2.5-flash   a fallback target (for a stream: until the last chunk)

Only the "chat" spans (1 per model call) carry the OpenTelemetry GenAI names
(gen_ai.provider.name, gen_ai.usage.input_tokens, gen_ai.usage.cost, ...), so tools such
as Langfuse count each call's tokens and cost exactly once. The root span holds the
request's outcome under gateway.* names. Prompt and reply text are never put on spans.

GATEWAY_TRACING picks where spans go: "off" (default), "console" (printed as JSON) or
"langfuse" (needs GATEWAY_LANGFUSE_PUBLIC_KEY and GATEWAY_LANGFUSE_SECRET_KEY).
"""

import base64
import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import NoOpTracerProvider, SpanKind, Status, StatusCode, Tracer
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hardened_llm_gateway import __version__
from hardened_llm_gateway.config import Settings

logger = logging.getLogger(__name__)

TRACER_NAME = "hardened_llm_gateway"


def build_tracer_provider(settings: Settings) -> TracerProvider | None:
    """A tracer provider for GATEWAY_TRACING, or None when tracing is off.

    Spans are sent in batches from a background thread, so exporting never slows a
    request; if Langfuse is unreachable the exporter logs it and the gateway carries on.
    """
    if settings.tracing == "off":
        return None
    resource = Resource.create(
        {"service.name": "hardened-llm-gateway", "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    exporter = (
        ConsoleSpanExporter() if settings.tracing == "console" else langfuse_exporter(settings)
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    logger.info("tracing to %s", settings.tracing)
    return provider


def langfuse_exporter(settings: Settings) -> OTLPSpanExporter:
    endpoint, headers = langfuse_endpoint(settings)
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def langfuse_endpoint(settings: Settings) -> tuple[str, dict[str, str]]:
    """Langfuse's OpenTelemetry URL and headers: OTLP over HTTP, project keys as Basic auth."""
    assert settings.langfuse_public_key and settings.langfuse_secret_key  # checked in config
    pair = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key.get_secret_value()}"
    url = settings.langfuse_base_url.rstrip("/") + "/api/public/otel/v1/traces"
    return url, {
        "authorization": "Basic " + base64.b64encode(pair.encode()).decode(),
        # Without it, Langfuse may take up to 10 minutes to show new traces.
        "x-langfuse-ingestion-version": "4",
    }


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
