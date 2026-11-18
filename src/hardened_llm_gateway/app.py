"""The gateway's FastAPI application."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider

from hardened_llm_gateway import __version__
from hardened_llm_gateway.api import GatewayError, gateway_error_handler, router
from hardened_llm_gateway.config import Settings, get_settings
from hardened_llm_gateway.models import ModelDirectory
from hardened_llm_gateway.pricing import PriceTable
from hardened_llm_gateway.providers import Provider
from hardened_llm_gateway.providers.catalog import build_providers
from hardened_llm_gateway.ratelimit import RateLimiter
from hardened_llm_gateway.requestlog import NullRequestLog, RequestLog, RequestRecorder
from hardened_llm_gateway.tracing import TracingMiddleware, build_tracer_provider, get_tracer

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    providers: Mapping[str, Provider] | None = None,
    prices: PriceTable | None = None,
    request_log: RequestRecorder | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Build a new gateway app. Used by uvicorn with --factory and by tests.

    Tests pass `providers`, `request_log` and `tracer_provider` directly. Otherwise, at
    startup, providers are built from settings and share one HTTP client, and the
    request log opens the database at GATEWAY_REQUEST_LOG_URL; both are closed on
    shutdown. Tracing follows GATEWAY_TRACING, and spans still buffered are sent on
    shutdown. `prices` defaults to the shipped prices.toml.
    """
    resolved_settings = settings or get_settings()
    if not resolved_settings.clients:
        logger.warning("no clients configured (GATEWAY_CLIENTS): every /v1 request will get 401")
    own_tracer_provider = None
    if tracer_provider is None:
        tracer_provider = own_tracer_provider = build_tracer_provider(resolved_settings)
    tracer = get_tracer(tracer_provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if own_tracer_provider is not None:
                stack.callback(own_tracer_provider.shutdown)  # runs last: sends buffered spans
            if request_log is not None:
                app.state.request_log = request_log
            elif resolved_settings.request_log_url:
                log = RequestLog(resolved_settings.request_log_url)
                await log.start()
                stack.push_async_callback(log.close)  # writes what is still queued
                app.state.request_log = log
            else:
                app.state.request_log = NullRequestLog()

            if providers is not None:
                app.state.providers = providers
            else:
                client = await stack.enter_async_context(httpx.AsyncClient())
                app.state.providers = build_providers(resolved_settings, client)
                logger.info("enabled providers: %s", ", ".join(app.state.providers) or "none")
            yield

    app = FastAPI(title="Hardened LLM Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.rate_limiter = RateLimiter()
    app.state.model_directory = ModelDirectory()
    app.state.prices = prices if prices is not None else PriceTable.load()
    app.state.tracer = tracer
    app.add_middleware(TracingMiddleware, tracer=tracer)
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check: answers as long as the server is running."""
        return {"status": "ok"}

    return app
