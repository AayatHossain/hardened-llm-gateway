"""The gateway's FastAPI application."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from hardened_llm_gateway import __version__
from hardened_llm_gateway.api import GatewayError, gateway_error_handler, router
from hardened_llm_gateway.config import Settings, get_settings
from hardened_llm_gateway.models import ModelDirectory
from hardened_llm_gateway.pricing import PriceTable
from hardened_llm_gateway.providers import Provider
from hardened_llm_gateway.providers.catalog import build_providers
from hardened_llm_gateway.ratelimit import RateLimiter

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    providers: Mapping[str, Provider] | None = None,
    prices: PriceTable | None = None,
) -> FastAPI:
    """Build a new gateway app. Used by uvicorn with --factory and by tests.

    Tests pass `providers` directly; otherwise they are built from settings at startup,
    sharing one HTTP client that is closed on shutdown. `prices` defaults to the
    shipped prices.toml.
    """
    resolved_settings = settings or get_settings()
    if not resolved_settings.clients:
        logger.warning("no clients configured (GATEWAY_CLIENTS): every /v1 request will get 401")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if providers is not None:
            app.state.providers = providers
            yield
            return
        async with httpx.AsyncClient() as client:
            app.state.providers = build_providers(resolved_settings, client)
            logger.info("enabled providers: %s", ", ".join(app.state.providers) or "none")
            yield

    app = FastAPI(title="Hardened LLM Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.rate_limiter = RateLimiter()
    app.state.model_directory = ModelDirectory()
    app.state.prices = prices if prices is not None else PriceTable.load()
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check: answers as long as the server is running."""
        return {"status": "ok"}

    return app
