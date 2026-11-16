"""The gateway's FastAPI application."""

from fastapi import FastAPI

from hardened_llm_gateway import __version__


def create_app() -> FastAPI:
    """Build a new gateway app. Used by uvicorn with --factory and by tests."""
    app = FastAPI(title="Hardened LLM Gateway", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check: answers as long as the server is running."""
        return {"status": "ok"}

    return app
