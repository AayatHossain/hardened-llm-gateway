"""Gateway settings, read from environment variables and an optional .env file."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings use the GATEWAY_ prefix, e.g. GATEWAY_GROQ_API_KEY.

    API keys are SecretStr so they never appear in logs, reprs or error messages.
    A provider whose key is missing is simply not available.
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    nvidia_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    request_timeout_seconds: float = Field(default=30.0, gt=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
