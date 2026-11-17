"""Gateway settings, read from environment variables and an optional .env file."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _check_target(value: str) -> str:
    provider, _, model = value.partition("/")
    if not provider or not model:
        raise ValueError(f'{value!r} must be "<provider>/<model>"')
    return value


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

    # Retries of the same provider, for timeouts, network errors, 429 and 5xx.
    max_retries: int = Field(default=1, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    # Never wait longer than this before a retry; a longer Retry-After means "use a fallback".
    max_retry_wait_seconds: float = Field(default=5.0, ge=0)

    # Where to go when a target keeps failing, in order. Given as JSON, e.g.
    # GATEWAY_FALLBACKS='{"groq/openai/gpt-oss-20b": ["gemini/gemini-2.5-flash-lite"]}'
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("fallbacks")
    @classmethod
    def _targets_have_provider_prefix(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for primary, backups in value.items():
            _check_target(primary)
            for backup in backups:
                _check_target(backup)
        return value


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
