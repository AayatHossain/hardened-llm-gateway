"""Gateway settings, read from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _check_target(value: str) -> str:
    provider, _, model = value.partition("/")
    if not provider or not model:
        raise ValueError(f'{value!r} must be "<provider>/<model>"')
    return value


class ClientConfig(BaseModel):
    """One app allowed to call the gateway.

    Only the SHA-256 of its key is stored, so a leaked .env does not leak client keys.
    Create a key and its hash with `uv run python -m hardened_llm_gateway.keys <name>`.
    """

    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requests_per_minute: int = Field(default=60, ge=1)


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

    # Apps allowed to call /v1, by name. Given as JSON, e.g.
    # GATEWAY_CLIENTS='{"my-app": {"key_sha256": "<64 hex chars>", "requests_per_minute": 20}}'
    # With no clients, every /v1 request is rejected.
    clients: dict[str, ClientConfig] = Field(default_factory=dict)

    # Where the request log is written (SQLAlchemy URL). Empty turns the log off.
    # Postgres later: postgresql+asyncpg://user:password@host/db
    request_log_url: str = "sqlite+aiosqlite:///gateway.db"

    # Where OpenTelemetry spans go: "off", "console" (printed as JSON) or "langfuse".
    # See tracing.py.
    tracing: Literal["off", "console", "langfuse"] = "off"
    # Langfuse project keys (Settings > API Keys), needed when tracing is "langfuse".
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @field_validator("fallbacks")
    @classmethod
    def _targets_have_provider_prefix(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for primary, backups in value.items():
            _check_target(primary)
            for backup in backups:
                _check_target(backup)
        return value

    @field_validator("clients")
    @classmethod
    def _keys_are_unique(cls, value: dict[str, ClientConfig]) -> dict[str, ClientConfig]:
        hashes = [client.key_sha256 for client in value.values()]
        if len(hashes) != len(set(hashes)):
            raise ValueError("2 clients share the same key")
        return value

    @model_validator(mode="after")
    def _langfuse_has_keys(self) -> Self:
        if self.tracing == "langfuse" and not (
            self.langfuse_public_key and self.langfuse_secret_key
        ):
            raise ValueError(
                "GATEWAY_TRACING=langfuse needs GATEWAY_LANGFUSE_PUBLIC_KEY and "
                "GATEWAY_LANGFUSE_SECRET_KEY"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
