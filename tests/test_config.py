from pathlib import Path

import pytest
from pydantic import ValidationError

from hardened_llm_gateway.config import Settings

# The autouse clean_env fixture in conftest.py removes GATEWAY_ variables and
# moves each test into an empty folder, so the repo's real .env is never read.


def test_defaults_without_any_config() -> None:
    settings = Settings()
    assert settings.groq_api_key is None
    assert settings.request_timeout_seconds == 30.0


def test_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GATEWAY_REQUEST_TIMEOUT_SECONDS", "5")
    settings = Settings()
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "gsk-test"
    assert settings.request_timeout_seconds == 5.0


def test_reads_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GATEWAY_GEMINI_API_KEY=from-file\n", encoding="utf-8")
    settings = Settings()
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "from-file"


def test_api_keys_are_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_GROQ_API_KEY", "gsk-very-secret")
    assert "gsk-very-secret" not in repr(Settings())


def test_rejects_non_positive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_REQUEST_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_retry_defaults() -> None:
    settings = Settings()
    assert settings.max_retries == 1
    assert settings.fallbacks == {}


def test_reads_fallbacks_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GATEWAY_FALLBACKS",
        '{"groq/openai/gpt-oss-20b": ["gemini/gemini-2.5-flash-lite", "nvidia/some/model"]}',
    )
    settings = Settings()
    assert settings.fallbacks == {
        "groq/openai/gpt-oss-20b": ["gemini/gemini-2.5-flash-lite", "nvidia/some/model"]
    }


@pytest.mark.parametrize(
    "fallbacks",
    ['{"groq/m1": ["no-provider-prefix"]}', '{"no-prefix": ["gemini/m2"]}', '{"groq/m1": ["/m2"]}'],
)
def test_rejects_fallback_without_provider(monkeypatch: pytest.MonkeyPatch, fallbacks: str) -> None:
    monkeypatch.setenv("GATEWAY_FALLBACKS", fallbacks)
    with pytest.raises(ValidationError):
        Settings()


def test_rejects_too_many_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MAX_RETRIES", "10")
    with pytest.raises(ValidationError):
        Settings()


HASH_A = "a" * 64
HASH_B = "b" * 64


def test_no_clients_by_default() -> None:
    assert Settings().clients == {}


def test_reads_clients_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GATEWAY_CLIENTS",
        f'{{"web": {{"key_sha256": "{HASH_A}", "requests_per_minute": 20}},'
        f' "cli": {{"key_sha256": "{HASH_B}"}}}}',
    )
    clients = Settings().clients
    assert clients["web"].requests_per_minute == 20
    assert clients["cli"].requests_per_minute == 60  # the default


@pytest.mark.parametrize(
    "clients",
    [
        '{"web": {"key_sha256": "not-a-hash"}}',
        f'{{"web": {{"key_sha256": "{HASH_A}", "requests_per_minute": 0}}}}',
        f'{{"web": {{"key_sha256": "{HASH_A}"}}, "cli": {{"key_sha256": "{HASH_A}"}}}}',
    ],
    ids=["bad-hash", "zero-limit", "shared-key"],
)
def test_rejects_bad_clients(monkeypatch: pytest.MonkeyPatch, clients: str) -> None:
    monkeypatch.setenv("GATEWAY_CLIENTS", clients)
    with pytest.raises(ValidationError):
        Settings()
