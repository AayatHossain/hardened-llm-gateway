import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from hardened_llm_gateway.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test with no GATEWAY_ variables and no .env from the repo."""
    for name in list(os.environ):
        if name.startswith("GATEWAY_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


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
