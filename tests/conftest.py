import os
from pathlib import Path

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests (marked with pytest.mark.anyio) on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test with no GATEWAY_ variables and away from the repo's real .env."""
    for name in list(os.environ):
        if name.startswith("GATEWAY_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
