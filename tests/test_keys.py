import json

import pytest

from hardened_llm_gateway.config import ClientConfig
from hardened_llm_gateway.keys import Client, find_client, hash_key, main, new_key


def test_new_keys_are_random_and_prefixed() -> None:
    first, second = new_key(), new_key()
    assert first.startswith("gw-")
    assert first != second
    assert len(first) > 40


def test_find_client_by_key() -> None:
    clients = {
        "web": ClientConfig(key_sha256=hash_key("gw-web"), requests_per_minute=20),
        "cli": ClientConfig(key_sha256=hash_key("gw-cli")),
    }
    assert find_client("gw-web", clients) == Client("web", 20)
    assert find_client("gw-cli", clients) == Client("cli", 60)
    assert find_client("gw-other", clients) is None
    assert find_client("gw-web", {}) is None


def test_command_prints_key_and_matching_entry(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["my-app"]) == 0
    lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    key = lines[1]
    entry = json.loads("{" + lines[-1] + "}")
    config = ClientConfig.model_validate(entry["my-app"])
    assert find_client(key, {"my-app": config}) == Client("my-app", 60)


def test_command_needs_a_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err
