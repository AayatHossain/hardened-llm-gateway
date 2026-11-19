"""A small persistent cache for eval runs: model replies and judge verdicts.

Entries are kept in a JSON Lines file, 1 line per entry, appended as they arrive,
so a run that crashes keeps everything it already paid for. Keys are hashes of the
exact request (see `request_key`), so a changed prompt, model or setting is a miss.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def request_key(**parts: Any) -> str:
    """A stable hash of the parts of a request, independent of their order."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class JsonlCache:
    """Values by key, in a JSON Lines file, or only in memory when `path` is None."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry["value"]

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._entries[key] = value
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps({"key": key, "value": value}) + "\n")
