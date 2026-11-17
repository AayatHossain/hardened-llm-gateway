"""Client API keys: create them, hash them and look them up.

Run `uv run python -m hardened_llm_gateway.keys <client-name>` to create a key. The key
is printed once for the client app; only its SHA-256 goes into GATEWAY_CLIENTS.

A plain SHA-256 is enough here (no slow password hash) because the keys are long and
random: there is nothing to guess from a word list.
"""

import hashlib
import hmac
import json
import secrets
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from hardened_llm_gateway.config import ClientConfig

KEY_PREFIX = "gw-"


@dataclass(frozen=True)
class Client:
    name: str
    requests_per_minute: int


def new_key() -> str:
    """A new random client key, e.g. "gw-3q2-…" (256 bits of randomness)."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def find_client(key: str, clients: Mapping[str, ClientConfig]) -> Client | None:
    """The client that owns `key`, or None.

    Every stored hash is compared in constant time, so response timing does not reveal
    how much of a guessed key's hash matched.
    """
    digest = hash_key(key)
    found = None
    for name, config in clients.items():
        if hmac.compare_digest(digest, config.key_sha256):
            found = Client(name, config.requests_per_minute)
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m hardened_llm_gateway.keys <client-name>", file=sys.stderr)
        return 2
    name = args[0]
    key = new_key()
    entry = json.dumps({name: {"key_sha256": hash_key(key), "requests_per_minute": 60}})
    print(f"Key for {name!r} (shown only once, give it to the app):\n  {key}\n")
    print(f"Add this entry to GATEWAY_CLIENTS in .env:\n  {entry[1:-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
