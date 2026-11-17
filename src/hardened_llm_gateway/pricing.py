"""What a reply would cost on the providers' paid tiers, from the price table in prices.toml.

Token counts come from the provider's own `usage` report, so they match its tokenizer
exactly. The gateway does not count tokens itself.
"""

import datetime
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

from hardened_llm_gateway.providers import Usage

_PER_TOKENS = 1_000_000


class Rate(BaseModel):
    """USD per 1 million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class PriceChange(Rate):
    starts: datetime.date = Field(alias="from")


class ModelPrice(Rate):
    changes: list[PriceChange] = Field(default_factory=list)

    def rate_on(self, day: datetime.date) -> Rate:
        """The price in force on `day`: the latest change that has started, or the base price."""
        started = [c for c in self.changes if c.starts <= day]
        if not started:
            return self
        return max(started, key=lambda change: change.starts)


class _PriceFile(BaseModel):
    models: dict[str, ModelPrice]


class PriceTable:
    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Read a price file; by default the prices.toml shipped with the gateway."""
        if path is None:
            shipped = resources.files("hardened_llm_gateway").joinpath("prices.toml")
            text = shipped.read_text(encoding="utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        return cls(_PriceFile.model_validate(tomllib.loads(text)).models)

    def __contains__(self, target: str) -> bool:
        return target in self._prices

    def cost_usd(self, target: str, usage: Usage, day: datetime.date | None = None) -> float | None:
        """Cost of one reply from `target` ("<provider>/<model>"), or None if not priced."""
        price = self._prices.get(target)
        if price is None:
            return None
        rate = price.rate_on(day or datetime.datetime.now(datetime.UTC).date())
        return (
            usage.prompt_tokens * rate.input + usage.completion_tokens * rate.output
        ) / _PER_TOKENS
