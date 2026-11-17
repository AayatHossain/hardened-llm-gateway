"""The combined model list behind GET /v1/models, cached per provider.

Provider model lists change rarely, so each one is kept for `ttl_seconds`. If a
provider fails to answer, its last known list is used (or it is left out), so one
broken provider never breaks the whole list.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping

from hardened_llm_gateway.providers import Provider, ProviderError

logger = logging.getLogger(__name__)


class ModelDirectory:
    def __init__(
        self, ttl_seconds: float = 300.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._cache: dict[str, tuple[float, list[str]]] = {}  # provider -> (fetched at, ids)

    async def model_ids(self, providers: Mapping[str, Provider]) -> list[str]:
        """Every model as "<provider>/<model>", asking all providers at the same time."""
        names = sorted(providers)
        lists = await asyncio.gather(*(self._for(name, providers[name]) for name in names))
        return [f"{name}/{model}" for name, ids in zip(names, lists, strict=True) for model in ids]

    async def _for(self, name: str, provider: Provider) -> list[str]:
        now = self._clock()
        cached = self._cache.get(name)
        if cached is not None and now - cached[0] < self._ttl:
            return cached[1]
        try:
            ids = await provider.list_models()
        except ProviderError as exc:
            logger.warning("could not list models of %s: %s", name, exc)
            return cached[1] if cached is not None else []
        self._cache[name] = (now, ids)
        return ids
