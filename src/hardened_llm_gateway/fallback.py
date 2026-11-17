"""Retries with backoff, then fallback to the next target.

A target is one "<provider>/<model>". For each target in order:

- retryable failures (timeout, network error, 429, 5xx) are retried up to
  `max_retries` times, waiting with exponential backoff and jitter, or for the
  provider's Retry-After when it is short enough;
- other provider failures (bad key, unknown model at that provider, malformed
  reply) skip straight to the next target;
- an invalid request (HTTP 400/422) stops everything, because every target would
  reject it too.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.providers import Provider, ProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)

_INVALID_REQUEST_STATUSES = (400, 422)


@dataclass(frozen=True)
class Target:
    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> Self | None:
        provider, _, model = value.partition("/")
        if not provider or not model:
            return None
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 1
    backoff_seconds: float = 0.5
    max_wait_seconds: float = 5.0

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            max_retries=settings.max_retries,
            backoff_seconds=settings.retry_backoff_seconds,
            max_wait_seconds=settings.max_retry_wait_seconds,
        )

    def delay(self, retry_number: int, error: ProviderError) -> float | None:
        """Seconds to wait before retry `retry_number` (1-based), or None to stop retrying."""
        if not error.retryable or retry_number > self.max_retries:
            return None
        if error.retry_after_seconds is not None:
            if error.retry_after_seconds > self.max_wait_seconds:
                return None  # waiting that long is worse than trying a fallback
            return error.retry_after_seconds
        backoff = min(self.backoff_seconds * 2.0 ** (retry_number - 1), self.max_wait_seconds)
        return backoff / 2 + random.uniform(0, backoff / 2)  # "equal jitter"


@dataclass(frozen=True)
class Attempt:
    target: Target
    error: str | None = None  # None means this attempt succeeded


async def run_with_fallback[T](
    targets: Sequence[Target],
    providers: Mapping[str, Provider],
    call: Callable[[Provider, Target], Awaitable[T]],
    policy: RetryPolicy,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[T, Target, list[Attempt]]:
    """Call targets in order until one succeeds.

    Returns the result, the target that produced it and every attempt made.
    If all targets fail, the last ProviderError is raised.
    """
    attempts: list[Attempt] = []
    last_error: ProviderError | None = None
    for target in targets:
        provider = providers.get(target.provider)
        if provider is None:
            attempts.append(Attempt(target, "provider not enabled"))
            continue
        retry_number = 0
        while True:
            try:
                result = await call(provider, target)
            except ProviderError as exc:
                logger.warning("attempt on %s failed: %s", target, exc)
                attempts.append(Attempt(target, _short(exc)))
                last_error = exc
                if exc.status_code in _INVALID_REQUEST_STATUSES:
                    raise
                retry_number += 1
                wait = policy.delay(retry_number, exc)
                if wait is None:
                    break
                await sleep(wait)
                continue
            attempts.append(Attempt(target))
            return result, target, attempts

    if last_error is None:
        raise ProviderError("gateway", "no enabled provider for any target")
    raise last_error


def _short(exc: ProviderError) -> str:
    if exc.status_code is not None:
        return f"HTTP {exc.status_code}"
    return "timeout" if isinstance(exc, ProviderTimeoutError) else "network error"
