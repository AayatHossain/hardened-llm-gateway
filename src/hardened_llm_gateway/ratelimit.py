"""Per-client request limits, with a token bucket kept in memory.

Each client has a bucket holding up to `requests_per_minute` tokens. A request takes
1 token; tokens come back at requests_per_minute / 60 per second. So a client can send
a short burst of up to a minute's worth of requests, and then a steady stream at its
limit.

The buckets live in this process. With several gateway processes, each would keep its
own count; a shared store (e.g. Redis) would be needed then.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: float  # 0 when allowed
    reset_seconds: float  # until the bucket is full again

    def headers(self) -> dict[str, str]:
        """The x-ratelimit-* headers OpenAI uses, so clients can pace themselves."""
        return {
            "x-ratelimit-limit-requests": str(self.limit),
            "x-ratelimit-remaining-requests": str(self.remaining),
            "x-ratelimit-reset-requests": f"{math.ceil(self.reset_seconds)}s",
        }


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, updated at)

    def check(self, client: str, requests_per_minute: int) -> RateLimitResult:
        """Take 1 token from the client's bucket if there is one."""
        now = self._clock()
        capacity = float(requests_per_minute)
        refill_per_second = capacity / 60
        tokens, updated = self._buckets.get(client, (capacity, now))
        tokens = min(capacity, tokens + (now - updated) * refill_per_second)

        allowed = tokens >= 1
        if allowed:
            tokens -= 1
        self._buckets[client] = (tokens, now)

        return RateLimitResult(
            allowed=allowed,
            limit=requests_per_minute,
            remaining=math.floor(tokens),
            retry_after_seconds=0.0 if allowed else (1 - tokens) / refill_per_second,
            reset_seconds=(capacity - tokens) / refill_per_second,
        )
