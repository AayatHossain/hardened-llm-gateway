"""Tests for the token-bucket rate limiter, with a fake clock (no real waiting)."""

import pytest

from hardened_llm_gateway.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_allows_a_burst_up_to_the_limit_then_refuses(clock: FakeClock) -> None:
    limiter = RateLimiter(clock)
    results = [limiter.check("app", 3) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    # 3 per minute refills 1 token every 20 seconds.
    assert results[-1].retry_after_seconds == pytest.approx(20)


def test_tokens_come_back_over_time(clock: FakeClock) -> None:
    limiter = RateLimiter(clock)
    for _ in range(3):
        limiter.check("app", 3)
    clock.now += 19
    assert not limiter.check("app", 3).allowed
    clock.now += 1
    assert limiter.check("app", 3).allowed


def test_bucket_never_holds_more_than_the_limit(clock: FakeClock) -> None:
    limiter = RateLimiter(clock)
    limiter.check("app", 3)
    clock.now += 3600  # a long quiet time does not build up extra requests
    results = [limiter.check("app", 3) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]


def test_clients_have_separate_buckets(clock: FakeClock) -> None:
    limiter = RateLimiter(clock)
    assert limiter.check("a", 1).allowed
    assert not limiter.check("a", 1).allowed
    assert limiter.check("b", 1).allowed


def test_headers(clock: FakeClock) -> None:
    result = RateLimiter(clock).check("app", 60)
    assert result.headers() == {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "59",
        "x-ratelimit-reset-requests": "1s",
    }
