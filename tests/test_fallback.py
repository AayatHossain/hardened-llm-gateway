"""Tests for retries and fallback, with scripted fake providers and no real waiting."""

from collections.abc import AsyncGenerator

import pytest

from hardened_llm_gateway.fallback import Attempt, RetryPolicy, Target, run_with_fallback
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

pytestmark = pytest.mark.anyio


class ScriptedProvider:
    """Fails with the scripted errors in order, then answers."""

    def __init__(self, name: str, errors: list[ProviderError] | None = None) -> None:
        self._name = name
        self.errors = list(errors or [])
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ChatResponse(
            id="x",
            provider=self._name,
            model=request.model,
            content=f"from {self._name}",
            finish_reason="stop",
            usage=Usage(),
            latency_ms=1.0,
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        raise NotImplementedError
        yield  # makes this an async generator

    async def list_models(self) -> list[str]:
        raise NotImplementedError


def server_error(name: str = "a") -> ProviderError:
    return ProviderError(name, "HTTP 503", status_code=503, retryable=True)


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


A = Target("a", "m1")
B = Target("b", "m2")
POLICY = RetryPolicy(max_retries=1, backoff_seconds=0.5, max_wait_seconds=5.0)


async def run(
    providers: dict[str, ScriptedProvider],
    targets: list[Target],
    policy: RetryPolicy = POLICY,
) -> tuple[str, Target, int, Sleeps]:
    sleeps = Sleeps()
    as_providers: dict[str, Provider] = dict(providers)

    async def call(provider: Provider, target: Target) -> ChatResponse:
        message = ChatMessage(role="user", content="hi")
        return await provider.chat(ChatRequest(model=target.model, messages=[message]))

    reply, target, attempts = await run_with_fallback(
        targets, as_providers, call, policy, sleep=sleeps
    )
    return reply.content, target, len(attempts), sleeps


async def test_first_target_answers() -> None:
    content, target, attempts, sleeps = await run({"a": ScriptedProvider("a")}, [A, B])
    assert (content, target, attempts) == ("from a", A, 1)
    assert sleeps.delays == []


async def test_retryable_error_is_retried_with_backoff() -> None:
    a = ScriptedProvider("a", [server_error()])
    content, target, attempts, sleeps = await run({"a": a}, [A])
    assert (content, target, attempts) == ("from a", A, 2)
    assert len(sleeps.delays) == 1
    assert 0.25 <= sleeps.delays[0] <= 0.5  # equal jitter on a 0.5 s backoff


async def test_falls_back_after_retries_run_out() -> None:
    a = ScriptedProvider("a", [server_error(), server_error()])
    b = ScriptedProvider("b")
    content, target, attempts, _ = await run({"a": a, "b": b}, [A, B])
    assert (content, target, attempts) == ("from b", B, 3)
    assert a.calls == 2


async def test_short_retry_after_is_respected() -> None:
    error = ProviderError("a", "HTTP 429", status_code=429, retryable=True, retry_after_seconds=2)
    _, target, _, sleeps = await run({"a": ScriptedProvider("a", [error])}, [A])
    assert target == A
    assert sleeps.delays == [2]


async def test_long_retry_after_skips_to_fallback() -> None:
    error = ProviderError("a", "HTTP 429", status_code=429, retryable=True, retry_after_seconds=60)
    a = ScriptedProvider("a", [error])
    _, target, _, sleeps = await run({"a": a, "b": ScriptedProvider("b")}, [A, B])
    assert target == B
    assert a.calls == 1
    assert sleeps.delays == []


async def test_non_retryable_error_falls_back_without_retry() -> None:
    a = ScriptedProvider("a", [ProviderError("a", "HTTP 401", status_code=401)])
    _, target, _, sleeps = await run({"a": a, "b": ScriptedProvider("b")}, [A, B])
    assert target == B
    assert a.calls == 1
    assert sleeps.delays == []


async def test_invalid_request_stops_without_fallback() -> None:
    a = ScriptedProvider("a", [ProviderError("a", "HTTP 400", status_code=400)])
    b = ScriptedProvider("b")
    with pytest.raises(ProviderError) as info:
        await run({"a": a, "b": b}, [A, B])
    assert info.value.status_code == 400
    assert b.calls == 0


async def test_disabled_fallback_is_skipped() -> None:
    a = ScriptedProvider("a", [server_error(), server_error()])
    c = ScriptedProvider("c")
    _, target, attempts, _ = await run({"a": a, "c": c}, [A, B, Target("c", "m3")])
    assert target == Target("c", "m3")
    assert attempts == 4  # a, a again, b skipped, c


async def test_all_targets_failing_raises_the_last_error() -> None:
    a = ScriptedProvider("a", [server_error("a"), server_error("a")])
    b = ScriptedProvider("b", [ProviderTimeoutError("b"), ProviderTimeoutError("b")])
    with pytest.raises(ProviderTimeoutError):
        await run({"a": a, "b": b}, [A, B])


async def test_zero_retries_means_one_attempt_per_target() -> None:
    a = ScriptedProvider("a", [server_error()])
    policy = RetryPolicy(max_retries=0)
    _, target, _, sleeps = await run({"a": a, "b": ScriptedProvider("b")}, [A, B], policy)
    assert target == B
    assert a.calls == 1
    assert sleeps.delays == []


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(max_retries=5, backoff_seconds=1.0, max_wait_seconds=3.0)
    error = server_error()
    for retry_number, upper in [(1, 1.0), (2, 2.0), (3, 3.0), (4, 3.0)]:
        delay = policy.delay(retry_number, error)
        assert delay is not None
        assert upper / 2 <= delay <= upper


@pytest.mark.parametrize(
    ("value", "expected"),
    [("groq/openai/gpt-oss-20b", Target("groq", "openai/gpt-oss-20b")), ("groq", None)],
)
def test_target_parse(value: str, expected: Target | None) -> None:
    assert Target.parse(value) == expected


async def test_attempts_list_is_filled_even_when_everything_fails() -> None:
    a = ScriptedProvider("a", [ProviderError("a", "HTTP 401", status_code=401)])
    attempts: list[Attempt] = []

    async def call(provider: Provider, target: Target) -> ChatResponse:
        message = ChatMessage(role="user", content="hi")
        return await provider.chat(ChatRequest(model=target.model, messages=[message]))

    with pytest.raises(ProviderError):
        await run_with_fallback([A], {"a": a}, call, POLICY, attempts=attempts)
    assert attempts == [Attempt(A, "HTTP 401")]
