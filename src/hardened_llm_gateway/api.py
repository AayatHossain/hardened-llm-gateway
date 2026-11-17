"""The OpenAI-compatible endpoints: POST /v1/chat/completions and GET /v1/models.

Clients send model="<provider>/<model>", e.g. "groq/openai/gpt-oss-20b".
Replies and errors use the OpenAI format, so OpenAI client libraries work unchanged,
including streaming (stream=true returns server-sent events).

Every /v1 request needs a client key ("Authorization: Bearer <key>", see keys.py) and
counts against that client's requests-per-minute limit (see ratelimit.py).
Each request is retried and, if configured, sent to fallback targets when the
requested provider keeps failing (see fallback.py).
"""

import dataclasses
import datetime
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.fallback import Attempt, RetryPolicy, Target, run_with_fallback
from hardened_llm_gateway.keys import Client, find_client
from hardened_llm_gateway.models import ModelDirectory
from hardened_llm_gateway.pricing import PriceTable
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)
from hardened_llm_gateway.ratelimit import RateLimiter, RateLimitResult
from hardened_llm_gateway.requestlog import RequestRecord, RequestRecorder

logger = logging.getLogger(__name__)


class ChatCompletionRequest(ChatRequest):
    """The OpenAI chat request. `model` is "<provider>/<model>"."""

    stream: bool = False


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0  # not every provider reports it
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class GatewayError(Exception):
    """An error returned to the client in the OpenAI error format."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        headers: dict[str, str] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.headers = headers
        self.code = code

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"message": self.message, "type": self.error_type}
        if self.code is not None:
            error["code"] = self.code
        return {"error": error}


async def gateway_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return JSONResponse(exc.body(), status_code=exc.status_code, headers=exc.headers)


def get_providers(request: Request) -> Mapping[str, Provider]:
    providers: Mapping[str, Provider] = request.app.state.providers
    return providers


def get_gateway_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


# auto_error=False: a missing key gets our OpenAI-style error, not FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False, description="A client key made with the keys command")


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_gateway_settings)],
) -> Client:
    """The client calling the gateway, identified by its "Authorization: Bearer" key."""
    if credentials is None:
        message = "missing API key: send the header 'Authorization: Bearer <key>'"
        client = None
    else:
        message = "invalid API key"
        client = find_client(credentials.credentials, settings.clients)
    if client is None:
        raise GatewayError(
            401,
            message,
            "invalid_request_error",
            headers={"www-authenticate": "Bearer"},
            code="invalid_api_key",
        )
    return client


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def enforce_rate_limit(
    client: Annotated[Client, Depends(authenticate)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> RateLimitResult:
    """Count the request against the client's per-minute limit, or answer 429."""
    result = limiter.check(client.name, client.requests_per_minute)
    if not result.allowed:
        raise GatewayError(
            429,
            f"rate limit reached for client {client.name!r}: "
            f"{client.requests_per_minute} requests per minute",
            "rate_limit_error",
            headers={
                **result.headers(),
                "retry-after": str(math.ceil(result.retry_after_seconds)),
            },
            code="client_rate_limit_exceeded",
        )
    return result


def get_model_directory(request: Request) -> ModelDirectory:
    directory: ModelDirectory = request.app.state.model_directory
    return directory


def get_prices(request: Request) -> PriceTable:
    prices: PriceTable = request.app.state.prices
    return prices


def _format_cost(cost: float | None) -> str:
    """USD with up to 8 decimals, e.g. "0.000225"; "unknown" when the model has no price."""
    if cost is None:
        return "unknown"
    return f"{cost:.8f}".rstrip("0").rstrip(".")


# Every /v1 route needs a client key and counts against the client's limit. Routes that
# also take `limits` get the same result: FastAPI runs a dependency once per request.
router = APIRouter(dependencies=[Depends(enforce_rate_limit)])


@router.get("/v1/models", response_model=ModelList)
async def list_models(
    response: Response,
    providers: Annotated[Mapping[str, Provider], Depends(get_providers)],
    directory: Annotated[ModelDirectory, Depends(get_model_directory)],
    limits: Annotated[RateLimitResult, Depends(enforce_rate_limit)],
) -> ModelList:
    """The chat models of every enabled provider, named as /v1/chat/completions expects."""
    response.headers.update(limits.headers())
    ids = await directory.model_ids(providers)
    return ModelList(data=[ModelCard(id=i, owned_by=i.partition("/")[0]) for i in ids])


def _to_gateway_error(exc: ProviderError) -> GatewayError:
    """Map a provider failure to what the client sees. Details stay in the server log."""
    name = exc.provider
    if isinstance(exc, ProviderTimeoutError):
        return GatewayError(504, f"provider {name!r} did not answer in time", "provider_timeout")
    if exc.status_code == 429:
        headers = None
        if exc.retry_after_seconds is not None:
            headers = {"retry-after": str(math.ceil(exc.retry_after_seconds))}
        return GatewayError(
            429, f"provider {name!r} rate limit reached", "rate_limit_error", headers
        )
    if exc.status_code in (400, 404, 422):
        # The request itself was wrong (e.g. an unknown model): pass the provider's reason on.
        reason = exc.reason or f"HTTP {exc.status_code}"
        return GatewayError(
            400, f"provider {name!r} rejected the request: {reason}", "invalid_request_error"
        )
    return GatewayError(502, f"provider {name!r} failed", "provider_error")


def _targets(
    body: ChatCompletionRequest, providers: Mapping[str, Provider], settings: Settings
) -> list[Target]:
    """The requested target first, then its configured fallbacks."""
    primary = Target.parse(body.model)
    if primary is None:
        raise GatewayError(
            400,
            'model must be "<provider>/<model>", e.g. "groq/openai/gpt-oss-20b"',
            "invalid_request_error",
        )
    if primary.provider not in providers:
        enabled = ", ".join(sorted(providers)) or "none"
        raise GatewayError(
            400,
            f"provider {primary.provider!r} is unknown or has no API key (enabled: {enabled})",
            "invalid_request_error",
        )
    backups = [Target.parse(value) for value in settings.fallbacks.get(body.model, [])]
    return [primary, *(t for t in backups if t is not None and t != primary)]


def _decision_headers(target: Target, attempts: list[Attempt]) -> dict[str, str]:
    """Headers that explain which target answered and how many tries it took."""
    return {
        "x-gateway-provider": target.provider,
        "x-gateway-target": str(target),
        "x-gateway-attempts": str(len(attempts)),
    }


def get_request_log(request: Request) -> RequestRecorder:
    request_log: RequestRecorder = request.app.state.request_log
    return request_log


@dataclasses.dataclass
class _Trace:
    """What the request log needs, collected while 1 chat request runs."""

    log: RequestRecorder
    client: str
    requested_model: str
    stream: bool
    id: str = dataclasses.field(default_factory=lambda: f"req-{uuid.uuid4().hex}")
    created_at: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    started: float = dataclasses.field(default_factory=time.perf_counter)
    attempts: list[Attempt] = dataclasses.field(default_factory=list)  # filled by the fallback

    def finish(
        self,
        status_code: int,
        *,
        error_type: str | None = None,
        usage: Usage | None = None,
        cost_usd: float | None = None,
        first_chunk_ms: float | None = None,
    ) -> None:
        last = self.attempts[-1].target if self.attempts else None
        self.log.record(
            RequestRecord(
                id=self.id,
                created_at=self.created_at,
                client=self.client,
                requested_model=self.requested_model,
                target=str(last) if last else None,
                provider=last.provider if last else None,
                stream=self.stream,
                status_code=status_code,
                error_type=error_type,
                attempts=len(self.attempts),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                cost_usd=cost_usd,
                latency_ms=(time.perf_counter() - self.started) * 1000,
                first_chunk_ms=first_chunk_ms,
            )
        )


@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    providers: Annotated[Mapping[str, Provider], Depends(get_providers)],
    settings: Annotated[Settings, Depends(get_gateway_settings)],
    limits: Annotated[RateLimitResult, Depends(enforce_rate_limit)],
    prices: Annotated[PriceTable, Depends(get_prices)],
    client: Annotated[Client, Depends(authenticate)],
    request_log: Annotated[RequestRecorder, Depends(get_request_log)],
) -> ChatCompletion | StreamingResponse:
    """Every request that gets past the key check and rate limit is written to the log."""
    trace = _Trace(request_log, client.name, body.model, body.stream)
    try:
        targets = _targets(body, providers, settings)
        base = ChatRequest.model_validate(body.model_dump(exclude={"stream"}))
        policy = RetryPolicy.from_settings(settings)

        def request_for(target: Target) -> ChatRequest:
            return base.model_copy(update={"model": target.model})

        if body.stream:
            headers = {**limits.headers(), "x-gateway-request-id": trace.id}
            return await _stream(targets, providers, policy, request_for, headers, prices, trace)

        async def call(provider: Provider, target: Target) -> ChatResponse:
            return await provider.chat(request_for(target))

        try:
            reply, target, attempts = await run_with_fallback(
                targets, providers, call, policy, attempts=trace.attempts
            )
        except ProviderError as exc:
            raise _to_gateway_error(exc) from None
    except GatewayError as exc:
        trace.finish(exc.status_code, error_type=exc.error_type)
        exc.headers = {**(exc.headers or {}), "x-gateway-request-id": trace.id}
        raise

    cost = prices.cost_usd(str(target), reply.usage)
    trace.finish(200, usage=reply.usage, cost_usd=cost)

    response.headers.update(limits.headers())
    response.headers.update(_decision_headers(target, attempts))
    response.headers["x-gateway-request-id"] = trace.id
    response.headers["x-gateway-latency-ms"] = f"{reply.latency_ms:.0f}"
    response.headers["x-gateway-cost-usd"] = _format_cost(cost)
    return ChatCompletion(
        id=reply.id or f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=f"{reply.provider}/{reply.model}",
        choices=[
            Choice(
                message=AssistantMessage(content=reply.content),
                finish_reason=reply.finish_reason,
            )
        ],
        usage=reply.usage,
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


type _OpenedStream = tuple[AsyncGenerator[ChatChunk], ChatChunk | None]


async def _stream(
    targets: list[Target],
    providers: Mapping[str, Provider],
    policy: RetryPolicy,
    request_for: Callable[[Target], ChatRequest],
    extra_headers: dict[str, str],
    prices: PriceTable,
    trace: _Trace,
) -> StreamingResponse:
    """Stream the reply as OpenAI chat.completion.chunk events, ending with [DONE].

    The cost is only known at the end, after the headers were sent, so it travels in
    the event that carries `usage`, as "x_gateway": {"cost_usd": ...} (like Groq's x_groq).

    Opening a stream means waiting for its first chunk. Retries and fallback apply
    until then, so failures before any text (rate limit, timeout, bad model) are
    handled like non-streamed calls. A failure after that can only be reported as an
    error event inside the stream.
    """
    started = time.perf_counter()

    async def open_stream(provider: Provider, target: Target) -> _OpenedStream:
        chunks = provider.stream(request_for(target))
        try:
            return chunks, await anext(chunks)
        except StopAsyncIteration:
            return chunks, None
        except BaseException:
            await chunks.aclose()
            raise

    try:
        (chunks, first), target, attempts = await run_with_fallback(
            targets, providers, open_stream, policy, attempts=trace.attempts
        )
    except ProviderError as exc:
        raise _to_gateway_error(exc) from None
    first_chunk_ms = (time.perf_counter() - started) * 1000

    completion_id = (first.id if first else "") or f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def event(chunk: ChatChunk, with_role: bool) -> str:
        delta: dict[str, str] = {"content": chunk.delta}
        if with_role:
            delta["role"] = "assistant"
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(target),
            "choices": [{"index": 0, "delta": delta, "finish_reason": chunk.finish_reason}],
        }
        if chunk.usage is not None:
            payload["usage"] = chunk.usage.model_dump()
            cost = prices.cost_usd(str(target), chunk.usage)
            payload["x_gateway"] = {"cost_usd": cost}
        return _sse(payload)

    async def events() -> AsyncIterator[str]:
        usage = first.usage if first else None
        error_type: str | None = "client_disconnected"  # until the reply ends normally
        try:
            if first is not None:
                yield event(first, with_role=True)
            async for chunk in chunks:
                usage = chunk.usage or usage
                yield event(chunk, with_role=False)
            error_type = None
        except ProviderError as exc:
            logger.warning("provider stream failed mid-reply: %s", exc)
            failure = _to_gateway_error(exc)
            error_type = failure.error_type
            yield _sse(failure.body())
            return
        finally:
            await chunks.aclose()  # also runs when the client disconnects early
            # The client already got status 200 with the headers, whatever happened next.
            trace.finish(
                200,
                error_type=error_type,
                usage=usage,
                cost_usd=prices.cost_usd(str(target), usage) if usage else None,
                first_chunk_ms=first_chunk_ms,
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            **extra_headers,
            **_decision_headers(target, attempts),
            "x-gateway-first-chunk-ms": f"{first_chunk_ms:.0f}",
            "cache-control": "no-cache",
        },
    )
