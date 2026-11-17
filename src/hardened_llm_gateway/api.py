"""The OpenAI-compatible chat endpoint.

Clients send model="<provider>/<model>", e.g. "groq/openai/gpt-oss-20b".
Replies and errors use the OpenAI format, so OpenAI client libraries work unchanged,
including streaming (stream=true returns server-sent events).

Every /v1 request needs a client key ("Authorization: Bearer <key>", see keys.py) and
counts against that client's requests-per-minute limit (see ratelimit.py).
Each request is retried and, if configured, sent to fallback targets when the
requested provider keeps failing (see fallback.py).
"""

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


router = APIRouter(dependencies=[Depends(authenticate)])


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
        # The request itself was wrong (e.g. an unknown model): pass the reason on.
        return GatewayError(400, str(exc), "invalid_request_error")
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


@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    providers: Annotated[Mapping[str, Provider], Depends(get_providers)],
    settings: Annotated[Settings, Depends(get_gateway_settings)],
    limits: Annotated[RateLimitResult, Depends(enforce_rate_limit)],
) -> ChatCompletion | StreamingResponse:
    targets = _targets(body, providers, settings)
    base = ChatRequest.model_validate(body.model_dump(exclude={"stream"}))
    policy = RetryPolicy.from_settings(settings)

    def request_for(target: Target) -> ChatRequest:
        return base.model_copy(update={"model": target.model})

    if body.stream:
        return await _stream(targets, providers, policy, request_for, limits.headers())

    async def call(provider: Provider, target: Target) -> ChatResponse:
        return await provider.chat(request_for(target))

    try:
        reply, target, attempts = await run_with_fallback(targets, providers, call, policy)
    except ProviderError as exc:
        raise _to_gateway_error(exc) from None

    response.headers.update(limits.headers())
    response.headers.update(_decision_headers(target, attempts))
    response.headers["x-gateway-latency-ms"] = f"{reply.latency_ms:.0f}"
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
) -> StreamingResponse:
    """Stream the reply as OpenAI chat.completion.chunk events, ending with [DONE].

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
            targets, providers, open_stream, policy
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
        return _sse(payload)

    async def events() -> AsyncIterator[str]:
        try:
            if first is not None:
                yield event(first, with_role=True)
            async for chunk in chunks:
                yield event(chunk, with_role=False)
        except ProviderError as exc:
            logger.warning("provider stream failed mid-reply: %s", exc)
            yield _sse(_to_gateway_error(exc).body())
            return
        finally:
            await chunks.aclose()  # also runs when the client disconnects early
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
