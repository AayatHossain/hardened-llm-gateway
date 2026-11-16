"""The OpenAI-compatible chat endpoint.

Clients send model="<provider>/<model>", e.g. "groq/openai/gpt-oss-20b".
Replies and errors use the OpenAI format, so OpenAI client libraries work unchanged,
including streaming (stream=true returns server-sent events).
"""

import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.headers = headers

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.error_type}}


async def gateway_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return JSONResponse(exc.body(), status_code=exc.status_code, headers=exc.headers)


def get_providers(request: Request) -> Mapping[str, Provider]:
    providers: Mapping[str, Provider] = request.app.state.providers
    return providers


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


def _resolve(
    body: ChatCompletionRequest, providers: Mapping[str, Provider]
) -> tuple[Provider, ChatRequest]:
    """Split "<provider>/<model>" and return the provider plus the request it should get."""
    provider_name, _, model = body.model.partition("/")
    if not model:
        raise GatewayError(
            400,
            'model must be "<provider>/<model>", e.g. "groq/openai/gpt-oss-20b"',
            "invalid_request_error",
        )
    provider = providers.get(provider_name)
    if provider is None:
        enabled = ", ".join(sorted(providers)) or "none"
        raise GatewayError(
            400,
            f"provider {provider_name!r} is unknown or has no API key (enabled: {enabled})",
            "invalid_request_error",
        )
    request = ChatRequest.model_validate({**body.model_dump(exclude={"stream"}), "model": model})
    return provider, request


@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    providers: Annotated[Mapping[str, Provider], Depends(get_providers)],
) -> ChatCompletion | StreamingResponse:
    provider, request = _resolve(body, providers)
    if body.stream:
        return await _stream(provider, request, body.model)

    try:
        reply = await provider.chat(request)
    except ProviderError as exc:
        logger.warning("provider call failed: %s", exc)
        raise _to_gateway_error(exc) from None

    response.headers["x-gateway-provider"] = reply.provider
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


async def _stream(provider: Provider, request: ChatRequest, model: str) -> StreamingResponse:
    """Stream the reply as OpenAI chat.completion.chunk events, ending with [DONE].

    The first chunk is awaited before responding, so failures that happen before any
    text (rate limit, timeout, bad model) still get a proper HTTP status. A failure after
    that can only be reported as an error event inside the stream.
    """
    started = time.perf_counter()
    chunks = provider.stream(request)
    first: ChatChunk | None
    try:
        first = await anext(chunks)
    except StopAsyncIteration:
        first = None
    except ProviderError as exc:
        logger.warning("provider stream failed before the first chunk: %s", exc)
        await chunks.aclose()
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
            "model": model,
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
            "x-gateway-provider": provider.name,
            "x-gateway-first-chunk-ms": f"{first_chunk_ms:.0f}",
            "cache-control": "no-cache",
        },
    )
