"""The OpenAI-compatible chat endpoint.

Clients send model="<provider>/<model>", e.g. "groq/openai/gpt-oss-20b".
Replies and errors use the OpenAI format, so OpenAI client libraries work unchanged.
"""

import logging
import math
import time
import uuid
from collections.abc import Mapping
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hardened_llm_gateway.providers import (
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


async def gateway_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return JSONResponse(
        {"error": {"message": exc.message, "type": exc.error_type}},
        status_code=exc.status_code,
        headers=exc.headers,
    )


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


@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    providers: Annotated[Mapping[str, Provider], Depends(get_providers)],
) -> ChatCompletion:
    if body.stream:
        raise GatewayError(400, "streaming is not supported yet", "invalid_request_error")

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
