"""One adapter for every provider that speaks the OpenAI chat format.

Groq, Gemini and Cohere (their OpenAI-compatible endpoints), NVIDIA NIM, OpenRouter
and Z.ai all accept POST {base_url}/chat/completions and GET {base_url}/models with a
Bearer key, so they share this code and differ only in base URL and key.
"""

import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from hardened_llm_gateway.providers.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

_ERROR_BODY_LIMIT = 200


class _Message(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Completion(BaseModel):
    """The parts of an OpenAI-format reply the gateway uses; extra fields are ignored."""

    id: str = ""
    model: str
    choices: list[_Choice] = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)


class _Delta(BaseModel):
    content: str | None = None


class _ChunkChoice(BaseModel):
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _GroqExtra(BaseModel):
    usage: Usage | None = None


class _CompletionChunk(BaseModel):
    """One server-sent event of a streamed OpenAI-format reply."""

    id: str = ""
    model: str = ""
    choices: list[_ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None
    x_groq: _GroqExtra | None = None  # Groq reports usage here on the last chunk


class _ModelEntry(BaseModel):
    id: str
    output_modalities: list[str] | None = None  # Groq says this; others leave it out


class _ModelList(BaseModel):
    data: list[_ModelEntry]


def _chat_model_ids(models: _ModelList) -> list[str]:
    """Model ids that can chat, sorted.

    Speech and transcription models are dropped when the provider says what a model
    outputs. Gemini's "models/" prefix is removed: its chat endpoint takes the bare name.
    """
    ids = {
        model.id.removeprefix("models/")
        for model in models.data
        if model.output_modalities is None or "text" in model.output_modalities
    }
    return sorted(ids)


def _billed(usage: Usage) -> Usage:
    """Usage with hidden reasoning tokens counted as output, as they are billed.

    Gemini leaves its thinking tokens out of completion_tokens but keeps them in
    total_tokens, so a reply looks several times cheaper than it is. Providers that
    already count reasoning as output (Groq) have nothing left over and are unchanged.
    """
    hidden = usage.total_tokens - usage.prompt_tokens - usage.completion_tokens
    if hidden <= 0:
        return usage
    return usage.model_copy(update={"completion_tokens": usage.completion_tokens + hidden})


def _error_reason(response: httpx.Response) -> str | None:
    """The provider's error message, from {"error": {"message": ...}}.

    Gemini wraps the same object in a 1-item list. Returns None for any other body.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return None
    if isinstance(body, list) and body:
        body = body[0]
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message.strip():
        return None
    return message.strip()[:_ERROR_BODY_LIMIT]


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: SecretStr,
        client: httpx.AsyncClient,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._models_url = base_url.rstrip("/") + "/models"
        self._api_key = api_key
        self._client = client
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self._name!r}, url={self._url!r})"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    def _http_error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        reason = _error_reason(response)
        return ProviderError(
            self._name,
            f"HTTP {status}: {reason or response.text[:_ERROR_BODY_LIMIT]}",
            status_code=status,
            retryable=status == 429 or status >= 500,
            retry_after_seconds=_retry_after(response),
            reason=reason,
        )

    def _network_error(self, exc: httpx.TransportError) -> ProviderError:
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(self._name)
        return ProviderError(self._name, f"network error: {type(exc).__name__}", retryable=True)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = request.model_dump(exclude_none=True)
        started = time.perf_counter()
        try:
            response = await self._client.post(
                self._url, json=payload, headers=self._headers(), timeout=self._timeout
            )
        except httpx.TransportError as exc:
            raise self._network_error(exc) from None
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise self._http_error(response)

        try:
            completion = _Completion.model_validate_json(response.content)
        except ValidationError:
            raise ProviderError(
                self._name, "reply is not in the OpenAI chat format", status_code=200
            ) from None

        choice = completion.choices[0]
        return ChatResponse(
            id=completion.id,
            provider=self._name,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=_billed(completion.usage),
            latency_ms=latency_ms,
        )

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get(
                self._models_url, headers=self._headers(), timeout=self._timeout
            )
        except httpx.TransportError as exc:
            raise self._network_error(exc) from None
        if response.status_code != 200:
            raise self._http_error(response)
        try:
            models = _ModelList.model_validate_json(response.content)
        except ValidationError:
            raise ProviderError(
                self._name, "model list is not in the OpenAI format", status_code=200
            ) from None
        return _chat_model_ids(models)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the reply piece by piece, parsed from the provider's server-sent events."""
        payload: dict[str, Any] = request.model_dump(exclude_none=True)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        try:
            async with self._client.stream(
                "POST", self._url, json=payload, headers=self._headers(), timeout=self._timeout
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise self._http_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators and ": keep-alive" comments
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return
                    yield self._parse_chunk(data)
        except httpx.TransportError as exc:
            raise self._network_error(exc) from None

    def _parse_chunk(self, data: str) -> ChatChunk:
        try:
            event = _CompletionChunk.model_validate_json(data)
        except ValidationError:
            raise ProviderError(
                self._name, "stream event is not in the OpenAI chunk format", status_code=200
            ) from None
        choice = event.choices[0] if event.choices else _ChunkChoice()
        usage = event.usage or (event.x_groq.usage if event.x_groq else None)
        return ChatChunk(
            id=event.id,
            provider=self._name,
            model=event.model,
            delta=choice.delta.content or "",
            finish_reason=choice.finish_reason,
            usage=_billed(usage) if usage else None,
        )
