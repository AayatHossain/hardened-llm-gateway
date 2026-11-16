"""One adapter for every provider that speaks the OpenAI chat format.

Groq, Gemini (OpenAI-compatible endpoint), NVIDIA NIM and OpenRouter all accept
POST {base_url}/chat/completions with a Bearer key, so they share this code and
differ only in base URL and key.
"""

import time

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from hardened_llm_gateway.providers.types import (
    ChatRequest,
    ChatResponse,
    ProviderError,
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
        self._api_key = api_key
        self._client = client
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self._name!r}, url={self._url!r})"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        payload = request.model_dump(exclude_none=True)
        started = time.perf_counter()
        try:
            response = await self._client.post(
                self._url, json=payload, headers=headers, timeout=self._timeout
            )
        except httpx.TimeoutException:
            raise ProviderError(self._name, "request timed out", retryable=True) from None
        except httpx.TransportError as exc:
            raise ProviderError(
                self._name, f"network error: {type(exc).__name__}", retryable=True
            ) from None
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            status = response.status_code
            raise ProviderError(
                self._name,
                f"HTTP {status}: {response.text[:_ERROR_BODY_LIMIT]}",
                status_code=status,
                retryable=status == 429 or status >= 500,
                retry_after_seconds=_retry_after(response),
            )

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
            usage=completion.usage,
            latency_ms=latency_ms,
        )
