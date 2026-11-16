"""Request and reply types shared by every provider.

The rest of the gateway only sees these types, so it never needs to know
which provider answered.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """A chat request in the OpenAI format."""

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """A provider's reply, plus what the gateway measured about the call."""

    id: str
    provider: str
    model: str
    content: str
    finish_reason: str | None
    usage: Usage
    latency_ms: float


class ProviderError(Exception):
    """A provider call failed.

    `retryable` is True when trying again (or trying another provider) may work:
    timeouts, network errors, rate limits (429) and server errors (5xx).
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
