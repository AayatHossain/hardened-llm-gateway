"""The interface every model provider implements."""

from collections.abc import AsyncGenerator
from typing import Protocol

from hardened_llm_gateway.providers.types import ChatChunk, ChatRequest, ChatResponse


class Provider(Protocol):
    """Something that can answer a chat request, whole or streamed.

    Implementations raise ProviderError on failure and never return partial replies
    from `chat`. `stream` may fail after some chunks were already yielded.
    """

    @property
    def name(self) -> str: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]: ...
