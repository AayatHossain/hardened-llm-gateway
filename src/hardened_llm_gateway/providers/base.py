"""The interface every model provider implements."""

from typing import Protocol

from hardened_llm_gateway.providers.types import ChatRequest, ChatResponse


class Provider(Protocol):
    """Something that can answer a chat request.

    Implementations raise ProviderError on failure and never return partial replies.
    """

    @property
    def name(self) -> str: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...
