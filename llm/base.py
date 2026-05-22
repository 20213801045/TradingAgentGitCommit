"""Base LLM client interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    """A chat message sent to an LLM provider."""

    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """A normalized response from an LLM provider."""

    content: str
    model: str
    provider: str
    raw: dict[str, Any] | None = None


class LLMError(RuntimeError):
    """Raised when an LLM provider request fails."""


class BaseLLMClient(ABC):
    """Abstract interface for LLM chat-completion clients."""

    provider: str
    model: str

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response for a list of chat messages."""
