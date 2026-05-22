"""Factory helpers for LLM clients."""

from config import DEFAULT_LLM_PROVIDER
from llm.base import BaseLLMClient
from llm.deepseek_client import DeepSeekClient
from llm.mock_client import MockLLMClient


def get_llm_client(
    provider_name: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
) -> BaseLLMClient:
    """Create an LLM client by provider name."""

    normalized_name = provider_name.lower().strip()
    if normalized_name == "mock":
        return MockLLMClient(model=model or "mock-llm")
    if normalized_name == "deepseek":
        return DeepSeekClient(model=model)
    raise ValueError(
        f"Unsupported LLM provider '{provider_name}'. "
        "Available providers: mock, deepseek."
    )

