"""Tests for LLM client abstractions."""

import json

import pytest

from llm import (
    BaseLLMClient,
    CachedLLMClient,
    LLMError,
    LLMMessage,
    LLMResponse,
    MockLLMClient,
    get_llm_client,
    parse_json_response,
)
from llm.deepseek_client import DeepSeekClient


def test_mock_llm_client_returns_deterministic_question() -> None:
    """Mock LLM should return a deterministic local response."""

    client = MockLLMClient()
    response = client.complete(
        [
            LLMMessage(role="user", content="Revenue growth looks positive."),
        ]
    )

    assert response.provider == "mock"
    assert response.content.endswith(("?", "？"))
    assert "增长" in response.content


def test_get_llm_client_supports_mock_and_rejects_unknown() -> None:
    """The LLM factory should expose mock and reject unknown providers."""

    assert isinstance(get_llm_client("mock"), MockLLMClient)
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client("unknown")


def test_deepseek_client_requires_api_key(monkeypatch) -> None:
    """DeepSeek client should fail clearly when no API key is configured."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()


def test_deepseek_client_sends_json_response_format(monkeypatch) -> None:
    """DeepSeek JSON mode should be forwarded to the chat-completions payload."""

    captured_payload: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": '{"question": "Why?"}'}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("llm.deepseek_client.urlopen", fake_urlopen)
    client = DeepSeekClient(api_key="test-key", model="deepseek-v4-pro")

    response = client.complete(
        [LLMMessage(role="user", content="Return JSON.")],
        response_format={"type": "json_object"},
    )

    assert response.content == '{"question": "Why?"}'
    assert captured_payload["model"] == "deepseek-v4-pro"
    assert captured_payload["response_format"] == {"type": "json_object"}


def test_deepseek_client_wraps_timeout_as_llm_error(monkeypatch) -> None:
    """Network read timeouts should allow agents to fall back deterministically."""

    def fake_urlopen(request, timeout):
        del request
        del timeout
        raise TimeoutError("simulated read timeout")

    monkeypatch.setattr("llm.deepseek_client.urlopen", fake_urlopen)
    client = DeepSeekClient(api_key="test-key", timeout_seconds=7)

    with pytest.raises(LLMError, match="timed out after 7s"):
        client.complete([LLMMessage(role="user", content="hello")])


def test_cached_llm_client_reuses_exact_prompt(tmp_path) -> None:
    """Identical LLM requests should be served from the local cache."""

    class CountingClient(BaseLLMClient):
        provider = "deepseek"
        model = "test-model"

        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            messages: list[LLMMessage],
            temperature: float = 0.0,
            response_format: dict[str, object] | None = None,
        ) -> LLMResponse:
            del messages
            del temperature
            del response_format
            self.calls += 1
            return LLMResponse(
                content=f"answer-{self.calls}",
                model=self.model,
                provider=self.provider,
            )

    client = CountingClient()
    cached_client = CachedLLMClient(client, cache_dir=tmp_path, ttl_hours=1)
    messages = [LLMMessage(role="user", content="Analyze MU.")]

    first = cached_client.complete(messages, response_format={"type": "json_object"})
    second = cached_client.complete(messages, response_format={"type": "json_object"})
    third = cached_client.complete([LLMMessage(role="user", content="Analyze INTC.")])

    assert first.content == "answer-1"
    assert second.content == "answer-1"
    assert third.content == "answer-2"
    assert client.calls == 2


def test_cached_llm_client_can_be_disabled(tmp_path) -> None:
    """Disabled cache should call the wrapped client every time."""

    class CountingClient(MockLLMClient):
        provider = "deepseek"

        def __init__(self) -> None:
            super().__init__(model="test-model")
            self.calls = 0

        def complete(
            self,
            messages: list[LLMMessage],
            temperature: float = 0.0,
            response_format: dict[str, object] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            return super().complete(messages, temperature, response_format)

    client = CountingClient()
    cached_client = CachedLLMClient(
        client,
        cache_dir=tmp_path,
        ttl_hours=1,
        enabled=False,
    )
    messages = [LLMMessage(role="user", content="Analyze MU.")]

    cached_client.complete(messages)
    cached_client.complete(messages)

    assert client.calls == 2


def test_parse_json_response_handles_fenced_json() -> None:
    """LLM JSON parser should recover fenced JSON objects."""

    parsed = parse_json_response('Here:\n```json\n{"question": "What changed?"}\n```')

    assert parsed == {"question": "What changed?"}
