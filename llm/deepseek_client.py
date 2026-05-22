"""DeepSeek OpenAI-compatible chat client."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
)
from llm.base import BaseLLMClient, LLMError, LLMMessage, LLMResponse


class DeepSeekClient(BaseLLMClient):
    """Minimal DeepSeek client using the OpenAI-compatible chat API."""

    provider = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds or DEFAULT_DEEPSEEK_TIMEOUT_SECONDS

        if not self.api_key:
            raise LLMError("DEEPSEEK_API_KEY is required for DeepSeekClient.")

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response using DeepSeek's chat-completions endpoint."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            raise LLMError(f"DeepSeek HTTP error {error.code}: {message}") from error
        except URLError as error:
            raise LLMError(f"DeepSeek request failed: {error}") from error
        except TimeoutError as error:
            raise LLMError(
                f"DeepSeek request timed out after {self.timeout_seconds}s."
            ) from error
        except json.JSONDecodeError as error:
            raise LLMError("DeepSeek returned invalid JSON.") from error

        content = _extract_content(response_data)
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            raw=response_data,
        )


def _extract_content(response_data: dict[str, Any]) -> str:
    """Extract the assistant content from a chat-completions response."""

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError("DeepSeek response did not include assistant content.") from error

    if not isinstance(content, str) or not content.strip():
        raise LLMError("DeepSeek response content was empty.")
    return content.strip()
