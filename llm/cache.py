"""Local file cache wrapper for LLM clients."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import DEFAULT_CACHE_DIR, LLM_CACHE_TTL_HOURS
from llm.base import BaseLLMClient, LLMMessage, LLMResponse
from memory.storage import load_json, save_json


class CachedLLMClient(BaseLLMClient):
    """Cache deterministic LLM responses by exact request payload."""

    provider: str
    model: str

    def __init__(
        self,
        client: BaseLLMClient,
        cache_dir: str | Path = DEFAULT_CACHE_DIR / "llm",
        ttl_hours: int = LLM_CACHE_TTL_HOURS,
        enabled: bool = True,
    ) -> None:
        self.client = client
        self.provider = client.provider
        self.model = client.model
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.enabled = enabled

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a cached response when the exact LLM request is still fresh."""

        cache_key = build_llm_cache_key(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        if self.enabled and self.ttl_hours > 0:
            cached_response = self._load_response(cache_key)
            if cached_response is not None:
                return cached_response

        response = self.client.complete(
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        if self.enabled and self.ttl_hours > 0:
            self._save_response(cache_key, response)
        return response

    def _load_response(self, cache_key: str) -> LLMResponse | None:
        """Load a fresh cached response, or return None."""

        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            payload = load_json(path)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        created_at = _parse_created_at(payload.get("created_at"))
        if created_at is None:
            return None
        if datetime.now(timezone.utc) - created_at > timedelta(hours=self.ttl_hours):
            return None

        response_data = payload.get("response")
        if not isinstance(response_data, dict):
            return None
        content = response_data.get("content")
        model = response_data.get("model")
        provider = response_data.get("provider")
        raw = response_data.get("raw")
        if not isinstance(content, str) or not isinstance(model, str):
            return None
        if not isinstance(provider, str):
            return None
        if raw is not None and not isinstance(raw, dict):
            raw = None
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            raw=raw,
        )

    def _save_response(self, cache_key: str, response: LLMResponse) -> None:
        """Persist a normalized response for future identical requests."""

        save_json(
            self._cache_path(cache_key),
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider,
                "model": self.model,
                "cache_key": cache_key,
                "ttl_hours": self.ttl_hours,
                "response": {
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "raw": response.raw,
                },
            },
        )

    def _cache_path(self, cache_key: str) -> Path:
        """Build the provider-scoped cache path."""

        return self.cache_dir / _safe_name(self.provider) / f"{cache_key}.json"


def build_llm_cache_key(
    provider: str,
    model: str,
    messages: list[LLMMessage],
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Build a stable request hash for an exact LLM completion call."""

    request_payload = {
        "cache_version": 1,
        "provider": provider,
        "model": model,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        "temperature": temperature,
        "response_format": response_format,
    }
    serialized = json.dumps(
        request_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_created_at(value: Any) -> datetime | None:
    """Parse a cache timestamp as UTC."""

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_name(value: str) -> str:
    """Return a filesystem-safe cache segment."""

    return "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    )
