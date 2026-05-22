"""Small JSON cache helpers for external data providers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import DEFAULT_CACHE_DIR, ENABLE_DATA_CACHE
from memory.storage import load_json, save_json


def load_cache(category: str, key: str, max_age_days: int) -> Any | None:
    """Load a fresh cache payload, or return None when absent or stale."""

    if not ENABLE_DATA_CACHE:
        return None

    payload = _load_payload(category, key)
    if payload is None:
        return None

    created_at = _parse_created_at(payload.get("created_at"))
    if created_at is None:
        return None
    if datetime.now(timezone.utc) - created_at > timedelta(days=max_age_days):
        return None
    return payload.get("data")


def load_stale_cache(category: str, key: str) -> Any | None:
    """Load any cache payload regardless of age."""

    if not ENABLE_DATA_CACHE:
        return None

    payload = _load_payload(category, key)
    if payload is None:
        return None
    return payload.get("data")


def save_cache(category: str, key: str, data: Any) -> Path | None:
    """Save a cache payload and return its path."""

    if not ENABLE_DATA_CACHE:
        return None

    return save_json(
        _cache_path(category, key),
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        },
    )


def _load_payload(category: str, key: str) -> dict[str, Any] | None:
    """Load a cache wrapper if it exists and is well-formed."""

    path = _cache_path(category, key)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _cache_path(category: str, key: str) -> Path:
    """Build a normalized cache file path."""

    safe_category = _safe_name(category)
    safe_key = _safe_name(key)
    return DEFAULT_CACHE_DIR / safe_category / f"{safe_key}.json"


def _safe_name(value: str) -> str:
    """Return a filesystem-safe cache name."""

    return "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    )


def _parse_created_at(value: Any) -> datetime | None:
    """Parse a cache timestamp."""

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
