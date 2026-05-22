"""Robust JSON parsing helpers for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

from llm.base import LLMError


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response.

    The helper accepts raw JSON, fenced Markdown JSON, or a response with prose
    around one JSON object. It raises LLMError when no valid JSON object can be
    recovered.
    """

    candidates = [
        content.strip(),
        *_fenced_json_blocks(content),
        _first_json_object(content),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMError("LLM response did not contain a valid JSON object.")


def _fenced_json_blocks(content: str) -> list[str]:
    """Extract JSON-looking Markdown fenced code blocks."""

    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]


def _first_json_object(content: str) -> str | None:
    """Extract the first balanced top-level JSON object substring."""

    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(content[start:], start=start):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return content[start:index + 1]
    return None
