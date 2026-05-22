"""Local JSON storage helpers for EVIR."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _to_jsonable(data: Any) -> Any:
    """Convert Pydantic models and nested containers into JSON-safe data."""

    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, dict):
        return {key: _to_jsonable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_jsonable(item) for item in data]
    return data


def save_json(path: str | Path, data: Any) -> Path:
    """Save JSON-serializable data to a local file and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, indent=2, ensure_ascii=False)

    return output_path


def load_json(path: str | Path) -> Any:
    """Load JSON data from a local file."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)

