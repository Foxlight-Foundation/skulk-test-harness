"""Small utility helpers."""

from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def unwrap_tagged(payload: object) -> tuple[str, dict[str, object]] | None:
    """Return ``(tag, body)`` for Skulk's tagged-union JSON objects."""

    if not isinstance(payload, dict) or len(payload) != 1:
        return None
    key = next(iter(payload))
    value = payload[key]
    if not isinstance(key, str) or not isinstance(value, dict):
        return None
    return key, value


def extract_first_code_block(text: str) -> str | None:
    """Extract the first fenced code block body from text."""

    match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", text, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def maybe_write_artifact(output_dir: Path, name: str, text: str) -> Path:
    """Write a generated artifact and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(text)
    return path

