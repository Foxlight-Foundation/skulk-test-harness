"""Stable, privacy-preserving identities for effective harness requests."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from skulk_test_harness.models import PromptTest

_NON_PROTOCOL_FIELDS: Final[frozenset[str]] = frozenset(
    {"name", "description", "repetitions", "success"}
)


def _effective_request(
    test: PromptTest, *, thinking_default: bool | None
) -> dict[str, object]:
    """Return request-affecting inputs with every harness default resolved.

    The returned object is an in-process hash preimage only. Callers must emit
    the digest, never this object, because it can contain prompt and system text.
    """

    dumped = test.model_dump(mode="json")
    request = {
        key: value for key, value in dumped.items() if key not in _NON_PROTOCOL_FIELDS
    }
    if test.enable_thinking is None:
        request["enable_thinking"] = thinking_default
    return request


def _digest(payload: dict[str, object]) -> str:
    """Return the SHA-256 digest of canonical JSON for ``payload``."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def protocol_ids(
    test: PromptTest, *, thinking_default: bool | None
) -> tuple[str, str]:
    """Return exact and concurrency-family protocol identities for one test."""

    effective = _effective_request(test, thinking_default=thinking_default)
    family = dict(effective)
    family.pop("concurrency", None)
    return _digest(effective), _digest(family)
