"""Drive a live cluster the way a third-party client does.

`compat_api` holds the wire contract as pure functions. This module performs
the requests that feed them, using a plain `httpx.Client` rather than
`SkulkClient` because the point is to behave like software that has never
heard of Skulk: no helper paths, no tagged-union unwrapping, no retry policy
beyond what an ordinary client would do.

Each surface runs a battery rather than a single call, because the failures
worth catching live in the corners: the second prefix an Ollama client tries,
the error body a client shows its user, the terminator that tells a stream
reader the turn ended.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .compat_api import (
    HttpOutcome,
    validate_anthropic_message,
    validate_ollama_chat,
    validate_ollama_chat_stream,
    validate_ollama_generate,
    validate_ollama_tags,
    validate_ollama_version,
    validate_openai_chat_completion,
    validate_openai_chat_stream,
    validate_openai_embeddings,
    validate_openai_error_envelope,
    validate_openai_model_list,
)

PROBE_PROMPT = "Reply with exactly the word: ready"
"""Deliberately trivial so a failure is a wire failure, not a model failure."""

UNKNOWN_MODEL_ID = "skulk-harness/definitely-not-a-real-model"
"""Used to assert the not-found path carries a proper error envelope."""


@dataclass
class CompatProbeReport:
    """Result of running one surface's battery."""

    problems: list[str] = field(default_factory=list)
    checks_run: int = 0
    sample_text: str = ""
    elapsed_s: float = 0.0

    def record(self, problems: list[str]) -> None:
        """Fold one check's problems into the report."""

        self.checks_run += 1
        self.problems.extend(problems)


def _fetch(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> HttpOutcome:
    """Perform one request and reduce it to an `HttpOutcome`.

    Transport failures become an outcome with status 0 rather than an
    exception, so one unreachable endpoint reports as a finding instead of
    abandoning the rest of the battery.
    """

    started = time.perf_counter()
    try:
        response = client.request(
            method,
            path,
            json=json_body,
            headers=headers,
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        return HttpOutcome(
            status_code=0,
            text="",
            json_body=None,
            json_error=f"transport error: {exc}",
            elapsed_s=time.perf_counter() - started,
        )
    elapsed = time.perf_counter() - started
    text = response.text
    parsed: Any | None = None
    parse_error: str | None = None
    if text.strip():
        try:
            parsed = response.json()
        except ValueError as exc:
            parse_error = str(exc)
    return HttpOutcome(
        status_code=response.status_code,
        text=text,
        json_body=parsed,
        json_error=parse_error,
        elapsed_s=elapsed,
        headers=dict(response.headers),
    )


def run_openai_probes(
    client: httpx.Client,
    *,
    model_id: str,
    serving_model_ids: tuple[str, ...] = (),
    embedding_model_id: str | None = None,
    include_streaming: bool = True,
    generation_timeout_s: float = 300.0,
) -> CompatProbeReport:
    """Exercise the OpenAI-compatible surface end to end."""

    report = CompatProbeReport()
    started = time.perf_counter()

    report.record(
        validate_openai_model_list(
            _fetch(client, "GET", "/v1/models"),
            expected_model_ids=serving_model_ids,
        )
    )

    completion = _fetch(
        client,
        "POST",
        "/v1/chat/completions",
        json_body={
            "model": model_id,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout_s=generation_timeout_s,
    )
    report.record(validate_openai_chat_completion(completion, expected_model_id=model_id))
    if isinstance(completion.json_body, dict):
        choices = completion.json_body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                report.sample_text = message["content"]

    if include_streaming:
        stream = _fetch(
            client,
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model_id,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "max_tokens": 32,
                "temperature": 0,
                "stream": True,
            },
            timeout_s=generation_timeout_s,
        )
        stream_problems, streamed_text = validate_openai_chat_stream(stream)
        report.record(stream_problems)
        if streamed_text and not report.sample_text:
            report.sample_text = streamed_text

    # A client shows error.message to its user, so the envelope matters as much
    # as the status.
    report.record(
        validate_openai_error_envelope(
            _fetch(
                client,
                "POST",
                "/v1/chat/completions",
                json_body={
                    "model": UNKNOWN_MODEL_ID,
                    "messages": [{"role": "user", "content": PROBE_PROMPT}],
                    "max_tokens": 8,
                },
                timeout_s=60.0,
            ),
            "POST /v1/chat/completions (unknown model)",
            expected_status=404,
        )
    )

    malformed = _fetch(
        client,
        "POST",
        "/v1/chat/completions",
        json_body={"model": model_id},
        timeout_s=60.0,
    )
    report.checks_run += 1
    if malformed.is_success:
        report.problems.append(
            "POST /v1/chat/completions accepted a request with no messages "
            f"(HTTP {malformed.status_code}); a client sending a malformed body "
            "must receive a 4xx"
        )
    elif malformed.status_code >= 500:
        report.problems.append(
            "POST /v1/chat/completions answered a malformed request with "
            f"HTTP {malformed.status_code}; a client error must not surface as a "
            "server error"
        )

    if embedding_model_id:
        report.record(
            validate_openai_embeddings(
                _fetch(
                    client,
                    "POST",
                    "/v1/embeddings",
                    json_body={
                        "model": embedding_model_id,
                        "input": "skulk external api compatibility probe",
                    },
                    timeout_s=120.0,
                )
            )
        )
        # Asking an embedding model to chat is a mistake real users make in
        # clients that list every model in one picker.
        report.record(
            validate_openai_error_envelope(
                _fetch(
                    client,
                    "POST",
                    "/v1/chat/completions",
                    json_body={
                        "model": embedding_model_id,
                        "messages": [{"role": "user", "content": PROBE_PROMPT}],
                        "max_tokens": 8,
                    },
                    timeout_s=60.0,
                ),
                "POST /v1/chat/completions (embedding model)",
                expected_status=400,
            )
        )

    report.elapsed_s = time.perf_counter() - started
    return report


def run_anthropic_probes(
    client: httpx.Client,
    *,
    model_id: str,
    generation_timeout_s: float = 300.0,
) -> CompatProbeReport:
    """Exercise the Anthropic Messages surface, which Claude Code speaks."""

    report = CompatProbeReport()
    started = time.perf_counter()

    message = _fetch(
        client,
        "POST",
        "/v1/messages",
        json_body={
            "model": model_id,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
        },
        headers={"anthropic-version": "2023-06-01"},
        timeout_s=generation_timeout_s,
    )
    report.record(validate_anthropic_message(message, expected_model_id=model_id))
    if isinstance(message.json_body, dict):
        content = message.json_body.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    report.sample_text = block["text"]
                    break

    # Anthropic clients always send a system prompt separately from messages,
    # so a surface that only accepts it inline is unusable by them.
    report.record(
        validate_anthropic_message(
            _fetch(
                client,
                "POST",
                "/v1/messages",
                json_body={
                    "model": model_id,
                    "max_tokens": 32,
                    "system": "You answer in one word.",
                    "messages": [{"role": "user", "content": PROBE_PROMPT}],
                },
                headers={"anthropic-version": "2023-06-01"},
                timeout_s=generation_timeout_s,
            ),
            expected_model_id=model_id,
        )
    )

    report.record(
        validate_openai_error_envelope(
            _fetch(
                client,
                "POST",
                "/v1/messages",
                json_body={
                    "model": UNKNOWN_MODEL_ID,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": PROBE_PROMPT}],
                },
                headers={"anthropic-version": "2023-06-01"},
                timeout_s=60.0,
            ),
            "POST /v1/messages (unknown model)",
            expected_status=404,
        )
    )

    report.elapsed_s = time.perf_counter() - started
    return report


OLLAMA_CHAT_ALIASES = ("/ollama/api/chat", "/ollama/api/api/chat", "/ollama/api/v1/chat")
"""Every chat prefix the server accepts.

Real Ollama clients disagree about whether the base URL already ends in
`/api`, so they produce doubled and versioned prefixes. Skulk answers all
three deliberately; without a test, a future tidy-up silently breaks whichever
client depends on the alias.
"""

OLLAMA_TAG_ALIASES = ("/ollama/api/tags", "/ollama/api/api/tags", "/ollama/api/v1/tags")
"""The same disagreement, on the model-listing route."""


def run_ollama_probes(
    client: httpx.Client,
    *,
    model_id: str,
    serving_model_ids: tuple[str, ...] = (),
    include_streaming: bool = True,
    generation_timeout_s: float = 300.0,
) -> CompatProbeReport:
    """Exercise the Ollama-compatible surface, including its prefix aliases."""

    report = CompatProbeReport()
    started = time.perf_counter()

    report.record(
        validate_ollama_version(
            _fetch(client, "GET", "/ollama/api/version", timeout_s=30.0),
            "/ollama/api/version",
        )
    )

    for path in OLLAMA_TAG_ALIASES:
        report.record(
            validate_ollama_tags(
                _fetch(client, "GET", path, timeout_s=60.0),
                path,
                expected_model_ids=serving_model_ids,
            )
        )

    for path in OLLAMA_CHAT_ALIASES:
        outcome = _fetch(
            client,
            "POST",
            path,
            json_body={
                "model": model_id,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "stream": False,
            },
            timeout_s=generation_timeout_s,
        )
        report.record(validate_ollama_chat(outcome, path))
        if not report.sample_text and isinstance(outcome.json_body, dict):
            message = outcome.json_body.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                report.sample_text = message["content"]

    report.record(
        validate_ollama_generate(
            _fetch(
                client,
                "POST",
                "/ollama/api/generate",
                json_body={"model": model_id, "prompt": PROBE_PROMPT, "stream": False},
                timeout_s=generation_timeout_s,
            ),
            "/ollama/api/generate",
        )
    )

    show = _fetch(
        client,
        "POST",
        "/ollama/api/show",
        json_body={"name": model_id},
        timeout_s=60.0,
    )
    report.checks_run += 1
    if not show.is_success:
        report.problems.append(
            f"POST /ollama/api/show returned HTTP {show.status_code}; clients call "
            "it to learn a model's context window before chatting"
        )
    elif not isinstance(show.json_body, dict):
        report.problems.append("POST /ollama/api/show must return a JSON object")

    if include_streaming:
        stream = _fetch(
            client,
            "POST",
            "/ollama/api/chat",
            json_body={
                "model": model_id,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "stream": True,
            },
            timeout_s=generation_timeout_s,
        )
        stream_problems, streamed = validate_ollama_chat_stream(stream, "/ollama/api/chat")
        report.record(stream_problems)
        if streamed and not report.sample_text:
            report.sample_text = streamed

    report.elapsed_s = time.perf_counter() - started
    return report


def build_probe_client(base_url: str, *, timeout_s: float = 60.0) -> httpx.Client:
    """Build a deliberately plain HTTP client for probing.

    No Skulk headers and no retry policy: if a request needs special handling
    to succeed, a third-party client will not know to apply it.
    """

    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=timeout_s,
        headers={"content-type": "application/json"},
    )


def summarize(report: CompatProbeReport, *, limit: int = 12) -> str:
    """Render a bounded human summary of a probe report."""

    if not report.problems:
        return f"{report.checks_run} checks passed"
    shown = report.problems[:limit]
    remainder = len(report.problems) - len(shown)
    text = json.dumps(shown, indent=None)
    if remainder > 0:
        text += f" (+{remainder} more)"
    return text
