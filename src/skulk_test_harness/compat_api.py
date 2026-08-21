"""Wire-shape validation for the API surfaces third-party clients speak.

Most Skulk use is through its HTTP API rather than the built-in chat, and the
clients on the other end are not written against Skulk. They are written
against OpenAI, Anthropic and Ollama, and they branch on status codes and
field names before they ever look at generated text. A response that is
plausible to a human reader but wrong in shape breaks them, and breaks them far
from the cause.

This module holds that contract as pure functions: each validator takes an
already-fetched response and returns a list of human-readable problems. Nothing
here performs I/O, so the whole contract is exercised by the offline test suite,
and the orchestrator only has to supply responses.

Two invariants are worth stating outright because they are what real clients
actually depend on:

- A success status must carry a body that parses and matches the documented
  shape. A 2xx with an empty or unparseable body is the worst possible answer,
  because it defeats client error handling entirely.
- A failure must carry the provider's documented error envelope, not a bare
  status. Clients surface `error.message` to their users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CompatSurface = Literal["openai", "anthropic", "ollama"]
"""Which third-party wire format a probe speaks."""

FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call"}
)
"""Finish reasons an OpenAI client is prepared to see."""

ANTHROPIC_STOP_REASONS = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"}
)
"""Stop reasons an Anthropic client is prepared to see."""


@dataclass(frozen=True)
class HttpOutcome:
    """One HTTP response, reduced to what the contract depends on.

    `text` is retained separately from `json_body` so a success carrying an
    unparseable body is representable: that combination is itself a finding.
    """

    status_code: int
    text: str
    json_body: Any | None
    json_error: str | None = None
    elapsed_s: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Whether the status is in the 2xx range."""

        return 200 <= self.status_code < 300


def _describe(value: Any, limit: int = 120) -> str:
    """Render a value for a finding message without flooding the report."""

    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def _require_mapping(payload: Any, where: str) -> list[str]:
    """Require a JSON object, the base assumption of every shape below."""

    if not isinstance(payload, dict):
        return [f"{where} must be a JSON object, got {_describe(payload)}"]
    return []


def _require_str(payload: dict[str, Any], key: str, where: str) -> list[str]:
    """Require a present, non-empty string field."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return [f"{where}.{key} must be a non-empty string, got {_describe(value)}"]
    return []


def _require_int(payload: dict[str, Any], key: str, where: str) -> list[str]:
    """Require a present integer field, rejecting bools."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return [f"{where}.{key} must be an integer, got {_describe(value)}"]
    return []


def validate_success_body_present(outcome: HttpOutcome, where: str) -> list[str]:
    """Assert a 2xx response carries a parseable body.

    This is the single most important check in the module. A client that
    receives 200 treats the call as successful and moves straight to parsing,
    so an empty or malformed body surfaces as an unrelated parse error rather
    than as the failure it actually is.
    """

    if not outcome.is_success:
        return []
    if not outcome.text.strip():
        return [
            f"{where} returned HTTP {outcome.status_code} with an empty body; "
            "a success status must carry a response object"
        ]
    if outcome.json_body is None:
        return [
            f"{where} returned HTTP {outcome.status_code} with a body that is not "
            f"valid JSON ({outcome.json_error or 'parse failed'})"
        ]
    return []


def validate_openai_error_envelope(
    outcome: HttpOutcome, where: str, *, expected_status: int | None = None
) -> list[str]:
    """Assert a failure carries OpenAI's documented error envelope."""

    problems: list[str] = []
    if expected_status is not None and outcome.status_code != expected_status:
        problems.append(
            f"{where} expected HTTP {expected_status}, got {outcome.status_code}"
        )
    if outcome.is_success:
        problems.append(
            f"{where} expected a failure status, got HTTP {outcome.status_code}"
        )
        return problems
    body = outcome.json_body
    if not isinstance(body, dict):
        problems.append(f"{where} error body must be a JSON object")
        return problems
    error = body.get("error")
    if not isinstance(error, dict):
        problems.append(f"{where} body must carry an 'error' object")
        return problems
    problems.extend(_require_str(error, "message", f"{where}.error"))
    problems.extend(_require_str(error, "type", f"{where}.error"))
    return problems


def validate_openai_model_list(
    outcome: HttpOutcome, *, expected_model_ids: tuple[str, ...] = ()
) -> list[str]:
    """Validate `GET /v1/models` against the OpenAI list shape.

    Clients call this first to populate a model picker, so an id here that
    cannot subsequently be used is a direct route to a confusing failure.
    """

    where = "GET /v1/models"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    if body.get("object") != "list":
        problems.append(f"{where}.object must be 'list', got {_describe(body.get('object'))}")
    data = body.get("data")
    if not isinstance(data, list):
        return problems + [f"{where}.data must be an array, got {_describe(data)}"]
    seen: set[str] = set()
    for index, entry in enumerate(data):
        entry_where = f"{where}.data[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{entry_where} must be a JSON object")
            continue
        problems.extend(_require_str(entry, "id", entry_where))
        if entry.get("object") != "model":
            problems.append(
                f"{entry_where}.object must be 'model', got {_describe(entry.get('object'))}"
            )
        problems.extend(_require_int(entry, "created", entry_where))
        problems.extend(_require_str(entry, "owned_by", entry_where))
        model_id = entry.get("id")
        if isinstance(model_id, str):
            seen.add(model_id)
    for expected in expected_model_ids:
        if expected not in seen:
            problems.append(
                f"{where} omits {expected!r}, which is serving on this cluster"
            )
    return problems


def validate_openai_chat_completion(
    outcome: HttpOutcome, *, expected_model_id: str | None = None
) -> list[str]:
    """Validate a non-streaming `POST /v1/chat/completions` response."""

    where = "POST /v1/chat/completions"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}: {outcome.text[:200]}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    problems.extend(_require_str(body, "id", where))
    if body.get("object") != "chat.completion":
        problems.append(
            f"{where}.object must be 'chat.completion', got {_describe(body.get('object'))}"
        )
    problems.extend(_require_int(body, "created", where))
    problems.extend(_require_str(body, "model", where))
    if expected_model_id is not None and body.get("model") != expected_model_id:
        problems.append(
            f"{where}.model must echo the requested model {expected_model_id!r}, "
            f"got {_describe(body.get('model'))}"
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return problems + [f"{where}.choices must be a non-empty array"]
    choice = choices[0]
    choice_where = f"{where}.choices[0]"
    if not isinstance(choice, dict):
        return problems + [f"{choice_where} must be a JSON object"]
    message = choice.get("message")
    if not isinstance(message, dict):
        problems.append(f"{choice_where}.message must be a JSON object")
    else:
        if message.get("role") != "assistant":
            problems.append(
                f"{choice_where}.message.role must be 'assistant', "
                f"got {_describe(message.get('role'))}"
            )
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        text_present = isinstance(content, str) and content.strip()
        tools_present = isinstance(tool_calls, list) and bool(tool_calls)
        if not text_present and not tools_present:
            problems.append(
                f"{choice_where}.message must carry non-empty content or tool_calls"
            )
    finish_reason = choice.get("finish_reason")
    if finish_reason not in FINISH_REASONS:
        problems.append(
            f"{choice_where}.finish_reason must be one of {sorted(FINISH_REASONS)}, "
            f"got {_describe(finish_reason)}"
        )
    usage = body.get("usage")
    if not isinstance(usage, dict):
        problems.append(f"{where}.usage must be a JSON object")
    else:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            problems.extend(_require_int(usage, key, f"{where}.usage"))
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if all(isinstance(v, int) and not isinstance(v, bool) for v in (prompt, completion, total)):
            assert isinstance(prompt, int) and isinstance(completion, int)
            assert isinstance(total, int)
            if prompt + completion != total:
                problems.append(
                    f"{where}.usage.total_tokens must equal prompt + completion "
                    f"({prompt} + {completion} != {total})"
                )
    return problems


@dataclass(frozen=True)
class SseStream:
    """A parsed Server-Sent Events response."""

    events: tuple[str, ...]
    saw_done_sentinel: bool
    malformed_lines: tuple[str, ...]


def parse_sse(raw: str) -> SseStream:
    """Split an SSE body into event payloads.

    Tolerates comment lines and blank separators, which the spec allows, and
    records any non-blank line that is neither, since a client using a strict
    SSE reader would choke on it.
    """

    events: list[str] = []
    malformed: list[str] = []
    saw_done = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            malformed.append(stripped[:120])
            continue
        payload = stripped[len("data:") :].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        events.append(payload)
    return SseStream(
        events=tuple(events),
        saw_done_sentinel=saw_done,
        malformed_lines=tuple(malformed),
    )


def validate_openai_chat_stream(
    outcome: HttpOutcome, *, min_chunks: int = 2
) -> tuple[list[str], str]:
    """Validate a streaming chat completion and return the assembled text.

    A client reading this stream needs three things: every line framed as SSE
    data, every payload shaped as a chunk, and an explicit `[DONE]` so it knows
    the turn ended rather than the connection dropping.
    """

    where = "POST /v1/chat/completions (stream)"
    if not outcome.is_success:
        return ([f"{where} returned HTTP {outcome.status_code}"], "")
    if not outcome.text.strip():
        return ([f"{where} returned HTTP {outcome.status_code} with an empty body"], "")
    import json

    stream = parse_sse(outcome.text)
    problems: list[str] = []
    for line in stream.malformed_lines:
        problems.append(f"{where} emitted a line that is not SSE data: {line!r}")
    if not stream.saw_done_sentinel:
        problems.append(f"{where} did not terminate with 'data: [DONE]'")
    assembled: list[str] = []
    for index, payload in enumerate(stream.events):
        chunk_where = f"{where} chunk[{index}]"
        try:
            chunk = json.loads(payload)
        except ValueError as exc:
            problems.append(f"{chunk_where} is not valid JSON: {exc}")
            continue
        if not isinstance(chunk, dict):
            problems.append(f"{chunk_where} must be a JSON object")
            continue
        if chunk.get("object") != "chat.completion.chunk":
            problems.append(
                f"{chunk_where}.object must be 'chat.completion.chunk', "
                f"got {_describe(chunk.get('object'))}"
            )
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            problems.append(f"{chunk_where}.choices must be a non-empty array")
            continue
        first = choices[0]
        if not isinstance(first, dict):
            problems.append(f"{chunk_where}.choices[0] must be a JSON object")
            continue
        delta = first.get("delta")
        if delta is None and first.get("finish_reason") is None:
            problems.append(
                f"{chunk_where}.choices[0] must carry a delta or a finish_reason"
            )
        if isinstance(delta, dict):
            piece = delta.get("content")
            if isinstance(piece, str):
                assembled.append(piece)
    if len(stream.events) < min_chunks:
        problems.append(
            f"{where} produced {len(stream.events)} chunks, expected at least {min_chunks}"
        )
    text = "".join(assembled)
    if not text.strip():
        problems.append(f"{where} assembled no content across {len(stream.events)} chunks")
    return (problems, text)


def validate_openai_embeddings(outcome: HttpOutcome) -> list[str]:
    """Validate `POST /v1/embeddings` against the OpenAI shape."""

    where = "POST /v1/embeddings"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}: {outcome.text[:200]}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    if body.get("object") != "list":
        problems.append(f"{where}.object must be 'list', got {_describe(body.get('object'))}")
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return problems + [f"{where}.data must be a non-empty array"]
    entry = data[0]
    entry_where = f"{where}.data[0]"
    if not isinstance(entry, dict):
        return problems + [f"{entry_where} must be a JSON object"]
    if entry.get("object") != "embedding":
        problems.append(
            f"{entry_where}.object must be 'embedding', got {_describe(entry.get('object'))}"
        )
    problems.extend(_require_int(entry, "index", entry_where))
    vector = entry.get("embedding")
    if not isinstance(vector, list) or not vector:
        problems.append(f"{entry_where}.embedding must be a non-empty array")
    elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector):
        problems.append(f"{entry_where}.embedding must contain only numbers")
    problems.extend(_require_str(body, "model", where))
    return problems


def validate_anthropic_message(
    outcome: HttpOutcome, *, expected_model_id: str | None = None
) -> list[str]:
    """Validate `POST /v1/messages` against the Anthropic Messages shape.

    This is the surface Claude Code speaks, so its field names matter as much
    as the text: a client reads `content[].text` and `stop_reason`, not
    `choices`.
    """

    where = "POST /v1/messages"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}: {outcome.text[:200]}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    problems.extend(_require_str(body, "id", where))
    if body.get("type") != "message":
        problems.append(f"{where}.type must be 'message', got {_describe(body.get('type'))}")
    if body.get("role") != "assistant":
        problems.append(f"{where}.role must be 'assistant', got {_describe(body.get('role'))}")
    problems.extend(_require_str(body, "model", where))
    if expected_model_id is not None and body.get("model") != expected_model_id:
        problems.append(
            f"{where}.model must echo the requested model {expected_model_id!r}, "
            f"got {_describe(body.get('model'))}"
        )
    content = body.get("content")
    if not isinstance(content, list) or not content:
        problems.append(f"{where}.content must be a non-empty array")
    else:
        text_blocks = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not text_blocks:
            problems.append(f"{where}.content must carry at least one text block")
        elif not any(
            isinstance(block.get("text"), str) and block["text"].strip()
            for block in text_blocks
        ):
            problems.append(f"{where}.content text blocks are all empty")
    stop_reason = body.get("stop_reason")
    if stop_reason is not None and stop_reason not in ANTHROPIC_STOP_REASONS:
        problems.append(
            f"{where}.stop_reason must be one of {sorted(ANTHROPIC_STOP_REASONS)} or null, "
            f"got {_describe(stop_reason)}"
        )
    usage = body.get("usage")
    if not isinstance(usage, dict):
        problems.append(f"{where}.usage must be a JSON object")
    else:
        for key in ("input_tokens", "output_tokens"):
            problems.extend(_require_int(usage, key, f"{where}.usage"))
    return problems


def validate_ollama_version(outcome: HttpOutcome, path: str) -> list[str]:
    """Validate an Ollama version response."""

    where = f"GET {path}"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    problems.extend(_require_str(body, "version", where))
    return problems


def validate_ollama_tags(
    outcome: HttpOutcome, path: str, *, expected_model_ids: tuple[str, ...] = ()
) -> list[str]:
    """Validate an Ollama tag listing.

    Ollama clients populate their model picker from this and then send the
    `name` back verbatim, so a name here that the chat route will not accept
    strands the user inside the client.
    """

    where = f"GET {path}"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    models = body.get("models")
    if not isinstance(models, list):
        return problems + [f"{where}.models must be an array, got {_describe(models)}"]
    seen: set[str] = set()
    for index, entry in enumerate(models):
        entry_where = f"{where}.models[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{entry_where} must be a JSON object")
            continue
        problems.extend(_require_str(entry, "name", entry_where))
        problems.extend(_require_str(entry, "model", entry_where))
        problems.extend(_require_str(entry, "modified_at", entry_where))
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int):
            problems.append(f"{entry_where}.size must be an integer")
        if not isinstance(entry.get("details"), dict):
            problems.append(f"{entry_where}.details must be a JSON object")
        name = entry.get("name")
        if isinstance(name, str):
            seen.add(name)
    for expected in expected_model_ids:
        if expected not in seen:
            problems.append(f"{where} omits {expected!r}, which is serving on this cluster")
    return problems


def validate_ollama_chat(
    outcome: HttpOutcome, path: str, *, expect_done: bool = True
) -> list[str]:
    """Validate a non-streaming Ollama chat response."""

    where = f"POST {path}"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}: {outcome.text[:200]}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    problems.extend(_require_str(body, "model", where))
    problems.extend(_require_str(body, "created_at", where))
    message = body.get("message")
    if not isinstance(message, dict):
        problems.append(f"{where}.message must be a JSON object")
    else:
        if message.get("role") != "assistant":
            problems.append(
                f"{where}.message.role must be 'assistant', got {_describe(message.get('role'))}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            problems.append(f"{where}.message.content must be a non-empty string")
    if expect_done and body.get("done") is not True:
        problems.append(f"{where}.done must be true, got {_describe(body.get('done'))}")
    return problems


def validate_ollama_generate(outcome: HttpOutcome, path: str) -> list[str]:
    """Validate a non-streaming Ollama generate response."""

    where = f"POST {path}"
    problems = validate_success_body_present(outcome, where)
    if problems:
        return problems
    if not outcome.is_success:
        return [f"{where} returned HTTP {outcome.status_code}: {outcome.text[:200]}"]
    body = outcome.json_body
    problems.extend(_require_mapping(body, where))
    if problems:
        return problems
    assert isinstance(body, dict)
    problems.extend(_require_str(body, "model", where))
    response = body.get("response")
    if not isinstance(response, str) or not response.strip():
        problems.append(f"{where}.response must be a non-empty string")
    if body.get("done") is not True:
        problems.append(f"{where}.done must be true, got {_describe(body.get('done'))}")
    return problems


def validate_ollama_chat_stream(outcome: HttpOutcome, path: str) -> tuple[list[str], str]:
    """Validate a streaming Ollama chat response and return assembled text.

    Ollama streams newline-delimited JSON rather than SSE, and the final object
    carries `done: true`. A client reading this does not tolerate SSE framing.
    """

    where = f"POST {path} (stream)"
    if not outcome.is_success:
        return ([f"{where} returned HTTP {outcome.status_code}"], "")
    if not outcome.text.strip():
        return ([f"{where} returned HTTP {outcome.status_code} with an empty body"], "")
    import json

    problems: list[str] = []
    assembled: list[str] = []
    saw_done = False
    lines = [line for line in outcome.text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.lstrip().startswith("data:"):
            problems.append(
                f"{where} line[{index}] uses SSE framing; the Ollama surface must "
                "emit newline-delimited JSON objects"
            )
            continue
        try:
            chunk = json.loads(line)
        except ValueError as exc:
            problems.append(f"{where} line[{index}] is not valid JSON: {exc}")
            continue
        if not isinstance(chunk, dict):
            problems.append(f"{where} line[{index}] must be a JSON object")
            continue
        message = chunk.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            assembled.append(message["content"])
        if chunk.get("done") is True:
            saw_done = True
    if not saw_done:
        problems.append(f"{where} never emitted a final object with done=true")
    text = "".join(assembled)
    if not text.strip():
        problems.append(f"{where} assembled no content across {len(lines)} lines")
    return (problems, text)
