"""Tool-calling path taxonomy and pure validators.

Skulk recovers tool calls along several independent paths, and a defect in one
says nothing about the others. This module names those paths and holds the
checks that apply to all of them, so a suite states *which* path it exercises
and the shared expectations live in one place rather than being re-listed, and
re-mistyped, in every YAML case.

Everything here is pure: it takes what a completed generation produced and
returns human-readable problems. Nothing reaches the network, so the rules can
be unit-tested without a cluster, and a rule that is wrong fails in CI rather
than being discovered against live hardware.

The paths, and why each is separate:

``generic``
    Marker-delimited ``<tool_call>`` blocks carrying Hermes JSON or Qwen3 XML.
    Driven by the MLX marker scanner and by llama.cpp's own handler.
``unmarked``
    Llama writes the call as a bare JSON object with no opening marker and ends
    the message rather than closing a block, so neither half of the marker
    mechanism applies.
``gemma4``
    ``call:NAME{...}`` with ``<|"|>``-delimited strings, opened and closed by
    its own markers.
``gpt_oss``
    Harmony channels, decoded from token ids rather than text, by a parser that
    is selected ahead of the marker path entirely.
``dsml``
    DeepSeek V3.2's DSML invoke blocks, also decoded ahead of the marker path.
``served``
    llama-server and vLLM parse tool calls themselves and hand back structured
    calls, so none of the in-process recovery runs at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

ToolPath = str

# Markers that belong to a dialect's scaffolding and must never reach a caller
# as content. Grouped by path so a suite can forbid one dialect's markers, and
# so adding a dialect updates every case that forbids "all" at once.
#
# Every marker here leaked from a real model during live testing. They are not
# a guess at what a dialect might emit.
SCAFFOLDING_MARKERS: Mapping[ToolPath, tuple[str, ...]] = {
    "generic": ("<tool_call>", "</tool_call>"),
    "unmarked": ("<|python_tag|>", "<|eom_id|>", "<|eot_id|>", "<|start_header_id|>"),
    "gemma4": ("<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>"),
    "gpt_oss": ("<|channel|>", "<|message|>", "<|start|>", "<|end|>", "to=functions."),
    "dsml": ("<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>"),
    "mistral": ("[TOOL_CALLS]",),
}

ALL_PATHS: tuple[ToolPath, ...] = tuple(SCAFFOLDING_MARKERS)


class ToolCallLike(Protocol):
    """The shape this module needs from a recorded tool call."""

    name: str
    arguments_text: str
    id: str


def scaffolding_markers(paths: Iterable[ToolPath] | None = None) -> tuple[str, ...]:
    """Return the markers to forbid for ``paths``, or for every path.

    Passing every path is the useful default for a case that does not care
    which dialect served it: a marker from the wrong dialect appearing in the
    answer is a defect whichever model produced it.
    """

    selected = tuple(paths) if paths is not None else ALL_PATHS
    markers: list[str] = []
    for path in selected:
        for marker in SCAFFOLDING_MARKERS.get(path, ()):
            if marker not in markers:
                markers.append(marker)
    return tuple(markers)


def validate_call_identity(calls: Sequence[ToolCallLike]) -> list[str]:
    """Check each call carries what a caller needs to answer it.

    A call with no id cannot be replied to, because the tool result message has
    to name the call it answers, and two calls sharing one id make the results
    ambiguous. The client records an omitted id as an empty string rather than
    inventing one, so this sees what the API actually returned.

    Index is deliberately not checked: the client backfills it from position
    when the API omits it, so asserting it here would only confirm the client's
    own default.
    """

    problems: list[str] = []
    seen_ids: set[str] = set()
    for position, call in enumerate(calls):
        if not call.name:
            problems.append(f"tool call at position {position} has no name")
        if not call.id:
            problems.append(f"tool call {call.name!r} came back with no id")
        elif call.id in seen_ids:
            problems.append(f"tool call id {call.id!r} was reused within one message")
        else:
            seen_ids.add(call.id)
    return problems


def validate_call_arguments(calls: Sequence[ToolCallLike]) -> list[str]:
    """Check each call's arguments are a JSON object a caller can act on.

    OpenAI's contract is that ``arguments`` is a JSON object encoded as a
    string. A dialect recovered as text can produce something that looks right
    and does not parse, or parses to a list or a bare string, and a caller only
    finds out when it tries to dispatch.
    """

    problems: list[str] = []
    for call in calls:
        text = call.arguments_text
        if text is None or text == "":
            problems.append(f"tool call {call.name!r} carried no arguments text")
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            problems.append(
                f"tool call {call.name!r} arguments are not valid JSON: {text[:120]!r}"
            )
            continue
        if not isinstance(parsed, dict):
            problems.append(
                f"tool call {call.name!r} arguments decoded to "
                f"{type(parsed).__name__}, expected a JSON object"
            )
    return problems


def validate_calls_were_offered(
    calls: Sequence[ToolCallLike], offered_names: Iterable[str]
) -> list[str]:
    """Check every call names a tool the request actually offered.

    Models reach for their own built-ins: Llama answers some plain questions
    with a call to ``print``, and gpt-oss has ``python`` and ``browser``. A
    caller has no implementation for those, so returning one hands them a name
    they cannot dispatch.
    """

    offered = {name for name in offered_names if name}
    if not offered:
        return []
    problems: list[str] = []
    for call in calls:
        if call.name and call.name not in offered:
            problems.append(
                f"tool call {call.name!r} names no offered tool "
                f"(offered: {sorted(offered)})"
            )
    return problems


def offered_tool_names(tools: Iterable[Mapping[str, object]] | None) -> tuple[str, ...]:
    """Pull the function names out of an OpenAI-style tools array."""

    if not tools:
        return ()
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)
