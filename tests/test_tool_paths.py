"""Offline coverage for the tool-calling path rules.

These rules decide whether a live run passes, so a rule that is wrong makes a
broken integration look healthy. They are pure, so they are checked here rather
than against hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from skulk_test_harness.tool_paths import (
    ALL_PATHS,
    SCAFFOLDING_MARKERS,
    offered_tool_names,
    scaffolding_markers,
    validate_call_arguments,
    validate_call_identity,
    validate_calls_were_offered,
)


@dataclass
class Call:
    """Stand-in for a recorded tool call."""

    name: str = "get_weather"
    arguments_text: str = '{"location": "Denver"}'
    id: str = "call-1"


class TestScaffoldingMarkers:
    def test_every_path_contributes_markers(self) -> None:
        markers = scaffolding_markers()
        for path in ALL_PATHS:
            for marker in SCAFFOLDING_MARKERS[path]:
                assert marker in markers, f"{path} marker {marker!r} missing"

    def test_a_single_path_narrows_the_set(self) -> None:
        markers = scaffolding_markers(["unmarked"])
        assert "<|python_tag|>" in markers
        assert "<tool_call>" not in markers

    def test_markers_are_deduplicated_across_paths(self) -> None:
        markers = scaffolding_markers()
        assert len(markers) == len(set(markers))

    def test_an_unknown_path_contributes_nothing_rather_than_raising(self) -> None:
        # A suite naming a path that does not exist should not crash a live run
        # mid-flight; the empty result shows up as a case that checks nothing.
        assert scaffolding_markers(["not-a-path"]) == ()


class TestCallIdentity:
    def test_a_complete_call_is_accepted(self) -> None:
        assert validate_call_identity([Call()]) == []

    def test_a_call_with_no_id_cannot_be_replied_to(self) -> None:
        # The client records an omitted id as an empty string rather than
        # inventing one, so this is what a missing id really looks like.
        problems = validate_call_identity([Call(id="")])
        assert any("no id" in problem for problem in problems)

    def test_reused_ids_within_one_message_are_rejected(self) -> None:
        # Two calls sharing an id make the tool results ambiguous.
        problems = validate_call_identity([Call(), Call(name="get_time")])
        assert any("reused" in problem for problem in problems)

    def test_distinct_ids_are_accepted(self) -> None:
        calls = [Call(), Call(name="get_time", id="call-2")]
        assert validate_call_identity(calls) == []


class TestCallArguments:
    def test_a_json_object_is_accepted(self) -> None:
        assert validate_call_arguments([Call()]) == []

    def test_unparseable_arguments_are_rejected(self) -> None:
        problems = validate_call_arguments([Call(arguments_text='{"location": Denver')])
        assert any("not valid JSON" in problem for problem in problems)

    def test_a_json_array_is_not_an_arguments_object(self) -> None:
        problems = validate_call_arguments([Call(arguments_text='["Denver"]')])
        assert any("expected a JSON object" in problem for problem in problems)

    def test_a_bare_string_is_not_an_arguments_object(self) -> None:
        problems = validate_call_arguments([Call(arguments_text='"Denver"')])
        assert any("expected a JSON object" in problem for problem in problems)

    def test_empty_arguments_are_rejected(self) -> None:
        problems = validate_call_arguments([Call(arguments_text="")])
        assert any("no arguments text" in problem for problem in problems)

    def test_an_empty_object_is_fine(self) -> None:
        # A tool with no required parameters legitimately gets "{}".
        assert validate_call_arguments([Call(arguments_text="{}")]) == []


class TestCallsWereOffered:
    def test_an_offered_call_is_accepted(self) -> None:
        assert validate_calls_were_offered([Call()], ["get_weather"]) == []

    def test_a_builtin_the_caller_never_offered_is_rejected(self) -> None:
        problems = validate_calls_were_offered([Call(name="print")], ["get_weather"])
        assert any("names no offered tool" in problem for problem in problems)

    def test_no_offered_names_disables_the_check(self) -> None:
        # Nothing to check against is not the same as nothing being allowed;
        # the caller decides whether an empty offer is itself a failure.
        assert validate_calls_were_offered([Call(name="print")], []) == []


class TestOfferedToolNames:
    def test_names_are_pulled_from_an_openai_tools_array(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ]
        assert offered_tool_names(tools) == ("get_weather", "get_time")

    def test_malformed_entries_are_skipped_rather_than_raising(self) -> None:
        tools = [
            {"type": "function"},
            {"type": "function", "function": {"description": "no name"}},
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        assert offered_tool_names(tools) == ("get_weather",)

    def test_no_tools_yields_no_names(self) -> None:
        assert offered_tool_names(None) == ()
        assert offered_tool_names([]) == ()
