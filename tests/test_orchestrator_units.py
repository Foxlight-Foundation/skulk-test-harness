import json
import math
import shlex
from pathlib import Path

import httpx
import pytest

from skulk_test_harness.client import (
    AudioSpeechExecution,
    AudioTranscriptionExecution,
    ChatExecution,
    SkulkApiError,
    SkulkClient,
    _extract_stream_delta,
    _extract_stream_logprobs,
)
from skulk_test_harness.models import (
    ExpectedToolCall,
    GenerationMetrics,
    HarnessConfig,
    Issue,
    ModelRef,
    ModelSelector,
    PlacementResult,
    PromptImage,
    PromptTest,
    RunReport,
    RunSpec,
    SuccessCriteria,
    ToolCallRecord,
    ToolMock,
)
from skulk_test_harness.models import (
    TestSet as HarnessTestSet,
)
from skulk_test_harness.orchestrator import (
    HarnessRunner,
    _clear_deferred_placement_issues,
    _first_stt_model_id,
    _messages_for_test,
    _placement_from_preview,
    _score_audio_output,
    _score_output,
    _score_streaming_audio_output,
    _select_catalog_models,
    _store_registry_entries,
    _tool_roundtrip_messages,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.specs import load_config, load_model_sets, load_test_sets


def test_store_registry_entries_supports_live_entries_shape() -> None:
    registry = {
        "entries": [
            {"model_id": "mlx-community/Foo-4bit"},
            "bad",
            {"model_id": "mlx-community/Bar-8bit"},
        ]
    }

    assert _store_registry_entries(registry) == [
        {"model_id": "mlx-community/Foo-4bit"},
        {"model_id": "mlx-community/Bar-8bit"},
    ]


def test_public_and_foxlight_example_configs_load() -> None:
    root = Path(__file__).parents[1]
    public_path = root / "skulk-harness.example.yaml"
    foxlight_path = root / "examples/foxlight/skulk-harness.yaml"
    stability_path = root / "examples/foxlight/skulk-harness.stability.example.yaml"

    assert public_path.exists()
    assert foxlight_path.exists()
    assert stability_path.exists()

    public_config = load_config(public_path)
    foxlight_config = load_config(foxlight_path)
    stability_config = load_config(stability_path)

    assert public_config.api_base_url == "http://localhost:52415"
    assert public_config.model_sets_path == Path("configs/model_sets.yaml")
    assert public_config.cluster_nodes == {}
    assert foxlight_config.api_base_url == "http://kite1:52415"
    assert foxlight_config.model_sets_path == Path("examples/foxlight/model_sets.yaml")
    assert foxlight_config.test_sets_path == Path("examples/foxlight/test_sets.yaml")
    assert foxlight_config.cluster_nodes == {}
    assert sorted(stability_config.cluster_nodes) == ["node-a", "node-b"]
    assert all(
        node.relaunch_command is not None
        for node in stability_config.cluster_nodes.values()
    )


def test_placement_from_preview_extracts_nodes_and_runners() -> None:
    preview = {
        "sharding": "Pipeline",
        "instance_meta": "MlxRing",
        "instance": {
            "MlxRingInstance": {
                "shardAssignments": {
                    "nodeToRunner": {"node-a": "runner-a"},
                    "runnerToShard": {"runner-a": {"PipelineShardMetadata": {}}},
                }
            }
        },
    }

    placement = _placement_from_preview("mlx-community/Foo", preview)

    assert placement.node_ids == ["node-a"]
    assert placement.runner_ids == ["runner-a"]
    assert placement.sharding == "Pipeline"
    assert placement.instance_meta == "MlxRingInstance"


def test_score_output_accepts_single_file_asteroids_artifact() -> None:
    text = """```html
<!doctype html>
<html>
<body>
<canvas id="game"></canvas>
<script>
function loop(){ requestAnimationFrame(loop); }
addEventListener("keydown", () => {});
loop();
</script>
</body>
</html>
```"""
    issues = _score_output(
        "model",
        "asteroids",
        text,
        SuccessCriteria(
            min_code_block_chars=40,
            require_html_artifact=True,
            required_substrings=["canvas", "requestAnimationFrame"],
        ),
    )

    assert issues == []


def test_score_output_accepts_in_order_integer_sequence() -> None:
    text = "Sure: " + " ".join(str(n) for n in range(1, 31))
    issues = _score_output(
        "model",
        "coherence",
        text,
        SuccessCriteria(in_order_integers=30),
    )

    assert issues == []


def test_score_output_flags_transposed_integer_sequence() -> None:
    # 8 and 12 swapped, exactly what a per-token delivery reorder produces.
    order = list(range(1, 31))
    order[7], order[11] = order[11], order[7]
    text = " ".join(str(n) for n in order)

    issues = _score_output(
        "model",
        "coherence",
        text,
        SuccessCriteria(in_order_integers=30),
    )

    assert len(issues) == 1
    assert "ascending order" in issues[0].message
    # The transposition survives into the evidence: 12 now sits where 8 was.
    emitted_sequence = issues[0].evidence["emitted_sequence"]
    assert isinstance(emitted_sequence, list)
    assert emitted_sequence[7] == 12


def test_score_output_in_order_integers_ignores_absent_sequence() -> None:
    # Model answered something else; too few integers to judge ordering, so the
    # coherence criterion stays silent (min_chars/substrings cover content).
    issues = _score_output(
        "model",
        "coherence",
        "I would rather not count today.",
        SuccessCriteria(min_chars=0, in_order_integers=30),
    )

    assert issues == []


def test_score_output_accepts_numbered_structured_list() -> None:
    text = "1. first reason\n2. second reason\n3. third reason"
    issues = _score_output(
        "model",
        "structured-list",
        text,
        SuccessCriteria(min_chars=0, min_list_items=3),
    )

    assert issues == []


def test_score_output_accepts_unicode_bullet_structured_list() -> None:
    text = "• first reason\n• second reason\n• third reason"
    issues = _score_output(
        "model",
        "structured-list",
        text,
        SuccessCriteria(min_chars=0, min_list_items=3),
    )

    assert issues == []


def test_score_output_flags_too_few_structured_list_items() -> None:
    issues = _score_output(
        "model",
        "structured-list",
        "1. one item\nplain continuation",
        SuccessCriteria(min_chars=0, min_list_items=3),
    )

    assert len(issues) == 1
    assert "too few structured list items" in issues[0].message


def test_extract_stream_delta_captures_tool_calls() -> None:
    event: dict[str, object] = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call-weather",
                            "index": 0,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": (
                                    '{"location": "Cedar Rapids, Iowa", '
                                    '"units": "fahrenheit"}'
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }

    content, reasoning, tool_calls = _extract_stream_delta(event)

    assert content == ""
    assert reasoning == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {
        "location": "Cedar Rapids, Iowa",
        "units": "fahrenheit",
    }


def test_score_output_accepts_expected_tool_calls() -> None:
    issues = _score_output(
        "model",
        "calculator",
        "",
        SuccessCriteria(
            min_chars=0,
            min_tool_calls=1,
            expected_tool_calls=[
                ExpectedToolCall(
                    name="calculate",
                    required_arguments=["expression"],
                    argument_substrings={"expression": "187"},
                )
            ],
        ),
        tool_calls=[
            ToolCallRecord(
                id="call-calculate",
                name="calculate",
                arguments_text='{"expression": "187 * 493"}',
                arguments={"expression": "187 * 493"},
                index=0,
            )
        ],
    )

    assert issues == []


def test_score_output_treats_tool_call_as_coherent_completion() -> None:
    issues = _score_output(
        "model",
        "tool-call-path",
        "",
        SuccessCriteria(min_chars=2),
        tool_calls=[
            ToolCallRecord(
                id="call-weather",
                name="get_weather",
                arguments_text='{"city": "San Francisco"}',
                arguments={"city": "San Francisco"},
                index=0,
            )
        ],
    )

    assert issues == []


def test_score_output_flags_marker_leak_in_reasoning_channel() -> None:
    # The Gemma channel parser must strip <|channel> markers from BOTH channels;
    # a marker surfacing in reasoning_content is a regression even when content
    # is clean. forbid_in_reasoning is on by default.
    issues = _score_output(
        "gemma",
        "reasoning-split",
        "They meet at 11:00 am.",
        SuccessCriteria(forbidden_substrings=["<|channel>", "<channel|>"]),
        reasoning_text="<|channel>thought\nlet me compute the closing speed",
    )

    assert len(issues) == 1
    assert "forbidden substring" in issues[0].message
    assert "reasoning" in issues[0].message


def test_score_output_marker_leak_ignored_when_forbid_in_reasoning_false() -> None:
    issues = _score_output(
        "gemma",
        "reasoning-split",
        "Clean visible answer.",
        SuccessCriteria(
            forbidden_substrings=["<|channel>"],
            forbid_in_reasoning=False,
        ),
        reasoning_text="<|channel>thought\nstill thinking",
    )

    assert issues == []


def test_score_output_requires_separated_reasoning_present() -> None:
    criteria = SuccessCriteria(min_chars=0, min_reasoning_chars=40)
    # Reasoning swallowed into content (empty reasoning channel) -> fail.
    swallowed = _score_output(
        "gemma", "reasoning-split", "long visible answer " * 10, criteria
    )
    assert len(swallowed) == 1
    assert "reasoning not split" in swallowed[0].message
    # Properly separated reasoning of sufficient length -> pass.
    split = _score_output(
        "gemma",
        "reasoning-split",
        "11:00 am.",
        criteria,
        reasoning_text="closing speed is 110 mph; remaining gap after 9:30 is 180 miles",
    )
    assert split == []


def test_score_output_accepts_reasoning_only_generated_text() -> None:
    criteria = SuccessCriteria(min_chars=0, min_generated_chars=40)

    issues = _score_output(
        "reasoning-model",
        "generated-presence",
        "",
        criteria,
        reasoning_text="reasoning-only generation can still be a healthy response",
    )

    assert issues == []

    empty = _score_output("reasoning-model", "generated-presence", "", criteria)
    assert len(empty) == 1
    assert "content + reasoning" in empty[0].message
    assert empty[0].evidence == {"content_chars": 0, "reasoning_chars": 0}


def test_score_output_throughput_floor_catches_silent_mtp_fallback() -> None:
    criteria = SuccessCriteria(min_chars=0, min_wall_tps=16.0)
    # Below the floor (a silent draft-mtp fallback) -> RED.
    slow = _score_output("served", "throughput", "ok", criteria, wall_tps=12.7)
    assert len(slow) == 1
    assert "below required floor" in slow[0].message
    assert slow[0].evidence["wall_tps"] == 12.7
    # At/above the floor (MTP active) -> pass.
    fast = _score_output("served", "throughput", "ok", criteria, wall_tps=28.0)
    assert fast == []
    # No measurable rate (failed/empty generation) -> the floor is skipped; the
    # content checks speak to that case instead.
    no_rate = _score_output("served", "throughput", "ok", criteria, wall_tps=None)
    assert no_rate == []


def test_score_audio_output_accepts_wav_bytes() -> None:
    audio = b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 2048)

    issues = _score_audio_output(
        "tts-model",
        "speech",
        audio,
        SuccessCriteria(min_audio_bytes=1024),
        response_format="wav",
        media_type="audio/wav",
    )

    assert issues == []


def test_score_audio_output_flags_short_non_audio_response() -> None:
    issues = _score_audio_output(
        "tts-model",
        "speech",
        b"oops",
        SuccessCriteria(min_audio_bytes=1024),
        response_format="wav",
        media_type="application/json",
    )

    assert [issue.message for issue in issues] == [
        "Audio response shorter than required minimum (4 < 1024 bytes)",
        "Audio response had non-audio media type 'application/json'",
        "WAV response did not contain a RIFF/WAVE header",
    ]


def test_tool_roundtrip_messages_include_assistant_and_tool_results() -> None:
    issues = []
    messages = _tool_roundtrip_messages(
        [{"role": "user", "content": "weather?"}],
        [
            ToolCallRecord(
                id="call-weather",
                name="get_weather",
                arguments_text='{"location": "Cedar Rapids"}',
                arguments={"location": "Cedar Rapids"},
                index=0,
            )
        ],
        [ToolMock(name="get_weather", content='{"temperature_f":72}')],
        model_id="model",
        test_name="weather",
        issues=issues,
    )

    assert issues == []
    assert messages[-2]["role"] == "assistant"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-weather",
        "name": "get_weather",
        "content": '{"temperature_f":72}',
    }


def test_messages_for_test_builds_multimodal_content() -> None:
    test = PromptTest(
        name="vision",
        prompt="what color?",
        images=[PromptImage(url="data:image/png;base64,AAAA")],
    )

    messages = _messages_for_test(test)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what color?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]


def test_served_spec_selector_matches_runtime_field() -> None:
    selector = load_model_sets(
        Path(__file__).parents[1] / "configs/model_sets.yaml"
    ).model_sets["served-spec-draft-eagle3"].selectors[0]
    catalog = [
        {"id": "org/plain", "runtime": {"served_spec_type": "draft_mtp"}},
        {"id": "org/eagle", "runtime": {"served_spec_type": "draft_eagle3"}},
    ]

    selected = _select_catalog_models(catalog, selector)

    assert [item["id"] for item in selected] == ["org/eagle"]


def test_capability_selector_matches_resolved_speech_flags() -> None:
    selector = ModelSelector(source="both", capabilities_any=["tts"])
    catalog = [
        {
            "id": "org/plain",
            "resolved_capabilities": {"supports_transcription": True},
        },
        {
            "id": "org/tts",
            "resolved_capabilities": {"supports_speech_synthesis": True},
        },
    ]

    selected = _select_catalog_models(catalog, selector)

    assert [item["id"] for item in selected] == ["org/tts"]


def test_streaming_audio_selector_requires_streaming_card_metadata() -> None:
    selector = ModelSelector(
        source="both",
        capabilities_any=["tts"],
        require_audio_streaming=True,
    )
    catalog = [
        {
            "id": "org/batch-tts",
            "capabilities": ["TextToSpeech"],
            "audio": {"kind": "tts", "supports_streaming": False},
        },
        {
            "id": "org/streaming-tts",
            "capabilities": ["TextToSpeech"],
            "audio": {"kind": "tts", "supports_streaming": True},
        },
    ]

    selected = _select_catalog_models(catalog, selector)

    assert [item["id"] for item in selected] == ["org/streaming-tts"]


def test_capability_selector_matches_raw_speech_aliases() -> None:
    tts_selector = ModelSelector(source="both", capabilities_any=["tts"])
    stt_selector = ModelSelector(source="both", capabilities_any=["stt"])
    catalog = [
        {"id": "org/text-to-speech", "capabilities": ["TextToSpeech"]},
        {"id": "org/speech-synthesis", "capabilities": ["speech_synthesis"]},
        {"id": "org/speech-to-text", "capabilities": ["SpeechToText"]},
    ]

    selected_tts = _select_catalog_models(catalog, tts_selector)
    selected_stt = _select_catalog_models(catalog, stt_selector)

    assert [item["id"] for item in selected_tts] == [
        "org/text-to-speech",
        "org/speech-synthesis",
    ]
    assert [item["id"] for item in selected_stt] == ["org/speech-to-text"]


def test_first_stt_model_id_reads_speech_metadata() -> None:
    catalog = [
        {"id": "org/TTS", "tags": ["tts"]},
        {
            "id": "org/ResolvedSTT",
            "resolved_capabilities": {"supports_transcription": True},
        },
        {"id": "org/TaggedSTT", "tags": ["stt"]},
    ]

    assert _first_stt_model_id(catalog, exclude_model_id="org/TTS") == (
        "org/ResolvedSTT"
    )
    assert _first_stt_model_id(catalog, exclude_model_id="org/ResolvedSTT") == (
        "org/TaggedSTT"
    )


def test_public_default_sets_are_cluster_neutral() -> None:
    root = Path(__file__).parents[1]
    model_sets = load_model_sets(root / "configs/model_sets.yaml").model_sets
    test_sets = load_test_sets(root / "configs/test_sets.yaml").test_sets

    assert {
        "store-smoke",
        "store-all",
        "catalog-small-text",
        "embeddings",
        "speech-tts",
        "speech-tts-streaming",
        "speech-roundtrip-tts",
        "speech-stt",
        "vision",
        "served-spec-draft-simple",
        "served-spec-draft-eagle3",
    } <= set(model_sets)
    assert {
        "chat-tests",
        "code-tests",
        "tool-tests",
        "throughput",
        "cancellation",
        "context-admission",
        "embeddings",
        "speech-synthesis",
        "speech-synthesis-streaming",
        "speech-data-pressure",
        "speech-roundtrip",
        "vision",
        "served-speculation",
    } <= set(test_sets)
    assert "gpt-oss-20b" not in model_sets
    assert model_sets["speech-tts"].models == []
    assert model_sets["speech-roundtrip-tts"].models == []
    assert model_sets["speech-stt"].models == []
    assert model_sets["speech-tts"].selectors
    assert model_sets["speech-roundtrip-tts"].selectors
    assert model_sets["speech-stt"].selectors
    roundtrip_test = test_sets["speech-roundtrip"].tests[0]
    assert roundtrip_test.transcription_model_id is None

    tool_suite = test_sets["tool-tests"]
    node_test = next(
        test
        for test in tool_suite.tests
        if test.name == "parallel-node-diagnostic-tool-calls"
    )
    expected_hosts = [
        call.argument_substrings["hostname"]
        for call in node_test.success.expected_tool_calls
    ]
    assert expected_hosts == ["node-a", "node-b"]


def test_foxlight_gpt_oss_complete_suite_loads_tool_tests() -> None:
    root = Path(__file__).parents[1]
    model_sets = load_model_sets(
        root / "examples/foxlight/model_sets.yaml"
    ).model_sets
    test_sets = load_test_sets(root / "examples/foxlight/test_sets.yaml").test_sets

    assert "gpt-oss-20b" in model_sets
    suite = test_sets["gpt-oss-20b-complete"]
    tool_tests = [test for test in suite.tests if test.kind == "tool"]

    assert model_sets["gpt-oss-20b"].models == ["mlx-community/gpt-oss-20b-MXFP4-Q8"]
    assert len(suite.tests) >= 8
    assert len(tool_tests) == 3
    assert all(test.tools for test in tool_tests)
    assert sum(1 for test in tool_tests if test.tool_mocks) == 2


def test_foxlight_e2e_battery_references_defined_sets() -> None:
    root = Path(__file__).parents[1]
    model_sets = load_model_sets(
        root / "examples/foxlight/model_sets.yaml"
    ).model_sets
    test_sets = load_test_sets(root / "examples/foxlight/test_sets.yaml").test_sets
    battery = root / "examples/foxlight/run_e2e_battery.sh"

    cells = [
        shlex.split(line.strip())
        for line in battery.read_text().splitlines()
        if line.strip().startswith("cell ")
    ]

    assert cells
    assert all(cell[1] in model_sets for cell in cells)
    assert all(cell[2] in test_sets for cell in cells)


def test_score_output_require_logprobs_fails_when_absent() -> None:
    issues = _score_output(
        "model",
        "logprobs-parity",
        "hello",
        SuccessCriteria(min_chars=1, require_logprobs=True),
        logprob_tokens=0,
    )
    assert any("logprobs" in issue.message.lower() for issue in issues)


def test_score_output_require_logprobs_passes_when_present() -> None:
    issues = _score_output(
        "model",
        "logprobs-parity",
        "hello",
        SuccessCriteria(min_chars=1, require_logprobs=True),
        logprob_tokens=5,
    )
    assert issues == []


def test_extract_stream_logprobs_counts_tokens_and_top() -> None:
    event: dict[str, object] = {
        "choices": [
            {
                "delta": {"content": "Hi"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hi",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "Hi", "logprob": -0.2},
                                {"token": "Hey", "logprob": -1.5},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    assert _extract_stream_logprobs(event) == (1, 1)


def test_extract_stream_logprobs_zero_without_logprobs() -> None:
    no_logprobs: dict[str, object] = {"choices": [{"delta": {"content": "x"}}]}
    assert _extract_stream_logprobs(no_logprobs) == (0, 0)
    assert _extract_stream_logprobs({}) == (0, 0)
    # logprob entry without ranked alternatives counts as a token, not a top.
    event: dict[str, object] = {
        "choices": [{"logprobs": {"content": [{"token": "a", "logprob": -1.0}]}}]
    }
    assert _extract_stream_logprobs(event) == (1, 0)


def test_llama_cpp_suite_and_gguf_set_load() -> None:
    root = Path(__file__).parents[1]
    model_sets = load_model_sets(
        root / "examples/foxlight/model_sets.yaml"
    ).model_sets
    test_sets = load_test_sets(root / "examples/foxlight/test_sets.yaml").test_sets

    assert model_sets["gguf-llama-cpp"].models == ["unsloth/Llama-3.2-1B-Instruct-GGUF"]
    suite = test_sets["llama-cpp"]
    names = {test.name for test in suite.tests}
    assert {
        "ordered-integers-coherence",
        "tool-call-path",
        "harmony-marker-leak-guard",
    } <= names

    # logprobs-parity was removed from the default llama.cpp suite: per-token
    # logprobs need a logits_all-enabled placement (opt-in), and a normal GGUF
    # placement correctly returns none, so requiring them flagged expected
    # behavior as a failure. Testing logprobs belongs in a dedicated placement.
    assert "logprobs-parity" not in names


def test_chat_and_llama_concise_tests_have_reasoning_budget() -> None:
    root = Path(__file__).parents[1]
    public_sets = load_test_sets(root / "configs/test_sets.yaml").test_sets
    foxlight_sets = load_test_sets(root / "examples/foxlight/test_sets.yaml").test_sets

    for test_sets, set_name in (
        (public_sets, "chat-tests"),
        (foxlight_sets, "llama-cpp"),
    ):
        concise = next(
            test
            for test in test_sets[set_name].tests
            if test.name == "concise-factual-answer"
        )
        assert concise.max_tokens >= 256


def test_vision_smoke_fixture_expects_blue_square() -> None:
    root = Path(__file__).parents[1]
    test_sets = load_test_sets(root / "configs/test_sets.yaml").test_sets
    vision = test_sets["vision"].tests[0]

    assert vision.success.required_substrings == ["blue"]
    assert "ABAAAAAQ" in vision.images[0].url


def test_default_placement_appearance_timeout_handles_large_cached_models() -> None:
    assert HarnessConfig().placement_appearance_timeout_s >= 300


# --- teardown sweep + thinking-default regression tests -------------------


def _runner() -> HarnessRunner:
    return HarnessRunner(config=HarnessConfig(), model_sets={}, test_sets={})


def _report() -> RunReport:
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    return RunReport.start("run-1", spec, [])


class _FakeClient:
    """Minimal SkulkClient stand-in recording teardown + chat calls."""

    def __init__(
        self,
        *,
        live_placements: list[PlacementResult] | None = None,
        not_found_ids: set[str] | None = None,
        models: list[dict[str, object]] | None = None,
    ) -> None:
        self._live = live_placements or []
        self._not_found = not_found_ids or set()
        self._models = models or []
        self.deleted: list[str] = []
        self.evicted: list[str] = []
        self.thinking_seen: list[bool | None] = []
        self.speech_requests: list[dict[str, object]] = []
        self.transcription_requests: list[dict[str, object]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def get_cluster_api_urls(self) -> list[str]:
        return ["http://owner-a", "http://owner-b"]

    def find_placements_for_model(self, model_id: str) -> list[PlacementResult]:
        return [p for p in self._live if p.model_id == model_id]

    def delete_instance(self, instance_id: str) -> None:
        self.deleted.append(instance_id)
        if instance_id in self._not_found:
            raise SkulkApiError("DELETE", f"/instance/{instance_id}", 404, "missing")

    def delete_store_model(
        self, model_id: str, *, timeout_s: float | None = None
    ) -> None:
        del timeout_s
        self.evicted.append(model_id)

    def list_models(self) -> list[dict[str, object]]:
        return self._models

    def audio_speech(
        self,
        *,
        model_id: str,
        input_text: str,
        response_format: str = "wav",
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        streaming_interval: float | None = None,
        read_delay_s: float = 0.0,
    ) -> AudioSpeechExecution:
        self.speech_requests.append(
            {
                "model_id": model_id,
                "input_text": input_text,
                "response_format": response_format,
                "voice": voice,
                "speed": speed,
                "stream": stream,
                "streaming_interval": streaming_interval,
                "read_delay_s": read_delay_s,
            }
        )
        audio = (
            b"ID3" + (b"\x00" * 2048)
            if response_format == "mp3"
            else b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 2048)
        )
        return AudioSpeechExecution(
            audio=audio,
            media_type="audio/mpeg" if response_format == "mp3" else "audio/wav",
            elapsed_s=0.8 if stream else 0.02,
            response_format=response_format,
            chunks=3 if stream else 1,
            first_byte_s=0.01 if stream else None,
            chunk_sizes=[3, 1024, 1024] if stream else [len(audio)],
            chunk_arrival_s=[0.01, 0.35, 0.75] if stream else [],
            streaming=stream,
        )

    def audio_transcription(
        self,
        *,
        model_id: str,
        audio: bytes,
        filename: str,
        media_type: str,
        response_format: str = "json",
        language: str | None = None,
        prompt: str | None = None,
    ) -> AudioTranscriptionExecution:
        self.transcription_requests.append(
            {
                "model_id": model_id,
                "audio": audio,
                "filename": filename,
                "media_type": media_type,
                "response_format": response_format,
                "language": language,
                "prompt": prompt,
            }
        )
        return AudioTranscriptionExecution(
            text="hello world",
            media_type="application/json",
            elapsed_s=0.03,
            response_format=response_format,
            raw_response={"text": "hello world"},
        )

    def stream_chat(self, *, enable_thinking=None, **_kwargs) -> ChatExecution:
        self.thinking_seen.append(enable_thinking)
        return ChatExecution(
            text="ok",
            reasoning_text="",
            tool_calls=[],
            metrics=GenerationMetrics(elapsed_s=0.01),
            command_id=None,
            raw_events=[],
        )


def test_teardown_sweeps_reissued_orphan_instance() -> None:
    # Original id 404s (superseded), but a re-IDed instance is still live and
    # must be swept so it does not starve the next cell.
    client = _FakeClient(
        live_placements=[PlacementResult(model_id="m/Foo", instance_id="new-id")],
        not_found_ids={"orig-id"},
    )
    runner = _runner()
    report = _report()

    deleted = runner._teardown_harness_instances(client, "m/Foo", "orig-id", report)  # type: ignore[arg-type]

    assert client.deleted == ["orig-id", "new-id"]
    assert deleted is True
    # The 404 on the original id is benign and must not raise a warning.
    assert [i.message for i in report.issues] == []


def test_teardown_surfaces_non_404_delete_failure() -> None:
    class _Boom(_FakeClient):
        def delete_instance(self, instance_id: str) -> None:
            self.deleted.append(instance_id)
            raise SkulkApiError("DELETE", f"/instance/{instance_id}", 500, "boom")

    client = _Boom()
    runner = _runner()
    report = _report()

    deleted = runner._teardown_harness_instances(client, "m/Foo", "only-id", report)  # type: ignore[arg-type]

    assert client.deleted == ["only-id"]
    assert deleted is False
    assert any("Failed to delete" in i.message for i in report.issues)


def test_resolved_thinking_toggle_reads_resolved_capabilities() -> None:
    client = SkulkClient("http://localhost:9")
    models = [
        {
            "id": "m/Toggle",
            "resolved_capabilities": {"supports_thinking_toggle": True},
        },
        {
            "id": "m/NoToggle",
            "resolved_capabilities": {"supports_thinking_toggle": False},
        },
        {"id": "m/Missing"},
    ]
    client.list_models = lambda: models  # type: ignore[method-assign]
    try:
        toggles = client.resolved_thinking_toggle_by_model()
    finally:
        client.close()
    # m/Missing has no resolved_capabilities block, so it is absent from the
    # map -- the harness then falls back to omitting the toggle for it.
    assert toggles == {"m/Toggle": True, "m/NoToggle": False}


def test_run_test_defaults_thinking_off_for_toggle_models(tmp_path: Path) -> None:
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(name="t", prompt="hi")  # enable_thinking unset
    runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="m/Foo",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
        thinking_default=False,
    )
    assert client.thinking_seen == [False]


def test_run_test_explicit_thinking_overrides_default(tmp_path: Path) -> None:
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(name="t", prompt="hi", enable_thinking=True)
    runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="m/Foo",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
        thinking_default=False,
    )
    assert client.thinking_seen == [True]


def test_run_test_dispatches_audio_speech(tmp_path: Path) -> None:
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(
        name="tts",
        kind="audio_speech",
        prompt="hello",
        success=SuccessCriteria(min_chars=0, min_audio_bytes=1024),
    )

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    assert "audio_bytes=" in result.output_text
    assert client.speech_requests[0]["model_id"] == "org/TTS"
    assert result.artifact_path is not None
    assert result.artifact_path == tmp_path / "org-tts--tts--rep-1.wav"
    assert result.artifact_path.read_bytes() == (
        b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 2048)
    )


def test_run_test_dispatches_streaming_audio_speech(tmp_path: Path) -> None:
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(
        name="tts-stream",
        kind="audio_speech_streaming",
        prompt="hello",
        audio_response_format="mp3",
        speech_streaming_interval=0.25,
        success=SuccessCriteria(
            min_chars=0,
            min_audio_bytes=1024,
            min_stream_chunks=2,
            min_stream_span_s=0.5,
        ),
    )

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    artifact_path = result.artifact_path
    assert artifact_path == tmp_path / "org-tts--tts-stream--rep-1.mp3"
    assert artifact_path is not None
    assert artifact_path.read_bytes().startswith(b"ID3")
    assert result.metrics.ttft_s == 0.01
    assert result.metrics.chunks == 3
    assert "streaming=True" in result.output_text
    assert "chunks=3" in result.output_text
    assert "stream_span_s=0.740" in result.output_text
    assert client.speech_requests[0]["stream"] is True
    assert client.speech_requests[0]["streaming_interval"] == 0.25

    sidecar = artifact_path.with_suffix(".mp3.stream.json")
    assert f"stream_metadata={sidecar}" in result.output_text
    assert json.loads(sidecar.read_text()) == {
        "chunks": 3,
        "first_byte_s": 0.01,
        "stream_span_s": 0.74,
        "chunk_sizes": [3, 1024, 1024],
        "chunk_arrival_s": [0.01, 0.35, 0.75],
    }


def test_run_test_drives_multi_owner_speech_pressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    pressure_clients: list[_FakeClient] = []
    runner = _runner()

    def client_for_url(_url: str) -> _FakeClient:
        pressure_client = _FakeClient()
        pressure_clients.append(pressure_client)
        return pressure_client

    monkeypatch.setattr(runner, "_client_for_url", client_for_url)
    test = PromptTest(
        name="tts-pressure",
        kind="audio_speech_pressure",
        prompt="hello",
        audio_response_format="mp3",
        speech_concurrency=4,
        speech_requests_per_worker=2,
        speech_owner_count=2,
        speech_slow_workers=1,
        speech_slow_reader_delay_s=0.25,
        success=SuccessCriteria(
            min_chars=0,
            min_audio_bytes=1024,
            min_stream_chunks=2,
            min_stream_span_s=0.5,
        ),
    )

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    assert "requests=8 successes=8 failures=0 owners=2" in result.output_text
    assert len(pressure_clients) == 4
    assert sum(len(item.speech_requests) for item in pressure_clients) == 8
    assert sum(
        request["read_delay_s"] == 0.25
        for item in pressure_clients
        for request in item.speech_requests
    ) == 2
    assert len(list(tmp_path.glob("*.mp3"))) == 8
    assert len(list(tmp_path.glob("*.stream.json"))) == 8


def test_score_streaming_audio_output_rejects_burst_chunks() -> None:
    execution = AudioSpeechExecution(
        audio=b"ID3" + (b"\x00" * 2048),
        media_type="audio/mpeg",
        elapsed_s=31.63,
        response_format="mp3",
        chunks=46,
        first_byte_s=31.62,
        chunk_sizes=[1024, 1024],
        chunk_arrival_s=[31.62, 31.626],
        streaming=True,
    )

    issues = _score_streaming_audio_output(
        "org/TTS",
        "tts-stream",
        execution,
        SuccessCriteria(min_stream_chunks=2, min_stream_span_s=0.5),
    )

    assert len(issues) == 1
    assert issues[0].message == "Streaming response did not span the configured duration"
    assert issues[0].evidence["min_stream_span_s"] == 0.5
    stream_span_s = issues[0].evidence["stream_span_s"]
    assert isinstance(stream_span_s, float)
    assert math.isclose(stream_span_s, 0.006, abs_tol=0.001)


def test_run_test_dispatches_audio_transcription(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 32))
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(
        name="stt",
        kind="audio_transcription",
        prompt="transcribe this",
        input_audio_path=audio_path,
        success=SuccessCriteria(min_chars=5, required_substrings=["hello"]),
    )

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/STT",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    assert result.output_text == "hello world"
    assert client.transcription_requests[0]["filename"] == "sample.wav"


def test_audio_transcription_infers_fixture_media_type(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"ID3" + (b"\x00" * 32))
    client = _FakeClient()
    runner = _runner()
    test = PromptTest(
        name="stt",
        kind="audio_transcription",
        prompt="transcribe this",
        input_audio_path=audio_path,
        success=SuccessCriteria(min_chars=5, required_substrings=["hello"]),
    )

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/STT",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    assert client.transcription_requests[0]["media_type"] == "audio/mpeg"


def test_speech_roundtrip_records_secondary_placement_transport_error(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    runner = _runner()

    def _raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise httpx.ReadTimeout("preview timed out")

    runner._ensure_model_placed = _raise_timeout  # type: ignore[method-assign]
    test = PromptTest(
        name="roundtrip",
        kind="speech_roundtrip",
        prompt="hello",
        transcription_model_id="org/STT",
    )
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    report = _report()

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
        spec=spec,
        report=report,
    )

    assert result.passed is False
    assert result.issues[0].message == "Speech roundtrip request failed"
    assert "preview timed out" in str(result.issues[0].evidence["error"])
    assert result.issues[0].evidence["transcription_model_id"] == "org/STT"


def test_speech_roundtrip_records_stt_discovery_transport_error(
    tmp_path: Path,
) -> None:
    class _TimeoutListModelsClient(_FakeClient):
        def list_models(self) -> list[dict[str, object]]:
            raise httpx.ReadTimeout("models timed out")

    runner = _runner()
    test = PromptTest(
        name="roundtrip",
        kind="speech_roundtrip",
        prompt="hello",
    )
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    report = _report()

    result = runner._run_test(
        _TimeoutListModelsClient(),  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
        spec=spec,
        report=report,
    )

    assert result.passed is False
    assert result.issues[0].message == "Speech roundtrip request failed"
    assert "models timed out" in str(result.issues[0].evidence["error"])
    assert result.issues[0].evidence["transcription_model_id"] is None


def test_speech_roundtrip_persists_generated_audio_artifact(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    runner = _runner()
    runner._ensure_model_placed = lambda *_args, **_kwargs: PlacementResult(  # type: ignore[method-assign]
        model_id="org/STT",
        instance_id="stt-instance",
        ready=True,
        created_by_harness=False,
    )
    test = PromptTest(
        name="roundtrip",
        kind="speech_roundtrip",
        prompt="hello",
        transcription_model_id="org/STT",
        success=SuccessCriteria(
            min_chars=5,
            min_audio_bytes=1024,
            required_substrings=["hello"],
        ),
    )
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    report = _report()

    result = runner._run_test(
        client,  # type: ignore[arg-type]
        model_id="org/TTS",
        test=test,
        repetition=1,
        artifact_dir=tmp_path,
        spec=spec,
        report=report,
    )

    assert result.passed is True
    assert result.output_text == "hello world"
    assert result.artifact_path is not None
    assert result.artifact_path == tmp_path / "org-tts--roundtrip--rep-1.wav"
    audio = result.artifact_path.read_bytes()
    assert audio.startswith(b"RIFF")
    assert client.transcription_requests[0]["audio"] == audio


def test_ensure_model_placed_fast_fails_when_instance_never_appears() -> None:
    class _NoAppearanceClient(_FakeClient):
        def get_store_registry(self) -> None:
            return None

        def get_placement_previews(
            self, model_id: str, *, excluded_node_ids=None
        ) -> list[dict[str, object]]:
            del model_id, excluded_node_ids
            return [
                {
                    "sharding": "Pipeline",
                    "instance_meta": "MlxRing",
                    "instance": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "m/Foo",
                                "nodeToRunner": {"node-a": "runner-a"},
                                "runnerToShard": {"runner-a": {}},
                            }
                        }
                    },
                }
            ]

        def place_model(self, **_kwargs) -> None:
            return None

    client = _NoAppearanceClient(models=[{"id": "m/Foo"}])
    runner = HarnessRunner(
        config=HarnessConfig(
            placement_appearance_timeout_s=0.01,
            poll_interval_s=0.001,
        ),
        model_sets={},
        test_sets={},
    )
    report = _report()
    spec = RunSpec(model_set="m", test_set="t", mode="execute")

    placement = runner._ensure_model_placed(client, "m/Foo", spec, report)  # type: ignore[arg-type]

    assert placement is not None
    assert placement.created_by_harness is True
    assert placement.ready is False
    assert any("placement refusal" in issue.message for issue in report.issues)


def test_ensure_store_download_reports_transport_timeout() -> None:
    class _TimeoutDownloadClient(_FakeClient):
        def request_store_download(self, model_id: str) -> None:
            del model_id
            raise httpx.ReadTimeout("timed out")

    runner = _runner()
    report = _report()

    runner._ensure_store_download(
        _TimeoutDownloadClient(),  # type: ignore[arg-type]
        "m/Slow",
        report,
    )

    assert [(issue.severity, issue.message) for issue in report.issues] == [
        ("warning", "Failed to request model-store download")
    ]
    assert "timed out" in str(report.issues[0].evidence["error"])


def test_clear_deferred_placement_issues_keeps_real_run_errors() -> None:
    report = _report()
    report.issues.extend(
        [
            Issue(
                severity="error",
                model_id="m/Foo",
                message="No usable placement preview found before execution",
            ),
            Issue(
                severity="error",
                model_id="m/Foo",
                message="Placement request failed",
            ),
            Issue(
                severity="error",
                model_id="m/Foo",
                message=(
                    "Timed out waiting for placed model to appear in cluster "
                    "state; treating as a placement refusal/give-up"
                ),
            ),
            Issue(
                severity="error",
                model_id="m/Foo",
                message="Model run failed; continuing to next model",
            ),
            Issue(
                severity="error",
                model_id="m/Bar",
                message="No usable placement preview found before execution",
            ),
        ]
    )

    _clear_deferred_placement_issues(report, "m/Foo")

    assert [(issue.model_id, issue.message) for issue in report.issues] == [
        ("m/Foo", "Model run failed; continuing to next model"),
        ("m/Bar", "No usable placement preview found before execution"),
    ]


# --- staged-model eviction gating (only after a real harness teardown) -----


def _drive_lifecycle(
    runner: HarnessRunner,
    client: "_FakeClient",
    *,
    retain_instances: bool,
    delete_staged_models: bool,
    created_by_harness: bool,
    tmp_path: Path,
) -> None:
    placement = PlacementResult(
        model_id="m/Foo",
        instance_id="inst-1",
        created_by_harness=created_by_harness,
        ready=True,
    )
    runner._ensure_model_placed = lambda *_a, **_k: placement  # type: ignore[method-assign]
    spec = RunSpec(
        model_set="m",
        test_set="t",
        mode="execute",
        retain_instances=retain_instances,
        delete_staged_models=delete_staged_models,
    )
    report = RunReport.start("run-1", spec, [])
    runner._run_model_lifecycle(
        client,  # type: ignore[arg-type]
        ModelRef(model_id="m/Foo", source="explicit"),
        spec,
        report,
        HarnessTestSet(name="t", tests=[]),
        ReportWriter(tmp_path),
        {},
    )


def test_eviction_skipped_when_instance_retained(tmp_path: Path) -> None:
    # --delete-staged-models WITHOUT --delete-created-instances: the instance is
    # kept, so its weights MUST NOT be evicted out from under it.
    client = _FakeClient()
    _drive_lifecycle(
        _runner(),
        client,
        retain_instances=True,
        delete_staged_models=True,
        created_by_harness=True,
        tmp_path=tmp_path,
    )
    assert client.deleted == []  # retained: no teardown
    assert client.evicted == []  # and therefore no eviction


def test_eviction_skipped_for_reused_user_instance(tmp_path: Path) -> None:
    # Placement reused an existing (user-owned) instance the harness did not
    # create: never tear it down, never evict its weights.
    client = _FakeClient()
    _drive_lifecycle(
        _runner(),
        client,
        retain_instances=False,
        delete_staged_models=True,
        created_by_harness=False,
        tmp_path=tmp_path,
    )
    assert client.deleted == []
    assert client.evicted == []


def test_eviction_runs_after_harness_teardown(tmp_path: Path) -> None:
    # The intended path: a harness-created instance with teardown enabled is torn
    # down, and only then are its staged weights evicted.
    client = _FakeClient()
    _drive_lifecycle(
        _runner(),
        client,
        retain_instances=False,
        delete_staged_models=True,
        created_by_harness=True,
        tmp_path=tmp_path,
    )
    assert client.deleted == ["inst-1"]
    assert client.evicted == ["m/Foo"]


def test_eviction_skipped_when_teardown_delete_fails(tmp_path: Path) -> None:
    class _Boom(_FakeClient):
        def delete_instance(self, instance_id: str) -> None:
            self.deleted.append(instance_id)
            raise SkulkApiError("DELETE", f"/instance/{instance_id}", 500, "boom")

    client = _Boom()
    _drive_lifecycle(
        _runner(),
        client,
        retain_instances=False,
        delete_staged_models=True,
        created_by_harness=True,
        tmp_path=tmp_path,
    )

    assert client.deleted == ["inst-1"]
    assert client.evicted == []


def test_eviction_skipped_when_placement_never_appears(tmp_path: Path) -> None:
    client = _FakeClient()
    runner = _runner()
    runner._ensure_model_placed = lambda *_a, **_k: PlacementResult(  # type: ignore[method-assign]
        model_id="m/Foo",
        created_by_harness=True,
        ready=False,
    )
    spec = RunSpec(
        model_set="m",
        test_set="t",
        mode="execute",
        delete_staged_models=True,
    )
    report = RunReport.start("run-1", spec, [])

    placed = runner._run_model_lifecycle(
        client,  # type: ignore[arg-type]
        ModelRef(model_id="m/Foo", source="explicit"),
        spec,
        report,
        HarnessTestSet(name="t", tests=[]),
        ReportWriter(tmp_path),
        {},
    )

    assert placed is False
    assert client.deleted == []
    assert client.evicted == []
