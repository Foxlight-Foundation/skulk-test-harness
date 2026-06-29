from pathlib import Path

from skulk_test_harness.client import (
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
    ModelRef,
    PlacementResult,
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
    _placement_from_preview,
    _score_output,
    _store_registry_entries,
    _tool_roundtrip_messages,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.specs import load_model_sets, load_test_sets


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
    # 8 and 12 swapped — exactly what a per-token delivery reorder produces.
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


def test_gpt_oss_complete_suite_loads_tool_tests() -> None:
    root = Path(__file__).parents[1]
    model_sets = load_model_sets(root / "configs/model_sets.yaml").model_sets
    test_sets = load_test_sets(root / "configs/test_sets.yaml").test_sets

    assert "gpt-oss-20b" in model_sets
    suite = test_sets["gpt-oss-20b-complete"]
    tool_tests = [test for test in suite.tests if test.kind == "tool"]

    assert model_sets["gpt-oss-20b"].models == ["mlx-community/gpt-oss-20b-MXFP4-Q8"]
    assert len(suite.tests) >= 8
    assert len(tool_tests) == 3
    assert all(test.tools for test in tool_tests)
    assert sum(1 for test in tool_tests if test.tool_mocks) == 2


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
    model_sets = load_model_sets(root / "configs/model_sets.yaml").model_sets
    test_sets = load_test_sets(root / "configs/test_sets.yaml").test_sets

    assert model_sets["gguf-llama-cpp"].models == ["unsloth/Llama-3.2-1B-Instruct-GGUF"]
    suite = test_sets["llama-cpp"]
    names = {test.name for test in suite.tests}
    assert {"ordered-integers-coherence", "tool-call-path"} <= names

    # logprobs-parity was removed from the default llama.cpp suite: per-token
    # logprobs need a logits_all-enabled placement (opt-in), and a normal GGUF
    # placement correctly returns none, so requiring them flagged expected
    # behavior as a failure. Testing logprobs belongs in a dedicated placement.
    assert "logprobs-parity" not in names


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

    def find_placements_for_model(self, model_id: str) -> list[PlacementResult]:
        return [p for p in self._live if p.model_id == model_id]

    def delete_instance(self, instance_id: str) -> None:
        self.deleted.append(instance_id)
        if instance_id in self._not_found:
            raise SkulkApiError("DELETE", f"/instance/{instance_id}", 404, "missing")

    def delete_store_model(self, model_id: str) -> None:
        self.evicted.append(model_id)

    def list_models(self) -> list[dict[str, object]]:
        return self._models

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

    runner._teardown_harness_instances(client, "m/Foo", "orig-id", report)  # type: ignore[arg-type]

    assert client.deleted == ["orig-id", "new-id"]
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

    runner._teardown_harness_instances(client, "m/Foo", "only-id", report)  # type: ignore[arg-type]

    assert client.deleted == ["only-id"]
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
