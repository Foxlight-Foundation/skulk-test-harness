from pathlib import Path

from skulk_test_harness.client import _extract_stream_delta, _extract_stream_logprobs
from skulk_test_harness.models import (
    ExpectedToolCall,
    SuccessCriteria,
    ToolCallRecord,
    ToolMock,
)
from skulk_test_harness.orchestrator import (
    _placement_from_preview,
    _score_output,
    _store_registry_entries,
    _tool_roundtrip_messages,
)
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

    assert model_sets["gpt-oss-20b"].models == [
        "mlx-community/gpt-oss-20b-MXFP4-Q8"
    ]
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

    assert model_sets["gguf-llama-cpp"].models == [
        "unsloth/Llama-3.2-1B-Instruct-GGUF"
    ]
    suite = test_sets["llama-cpp"]
    names = {test.name for test in suite.tests}
    assert {"ordered-integers-coherence", "logprobs-parity", "tool-call-path"} <= names

    logprobs_test = next(t for t in suite.tests if t.name == "logprobs-parity")
    assert logprobs_test.top_logprobs == 3
    assert logprobs_test.success.require_logprobs is True
