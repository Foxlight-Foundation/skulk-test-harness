from skulk_test_harness.client import ChatExecution
from skulk_test_harness.models import GenerationMetrics
from skulk_test_harness.stability import (
    _percentile,
    _placements_for_model_from_state,
    classify_placement_outcome,
    completion_is_coherent,
    summarize_latency,
)

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"


def _execution(text: str, *, chunks: int, elapsed_s: float = 1.0) -> ChatExecution:
    return ChatExecution(
        text=text,
        reasoning_text="",
        tool_calls=[],
        metrics=GenerationMetrics(elapsed_s=elapsed_s, chunks=chunks, output_chars=len(text)),
        command_id="cmd-1",
        raw_events=[],
    )


def _state(*, ready: bool, node_ids: list[str]) -> dict[str, object]:
    node_to_runner = {node: f"runner-{node}" for node in node_ids}
    runner_to_shard = {f"runner-{node}": {"PipelineShardMetadata": {}} for node in node_ids}
    status_tag = "RunnerReady" if ready else "RunnerStarting"
    runners = {f"runner-{node}": {status_tag: {}} for node in node_ids}
    return {
        "instances": {
            "instance-1": {
                "MlxRingInstance": {
                    "shardAssignments": {
                        "modelId": MODEL_ID,
                        "nodeToRunner": node_to_runner,
                        "runnerToShard": runner_to_shard,
                    }
                }
            }
        },
        "runners": runners,
    }


# --- latency aggregation ---------------------------------------------------


def test_percentile_nearest_rank() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(samples, 0.50) == 3.0
    assert _percentile(samples, 0.95) == 5.0
    assert _percentile(samples, 0.0) == 1.0


def test_percentile_empty_returns_none() -> None:
    assert _percentile([], 0.5) is None


def test_summarize_latency_reports_p50_p95_and_failures() -> None:
    samples = [float(n) for n in range(1, 21)]  # 1..20
    summary = summarize_latency(samples, failures=3)

    assert summary.count == 20
    assert summary.failures == 3
    assert summary.min_s == 1.0
    assert summary.max_s == 20.0
    assert summary.mean_s == sum(samples) / len(samples)
    assert summary.p50_s == 10.0
    assert summary.p95_s == 19.0


def test_summarize_latency_empty_keeps_failure_count() -> None:
    summary = summarize_latency([], failures=2)
    assert summary.count == 0
    assert summary.failures == 2
    assert summary.p50_s is None


# --- coherence -------------------------------------------------------------


def test_completion_is_coherent_requires_text_and_chunks() -> None:
    assert completion_is_coherent(_execution("1 2 3", chunks=3)) is True


def test_completion_not_coherent_when_empty() -> None:
    assert completion_is_coherent(_execution("   ", chunks=2)) is False


def test_completion_not_coherent_when_no_chunks() -> None:
    assert completion_is_coherent(_execution("text", chunks=0)) is False


# --- placement-refusal classification --------------------------------------


def test_classify_refused_when_no_instance() -> None:
    verdict, placements = classify_placement_outcome(
        {"instances": {}},
        MODEL_ID,
        expected_min_nodes=10,
        live_node_count=3,
    )
    assert verdict == "refused"
    assert placements == []


def test_classify_replaced_wider_for_ready_fitting_instance() -> None:
    state = _state(ready=True, node_ids=["node-a", "node-b"])
    verdict, placements = classify_placement_outcome(
        state, MODEL_ID, expected_min_nodes=10, live_node_count=3
    )
    assert verdict == "replaced_wider"
    assert len(placements) == 1
    assert placements[0].ready is True


def test_classify_partial_when_not_ready() -> None:
    state = _state(ready=False, node_ids=["node-a", "node-b"])
    verdict, _ = classify_placement_outcome(
        state, MODEL_ID, expected_min_nodes=10, live_node_count=3
    )
    assert verdict == "partial"


def test_classify_partial_when_more_nodes_than_live() -> None:
    state = _state(ready=True, node_ids=["node-a", "node-b", "node-c", "node-d"])
    verdict, _ = classify_placement_outcome(
        state, MODEL_ID, expected_min_nodes=10, live_node_count=3
    )
    assert verdict == "partial"


def test_placements_for_model_ignores_other_models() -> None:
    state = _state(ready=True, node_ids=["node-a"])
    assert _placements_for_model_from_state(state, "other/Model") == []
    assert len(_placements_for_model_from_state(state, MODEL_ID)) == 1
