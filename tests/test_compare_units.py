"""Unit tests for the like-for-like run comparison core (Phase 2)."""

from __future__ import annotations

from datetime import UTC, datetime

from skulk_test_harness.compare import compare, summarize
from skulk_test_harness.models import (
    CacheState,
    ClusterFingerprint,
    GenerationMetrics,
    Issue,
    PlacementResult,
    ReportFingerprint,
    RunReport,
    RunSpec,
    TestResult,
)


def _metrics(tps: float, tokens: int, ttft: float = 0.2) -> GenerationMetrics:
    return GenerationMetrics(
        elapsed_s=1.0,
        ttft_s=ttft,
        approx_output_tokens=tokens,
        skulk_generation_tokens=tokens,
        skulk_generation_tps=tps,
        wall_tps=tps,
    )


def _result(model: str, tps: float, tokens: int, *, passed: bool = True,
            issues: list[Issue] | None = None) -> TestResult:
    return TestResult(
        model_id=model,
        test_name="t",
        repetition=1,
        passed=passed,
        output_text="x" * tokens,
        metrics=_metrics(tps, tokens),
        issues=issues or [],
    )


def _report(run_id: str, results: list[TestResult], *, topology: str = "kite1-kite2",
            cache: str = "warm", node_count: int = 1,
            fingerprint: bool = True) -> RunReport:
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    fp = None
    if fingerprint:
        fp = ReportFingerprint(
            cluster=ClusterFingerprint(topology_label=topology),
            cache_state=CacheState(classification=cache),  # type: ignore[arg-type]
        )
    placements = [
        PlacementResult(model_id=m, node_ids=[f"n{i}" for i in range(node_count)])
        for m in {r.model_id for r in results}
    ]
    return RunReport(
        run_id=run_id,
        started_at=datetime.now(tz=UTC),
        spec=spec,
        results=results,
        placements=placements,
        fingerprint=fp,
    )


def test_summarize_excludes_short_outputs_from_median() -> None:
    # Three substantive samples at ~50 tok/s and one 3-token junk sample that,
    # if counted, would blow the median up. It must be excluded but counted.
    results = [
        _result("m/A", 50.0, 100),
        _result("m/A", 52.0, 100),
        _result("m/A", 48.0, 100),
        _result("m/A", 999.0, 3),
    ]
    summary = summarize([_report("r1", results)])
    decode = summary["m/A"].metrics["decode_tps"]
    assert decode.median == 50.0
    assert decode.sample_count == 3
    assert decode.short_sample_count == 1


def test_compare_computes_percent_delta() -> None:
    base = _report("b1", [_result("m/A", 40.0, 100) for _ in range(3)])
    cand = _report("c1", [_result("m/A", 44.0, 100) for _ in range(3)])
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    model = record.models[0]
    decode = next(d for d in model.deltas if d.metric == "decode_tps")
    assert decode.baseline == 40.0
    assert decode.candidate == 44.0
    assert decode.percent_delta is not None
    assert round(decode.percent_delta, 1) == 10.0
    assert decode.higher_is_better is True
    assert "low_sample" not in model.guards


def test_low_sample_guard_fires_under_threshold() -> None:
    base = _report("b1", [_result("m/A", 40.0, 100)])
    cand = _report("c1", [_result("m/A", 44.0, 100)])
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    assert "low_sample" in record.models[0].guards
    assert "low_sample" in record.guards


def test_node_set_mismatch_guard() -> None:
    base = _report("b1", [_result("m/A", 40.0, 100) for _ in range(3)], node_count=1)
    cand = _report("c1", [_result("m/A", 44.0, 100) for _ in range(3)], node_count=2)
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    assert "node_set_mismatch" in record.models[0].guards


def test_cache_mismatch_and_issue_guards() -> None:
    warn = Issue(severity="warning", message="noisy")
    base = _report(
        "b1", [_result("m/A", 40.0, 100) for _ in range(3)], cache="warm"
    )
    cand = _report(
        "c1",
        [_result("m/A", 44.0, 100, issues=[warn]) for _ in range(3)],
        cache="cold",
    )
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    assert "cache_mismatch" in record.guards
    assert "issue_marked" in record.models[0].guards


def test_missing_fingerprint_guard() -> None:
    base = _report(
        "b1", [_result("m/A", 40.0, 100) for _ in range(3)], fingerprint=False
    )
    cand = _report("c1", [_result("m/A", 44.0, 100) for _ in range(3)])
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    assert "missing_fingerprint" in record.guards


def test_model_only_one_side() -> None:
    base = _report("b1", [_result("m/A", 40.0, 100) for _ in range(3)])
    cand = _report("c1", [_result("m/B", 44.0, 100) for _ in range(3)])
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    by_model = {m.model_id: m for m in record.models}
    assert "model_only_one_side" in by_model["m/A"].guards
    assert "model_only_one_side" in by_model["m/B"].guards
    assert "model_only_one_side" in record.guards


def test_decode_tps_unavailable_falls_back_to_wall() -> None:
    # A result with no skulk_generation_tps but a wall_tps: decode falls back to
    # wall and the guard flags the substitution.
    def wall_only(model: str, wall: float, tokens: int) -> TestResult:
        m = GenerationMetrics(
            elapsed_s=1.0, approx_output_tokens=tokens,
            skulk_generation_tokens=tokens, wall_tps=wall,
        )
        return TestResult(
            model_id=model, test_name="t", repetition=1, passed=True,
            output_text="x" * tokens, metrics=m,
        )

    base = _report("b1", [wall_only("m/A", 40.0, 100) for _ in range(3)])
    cand = _report("c1", [wall_only("m/A", 44.0, 100) for _ in range(3)])
    record = compare([base], [cand], baseline_label="main", candidate_label="branch")
    model = record.models[0]
    decode = next(d for d in model.deltas if d.metric == "decode_tps")
    # Fallback still produces a number (from wall_tps)...
    assert decode.baseline == 40.0
    # ...but decode_tps is only unavailable when the median itself is None; with
    # wall fallback present, the guard should NOT fire here.
    assert "decode_tps_unavailable" not in model.guards
