"""Unit tests for provenance-first run comparison."""

from __future__ import annotations

from datetime import UTC, datetime

from skulk_test_harness.compare import compare, summarize
from skulk_test_harness.models import (
    CacheState,
    ClusterFingerprint,
    ClusterNodeFingerprint,
    GenerationMetrics,
    Issue,
    PlacementResult,
    ReportFingerprint,
    RunReport,
    RunSpec,
)
from skulk_test_harness.models import (
    TestResult as HarnessTestResult,
)


def _metrics(
    tps: float,
    tokens: int,
    *,
    ttft: float = 0.2,
    approximate_tokens: int | None = None,
) -> GenerationMetrics:
    decode_elapsed_s = tokens / tps
    return GenerationMetrics(
        elapsed_s=ttft + decode_elapsed_s,
        ttft_s=ttft,
        chunks=2,
        approx_output_tokens=(
            tokens if approximate_tokens is None else approximate_tokens
        ),
        decode_elapsed_s=decode_elapsed_s,
        observed_decode_tps=tps,
        skulk_generation_tokens=tokens,
        skulk_generation_tps=tps + 1,
        wall_tps=tps - 1,
    )


def _result(
    model: str,
    tps: float,
    tokens: int,
    *,
    test_name: str = "ordered",
    protocol_id: str | None = "a" * 64,
    kind: str = "chat",
    passed: bool = True,
    issues: list[Issue] | None = None,
    repetition: int = 1,
) -> HarnessTestResult:
    return HarnessTestResult(
        model_id=model,
        test_name=test_name,
        kind=kind,  # type: ignore[arg-type]
        protocol_id=protocol_id,
        protocol_family_id="f" * 64 if protocol_id else None,
        repetition=repetition,
        passed=passed,
        output_text="x" * tokens,
        metrics=_metrics(tps, tokens),
        issues=issues or [],
    )


def _report(
    run_id: str,
    results: list[HarnessTestResult],
    *,
    backend: str | None = "mlx-metal",
    accelerator_name: str | None = "M4 Max",
    cache: str = "warm",
    fingerprint: bool = True,
) -> RunReport:
    spec = RunSpec(model_set="m", test_set="decode-suite", mode="execute")
    report_fingerprint = None
    if fingerprint:
        report_fingerprint = ReportFingerprint(
            cluster=ClusterFingerprint(
                topology_label="one-node",
                nodes=[
                    ClusterNodeFingerprint(
                        node_id="node-a",
                        accelerator_vendor="apple",
                        accelerator_name=accelerator_name,
                        ram_total_bytes=64_000_000_000,
                        system_telemetry_present=True,
                        memory_telemetry_present=True,
                    )
                ],
            ),
            cache_state=CacheState(classification=cache),  # type: ignore[arg-type]
        )
    placements = [
        PlacementResult(
            model_id=model,
            node_ids=["node-a"],
            runner_ids=["runner-a"],
            resolved_backends=[backend] if backend else [],
            shard_types=["PipelineShardMetadata"],
            sharding="Pipeline",
            instance_meta="MlxRingInstance",
            ready=True,
        )
        for model in {result.model_id for result in results}
    ]
    return RunReport(
        run_id=run_id,
        started_at=datetime.now(tz=UTC),
        spec=spec,
        results=results,
        placements=placements,
        fingerprint=report_fingerprint,
    )


def _exact_summary(report: RunReport):
    return next(
        summary
        for summary in summarize([report]).values()
        if summary.series is not None
        and summary.series.metric_source == "client_exact"
    )


def test_summarize_reduces_repetitions_to_one_run_point() -> None:
    report = _report(
        "r1",
        [
            _result("m/A", 40.0, 100, repetition=1),
            _result("m/A", 50.0, 100, repetition=2),
            _result("m/A", 60.0, 100, repetition=3),
        ],
    )
    summary = _exact_summary(report)

    assert summary.metrics["decode_tps"].median == 50.0
    assert summary.metrics["decode_tps"].sample_count == 1
    assert summary.repetition_count == 3


def test_summarize_keeps_tests_and_metric_sources_separate() -> None:
    summaries = summarize(
        [
            _report(
                "r1",
                [
                    _result("m/A", 40.0, 100, test_name="ordered"),
                    _result("m/A", 80.0, 100, test_name="harmony"),
                ],
            )
        ]
    )

    identities = [summary.series for summary in summaries.values()]
    assert len(identities) == 6
    assert {identity.test_name for identity in identities if identity} == {
        "ordered",
        "harmony",
    }
    assert {identity.metric_source for identity in identities if identity} == {
        "client_exact",
        "engine_reported",
        "client_approx",
    }


def test_compare_computes_delta_from_matching_run_level_medians() -> None:
    baseline = [
        _report("b1", [_result("m/A", 40.0, 100)]),
        _report("b2", [_result("m/A", 42.0, 100)]),
        _report("b3", [_result("m/A", 41.0, 100)]),
    ]
    candidate = [
        _report("c1", [_result("m/A", 44.0, 100)]),
        _report("c2", [_result("m/A", 46.0, 100)]),
        _report("c3", [_result("m/A", 45.0, 100)]),
    ]
    record = compare(
        baseline, candidate, baseline_label="main", candidate_label="branch"
    )
    exact = next(
        item
        for item in record.models
        if item.series is not None and item.series.metric_source == "client_exact"
    )
    delta = next(value for value in exact.deltas if value.metric == "decode_tps")

    assert delta.baseline == 41.0
    assert delta.candidate == 45.0
    assert delta.percent_delta is not None
    assert round(delta.percent_delta, 2) == 9.76
    assert "low_sample" not in exact.guards


def test_protocol_hardware_and_backend_mismatches_are_not_comparable() -> None:
    cases = [
        (
            _report("b1", [_result("m/A", 40.0, 100, protocol_id="a" * 64)]),
            _report("c1", [_result("m/A", 44.0, 100, protocol_id="b" * 64)]),
        ),
        (
            _report("b1", [_result("m/A", 40.0, 100)]),
            _report(
                "c1", [_result("m/A", 44.0, 100)], accelerator_name="M3 Max"
            ),
        ),
        (
            _report("b1", [_result("m/A", 40.0, 100)]),
            _report("c1", [_result("m/A", 44.0, 100)], backend="mlx"),
        ),
    ]

    for baseline, candidate in cases:
        record = compare(
            [baseline], [candidate], baseline_label="main", candidate_label="branch"
        )
        assert all(not item.deltas for item in record.models)
        assert "series_only_one_side" in record.guards


def test_missing_protocol_or_backend_is_incomplete_and_never_compared() -> None:
    for report in (
        _report("r1", [_result("m/A", 40.0, 100, protocol_id=None)]),
        _report("r1", [_result("m/A", 40.0, 100)], backend=None),
        _report(
            "r1", [_result("m/A", 40.0, 100)], accelerator_name=None
        ),
    ):
        record = compare(
            [report], [report], baseline_label="main", candidate_label="branch"
        )
        assert all(not item.deltas for item in record.models)
        assert "series_identity_incomplete" in record.guards


def test_failed_specialized_and_short_results_do_not_feed_decode_points() -> None:
    report = _report(
        "r1",
        [
            _result("m/A", 40.0, 100, passed=False),
            _result("m/A", 200.0, 100, kind="tool", test_name="tool-call"),
            _result("m/A", 999.0, 3, test_name="short"),
        ],
    )
    summaries = summarize([report])

    assert all(
        summary.series is not None and summary.series.test_name != "tool-call"
        for summary in summaries.values()
    )
    short_exact = next(
        summary
        for summary in summaries.values()
        if summary.series is not None
        and summary.series.test_name == "short"
        and summary.series.metric_source == "client_exact"
    )
    assert short_exact.metrics["decode_tps"].sample_count == 0
    assert short_exact.metrics["decode_tps"].short_sample_count == 1


def test_approximate_source_uses_its_own_token_basis() -> None:
    result = _result("m/A", 50.0, 100)
    result = result.model_copy(
        update={"metrics": _metrics(50.0, 100, approximate_tokens=3)}
    )
    summaries = summarize([_report("r1", [result])])
    exact = next(
        summary
        for summary in summaries.values()
        if summary.series and summary.series.metric_source == "client_exact"
    )
    approximate = next(
        summary
        for summary in summaries.values()
        if summary.series and summary.series.metric_source == "client_approx"
    )

    assert exact.metrics["decode_tps"].sample_count == 1
    assert approximate.metrics["decode_tps"].sample_count == 0
    assert approximate.metrics["decode_tps"].short_sample_count == 1


def test_cache_issue_and_fingerprint_guards_remain_visible() -> None:
    warning = Issue(severity="warning", message="noisy")
    baseline = _report("b1", [_result("m/A", 40.0, 100)], cache="warm")
    candidate = _report(
        "c1",
        [_result("m/A", 44.0, 100, issues=[warning])],
        cache="cold",
    )
    record = compare(
        [baseline], [candidate], baseline_label="main", candidate_label="branch"
    )
    assert "cache_mismatch" in record.guards
    assert "issue_marked" in record.guards

    old = _report("old", [_result("m/A", 40.0, 100)], fingerprint=False)
    missing = compare([old], [old], baseline_label="old", candidate_label="old")
    assert "missing_fingerprint" in missing.guards
