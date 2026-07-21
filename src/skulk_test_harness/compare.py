"""Provenance-first, like-for-like comparison of harness run sets.

Each comparison row is one exact execution series: model, suite, test,
protocol, metric source, hardware, backend, instance type, and sharding. Test
cases and execution profiles are never pooled. Repetitions are first reduced to
one median per run; run-level medians are then summarized across each side.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from skulk_test_harness.models import (
    ComparisonGuardKind,
    ComparisonHardwareNode,
    ComparisonRecord,
    ComparisonSeriesIdentity,
    MetricAggregate,
    MetricDelta,
    MetricSource,
    ModelComparison,
    ModelMetricSummary,
    PlacementResult,
    RunReport,
    TestResult,
)

_TEXT_DECODE_KINDS = frozenset({"chat", "code", "artifact"})
_SHORT_OUTPUT_TOKENS = 20
_LOW_SAMPLE_THRESHOLD = 3
_SHORT_DOMINANT_FRACTION = 0.5


@dataclass
class _RunSeriesBucket:
    """Repetitions for one exact series inside one run."""

    identity: ComparisonSeriesIdentity
    run_id: str
    tps_values: list[float] = field(default_factory=list)
    ttft_values: list[float] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    issue_count: int = 0
    repetition_count: int = 0
    short_sample_count: int = 0


@dataclass
class _SeriesAccumulator:
    """Run-level points and audit counts for one exact series."""

    identity: ComparisonSeriesIdentity
    run_ids: list[str] = field(default_factory=list)
    tps_points: list[float] = field(default_factory=list)
    ttft_points: list[float] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    issue_count: int = 0
    repetition_count: int = 0
    short_sample_count: int = 0


def load_reports(run_dirs: list[Path]) -> list[RunReport]:
    """Load valid model-scoring ``report.json`` files from run directories."""

    reports: list[RunReport] = []
    for run_dir in run_dirs:
        report_path = run_dir / "report.json"
        if not report_path.is_file():
            continue
        try:
            raw = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or "results" not in raw or "suite" in raw:
            continue
        try:
            reports.append(RunReport.model_validate(raw))
        except ValueError:
            continue
    return reports


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _source_value(result: TestResult, source: MetricSource) -> float | None:
    metrics = result.metrics
    if source == "client_exact":
        return metrics.observed_decode_tps
    if source == "engine_reported":
        return metrics.skulk_generation_tps
    return metrics.wall_tps


def _source_tokens(result: TestResult, source: MetricSource) -> int | None:
    if source == "client_approx":
        return result.metrics.approx_output_tokens
    return result.metrics.skulk_generation_tokens


def _source_valid(result: TestResult, source: MetricSource) -> bool:
    metrics = result.metrics
    value = _source_value(result, source)
    tokens = _source_tokens(result, source)
    if not result.passed or not _finite_positive(value):
        return False
    if tokens is None or tokens < _SHORT_OUTPUT_TOKENS:
        return False
    if source == "client_exact":
        return (
            metrics.chunks >= 2
            and _finite_positive(metrics.decode_elapsed_s)
            and metrics.observed_decode_tps is not None
        )
    return True


def _placement_for(report: RunReport, model_id: str) -> PlacementResult | None:
    candidates = [p for p in report.placements if p.model_id == model_id]
    if not candidates:
        return None
    ready = [placement for placement in candidates if placement.ready]
    return (ready or candidates)[-1]


def _hardware_for(
    report: RunReport, placement: PlacementResult | None
) -> tuple[list[ComparisonHardwareNode], bool]:
    if report.fingerprint is None or placement is None or not placement.node_ids:
        return [], False
    by_id = {node.node_id: node for node in report.fingerprint.cluster.nodes}
    hardware: list[ComparisonHardwareNode] = []
    complete = True
    for node_id in placement.node_ids:
        node = by_id.get(node_id)
        if node is None:
            hardware.append(ComparisonHardwareNode())
            complete = False
            continue
        item = ComparisonHardwareNode(
            accelerator_vendor=node.accelerator_vendor,
            accelerator_name=node.accelerator_name,
            ram_total_bytes=node.ram_total_bytes,
            vram_total_bytes=node.vram_total_bytes,
            gtt_total_bytes=node.gtt_total_bytes,
        )
        if (
            item.accelerator_vendor is None
            or item.accelerator_name is None
            or item.ram_total_bytes is None
            or (
                item.accelerator_vendor != "apple"
                and item.vram_total_bytes is None
            )
        ):
            complete = False
        hardware.append(item)
    hardware.sort(
        key=lambda item: (
            item.accelerator_vendor or "",
            item.accelerator_name or "",
            item.ram_total_bytes or -1,
            item.vram_total_bytes or -1,
            item.gtt_total_bytes or -1,
        )
    )
    return hardware, complete


def _series_identity(
    report: RunReport, result: TestResult, source: MetricSource
) -> ComparisonSeriesIdentity:
    placement = _placement_for(report, result.model_id)
    hardware, hardware_complete = _hardware_for(report, placement)
    resolved_backends = placement.resolved_backends if placement is not None else []
    shard_types = placement.shard_types if placement is not None else []
    instance_meta = placement.instance_meta if placement is not None else None
    sharding = placement.sharding if placement is not None else None
    raw_identity: dict[str, object] = {
        "model_id": result.model_id,
        "test_set": report.spec.test_set,
        "test_name": result.test_name,
        "protocol_id": result.protocol_id,
        "metric_source": source,
        "hardware": [item.model_dump(mode="json") for item in hardware],
        "resolved_backends": sorted(resolved_backends),
        "instance_meta": instance_meta,
        "sharding": sharding,
        "shard_types": sorted(shard_types),
    }
    canonical = json.dumps(
        raw_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    complete = bool(
        result.protocol_id
        and hardware_complete
        and resolved_backends
        and instance_meta
        and sharding
        and shard_types
    )
    return ComparisonSeriesIdentity(
        series_id=hashlib.sha256(canonical).hexdigest(),
        model_id=result.model_id,
        test_set=report.spec.test_set,
        test_name=result.test_name,
        protocol_id=result.protocol_id,
        metric_source=source,
        hardware=hardware,
        resolved_backends=sorted(resolved_backends),
        instance_meta=instance_meta,
        sharding=sharding,
        shard_types=sorted(shard_types),
        complete=complete,
    )


def _available_sources(result: TestResult) -> tuple[MetricSource, ...]:
    sources: list[MetricSource] = []
    for source in ("client_exact", "engine_reported", "client_approx"):
        if _source_value(result, source) is not None:
            sources.append(source)
    return tuple(sources)


def _metric_aggregate(
    metric: str,
    unit: str,
    points: list[float],
    *,
    short_sample_count: int = 0,
) -> MetricAggregate:
    if not points:
        return MetricAggregate(
            metric=metric,
            unit=unit,
            sample_count=0,
            short_sample_count=short_sample_count,
        )
    return MetricAggregate(
        metric=metric,
        unit=unit,
        median=float(median(points)),
        minimum=min(points),
        maximum=max(points),
        sample_count=len(points),
        short_sample_count=short_sample_count,
    )


def summarize(reports: list[RunReport]) -> dict[str, ModelMetricSummary]:
    """Summarize matching repetitions into distinct run-level series points."""

    accumulators: dict[str, _SeriesAccumulator] = {}
    for report in reports:
        run_buckets: dict[str, _RunSeriesBucket] = {}
        for result in report.results:
            if result.kind not in _TEXT_DECODE_KINDS:
                continue
            for source in _available_sources(result):
                identity = _series_identity(report, result, source)
                bucket = run_buckets.setdefault(
                    identity.series_id,
                    _RunSeriesBucket(identity=identity, run_id=report.run_id),
                )
                bucket.repetition_count += 1
                bucket.pass_count += int(result.passed)
                bucket.fail_count += int(not result.passed)
                bucket.issue_count += len(result.issues)
                tokens = _source_tokens(result, source)
                if (
                    result.passed
                    and tokens is not None
                    and tokens < _SHORT_OUTPUT_TOKENS
                ):
                    bucket.short_sample_count += 1
                if _source_valid(result, source):
                    value = _source_value(result, source)
                    if value is not None:
                        bucket.tps_values.append(float(value))
                if result.passed and _finite_positive(result.metrics.ttft_s):
                    assert result.metrics.ttft_s is not None
                    bucket.ttft_values.append(result.metrics.ttft_s)

        for series_id, bucket in run_buckets.items():
            accumulator = accumulators.setdefault(
                series_id, _SeriesAccumulator(identity=bucket.identity)
            )
            accumulator.run_ids.append(bucket.run_id)
            accumulator.pass_count += bucket.pass_count
            accumulator.fail_count += bucket.fail_count
            accumulator.issue_count += bucket.issue_count
            accumulator.repetition_count += bucket.repetition_count
            accumulator.short_sample_count += bucket.short_sample_count
            if bucket.tps_values:
                accumulator.tps_points.append(float(median(bucket.tps_values)))
            if bucket.ttft_values:
                accumulator.ttft_points.append(float(median(bucket.ttft_values)))

    return {
        series_id: ModelMetricSummary(
            model_id=accumulator.identity.model_id,
            series=accumulator.identity,
            run_ids=sorted(set(accumulator.run_ids)),
            repetition_count=accumulator.repetition_count,
            pass_count=accumulator.pass_count,
            fail_count=accumulator.fail_count,
            issue_count=accumulator.issue_count,
            node_count_observed=sorted(
                {len(accumulator.identity.hardware)}
                if accumulator.identity.hardware
                else set()
            ),
            metrics={
                "decode_tps": _metric_aggregate(
                    "decode_tps",
                    "tok/s",
                    accumulator.tps_points,
                    short_sample_count=accumulator.short_sample_count,
                ),
                "ttft_s": _metric_aggregate(
                    "ttft_s", "s", accumulator.ttft_points
                ),
            },
        )
        for series_id, accumulator in accumulators.items()
    }


def _percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def _series_guards(
    baseline: ModelMetricSummary, candidate: ModelMetricSummary
) -> list[ComparisonGuardKind]:
    guards: list[ComparisonGuardKind] = []
    if (
        baseline.series is None
        or candidate.series is None
        or not baseline.series.complete
        or not candidate.series.complete
    ):
        guards.append("series_identity_incomplete")
    baseline_decode = baseline.metrics.get("decode_tps")
    candidate_decode = candidate.metrics.get("decode_tps")
    baseline_count = baseline_decode.sample_count if baseline_decode else 0
    candidate_count = candidate_decode.sample_count if candidate_decode else 0
    if min(baseline_count, candidate_count) < _LOW_SAMPLE_THRESHOLD:
        guards.append("low_sample")
    for aggregate in (baseline_decode, candidate_decode):
        if aggregate is None:
            continue
        total = aggregate.sample_count + aggregate.short_sample_count
        if total and aggregate.short_sample_count / total > _SHORT_DOMINANT_FRACTION:
            guards.append("short_output_dominant")
            break
    if baseline.issue_count or candidate.issue_count:
        guards.append("issue_marked")
    return guards


def _compare_series(
    baseline: ModelMetricSummary, candidate: ModelMetricSummary
) -> ModelComparison:
    guards = _series_guards(baseline, candidate)
    identity = baseline.series
    if "series_identity_incomplete" in guards:
        return ModelComparison(
            model_id=baseline.model_id,
            series=identity,
            guards=guards,
            baseline_summary=baseline,
            candidate_summary=candidate,
        )

    deltas: list[MetricDelta] = []
    for metric, unit, higher_is_better in (
        ("decode_tps", "tok/s", True),
        ("ttft_s", "s", False),
    ):
        baseline_aggregate = baseline.metrics.get(metric)
        candidate_aggregate = candidate.metrics.get(metric)
        baseline_value = baseline_aggregate.median if baseline_aggregate else None
        candidate_value = candidate_aggregate.median if candidate_aggregate else None
        if metric == "decode_tps" and (
            baseline_value is None or candidate_value is None
        ):
            guards.append("decode_tps_unavailable")
        absolute_delta = (
            candidate_value - baseline_value
            if baseline_value is not None and candidate_value is not None
            else None
        )
        deltas.append(
            MetricDelta(
                metric=metric,
                unit=unit,
                baseline=baseline_value,
                candidate=candidate_value,
                absolute_delta=absolute_delta,
                percent_delta=(
                    _percent(baseline_value, candidate_value)
                    if baseline_value is not None and candidate_value is not None
                    else None
                ),
                higher_is_better=higher_is_better,
            )
        )
    return ModelComparison(
        model_id=baseline.model_id,
        series=identity,
        deltas=deltas,
        guards=guards,
        baseline_summary=baseline,
        candidate_summary=candidate,
    )


def compare(
    baseline_reports: list[RunReport],
    candidate_reports: list[RunReport],
    *,
    baseline_label: str,
    candidate_label: str,
) -> ComparisonRecord:
    """Compare only full, matching series identities across two run sets."""

    baseline = summarize(baseline_reports)
    candidate = summarize(candidate_reports)
    comparisons: list[ModelComparison] = []
    record_guards: set[ComparisonGuardKind] = set()

    for series_id in sorted(set(baseline) | set(candidate)):
        baseline_summary = baseline.get(series_id)
        candidate_summary = candidate.get(series_id)
        if baseline_summary is None or candidate_summary is None:
            present = candidate_summary or baseline_summary
            assert present is not None
            guards: list[ComparisonGuardKind] = ["series_only_one_side"]
            if present.series is None or not present.series.complete:
                guards.append("series_identity_incomplete")
            comparisons.append(
                ModelComparison(
                    model_id=present.model_id,
                    series=present.series,
                    guards=guards,
                    baseline_summary=baseline_summary,
                    candidate_summary=candidate_summary,
                )
            )
            record_guards.update(guards)
            continue
        comparison = _compare_series(baseline_summary, candidate_summary)
        comparisons.append(comparison)
        record_guards.update(comparison.guards)

    if not _has_fingerprint(baseline_reports) or not _has_fingerprint(
        candidate_reports
    ):
        record_guards.add("missing_fingerprint")
    if _cache_classes(baseline_reports) != _cache_classes(candidate_reports):
        record_guards.add("cache_mismatch")

    comparisons.sort(
        key=lambda item: (
            item.model_id,
            item.series.test_set if item.series else "",
            item.series.test_name if item.series else "",
            item.series.metric_source if item.series else "",
            item.series.series_id if item.series else "",
        )
    )
    return ComparisonRecord(
        schema_version="2.0",
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_run_ids=sorted(report.run_id for report in baseline_reports),
        candidate_run_ids=sorted(report.run_id for report in candidate_reports),
        models=comparisons,
        guards=sorted(record_guards),
    )


def _has_fingerprint(reports: list[RunReport]) -> bool:
    return bool(reports) and all(report.fingerprint is not None for report in reports)


def _cache_classes(reports: list[RunReport]) -> set[str]:
    return {
        report.fingerprint.cache_state.classification
        for report in reports
        if report.fingerprint is not None
    }


def select_run_dirs(runs_root: Path, selector: str) -> list[Path]:
    """Resolve an explicit run path or substring selector to run directories."""

    explicit = Path(selector)
    if explicit.is_dir() and (explicit / "report.json").is_file():
        return [explicit]
    if not runs_root.is_dir():
        return []
    matches = [
        directory
        for directory in runs_root.iterdir()
        if directory.is_dir()
        and selector in directory.name
        and (directory / "report.json").is_file()
    ]
    return sorted(matches, key=lambda directory: directory.name)
