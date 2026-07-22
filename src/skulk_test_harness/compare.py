"""Like-for-like comparison of harness run sets (results-ledger Phase 2).

Loads two sets of ``report.json`` artifacts from ``runs/`` and produces a
:class:`ComparisonRecord`: per-model deltas for the headline metrics plus the
guards that make a comparison NOT trustworthy (different node set, cache
warmth, low sample count, short-output noise, issue-marked runs). This is the
programmatic form of the MLX-VLM 0.6.4 "is the new branch actually faster?"
analysis, so that judgement never again depends on chat memory.

The aggregation core is pure and unit-tested; the CLI (`skulk-harness compare`)
is a thin shell that resolves run selectors to directories, loads reports, and
renders the record.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from skulk_test_harness.models import (
    ComparisonGuardKind,
    ComparisonRecord,
    MetricAggregate,
    MetricDelta,
    ModelComparison,
    ModelMetricSummary,
    RunReport,
    TestResult,
)

# Outputs shorter than this many generated tokens are excluded from throughput
# medians: a 5-character answer produces a meaningless "415 tok/s" wall figure
# (see report-schema.md short-output caveat). They are still counted, as
# ``short_sample_count``, so the exclusion is visible, never silent.
_SHORT_OUTPUT_TOKENS = 20

# Below this many substantive samples per side, a delta is not a measurement.
_LOW_SAMPLE_THRESHOLD = 3

# Fraction of short samples above which the model's throughput is dominated by
# noise and the delta should be read with heavy suspicion.
_SHORT_DOMINANT_FRACTION = 0.5


class _MetricSpec:
    """One comparable metric: how to pull it from a result and read a delta."""

    __slots__ = ("key", "unit", "higher_is_better", "extract")

    def __init__(
        self,
        key: str,
        unit: str,
        higher_is_better: bool,
        extract: object,
    ) -> None:
        self.key = key
        self.unit = unit
        self.higher_is_better = higher_is_better
        self.extract = extract  # Callable[[TestResult], float | None]


def _decode_tps(r: TestResult) -> float | None:
    """Steady-state decode throughput: prefer Skulk's own number over wall.

    ``skulk_generation_tps`` excludes prompt/TTFT time and is the honest decode
    rate. When Skulk did not report it (older runs, some engines), fall back to
    ``wall_tps`` so a comparison is still possible, and let the
    ``decode_tps_unavailable`` guard flag that the fallback was used.
    """
    m = r.metrics
    if m.skulk_generation_tps is not None:
        return m.skulk_generation_tps
    return m.wall_tps


_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("decode_tps", "tok/s", True, _decode_tps),
    _MetricSpec("ttft_s", "s", False, lambda r: r.metrics.ttft_s),
    _MetricSpec("wall_tps", "tok/s", True, lambda r: r.metrics.wall_tps),
)


def _is_short(result: TestResult) -> bool:
    """True when a result's output is too short to time throughput honestly."""
    tokens = result.metrics.skulk_generation_tokens
    if tokens is None:
        tokens = result.metrics.approx_output_tokens
    if tokens is None:
        # No token count at all: treat as short so it cannot inflate a median.
        return True
    return tokens < _SHORT_OUTPUT_TOKENS


def load_reports(run_dirs: list[Path]) -> list[RunReport]:
    """Load and validate ``report.json`` from each run directory.

    Stability-suite reports (which carry ``suite`` and no scored ``results``)
    and any file that fails validation are skipped: a comparison operates only
    over model-scoring runs. Unreadable or non-conforming reports are ignored
    rather than aborting the whole comparison.
    """
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


def _aggregate_metric(spec: _MetricSpec, results: list[TestResult]) -> MetricAggregate:
    """Aggregate one metric over a model's results, splitting out short outputs."""
    values: list[float] = []
    short = 0
    extract = spec.extract
    for result in results:
        raw = extract(result)  # type: ignore[operator]
        if raw is None:
            continue
        if spec.key in ("decode_tps", "wall_tps") and _is_short(result):
            short += 1
            continue
        values.append(float(raw))
    if not values:
        return MetricAggregate(
            metric=spec.key, unit=spec.unit, sample_count=0, short_sample_count=short
        )
    return MetricAggregate(
        metric=spec.key,
        unit=spec.unit,
        median=float(median(values)),
        minimum=min(values),
        maximum=max(values),
        sample_count=len(values),
        short_sample_count=short,
    )


def summarize(reports: list[RunReport]) -> dict[str, ModelMetricSummary]:
    """Aggregate a set of reports into one summary per model.

    Groups every :class:`TestResult` across the reports by ``model_id`` and
    computes the headline metric aggregates plus pass/fail/issue counts and the
    node-set / topology observed for that model.
    """
    by_model: dict[str, list[TestResult]] = {}
    run_ids: dict[str, set[str]] = {}
    node_counts: dict[str, set[int]] = {}
    topologies: dict[str, set[str]] = {}

    for report in reports:
        topo = None
        if report.fingerprint is not None:
            topo = report.fingerprint.cluster.topology_label
        placement_nodes = {
            p.model_id: len(p.node_ids) for p in report.placements if p.node_ids
        }
        for result in report.results:
            by_model.setdefault(result.model_id, []).append(result)
            run_ids.setdefault(result.model_id, set()).add(report.run_id)
            if result.model_id in placement_nodes:
                node_counts.setdefault(result.model_id, set()).add(
                    placement_nodes[result.model_id]
                )
            if topo:
                topologies.setdefault(result.model_id, set()).add(topo)

    summaries: dict[str, ModelMetricSummary] = {}
    for model_id, results in by_model.items():
        summaries[model_id] = ModelMetricSummary(
            model_id=model_id,
            run_ids=sorted(run_ids.get(model_id, set())),
            pass_count=sum(1 for r in results if r.passed),
            fail_count=sum(1 for r in results if not r.passed),
            issue_count=sum(len(r.issues) for r in results),
            node_count_observed=sorted(node_counts.get(model_id, set())),
            topology_labels=sorted(topologies.get(model_id, set())),
            metrics={
                spec.key: _aggregate_metric(spec, results) for spec in _METRICS
            },
        )
    return summaries


def _percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def _model_guards(
    baseline: ModelMetricSummary, candidate: ModelMetricSummary
) -> list[ComparisonGuardKind]:
    guards: list[ComparisonGuardKind] = []
    b_decode = baseline.metrics.get("decode_tps")
    c_decode = candidate.metrics.get("decode_tps")
    b_n = b_decode.sample_count if b_decode else 0
    c_n = c_decode.sample_count if c_decode else 0
    if min(b_n, c_n) < _LOW_SAMPLE_THRESHOLD:
        guards.append("low_sample")
    for agg in (b_decode, c_decode):
        if agg is None:
            continue
        total = agg.sample_count + agg.short_sample_count
        if total and agg.short_sample_count / total > _SHORT_DOMINANT_FRACTION:
            guards.append("short_output_dominant")
            break
    if baseline.issue_count or candidate.issue_count:
        guards.append("issue_marked")
    if (
        baseline.node_count_observed
        and candidate.node_count_observed
        and set(baseline.node_count_observed) != set(candidate.node_count_observed)
    ):
        guards.append("node_set_mismatch")
    return guards


def _compare_model(
    baseline: ModelMetricSummary, candidate: ModelMetricSummary
) -> ModelComparison:
    deltas: list[MetricDelta] = []
    decode_missing = False
    for spec in _METRICS:
        b_agg = baseline.metrics.get(spec.key)
        c_agg = candidate.metrics.get(spec.key)
        b_val = b_agg.median if b_agg else None
        c_val = c_agg.median if c_agg else None
        if spec.key == "decode_tps" and (b_val is None or c_val is None):
            decode_missing = True
        abs_delta = (
            c_val - b_val if b_val is not None and c_val is not None else None
        )
        pct = (
            _percent(b_val, c_val)
            if b_val is not None and c_val is not None
            else None
        )
        deltas.append(
            MetricDelta(
                metric=spec.key,
                unit=spec.unit,
                baseline=b_val,
                candidate=c_val,
                absolute_delta=abs_delta,
                percent_delta=pct,
                higher_is_better=spec.higher_is_better,
            )
        )
    guards = _model_guards(baseline, candidate)
    if decode_missing:
        guards.append("decode_tps_unavailable")
    return ModelComparison(
        model_id=baseline.model_id,
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
    """Build a like-for-like comparison record from two loaded run sets."""
    baseline = summarize(baseline_reports)
    candidate = summarize(candidate_reports)

    models: list[ModelComparison] = []
    record_guards: set[ComparisonGuardKind] = set()

    for model_id in sorted(set(baseline) | set(candidate)):
        b = baseline.get(model_id)
        c = candidate.get(model_id)
        if b is None or c is None:
            present = c if b is None else b
            models.append(
                ModelComparison(
                    model_id=model_id,
                    guards=["model_only_one_side"],
                    baseline_summary=b,
                    candidate_summary=c,
                )
            )
            record_guards.add("model_only_one_side")
            _ = present
            continue
        comparison = _compare_model(b, c)
        models.append(comparison)
        record_guards.update(comparison.guards)

    if not _has_fingerprint(baseline_reports) or not _has_fingerprint(
        candidate_reports
    ):
        record_guards.add("missing_fingerprint")
    if _cache_classes(baseline_reports) != _cache_classes(candidate_reports):
        record_guards.add("cache_mismatch")

    return ComparisonRecord(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_run_ids=sorted(r.run_id for r in baseline_reports),
        candidate_run_ids=sorted(r.run_id for r in candidate_reports),
        models=models,
        guards=sorted(record_guards),
    )


def _has_fingerprint(reports: list[RunReport]) -> bool:
    return any(r.fingerprint is not None for r in reports)


def _cache_classes(reports: list[RunReport]) -> set[str]:
    classes: set[str] = set()
    for r in reports:
        if r.fingerprint is not None:
            classes.add(r.fingerprint.cache_state.classification)
    return classes


def select_run_dirs(runs_root: Path, selector: str) -> list[Path]:
    """Resolve a selector to a sorted list of run directories.

    A selector is either an explicit path to a run directory, or a substring
    matched against run-directory names (e.g. ``dense-singles`` or a run-id
    prefix like ``20260707``). Matching directories are returned in
    chronological order (run-id names sort chronologically).
    """
    explicit = Path(selector)
    if explicit.is_dir() and (explicit / "report.json").is_file():
        return [explicit]
    if not runs_root.is_dir():
        return []
    matches = [
        d
        for d in runs_root.iterdir()
        if d.is_dir() and selector in d.name and (d / "report.json").is_file()
    ]
    return sorted(matches, key=lambda d: d.name)
