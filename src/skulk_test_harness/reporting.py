"""Report writers for harness runs."""

from __future__ import annotations

import json
from pathlib import Path

from skulk_test_harness.models import Issue, RunReport, StabilityReport
from skulk_test_harness.utils import slugify


class ReportWriter:
    """Writes machine-readable and human-readable run artifacts."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run_dir(self, run_id: str) -> Path:
        """Return the output directory for ``run_id``."""

        return self.output_root / slugify(run_id)

    def write(self, report: RunReport) -> Path:
        """Write report JSON, JSONL event log, and Markdown summary."""

        run_dir = self.run_dir(report.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            report.model_dump_json(indent=2, serialize_as_any=True)
        )
        (run_dir / "events.jsonl").write_text(_jsonl(report))
        (run_dir / "summary.md").write_text(_markdown(report))
        return run_dir

    def write_stability(self, report: StabilityReport) -> Path:
        """Write a stability suite report's JSON and Markdown summary."""

        run_dir = self.run_dir(report.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            report.model_dump_json(indent=2, serialize_as_any=True)
        )
        (run_dir / "summary.md").write_text(_stability_markdown(report))
        return run_dir


def _jsonl(report: RunReport) -> str:
    rows: list[dict[str, object]] = []
    rows.append({"type": "run_started", "run_id": report.run_id, "at": report.started_at.isoformat()})
    for issue in report.issues:
        rows.append({"type": "issue", **issue.model_dump(mode="json")})
    for placement in report.placements:
        rows.append({"type": "placement", **placement.model_dump(mode="json")})
    for result in report.results:
        rows.append({"type": "test_result", **result.model_dump(mode="json")})
    if report.finished_at is not None:
        rows.append(
            {
                "type": "run_finished",
                "run_id": report.run_id,
                "at": report.finished_at.isoformat(),
            }
        )
    return "\n".join(json.dumps(row, default=str) for row in rows) + "\n"


def _markdown(report: RunReport) -> str:
    lines: list[str] = [
        f"# Skulk Harness Run: {report.run_id}",
        "",
        f"- Started: `{report.started_at.isoformat()}`",
        f"- Finished: `{report.finished_at.isoformat() if report.finished_at else 'running'}`",
        f"- Model set: `{report.spec.model_set}`",
        f"- Test set: `{report.spec.test_set}`",
        f"- Mode: `{report.spec.mode}`",
        "",
        "## Models",
        "",
    ]
    if report.models:
        for model in report.models:
            lines.append(f"- `{model.model_id}` ({model.source}) {model.detail}".rstrip())
    else:
        lines.append("- None")

    lines.extend(["", "## Placements", ""])
    if report.placements:
        lines.append("| Model | Instance | Nodes | Ready | Reused |")
        lines.append("|---|---:|---:|---:|---:|")
        for placement in report.placements:
            lines.append(
                "| "
                f"`{placement.model_id}` | "
                f"`{placement.instance_id or ''}` | "
                f"{len(placement.node_ids)} | "
                f"{'yes' if placement.ready else 'no'} | "
                f"{'yes' if placement.reused_existing else 'no'} |"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Results", ""])
    if report.results:
        lines.append(
            "| Model | Test | Rep | Pass | TTFT s | Wall TPS | Content Chars | Generated Chars | Artifact |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for result in report.results:
            metrics = result.metrics
            artifact = f"`{result.artifact_path}`" if result.artifact_path else ""
            lines.append(
                "| "
                f"`{result.model_id}` | "
                f"`{result.test_name}` | "
                f"{result.repetition} | "
                f"{'yes' if result.passed else 'no'} | "
                f"{_fmt(metrics.ttft_s)} | "
                f"{_fmt(metrics.wall_tps)} | "
                f"{metrics.output_chars} | "
                f"{metrics.generated_chars} | "
                f"{artifact} |"
            )
    else:
        lines.append("- None")

    all_issues = [*report.issues, *(issue for r in report.results for issue in r.issues)]
    lines.extend(["", "## Issues", ""])
    if all_issues:
        for issue in all_issues:
            lines.extend(_issue_lines(issue))
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _issue_lines(issue: Issue) -> list[str]:
    scope = ""
    if issue.model_id:
        scope += f" model=`{issue.model_id}`"
    if issue.test_name:
        scope += f" test=`{issue.test_name}`"
    return [f"- **{issue.severity}**{scope}: {issue.message}"]


def _stability_markdown(report: StabilityReport) -> str:
    lines: list[str] = [
        f"# Skulk Stability Suite: {report.suite}",
        "",
        f"- Run: `{report.run_id}`",
        f"- Model: `{report.model_id}`",
        f"- Started: `{report.started_at.isoformat()}`",
        f"- Finished: `{report.finished_at.isoformat() if report.finished_at else 'running'}`",
        f"- Result: **{'PASS' if report.passed else 'FAIL'}**",
        "",
    ]
    if report.latency is not None:
        latency = report.latency
        lines.extend(
            [
                "## Latency",
                "",
                f"- Successful completions: {latency.count}",
                f"- Failures: {latency.failures}",
                f"- p50: {_fmt(latency.p50_s)} s",
                f"- p95: {_fmt(latency.p95_s)} s",
                f"- max: {_fmt(latency.max_s)} s",
                "",
            ]
        )

    lines.extend(["## Observations", ""])
    if report.observations:
        for key in sorted(report.observations):
            lines.append(f"- `{key}`: {report.observations[key]}")
    else:
        lines.append("- None")

    lines.extend(["", "## Issues", ""])
    if report.issues:
        for issue in report.issues:
            lines.extend(_issue_lines(issue))
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"
