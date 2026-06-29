"""CLI-level tests for the `run` command's exit-code gating."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from skulk_test_harness import cli
from skulk_test_harness.models import (
    GenerationMetrics,
    HarnessConfig,
    RunReport,
    RunSpec,
)
from skulk_test_harness.models import TestResult as _TestResult

runner_cli = CliRunner()


def _report(spec: RunSpec, *, passed: bool) -> RunReport:
    rep = RunReport.start("test-run", spec, [])
    rep.results.append(
        _TestResult(
            model_id="m",
            test_name="t",
            repetition=1,
            passed=passed,
            output_text="ok" if passed else "",
            metrics=GenerationMetrics(elapsed_s=0.0),
        )
    )
    return rep


class _StubRunner:
    def __init__(self, *, passed: bool) -> None:
        self._passed = passed

    def execute(self, spec: RunSpec) -> RunReport:
        return _report(spec, passed=self._passed)

    def plan(self, spec: RunSpec) -> RunReport:
        return _report(spec, passed=self._passed)


def _patch(monkeypatch, tmp_path: Path, *, passed: bool) -> None:
    cfg = HarnessConfig(output_dir=tmp_path)
    monkeypatch.setattr(cli, "_load_runner", lambda _config: (cfg, _StubRunner(passed=passed)))


def test_run_exits_nonzero_on_failed_result(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch, tmp_path, passed=False)
    result = runner_cli.invoke(cli.app, ["run", "-m", "s", "-t", "t", "--execute"])
    assert result.exit_code == 1


def test_run_exits_zero_when_all_pass(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch, tmp_path, passed=True)
    result = runner_cli.invoke(cli.app, ["run", "-m", "s", "-t", "t", "--execute"])
    assert result.exit_code == 0


def test_run_no_fail_on_issue_stays_zero_despite_failure(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch, tmp_path, passed=False)
    result = runner_cli.invoke(
        cli.app, ["run", "-m", "s", "-t", "t", "--execute", "--no-fail-on-issue"]
    )
    assert result.exit_code == 0


def test_dry_run_does_not_gate(monkeypatch, tmp_path) -> None:
    # A plan (dry-run) has nothing to fail on; it must not exit non-zero even if a
    # (stub) result is marked failed.
    _patch(monkeypatch, tmp_path, passed=False)
    result = runner_cli.invoke(cli.app, ["run", "-m", "s", "-t", "t", "--dry-run"])
    assert result.exit_code == 0
