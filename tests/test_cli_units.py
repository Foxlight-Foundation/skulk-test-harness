"""CLI-level tests for the `run` command's exit-code gating."""

from __future__ import annotations

from pathlib import Path

import pytest
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
        self.model_sets = {"s": object()}
        self.test_sets = {"t": object()}
        self.observed_specs: list[RunSpec] = []

    def execute(self, spec: RunSpec) -> RunReport:
        self.observed_specs.append(spec)
        return _report(spec, passed=self._passed)

    def plan(self, spec: RunSpec) -> RunReport:
        self.observed_specs.append(spec)
        return _report(spec, passed=self._passed)


def _patch(monkeypatch, tmp_path: Path, *, passed: bool) -> _StubRunner:
    cfg = HarnessConfig(output_dir=tmp_path)
    stub = _StubRunner(passed=passed)
    monkeypatch.setattr(cli, "_load_runner", lambda _config: (cfg, stub))
    return stub


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


def test_run_treats_requested_placement_as_exact_contract(
    monkeypatch, tmp_path
) -> None:
    """Executed CLI flags must not silently fall back to another placement."""

    stub = _patch(monkeypatch, tmp_path, passed=True)

    result = runner_cli.invoke(
        cli.app,
        [
            "run",
            "-m",
            "s",
            "-t",
            "t",
            "--execute",
            "--sharding",
            "Tensor",
            "--instance-meta",
            "MlxJaccl",
            "--min-nodes",
            "2",
            "--max-nodes",
            "2",
        ],
    )

    assert result.exit_code == 0
    policy = stub.observed_specs[0].placement
    assert policy.strategy == "exact"
    assert policy.sharding == "Tensor"
    assert policy.instance_meta == "MlxJaccl"
    assert policy.min_nodes == 2
    assert policy.max_nodes == 2


def test_plan_uses_the_same_exact_placement_contract(monkeypatch, tmp_path) -> None:
    """A dry plan must describe the same strict shape the live run will use."""

    stub = _patch(monkeypatch, tmp_path, passed=True)

    result = runner_cli.invoke(
        cli.app,
        [
            "plan",
            "-m",
            "s",
            "-t",
            "t",
            "--sharding",
            "Tensor",
            "--instance-meta",
            "MlxRing",
            "--min-nodes",
            "2",
            "--max-nodes",
            "2",
        ],
    )

    assert result.exit_code == 0
    policy = stub.observed_specs[0].placement
    assert policy.strategy == "exact"
    assert policy.sharding == "Tensor"
    assert policy.instance_meta == "MlxRing"
    assert policy.min_nodes == 2
    assert policy.max_nodes == 2


def test_shipping_transport_requirement_accepts_uniform_fleet() -> None:
    cfg = HarnessConfig(required_data_transport="zenoh")
    state: dict[str, object] = {
        "nodeResources": {
            "peer-a": {"dataTransport": "zenoh"},
            "peer-b": {"dataTransport": "zenoh"},
        },
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "peer-b": {"friendlyName": "beta"},
        },
    }

    cli._require_shipping_data_transport(cfg, state)


def test_shipping_transport_requirement_rejects_mixed_fleet() -> None:
    cfg = HarnessConfig(required_data_transport="zenoh")
    state: dict[str, object] = {
        "nodeResources": {
            "peer-a": {"dataTransport": "zenoh"},
            "peer-b": {"dataTransport": "gossipsub"},
        },
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "peer-b": {"friendlyName": "beta"},
        },
    }

    with pytest.raises(
        ValueError,
        match="shipping-profile violation.*beta=gossipsub",
    ):
        cli._require_shipping_data_transport(cfg, state)


def test_shipping_transport_requirement_rejects_live_node_without_resources() -> None:
    cfg = HarnessConfig(required_data_transport="zenoh")
    state: dict[str, object] = {
        "nodeResources": {
            "peer-a": {"dataTransport": "zenoh"},
        },
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "peer-b": {"friendlyName": "beta"},
        },
    }

    with pytest.raises(
        ValueError,
        match="shipping-profile violation.*beta=missing",
    ):
        cli._require_shipping_data_transport(cfg, state)


def test_shipping_transport_requirement_rejects_resource_without_identity() -> None:
    cfg = HarnessConfig(required_data_transport="zenoh")
    state: dict[str, object] = {
        "nodeResources": {
            "peer-a": {"dataTransport": "zenoh"},
            "peer-b": {"dataTransport": "gossipsub"},
        },
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
        },
    }

    with pytest.raises(
        ValueError,
        match="shipping-profile violation.*peer-b=gossipsub",
    ):
        cli._require_shipping_data_transport(cfg, state)


def test_shipping_transport_requirement_rejects_missing_advertisements() -> None:
    cfg = HarnessConfig(required_data_transport="zenoh")

    with pytest.raises(ValueError, match="no nodeResources transport advertisements"):
        cli._require_shipping_data_transport(cfg, {})


def test_generic_profile_has_no_shipping_transport_requirement() -> None:
    cli._require_shipping_data_transport(HarnessConfig(), {})


def test_eligible_fleet_ignores_incidental_transport() -> None:
    cfg = HarnessConfig(
        required_data_transport="zenoh",
        eligible_fleet_nodes=["alpha", "beta"],
    )
    state: dict[str, object] = {
        "nodeResources": {
            "peer-a": {"dataTransport": "zenoh"},
            "peer-b": {"dataTransport": "zenoh"},
            "incidental": {"dataTransport": "gossipsub"},
        },
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "peer-b": {"friendlyName": "beta"},
            "incidental": {"friendlyName": "unmanaged"},
        },
    }

    cli._require_shipping_data_transport(cfg, state)
    assert cli._placement_scope_from_state(cfg, state) == (
        ["peer-a", "peer-b"],
        ["incidental"],
    )


def test_eligible_fleet_requires_every_configured_node() -> None:
    cfg = HarnessConfig(eligible_fleet_nodes=["alpha", "beta"])
    state: dict[str, object] = {
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "incidental": {"friendlyName": "unmanaged"},
        }
    }

    with pytest.raises(ValueError, match=r"required node\(s\) are absent.*beta"):
        cli._placement_scope_from_state(cfg, state)


def test_eligible_fleet_rejects_ambiguous_friendly_names() -> None:
    cfg = HarnessConfig(eligible_fleet_nodes=["alpha"])
    state: dict[str, object] = {
        "nodeIdentities": {
            "peer-a": {"friendlyName": "alpha"},
            "peer-b": {"friendlyName": "alpha"},
        }
    }

    with pytest.raises(ValueError, match="friendly name.*ambiguous"):
        cli._placement_scope_from_state(cfg, state)


def test_goal_execute_uses_shared_execution_preflight(monkeypatch, tmp_path) -> None:
    stub = _patch(monkeypatch, tmp_path, passed=True)
    observed: list[tuple[HarnessConfig, bool]] = []

    def record_preflight(
        cfg: HarnessConfig, *, force: bool
    ) -> tuple[list[str], list[str]]:
        observed.append((cfg, force))
        return ["eligible-a", "eligible-b"], ["incidental"]

    monkeypatch.setattr(cli, "_require_execution_preflight", record_preflight)

    result = runner_cli.invoke(cli.app, ["goal", "run s on t", "--execute"])

    assert result.exit_code == 0
    assert observed == [(HarnessConfig(output_dir=tmp_path), False)]
    assert stub.observed_specs[0].placement.eligible_nodes == [
        "eligible-a",
        "eligible-b",
    ]
    assert stub.observed_specs[0].placement.excluded_nodes == ["incidental"]


@pytest.mark.parametrize(
    "args",
    [
        ["stability", "failover"],
        ["stability", "churn"],
        ["stability", "refusal"],
    ],
)
def test_destructive_stability_commands_require_explicit_opt_in(monkeypatch, args) -> None:
    def fail_if_loaded(_config: Path) -> HarnessConfig:
        raise AssertionError("config should not load before destructive opt-in")

    monkeypatch.setattr(cli, "load_config", fail_if_loaded)

    result = runner_cli.invoke(cli.app, args)

    assert result.exit_code == 2
    assert "Refusing destructive stability command" in result.output
