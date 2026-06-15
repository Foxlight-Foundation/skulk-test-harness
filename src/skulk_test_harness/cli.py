"""Command-line interface for the Skulk test harness."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from skulk_test_harness import stability
from skulk_test_harness.client import SkulkClient
from skulk_test_harness.goal_parser import parse_goal
from skulk_test_harness.models import (
    HarnessConfig,
    PlacementPolicy,
    RunSpec,
    StabilityReport,
)
from skulk_test_harness.orchestrator import HarnessRunner
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.specs import load_config, load_model_sets, load_test_sets

app = typer.Typer(help="Agent-controlled Skulk end-to-end test and benchmark harness.")
models_app = typer.Typer(help="Inspect model sets and live Skulk model catalog.")
tests_app = typer.Typer(help="Inspect named test sets.")
stability_app = typer.Typer(help="Run cluster stability suites (failover/churn/soak/refusal).")
app.add_typer(models_app, name="models")
app.add_typer(tests_app, name="tests")
app.add_typer(stability_app, name="stability")

console = Console()

# A small model that can shard across the kite TP pair, used as the default
# target for the stability suites.
DEFAULT_STABILITY_MODEL = "mlx-community/Qwen3.5-9B-4bit"


ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Harness config YAML. Defaults to skulk-harness.yaml if present.",
    ),
]


def _load_runner(config_path: Path) -> tuple[HarnessConfig, HarnessRunner]:
    config = load_config(config_path)
    model_sets = load_model_sets(config.model_sets_path).model_sets
    test_sets = load_test_sets(config.test_sets_path).test_sets
    return config, HarnessRunner(
        config=config,
        model_sets=model_sets,
        test_sets=test_sets,
    )


@app.command()
def doctor(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """Check the configured Skulk API and print a compact cluster summary."""

    cfg = load_config(config)
    with SkulkClient(
        cfg.api_base_url,
        request_timeout_s=cfg.request_timeout_s,
        generation_timeout_s=cfg.generation_timeout_s,
    ) as client:
        node_id = client.get_node_id()
        state = client.get_state()
        models = client.list_models()
        issues = client.detect_runner_state_drift()

    instances = _dict_field(state, "instances")
    runners = _dict_field(state, "runners")
    memory = _dict_field(state, "nodeMemory")

    table = Table(title="Skulk Harness Doctor")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("API", cfg.api_base_url)
    table.add_row("API node", node_id)
    table.add_row("Known models", str(len(models)))
    table.add_row("Cluster memory nodes", str(len(memory)))
    table.add_row("Instances", str(len(instances)))
    table.add_row("Runner states", str(len(runners)))
    table.add_row("State drift issues", str(len(issues)))
    console.print(table)
    for issue in issues:
        console.print(f"[yellow]warning[/yellow] {issue.message} {issue.evidence}")


@models_app.command("sets")
def list_model_sets(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """List configured named model sets."""

    cfg = load_config(config)
    model_sets = load_model_sets(cfg.model_sets_path).model_sets
    table = Table(title="Model Sets")
    table.add_column("Name")
    table.add_column("Explicit")
    table.add_column("Selectors")
    table.add_column("HF Seeds")
    table.add_column("Description")
    for name, model_set in model_sets.items():
        table.add_row(
            name,
            str(len(model_set.models)),
            str(len(model_set.selectors)),
            str(len(model_set.huggingface_seeds)),
            model_set.description,
        )
    console.print(table)


@models_app.command("catalog")
def list_catalog(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """List live Skulk catalog models from the configured API."""

    cfg = load_config(config)
    with SkulkClient(cfg.api_base_url, request_timeout_s=cfg.request_timeout_s) as client:
        catalog = client.list_models()
    table = Table(title="Skulk Model Catalog")
    table.add_column("Model")
    table.add_column("Family")
    table.add_column("Tasks")
    table.add_column("Capabilities")
    for item in catalog:
        model_id = str(item.get("hugging_face_id") or item.get("id") or item.get("name"))
        table.add_row(
            model_id,
            str(item.get("family") or ""),
            ", ".join(_string_list(item.get("tasks"))),
            ", ".join(_string_list(item.get("capabilities"))),
        )
    console.print(table)


@models_app.command("store")
def list_store(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """List models currently registered in the Skulk model store."""

    cfg = load_config(config)
    with SkulkClient(cfg.api_base_url, request_timeout_s=cfg.request_timeout_s) as client:
        registry = client.get_store_registry() or {}
    entries = _registry_entries(registry)
    table = Table(title="Skulk Model Store")
    table.add_column("Model")
    table.add_column("Bytes")
    table.add_column("Downloaded")
    for entry in entries:
        table.add_row(
            str(entry.get("model_id") or entry.get("id") or ""),
            str(entry.get("total_bytes") or ""),
            str(entry.get("downloaded_at") or ""),
        )
    console.print(table)


@models_app.command("add")
def add_model(
    model_id: Annotated[str, typer.Argument(help="Hugging Face model ID.")],
    config: ConfigPath = Path("skulk-harness.yaml"),
) -> None:
    """Ask Skulk to add/fetch a model card."""

    cfg = load_config(config)
    with SkulkClient(cfg.api_base_url, request_timeout_s=cfg.request_timeout_s) as client:
        payload = client.add_model_card(model_id)
    console.print(payload or {"status": "ok", "model_id": model_id})


@models_app.command("download")
def download_model(
    model_id: Annotated[str, typer.Argument(help="Hugging Face model ID.")],
    config: ConfigPath = Path("skulk-harness.yaml"),
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = False,
) -> None:
    """Request a model-store download for a model."""

    cfg = load_config(config)
    with SkulkClient(
        cfg.api_base_url,
        request_timeout_s=cfg.request_timeout_s,
        generation_timeout_s=cfg.generation_timeout_s,
    ) as client:
        payload = client.request_store_download(model_id)
        console.print(payload or {"status": "requested", "model_id": model_id})
        if wait:
            deadline = time.monotonic() + cfg.store_download_timeout_s
            while time.monotonic() < deadline:
                status = client.get_store_download_status(model_id) or {}
                console.print(status)
                status_text = str(status.get("status") or status.get("state") or "").lower()
                if status_text in {"complete", "completed", "ready", "succeeded"}:
                    return
                if status_text in {"failed", "error"}:
                    raise typer.Exit(code=1)
                time.sleep(cfg.poll_interval_s)
            raise typer.Exit(code=124)


@tests_app.command("sets")
def list_test_sets(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """List configured named test sets."""

    cfg = load_config(config)
    test_sets = load_test_sets(cfg.test_sets_path).test_sets
    table = Table(title="Test Sets")
    table.add_column("Name")
    table.add_column("Tests")
    table.add_column("Description")
    for name, test_set in test_sets.items():
        table.add_row(name, str(len(test_set.tests)), test_set.description)
    console.print(table)


@app.command()
def plan(
    model_set: Annotated[str, typer.Option("--model-set", "-m")],
    test_set: Annotated[str, typer.Option("--test-set", "-t")],
    config: ConfigPath = Path("skulk-harness.yaml"),
    sharding: Annotated[str, typer.Option(help="Pipeline or Tensor")] = "Pipeline",
    instance_meta: Annotated[str, typer.Option(help="MlxRing or MlxJaccl")] = "MlxRing",
    min_nodes: Annotated[int | None, typer.Option(help="Minimum node count override")] = None,
) -> None:
    """Plan a harness run without mutating the cluster."""

    cfg, runner = _load_runner(config)
    spec = RunSpec(
        model_set=model_set,
        test_set=test_set,
        mode="plan",
        placement=PlacementPolicy(
            sharding=sharding,  # type: ignore[arg-type]
            instance_meta=instance_meta,  # type: ignore[arg-type]
            min_nodes=min_nodes,
        ),
    )
    report = runner.plan(spec)
    run_dir = ReportWriter(cfg.output_dir).write(report)
    _print_report_summary(report, run_dir)


@app.command()
def run(
    model_set: Annotated[str, typer.Option("--model-set", "-m")],
    test_set: Annotated[str, typer.Option("--test-set", "-t")],
    config: ConfigPath = Path("skulk-harness.yaml"),
    execute: Annotated[
        bool,
        typer.Option(
            "--execute/--dry-run",
            help="Actually mutate the cluster and run tests. Dry-run writes a plan.",
        ),
    ] = False,
    ensure_store_downloads: Annotated[
        bool, typer.Option(help="Request model-store downloads before placement.")
    ] = False,
    retain_instances: Annotated[
        bool,
        typer.Option("--retain-instances/--delete-created-instances"),
    ] = True,
    sharding: Annotated[str, typer.Option(help="Pipeline or Tensor")] = "Pipeline",
    instance_meta: Annotated[str, typer.Option(help="MlxRing or MlxJaccl")] = "MlxRing",
    min_nodes: Annotated[int | None, typer.Option(help="Minimum node count override")] = None,
) -> None:
    """Run or dry-run a named test set against a named model set."""

    cfg, runner = _load_runner(config)
    spec = RunSpec(
        model_set=model_set,
        test_set=test_set,
        mode="execute" if execute else "plan",
        ensure_store_downloads=ensure_store_downloads,
        retain_instances=retain_instances,
        placement=PlacementPolicy(
            sharding=sharding,  # type: ignore[arg-type]
            instance_meta=instance_meta,  # type: ignore[arg-type]
            min_nodes=min_nodes,
        ),
    )
    report = runner.execute(spec) if execute else runner.plan(spec)
    run_dir = ReportWriter(cfg.output_dir).write(report)
    _print_report_summary(report, run_dir)


@app.command()
def goal(
    text: Annotated[str, typer.Argument(help="Natural-language harness goal.")],
    config: ConfigPath = Path("skulk-harness.yaml"),
    execute: Annotated[
        bool,
        typer.Option("--execute/--dry-run", help="Execute the parsed goal."),
    ] = False,
) -> None:
    """Parse a constrained natural-language goal into a plan or run."""

    cfg, runner = _load_runner(config)
    spec = parse_goal(
        text,
        model_set_names=list(runner.model_sets),
        test_set_names=list(runner.test_sets),
        execute=execute,
    )
    report = runner.execute(spec) if execute else runner.plan(spec)
    run_dir = ReportWriter(cfg.output_dir).write(report)
    _print_report_summary(report, run_dir)


ModelOption = Annotated[
    str,
    typer.Option("--model", "-m", help="Model ID to exercise (multinode-capable)."),
]


def _stability_client(cfg: HarnessConfig) -> SkulkClient:
    return SkulkClient(
        cfg.api_base_url,
        request_timeout_s=cfg.request_timeout_s,
        generation_timeout_s=cfg.generation_timeout_s,
    )


def _write_stability(cfg: HarnessConfig, report: StabilityReport) -> None:
    run_dir = ReportWriter(cfg.output_dir).write_stability(report)
    _print_stability_summary(report, run_dir)


@stability_app.command("failover")
def stability_failover(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    min_nodes: Annotated[int, typer.Option(help="Minimum nodes to place across.")] = 2,
) -> None:
    """Crash the master mid-stream and assert the cluster survives (#273)."""

    cfg = load_config(config)
    with _stability_client(cfg) as client:
        report = stability.run_failover(client, cfg, model, min_nodes=min_nodes)
    _write_stability(cfg, report)


@stability_app.command("churn")
def stability_churn(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    rounds: Annotated[int, typer.Option(help="Kill/relaunch rounds to run.")] = 3,
) -> None:
    """Repeatedly crash and relaunch a non-master node, asserting recovery."""

    cfg = load_config(config)
    with _stability_client(cfg) as client:
        report = stability.run_churn(client, cfg, model, rounds=rounds)
    _write_stability(cfg, report)


@stability_app.command("soak")
def stability_soak(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    concurrency: Annotated[int, typer.Option(help="Concurrent completion workers.")] = 4,
    duration_s: Annotated[float, typer.Option(help="Soak duration in seconds.")] = 120.0,
) -> None:
    """Drive sustained concurrent load and report latency/failures."""

    cfg = load_config(config)
    with _stability_client(cfg) as client:
        report = stability.run_soak(
            client, cfg, model, concurrency=concurrency, duration_s=duration_s
        )
    _write_stability(cfg, report)


@stability_app.command("refusal")
def stability_refusal(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
) -> None:
    """Assert an impossible placement is refused or re-placed, not wedged (#290)."""

    cfg = load_config(config)
    with _stability_client(cfg) as client:
        report = stability.run_placement_refusal(client, cfg, model)
    _write_stability(cfg, report)


def _print_stability_summary(report: StabilityReport, run_dir: Path) -> None:
    error_count = sum(1 for issue in report.issues if issue.severity == "error")
    warning_count = sum(1 for issue in report.issues if issue.severity == "warning")
    table = Table(title=f"Stability {report.suite}: {report.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Model", report.model_id)
    table.add_row("Result", "PASS" if report.passed else "FAIL")
    table.add_row("Errors", str(error_count))
    table.add_row("Warnings", str(warning_count))
    if report.latency is not None:
        table.add_row("Completions", str(report.latency.count))
        table.add_row("Failures", str(report.latency.failures))
        if report.latency.p50_s is not None:
            table.add_row("p50 s", f"{report.latency.p50_s:.2f}")
        if report.latency.p95_s is not None:
            table.add_row("p95 s", f"{report.latency.p95_s:.2f}")
    table.add_row("Report dir", str(run_dir))
    console.print(table)
    for issue in report.issues:
        color = "red" if issue.severity == "error" else "yellow"
        console.print(f"[{color}]{issue.severity}[/{color}] {issue.message}")


def _print_report_summary(report, run_dir: Path) -> None:  # noqa: ANN001
    passed = sum(1 for result in report.results if result.passed)
    failed = sum(1 for result in report.results if not result.passed)
    table = Table(title=f"Harness Run {report.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Models", str(len(report.models)))
    table.add_row("Placements", str(len(report.placements)))
    table.add_row("Results passed", str(passed))
    table.add_row("Results failed", str(failed))
    table.add_row(
        "Issues",
        str(len(report.issues) + sum(len(result.issues) for result in report.results)),
    )
    table.add_row("Report dir", str(run_dir))
    console.print(table)


def _dict_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _registry_entries(registry: dict[str, object]) -> list[dict[str, object]]:
    entries = registry.get("entries")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []
