"""Command-line interface for the Skulk test harness."""

from __future__ import annotations

import json
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from skulk_test_harness import __version__, stability
from skulk_test_harness import submit as submit_module
from skulk_test_harness.client import SkulkClient
from skulk_test_harness.compare import compare, load_reports, select_run_dirs
from skulk_test_harness.fleet_lock import FleetLockStore
from skulk_test_harness.goal_parser import parse_goal
from skulk_test_harness.models import (
    ComparisonRecord,
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
fleet_app = typer.Typer(
    help="Coordinate exclusive access to a shared test fleet across agents."
)
app.add_typer(models_app, name="models")
app.add_typer(tests_app, name="tests")
app.add_typer(stability_app, name="stability")
app.add_typer(fleet_app, name="fleet")

console = Console()


def _version_callback(value: bool) -> None:
    """Print the installed harness version and exit (for ``--version``)."""

    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the harness version and exit.",
        ),
    ] = False,
) -> None:
    """Agent-controlled Skulk end-to-end test and benchmark harness."""


# A small public model that can shard across multiple nodes, used as the default
# target for the stability suites when the operator does not pass --model.
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


def _load_fleet_store(cfg: HarnessConfig) -> FleetLockStore | None:
    """Build the fleet-lock store, or ``None`` when the lease is not configured."""

    if cfg.fleet_lock is None:
        return None
    return FleetLockStore(cfg.fleet_lock)


def _require_fleet_or_refuse(cfg: HarnessConfig, *, force: bool) -> None:
    """Refuse to touch the fleet when another agent holds the lease.

    A safety net for the execute paths: the primary mechanism is the explicit
    ``fleet acquire`` / ``fleet release`` bracket an agent wraps a whole deploy
    session in. Does nothing when the lease is unconfigured (single-operator
    use) or already held by this agent. ``force`` overrides the refusal.
    """

    store = _load_fleet_store(cfg)
    if store is None:
        return
    assert cfg.fleet_lock is not None
    lease = store.read()
    now = datetime.now(UTC)
    if lease.is_held(now) and lease.holder != cfg.fleet_lock.holder:
        if force:
            console.print(
                f"[yellow]--force[/]: proceeding although the fleet is held by "
                f"{lease.holder} (branch {lease.branch})"
            )
            return
        console.print(
            f"[bold red]REFUSED[/]: the shared fleet is held by {lease.holder} "
            f"(branch {lease.branch}, expires {lease.expires_at}). Wait for it to "
            "free, or pass --force to override."
        )
        raise typer.Exit(code=1)


def _require_shipping_data_transport(
    cfg: HarnessConfig, state: dict[str, object]
) -> None:
    """Refuse an E2E run whose live fleet is not on the required DATA transport.

    The Foxlight battery is release qualification, so testing a hand-configured
    transport that differs from a fresh Skulk installation gives false
    confidence. ``nodeResources`` carries each node's startup-resolved transport;
    require a complete, uniform match before any placement or model-store
    mutation. Generic harness profiles leave the requirement unset.
    """
    required = cfg.required_data_transport
    if required is None:
        return
    resources = _dict_field(state, "nodeResources")
    identities = _dict_field(state, "nodeIdentities")
    if not resources:
        raise ValueError(
            f"required_data_transport={required!r}, but /state has no "
            "nodeResources transport advertisements"
        )

    mismatches: list[str] = []
    for node_id, raw_resource in resources.items():
        identity = identities.get(node_id)
        friendly_name = (
            identity.get("friendlyName")
            if isinstance(identity, dict)
            and isinstance(identity.get("friendlyName"), str)
            else node_id
        )
        observed = (
            raw_resource.get("dataTransport")
            if isinstance(raw_resource, dict)
            else None
        )
        if observed != required:
            rendered = observed if isinstance(observed, str) else "missing"
            mismatches.append(f"{friendly_name}={rendered}")

    if mismatches:
        raise ValueError(
            f"E2E shipping-profile violation: required DATA transport "
            f"{required!r}, observed {', '.join(sorted(mismatches))}. Refusing "
            "to qualify a runtime path different from the one we ship."
        )


def _print_lease(store: FleetLockStore) -> None:
    lease = store.read()
    now = datetime.now(UTC)
    table = Table(title="Fleet lease")
    table.add_column("Field")
    table.add_column("Value")
    held = lease.is_held(now)
    table.add_row("state", "HELD" if held else "free")
    if lease.state == "held" and not held:
        table.add_row("(note)", "lock is past its TTL and is treated as free")
    for label, value in (
        ("holder", lease.holder),
        ("branch", lease.branch),
        ("host", lease.host),
        ("battery", lease.battery),
        ("acquired_at", lease.acquired_at),
        ("expires_at", lease.expires_at),
        ("heartbeat_at", lease.heartbeat_at),
        ("note", lease.note),
    ):
        if value:
            table.add_row(label, str(value))
    console.print(table)


@fleet_app.command("status")
def fleet_status(config: ConfigPath = Path("skulk-harness.yaml")) -> None:
    """Show the current fleet lease."""

    cfg = load_config(config)
    store = _load_fleet_store(cfg)
    if store is None:
        console.print(
            "fleet lock is not configured (no `fleet_lock` in the harness "
            "config); shared-fleet coordination is disabled."
        )
        return
    _print_lease(store)


@fleet_app.command("acquire")
def fleet_acquire(
    branch: Annotated[
        str, typer.Option("--branch", help="Branch being deployed to the fleet.")
    ],
    config: ConfigPath = Path("skulk-harness.yaml"),
    host: Annotated[
        str | None,
        typer.Option("--host", help="Driving host (defaults to this machine)."),
    ] = None,
    battery: Annotated[
        str | None, typer.Option("--battery", help="Battery/suite being run.")
    ] = None,
    ttl_minutes: Annotated[
        float | None,
        typer.Option("--ttl-minutes", help="Lease lifetime; config default if unset."),
    ] = None,
    note: Annotated[str | None, typer.Option("--note")] = None,
) -> None:
    """Acquire the shared fleet, or refuse if another agent holds it."""

    cfg = load_config(config)
    store = _load_fleet_store(cfg)
    if store is None:
        console.print("fleet lock is not configured; nothing to acquire.")
        return
    outcome = store.acquire(
        branch=branch,
        host=host or socket.gethostname(),
        battery=battery,
        ttl_s=None if ttl_minutes is None else ttl_minutes * 60.0,
        note=note,
    )
    if outcome.ok:
        console.print(f"[bold green]OK[/]: {outcome.message}")
        _print_lease(store)
        return
    console.print(f"[bold red]REFUSED[/]: {outcome.message}")
    raise typer.Exit(code=1)


@fleet_app.command("extend")
def fleet_extend(
    config: ConfigPath = Path("skulk-harness.yaml"),
    ttl_minutes: Annotated[
        float | None,
        typer.Option("--ttl-minutes", help="New lifetime; config default if unset."),
    ] = None,
) -> None:
    """Push the lease TTL forward (holder only). Use during a long battery."""

    cfg = load_config(config)
    store = _load_fleet_store(cfg)
    if store is None:
        console.print("fleet lock is not configured; nothing to extend.")
        return
    outcome = store.extend(
        ttl_s=None if ttl_minutes is None else ttl_minutes * 60.0
    )
    if outcome.ok:
        console.print(f"[bold green]OK[/]: {outcome.message}")
        _print_lease(store)
        return
    console.print(f"[bold red]FAILED[/]: {outcome.message}")
    raise typer.Exit(code=1)


@fleet_app.command("release")
def fleet_release(
    config: ConfigPath = Path("skulk-harness.yaml"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Release even if another agent holds it (break a stuck lock).",
        ),
    ] = False,
) -> None:
    """Release the shared fleet so another agent can take it."""

    cfg = load_config(config)
    store = _load_fleet_store(cfg)
    if store is None:
        console.print("fleet lock is not configured; nothing to release.")
        return
    outcome = store.release(force=force)
    if outcome.ok:
        console.print(f"[bold green]OK[/]: {outcome.message}")
        return
    console.print(f"[bold red]FAILED[/]: {outcome.message}")
    raise typer.Exit(code=1)


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
    instance_meta: Annotated[str, typer.Option(help="MlxRing, MlxJaccl, or LlamaRpc")] = "MlxRing",
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
    delete_staged_models: Annotated[
        bool,
        typer.Option(
            "--delete-staged-models",
            help="Evict each model's staged weights from the store after its run "
            "(benchmark hygiene; off by default to keep the store warm).",
        ),
    ] = False,
    sharding: Annotated[str, typer.Option(help="Pipeline or Tensor")] = "Pipeline",
    instance_meta: Annotated[str, typer.Option(help="MlxRing, MlxJaccl, or LlamaRpc")] = "MlxRing",
    min_nodes: Annotated[int | None, typer.Option(help="Minimum node count override")] = None,
    exclude_nodes: Annotated[
        str | None,
        typer.Option(
            "--exclude-nodes",
            help="Comma-separated friendly node names (e.g. 'node-a') to exclude "
            "from placement. Used to force a model onto a specific node the "
            "planner would not otherwise pick -- e.g. exclude the larger AMD "
            "node so a GGUF/served cell lands on the smaller one for coverage.",
        ),
    ] = None,
    fail_on_issue: Annotated[
        bool,
        typer.Option(
            "--fail-on-issue/--no-fail-on-issue",
            help="Exit non-zero when any test result fails or an error-severity "
            "issue is recorded (execute mode only). On by default so a battery / "
            "CI goes red on a regression instead of silently green.",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Run or dry-run a named test set against a named model set."""

    cfg, runner = _load_runner(config)
    # An executed run mutates the shared fleet; refuse if another agent holds the
    # lease. A dry-run only reads/plans, so it never needs the fleet.
    if execute:
        _require_fleet_or_refuse(cfg, force=force)
        if cfg.required_data_transport is not None:
            with SkulkClient(
                cfg.api_base_url, request_timeout_s=cfg.request_timeout_s
            ) as client:
                try:
                    _require_shipping_data_transport(cfg, client.get_state())
                except ValueError as exception:
                    console.print(f"[bold red]REFUSED[/]: {exception}")
                    raise typer.Exit(code=1) from exception
    # Resolve friendly names to live libp2p node IDs before building the
    # spec: placement exclusion is by node ID, but node IDs are ephemeral so a
    # battery cell can only name a node by its stable friendly name.
    excluded_node_ids: list[str] = []
    requested = [n.strip() for n in (exclude_nodes or "").split(",") if n.strip()]
    if requested:
        with SkulkClient(
            cfg.api_base_url, request_timeout_s=cfg.request_timeout_s
        ) as client:
            excluded_node_ids = client.resolve_node_ids(requested)
    spec = RunSpec(
        model_set=model_set,
        test_set=test_set,
        mode="execute" if execute else "plan",
        ensure_store_downloads=ensure_store_downloads,
        retain_instances=retain_instances,
        delete_staged_models=delete_staged_models,
        placement=PlacementPolicy(
            sharding=sharding,  # type: ignore[arg-type]
            instance_meta=instance_meta,  # type: ignore[arg-type]
            min_nodes=min_nodes,
            excluded_nodes=excluded_node_ids,
        ),
    )
    report = runner.execute(spec) if execute else runner.plan(spec)
    run_dir = ReportWriter(cfg.output_dir).write(report)
    _print_report_summary(report, run_dir)

    # The exit code is the gate: without this, `run` always exited 0 even when a
    # result failed, so batteries (and the served-MTP correctness cell) looked
    # green on a real regression. Only judge an executed run -- a dry-run plan has
    # no results to fail on.
    if execute and fail_on_issue:
        result_failed = any(not r.passed for r in report.results)
        run_errored = any(i.severity == "error" for i in report.issues)
        # Coverage first: an incomplete run (some models never placed/tested) is
        # not a clean pass even with zero test failures. Report it explicitly so
        # "0 tests failed" can never be mistaken for "everything ran".
        tested_models = len({r.model_id for r in report.results})
        total_models = len(report.models)
        if tested_models < total_models:
            console.print(
                f"[bold yellow]COVERAGE[/]: only {tested_models}/{total_models} "
                "models placed and tested; the rest never became ready "
                "(see run-level issues)"
            )
        if result_failed or run_errored:
            failed = sum(1 for r in report.results if not r.passed)
            console.print(
                f"[bold red]FAIL[/]: {failed} test result(s) failed"
                + (" + run-level error issue(s)" if run_errored else "")
                + f"  [models tested {tested_models}/{total_models}]"
            )
            raise typer.Exit(code=1)
        # An executed run that placed NOTHING is a silent skip, not a pass:
        # the first pooled-rpc cell run no-opped (0 placements, 0 results,
        # rc=0) because no preview matched during the telemetry warm-up
        # window, and the battery read green. Zero placements with models
        # resolved means the cell never tested anything; fail loud.
        if report.results == [] and not report.placements:
            console.print(
                "[bold red]FAIL[/]: run placed no instances and produced no "
                "results (silent skip -- check placement previews/telemetry)"
            )
            raise typer.Exit(code=1)


@app.command()
def goal(
    text: Annotated[str, typer.Argument(help="Natural-language harness goal.")],
    config: ConfigPath = Path("skulk-harness.yaml"),
    execute: Annotated[
        bool,
        typer.Option("--execute/--dry-run", help="Execute the parsed goal."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Parse a constrained natural-language goal into a plan or run."""

    cfg, runner = _load_runner(config)
    if execute:
        _require_fleet_or_refuse(cfg, force=force)
    spec = parse_goal(
        text,
        model_set_names=list(runner.model_sets),
        test_set_names=list(runner.test_sets),
        execute=execute,
    )
    report = runner.execute(spec) if execute else runner.plan(spec)
    run_dir = ReportWriter(cfg.output_dir).write(report)
    _print_report_summary(report, run_dir)


@app.command("compare")
def compare_runs(
    baseline: Annotated[
        str,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline run selector: a run directory, or a substring matched "
            "against run-dir names (e.g. 'dense-singles' or a run-id prefix).",
        ),
    ],
    candidate: Annotated[
        str,
        typer.Option(
            "--candidate",
            "-n",
            help="Candidate run selector (same matching rules as --baseline).",
        ),
    ],
    config: ConfigPath = Path("skulk-harness.yaml"),
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write the machine-readable ComparisonRecord JSON here.",
        ),
    ] = None,
) -> None:
    """Compare two run sets like-for-like and surface trust guards.

    Aggregates each side's results per model (median decode tok/s and TTFT over
    substantive samples, short outputs excluded but counted) and prints the
    candidate-vs-baseline deltas together with the guards that make a delta
    untrustworthy (node-set/cache mismatch, low sample count, short-output
    noise, issue-marked runs, missing fingerprint). This is the reproducible
    form of a "is the new branch actually faster?" investigation.
    """

    cfg = load_config(config)
    runs_root = cfg.output_dir
    baseline_dirs = select_run_dirs(runs_root, baseline)
    candidate_dirs = select_run_dirs(runs_root, candidate)
    if not baseline_dirs:
        console.print(f"[red]No runs matched baseline selector[/red]: {baseline!r}")
        raise typer.Exit(code=2)
    if not candidate_dirs:
        console.print(f"[red]No runs matched candidate selector[/red]: {candidate!r}")
        raise typer.Exit(code=2)

    record = compare(
        load_reports(baseline_dirs),
        load_reports(candidate_dirs),
        baseline_label=baseline,
        candidate_label=candidate,
    )
    _print_comparison(record)
    if out is not None:
        out.write_text(record.model_dump_json(indent=2, by_alias=True))
        console.print(f"\nWrote comparison record -> [cyan]{out}[/cyan]")


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


def _require_destructive_opt_in(execute_destructive: bool) -> None:
    if execute_destructive:
        return
    console.print(
        "[bold red]Refusing destructive stability command.[/] "
        "Pass --execute-destructive to allow SSH kill/relaunch operations."
    )
    raise typer.Exit(code=2)


@stability_app.command("failover")
def stability_failover(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    min_nodes: Annotated[int, typer.Option(help="Minimum nodes to place across.")] = 2,
    execute_destructive: Annotated[
        bool,
        typer.Option(
            "--execute-destructive",
            help="Allow this command to kill/relaunch Skulk over SSH.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Crash the master mid-stream and assert the cluster survives (#273)."""

    _require_destructive_opt_in(execute_destructive)
    cfg = load_config(config)
    _require_fleet_or_refuse(cfg, force=force)
    with _stability_client(cfg) as client:
        report = stability.run_failover(client, cfg, model, min_nodes=min_nodes)
    _write_stability(cfg, report)


@stability_app.command("churn")
def stability_churn(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    rounds: Annotated[int, typer.Option(help="Kill/relaunch rounds to run.")] = 3,
    execute_destructive: Annotated[
        bool,
        typer.Option(
            "--execute-destructive",
            help="Allow this command to kill/relaunch Skulk over SSH.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Repeatedly crash and relaunch a non-master node, asserting recovery."""

    _require_destructive_opt_in(execute_destructive)
    cfg = load_config(config)
    _require_fleet_or_refuse(cfg, force=force)
    with _stability_client(cfg) as client:
        report = stability.run_churn(client, cfg, model, rounds=rounds)
    _write_stability(cfg, report)


@stability_app.command("soak")
def stability_soak(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    concurrency: Annotated[int, typer.Option(help="Concurrent completion workers.")] = 4,
    duration_s: Annotated[float, typer.Option(help="Soak duration in seconds.")] = 120.0,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Drive sustained concurrent load and report latency/failures."""

    cfg = load_config(config)
    _require_fleet_or_refuse(cfg, force=force)
    with _stability_client(cfg) as client:
        report = stability.run_soak(
            client, cfg, model, concurrency=concurrency, duration_s=duration_s
        )
    _write_stability(cfg, report)


@stability_app.command("refusal")
def stability_refusal(
    model: ModelOption = DEFAULT_STABILITY_MODEL,
    config: ConfigPath = Path("skulk-harness.yaml"),
    execute_destructive: Annotated[
        bool,
        typer.Option(
            "--execute-destructive",
            help="Allow this command to run a destructive placement-refusal suite.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if another agent holds the shared-fleet lease.",
        ),
    ] = False,
) -> None:
    """Assert an impossible placement is refused or re-placed, not wedged (#290)."""

    _require_destructive_opt_in(execute_destructive)
    cfg = load_config(config)
    _require_fleet_or_refuse(cfg, force=force)
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
    # Coverage is the headline signal: "0 tests failed" is misleading when only
    # some models actually placed and ran. Count DISTINCT models that produced at
    # least one result against the models the set asked for.
    tested_models = len({result.model_id for result in report.results})
    total_models = len(report.models)
    coverage = f"{tested_models}/{total_models}"
    if tested_models < total_models:
        coverage = f"[yellow]{coverage}  (incomplete)[/yellow]"
    table = Table(title=f"Harness Run {report.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Models tested", coverage)
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


def _print_comparison(record: ComparisonRecord) -> None:
    """Render a ComparisonRecord: decode-tps deltas per model plus guards."""
    title = f"{record.candidate_label}  vs  {record.baseline_label}  (decode tok/s)"
    table = Table(title=title)
    table.add_column("Model")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Guards")

    for model in record.models:
        decode = next((d for d in model.deltas if d.metric == "decode_tps"), None)
        base = f"{decode.baseline:.1f}" if decode and decode.baseline is not None else "-"
        cand = (
            f"{decode.candidate:.1f}"
            if decode and decode.candidate is not None
            else "-"
        )
        if decode is not None and decode.percent_delta is not None:
            pct = decode.percent_delta
            # Higher decode tok/s is better; green for a real gain, red for loss.
            color = "green" if pct >= 0 else "red"
            delta = f"[{color}]{pct:+.1f}%[/{color}]"
        else:
            delta = "-"
        guards = ", ".join(g.replace("_", " ") for g in model.guards) or ""
        guard_text = f"[yellow]{guards}[/yellow]" if guards else ""
        table.add_row(model.model_id.split("/")[-1], base, cand, delta, guard_text)

    console.print(table)
    if record.guards:
        console.print(
            "\n[yellow]run-set guards[/yellow]: "
            + ", ".join(g.replace("_", " ") for g in record.guards)
        )
    console.print(
        f"[dim]baseline: {len(record.baseline_run_ids)} run(s) · "
        f"candidate: {len(record.candidate_run_ids)} run(s)[/dim]"
    )


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


@app.command()
def submit(
    run_path: Annotated[Path, typer.Argument(help="Run directory or report.json to submit.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the exact payload instead of sending.")
    ] = False,
    github_token: Annotated[
        str | None, typer.Option("--github-token", help="GitHub token for attribution.")
    ] = None,
    ingest_url: Annotated[
        str | None,
        typer.Option(
            "--ingest-url",
            help="Ingest API base URL (also read from SKULK_INGEST_URL; "
            "defaults to the public ledger).",
        ),
    ] = None,
) -> None:
    """Submit a run to the Foxlight community benchmarks ledger.

    Slims and redacts the report CLIENT-side (no generated text and no
    operator identifiers ever leave the machine), authenticates with your
    GitHub account for attribution, and queues the run for manual review.
    Use --dry-run to inspect the exact payload first. Never contacts a
    Skulk cluster.
    """

    # Resolved at call time so SKULK_INGEST_URL set after import still applies.
    resolved_ingest_url = ingest_url or submit_module.default_ingest_url()
    try:
        report_path = submit_module.locate_report(run_path)
        raw = json.loads(report_path.read_text())
        payload = submit_module.slim_and_redact_report(raw)
        if dry_run:
            typer.echo(json.dumps(payload, indent=2))
            typer.echo(
                f"[dry-run] would POST to {resolved_ingest_url}/v1/submissions",
                err=True,
            )
            return
        token = submit_module.resolve_github_token(github_token)
        result = submit_module.post_submission(payload, token, resolved_ingest_url)
        typer.echo(json.dumps(result, indent=2))
    except submit_module.SubmitError as exc:
        typer.echo(f"submit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
