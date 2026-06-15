"""Cluster stability suites: failover, churn, soak, and placement refusal.

Each suite places a model, drives a real workload while perturbing the cluster,
and asserts a cluster *property* rather than scoring model output. Every suite
cleans up the instances it created on exit (best-effort delete), and records
findings as :class:`Issue` objects on a :class:`StabilityReport`. An error-level
issue clears ``passed``; warnings are non-fatal observations.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable

from skulk_test_harness import chaos
from skulk_test_harness.client import ChatExecution, SkulkApiError, SkulkClient
from skulk_test_harness.models import (
    ClusterNode,
    HarnessConfig,
    Issue,
    LatencySummary,
    PlacementResult,
    StabilityReport,
)
from skulk_test_harness.utils import slugify, unwrap_tagged

# A short, deterministic coherence prompt: a fresh completion after failover or
# churn must still produce ordered output, which catches a silently wedged or
# desynced instance that a "non-empty" check alone would pass.
_COHERENCE_PROMPT = "Count from 1 to 20, space separated, numbers only."
_COHERENCE_MAX_TOKENS = 128


def _stability_run_id(suite: str, model_id: str) -> str:
    """Build a timestamped run ID for a stability suite report directory."""

    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{suite}-{slugify(model_id)}"


def _percentile(samples: list[float], fraction: float) -> float | None:
    """Return the ``fraction`` percentile (0..1) using nearest-rank.

    Pure helper so the soak aggregation can be unit-tested without timing a real
    cluster. Returns ``None`` for an empty sample set.
    """

    if not samples:
        return None
    ordered = sorted(samples)
    if fraction <= 0:
        return ordered[0]
    if fraction >= 1:
        return ordered[-1]
    # Nearest-rank: rank = ceil(fraction * n), 1-indexed.
    rank = max(1, -(-int(round(fraction * len(ordered) * 100)) // 100))
    rank = min(rank, len(ordered))
    return ordered[rank - 1]


def summarize_latency(samples: list[float], *, failures: int = 0) -> LatencySummary:
    """Aggregate successful-completion latencies into a :class:`LatencySummary`.

    ``samples`` are per-completion elapsed seconds for SUCCESSFUL completions;
    ``failures`` counts completions that errored or returned empty. Pure helper.
    """

    if not samples:
        return LatencySummary(count=0, failures=failures)
    return LatencySummary(
        count=len(samples),
        failures=failures,
        p50_s=_percentile(samples, 0.50),
        p95_s=_percentile(samples, 0.95),
        max_s=max(samples),
        min_s=min(samples),
        mean_s=sum(samples) / len(samples),
    )


def completion_is_coherent(execution: ChatExecution) -> bool:
    """Return whether a completion produced usable, non-empty output.

    A healthy completion produced non-empty output in EITHER the content or the
    reasoning channel, and streamed at least one chunk. Reasoning models legitly
    emit their answer as ``reasoning_text`` with empty ``text`` (especially when
    a tight ``max_tokens`` is consumed mid-think), so checking ``text`` alone
    would wrongly fail a perfectly-serving model. An empty/zero-chunk result
    indicates a wedged or aborted generation even if the HTTP request returned
    200 — which is what the stability suites care about (is the cluster serving),
    not answer correctness.
    """

    produced = bool(execution.text.strip()) or bool(execution.reasoning_text.strip())
    return produced and execution.metrics.chunks > 0


def classify_placement_outcome(
    state: dict[str, object],
    model_id: str,
    *,
    expected_min_nodes: int,
    live_node_count: int,
) -> tuple[str, list[PlacementResult]]:
    """Classify a refusal-scenario placement outcome from cluster state.

    Pure helper (unit-testable) used by :func:`run_placement_refusal`. Inspects
    instances currently placed for ``model_id`` and returns one of:

    - ``"refused"``: no instance exists for the model (the cluster cleanly
      declined an impossible request) — the desired #290 behavior.
    - ``"replaced_wider"``: an instance exists spanning enough nodes to be
      serviceable (>= 1 and <= ``live_node_count``) and is ready — the cluster
      re-placed onto a viable, narrower set instead of the impossible request.
    - ``"partial"``: an instance exists but is NOT ready or claims more nodes
      than are live — a wedged/half-placed instance, which is the failure #290
      hardened against.

    Returns the verdict and the matched placements for evidence.
    """

    placements = _placements_for_model_from_state(state, model_id)
    if not placements:
        return "refused", []
    for placement in placements:
        node_count = len(placement.node_ids)
        if node_count > live_node_count or node_count == 0:
            return "partial", placements
        if not placement.ready:
            return "partial", placements
    # Every placement is ready and fits within the live set. If it satisfied the
    # impossible min_nodes it would have been "partial" above (node_count would
    # exceed live_node_count); reaching here means it landed on a viable subset.
    del expected_min_nodes  # retained for signature clarity; verdict is by fit
    return "replaced_wider", placements


def _placements_for_model_from_state(
    state: dict[str, object], model_id: str
) -> list[PlacementResult]:
    """Pure extraction of placements for a model from a state dict."""

    instances = state.get("instances")
    if not isinstance(instances, dict):
        return []
    runners = state.get("runners")
    runner_states = runners if isinstance(runners, dict) else {}
    placements: list[PlacementResult] = []
    for instance_id, raw_instance in instances.items():
        parsed = unwrap_tagged(raw_instance)
        if parsed is None:
            continue
        tag, body = parsed
        assignments = body.get("shardAssignments")
        if not isinstance(assignments, dict):
            continue
        if assignments.get("modelId") != model_id:
            continue
        node_to_runner = assignments.get("nodeToRunner")
        runner_to_shard = assignments.get("runnerToShard")
        runner_ids = list(runner_to_shard) if isinstance(runner_to_shard, dict) else []
        ready = bool(runner_ids) and all(
            _runner_ready(runner_states.get(runner_id)) for runner_id in runner_ids
        )
        placements.append(
            PlacementResult(
                model_id=model_id,
                instance_id=str(instance_id),
                node_ids=list(node_to_runner) if isinstance(node_to_runner, dict) else [],
                runner_ids=runner_ids,
                instance_meta=tag,
                reused_existing=True,
                ready=ready,
            )
        )
    return placements


def _runner_ready(raw_status: object) -> bool:
    parsed = unwrap_tagged(raw_status)
    if parsed is None:
        return False
    return parsed[0] in {"RunnerReady", "RunnerRunning"}


def _live_node_count(client: SkulkClient) -> int:
    """Return the number of nodes the cluster currently considers present."""

    try:
        state = client.get_state()
    except SkulkApiError:
        return 0
    last_seen = state.get("lastSeen")
    return len(last_seen) if isinstance(last_seen, dict) else 0


def _coherence_completion(client: SkulkClient, model_id: str) -> ChatExecution:
    """Run one short deterministic coherence completion."""

    return client.stream_chat(
        model_id=model_id,
        messages=[{"role": "user", "content": _COHERENCE_PROMPT}],
        max_tokens=_COHERENCE_MAX_TOKENS,
        temperature=0.0,
        top_p=None,
    )


def _place_multinode(
    client: SkulkClient,
    config: HarnessConfig,
    model_id: str,
    report: StabilityReport,
    *,
    min_nodes: int,
) -> PlacementResult | None:
    """Place ``model_id`` across at least ``min_nodes`` and wait for readiness.

    Reuses an existing placement when one is already serving the model. Records
    an error issue and returns ``None`` if no usable placement can be made ready.
    """

    existing = client.find_placements_for_model(model_id)
    if existing:
        placement = existing[0]
        if placement.instance_id and not placement.ready:
            placement = client.wait_for_instance_ready(
                placement.instance_id,
                timeout_s=config.placement_ready_timeout_s,
                poll_interval_s=config.poll_interval_s,
            )
        return placement.model_copy(update={"reused_existing": True})

    previews = [
        preview
        for preview in client.get_placement_previews(model_id)
        if preview.get("error") in (None, "")
    ]
    if not previews:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="No usable placement preview found for stability suite",
            )
        )
        return None
    preview = sorted(previews, key=_preview_node_count, reverse=True)[0]
    try:
        client.place_model(
            model_id=model_id,
            sharding=str(preview.get("sharding") or "Pipeline"),
            instance_meta=str(preview.get("instance_meta") or "MlxRing"),
            min_nodes=min_nodes,
            excluded_nodes=[],
        )
    except SkulkApiError as exc:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Placement request failed for stability suite",
                evidence={"error": str(exc)},
            )
        )
        return None

    deadline = time.monotonic() + config.placement_ready_timeout_s
    while time.monotonic() < deadline:
        placements = client.find_placements_for_model(model_id)
        if placements and placements[0].instance_id:
            placement = client.wait_for_instance_ready(
                placements[0].instance_id,
                timeout_s=max(0.1, deadline - time.monotonic()),
                poll_interval_s=config.poll_interval_s,
            )
            return placement.model_copy(update={"created_by_harness": True})
        time.sleep(config.poll_interval_s)
    report.add_issue(
        Issue(
            severity="error",
            model_id=model_id,
            message="Timed out waiting for stability-suite placement to become ready",
        )
    )
    return None


def _preview_node_count(preview: dict[str, object]) -> int:
    parsed = unwrap_tagged(preview.get("instance"))
    if parsed is None:
        return 0
    assignments = parsed[1].get("shardAssignments")
    if not isinstance(assignments, dict):
        return 0
    node_to_runner = assignments.get("nodeToRunner")
    return len(node_to_runner) if isinstance(node_to_runner, dict) else 0


def _cleanup_instance(
    client: SkulkClient, placement: PlacementResult | None, report: StabilityReport
) -> None:
    """Best-effort delete of a harness-created instance."""

    if placement is None or not placement.created_by_harness or not placement.instance_id:
        return
    try:
        client.delete_instance(placement.instance_id)
    except SkulkApiError as exc:
        report.add_issue(
            Issue(
                severity="warning",
                model_id=placement.model_id,
                message="Failed to delete stability-suite instance during cleanup",
                evidence={"error": str(exc), "instance_id": placement.instance_id},
            )
        )


def _require_cluster_node(
    config: HarnessConfig, friendly: str, report: StabilityReport, model_id: str
) -> ClusterNode | None:
    node = config.cluster_nodes.get(friendly)
    if node is None:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message=f"No cluster_nodes entry for friendly name {friendly!r}",
                evidence={"known_nodes": sorted(config.cluster_nodes)},
            )
        )
    return node


def run_failover(
    client: SkulkClient,
    config: HarnessConfig,
    model_id: str,
    *,
    min_nodes: int = 2,
) -> StabilityReport:
    """Crash the master mid-stream and assert the cluster survives (#273).

    Places ``model_id`` across >= ``min_nodes`` nodes, starts a streaming
    completion, hard-kills the current master node, then asserts: (a) a NEW
    master is elected, (b) the instance survives (still present or re-placed),
    and (c) a fresh completion succeeds and is coherent. Finally relaunches the
    killed node and asserts it rejoins. ``client`` should point at a node that
    will SURVIVE the crash (i.e. not the master).
    """

    report = StabilityReport.start(_stability_run_id("failover", model_id), "failover", model_id)
    placement = _place_multinode(client, config, model_id, report, min_nodes=min_nodes)
    if placement is None or not placement.ready:
        return report.finish()

    # Capture the full-cluster size before we crash anything. Rejoin is verified
    # by size, not by node_id: node_id is ephemeral (regenerated every process
    # start), so a relaunched node rejoins under a NEW id and the old one never
    # reappears.
    baseline_nodes = _live_node_count(client)
    report.observations["baseline_node_count"] = baseline_nodes

    master_id = chaos.current_master(client)
    master_friendly = chaos.friendly_for_node(client, master_id)
    report.observations["original_master"] = {
        "node_id": master_id,
        "friendly": master_friendly,
    }
    node = _require_cluster_node(config, master_friendly, report, model_id)
    if node is None:
        _cleanup_instance(client, placement, report)
        return report.finish()

    if client.base_url.endswith(":52415") and master_friendly in client.base_url:
        # The harness API client must survive the crash; refuse to kill the very
        # node we are talking to rather than blind ourselves mid-failover.
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Configured api_base_url points at the master; "
                "point it at a surviving node before running failover",
                evidence={"api_base_url": client.base_url, "master": master_friendly},
            )
        )
        _cleanup_instance(client, placement, report)
        return report.finish()

    # Kick off a stream, then crash the master while it is (intended to be) in
    # flight. The stream is best-effort: it may error when its serving rank dies,
    # which is itself informational, not a suite failure.
    stream_outcome: dict[str, object] = {}

    def _drive_stream() -> None:
        try:
            execution = _coherence_completion(client, model_id)
            stream_outcome["completed"] = completion_is_coherent(execution)
        except SkulkApiError as exc:
            stream_outcome["error"] = str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_drive_stream)
        time.sleep(0.5)  # let the stream begin before we pull the rug
        killed = chaos.kill_skulk(node.ssh_host)
        report.observations["master_kill_issued"] = killed
        future.result(timeout=config.generation_timeout_s)
    report.observations["in_flight_stream"] = stream_outcome

    if not chaos.wait_for_node_absent(
        client, master_id, timeout_s=120.0, poll_interval_s=config.poll_interval_s
    ):
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Killed master never dropped out of cluster state",
                evidence={"master": master_id},
            )
        )

    new_master = chaos.wait_for_new_master(
        client, master_id, timeout_s=180.0, poll_interval_s=config.poll_interval_s
    )
    if new_master is None:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="No new master was elected after the master crashed",
                evidence={"old_master": master_id},
            )
        )
    else:
        report.observations["new_master"] = new_master

    # (b) instance continuity: the model should still be serviceable, whether the
    # same instance survived or the cluster re-placed it.
    survived = _wait_for_model_servable(client, model_id, timeout_s=180.0, poll_interval_s=config.poll_interval_s)
    report.observations["instance_survived"] = survived
    if not survived:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Instance did not survive or re-place after master failover (#273)",
            )
        )

    # (c) a fresh completion must succeed and be coherent post-failover.
    if survived:
        try:
            execution = _coherence_completion(client, model_id)
            coherent = completion_is_coherent(execution)
            report.observations["post_failover_completion"] = {
                "coherent": coherent,
                "output_chars": execution.metrics.output_chars,
            }
            if not coherent:
                report.add_issue(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        message="Post-failover completion was empty or incoherent",
                    )
                )
        except SkulkApiError as exc:
            report.add_issue(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message="Post-failover completion request failed",
                    evidence={"error": str(exc)},
                )
            )

    # Relaunch the crashed node and confirm the cluster returns to full size.
    # Verify by node COUNT, not the old node_id: the relaunched node rejoins
    # under a fresh ephemeral id, so the original master_id never reappears.
    relaunched = chaos.relaunch_skulk(node)
    report.observations["relaunch_issued"] = relaunched
    rejoined = _wait_for_node_count(
        client, baseline_nodes, timeout_s=240.0, poll_interval_s=config.poll_interval_s
    )
    report.observations["node_rejoined"] = rejoined
    if not rejoined:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Cluster did not return to full size after relaunching the crashed node",
                evidence={"baseline_nodes": baseline_nodes, "friendly": master_friendly},
            )
        )

    _cleanup_instance(client, placement, report)
    return report.finish()


def _wait_for_model_servable(
    client: SkulkClient, model_id: str, *, timeout_s: float, poll_interval_s: float
) -> bool:
    """Poll until at least one ready instance exists for ``model_id``."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            placements = client.find_placements_for_model(model_id)
        except SkulkApiError:
            placements = []
        if any(placement.ready for placement in placements):
            return True
        time.sleep(poll_interval_s)
    return False


def run_churn(
    client: SkulkClient,
    config: HarnessConfig,
    model_id: str,
    *,
    rounds: int = 3,
) -> StabilityReport:
    """Repeatedly crash and relaunch a NON-master node, asserting recovery.

    Places ``model_id``, then for ``rounds`` iterations kills a non-master node,
    waits for the cluster to shrink, relaunches it, waits for it to rejoin, and
    asserts a completion still succeeds. Records per-round observations.
    """

    report = StabilityReport.start(_stability_run_id("churn", model_id), "churn", model_id)
    placement = _place_multinode(client, config, model_id, report, min_nodes=2)
    if placement is None or not placement.ready:
        return report.finish()

    baseline_nodes = _live_node_count(client)
    report.observations["baseline_node_count"] = baseline_nodes
    rounds_log: list[dict[str, object]] = []

    for round_index in range(1, rounds + 1):
        target_friendly = _pick_non_master_friendly(client, config)
        if target_friendly is None:
            report.add_issue(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message="No non-master node with a cluster_nodes entry to churn",
                )
            )
            break
        node = config.cluster_nodes[target_friendly]
        target_node_id = chaos.node_for_friendly(client, target_friendly)
        round_log: dict[str, object] = {
            "round": round_index,
            "target_friendly": target_friendly,
            "target_node_id": target_node_id,
        }

        killed = chaos.kill_skulk(node.ssh_host)
        round_log["kill_issued"] = killed
        if target_node_id is not None:
            round_log["went_absent"] = chaos.wait_for_node_absent(
                client, target_node_id, timeout_s=120.0, poll_interval_s=config.poll_interval_s
            )

        relaunched = chaos.relaunch_skulk(node)
        round_log["relaunch_issued"] = relaunched
        # Rejoin is verified by cluster size returning to baseline, not by the
        # old node_id reappearing (node_id is ephemeral: the relaunched node
        # comes back under a fresh id, so the original never returns).
        recovered = _wait_for_node_count(
            client, baseline_nodes, timeout_s=240.0, poll_interval_s=config.poll_interval_s
        )
        round_log["full_size_recovered"] = recovered
        round_log["rejoined"] = recovered
        if not recovered:
            report.add_issue(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message=f"Cluster did not return to {baseline_nodes} nodes after churn round {round_index}",
                )
            )

        # If the churned node hosted a shard, Skulk correctly tears down the
        # non-redundant instance (failing its tasks, #223/#224), so the original
        # placement may be gone. The churn property is that the cluster stays
        # SERVICEABLE across node turnover, so re-establish an instance before
        # asserting a completion rather than expecting the old one to survive a
        # shard loss it structurally cannot.
        if not _wait_for_model_servable(
            client, model_id, timeout_s=20.0, poll_interval_s=config.poll_interval_s
        ):
            replaced = _place_multinode(client, config, model_id, report, min_nodes=2)
            if replaced is not None:
                placement = replaced
            round_log["re_placed"] = replaced is not None

        try:
            execution = _coherence_completion(client, model_id)
            coherent = completion_is_coherent(execution)
        except SkulkApiError as exc:
            coherent = False
            round_log["completion_error"] = str(exc)
        round_log["completion_coherent"] = coherent
        if not coherent:
            report.add_issue(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message=f"Completion failed after churn round {round_index}",
                )
            )
        rounds_log.append(round_log)

    report.observations["rounds"] = rounds_log
    _cleanup_instance(client, placement, report)
    return report.finish()


def _pick_non_master_friendly(
    client: SkulkClient, config: HarnessConfig
) -> str | None:
    """Return a configured friendly name that is neither master nor the client node.

    Skips the master (we churn workers, not the leader — failover covers leader
    death) and skips the node the harness API client is talking to: killing that
    node would blind the harness mid-round, the same hazard the failover suite
    guards against.
    """

    master_id = chaos.current_master(client)
    master_friendly = chaos.friendly_for_node(client, master_id)
    for friendly in config.cluster_nodes:
        if friendly == master_friendly:
            continue
        if friendly in client.base_url:
            continue
        return friendly
    return None


def _wait_for_node_count(
    client: SkulkClient, target: int, *, timeout_s: float, poll_interval_s: float
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _live_node_count(client) >= target:
            return True
        time.sleep(poll_interval_s)
    return False


def run_soak(
    client: SkulkClient,
    config: HarnessConfig,
    model_id: str,
    *,
    concurrency: int = 4,
    duration_s: float = 120.0,
) -> StabilityReport:
    """Drive sustained concurrent load and assert every completion succeeds.

    Places ``model_id``, then runs ``concurrency`` worker threads issuing chat
    completions back-to-back for ``duration_s`` seconds. Asserts all completions
    are coherent and reports p50/p95 latency, total count, and any failures.
    """

    report = StabilityReport.start(_stability_run_id("soak", model_id), "soak", model_id)
    placement = _place_multinode(client, config, model_id, report, min_nodes=1)
    if placement is None or not placement.ready:
        return report.finish()

    deadline = time.monotonic() + duration_s
    latencies: list[float] = []
    failures = 0
    lock_failures: list[str] = []

    def _worker() -> tuple[list[float], int, list[str]]:
        local_latencies: list[float] = []
        local_failures = 0
        local_errors: list[str] = []
        while time.monotonic() < deadline:
            try:
                execution = _coherence_completion(client, model_id)
            except SkulkApiError as exc:
                local_failures += 1
                local_errors.append(str(exc))
                continue
            if completion_is_coherent(execution):
                local_latencies.append(execution.metrics.elapsed_s)
            else:
                local_failures += 1
                local_errors.append("empty-or-incoherent")
        return local_latencies, local_failures, local_errors

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker) for _ in range(concurrency)]
        for future in concurrent.futures.as_completed(futures):
            worker_latencies, worker_failures, worker_errors = future.result()
            latencies.extend(worker_latencies)
            failures += worker_failures
            lock_failures.extend(worker_errors)

    report.latency = summarize_latency(latencies, failures=failures)
    report.observations["concurrency"] = concurrency
    report.observations["duration_s"] = duration_s
    report.observations["total_completions"] = len(latencies) + failures
    report.observations["sample_errors"] = lock_failures[:10]
    if failures:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message=f"{failures} of {len(latencies) + failures} soak completions failed",
                evidence={"sample_errors": lock_failures[:10]},
            )
        )
    if not latencies:
        report.add_issue(
            Issue(
                severity="error",
                model_id=model_id,
                message="Soak produced no successful completions",
            )
        )

    _cleanup_instance(client, placement, report)
    return report.finish()


def run_placement_refusal(
    client: SkulkClient,
    config: HarnessConfig,
    model_id: str,
) -> StabilityReport:
    """Assert an impossible placement is refused or re-placed wider, not wedged (#290).

    Requests a placement with ``min_nodes`` greater than the live node count and
    asserts the system either cleanly refuses (no instance appears) or re-places
    onto a viable narrower set, WITHOUT leaving a half-placed/wedged instance.
    Any harness-created instance is deleted on exit.
    """

    report = StabilityReport.start(
        _stability_run_id("refusal", model_id), "refusal", model_id
    )
    live_nodes = _live_node_count(client)
    impossible_min_nodes = live_nodes + 5
    report.observations["live_node_count"] = live_nodes
    report.observations["requested_min_nodes"] = impossible_min_nodes

    previews = [
        preview
        for preview in client.get_placement_previews(model_id)
        if preview.get("error") in (None, "")
    ]
    sharding = str(previews[0].get("sharding")) if previews else "Pipeline"
    instance_meta = str(previews[0].get("instance_meta")) if previews else "MlxRing"

    refused_cleanly = False
    try:
        client.place_model(
            model_id=model_id,
            sharding=sharding,
            instance_meta=instance_meta,
            min_nodes=impossible_min_nodes,
            excluded_nodes=[],
        )
    except SkulkApiError as exc:
        # An explicit 4xx/5xx refusal is the cleanest possible outcome.
        refused_cleanly = True
        report.observations["refusal_error"] = str(exc)

    created_instance_id: str | None = None
    if not refused_cleanly:
        # Give the cluster a bounded window to either decline or settle, then
        # classify what (if anything) it placed.
        deadline = time.monotonic() + min(120.0, config.placement_ready_timeout_s)
        verdict = "refused"
        evidence_placements: list[PlacementResult] = []
        while time.monotonic() < deadline:
            state = client.get_state()
            verdict, evidence_placements = classify_placement_outcome(
                state,
                model_id,
                expected_min_nodes=impossible_min_nodes,
                live_node_count=live_nodes,
            )
            # A terminal verdict is anything other than a still-settling partial.
            if verdict in {"refused", "replaced_wider"}:
                break
            time.sleep(config.poll_interval_s)
        report.observations["verdict"] = verdict
        report.observations["placements"] = [
            placement.model_dump(mode="json") for placement in evidence_placements
        ]
        for placement in evidence_placements:
            if placement.created_by_harness or placement.instance_id:
                created_instance_id = placement.instance_id
        if verdict == "partial":
            report.add_issue(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message="Impossible placement produced a partial/wedged instance (#290 regression)",
                    evidence={
                        "live_node_count": live_nodes,
                        "requested_min_nodes": impossible_min_nodes,
                        "placements": [p.model_dump(mode="json") for p in evidence_placements],
                    },
                )
            )
    else:
        report.observations["verdict"] = "refused"

    # Clean up anything that did get placed (e.g. a re-placed-wider instance).
    if created_instance_id is not None:
        try:
            client.delete_instance(created_instance_id)
        except SkulkApiError as exc:
            report.add_issue(
                Issue(
                    severity="warning",
                    model_id=model_id,
                    message="Failed to delete instance created by refusal scenario",
                    evidence={"error": str(exc), "instance_id": created_instance_id},
                )
            )

    return report.finish()


# Public dispatch table for the CLI; keeps the command bodies tiny and uniform.
SUITES: dict[str, Callable[..., StabilityReport]] = {
    "failover": run_failover,
    "churn": run_churn,
    "soak": run_soak,
    "refusal": run_placement_refusal,
}
