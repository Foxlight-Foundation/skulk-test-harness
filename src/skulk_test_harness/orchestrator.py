"""Main planning and execution engine for the Skulk harness."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
import mimetypes
import re
import statistics
import time
import wave
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import httpx

from skulk_test_harness.client import (
    AudioSpeechExecution,
    ChatExecution,
    ClusterApiOwner,
    DataPlaneDiagnosticsSnapshot,
    ProviderCapabilityDiagnosticsSnapshot,
    RealtimeTranscriptionExecution,
    SkulkApiError,
    SkulkClient,
    StreamingAudioTranscriptionExecution,
    VisionMediaDiagnosticsSnapshot,
    concurrent_benchmark_client,
    stream_chat_async,
)
from skulk_test_harness.fingerprint import gather_fingerprint
from skulk_test_harness.models import (
    ExpectedToolCall,
    GenerationMetrics,
    HarnessConfig,
    Issue,
    ModelRef,
    ModelSelector,
    ModelSet,
    OwnerTopology,
    PlacementPolicy,
    PlacementResult,
    PromptImage,
    PromptTest,
    RunReport,
    RunSpec,
    SuccessCriteria,
    TestResult,
    TestSet,
    ToolCallRecord,
    ToolMock,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.utils import (
    extract_first_code_block,
    maybe_write_artifact,
    slugify,
    unwrap_tagged,
)

# One concurrent-benchmark request record: (execution or None, error text or
# None, started-at monotonic seconds, ended-at monotonic seconds).
_ConcurrentRecord = tuple[ChatExecution | None, str | None, float, float]


class _SpeechGenerationKwargs(TypedDict):
    """Explicit model-generation controls forwarded to speech synthesis."""

    temperature: float | None
    top_p: float | None
    max_tokens: int | None


class HarnessRunner:
    """Coordinates model resolution, placement, execution, and reporting."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        model_sets: dict[str, ModelSet],
        test_sets: dict[str, TestSet],
    ) -> None:
        self.config = config
        self.model_sets = model_sets
        self.test_sets = test_sets

    def plan(self, spec: RunSpec) -> RunReport:
        """Build a report describing what would run."""

        with self._client() as client:
            models = self.resolve_model_set(spec.model_set, client)
            report = RunReport.start(_run_id(spec), spec, models)
            # Stamp the suite description onto plan / dry-run reports too, so a
            # non-executed report is self-describing like an executed one.
            # Best-effort via the map (plan does not run tests, so an unknown
            # test set must not raise here as it would in execute()).
            planned_test_set = self.test_sets.get(spec.test_set)
            if planned_test_set is not None:
                report.test_set_description = planned_test_set.description
            report.issues.extend(client.detect_runner_state_drift())
            for model in models:
                existing = client.find_placements_for_model(model.model_id)
                if existing and spec.reuse_existing_instances:
                    report.placements.append(existing[0])
                    continue
                preview = self.choose_preview(client, model.model_id, spec.placement)
                if preview is None:
                    report.issues.append(
                        Issue(
                            severity="error",
                            model_id=model.model_id,
                            message="No usable placement preview found",
                        )
                    )
                else:
                    report.placements.append(
                        _placement_from_preview(model.model_id, preview)
                    )
            fingerprint, fp_issues = gather_fingerprint(
                client, spec, run_reason=spec.mode
            )
            report.issues.extend(fp_issues)
            report.fingerprint = fingerprint
            return report.finish()

    def execute(self, spec: RunSpec) -> RunReport:
        """Execute a full harness run."""

        with self._client() as client:
            models = self.resolve_model_set(spec.model_set, client)
            report = RunReport.start(_run_id(spec), spec, models)
            report.issues.extend(client.detect_runner_state_drift())
            writer = ReportWriter(self.config.output_dir)
            test_set = self._test_set(spec.test_set)
            # Stamp the suite's own description onto the report so the artifact
            # is self-describing (the results ledger explains what a suite
            # measures from the report, not a name-keyed lookup).
            report.test_set_description = test_set.description
            # Resolve each model's thinking-toggle support once so per-test
            # requests can default thinking OFF (dashboard parity) when a test
            # leaves it unspecified. Best-effort: a catalog hiccup leaves the
            # map empty and tests fall back to omitting the toggle.
            try:
                thinking_toggles = client.resolved_thinking_toggle_by_model()
            except Exception:  # noqa: BLE001 - non-fatal capability lookup
                thinking_toggles = {}
            deferred: list[ModelRef] = []
            for model in models:
                if not self._run_model_lifecycle(
                    client, model, spec, report, test_set, writer, thinking_toggles
                ):
                    deferred.append(model)
            # Deferred-retry pass: a model that couldn't place earlier (no viable
            # preview, typically transient memory pressure from a concurrently
            # staging/serving peer) gets one more attempt now that every other
            # model in the cell has been torn down -- a maximally-free cluster
            # where choose_preview can settle onto a larger node set that did not
            # fit under contention. Honors "retry refused placements later with a
            # bigger node set."
            for model in deferred:
                placed_after_retry = self._run_model_lifecycle(
                    client,
                    model,
                    spec,
                    report,
                    test_set,
                    writer,
                    thinking_toggles,
                    deferred_retry=True,
                )
                if placed_after_retry:
                    _clear_deferred_placement_issues(report, model.model_id)
            fingerprint, fp_issues = gather_fingerprint(
                client, spec, run_reason=spec.mode
            )
            report.issues.extend(fp_issues)
            report.fingerprint = fingerprint
            finished = report.finish()
            writer.write(finished)
            return finished

    def _run_model_lifecycle(
        self,
        client: SkulkClient,
        model: ModelRef,
        spec: RunSpec,
        report: RunReport,
        test_set: TestSet,
        writer: ReportWriter,
        thinking_toggles: Mapping[str, bool],
        *,
        deferred_retry: bool = False,
    ) -> bool:
        """Place a model, run its tests, and ALWAYS tear it down.

        Each model runs in its own try/finally so (a) one model's failure is
        isolated and recorded, never aborting the rest of the cell, and (b) any
        harness-created instance is always torn down, even on a crash mid-test --
        a leaked instance was what poisoned later cells with resource contention
        in the prior battery.

        Returns ``True`` if the model obtained a placement (or failed after
        placement); ``False`` only when placement was *refused* (no viable
        preview), so the caller may defer and retry it once on a freer cluster.
        On the deferred retry a still-refused model is recorded as a final
        failure rather than deferred again.
        """
        placement = None
        try:
            placement = self._ensure_model_placed(client, model.model_id, spec, report)
            if placement is None:
                if deferred_retry:
                    report.issues.append(
                        Issue(
                            severity="error",
                            model_id=model.model_id,
                            message=(
                                "Placement still refused after deferred retry on a "
                                "freed cluster (model likely too large for the fleet)"
                            ),
                        )
                    )
                    writer.write(report)
                return False
            if not placement.ready:
                # NEVER silent: an instance that was placed but did not become
                # ready is reported with a precise cause resolved by the
                # model-scoped readiness wait, plus the full history of observed
                # placement changes so a silent wait is diagnosable from the
                # report alone (no master-log archaeology). A retryable give-up
                # (nothing serving the model -- never appeared or torn down
                # without re-placement) defers with a visible warning; a hard
                # load failure or readiness timeout on a known instance is an
                # error.
                retryable = _is_retryable_placement_giveup(placement)
                report.issues.append(
                    Issue(
                        severity=(
                            "warning" if retryable and not deferred_retry else "error"
                        ),
                        model_id=model.model_id,
                        message=_not_ready_message(placement),
                        evidence={
                            "instance_id": placement.instance_id or "",
                            "unavailable_reason": placement.unavailable_reason or "",
                            "runner_failures": placement.runner_failure_messages,
                            "readiness_transitions": placement.readiness_transitions,
                        },
                    )
                )
                writer.write(report)
                return not retryable
            report.placements.append(placement)
            # Dashboard parity: when the model exposes a thinking toggle and the
            # test does not pin enable_thinking, default it OFF so the model
            # answers instead of emitting an all-reasoning, length-capped reply.
            thinking_default = False if thinking_toggles.get(model.model_id) else None
            for test in test_set.tests:
                for repetition in range(1, test.repetitions + 1):
                    result = self._run_test(
                        client,
                        model_id=model.model_id,
                        test=test,
                        repetition=repetition,
                        artifact_dir=writer.run_dir(report.run_id) / "artifacts",
                        thinking_default=thinking_default,
                        spec=spec,
                        report=report,
                        writer=writer,
                    )
                    # Copy the test's kind + description onto the result at the
                    # single append point rather than threading them through
                    # every per-kind handler that builds a TestResult.
                    report.results.append(
                        result.model_copy(
                            update={
                                "kind": test.kind,
                                "description": test.description,
                            }
                        )
                    )
                    writer.write(report)
            return True
        except Exception as exc:  # noqa: BLE001 - isolate per-model failure
            report.issues.append(
                Issue(
                    severity="error",
                    model_id=model.model_id,
                    message="Model run failed; continuing to next model",
                    evidence={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            writer.write(report)
            return True
        finally:
            instance_torn_down = False
            if (
                placement is not None
                and placement.created_by_harness
                and not spec.retain_instances
            ):
                instance_torn_down = self._teardown_harness_instances(
                    client,
                    model.model_id,
                    placement.instance_id,
                    report,
                    protected_instance_ids=frozenset(placement.protected_instance_ids),
                )
            # Evict staged weights ONLY after the harness actually tore down the
            # instance it created (opt-in via --delete-staged-models), so test
            # models do not accumulate on disk. Never evict out from under a
            # retained instance (--delete-staged-models without
            # --delete-created-instances) or a reused, user-owned placement --
            # that would pull weights from a live model.
            if spec.delete_staged_models and instance_torn_down:
                self._evict_staged_model(client, model.model_id, report)

    def _evict_staged_model(
        self, client: SkulkClient, model_id: str, report: RunReport
    ) -> None:
        """Best-effort: remove a model's staged weights from the store after a run.

        A 404 is benign (already absent). Other failures are recorded as warnings
        and never abort the run; the next cell still proceeds.
        """
        try:
            client.delete_store_model(
                model_id, timeout_s=self.config.store_delete_timeout_s
            )
        except SkulkApiError as exc:
            if exc.status_code != 404:
                report.issues.append(
                    Issue(
                        severity="warning",
                        model_id=model_id,
                        message="Failed to evict staged model from store",
                        evidence={"error": str(exc)},
                    )
                )
        except Exception as exc:  # noqa: BLE001 - eviction is best-effort
            report.issues.append(
                Issue(
                    severity="warning",
                    model_id=model_id,
                    message="Failed to evict staged model from store",
                    evidence={"error": str(exc)},
                )
            )

    def _teardown_harness_instances(
        self,
        client: SkulkClient,
        model_id: str,
        primary_instance_id: str | None,
        report: RunReport,
        protected_instance_ids: frozenset[str] = frozenset(),
    ) -> bool:
        """Delete every live instance the harness owns for ``model_id``.

        Teardown deletes by instance_id, but the cluster can re-place an
        instance under a *new* id mid-run (failover / re-placement carry-over).
        When that happens, deleting the id we were handed at creation 404s while
        the re-IDed instance is orphaned -- it then starves the next cell and
        reads as "the harness left the old instance running". So we delete the
        original id AND sweep the current state for any instance still serving
        this model. This branch only runs when the harness *created* the
        lineage (``created_by_harness``), so every live instance for the model is
        ours to reap -- EXCEPT ``protected_instance_ids``: instances that already
        existed for the model before a forced placement are operator-owned and
        excluded from both the primary target and the sweep, so a harness cell
        can never delete a model the operator was already running.
        """
        target_ids: list[str] = []
        if primary_instance_id and primary_instance_id not in protected_instance_ids:
            target_ids.append(primary_instance_id)
        all_deletes_succeeded = bool(target_ids)
        try:
            for live in client.find_placements_for_model(model_id):
                if (
                    live.instance_id
                    and live.instance_id not in target_ids
                    and live.instance_id not in protected_instance_ids
                ):
                    target_ids.append(live.instance_id)
                    all_deletes_succeeded = True
        except Exception as exc:  # noqa: BLE001 - sweep is best-effort
            all_deletes_succeeded = False
            report.issues.append(
                Issue(
                    severity="warning",
                    model_id=model_id,
                    message="Failed to enumerate instances for teardown sweep",
                    evidence={"error": str(exc)},
                )
            )
        for instance_id in target_ids:
            try:
                client.delete_instance(instance_id)
            except SkulkApiError as exc:
                # A 404 means this id was already superseded/removed -- benign
                # for the original id; only surface non-404 failures.
                if exc.status_code != 404:
                    all_deletes_succeeded = False
                    report.issues.append(
                        Issue(
                            severity="warning",
                            model_id=model_id,
                            message="Failed to delete harness-created instance",
                            evidence={
                                "error": str(exc),
                                "instance_id": instance_id,
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - teardown best-effort
                all_deletes_succeeded = False
                report.issues.append(
                    Issue(
                        severity="warning",
                        model_id=model_id,
                        message="Failed to delete harness-created instance",
                        evidence={"error": str(exc), "instance_id": instance_id},
                    )
                )
        return bool(target_ids) and all_deletes_succeeded

    def resolve_model_set(self, name: str, client: SkulkClient) -> list[ModelRef]:
        """Resolve explicit IDs and catalog selectors for one named model set."""

        model_set = self._model_set(name)
        catalog = client.list_models()
        # Fetched lazily: only selectors that target the store need it, and a
        # store-less node (a valid deployment) must not fail explicit-list or
        # catalog-only resolution over an endpoint it never needed.
        store_entries: list[dict[str, object]] | None = None
        refs: list[ModelRef] = []
        seen: set[str] = set()

        def add(model_id: str, source: str, detail: str = "") -> None:
            if model_id in seen:
                return
            seen.add(model_id)
            refs.append(
                ModelRef(
                    model_id=model_id,
                    source=source,  # type: ignore[arg-type]
                    detail=detail,
                )
            )

        for model_id in model_set.models:
            add(model_id, "explicit")

        for selector in model_set.selectors:
            candidate_sources: list[dict[str, object]] = []
            if selector.source in {"catalog", "both"}:
                candidate_sources.extend(catalog)
            if selector.source in {"store", "both"}:
                if store_entries is None:
                    store_entries = _store_registry_entries(client.get_store_registry())
                candidate_sources.extend(store_entries)
            for model in _select_catalog_models(candidate_sources, selector):
                model_id = _model_id_from_catalog_entry(model)
                if model_id:
                    add(model_id, "selector", detail=selector.model_dump_json())

        for seed in model_set.huggingface_seeds:
            if not seed.require_mlx_community or seed.model_id.startswith(
                "mlx-community/"
            ):
                add(seed.model_id, "huggingface_seed", detail=seed.reason)
        return refs

    def choose_preview(
        self, client: SkulkClient, model_id: str, placement: PlacementPolicy
    ) -> dict[str, object] | None:
        """Select the best placement preview for a model and policy.

        Retries the previews request while none are viable: right after a
        teardown the freed memory of the prior model lags in gossiped telemetry,
        so a single shot can transiently see no fit (the tear-down-then-place
        matrix loop). The cluster clears it within a few seconds; a real no-fit
        keeps returning empty and we give up after ``preview_settle_attempts``.
        """
        attempts = max(1, self.config.preview_settle_attempts)
        chosen: dict[str, object] | None = None
        for attempt in range(attempts):
            chosen = self._choose_preview_once(client, model_id, placement)
            if chosen is not None or attempt == attempts - 1:
                break
            time.sleep(self.config.poll_interval_s)
        return chosen

    def _choose_preview_once(
        self, client: SkulkClient, model_id: str, placement: PlacementPolicy
    ) -> dict[str, object] | None:
        """Single pass: pick the best viable preview for a model and policy."""

        previews = [
            preview
            for preview in client.get_placement_previews(
                model_id, excluded_node_ids=placement.excluded_nodes
            )
            if preview.get("error") in (None, "")
        ]
        if placement.strategy == "exact":
            previews = [
                p
                for p in previews
                if p.get("sharding") == placement.sharding
                and p.get("instance_meta") == placement.instance_meta
                and (
                    placement.min_nodes is None
                    or _preview_node_count(p) >= placement.min_nodes
                )
            ]
        elif placement.strategy == "single":
            previews = [p for p in previews if _preview_node_count(p) == 1]
        else:
            previews = [
                p
                for p in previews
                if p.get("sharding") == placement.sharding
                and p.get("instance_meta") == placement.instance_meta
            ] or previews

        if not previews:
            return None
        return sorted(
            previews, key=lambda p: (_preview_node_count(p), str(p.get("sharding")))
        )[0]

    def _ensure_model_placed(
        self,
        client: SkulkClient,
        model_id: str,
        spec: RunSpec,
        report: RunReport,
    ) -> PlacementResult | None:
        if spec.ensure_model_cards:
            self._ensure_model_card(client, model_id, report)
        if spec.ensure_store_downloads:
            self._ensure_store_download(client, model_id, report)

        existing = client.find_placements_for_model(model_id)
        # Instances that already exist for this model BEFORE we place. On a forced
        # placement (reuse_existing_instances=False) these are user-owned and must
        # never be adopted as the harness's own -- otherwise the harness would run
        # its "fresh placement" test against a pre-existing instance and, with
        # retain_instances=False, tear down a model the operator was running.
        preexisting_instance_ids = frozenset(
            p.instance_id for p in existing if p.instance_id
        )
        # An excluded node must not be reused: without this, a cell that asks to
        # exclude kite4 (to force a GGUF onto kite5) would silently reuse a prior
        # kite4 placement and never exercise the target node.
        if spec.placement.excluded_nodes:
            excluded = set(spec.placement.excluded_nodes)
            existing = [p for p in existing if not excluded.intersection(p.node_ids)]
        if existing and spec.reuse_existing_instances:
            # Resolve readiness by MODEL, not by a pinned instance id. If the
            # cluster tears the reused instance down and re-places the model
            # under a new id, follow the replacement instead of polling the id we
            # first saw (which would just time out on a vanished instance).
            return self._wait_for_model_ready(
                client,
                model_id,
                spec,
                report,
                created_by_harness=False,
                reused=True,
            )

        preview = self.choose_preview(client, model_id, spec.placement)
        if preview is None:
            report.issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message="No usable placement preview found before execution",
                )
            )
            return None

        min_nodes = spec.placement.min_nodes or max(1, _preview_node_count(preview))
        try:
            client.place_model(
                model_id=model_id,
                sharding=str(preview.get("sharding") or spec.placement.sharding),
                instance_meta=str(
                    preview.get("instance_meta") or spec.placement.instance_meta
                ),
                min_nodes=min_nodes,
                excluded_nodes=spec.placement.excluded_nodes,
            )
        except SkulkApiError as exc:
            report.issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    message="Placement request failed",
                    evidence={"error": str(exc)},
                )
            )
            return None

        return self._wait_for_model_ready(
            client,
            model_id,
            spec,
            report,
            created_by_harness=True,
            reused=False,
            ignore_instance_ids=preexisting_instance_ids,
        )

    def _wait_for_model_ready(
        self,
        client: SkulkClient,
        model_id: str,
        spec: RunSpec,
        report: RunReport,
        *,
        created_by_harness: bool,
        reused: bool,
        ignore_instance_ids: frozenset[str] = frozenset(),
    ) -> PlacementResult:
        """Wait for the model to have a dispatchable instance, following re-placement.

        Readiness is resolved by MODEL, re-reading which instances currently
        serve ``model_id`` on every poll, rather than pinning the first instance
        id we saw. This is the fix for the failure that motivated it: a placement
        that hit a node fault (e.g. a full staging disk) is torn down by Skulk and
        re-placed under a NEW id; pinning the old id polled a vanished instance
        for the full readiness timeout and reported a false "never became ready"
        while the replacement was up and serving.

        Behavior:
          * return as soon as ANY instance serving the model is ready, preferring
            a ready instance over stale/loading duplicates;
          * while only loading instances exist, keep waiting (bounded by
            ``placement_ready_timeout_s``) -- a genuine slow load;
          * if no viable instance exists (none placed, or only failed ones), allow
            ``placement_appearance_timeout_s`` for the cluster to (re-)place, then
            give up -- never poll a vanished instance to the full deadline;
          * bound the TOTAL wait across every re-anchored replacement with a hard
            wall-clock ceiling (``placement_ready_total_timeout_s``, defaulting to
            two readiness allowances plus one appearance window): each replacement
            gets a fresh per-lineage allowance, but placement churn cannot extend
            the wait without bound. Hitting the ceiling fails loudly with
            ``unavailable_reason: churn``;
          * record every observed placement change so a silent wait is
            diagnosable from the report alone.

        ``ignore_instance_ids`` are instances that existed BEFORE a forced
        placement; they are user-owned and never adopted, so the harness cannot
        run its test against -- and then tear down -- a pre-existing instance it
        did not create. They are invisible to this wait; only the harness's own
        placement (or the cluster's re-placement of it) can satisfy readiness.

        The returned ``PlacementResult`` carries ``unavailable_reason`` and
        ``readiness_transitions`` when not ready; the caller owns issue reporting.
        """
        excluded = set(spec.placement.excluded_nodes or ())

        def current_placements() -> list[PlacementResult]:
            placements = client.find_placements_for_model(model_id)
            if excluded:
                placements = [
                    p for p in placements if not excluded.intersection(p.node_ids)
                ]
            if ignore_instance_ids:
                placements = [
                    p for p in placements if p.instance_id not in ignore_instance_ids
                ]
            return placements

        start = time.monotonic()
        appearance_deadline = start + self.config.placement_appearance_timeout_s
        # The readiness clock starts only when the first viable (loading)
        # placement actually appears, NOT at request time: a placement can take
        # up to placement_appearance_timeout_s to surface, and that appearance
        # window must not eat into the runner-readiness allowance -- otherwise a
        # model that appears late in the window is falsely reported ready_timeout
        # despite loading for well under placement_ready_timeout_s. Until a
        # placement appears, the appearance deadline bounds the wait.
        # The readiness deadline is anchored to a specific loading instance. When
        # that instance fails/disappears and a REPLACEMENT loads, the deadline is
        # restarted for the new lineage, so a replacement observed late in the old
        # instance's window still gets its full runner-readiness allowance rather
        # than inheriting a near-expired deadline.
        ready_deadline: float | None = None
        ready_anchor: str | None = None
        # Re-anchoring is correct for ONE replacement but unbounded under
        # placement churn: every new placement grants a fresh readiness
        # allowance, so a cluster stuck in a re-place loop (e.g. an OOM race)
        # kept a run silently waiting for over an hour. The churn ceiling is a
        # hard wall-clock bound across ALL re-anchors, counted from the first
        # viable appearance; hitting it fails loudly with the observed
        # placement transitions so the churn itself becomes the finding.
        total_ready_allowance = (
            self.config.placement_ready_total_timeout_s
            if self.config.placement_ready_total_timeout_s is not None
            else (
                2.0 * self.config.placement_ready_timeout_s
                + self.config.placement_appearance_timeout_s
            )
        )
        churn_deadline: float | None = None
        re_anchor_count = 0
        transitions: list[dict[str, object]] = []
        last_signature: tuple[tuple[str, bool, bool], ...] | None = None
        seen_any = False
        no_viable_since: float | None = None
        last_seen: PlacementResult | None = None

        def not_ready(reason: str) -> PlacementResult:
            # Keep the last-known instance identity for a known-but-unhealthy
            # instance (load failure / ready timeout / churn) -- a hard failure.
            # Leave it None for re-placeable give-ups (never appeared /
            # disappeared) so the caller's retry heuristic defers them.
            known = (
                last_seen
                if reason in {"load_failed", "ready_timeout", "churn"}
                else None
            )
            return PlacementResult(
                model_id=model_id,
                created_by_harness=created_by_harness,
                reused_existing=reused,
                ready=False,
                instance_id=known.instance_id if known else None,
                terminal_failure=known.terminal_failure if known else False,
                runner_failure_messages=(
                    known.runner_failure_messages if known else []
                ),
                unavailable_reason=reason,
                readiness_transitions=transitions,
                protected_instance_ids=sorted(ignore_instance_ids),
            )

        while True:
            now = time.monotonic()
            placements = current_placements()
            signature = tuple(
                sorted(
                    (p.instance_id or "", p.ready, p.terminal_failure)
                    for p in placements
                )
            )
            if signature != last_signature:
                transitions.append(
                    {
                        "elapsed_s": round(now - start, 1),
                        "instances": [
                            {
                                "instance_id": p.instance_id,
                                "ready": p.ready,
                                "terminal_failure": p.terminal_failure,
                            }
                            for p in placements
                        ],
                    }
                )
                last_signature = signature

            ready_placements = [p for p in placements if p.ready]
            if ready_placements:
                return ready_placements[0].model_copy(
                    update={
                        "created_by_harness": created_by_harness,
                        "reused_existing": reused,
                        "readiness_transitions": transitions,
                        "protected_instance_ids": sorted(ignore_instance_ids),
                    }
                )

            # The churn ceiling outranks every per-lineage allowance: once the
            # first viable placement appeared, the TOTAL wait across all
            # re-anchors is bounded, however many fresh allowances re-placement
            # keeps granting.
            if churn_deadline is not None and now >= churn_deadline:
                transitions.append(
                    {
                        "elapsed_s": round(now - start, 1),
                        "churn_ceiling_s": total_ready_allowance,
                        "re_anchor_count": re_anchor_count,
                    }
                )
                return not_ready("churn")

            loading = [p for p in placements if not p.terminal_failure]
            if loading:
                # A live instance is still coming up; keep waiting for it.
                seen_any = True
                no_viable_since = None
                last_seen = loading[0]
                loading_ids = {p.instance_id for p in loading}
                if ready_deadline is None or ready_anchor not in loading_ids:
                    # First viable placement, OR the instance the deadline was
                    # anchored to is gone and a (re-)placement is now loading:
                    # start a fresh readiness allowance for this lineage. The
                    # clock is counted from appearance, never from request, so a
                    # late-appearing or re-placed instance gets its full budget.
                    if ready_anchor is not None:
                        re_anchor_count += 1
                    ready_deadline = now + self.config.placement_ready_timeout_s
                    ready_anchor = loading[0].instance_id
                    if churn_deadline is None:
                        # Total ceiling starts at the FIRST appearance, mirroring
                        # the per-lineage clock: the appearance window must not
                        # eat into any readiness allowance.
                        churn_deadline = now + total_ready_allowance
                # Only time out while a loading instance is actually present and
                # has exhausted its (re-anchored) allowance -- never during a
                # re-placement gap, which the else branch governs.
                if now >= ready_deadline:
                    return not_ready("ready_timeout")
            else:
                # Nothing viable right now: either no placement at all, or only
                # instances that entered a terminal failure. Give the cluster a
                # bounded window to (re-)place before giving up, so a re-placement
                # gap is bridged but a vanished instance is never polled forever.
                if placements:
                    seen_any = True
                    last_seen = placements[0]
                if no_viable_since is None:
                    no_viable_since = now
                grace_expired = (
                    now - no_viable_since
                    > self.config.placement_appearance_timeout_s
                )
                never_appeared = not seen_any and now > appearance_deadline
                if never_appeared:
                    return not_ready("never_appeared")
                if seen_any and grace_expired:
                    return not_ready(
                        "load_failed" if placements else "disappeared_without_replacement"
                    )
            time.sleep(self.config.poll_interval_s)

    def _ensure_model_card(
        self, client: SkulkClient, model_id: str, report: RunReport
    ) -> None:
        catalog_ids = {
            _model_id_from_catalog_entry(item) for item in client.list_models()
        }
        if model_id in catalog_ids:
            return
        try:
            client.add_model_card(model_id)
        except (SkulkApiError, httpx.HTTPError) as exc:
            report.issues.append(
                Issue(
                    severity="warning",
                    model_id=model_id,
                    message="Failed to add model card from Skulk/Hugging Face",
                    evidence={"error": str(exc)},
                )
            )

    def _ensure_store_download(
        self, client: SkulkClient, model_id: str, report: RunReport
    ) -> None:
        try:
            client.request_store_download(model_id)
        except (SkulkApiError, httpx.HTTPError) as exc:
            report.issues.append(
                Issue(
                    severity="warning",
                    model_id=model_id,
                    message="Failed to request model-store download",
                    evidence={"error": str(exc)},
                )
            )
            return
        deadline = time.monotonic() + self.config.store_download_timeout_s
        while time.monotonic() < deadline:
            try:
                status = client.get_store_download_status(model_id) or {}
            except (SkulkApiError, httpx.HTTPError):
                time.sleep(self.config.poll_interval_s)
                continue
            status_text = str(status.get("status") or status.get("state") or "").lower()
            if status_text in {"complete", "completed", "ready", "succeeded"}:
                return
            if status_text in {"failed", "error"}:
                report.issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        message="Model-store download failed",
                        evidence=status,
                    )
                )
                return
            time.sleep(self.config.poll_interval_s)
        report.issues.append(
            Issue(
                severity="warning",
                model_id=model_id,
                message="Timed out waiting for model-store download status",
            )
        )

    def _run_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        thinking_default: bool | None = None,
        spec: RunSpec | None = None,
        report: RunReport | None = None,
        writer: ReportWriter | None = None,
    ) -> TestResult:
        if test.kind == "cancel":
            return self._run_cancel_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                thinking_default=thinking_default,
            )
        if test.kind == "concurrent":
            return self._run_concurrent_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                thinking_default=thinking_default,
            )
        if test.kind == "error":
            return self._run_expected_error_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                thinking_default=thinking_default,
            )
        if test.kind == "embedding":
            return self._run_embedding_test(
                client, model_id=model_id, test=test, repetition=repetition
            )
        if test.kind == "vision_data_plane":
            return self._run_vision_data_plane_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                thinking_default=thinking_default,
            )
        if test.kind in {"audio_speech", "audio_speech_streaming"}:
            return self._run_audio_speech_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                stream=test.kind == "audio_speech_streaming",
            )
        if test.kind == "audio_voices":
            return self._run_audio_voices_test(
                client, model_id=model_id, test=test, repetition=repetition
            )
        if test.kind == "audio_speech_pressure":
            return self._run_audio_speech_pressure_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                writer=writer,
            )
        if test.kind == "audio_transcription":
            return self._run_audio_transcription_test(
                client, model_id=model_id, test=test, repetition=repetition
            )
        if test.kind == "audio_transcription_streaming":
            return self._run_streaming_audio_transcription_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                writer=writer,
            )
        if test.kind in {
            "realtime_transcription",
            "realtime_conversation",
            "fabric_speech_chain",
        }:
            return self._run_realtime_transcription_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                writer=writer,
            )
        if test.kind in {"speech_roundtrip", "speech_translation_roundtrip"}:
            return self._run_speech_roundtrip_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                writer=writer,
                translate_to_english=test.kind == "speech_translation_roundtrip",
            )
        if test.kind == "speech_reference_roundtrip":
            return self._run_speech_reference_roundtrip_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                writer=writer,
            )

        messages = _messages_for_test(test)
        # An explicit per-test value always wins; otherwise fall back to the
        # model's resolved toggle default (OFF for toggle-capable models).
        enable_thinking = (
            test.enable_thinking
            if test.enable_thinking is not None
            else thinking_default
        )
        try:
            execution = client.stream_chat(
                model_id=model_id,
                messages=messages,
                max_tokens=test.max_tokens,
                temperature=test.temperature,
                top_p=test.top_p,
                enable_thinking=enable_thinking,
                reasoning_effort=test.reasoning_effort,
                tools=test.tools,
                tool_choice=test.tool_choice,
                parallel_tool_calls=test.parallel_tool_calls,
                top_logprobs=test.top_logprobs,
            )
        except SkulkApiError as exc:
            issue = Issue(
                severity="error",
                model_id=model_id,
                test_name=test.name,
                message="Generation request failed",
                evidence={"error": str(exc)},
            )
            return TestResult(
                model_id=model_id,
                test_name=test.name,
                repetition=repetition,
                passed=False,
                output_text="",
                reasoning_text="",
                tool_calls=[],
                metrics=_empty_metrics(),
                issues=[issue],
            )

        roundtrip_issues: list[Issue] = []
        scored_execution = execution
        if test.tool_mocks and execution.tool_calls:
            roundtrip_messages = _tool_roundtrip_messages(
                messages,
                execution.tool_calls,
                test.tool_mocks,
                model_id=model_id,
                test_name=test.name,
                issues=roundtrip_issues,
            )
            if not any(issue.severity == "error" for issue in roundtrip_issues):
                try:
                    scored_execution = client.stream_chat(
                        model_id=model_id,
                        messages=roundtrip_messages,
                        max_tokens=test.max_tokens,
                        temperature=test.temperature,
                        top_p=test.top_p,
                        enable_thinking=test.enable_thinking,
                        reasoning_effort=test.reasoning_effort,
                    )
                except SkulkApiError as exc:
                    roundtrip_issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Tool-result follow-up generation failed",
                            evidence={"error": str(exc)},
                        )
                    )

        issues = _score_output(
            model_id,
            test.name,
            scored_execution.text,
            test.success,
            tool_calls=execution.tool_calls,
            logprob_tokens=execution.logprob_tokens,
            reasoning_text=scored_execution.reasoning_text,
            wall_tps=scored_execution.metrics.wall_tps,
        )
        issues.extend(roundtrip_issues)
        artifact_path = _artifact_path(
            artifact_dir, model_id, test, repetition, execution
        )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=scored_execution.text,
            reasoning_text=scored_execution.reasoning_text,
            tool_calls=execution.tool_calls,
            metrics=scored_execution.metrics,
            issues=issues,
            artifact_path=artifact_path,
        )

    def _client(self) -> SkulkClient:
        return SkulkClient(
            self.config.api_base_url,
            request_timeout_s=self.config.request_timeout_s,
            generation_timeout_s=self.config.generation_timeout_s,
            stream_read_timeout_s=self.config.stream_read_timeout_s,
        )

    def _client_for_url(self, base_url: str) -> SkulkClient:
        """Create an isolated client for one API owner in a pressure test."""

        return SkulkClient(
            base_url,
            request_timeout_s=self.config.request_timeout_s,
            generation_timeout_s=self.config.generation_timeout_s,
            stream_read_timeout_s=self.config.stream_read_timeout_s,
        )

    def _run_vision_data_plane_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        thinking_default: bool | None = None,
    ) -> TestResult:
        """Prove equivalent VLM input through local and remote API owners."""

        issues: list[Issue] = []
        if not test.images:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Vision DATA qualification requires at least one image",
                )
            )
            return _vision_data_plane_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        owners, serving_node_id = self._select_model_owners(
            client,
            model_id=model_id,
            test_name=test.name,
            owner_count=2,
            owner_topology="local_remote",
            workload_name="vision DATA qualification",
            issues=issues,
        )
        if not owners:
            return _vision_data_plane_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        serving_node_ids = {
            node_id
            for placement in client.find_placements_for_model(model_id)
            if placement.ready
            for node_id in placement.node_ids
        }
        reachable_owners = {
            owner.node_id: owner for owner in client.get_cluster_api_owners()
        }
        missing_serving_owners = serving_node_ids - reachable_owners.keys()
        if missing_serving_owners:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message=(
                        "Not every VLM serving participant has a reachable API "
                        "route for diagnostics"
                    ),
                    evidence={
                        "serving_participant_count": len(serving_node_ids),
                        "unreachable_participant_count": len(missing_serving_owners),
                    },
                )
            )
            return _vision_data_plane_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )
        remote_request_owners = [
            owner for owner in owners if owner.node_id != serving_node_id
        ]
        diagnostic_owners = [owners[0]]
        diagnostic_owners.extend(
            reachable_owners[node_id]
            for node_id in sorted(serving_node_ids)
            if node_id != serving_node_id
        )
        diagnostic_owners.extend(remote_request_owners)

        try:
            diagnostics_before = self._capture_vision_media_diagnostics(
                diagnostic_owners
            )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Unable to capture pre-request vision media diagnostics",
                    evidence={"error": str(exc)},
                )
            )
            return _vision_data_plane_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        issues.extend(
            _score_vision_media_baseline(
                model_id=model_id,
                test_name=test.name,
                owners=diagnostic_owners,
                serving_node_id=serving_node_id,
                serving_node_ids=serving_node_ids,
                snapshots=diagnostics_before,
            )
        )
        if any(issue.severity == "error" for issue in issues):
            return _vision_data_plane_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        messages = _messages_for_test(test)
        enable_thinking = (
            test.enable_thinking
            if test.enable_thinking is not None
            else thinking_default
        )
        output_records: list[tuple[str, ChatExecution]] = []
        successful_requests = 0
        elapsed_s = 0.0
        for owner_index, owner in enumerate(owners, start=1):
            role = (
                "serving_local"
                if owner.node_id == serving_node_id
                else "remote_owner"
            )
            label = f"owner-{owner_index}-{role}"
            try:
                with self._client_for_url(owner.base_url) as owner_client:
                    execution = owner_client.stream_chat(
                        model_id=model_id,
                        messages=messages,
                        max_tokens=test.max_tokens,
                        temperature=test.temperature,
                        top_p=test.top_p,
                        enable_thinking=enable_thinking,
                        reasoning_effort=test.reasoning_effort,
                        top_logprobs=test.top_logprobs,
                    )
            except (SkulkApiError, httpx.HTTPError) as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Vision request failed through selected API owner",
                        evidence={"owner": label, "error": str(exc)},
                    )
                )
                continue
            successful_requests += 1
            elapsed_s += execution.metrics.elapsed_s
            output_records.append((label, execution))
            for issue in _score_output(
                model_id,
                test.name,
                execution.text,
                test.success,
                logprob_tokens=execution.logprob_tokens,
                reasoning_text=execution.reasoning_text,
                wall_tps=execution.metrics.wall_tps,
            ):
                issues.append(
                    issue.model_copy(
                        update={"evidence": {**issue.evidence, "owner": label}}
                    )
                )

        diagnostics_artifact: Path | None = None
        try:
            diagnostics_after = self._wait_for_vision_media_idle(diagnostic_owners)
            diagnostic_issues, diagnostic_records = _score_vision_media_diagnostics(
                model_id=model_id,
                test_name=test.name,
                owners=diagnostic_owners,
                request_owners=owners,
                serving_node_id=serving_node_id,
                serving_node_ids=serving_node_ids,
                before=diagnostics_before,
                after=diagnostics_after,
                successful_requests=successful_requests,
            )
            issues.extend(diagnostic_issues)
            diagnostics_artifact = _vision_media_diagnostics_artifact_path(
                artifact_dir,
                model_id,
                test.name,
                repetition,
                diagnostic_records,
            )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Unable to validate post-request vision media diagnostics",
                    evidence={"error": str(exc)},
                )
            )

        output = "\n".join(
            f"{label}: {execution.text}" for label, execution in output_records
        )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            reasoning_text="\n".join(
                execution.reasoning_text for _, execution in output_records
            ),
            metrics=GenerationMetrics(
                elapsed_s=elapsed_s,
                output_chars=len(output),
                generated_chars=sum(
                    len(execution.text) + len(execution.reasoning_text)
                    for _, execution in output_records
                ),
            ),
            issues=issues,
            artifact_path=diagnostics_artifact,
        )

    def _run_concurrent_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        thinking_default: bool | None = None,
    ) -> TestResult:
        """Drive ``concurrency`` simultaneous chat completions and measure load.

        Runs ``test.concurrency`` asyncio worker tasks on ONE single-threaded
        event loop, each issuing ``concurrent_requests_per_worker`` streamed
        completions in sequence. The driver is deliberately asyncio rather than
        worker threads: N GIL-contended Python threads parsing SSE understate
        high-rate aggregates 25-35% at N=64 (measured against the same server
        class), so a threaded wall-clock aggregate reflects the measurement
        client, not the server. All workers share one async client whose pool
        holds ``concurrency`` connections but keeps NONE alive between requests
        (see ``concurrent_benchmark_client``): stream-closing servers otherwise
        turn every second pooled request into a transport error or a
        clean-looking empty stream. Two things are measured at once:

        * **Aggregate throughput** (``aggregate_generation_tps``): total generated
          tokens across every request that returned tokens -- regardless of
          scoring outcome -- divided by the wall span from the first request
          starting to the last finishing (measured from per-request timestamps,
          not the submit/collect wall). Tokens count whenever the request's time
          is in the span; excluding a scoring-failed request's tokens while its
          elapsed time widens the denominator would deflate the aggregate. This
          is the number a batching engine on a large GPU grows as concurrency
          rises, while a single-stream decode rate cannot. It is also copied
          into ``skulk_generation_tps`` so existing single-number readers
          (including the ledger) surface it.
        * **Correctness under load**: every request must succeed and satisfy the
          same success criteria as a chat test. Any error or failed scoring fails
          the whole test, so this doubles as the concurrent-load smoke that the
          harness previously never exercised. Scoring pass/fail stays a separate
          axis (``concurrent_succeeded`` / ``concurrent_failed``) from the
          throughput accounting above.

        Per-request decode rate and TTFT are reported as mean/p50/p90 so the
        expected per-stream slowdown (each stream shares the batch) is visible
        alongside the aggregate gain.
        """

        messages = _messages_for_test(test)
        enable_thinking = (
            test.enable_thinking
            if test.enable_thinking is not None
            else thinking_default
        )
        base_url = client.base_url
        total_requests = test.concurrency * test.concurrent_requests_per_worker
        if test.tool_mocks:
            # The concurrent kind is a single-turn load benchmark: it forwards
            # `tools` so tool-call emission can be exercised under load, but it
            # deliberately does not run the tool-result round trip the chat/tool
            # kinds do (that would turn each timed request into a variable number
            # of API calls and is the `tool` kind's job at concurrency 1). Scoring
            # a tool_mocks response here would silently measure only the tool-call
            # emission, not the follow-up answer, so reject it loudly instead.
            return TestResult(
                model_id=model_id,
                test_name=test.name,
                repetition=repetition,
                passed=False,
                output_text="",
                metrics=GenerationMetrics(
                    elapsed_s=0.0,
                    concurrency=test.concurrency,
                    concurrent_total_requests=total_requests,
                    concurrent_succeeded=0,
                    concurrent_failed=0,
                ),
                issues=[
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message=(
                            "kind: concurrent does not support tool_mocks (it is a "
                            "single-turn load benchmark and does not run the "
                            "tool-result round trip; use kind: tool for that)"
                        ),
                    )
                ],
            )
        records = asyncio.run(
            self._drive_concurrent_requests(
                base_url=base_url,
                model_id=model_id,
                test=test,
                messages=messages,
                enable_thinking=enable_thinking,
            )
        )
        # Span = first request start to last request finish (the real in-flight
        # window), not the submit/collect wall, so thread scheduling overhead
        # does not inflate the denominator of aggregate throughput. Each record
        # is (execution, error, started_at, ended_at).
        wall_span = (
            max(r[3] for r in records) - min(r[2] for r in records) if records else 0.0
        )

        issues: list[Issue] = []
        succeeded = 0
        failed = 0
        per_request_tps: list[float] = []
        ttfts: list[float] = []
        total_generation_tokens = 0
        sample_text = ""
        for execution, error, _started, _ended in records:
            if execution is None:
                failed += 1
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Concurrent request failed",
                        evidence={"error": error or "no execution"},
                    )
                )
                continue
            metrics = execution.metrics
            # Throughput accounting happens BEFORE scoring: every request that
            # returned tokens already contributed its elapsed time to the
            # wall-span denominator, so its tokens must contribute to the
            # numerator regardless of scoring outcome. Excluding scoring-failed
            # requests (the old behavior) deflated aggregate_generation_tps --
            # e.g. min_chars failures that still generated hundreds of chars
            # each counted as pure dead time. Scoring pass/fail is recorded
            # separately below via concurrent_succeeded/concurrent_failed.
            if metrics.skulk_generation_tokens is not None:
                total_generation_tokens += metrics.skulk_generation_tokens
            elif metrics.approx_output_tokens is not None:
                total_generation_tokens += metrics.approx_output_tokens
            # TTFT is a load-latency observation, valid for every request that
            # streamed output; scoring cannot retroactively change when the
            # first token arrived.
            if metrics.ttft_s is not None:
                ttfts.append(metrics.ttft_s)
            # Prefer Skulk's engine-reported decode rate, but fall back to the
            # client-computed wall_tps when a stream carries no generation_stats
            # (older runs or engines that only provide wall timing). Mirrors the
            # aggregate path's token fallback and compare._decode_tps, so the
            # per-request distribution is not blank in those environments. The
            # per-request distribution keeps its documented "successful
            # requests" definition, so it is appended only after scoring below.
            per_request_rate = (
                metrics.skulk_generation_tps
                if metrics.skulk_generation_tps is not None
                else metrics.wall_tps
            )
            response_issues = _score_output(
                model_id,
                test.name,
                execution.text,
                test.success,
                tool_calls=execution.tool_calls,
                logprob_tokens=execution.logprob_tokens,
                reasoning_text=execution.reasoning_text,
                wall_tps=execution.metrics.wall_tps,
            )
            if any(issue.severity == "error" for issue in response_issues):
                failed += 1
                issues.extend(response_issues)
                continue
            succeeded += 1
            if per_request_rate is not None:
                per_request_tps.append(per_request_rate)
            if not sample_text:
                sample_text = execution.text

        # A worker that aborts (an unexpected fault or a client-construction
        # failure) with concurrent_requests_per_worker > 1 records a single
        # failure and stops, so its remaining planned requests never reach
        # records. Count those unissued slots as failures so concurrent_failed
        # reflects all dropped work and succeeded + failed always equals the
        # planned total (this also covers the degenerate no-records case).
        dropped = total_requests - len(records)
        if dropped > 0:
            failed += dropped
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Concurrent worker aborted; planned requests were not issued",
                    evidence={"dropped_requests": dropped},
                )
            )
        aggregate_tps = (
            total_generation_tokens / wall_span
            if wall_span > 0 and total_generation_tokens > 0
            else None
        )
        aggregated_metrics = GenerationMetrics(
            elapsed_s=wall_span,
            wall_span_s=wall_span,
            skulk_generation_tps=aggregate_tps,
            aggregate_generation_tps=aggregate_tps,
            skulk_generation_tokens=total_generation_tokens or None,
            concurrency=test.concurrency,
            concurrent_total_requests=total_requests,
            concurrent_succeeded=succeeded,
            concurrent_failed=failed,
            per_request_generation_tps_mean=(
                statistics.fmean(per_request_tps) if per_request_tps else None
            ),
            per_request_generation_tps_p50=_percentile(per_request_tps, 50),
            per_request_generation_tps_p90=_percentile(per_request_tps, 90),
            ttft_p50_s=_percentile(ttfts, 50),
            ttft_p90_s=_percentile(ttfts, 90),
        )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=failed == 0 and succeeded == total_requests,
            output_text=sample_text,
            reasoning_text="",
            tool_calls=[],
            metrics=aggregated_metrics,
            issues=issues,
        )

    def _concurrent_async_client(
        self, base_url: str, *, concurrency: int
    ) -> httpx.AsyncClient:
        """Build the shared force-close async client for one concurrent test.

        Separated out as the seam unit tests replace; production behavior lives
        in ``concurrent_benchmark_client``.
        """

        return concurrent_benchmark_client(base_url, concurrency=concurrency)

    async def _concurrent_stream_chat(
        self,
        async_client: httpx.AsyncClient,
        *,
        model_id: str,
        messages: list[dict[str, object]],
        test: PromptTest,
        enable_thinking: bool | None,
    ) -> ChatExecution:
        """Issue one streamed completion for the concurrent benchmark."""

        return await stream_chat_async(
            async_client,
            model_id=model_id,
            messages=messages,
            max_tokens=test.max_tokens,
            temperature=test.temperature,
            top_p=test.top_p,
            enable_thinking=enable_thinking,
            reasoning_effort=test.reasoning_effort,
            # Mirror the chat request options so a concurrent test that
            # exercises tools or logprobs asks the backend for them (dropping
            # them would falsely fail tool/logprob criteria under load).
            tools=test.tools,
            tool_choice=test.tool_choice,
            parallel_tool_calls=test.parallel_tool_calls,
            top_logprobs=test.top_logprobs,
            request_timeout_s=self.config.request_timeout_s,
            stream_read_timeout_s=self.config.stream_read_timeout_s,
        )

    async def _drive_concurrent_requests(
        self,
        *,
        base_url: str,
        model_id: str,
        test: PromptTest,
        messages: list[dict[str, object]],
        enable_thinking: bool | None,
    ) -> list[_ConcurrentRecord]:
        """Run every planned concurrent request on one asyncio event loop.

        Single-threaded on purpose: worker THREADS parsing N simultaneous SSE
        streams contend on the GIL and understate high-rate aggregates, so the
        measured number would be the client's ceiling rather than the server's.
        All workers start together via ``asyncio.gather`` (no start barrier is
        needed -- task startup on one loop is effectively simultaneous), and
        every request uses a fresh connection from the shared force-close
        client so stream-closing servers cannot poison a keep-alive pool.
        """

        records: list[_ConcurrentRecord] = []
        async with self._concurrent_async_client(
            base_url, concurrency=test.concurrency
        ) as async_client:

            async def worker(_worker_index: int) -> list[_ConcurrentRecord]:
                samples: list[_ConcurrentRecord] = []
                for _ in range(test.concurrent_requests_per_worker):
                    started = time.monotonic()
                    try:
                        execution = await self._concurrent_stream_chat(
                            async_client,
                            model_id=model_id,
                            messages=messages,
                            test=test,
                            enable_thinking=enable_thinking,
                        )
                        samples.append((execution, None, started, time.monotonic()))
                    except (SkulkApiError, httpx.HTTPError) as exc:
                        samples.append((None, str(exc), started, time.monotonic()))
                    except Exception as exc:  # noqa: BLE001
                        # A benchmark must record a failure and keep running,
                        # never abort the whole run. An unexpected fault ends
                        # THIS worker's remaining planned requests (the caller
                        # counts those dropped slots as failures) but leaves
                        # every other worker running.
                        samples.append(
                            (
                                None,
                                f"worker error: {exc}",
                                started,
                                time.monotonic(),
                            )
                        )
                        break
                return samples

            worker_results = await asyncio.gather(
                *(worker(index) for index in range(test.concurrency)),
                return_exceptions=True,
            )
        for worker_result in worker_results:
            if isinstance(worker_result, BaseException):
                # The worker already converts its own faults to failed samples;
                # this guard covers anything that still escapes (task-level
                # errors) so one worker cannot take down the run.
                now = time.monotonic()
                records.append((None, f"worker crashed: {worker_result}", now, now))
            else:
                records.extend(worker_result)
        return records

    def _run_cancel_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        thinking_default: bool | None = None,
    ) -> TestResult:
        issues: list[Issue] = []
        cancel_after = max(1, test.cancel_after_chunks)
        enable_thinking = (
            test.enable_thinking
            if test.enable_thinking is not None
            else thinking_default
        )
        try:
            canceled = client.stream_chat(
                model_id=model_id,
                messages=_messages_for_test(test),
                max_tokens=test.max_tokens,
                temperature=test.temperature,
                top_p=test.top_p,
                enable_thinking=enable_thinking,
                reasoning_effort=test.reasoning_effort,
                tools=test.tools,
                tool_choice=test.tool_choice,
                parallel_tool_calls=test.parallel_tool_calls,
                cancel_after_chunks=cancel_after,
            )
        except SkulkApiError as exc:
            issue = Issue(
                severity="error",
                model_id=model_id,
                test_name=test.name,
                message="Cancellable generation request failed before cancellation",
                evidence={"error": str(exc)},
            )
            return TestResult(
                model_id=model_id,
                test_name=test.name,
                repetition=repetition,
                passed=False,
                output_text="",
                metrics=_empty_metrics(),
                issues=[issue],
            )
        if not canceled.canceled:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Stream completed before the harness could cancel it",
                    evidence={"chunks": canceled.metrics.chunks},
                )
            )
        if canceled.metrics.chunks < cancel_after:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message=(
                        "Stream produced fewer chunks than the configured "
                        "cancellation point"
                    ),
                    evidence={
                        "chunks": canceled.metrics.chunks,
                        "cancel_after_chunks": cancel_after,
                    },
                )
            )

        followup_test = test.model_copy(
            update={
                "prompt": test.followup_prompt
                or "Reply with exactly this token: CANCEL-HEALTHY",
                "prompt_repetitions": 1,
                "images": [],
                "tools": [],
                "tool_choice": None,
                "parallel_tool_calls": None,
            }
        )
        try:
            followup = client.stream_chat(
                model_id=model_id,
                messages=_messages_for_test(followup_test),
                max_tokens=min(test.max_tokens, 96),
                temperature=0,
                top_p=None,
                enable_thinking=enable_thinking,
                reasoning_effort=None,
            )
        except SkulkApiError as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Follow-up generation failed after cancellation",
                    evidence={"error": str(exc)},
                )
            )
            followup = ChatExecution(
                text="",
                reasoning_text="",
                tool_calls=[],
                metrics=_empty_metrics(),
                command_id=None,
                raw_events=[],
            )
        issues.extend(
            _score_output(
                model_id,
                test.name,
                followup.text,
                test.success,
                reasoning_text=followup.reasoning_text,
                wall_tps=followup.metrics.wall_tps,
            )
        )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=followup.text,
            reasoning_text=followup.reasoning_text,
            metrics=followup.metrics,
            issues=issues,
        )

    def _run_expected_error_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        thinking_default: bool | None = None,
    ) -> TestResult:
        issues: list[Issue] = []
        error_text = ""
        enable_thinking = (
            test.enable_thinking
            if test.enable_thinking is not None
            else thinking_default
        )
        try:
            execution = client.stream_chat(
                model_id=model_id,
                messages=_messages_for_test(test),
                max_tokens=test.max_tokens,
                temperature=test.temperature,
                top_p=test.top_p,
                enable_thinking=enable_thinking,
                reasoning_effort=test.reasoning_effort,
            )
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Expected generation request to fail, but it succeeded",
                    evidence={"output_chars": execution.metrics.output_chars},
                )
            )
        except SkulkApiError as exc:
            error_text = exc.body
            if (
                test.expected_error_statuses
                and exc.status_code not in test.expected_error_statuses
            ):
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Generation failed with unexpected HTTP status",
                        evidence={
                            "expected": test.expected_error_statuses,
                            "actual": exc.status_code,
                            "body": exc.body,
                        },
                    )
                )
            for substring in test.expected_error_substrings:
                if substring.lower() not in exc.body.lower():
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message=(
                                "Expected error body to contain substring "
                                f"{substring!r}"
                            ),
                            evidence={"body": exc.body},
                        )
                    )

        if test.followup_prompt:
            followup_test = test.model_copy(
                update={
                    "prompt": test.followup_prompt,
                    "prompt_repetitions": 1,
                    "images": [],
                }
            )
            try:
                followup = client.stream_chat(
                    model_id=model_id,
                    messages=_messages_for_test(followup_test),
                    max_tokens=96,
                    temperature=0,
                    top_p=None,
                    enable_thinking=enable_thinking,
                    reasoning_effort=None,
                )
            except SkulkApiError as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Follow-up generation failed after expected error",
                        evidence={"error": str(exc)},
                    )
                )
            else:
                if not followup.text.strip() and not followup.reasoning_text.strip():
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message=(
                                "Follow-up generation after expected error was empty"
                            ),
                        )
                    )

        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=error_text,
            metrics=_empty_metrics(),
            issues=issues,
        )

    def _run_embedding_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
    ) -> TestResult:
        issues: list[Issue] = []
        try:
            execution = client.embeddings(
                model_id=model_id,
                input_text=test.embedding_input or _expanded_prompt(test),
            )
        except (SkulkApiError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Embedding request failed",
                    evidence={"error": str(exc)},
                )
            )
            execution = None
        if execution is not None:
            if test.expected_embedding_dimensions is not None and any(
                dim != test.expected_embedding_dimensions
                for dim in execution.dimensions
            ):
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Embedding vector dimensionality did not match",
                        evidence={
                            "expected": test.expected_embedding_dimensions,
                            "actual": execution.dimensions,
                        },
                    )
                )
            if any(norm < test.min_embedding_norm for norm in execution.norms):
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Embedding vector norm below required minimum",
                        evidence={
                            "min_embedding_norm": test.min_embedding_norm,
                            "actual_norms": execution.norms,
                        },
                    )
                )
        output = ""
        elapsed = 0.0
        if execution is not None:
            elapsed = execution.elapsed_s
            output = (
                f"dimensions={execution.dimensions} "
                f"norms={[round(norm, 4) for norm in execution.norms]}"
            )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=elapsed,
                output_chars=len(output),
                generated_chars=len(output),
            ),
            issues=issues,
        )

    def _run_audio_speech_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        stream: bool = False,
    ) -> TestResult:
        """Run a TTS request and assert Skulk returns plausible audio bytes."""

        issues: list[Issue] = []
        output = ""
        elapsed = 0.0
        artifact_path: Path | None = None
        execution: AudioSpeechExecution | None = None
        try:
            execution = client.audio_speech(
                model_id=model_id,
                input_text=_expanded_prompt(test),
                response_format=test.audio_response_format,
                voice=test.speech_voice,
                speed=test.speech_speed,
                **_speech_generation_kwargs(test),
                stream=stream,
                streaming_interval=test.speech_streaming_interval,
            )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Speech synthesis request failed",
                    evidence={"error": str(exc)},
                )
            )
        else:
            elapsed = execution.elapsed_s
            issues.extend(
                _score_audio_output(
                    model_id,
                    test.name,
                    execution.audio,
                    test.success,
                    response_format=test.audio_response_format,
                    media_type=execution.media_type,
                )
            )
            if stream:
                issues.extend(
                    _score_streaming_audio_output(
                        model_id,
                        test.name,
                        execution,
                        test.success,
                    )
                )
            artifact_path = _audio_artifact_path(
                artifact_dir,
                model_id,
                test.name,
                repetition,
                execution.response_format,
                execution.audio,
            )
            stream_metadata_path: Path | None = None
            if stream:
                stream_span_s = _stream_span_s(execution.chunk_arrival_s)
                stream_metadata_path = _audio_stream_metadata_path(
                    artifact_path,
                    execution.chunks,
                    execution.first_byte_s,
                    stream_span_s,
                    execution.chunk_sizes,
                    execution.chunk_arrival_s,
                )
            output = (
                f"audio_bytes={len(execution.audio)} "
                f"media_type={execution.media_type} "
                f"format={execution.response_format} "
                f"streaming={execution.streaming} "
                f"chunks={execution.chunks} "
                f"first_byte_s={_fmt_optional_float(execution.first_byte_s)} "
                f"stream_span_s={_fmt_optional_float(_stream_span_s(execution.chunk_arrival_s))} "
                f"artifact={artifact_path}"
            )
            if stream_metadata_path is not None:
                output = f"{output} stream_metadata={stream_metadata_path}"
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=elapsed,
                ttft_s=execution.first_byte_s if execution is not None else None,
                output_chars=len(output),
                generated_chars=len(output),
                chunks=execution.chunks if execution is not None else 0,
            ),
            issues=issues,
            artifact_path=artifact_path,
        )

    def _run_audio_voices_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
    ) -> TestResult:
        """Require a mounted model's static voice catalog to match expectations."""

        issues: list[Issue] = []
        voices: list[str] = []
        started_at = time.monotonic()
        try:
            voices = client.audio_voices(model_id)
            missing = sorted(set(test.expected_voice_ids) - set(voices))
            if missing:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Voice catalog omitted required identifiers",
                        evidence={"missing": missing, "actual": voices},
                    )
                )
            if not voices:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Voice catalog was empty",
                    )
                )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Voice catalog request failed",
                    evidence={"error": str(exc)},
                )
            )
        output = f"voices={voices}"
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=time.monotonic() - started_at,
                output_chars=len(output),
            ),
            issues=issues,
        )

    def _run_audio_speech_pressure_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
    ) -> TestResult:
        """Run pressure and always release a harness-created secondary model."""

        secondary_placements: list[tuple[str, PlacementResult]] = []
        try:
            return self._run_audio_speech_pressure_test_inner(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                secondary_placements=secondary_placements,
            )
        finally:
            if spec is not None and report is not None and not spec.retain_instances:
                for secondary_model_id, placement in secondary_placements:
                    if (
                        secondary_model_id == model_id
                        or not placement.created_by_harness
                    ):
                        continue
                    torn_down = self._teardown_harness_instances(
                        client,
                        secondary_model_id,
                        placement.instance_id,
                        report,
                        protected_instance_ids=frozenset(
                            placement.protected_instance_ids
                        ),
                    )
                    if spec.delete_staged_models and torn_down:
                        self._evict_staged_model(client, secondary_model_id, report)
                if secondary_placements and writer is not None:
                    writer.write(report)

    def _run_audio_speech_pressure_test_inner(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        secondary_placements: list[tuple[str, PlacementResult]],
    ) -> TestResult:
        """Drive concurrent TTS and optional chat through known API owners."""

        issues: list[Issue] = []
        owners, serving_node_id = self._select_speech_owners(
            client, model_id, test, issues
        )
        if not owners:
            return _speech_pressure_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
                elapsed_s=0.0,
            )

        chat_model_id = test.speech_chat_model_id
        chat_placement: PlacementResult | None = None
        if test.speech_chat_concurrency > 0:
            if spec is None or report is None or not chat_model_id:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message=(
                            "Mixed speech pressure requires speech_chat_model_id "
                            "and an execution RunSpec/report"
                        ),
                    )
                )
                return _speech_pressure_result(
                    model_id=model_id,
                    test=test,
                    repetition=repetition,
                    issues=issues,
                    elapsed_s=0.0,
                )
            chat_placement = self._ensure_model_placed(
                client, chat_model_id, spec, report
            )
            if chat_placement is not None:
                secondary_placements.append((chat_model_id, chat_placement))
            if chat_placement is None or not chat_placement.ready:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Chat model could not be made ready for mixed pressure",
                        evidence={"chat_model_id": chat_model_id},
                    )
                )
                return _speech_pressure_result(
                    model_id=model_id,
                    test=test,
                    repetition=repetition,
                    issues=issues,
                    elapsed_s=0.0,
                )
            _append_unique_placement(report, chat_placement)

        diagnostics_before: dict[str, DataPlaneDiagnosticsSnapshot] = {}
        if test.speech_assert_data_plane_diagnostics:
            try:
                diagnostics_before = self._capture_data_plane_diagnostics(owners)
            except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Unable to capture pre-pressure DATA diagnostics",
                        evidence={"error": str(exc)},
                    )
                )
            else:
                issues.extend(
                    _score_data_plane_baseline(
                        model_id=model_id,
                        test_name=test.name,
                        owners=owners,
                        serving_node_id=serving_node_id,
                        snapshots=diagnostics_before,
                    )
                )
                if issues:
                    return _speech_pressure_result(
                        model_id=model_id,
                        test=test,
                        repetition=repetition,
                        issues=issues,
                        elapsed_s=0.0,
                    )

        started_at = time.monotonic()

        def run_speech_worker(
            worker_index: int,
        ) -> list[tuple[int, AudioSpeechExecution | None, str | None]]:
            owner = owners[worker_index % len(owners)]
            samples: list[tuple[int, AudioSpeechExecution | None, str | None]] = []
            read_delay_s = (
                test.speech_slow_reader_delay_s
                if worker_index < test.speech_slow_workers
                else 0.0
            )
            with self._client_for_url(owner.base_url) as owner_client:
                for request_index in range(test.speech_requests_per_worker):
                    try:
                        execution = owner_client.audio_speech(
                            model_id=model_id,
                            input_text=_expanded_prompt(test),
                            response_format=test.audio_response_format,
                            voice=test.speech_voice,
                            speed=test.speech_speed,
                            **_speech_generation_kwargs(test),
                            stream=True,
                            streaming_interval=test.speech_streaming_interval,
                            read_delay_s=read_delay_s,
                        )
                    except (
                        SkulkApiError,
                        httpx.HTTPError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        samples.append((request_index, None, str(exc)))
                    else:
                        samples.append((request_index, execution, None))
            return samples

        chat_thinking_default: bool | None = None
        if chat_model_id:
            try:
                if client.resolved_thinking_toggle_by_model().get(chat_model_id):
                    chat_thinking_default = False
            except (SkulkApiError, httpx.HTTPError, TypeError, ValueError):
                pass

        def run_chat_worker(
            worker_index: int,
        ) -> tuple[ChatExecution | None, str | None]:
            assert chat_model_id is not None
            owner = owners[worker_index % len(owners)]
            with self._client_for_url(owner.base_url) as owner_client:
                try:
                    execution = owner_client.stream_chat(
                        model_id=chat_model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": test.speech_chat_prompt or test.prompt,
                            }
                        ],
                        max_tokens=test.max_tokens,
                        temperature=test.temperature,
                        top_p=test.top_p,
                        enable_thinking=chat_thinking_default,
                    )
                except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
                    return None, str(exc)
            return execution, None

        speech_samples: list[
            tuple[int, list[tuple[int, AudioSpeechExecution | None, str | None]]]
        ] = []
        chat_samples: list[tuple[int, ChatExecution | None, str | None]] = []
        max_workers = test.speech_concurrency + test.speech_chat_concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            speech_futures = {
                pool.submit(run_speech_worker, worker_index): worker_index
                for worker_index in range(test.speech_concurrency)
            }
            chat_futures = {
                pool.submit(run_chat_worker, worker_index): worker_index
                for worker_index in range(test.speech_chat_concurrency)
            }
            for future in concurrent.futures.as_completed(speech_futures):
                speech_samples.append((speech_futures[future], future.result()))
            for future in concurrent.futures.as_completed(chat_futures):
                execution, error = future.result()
                chat_samples.append((chat_futures[future], execution, error))

        elapsed_s = time.monotonic() - started_at
        executions: list[AudioSpeechExecution] = []
        chat_executions: list[ChatExecution] = []
        artifacts: list[Path] = []
        failures = 0
        for worker_index, samples in sorted(speech_samples):
            for request_index, execution, error in samples:
                sample_name = (
                    f"{test.name}-worker-{worker_index + 1}-request-{request_index + 1}"
                )
                if execution is None:
                    failures += 1
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Concurrent speech synthesis request failed",
                            evidence={
                                "worker": worker_index + 1,
                                "request": request_index + 1,
                                "error": error or "unknown failure",
                            },
                        )
                    )
                    continue
                executions.append(execution)
                issues.extend(
                    _score_audio_output(
                        model_id,
                        sample_name,
                        execution.audio,
                        test.success,
                        response_format=test.audio_response_format,
                        media_type=execution.media_type,
                    )
                )
                issues.extend(
                    _score_streaming_audio_output(
                        model_id,
                        sample_name,
                        execution,
                        (
                            test.success.model_copy(update={"min_stream_span_s": 0.0})
                            if worker_index < test.speech_slow_workers
                            else test.success
                        ),
                    )
                )
                artifact = _audio_artifact_path(
                    artifact_dir,
                    model_id,
                    sample_name,
                    repetition,
                    execution.response_format,
                    execution.audio,
                )
                artifacts.append(artifact)
                _audio_stream_metadata_path(
                    artifact,
                    execution.chunks,
                    execution.first_byte_s,
                    _stream_span_s(execution.chunk_arrival_s),
                    execution.chunk_sizes,
                    execution.chunk_arrival_s,
                )

        for worker_index, execution, error in sorted(chat_samples):
            if execution is None:
                failures += 1
                issues.append(
                    Issue(
                        severity="error",
                        model_id=chat_model_id,
                        test_name=test.name,
                        message="Concurrent chat request failed during speech pressure",
                        evidence={
                            "worker": worker_index + 1,
                            "error": error or "unknown failure",
                        },
                    )
                )
                continue
            chat_executions.append(execution)
            issues.extend(
                _score_output(
                    chat_model_id or model_id,
                    f"{test.name}-chat-worker-{worker_index + 1}",
                    execution.text + execution.reasoning_text,
                    test.success,
                )
            )

        diagnostics_artifact: Path | None = None
        if test.speech_assert_data_plane_diagnostics and diagnostics_before:
            try:
                diagnostics_after = self._wait_for_data_plane_idle(owners)
                diagnostic_issues, diagnostic_records = _score_data_plane_diagnostics(
                    model_id=model_id,
                    test_name=test.name,
                    owners=owners,
                    serving_node_id=serving_node_id,
                    before=diagnostics_before,
                    after=diagnostics_after,
                    successful_streams=len(executions) + len(chat_executions),
                    require_local_remote=(test.speech_owner_topology == "local_remote"),
                )
                issues.extend(diagnostic_issues)
                diagnostics_artifact = _data_plane_diagnostics_artifact_path(
                    artifact_dir,
                    model_id,
                    test.name,
                    repetition,
                    diagnostic_records,
                )
            except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Unable to validate post-pressure DATA diagnostics",
                        evidence={"error": str(exc)},
                    )
                )

        first_bytes = [
            value
            for value in (
                [execution.first_byte_s for execution in executions]
                + [execution.metrics.ttft_s for execution in chat_executions]
            )
            if value is not None
        ]
        speech_request_count = test.speech_concurrency * test.speech_requests_per_worker
        output = (
            f"speech_requests={speech_request_count} "
            f"speech_successes={len(executions)} "
            f"chat_requests={test.speech_chat_concurrency} "
            f"chat_successes={len(chat_executions)} failures={failures} "
            f"owners={len(owners)} topology={test.speech_owner_topology} "
            f"slow_workers={test.speech_slow_workers} artifacts={len(artifacts)}"
        )
        if diagnostics_artifact is not None:
            output += f" diagnostics={diagnostics_artifact}"

        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=elapsed_s,
                ttft_s=statistics.median(first_bytes) if first_bytes else None,
                chunks=(
                    sum(execution.chunks for execution in executions)
                    + sum(execution.metrics.chunks for execution in chat_executions)
                ),
                output_chars=len(output),
                generated_chars=len(output),
            ),
            issues=issues,
            artifact_path=artifacts[0] if artifacts else None,
        )

    def _select_speech_owners(
        self,
        client: SkulkClient,
        model_id: str,
        test: PromptTest,
        issues: list[Issue],
    ) -> tuple[list[ClusterApiOwner], str | None]:
        """Choose API owners, optionally relative to the model's serving node."""

        return self._select_model_owners(
            client,
            model_id=model_id,
            test_name=test.name,
            owner_count=test.speech_owner_count,
            owner_topology=test.speech_owner_topology,
            workload_name="speech test",
            issues=issues,
        )

    def _select_model_owners(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test_name: str,
        owner_count: int,
        owner_topology: OwnerTopology,
        workload_name: str,
        issues: list[Issue],
    ) -> tuple[list[ClusterApiOwner], str | None]:
        """Choose reachable API owners relative to a mounted model placement."""

        owners = client.get_cluster_api_owners()
        serving_node_ids = {
            node_id
            for placement in client.find_placements_for_model(model_id)
            if placement.ready
            for node_id in placement.node_ids
        }
        if owner_topology == "any":
            selected = owners[:owner_count]
            serving_node_id = next(iter(sorted(serving_node_ids)), None)
        else:
            if owner_count < 2:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test_name,
                        message=(
                            f"{workload_name} local_remote topology requires at "
                            "least two owners"
                        ),
                    )
                )
                return [], None
            local = sorted(
                (owner for owner in owners if owner.node_id in serving_node_ids),
                key=lambda owner: owner.node_id,
            )
            remote = sorted(
                (owner for owner in owners if owner.node_id not in serving_node_ids),
                key=lambda owner: owner.node_id,
            )
            if not local or len(remote) < owner_count - 1:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test_name,
                        message=(
                            "Could not resolve the requested deterministic local/remote "
                            "API owner topology"
                        ),
                        evidence={
                            "serving_owner_count": len(local),
                            "remote_owner_count": len(remote),
                            "required_owner_count": owner_count,
                        },
                    )
                )
                return [], None
            selected = [local[0], *remote[: owner_count - 1]]
            serving_node_id = local[0].node_id

        if len(selected) < owner_count:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"Not enough reachable API owners for {workload_name}",
                    evidence={
                        "required_owner_count": owner_count,
                        "reachable_owner_count": len(owners),
                    },
                )
            )
            return [], serving_node_id
        return selected, serving_node_id

    def _capture_vision_media_diagnostics(
        self, owners: list[ClusterApiOwner]
    ) -> dict[str, VisionMediaDiagnosticsSnapshot]:
        """Read one vision media snapshot from every selected owner."""

        snapshots: dict[str, VisionMediaDiagnosticsSnapshot] = {}
        for owner in owners:
            with self._client_for_url(owner.base_url) as owner_client:
                snapshot = owner_client.get_vision_media_diagnostics()
            if snapshot.node_id != owner.node_id:
                raise ValueError(
                    "Resolved API route returned diagnostics for another node"
                )
            snapshots[owner.node_id] = snapshot
        return snapshots

    def _wait_for_vision_media_idle(
        self, owners: list[ClusterApiOwner]
    ) -> dict[str, VisionMediaDiagnosticsSnapshot]:
        """Wait briefly for vision transfer and retained-media cleanup."""

        deadline = time.monotonic() + 10.0
        snapshots = self._capture_vision_media_diagnostics(owners)
        while time.monotonic() < deadline:
            if all(_vision_media_snapshot_is_idle(item) for item in snapshots.values()):
                return snapshots
            time.sleep(0.1)
            snapshots = self._capture_vision_media_diagnostics(owners)
        return snapshots

    def _capture_data_plane_diagnostics(
        self, owners: list[ClusterApiOwner]
    ) -> dict[str, DataPlaneDiagnosticsSnapshot]:
        """Read one DATA snapshot from every selected owner."""

        snapshots: dict[str, DataPlaneDiagnosticsSnapshot] = {}
        for owner in owners:
            with self._client_for_url(owner.base_url) as owner_client:
                snapshot = owner_client.get_data_plane_diagnostics()
            if snapshot.node_id != owner.node_id:
                raise ValueError(
                    "Resolved API route returned diagnostics for another node"
                )
            snapshots[owner.node_id] = snapshot
        return snapshots

    def _wait_for_data_plane_idle(
        self, owners: list[ClusterApiOwner]
    ) -> dict[str, DataPlaneDiagnosticsSnapshot]:
        """Wait briefly for terminal delivery and egress queue cleanup."""

        deadline = time.monotonic() + 10.0
        snapshots = self._capture_data_plane_diagnostics(owners)
        while time.monotonic() < deadline:
            if all(
                snapshot.active_streams == 0
                and snapshot.active_stream_queues == 0
                and snapshot.queue_depth == 0
                for snapshot in snapshots.values()
            ):
                return snapshots
            time.sleep(0.1)
            snapshots = self._capture_data_plane_diagnostics(owners)
        return snapshots

    def _run_audio_transcription_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
    ) -> TestResult:
        """Run an STT request against a configured local audio fixture."""

        issues: list[Issue] = []
        output = ""
        elapsed = 0.0
        if test.input_audio_path is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="audio_transcription test requires input_audio_path",
                )
            )
        else:
            audio_path = _resolve_audio_input_path(test.input_audio_path)
            try:
                audio = audio_path.read_bytes()
                execution = client.audio_transcription(
                    model_id=model_id,
                    audio=audio,
                    filename=audio_path.name,
                    media_type=(
                        test.input_audio_mime_type
                        or _guess_audio_media_type(audio_path)
                    ),
                    response_format=test.transcription_response_format,
                    language=test.transcription_language,
                    prompt=test.prompt,
                )
            except (OSError, SkulkApiError, TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Audio transcription request failed",
                        evidence={"error": str(exc)},
                    )
                )
            else:
                elapsed = execution.elapsed_s
                output = execution.text
                issues.extend(
                    _score_output(
                        model_id,
                        test.name,
                        execution.text,
                        test.success,
                    )
                )
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=elapsed,
                output_chars=len(output),
                generated_chars=len(output),
            ),
            issues=issues,
        )

    def _run_streaming_audio_transcription_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
    ) -> TestResult:
        """Run complete and early-close uploaded-audio transcription streams."""

        issues: list[Issue] = []
        execution: StreamingAudioTranscriptionExecution | None = None
        cancellation: StreamingAudioTranscriptionExecution | None = None
        artifact_path: Path | None = None
        fixture_artifact_path: Path | None = None
        secondary_placement: tuple[str, PlacementResult] | None = None
        audio: bytes | None = None
        filename = "streaming-transcription.wav"
        media_type = "audio/wav"
        if test.input_audio_path is not None:
            audio_path = _resolve_audio_input_path(test.input_audio_path)
            try:
                audio = audio_path.read_bytes()
                filename = audio_path.name
                media_type = test.input_audio_mime_type or _guess_audio_media_type(audio_path)
            except OSError as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Streaming transcription fixture could not be read",
                        evidence={"error": str(exc)},
                    )
                )
        elif spec is None or report is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message=(
                        "audio_transcription_streaming requires an input fixture "
                        "or execution RunSpec/report"
                    ),
                )
            )
        else:
            tts_model_id = test.speech_synthesis_model_id or _first_tts_model_id(
                client.list_models(), exclude_model_id=model_id
            )
            if tts_model_id is None:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="No TTS model found for streaming STT fixture",
                    )
                )
            else:
                try:
                    placement = self._ensure_model_placed(
                        client, tts_model_id, spec, report
                    )
                    if placement is None or not placement.ready:
                        raise ValueError("TTS fixture model did not become ready")
                    secondary_placement = (tts_model_id, placement)
                    _append_unique_placement(report, placement)
                    speech = client.audio_speech(
                        model_id=tts_model_id,
                        input_text=_expanded_prompt(test),
                        response_format="wav",
                        voice=test.speech_voice,
                        speed=test.speech_speed,
                        **_speech_generation_kwargs(test),
                    )
                    source_audio = speech.audio
                    issues.extend(
                        _score_audio_output(
                            tts_model_id,
                            test.name,
                            source_audio,
                            test.success,
                            response_format="wav",
                            media_type=speech.media_type,
                        )
                    )
                    pcm16, sample_rate = _pcm16_from_wav(
                        source_audio,
                        target_sample_rate=24_000,
                    )
                    audio = _wav_from_pcm16(pcm16, sample_rate=sample_rate)
                    fixture_artifact_path = _audio_artifact_path(
                        artifact_dir,
                        model_id,
                        f"{test.name}-input",
                        repetition,
                        "wav",
                        audio,
                    )
                except (
                    OSError,
                    SkulkApiError,
                    httpx.HTTPError,
                    TypeError,
                    ValueError,
                ) as exc:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Streaming transcription fixture generation failed",
                            evidence={"error": str(exc)},
                        )
                    )

        if audio is not None:
            try:
                if test.transcription_cancel_after_deltas > 0:
                    cancellation = client.streaming_audio_transcription(
                        model_id=model_id,
                        audio=audio,
                        filename=filename,
                        media_type=media_type,
                        language=test.transcription_language,
                        prompt=test.prompt,
                        cancel_after_deltas=test.transcription_cancel_after_deltas,
                    )
                    if not cancellation.canceled:
                        raise ValueError("stream cancellation probe reached no delta")
                execution = client.streaming_audio_transcription(
                    model_id=model_id,
                    audio=audio,
                    filename=filename,
                    media_type=media_type,
                    language=test.transcription_language,
                    prompt=test.prompt,
                )
            except (OSError, SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="Streaming audio transcription request failed",
                        evidence={"error": str(exc)},
                    )
                )
            else:
                issues.extend(
                    _score_output(model_id, test.name, execution.text, test.success)
                )
                if execution.transcript_deltas < test.success.min_transcript_deltas:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Streaming transcription emitted too few deltas",
                            evidence={
                                "actual": execution.transcript_deltas,
                                "minimum": test.success.min_transcript_deltas,
                            },
                        )
                    )
                required_terminals = {
                    "transcription.completed",
                    "transcription.usage",
                }
                missing = sorted(required_terminals - set(execution.event_types))
                if missing:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Streaming transcription omitted terminal events",
                            evidence={"missing": missing},
                        )
                    )
                timeline = {
                    "model_id": model_id,
                    "input_bytes": execution.input_bytes,
                    "text": execution.text,
                    "first_transcript_s": execution.first_transcript_s,
                    "elapsed_s": execution.elapsed_s,
                    "event_arrival_s": execution.event_arrival_s,
                    "events": execution.events,
                    "input_audio_artifact": (
                        None
                        if fixture_artifact_path is None
                        else str(fixture_artifact_path)
                    ),
                    "cancellation_probe": (
                        None
                        if cancellation is None
                        else {
                            "elapsed_s": cancellation.elapsed_s,
                            "event_arrival_s": cancellation.event_arrival_s,
                            "events": cancellation.events,
                            "canceled": cancellation.canceled,
                        }
                    ),
                }
                artifact_path = maybe_write_artifact(
                    artifact_dir,
                    (
                        f"{slugify(model_id)}--{slugify(test.name)}--"
                        f"rep-{repetition}.json"
                    ),
                    json.dumps(timeline, indent=2, sort_keys=True),
                )
        if (
            secondary_placement is not None
            and spec is not None
            and report is not None
            and not spec.retain_instances
        ):
            secondary_model_id, placement = secondary_placement
            if secondary_model_id != model_id and placement.created_by_harness:
                torn_down = self._teardown_harness_instances(
                    client,
                    secondary_model_id,
                    placement.instance_id,
                    report,
                    protected_instance_ids=frozenset(
                        placement.protected_instance_ids
                    ),
                )
                if spec.delete_staged_models and torn_down:
                    self._evict_staged_model(client, secondary_model_id, report)
                if writer is not None:
                    writer.write(report)
        output = execution.text if execution is not None else ""
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=execution.elapsed_s if execution is not None else 0.0,
                ttft_s=(
                    execution.first_transcript_s if execution is not None else None
                ),
                output_chars=len(output),
                generated_chars=len(output),
                chunks=(
                    execution.transcript_deltas if execution is not None else 0
                ),
            ),
            issues=issues,
            artifact_path=artifact_path,
        )

    def _run_realtime_transcription_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
    ) -> TestResult:
        """Generate speech, then exercise realtime STT across selected API owners."""

        secondary_placements: list[tuple[str, PlacementResult]] = []
        try:
            return self._run_realtime_transcription_test_inner(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
                artifact_dir=artifact_dir,
                spec=spec,
                report=report,
                secondary_placements=secondary_placements,
            )
        finally:
            if spec is not None and report is not None and not spec.retain_instances:
                for secondary_model_id, placement in secondary_placements:
                    if (
                        secondary_model_id == model_id
                        or not placement.created_by_harness
                    ):
                        continue
                    torn_down = self._teardown_harness_instances(
                        client,
                        secondary_model_id,
                        placement.instance_id,
                        report,
                        protected_instance_ids=frozenset(
                            placement.protected_instance_ids
                        ),
                    )
                    if spec.delete_staged_models and torn_down:
                        self._evict_staged_model(client, secondary_model_id, report)
                if secondary_placements and writer is not None:
                    writer.write(report)

    def _run_realtime_transcription_test_inner(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        secondary_placements: list[tuple[str, PlacementResult]],
    ) -> TestResult:
        """Execute the realtime workload after secondary-placement ownership setup."""

        issues: list[Issue] = []
        if spec is None or report is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="realtime_transcription requires an execution RunSpec/report",
                )
            )
            return _realtime_transcription_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )
        if test.audio_response_format != "wav":
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="realtime_transcription requires audio_response_format: wav",
                )
            )
            return _realtime_transcription_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        tts_model_id = test.speech_synthesis_model_id or _first_tts_model_id(
            client.list_models(), exclude_model_id=model_id
        )
        if tts_model_id is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="No TTS model found for realtime transcription fixture",
                )
            )
            return _realtime_transcription_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )

        try:
            tts_placement = self._ensure_model_placed(
                client,
                tts_model_id,
                spec,
                report,
            )
        except (
            OSError,
            SkulkApiError,
            httpx.HTTPError,
            TypeError,
            ValueError,
        ) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Realtime transcription roundtrip failed",
                    evidence={
                        "error": str(exc),
                        "speech_synthesis_model_id": tts_model_id,
                    },
                )
            )
            return _realtime_transcription_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )
        if tts_placement is not None:
            secondary_placements.append((tts_model_id, tts_placement))
        if tts_placement is None or not tts_placement.ready:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="TTS model could not be made ready for realtime fixture",
                    evidence={"speech_synthesis_model_id": tts_model_id},
                )
            )
            return _realtime_transcription_result(
                model_id=model_id,
                test=test,
                repetition=repetition,
                issues=issues,
            )
        _append_unique_placement(report, tts_placement)

        response_model_id: str | None = None
        response_tts_model_id: str | None = None
        if test.kind in {"realtime_conversation", "fabric_speech_chain"}:
            catalog = client.list_models()
            response_tts_model_id = (
                test.realtime_response_tts_model_id or tts_model_id
            )
            response_model_id = test.realtime_response_model_id or _first_chat_model_id(
                catalog,
                exclude_model_ids={model_id, tts_model_id, response_tts_model_id},
            )
            if response_model_id is None or response_tts_model_id is None:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message=(
                            "conversational realtime requires response chat "
                            "and TTS model IDs"
                        ),
                    )
                )
                return _realtime_transcription_result(
                    model_id=model_id,
                    test=test,
                    repetition=repetition,
                    issues=issues,
                )
            for participant_model_id in (response_model_id, response_tts_model_id):
                if participant_model_id in {model_id, tts_model_id}:
                    continue
                try:
                    participant_placement = self._ensure_model_placed(
                        client,
                        participant_model_id,
                        spec,
                        report,
                    )
                except (
                    OSError,
                    SkulkApiError,
                    httpx.HTTPError,
                    TypeError,
                    ValueError,
                ) as exc:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Fabric speech participant placement failed",
                            evidence={
                                "participant_model_id": participant_model_id,
                                "error": str(exc),
                            },
                        )
                    )
                    return _realtime_transcription_result(
                        model_id=model_id,
                        test=test,
                        repetition=repetition,
                        issues=issues,
                    )
                if participant_placement is not None:
                    secondary_placements.append(
                        (participant_model_id, participant_placement)
                    )
                if participant_placement is None or not participant_placement.ready:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Fabric speech participant was not ready",
                            evidence={"participant_model_id": participant_model_id},
                        )
                    )
                    return _realtime_transcription_result(
                        model_id=model_id,
                        test=test,
                        repetition=repetition,
                        issues=issues,
                    )
                _append_unique_placement(report, participant_placement)

        artifact_path: Path | None = None
        metadata_path: Path | None = None
        started_at = time.monotonic()
        executions: list[RealtimeTranscriptionExecution] = []
        owner_records: list[dict[str, object]] = []
        cancellation_execution: RealtimeTranscriptionExecution | None = None
        diagnostic_owners: list[ClusterApiOwner] = []
        diagnostics_before: dict[str, ProviderCapabilityDiagnosticsSnapshot] = {}
        diagnostics_after: dict[str, ProviderCapabilityDiagnosticsSnapshot] = {}
        try:
            speech = client.audio_speech(
                model_id=tts_model_id,
                input_text=_expanded_prompt(test),
                response_format="wav",
                voice=test.speech_voice,
                speed=test.speech_speed,
                **_speech_generation_kwargs(test),
            )
            issues.extend(
                _score_audio_output(
                    tts_model_id,
                    test.name,
                    speech.audio,
                    test.success,
                    response_format="wav",
                    media_type=speech.media_type,
                )
            )
            pcm16, sample_rate = _pcm16_from_wav(
                speech.audio,
                target_sample_rate=24_000,
            )
            realtime_fixture = _wav_from_pcm16(pcm16, sample_rate=sample_rate)
            artifact_path = _audio_artifact_path(
                artifact_dir,
                model_id,
                test.name,
                repetition,
                "wav",
                realtime_fixture,
            )
            owners, serving_node_id = self._select_speech_owners(
                client,
                model_id,
                test,
                issues,
            )
            if not owners:
                return _realtime_transcription_result(
                    model_id=model_id,
                    test=test,
                    repetition=repetition,
                    issues=issues,
                    elapsed_s=time.monotonic() - started_at,
                    artifact_path=artifact_path,
                )

            if (
                test.realtime_assert_provider_diagnostics
                or test.realtime_cancel_after_frames > 0
            ):
                diagnostic_owners = client.get_cluster_api_owners()
                diagnostics_before = self._capture_provider_diagnostics(
                    diagnostic_owners
                )

            if test.realtime_cancel_after_frames > 0:
                cancel_owner = owners[-1]
                with self._client_for_url(cancel_owner.base_url) as owner_client:
                    cancellation_execution = owner_client.realtime_transcription(
                        model_id=model_id,
                        pcm16=pcm16,
                        sample_rate=sample_rate,
                        frame_duration_ms=test.realtime_frame_duration_ms,
                        pace_audio=test.realtime_pace_audio,
                        cancel_after_frames=test.realtime_cancel_after_frames,
                        fabric_chain=test.kind == "fabric_speech_chain",
                        response_model_id=response_model_id,
                        response_tts_model_id=response_tts_model_id,
                        response_voice=test.speech_voice,
                        response_max_output_tokens=test.max_tokens,
                        response_enable_thinking=test.enable_thinking,
                        server_vad=(
                            test.realtime_server_vad
                            or test.kind == "realtime_conversation"
                        ),
                        turn_count=test.realtime_turn_count,
                        barge_in=test.realtime_barge_in,
                    )
                if not cancellation_execution.canceled:
                    issues.append(
                        Issue(
                            severity="error",
                            model_id=model_id,
                            test_name=test.name,
                            message="Realtime disconnect probe did not cancel its session",
                        )
                    )
                self._wait_for_realtime_cancellation_release(
                    diagnostic_owners,
                    before=diagnostics_before,
                )

            for owner_index, owner in enumerate(owners, start=1):
                role = (
                    "serving_local"
                    if owner.node_id == serving_node_id
                    else "remote_owner"
                )
                with self._client_for_url(owner.base_url) as owner_client:
                    execution = self._run_realtime_transcription_after_release(
                        owner_client,
                        model_id=model_id,
                        pcm16=pcm16,
                        sample_rate=sample_rate,
                        frame_duration_ms=test.realtime_frame_duration_ms,
                        pace_audio=test.realtime_pace_audio,
                        fabric_chain=test.kind == "fabric_speech_chain",
                        response_model_id=response_model_id,
                        response_tts_model_id=response_tts_model_id,
                        response_voice=test.speech_voice,
                        response_max_output_tokens=test.max_tokens,
                        response_enable_thinking=test.enable_thinking,
                        server_vad=(
                            test.realtime_server_vad
                            or test.kind == "realtime_conversation"
                        ),
                        turn_count=test.realtime_turn_count,
                        barge_in=test.realtime_barge_in,
                    )
                executions.append(execution)
                owner_label = f"owner-{owner_index}-{role}"
                if test.kind in {
                    "realtime_conversation",
                    "fabric_speech_chain",
                } and execution.response_audio:
                    _audio_artifact_path(
                        artifact_dir,
                        response_tts_model_id or "fabric-response-tts",
                        f"{test.name}-{owner_label}-response",
                        repetition,
                        "mp3",
                        execution.response_audio,
                    )
                issues.extend(
                    _score_realtime_transcription(
                        model_id=model_id,
                        test_name=f"{test.name}-{owner_label}",
                        execution=execution,
                        criteria=test.success,
                    )
                )
                if test.kind == "fabric_speech_chain":
                    issues.extend(
                        _score_fabric_speech_chain(
                            model_id=model_id,
                            test_name=f"{test.name}-{owner_label}",
                            execution=execution,
                            criteria=test.success,
                        )
                    )
                if test.kind == "realtime_conversation":
                    issues.extend(
                        _score_realtime_conversation(
                            model_id=model_id,
                            test_name=f"{test.name}-{owner_label}",
                            execution=execution,
                            turn_count=test.realtime_turn_count,
                            require_barge_in=test.realtime_barge_in,
                            criteria=test.success,
                        )
                    )
                owner_records.append(
                    _sanitized_realtime_execution(owner_label, execution)
                )

            diagnostics_records: list[dict[str, object]] = []
            if test.realtime_assert_provider_diagnostics:
                diagnostics_after = self._wait_for_provider_diagnostics(
                    diagnostic_owners,
                    before=diagnostics_before,
                    expected_completed=sum(
                        execution.provider_sessions for execution in executions
                    ),
                    expected_cancelled=(1 if cancellation_execution is not None else 0),
                )
                diagnostic_issues, diagnostics_records = (
                    _score_realtime_provider_diagnostics(
                        model_id=model_id,
                        test_name=test.name,
                        owners=diagnostic_owners,
                        serving_node_id=serving_node_id,
                        before=diagnostics_before,
                        after=diagnostics_after,
                        successful_sessions=executions,
                        cancellation_session=cancellation_execution,
                    )
                )
                issues.extend(diagnostic_issues)

            if artifact_path is not None:
                metadata_path = _realtime_metadata_artifact_path(
                    artifact_path,
                    speech_synthesis_model_id=tts_model_id,
                    response_model_id=response_model_id,
                    response_tts_model_id=response_tts_model_id,
                    sample_rate=sample_rate,
                    frame_duration_ms=test.realtime_frame_duration_ms,
                    paced=test.realtime_pace_audio,
                    sessions=owner_records,
                    cancellation=(
                        _sanitized_realtime_execution(
                            "disconnect-probe",
                            cancellation_execution,
                        )
                        if cancellation_execution is not None
                        else None
                    ),
                    provider_diagnostics=diagnostics_records,
                )
        except (OSError, SkulkApiError, TypeError, ValueError, wave.Error) as exc:
            error_detail = exc.body if isinstance(exc, SkulkApiError) else str(exc)
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Realtime transcription roundtrip failed",
                    evidence={
                        "error": error_detail,
                        "speech_synthesis_model_id": tts_model_id,
                    },
                )
            )

        elapsed_s = time.monotonic() - started_at
        transcripts = [execution.text for execution in executions]
        assistant_outputs = [
            execution.assistant_text
            for execution in executions
            if execution.assistant_text
        ]
        output = "\n".join((*transcripts, *assistant_outputs))
        if metadata_path is not None:
            output = f"{output}\nrealtime_metadata={metadata_path}".strip()
        first_transcripts = [
            execution.first_transcript_s
            for execution in executions
            if execution.first_transcript_s is not None
        ]
        return TestResult(
            model_id=model_id,
            test_name=test.name,
            repetition=repetition,
            passed=not any(issue.severity == "error" for issue in issues),
            output_text=output,
            metrics=GenerationMetrics(
                elapsed_s=elapsed_s,
                ttft_s=(
                    statistics.median(first_transcripts) if first_transcripts else None
                ),
                output_chars=sum(len(text) for text in transcripts),
                generated_chars=sum(len(text) for text in transcripts),
                chunks=sum(execution.transcript_deltas + 1 for execution in executions),
            ),
            issues=issues,
            artifact_path=artifact_path,
        )

    def _capture_provider_diagnostics(
        self,
        owners: list[ClusterApiOwner],
    ) -> dict[str, ProviderCapabilityDiagnosticsSnapshot]:
        """Capture realtime STT provider counters from reachable API nodes."""

        snapshots: dict[str, ProviderCapabilityDiagnosticsSnapshot] = {}
        for owner in owners:
            with self._client_for_url(owner.base_url) as owner_client:
                snapshot = owner_client.get_provider_capability_diagnostics(
                    "stt.realtime@1.0.0"
                )
            if snapshot.node_id != owner.node_id:
                raise ValueError(
                    "Resolved API route returned provider diagnostics for another node"
                )
            snapshots[owner.node_id] = snapshot
        return snapshots

    def _wait_for_provider_diagnostics(
        self,
        owners: list[ClusterApiOwner],
        *,
        before: dict[str, ProviderCapabilityDiagnosticsSnapshot],
        expected_completed: int,
        expected_cancelled: int,
    ) -> dict[str, ProviderCapabilityDiagnosticsSnapshot]:
        """Wait for realtime provider terminal counters and queue gauges to settle."""

        deadline = time.monotonic() + 10.0
        snapshots = self._capture_provider_diagnostics(owners)
        while time.monotonic() < deadline:
            completed = sum(
                snapshot.completed_streams
                - before.get(snapshot.node_id, snapshot).completed_streams
                for snapshot in snapshots.values()
            )
            cancelled = sum(
                snapshot.cancellation_requests
                - before.get(snapshot.node_id, snapshot).cancellation_requests
                for snapshot in snapshots.values()
            )
            drained = all(
                snapshot.active_streams
                <= before.get(snapshot.node_id, snapshot).active_streams
                and snapshot.input_queue_depth
                <= before.get(snapshot.node_id, snapshot).input_queue_depth
                for snapshot in snapshots.values()
            )
            if (
                completed >= expected_completed
                and cancelled >= expected_cancelled
                and drained
            ):
                return snapshots
            time.sleep(0.1)
            snapshots = self._capture_provider_diagnostics(owners)
        return snapshots

    @staticmethod
    def _run_realtime_transcription_after_release(
        client: SkulkClient,
        *,
        model_id: str,
        pcm16: bytes,
        sample_rate: int,
        frame_duration_ms: int,
        pace_audio: bool,
        fabric_chain: bool = False,
        response_model_id: str | None = None,
        response_tts_model_id: str | None = None,
        response_voice: str | None = None,
        response_max_output_tokens: int | None = None,
        response_enable_thinking: bool | None = None,
        server_vad: bool = False,
        turn_count: int = 1,
        barge_in: bool = False,
    ) -> RealtimeTranscriptionExecution:
        """Retry only transient admission races before realtime audio is accepted."""

        deadline = time.monotonic() + 10.0
        while True:
            try:
                return client.realtime_transcription(
                    model_id=model_id,
                    pcm16=pcm16,
                    sample_rate=sample_rate,
                    frame_duration_ms=frame_duration_ms,
                    pace_audio=pace_audio,
                    fabric_chain=fabric_chain,
                    response_model_id=response_model_id,
                    response_tts_model_id=response_tts_model_id,
                    response_voice=response_voice,
                    response_max_output_tokens=response_max_output_tokens,
                    response_enable_thinking=response_enable_thinking,
                    server_vad=server_vad,
                    turn_count=turn_count,
                    barge_in=barge_in,
                )
            except SkulkApiError as exc:
                message = str(exc).casefold()
                retryable = (
                    "all realtime stt runners for this model are already busy"
                    in message
                    or "1013 (try again later)" in message
                )
                remaining = deadline - time.monotonic()
                if not retryable or remaining <= 0:
                    raise
                time.sleep(min(0.1, remaining))

    def _wait_for_realtime_cancellation_release(
        self,
        owners: list[ClusterApiOwner],
        *,
        before: dict[str, ProviderCapabilityDiagnosticsSnapshot],
    ) -> dict[str, ProviderCapabilityDiagnosticsSnapshot]:
        """Wait until a disconnect probe releases its provider stream slot."""

        deadline = time.monotonic() + 10.0
        snapshots = self._capture_provider_diagnostics(owners)
        while True:
            cancellation_observed = (
                sum(
                    snapshot.cancellation_requests
                    - before.get(snapshot.node_id, snapshot).cancellation_requests
                    for snapshot in snapshots.values()
                )
                >= 1
            )
            drained = all(
                snapshot.active_streams
                <= before.get(snapshot.node_id, snapshot).active_streams
                and snapshot.stream_slots_in_use
                <= before.get(snapshot.node_id, snapshot).stream_slots_in_use
                and snapshot.input_queue_depth
                <= before.get(snapshot.node_id, snapshot).input_queue_depth
                for snapshot in snapshots.values()
            )
            if cancellation_observed and drained:
                return snapshots
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "realtime disconnect probe did not release provider capacity"
                )
            time.sleep(0.1)
            snapshots = self._capture_provider_diagnostics(owners)

    def _run_speech_roundtrip_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
        translate_to_english: bool = False,
    ) -> TestResult:
        """Generate speech with a TTS model, then transcribe it with an STT model."""

        issues: list[Issue] = []
        output = ""
        elapsed = 0.0
        artifact_path: Path | None = None
        word_error_rate: float | None = None
        if spec is None or report is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="speech_roundtrip requires an execution RunSpec/report",
                )
            )
            return _speech_result(
                model_id,
                test.name,
                repetition,
                output,
                elapsed,
                issues,
                artifact_path=artifact_path,
            )

        transcription_model_id: str | None = None
        stt_placement: PlacementResult | None = None
        try:
            transcription_model_id = test.transcription_model_id
            if transcription_model_id is None:
                transcription_model_id = (
                    _first_translation_model_id(
                        client.list_models(), exclude_model_id=model_id
                    )
                    if translate_to_english
                    else _first_stt_model_id(
                        client.list_models(), exclude_model_id=model_id
                    )
                )
            if transcription_model_id is None:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="No STT model found for speech roundtrip",
                    )
                )
                return _speech_result(
                    model_id,
                    test.name,
                    repetition,
                    output,
                    elapsed,
                    issues,
                    artifact_path=artifact_path,
                )
            stt_placement = self._ensure_model_placed(
                client, transcription_model_id, spec, report
            )
            if stt_placement is None or not stt_placement.ready:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test.name,
                        message="STT model could not be made ready for speech roundtrip",
                        evidence={"transcription_model_id": transcription_model_id},
                    )
                )
                return _speech_result(
                    model_id,
                    test.name,
                    repetition,
                    output,
                    elapsed,
                    issues,
                    artifact_path=artifact_path,
                )
            _append_unique_placement(report, stt_placement)

            speech = client.audio_speech(
                model_id=model_id,
                input_text=_expanded_prompt(test),
                response_format=test.audio_response_format,
                voice=test.speech_voice,
                speed=test.speech_speed,
                **_speech_generation_kwargs(test),
            )
            elapsed = speech.elapsed_s
            issues.extend(
                _score_audio_output(
                    model_id,
                    test.name,
                    speech.audio,
                    test.success,
                    response_format=test.audio_response_format,
                    media_type=speech.media_type,
                )
            )
            artifact_path = _audio_artifact_path(
                artifact_dir,
                model_id,
                test.name,
                repetition,
                speech.response_format,
                speech.audio,
            )
            transcription_method = (
                client.audio_translation
                if translate_to_english
                else client.audio_transcription
            )
            transcript = transcription_method(
                model_id=transcription_model_id,
                audio=speech.audio,
                filename=f"{slugify(test.name)}.{test.audio_response_format}",
                media_type=speech.media_type
                or _audio_media_type_for_format(test.audio_response_format),
                response_format=test.transcription_response_format,
                language=test.transcription_language,
                prompt=None,
            )
            elapsed += transcript.elapsed_s
            output = transcript.text
            issues.extend(
                _score_output(
                    model_id,
                    test.name,
                    transcript.text,
                    test.success,
                )
            )
            if not translate_to_english:
                word_error_rate = _word_error_rate(
                    _expanded_prompt(test), transcript.text
                )
                issues.extend(
                    _score_transcript_fidelity(
                        model_id,
                        test.name,
                        reference=_expanded_prompt(test),
                        transcript=transcript.text,
                        criteria=test.success,
                    )
                )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Speech roundtrip request failed",
                    evidence={
                        "error": str(exc),
                        "transcription_model_id": transcription_model_id,
                    },
                )
            )
        finally:
            if (
                stt_placement is not None
                and transcription_model_id is not None
                and stt_placement.created_by_harness
                and transcription_model_id != model_id
                and not spec.retain_instances
            ):
                torn_down = self._teardown_harness_instances(
                    client,
                    transcription_model_id,
                    stt_placement.instance_id,
                    report,
                    protected_instance_ids=frozenset(
                        stt_placement.protected_instance_ids
                    ),
                )
                if spec.delete_staged_models and torn_down:
                    self._evict_staged_model(client, transcription_model_id, report)
                if writer is not None:
                    writer.write(report)
        return _speech_result(
            model_id,
            test.name,
            repetition,
            output,
            elapsed,
            issues,
            artifact_path=artifact_path,
            word_error_rate=word_error_rate,
        )

    def _run_speech_reference_roundtrip_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        artifact_dir: Path,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
    ) -> TestResult:
        """Generate a donor voice clip and use it to condition a TTS request."""

        issues: list[Issue] = []
        elapsed = 0.0
        output = ""
        output_artifact: Path | None = None
        word_error_rate: float | None = None
        donor_model_id = test.reference_model_id
        donor_placement: PlacementResult | None = None
        transcription_model_id = test.transcription_model_id
        stt_placement: PlacementResult | None = None
        if spec is None or report is None or donor_model_id is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message=(
                        "speech_reference_roundtrip requires an execution "
                        "RunSpec/report and reference_model_id"
                    ),
                )
            )
            return _speech_result(
                model_id,
                test.name,
                repetition,
                "",
                elapsed,
                issues,
                artifact_path=output_artifact,
            )

        reference_text = test.reference_text or test.prompt
        try:
            donor_placement = self._ensure_model_placed(
                client, donor_model_id, spec, report
            )
            if donor_placement is None or not donor_placement.ready:
                raise ValueError("Reference TTS model could not be made ready")
            _append_unique_placement(report, donor_placement)

            if (
                transcription_model_id is None
                and test.success.max_word_error_rate is not None
            ):
                transcription_model_id = _first_stt_model_id(
                    client.list_models(), exclude_model_id=model_id
                )
                if transcription_model_id is None:
                    raise ValueError(
                        "No STT model found for reference-roundtrip semantic scoring"
                    )
            if transcription_model_id is not None:
                stt_placement = self._ensure_model_placed(
                    client, transcription_model_id, spec, report
                )
                if stt_placement is None or not stt_placement.ready:
                    raise ValueError(
                        "STT model could not be made ready for reference roundtrip"
                    )
                _append_unique_placement(report, stt_placement)

            reference = client.audio_speech(
                model_id=donor_model_id,
                input_text=reference_text,
                response_format="wav",
                **_speech_generation_kwargs(test),
            )
            elapsed += reference.elapsed_s
            issues.extend(
                _score_audio_output(
                    model_id,
                    test.name,
                    reference.audio,
                    test.success,
                    response_format="wav",
                    media_type=reference.media_type,
                )
            )
            _audio_artifact_path(
                artifact_dir,
                model_id,
                f"{test.name}-reference",
                repetition,
                "wav",
                reference.audio,
            )

            conditioned = client.audio_speech(
                model_id=model_id,
                input_text=_expanded_prompt(test),
                response_format=test.audio_response_format,
                voice=test.speech_voice,
                speed=test.speech_speed,
                **_speech_generation_kwargs(test),
                reference_audio=reference.audio,
                reference_audio_filename="reference.wav",
                reference_audio_media_type=reference.media_type or "audio/wav",
                reference_text=reference_text,
            )
            elapsed += conditioned.elapsed_s
            issues.extend(
                _score_audio_output(
                    model_id,
                    test.name,
                    conditioned.audio,
                    test.success,
                    response_format=test.audio_response_format,
                    media_type=conditioned.media_type,
                )
            )
            output_artifact = _audio_artifact_path(
                artifact_dir,
                model_id,
                test.name,
                repetition,
                conditioned.response_format,
                conditioned.audio,
            )
            if transcription_model_id is not None:
                transcript = client.audio_transcription(
                    model_id=transcription_model_id,
                    audio=conditioned.audio,
                    filename=(f"{slugify(test.name)}.{test.audio_response_format}"),
                    media_type=conditioned.media_type
                    or _audio_media_type_for_format(test.audio_response_format),
                    response_format=test.transcription_response_format,
                    language=test.transcription_language,
                    prompt=None,
                )
                elapsed += transcript.elapsed_s
                output = transcript.text
                issues.extend(_score_output(model_id, test.name, output, test.success))
                word_error_rate = _word_error_rate(_expanded_prompt(test), output)
                issues.extend(
                    _score_transcript_fidelity(
                        model_id,
                        test.name,
                        reference=_expanded_prompt(test),
                        transcript=output,
                        criteria=test.success,
                    )
                )
        except (SkulkApiError, httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test.name,
                    message="Reference-conditioned speech request failed",
                    evidence={
                        "error": str(exc),
                        "reference_model_id": donor_model_id,
                        "transcription_model_id": transcription_model_id,
                    },
                )
            )
        finally:
            if (
                donor_placement is not None
                and donor_placement.created_by_harness
                and donor_model_id != model_id
                and not spec.retain_instances
            ):
                torn_down = self._teardown_harness_instances(
                    client,
                    donor_model_id,
                    donor_placement.instance_id,
                    report,
                    protected_instance_ids=frozenset(
                        donor_placement.protected_instance_ids
                    ),
                )
                if spec.delete_staged_models and torn_down:
                    self._evict_staged_model(client, donor_model_id, report)
                if writer is not None:
                    writer.write(report)
            if (
                stt_placement is not None
                and transcription_model_id is not None
                and stt_placement.created_by_harness
                and transcription_model_id not in {model_id, donor_model_id}
                and not spec.retain_instances
            ):
                torn_down = self._teardown_harness_instances(
                    client,
                    transcription_model_id,
                    stt_placement.instance_id,
                    report,
                    protected_instance_ids=frozenset(
                        stt_placement.protected_instance_ids
                    ),
                )
                if spec.delete_staged_models and torn_down:
                    self._evict_staged_model(client, transcription_model_id, report)
                if writer is not None:
                    writer.write(report)
        return _speech_result(
            model_id,
            test.name,
            repetition,
            output,
            elapsed,
            issues,
            artifact_path=output_artifact,
            word_error_rate=word_error_rate,
        )

    def _model_set(self, name: str) -> ModelSet:
        try:
            return self.model_sets[name]
        except KeyError as exc:
            raise ValueError(f"Unknown model set {name!r}") from exc

    def _test_set(self, name: str) -> TestSet:
        try:
            return self.test_sets[name]
        except KeyError as exc:
            raise ValueError(f"Unknown test set {name!r}") from exc


def _select_catalog_models(
    catalog: list[dict[str, object]], selector: ModelSelector
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    regex = re.compile(selector.id_regex, re.IGNORECASE) if selector.id_regex else None
    for model in catalog:
        model_id = _model_id_from_catalog_entry(model)
        if not model_id:
            continue
        if (
            selector.family
            and str(model.get("family") or "").lower() != selector.family.lower()
        ):
            continue
        if (
            selector.id_contains
            and selector.id_contains.lower() not in model_id.lower()
        ):
            continue
        if regex and regex.search(model_id) is None:
            continue
        if selector.tags_any and not _has_any(model.get("tags"), selector.tags_any):
            continue
        if selector.tasks_any and not _has_any(model.get("tasks"), selector.tasks_any):
            continue
        if selector.capabilities_any and not _has_any_capability(
            model, selector.capabilities_any
        ):
            continue
        if selector.served_spec_types_any and _served_spec_type(model) not in {
            value.lower() for value in selector.served_spec_types_any
        }:
            continue
        if (
            selector.require_audio_streaming
            and not _catalog_entry_supports_streaming_audio(model)
        ):
            continue
        if (
            selector.require_audio_realtime
            and not _catalog_entry_supports_realtime_audio(model)
        ):
            continue
        selected.append(model)
        if selector.max_models is not None and len(selected) >= selector.max_models:
            break
    return selected


def _served_spec_type(model: dict[str, object]) -> str:
    runtime = model.get("runtime")
    if isinstance(runtime, dict):
        value = runtime.get("served_spec_type")
        if isinstance(value, str):
            return value.lower()
    value = model.get("served_spec_type")
    return value.lower() if isinstance(value, str) else ""


def _is_retryable_placement_giveup(placement: PlacementResult) -> bool:
    return placement.created_by_harness and placement.instance_id is None


def _not_ready_message(placement: PlacementResult) -> str:
    """Human-readable cause for a placement that never became ready.

    Keyed off the model-scoped readiness wait's ``unavailable_reason`` so the
    report names the actual failure mode (re-placement lost, load failure, slow
    load, refusal) instead of a single generic string.
    """
    reason = placement.unavailable_reason
    if reason == "never_appeared":
        return (
            "Requested placement never appeared in cluster state; treating as a "
            "placement refusal/give-up"
        )
    if reason == "disappeared_without_replacement":
        return (
            "Instance was torn down and not re-placed within the appearance "
            "window; stopped the readiness wait instead of polling a vanished "
            "instance id"
        )
    if reason == "load_failed" or placement.terminal_failure:
        return "Instance runner failed while loading"
    if reason == "ready_timeout":
        return (
            "Instance was placed but never reached a dispatchable runner within "
            "the readiness timeout; see master logs"
        )
    if reason == "churn":
        return (
            "Placement churned: the cluster kept replacing the loading instance "
            "and no lineage became ready before the total readiness ceiling; "
            "the churn itself is the failure (readiness_transitions lists every "
            "observed placement)"
        )
    return (
        "Instance was placed but never became ready (torn down by cluster "
        "recovery, or load failed); see master logs"
    )


def _speech_generation_kwargs(test: PromptTest) -> _SpeechGenerationKwargs:
    """Forward only generation controls explicitly configured for a TTS test."""

    configured = test.model_fields_set
    tts_max_tokens = (
        test.max_tokens
        if "max_tokens" in configured
        and test.kind
        not in {
            "audio_speech_pressure",
            "realtime_conversation",
            "fabric_speech_chain",
        }
        else None
    )
    return {
        "temperature": test.temperature if "temperature" in configured else None,
        "top_p": test.top_p if "top_p" in configured else None,
        "max_tokens": tts_max_tokens,
    }


def _expanded_prompt(test: PromptTest) -> str:
    return test.prompt * test.prompt_repetitions


def _messages_for_test(test: PromptTest) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if test.system:
        messages.append({"role": "system", "content": test.system})
    prompt = _expanded_prompt(test)
    if not test.images:
        messages.append({"role": "user", "content": prompt})
        return messages
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for image in test.images:
        image_url: dict[str, object] = {"url": _prompt_image_url(image)}
        if image.detail is not None:
            image_url["detail"] = image.detail
        content.append({"type": "image_url", "image_url": image_url})
    messages.append({"role": "user", "content": content})
    return messages


def _prompt_image_url(image: PromptImage) -> str:
    """Return a request-ready URL, encoding a local fixture when configured."""
    if image.url is not None:
        return image.url
    assert image.input_path is not None
    image_path = (
        image.input_path
        if image.input_path.is_absolute()
        else Path.cwd() / image.input_path
    )
    media_type = image.media_type or mimetypes.guess_type(image_path)[0]
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(f"Unable to determine image media type for {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _resolve_audio_input_path(path: Path) -> Path:
    """Resolve an audio fixture path from the harness working directory."""

    return path if path.is_absolute() else Path.cwd() / path


def _guess_audio_media_type(path: Path) -> str:
    """Best-effort MIME type for a local audio fixture."""

    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "audio/wav"


def _audio_media_type_for_format(response_format: str) -> str:
    """Return the OpenAI audio media type for an encoded response format."""

    if response_format == "mp3":
        return "audio/mpeg"
    if response_format == "wav":
        return "audio/wav"
    if response_format == "flac":
        return "audio/flac"
    if response_format == "ogg":
        return "audio/ogg"
    if response_format == "opus":
        return "audio/opus"
    return "application/octet-stream"


def _pcm16_from_wav(
    audio: bytes,
    *,
    target_sample_rate: int | None = None,
) -> tuple[bytes, int]:
    """Extract mono PCM16 and optionally resample it to a required rate."""

    with wave.open(io.BytesIO(audio), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        compression = reader.getcomptype()
        frames = reader.readframes(reader.getnframes())
    if channels < 1:
        raise ValueError("TTS WAV did not contain an audio channel")
    if sample_width != 2 or compression != "NONE":
        raise ValueError("Realtime fixture requires uncompressed 16-bit PCM WAV")
    frame_width = channels * sample_width
    if not frames or len(frames) % frame_width != 0:
        raise ValueError("TTS WAV contained incomplete PCM frames")
    if channels == 1:
        mono = frames
    else:
        downmixed = bytearray()
        for frame_offset in range(0, len(frames), frame_width):
            total = 0
            for channel in range(channels):
                offset = frame_offset + channel * sample_width
                total += int.from_bytes(
                    frames[offset : offset + sample_width],
                    byteorder="little",
                    signed=True,
                )
            sample = max(-32768, min(32767, round(total / channels)))
            downmixed.extend(sample.to_bytes(2, byteorder="little", signed=True))
        mono = bytes(downmixed)

    if target_sample_rate is None or target_sample_rate == sample_rate:
        return mono, sample_rate
    if target_sample_rate <= 0:
        raise ValueError("Target sample rate must be positive")
    return (
        _resample_pcm16_linear(
            mono,
            input_sample_rate=sample_rate,
            output_sample_rate=target_sample_rate,
        ),
        target_sample_rate,
    )


def _resample_pcm16_linear(
    pcm16: bytes,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> bytes:
    """Resample complete mono PCM16 with duration-preserving interpolation."""

    if input_sample_rate <= 0 or output_sample_rate <= 0:
        raise ValueError("PCM sample rates must be positive")
    if not pcm16 or len(pcm16) % 2 != 0:
        raise ValueError("PCM16 input must contain complete samples")
    if input_sample_rate == output_sample_rate:
        return pcm16

    samples = [
        int.from_bytes(pcm16[offset : offset + 2], "little", signed=True)
        for offset in range(0, len(pcm16), 2)
    ]
    output_count = max(
        1,
        (len(samples) * output_sample_rate + input_sample_rate // 2)
        // input_sample_rate,
    )
    output = bytearray()
    for output_index in range(output_count):
        position_numerator = output_index * input_sample_rate
        left_index, fraction_numerator = divmod(
            position_numerator,
            output_sample_rate,
        )
        if left_index >= len(samples) - 1:
            sample = samples[-1]
        else:
            left = samples[left_index]
            right = samples[left_index + 1]
            sample = round(
                (
                    left * (output_sample_rate - fraction_numerator)
                    + right * fraction_numerator
                )
                / output_sample_rate
            )
        clipped = max(-32768, min(32767, sample))
        output.extend(clipped.to_bytes(2, "little", signed=True))
    return bytes(output)


def _wav_from_pcm16(pcm16: bytes, *, sample_rate: int) -> bytes:
    """Wrap mono little-endian PCM16 in a WAV container."""

    if sample_rate <= 0:
        raise ValueError("WAV sample rate must be positive")
    if not pcm16 or len(pcm16) % 2 != 0:
        raise ValueError("PCM16 input must contain complete samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16)
    return output.getvalue()


def _score_audio_output(
    model_id: str,
    test_name: str,
    audio: bytes,
    criteria: SuccessCriteria,
    *,
    response_format: str,
    media_type: str,
) -> list[Issue]:
    """Score a binary TTS response without trying to inspect waveform semantics."""

    issues: list[Issue] = []
    if len(audio) < criteria.min_audio_bytes:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Audio response shorter than required minimum "
                    f"({len(audio)} < {criteria.min_audio_bytes} bytes)"
                ),
                evidence={"audio_bytes": len(audio)},
            )
        )
    if media_type and not media_type.startswith("audio/"):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=f"Audio response had non-audio media type {media_type!r}",
            )
        )
    if (
        response_format == "wav"
        and audio
        and not (audio.startswith(b"RIFF") and b"WAVE" in audio[:16])
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="WAV response did not contain a RIFF/WAVE header",
            )
        )
    return issues


def _transcript_words(text: str) -> list[str]:
    """Normalize speech text into case-folded words for semantic comparison."""

    return re.findall(r"\w+(?:['’]\w+)?", text.casefold(), flags=re.UNICODE)


def _word_error_rate(reference: str, transcript: str) -> float:
    """Return Levenshtein word error rate for a reference and transcript."""

    reference_words = _transcript_words(reference)
    transcript_words = _transcript_words(transcript)
    if not reference_words:
        return 0.0 if not transcript_words else float(len(transcript_words))

    previous_row = list(range(len(transcript_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]
        for transcript_index, transcript_word in enumerate(transcript_words, start=1):
            substitution_cost = int(reference_word != transcript_word)
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[transcript_index] + 1,
                    previous_row[transcript_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1] / len(reference_words)


def _score_transcript_fidelity(
    model_id: str,
    test_name: str,
    *,
    reference: str,
    transcript: str,
    criteria: SuccessCriteria,
) -> list[Issue]:
    """Score roundtrip transcript semantics when a WER ceiling is configured."""

    if criteria.max_word_error_rate is None:
        return []
    word_error_rate = _word_error_rate(reference, transcript)
    if word_error_rate <= criteria.max_word_error_rate:
        return []
    return [
        Issue(
            severity="error",
            model_id=model_id,
            test_name=test_name,
            message=(
                "Speech roundtrip word error rate exceeded the maximum "
                f"({word_error_rate:.3f} > {criteria.max_word_error_rate:.3f})"
            ),
            evidence={
                "word_error_rate": word_error_rate,
                "max_word_error_rate": criteria.max_word_error_rate,
                "reference_words": len(_transcript_words(reference)),
                "transcript_words": len(_transcript_words(transcript)),
            },
        )
    ]


def _score_streaming_audio_output(
    model_id: str,
    test_name: str,
    execution: AudioSpeechExecution,
    criteria: SuccessCriteria,
) -> list[Issue]:
    """Score streaming transport evidence for a TTS response."""

    issues: list[Issue] = []
    if execution.chunks < criteria.min_stream_chunks:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Streaming response yielded fewer chunks than required "
                    f"({execution.chunks} < {criteria.min_stream_chunks})"
                ),
                evidence={"chunks": execution.chunks},
            )
        )
    if criteria.max_first_byte_s is not None and (
        execution.first_byte_s is None
        or execution.first_byte_s > criteria.max_first_byte_s
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Streaming first byte exceeded the configured limit",
                evidence={
                    "first_byte_s": execution.first_byte_s,
                    "max_first_byte_s": criteria.max_first_byte_s,
                },
            )
        )
    stream_span_s = _stream_span_s(execution.chunk_arrival_s)
    if criteria.min_stream_span_s > 0 and (
        stream_span_s is None or stream_span_s < criteria.min_stream_span_s
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Streaming response did not span the configured duration",
                evidence={
                    "stream_span_s": stream_span_s,
                    "min_stream_span_s": criteria.min_stream_span_s,
                },
            )
        )
    return issues


def _score_realtime_transcription(
    *,
    model_id: str,
    test_name: str,
    execution: RealtimeTranscriptionExecution,
    criteria: SuccessCriteria,
) -> list[Issue]:
    """Score realtime transcript semantics, deltas, and first-result latency."""

    issues = _score_output(model_id, test_name, execution.text, criteria)
    if execution.canceled:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Successful realtime session was unexpectedly cancelled",
            )
        )
    if execution.input_bytes <= 0 or execution.input_frames <= 0:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime session sent no PCM audio frames",
            )
        )
    if execution.transcript_deltas < criteria.min_transcript_deltas:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime session emitted too few transcript deltas",
                evidence={
                    "transcript_deltas": execution.transcript_deltas,
                    "min_transcript_deltas": criteria.min_transcript_deltas,
                },
            )
        )
    if criteria.max_first_byte_s is not None and (
        execution.first_transcript_s is None
        or execution.first_transcript_s > criteria.max_first_byte_s
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime first transcript exceeded the configured limit",
                evidence={
                    "first_transcript_s": execution.first_transcript_s,
                    "max_first_byte_s": criteria.max_first_byte_s,
                },
            )
        )
    return issues


def _score_fabric_speech_chain(
    *,
    model_id: str,
    test_name: str,
    execution: RealtimeTranscriptionExecution,
    criteria: SuccessCriteria,
) -> list[Issue]:
    """Require completed assistant text and synthesized audio from one chain."""

    issues: list[Issue] = []
    if execution.response_status != "completed":
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Fabric speech response did not complete",
                evidence={"response_status": execution.response_status},
            )
        )
    if not execution.assistant_text.strip():
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Fabric speech chain returned no assistant text",
            )
        )
    if len(execution.response_audio) < criteria.min_audio_bytes:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Fabric speech chain returned too little response audio",
                evidence={
                    "audio_bytes": len(execution.response_audio),
                    "min_audio_bytes": criteria.min_audio_bytes,
                },
            )
        )
    if execution.response_audio_chunks <= 0:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Fabric speech chain emitted no response audio chunks",
            )
        )
    return issues


def _score_realtime_conversation(
    *,
    model_id: str,
    test_name: str,
    execution: RealtimeTranscriptionExecution,
    turn_count: int,
    require_barge_in: bool,
    criteria: SuccessCriteria,
) -> list[Issue]:
    """Require multi-turn VAD, response, audio, and interruption evidence."""

    issues: list[Issue] = []
    counts = {
        "transcripts": len(execution.transcripts),
        "responses": len(execution.response_statuses),
        "speech_started": execution.speech_started_events,
        "speech_stopped": execution.speech_stopped_events,
        "provider_sessions": execution.provider_sessions,
    }
    missing = {
        name: {"actual": value, "minimum": turn_count}
        for name, value in counts.items()
        if value < turn_count
    }
    if missing:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime conversation did not complete every VAD turn",
                evidence={"missing_turn_evidence": missing},
            )
        )
    if not execution.response_statuses or execution.response_statuses[-1] != "completed":
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime conversation final response did not complete",
                evidence={"response_statuses": execution.response_statuses},
            )
        )
    if not execution.assistant_turns or not execution.assistant_turns[-1].strip():
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime conversation returned no final assistant text",
            )
        )
    final_audio = (
        execution.response_audio_turns[-1] if execution.response_audio_turns else b""
    )
    if len(final_audio) < criteria.min_audio_bytes:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime conversation returned too little final response audio",
                evidence={
                    "audio_bytes": len(final_audio),
                    "min_audio_bytes": criteria.min_audio_bytes,
                },
            )
        )
    if require_barge_in and (
        not execution.barge_in_sent
        or "cancelled" not in execution.response_statuses[:-1]
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Realtime conversation did not prove response barge-in",
                evidence={
                    "barge_in_sent": execution.barge_in_sent,
                    "response_statuses": execution.response_statuses,
                },
            )
        )
    return issues


def _stream_span_s(chunk_arrival_s: list[float]) -> float | None:
    """Return elapsed seconds between first and last streamed chunks."""

    if len(chunk_arrival_s) < 2:
        return None
    return max(0.0, chunk_arrival_s[-1] - chunk_arrival_s[0])


def _speech_result(
    model_id: str,
    test_name: str,
    repetition: int,
    output: str,
    elapsed: float,
    issues: list[Issue],
    *,
    artifact_path: Path | None = None,
    word_error_rate: float | None = None,
) -> TestResult:
    """Build a standard result for speech endpoint tests."""

    return TestResult(
        model_id=model_id,
        test_name=test_name,
        repetition=repetition,
        passed=not any(issue.severity == "error" for issue in issues),
        output_text=output,
        metrics=GenerationMetrics(
            elapsed_s=elapsed,
            output_chars=len(output),
            generated_chars=len(output),
            word_error_rate=word_error_rate,
        ),
        issues=issues,
        artifact_path=artifact_path,
    )


def _realtime_transcription_result(
    *,
    model_id: str,
    test: PromptTest,
    repetition: int,
    issues: list[Issue],
    elapsed_s: float = 0.0,
    artifact_path: Path | None = None,
) -> TestResult:
    """Build a failed realtime result for pre-workload validation errors."""

    return TestResult(
        model_id=model_id,
        test_name=test.name,
        repetition=repetition,
        passed=False,
        output_text="realtime transcription did not complete",
        metrics=GenerationMetrics(elapsed_s=elapsed_s),
        issues=issues,
        artifact_path=artifact_path,
    )


def _fmt_optional_float(value: float | None) -> str:
    """Format an optional float for compact result text."""

    return "None" if value is None else f"{value:.3f}"


def _append_unique_placement(report: RunReport, placement: PlacementResult) -> None:
    """Record a secondary placement once, avoiding duplicate report rows."""

    if not any(
        existing.model_id == placement.model_id
        and existing.instance_id == placement.instance_id
        for existing in report.placements
    ):
        report.placements.append(placement)


def _first_stt_model_id(
    catalog: list[dict[str, object]], *, exclude_model_id: str
) -> str | None:
    """Pick the first catalog model advertising speech-to-text support."""

    for model in catalog:
        model_id = _model_id_from_catalog_entry(model)
        if not model_id or model_id == exclude_model_id:
            continue
        if _catalog_entry_supports_stt(model):
            return model_id
    return None


def _first_translation_model_id(
    catalog: list[dict[str, object]], *, exclude_model_id: str
) -> str | None:
    """Pick the first catalog model explicitly advertising speech translation."""

    for model in catalog:
        model_id = _model_id_from_catalog_entry(model)
        if not model_id or model_id == exclude_model_id:
            continue
        if _catalog_entry_supports_translation(model):
            return model_id
    return None


def _first_tts_model_id(
    catalog: list[dict[str, object]], *, exclude_model_id: str
) -> str | None:
    """Pick the first catalog model advertising text-to-speech support."""

    for model in catalog:
        model_id = _model_id_from_catalog_entry(model)
        if not model_id or model_id == exclude_model_id:
            continue
        if _catalog_entry_supports_tts(model):
            return model_id
    return None


def _first_chat_model_id(
    catalog: list[dict[str, object]], *, exclude_model_ids: set[str]
) -> str | None:
    """Pick the first catalog model advertising text-generation support."""

    for model in catalog:
        model_id = _model_id_from_catalog_entry(model)
        if not model_id or model_id in exclude_model_ids:
            continue
        if _has_any(model.get("tasks"), ["TextGeneration"]):
            return model_id
    return None


def _catalog_entry_supports_tts(model: dict[str, object]) -> bool:
    """Return whether a `/models` entry looks usable as a TTS model."""

    resolved = model.get("resolved_capabilities")
    if isinstance(resolved, dict) and bool(
        resolved.get("supports_speech_synthesis")
        or resolved.get("supports_audio_output")
    ):
        return True
    audio = model.get("audio")
    if isinstance(audio, dict) and str(audio.get("kind") or "").lower() == "tts":
        return True
    return (
        _has_any(model.get("tags"), ["tts"])
        or _has_any_capability(model, ["tts"])
        or _has_any(model.get("tasks"), ["TextToSpeech"])
    )


def _catalog_entry_supports_stt(model: dict[str, object]) -> bool:
    """Return whether a `/models` entry looks usable as an STT model."""

    resolved = model.get("resolved_capabilities")
    if isinstance(resolved, dict) and (
        bool(resolved.get("supports_transcription"))
        or bool(resolved.get("supports_speech_translation"))
    ):
        return True
    audio = model.get("audio")
    if isinstance(audio, dict) and str(audio.get("kind") or "").lower() == "stt":
        return True
    return (
        _has_any(model.get("tags"), ["stt"])
        or _has_any_capability(model, ["stt"])
        or _has_any(model.get("tasks"), ["SpeechToText", "SpeechTranslation"])
    )


def _catalog_entry_supports_translation(model: dict[str, object]) -> bool:
    """Return whether a `/models` entry explicitly supports speech translation."""

    resolved = model.get("resolved_capabilities")
    if isinstance(resolved, dict) and bool(resolved.get("supports_speech_translation")):
        return True
    audio = model.get("audio")
    if isinstance(audio, dict) and audio.get("supports_translation") is True:
        return True
    return _has_any(model.get("tasks"), ["SpeechTranslation"])


def _catalog_entry_supports_streaming_audio(model: dict[str, object]) -> bool:
    """Return whether a `/models` entry declares streaming speech output."""

    audio = model.get("audio")
    if isinstance(audio, dict) and audio.get("supports_streaming") is True:
        return True
    resolved = model.get("resolved_capabilities")
    return isinstance(resolved, dict) and bool(
        resolved.get("supports_audio_output_streaming")
        or resolved.get("supports_speech_synthesis_streaming")
    )


def _catalog_entry_supports_realtime_audio(model: dict[str, object]) -> bool:
    """Return whether a `/models` entry truthfully declares realtime STT."""

    audio = model.get("audio")
    if isinstance(audio, dict) and (
        str(audio.get("kind") or "").lower() == "stt"
        and audio.get("supports_streaming") is True
        and audio.get("supports_realtime") is True
    ):
        return True
    resolved = model.get("resolved_capabilities")
    return isinstance(resolved, dict) and bool(
        resolved.get("supports_transcription")
        and resolved.get("supports_realtime_audio")
    )


def _model_id_from_catalog_entry(model: dict[str, object]) -> str:
    for key in ("model_id", "hugging_face_id", "id", "name"):
        value = model.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


_DEFERRED_PLACEMENT_MESSAGE_PREFIXES = (
    "No usable placement preview found before execution",
    "Placement request failed",
    "Timed out waiting for placed model to appear in cluster state",
)


def _clear_deferred_placement_issues(report: RunReport, model_id: str) -> None:
    """Drop provisional placement-refusal issues after a successful retry."""

    report.issues = [
        issue
        for issue in report.issues
        if not (
            issue.model_id == model_id
            and issue.severity == "error"
            and issue.message.startswith(_DEFERRED_PLACEMENT_MESSAGE_PREFIXES)
        )
    ]


def _store_registry_entries(
    registry: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if registry is None:
        return []
    entries = registry.get("entries")
    if isinstance(entries, list):
        return [item for item in entries if isinstance(item, dict)]
    models = registry.get("models")
    if isinstance(models, list):
        return [item for item in models if isinstance(item, dict)]
    data = registry.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _has_any(raw: object, needles: list[str]) -> bool:
    if not isinstance(raw, list):
        return False
    haystack = {str(item).lower() for item in raw}
    return any(needle.lower() in haystack for needle in needles)


def _has_any_capability(model: dict[str, object], needles: list[str]) -> bool:
    """Match capability selectors across legacy lists and resolved API flags."""

    capabilities = _normalized_string_set(model.get("capabilities"))
    wanted = {_normalized_capability(needle) for needle in needles}
    if capabilities & wanted:
        return True
    if capabilities & _TTS_CAPABILITY_ALIASES and wanted & _TTS_CAPABILITY_ALIASES:
        return True
    if capabilities & _STT_CAPABILITY_ALIASES and wanted & _STT_CAPABILITY_ALIASES:
        return True
    audio_kind = _audio_kind(model)
    if audio_kind in _TTS_CAPABILITY_ALIASES and wanted & _TTS_CAPABILITY_ALIASES:
        return True
    if audio_kind in _STT_CAPABILITY_ALIASES and wanted & _STT_CAPABILITY_ALIASES:
        return True
    resolved = model.get("resolved_capabilities")
    if not isinstance(resolved, dict):
        return False
    if wanted & _TTS_CAPABILITY_ALIASES and (
        bool(resolved.get("supports_speech_synthesis"))
        or bool(resolved.get("supports_audio_output"))
    ):
        return True
    if wanted & _STT_CAPABILITY_ALIASES and (
        bool(resolved.get("supports_transcription"))
        or bool(resolved.get("supports_speech_translation"))
    ):
        return True
    return "vision" in wanted and bool(resolved.get("supports_image_input"))


def _audio_kind(model: dict[str, object]) -> str:
    audio = model.get("audio")
    if not isinstance(audio, dict):
        return ""
    return _normalized_capability(str(audio.get("kind") or ""))


def _normalized_capability(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _normalized_string_set(raw: object) -> set[str]:
    if not isinstance(raw, list):
        return set()
    return {_normalized_capability(str(item)) for item in raw}


_TTS_CAPABILITY_ALIASES = frozenset(
    {"tts", "texttospeech", "speechsynthesis", "speechoutput", "audiooutput"}
)
_STT_CAPABILITY_ALIASES = frozenset(
    {
        "stt",
        "speechtotext",
        "transcription",
        "speechtranscription",
        "speechtranslation",
    }
)


def _preview_node_count(preview: dict[str, object]) -> int:
    instance = preview.get("instance")
    parsed = unwrap_tagged(instance)
    if parsed is None:
        return 0
    _tag, body = parsed
    assignments = body.get("shardAssignments")
    if not isinstance(assignments, dict):
        return 0
    node_to_runner = assignments.get("nodeToRunner")
    return len(node_to_runner) if isinstance(node_to_runner, dict) else 0


def _placement_from_preview(
    model_id: str, preview: dict[str, object]
) -> PlacementResult:
    instance = preview.get("instance")
    parsed = unwrap_tagged(instance)
    if parsed is None:
        return PlacementResult(model_id=model_id)
    tag, body = parsed
    assignments = body.get("shardAssignments")
    node_ids: list[str] = []
    runner_ids: list[str] = []
    if isinstance(assignments, dict):
        node_to_runner = assignments.get("nodeToRunner")
        runner_to_shard = assignments.get("runnerToShard")
        if isinstance(node_to_runner, dict):
            node_ids = list(node_to_runner)
        if isinstance(runner_to_shard, dict):
            runner_ids = list(runner_to_shard)
    return PlacementResult(
        model_id=model_id,
        node_ids=node_ids,
        runner_ids=runner_ids,
        sharding=str(preview.get("sharding") or ""),
        instance_meta=tag,
    )


def _score_output(
    model_id: str,
    test_name: str,
    text: str,
    criteria: SuccessCriteria,
    *,
    tool_calls: list[ToolCallRecord] | None = None,
    logprob_tokens: int = 0,
    reasoning_text: str = "",
    wall_tps: float | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    tool_calls = tool_calls or []
    generated_chars = len(text) + len(reasoning_text)
    if criteria.require_logprobs and logprob_tokens <= 0:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Expected per-token logprobs but the stream returned none "
                    "(the serving runner/build did not produce logprobs)"
                ),
            )
        )
    # A native OpenAI tool_call is a coherent assistant completion even when the
    # visible content field is intentionally empty. Other content-specific gates
    # (required substrings, regexes, list counts) still check visible text.
    if len(text) < criteria.min_chars and not tool_calls:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=f"Output shorter than required minimum ({len(text)} < {criteria.min_chars})",
            )
        )
    if criteria.min_generated_chars and generated_chars < criteria.min_generated_chars:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Generated text shorter than required minimum "
                    f"({generated_chars} < {criteria.min_generated_chars} chars "
                    "across content + reasoning)"
                ),
                evidence={
                    "content_chars": len(text),
                    "reasoning_chars": len(reasoning_text),
                },
            )
        )
    for substring in criteria.required_substrings:
        if substring.lower() not in text.lower():
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"Output missing required substring {substring!r}",
                )
            )
    if criteria.min_list_items and _list_item_count(text) < criteria.min_list_items:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Output had too few structured list items "
                    f"({_list_item_count(text)} < {criteria.min_list_items})"
                ),
            )
        )
    # Forbidden substrings are checked against the visible content and, when
    # forbid_in_reasoning is set, the separated reasoning channel too -- a leaked
    # control marker (e.g. Gemma 4's literal '<|channel>') is a regression
    # whichever channel it surfaces in.
    forbidden_haystack = text
    if criteria.forbid_in_reasoning and reasoning_text:
        forbidden_haystack = f"{text}\n{reasoning_text}"
    for substring in criteria.forbidden_substrings:
        if substring.lower() in forbidden_haystack.lower():
            where = (
                "output/reasoning"
                if criteria.forbid_in_reasoning and reasoning_text
                else "output"
            )
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"{where} contained forbidden substring {substring!r}",
                )
            )
    if len(reasoning_text) < criteria.min_reasoning_chars:
        # A reasoning model must surface its thinking in the SEPARATED reasoning
        # channel. Too little reasoning_content means the parser swallowed the
        # thought into content (or lost the split) -- the Gemma 4 served channel
        # parser regression this guards against.
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Separated reasoning shorter than required minimum "
                    f"({len(reasoning_text)} < {criteria.min_reasoning_chars} chars) "
                    "-- reasoning not split into its own channel"
                ),
            )
        )
    # A throughput floor makes a SILENT speculative/MTP fallback visible: the text
    # stays correct, only the decode rate drops. The gate is skipped when the run
    # produced no measurable rate (e.g. an empty/failed generation, which the
    # content checks already flag).
    if (
        criteria.min_wall_tps is not None
        and wall_tps is not None
        and wall_tps < criteria.min_wall_tps
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    f"Decode throughput {wall_tps:.1f} tok/s below required "
                    f"floor {criteria.min_wall_tps:.1f} tok/s -- speculative/"
                    "MTP decoding may have silently fallen back"
                ),
                evidence={"wall_tps": wall_tps, "min_wall_tps": criteria.min_wall_tps},
            )
        )
    for pattern in criteria.required_regexes:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"Output did not match required regex {pattern!r}",
                )
            )
    code_block = extract_first_code_block(text)
    if (
        criteria.min_code_block_chars
        and len(code_block or "") < criteria.min_code_block_chars
    ):
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Output did not include a large enough fenced code block",
            )
        )
    if criteria.require_html_artifact:
        candidate = code_block or text
        if "<canvas" not in candidate.lower() or "<script" not in candidate.lower():
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Output did not look like a playable single-file HTML/canvas artifact",
                )
            )
    if criteria.in_order_integers:
        target = criteria.in_order_integers
        # Integers in [1, target] in emission order must be strictly ascending.
        # A delivery reorder ("12" before "8", or a displaced "26") breaks
        # monotonicity; this is the order-blindness that let a data-plane
        # transposition pass presence-only checks (Skulk #297, fixed by #301).
        emitted = [
            value
            for value in (int(match) for match in re.findall(r"\d+", text))
            if 1 <= value <= target
        ]
        if len(emitted) < target // 2:
            # The model never really produced the sequence (e.g. it refused or
            # answered something else), which is a content failure, not a
            # reorder, and the other criteria/min_chars already speak to it.
            pass
        elif any(
            later <= earlier
            for earlier, later in zip(emitted, emitted[1:], strict=False)
        ):
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=(
                        f"Output integers 1..{target} were not in ascending "
                        "order (token/sub-word transposition)"
                    ),
                    evidence={"emitted_sequence": emitted},
                )
            )
    if len(tool_calls) < criteria.min_tool_calls:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "Too few tool calls emitted "
                    f"({len(tool_calls)} < {criteria.min_tool_calls})"
                ),
                evidence={"tool_call_names": [call.name for call in tool_calls]},
            )
        )
    issues.extend(
        _score_expected_tool_calls(
            model_id,
            test_name,
            tool_calls,
            criteria.expected_tool_calls,
        )
    )
    return issues


def _list_item_count(text: str) -> int:
    """Count common Markdown/plaintext list markers in model output."""

    marker = re.compile(
        r"^\s*(?:[-*•‣·–▪]|\d+[.)]|[a-z][.)])\s+",
        flags=re.IGNORECASE,
    )
    return sum(1 for line in text.splitlines() if marker.search(line))


def _score_expected_tool_calls(
    model_id: str,
    test_name: str,
    tool_calls: list[ToolCallRecord],
    expected_tool_calls: list[ExpectedToolCall],
) -> list[Issue]:
    issues: list[Issue] = []
    matched_indexes: set[int] = set()
    for expected in expected_tool_calls:
        match_index = _find_matching_tool_call(
            tool_calls,
            expected,
            ignored_indexes=matched_indexes,
        )
        if match_index is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"Expected tool call {expected.name!r} was not emitted",
                    evidence={
                        "expected": expected.model_dump(mode="json"),
                        "actual_tool_calls": [
                            call.model_dump(mode="json") for call in tool_calls
                        ],
                    },
                )
            )
        else:
            matched_indexes.add(match_index)
    return issues


def _find_matching_tool_call(
    tool_calls: list[ToolCallRecord],
    expected: ExpectedToolCall,
    *,
    ignored_indexes: set[int],
) -> int | None:
    for index, tool_call in enumerate(tool_calls):
        if index in ignored_indexes or tool_call.name != expected.name:
            continue
        if _tool_call_arguments_match(tool_call, expected):
            return index
    return None


def _tool_call_arguments_match(
    tool_call: ToolCallRecord,
    expected: ExpectedToolCall,
) -> bool:
    arguments = tool_call.arguments
    if arguments is None:
        return not (
            expected.required_arguments
            or expected.arguments_contains
            or expected.argument_substrings
        )
    for key in expected.required_arguments:
        if key not in arguments:
            return False
    for key, expected_value in expected.arguments_contains.items():
        if arguments.get(key) != expected_value:
            return False
    for key, expected_substring in expected.argument_substrings.items():
        value = arguments.get(key)
        if expected_substring.lower() not in str(value or "").lower():
            return False
    return True


def _tool_roundtrip_messages(
    base_messages: list[dict[str, object]],
    tool_calls: list[ToolCallRecord],
    tool_mocks: list[ToolMock],
    *,
    model_id: str,
    test_name: str,
    issues: list[Issue],
) -> list[dict[str, object]]:
    mocks_by_name = {mock.name: mock.content for mock in tool_mocks}
    messages = [dict(message) for message in base_messages]
    assistant_tool_calls: list[dict[str, object]] = []
    tool_messages: list[dict[str, object]] = []

    for index, tool_call in enumerate(tool_calls):
        call_id = tool_call.id or f"call-{index}-{slugify(tool_call.name)}"
        assistant_tool_calls.append(
            {
                "id": call_id,
                "index": tool_call.index if tool_call.index is not None else index,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_text,
                },
            }
        )
        content = mocks_by_name.get(tool_call.name)
        if content is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"No mock result configured for tool {tool_call.name!r}",
                    evidence={"tool_call": tool_call.model_dump(mode="json")},
                )
            )
            continue
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_call.name,
                "content": content,
            }
        )

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        }
    )
    messages.extend(tool_messages)
    return messages


def _artifact_path(
    artifact_dir: Path,
    model_id: str,
    test: PromptTest,
    repetition: int,
    execution: ChatExecution,
) -> Path | None:
    if test.kind not in {"artifact", "code"}:
        return None
    code = extract_first_code_block(execution.text) or execution.text
    extension = (
        "html" if "<html" in code.lower() or "<canvas" in code.lower() else "txt"
    )
    filename = (
        f"{slugify(model_id)}--{slugify(test.name)}--rep-{repetition}.{extension}"
    )
    return maybe_write_artifact(artifact_dir, filename, code)


def _audio_artifact_path(
    artifact_dir: Path,
    model_id: str,
    test_name: str,
    repetition: int,
    response_format: str,
    audio: bytes,
) -> Path:
    """Persist generated audio bytes and return the reportable artifact path."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{slugify(model_id)}--{slugify(test_name)}--rep-{repetition}."
        f"{slugify(response_format)}"
    )
    path = artifact_dir / filename
    path.write_bytes(audio)
    return path


def _audio_stream_metadata_path(
    audio_path: Path,
    chunks: int,
    first_byte_s: float | None,
    stream_span_s: float | None,
    chunk_sizes: list[int],
    chunk_arrival_s: list[float],
) -> Path:
    """Persist per-chunk streaming timing evidence next to an audio artifact."""

    path = audio_path.with_suffix(f"{audio_path.suffix}.stream.json")
    path.write_text(
        json.dumps(
            {
                "chunks": chunks,
                "first_byte_s": first_byte_s,
                "stream_span_s": stream_span_s,
                "chunk_sizes": chunk_sizes,
                "chunk_arrival_s": chunk_arrival_s,
            },
            indent=2,
        )
    )
    return path


def _sanitized_realtime_execution(
    owner: str,
    execution: RealtimeTranscriptionExecution,
) -> dict[str, object]:
    """Return media-bounded realtime evidence without node routes or identifiers."""

    return {
        "owner": owner,
        "canceled": execution.canceled,
        "elapsed_s": execution.elapsed_s,
        "first_transcript_s": execution.first_transcript_s,
        "input_bytes": execution.input_bytes,
        "input_frames": execution.input_frames,
        "transcript_chars": len(execution.text),
        "transcript_deltas": execution.transcript_deltas,
        "assistant_chars": len(execution.assistant_text),
        "response_audio_bytes": len(execution.response_audio),
        "response_audio_chunks": execution.response_audio_chunks,
        "response_status": execution.response_status,
        "turns": len(execution.transcripts),
        "assistant_turns": len(execution.assistant_turns),
        "response_statuses": execution.response_statuses,
        "speech_started_events": execution.speech_started_events,
        "speech_stopped_events": execution.speech_stopped_events,
        "provider_sessions": execution.provider_sessions,
        "provider_input_bytes_min": execution.provider_input_bytes_min,
        "barge_in_sent": execution.barge_in_sent,
        "event_types": execution.event_types,
        "events": execution.events,
    }


def _realtime_metadata_artifact_path(
    audio_path: Path,
    *,
    speech_synthesis_model_id: str,
    response_model_id: str | None,
    response_tts_model_id: str | None,
    sample_rate: int,
    frame_duration_ms: int,
    paced: bool,
    sessions: list[dict[str, object]],
    cancellation: dict[str, object] | None,
    provider_diagnostics: list[dict[str, object]],
) -> Path:
    """Persist sanitized realtime protocol, timing, and provider evidence."""

    path = audio_path.with_suffix(f"{audio_path.suffix}.realtime.json")
    path.write_text(
        json.dumps(
            {
                "speech_synthesis_model_id": speech_synthesis_model_id,
                "response_model_id": response_model_id,
                "response_tts_model_id": response_tts_model_id,
                "sample_rate": sample_rate,
                "frame_duration_ms": frame_duration_ms,
                "paced": paced,
                "sessions": sessions,
                "cancellation": cancellation,
                "provider_diagnostics": provider_diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path


_PROVIDER_COUNTER_FIELDS = (
    "admitted_streams",
    "input_frames",
    "input_media_bytes",
    "output_frames",
    "output_media_bytes",
    "completed_streams",
    "failed_streams",
    "cancelled_streams",
    "missing_terminal_streams",
    "cancellation_requests",
)


def _provider_counter_delta(
    before: ProviderCapabilityDiagnosticsSnapshot,
    after: ProviderCapabilityDiagnosticsSnapshot,
) -> dict[str, int]:
    """Subtract cumulative provider counters from same-process snapshots."""

    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in _PROVIDER_COUNTER_FIELDS
    }


def _sanitized_provider_snapshot(
    snapshot: ProviderCapabilityDiagnosticsSnapshot,
) -> dict[str, int]:
    """Return provider diagnostics without node or capability identifiers."""

    values = asdict(snapshot)
    values.pop("node_id")
    values.pop("capability_id")
    return {key: value for key, value in values.items() if isinstance(value, int)}


def _score_realtime_provider_diagnostics(
    *,
    model_id: str,
    test_name: str,
    owners: list[ClusterApiOwner],
    serving_node_id: str | None,
    before: dict[str, ProviderCapabilityDiagnosticsSnapshot],
    after: dict[str, ProviderCapabilityDiagnosticsSnapshot],
    successful_sessions: list[RealtimeTranscriptionExecution],
    cancellation_session: RealtimeTranscriptionExecution | None,
) -> tuple[list[Issue], list[dict[str, object]]]:
    """Require provider lifecycle/media evidence for realtime success and cancel."""

    issues: list[Issue] = []
    records: list[dict[str, object]] = []
    deltas: list[dict[str, int]] = []
    for owner_index, owner in enumerate(owners, start=1):
        before_snapshot = before.get(owner.node_id)
        after_snapshot = after.get(owner.node_id)
        role = "serving_local" if owner.node_id == serving_node_id else "remote_owner"
        label = f"owner-{owner_index}-{role}"
        if before_snapshot is None or after_snapshot is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Provider diagnostics snapshot missing for selected owner",
                    evidence={"owner": label},
                )
            )
            continue
        delta = _provider_counter_delta(before_snapshot, after_snapshot)
        deltas.append(delta)
        records.append(
            {
                "owner": label,
                "before": _sanitized_provider_snapshot(before_snapshot),
                "after": _sanitized_provider_snapshot(after_snapshot),
                "delta": delta,
            }
        )
        negative = {key: value for key, value in delta.items() if value < 0}
        if negative:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Provider diagnostics counters reset during realtime test",
                    evidence={"owner": label, "negative_deltas": negative},
                )
            )
        if (
            after_snapshot.active_streams > before_snapshot.active_streams
            or after_snapshot.input_queue_depth > before_snapshot.input_queue_depth
        ):
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Realtime provider streams or input queues did not drain",
                    evidence={
                        "owner": label,
                        "active_streams_before": before_snapshot.active_streams,
                        "active_streams_after": after_snapshot.active_streams,
                        "input_queue_depth_before": before_snapshot.input_queue_depth,
                        "input_queue_depth_after": after_snapshot.input_queue_depth,
                    },
                )
            )
        anomalies = {
            key: delta[key]
            for key in ("failed_streams", "missing_terminal_streams")
            if delta[key] > 0
        }
        if anomalies:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Realtime provider diagnostics recorded lifecycle anomalies",
                    evidence={"owner": label, "anomalies": anomalies},
                )
            )

    totals = {
        field_name: sum(delta[field_name] for delta in deltas)
        for field_name in _PROVIDER_COUNTER_FIELDS
    }
    successful_provider_sessions = sum(
        execution.provider_sessions for execution in successful_sessions
    )
    expected_sessions = successful_provider_sessions + (
        1 if cancellation_session is not None else 0
    )
    expected_input_bytes = sum(
        execution.provider_input_bytes_min or execution.input_bytes
        for execution in successful_sessions
    ) + (cancellation_session.input_bytes if cancellation_session is not None else 0)
    expected_input_frames = sum(
        execution.input_frames for execution in successful_sessions
    ) + (cancellation_session.input_frames if cancellation_session is not None else 0)

    requirements = {
        "admitted_streams": expected_sessions,
        "completed_streams": successful_provider_sessions,
        "input_media_bytes": expected_input_bytes,
        "input_frames": expected_input_frames,
        "output_frames": successful_provider_sessions * 2,
    }
    if cancellation_session is not None:
        requirements["cancellation_requests"] = 1
    missing = {
        field_name: {"actual": totals[field_name], "minimum": minimum}
        for field_name, minimum in requirements.items()
        if totals[field_name] < minimum
    }
    if missing:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Provider diagnostics did not cover realtime sessions",
                evidence={"missing_counter_evidence": missing},
            )
        )
    return issues, records


_VISION_MEDIA_COUNTER_FIELDS = (
    "local_short_circuits",
    "remote_frames_enqueued",
    "remote_frames_published",
    "remote_frames_dropped",
    "remote_publish_failures",
    "inbound_frames_dropped",
    "idle_stream_reclaims",
    "completed_streams",
    "rejected_streams",
    "expired_streams",
)

_VISION_MEDIA_ANOMALY_FIELDS = (
    "remote_frames_dropped",
    "remote_publish_failures",
    "inbound_frames_dropped",
    "idle_stream_reclaims",
    "rejected_streams",
    "expired_streams",
)

_VISION_MEDIA_LIVE_GAUGE_FIELDS = (
    "active_stream_queues",
    "queue_depth",
    "inbound_payload_queue_depth",
    "inbound_terminal_queue_depth",
    "pending_api_commands",
    "pending_api_bytes",
    "active_api_commands",
    "active_api_bytes",
    "pending_worker_acknowledgements",
    "active_streams",
    "pending_frames",
    "retained_bytes",
    "verified_streams",
    "pending_failures",
)


def _vision_media_snapshot_is_idle(
    snapshot: VisionMediaDiagnosticsSnapshot,
) -> bool:
    """Return whether all request-scoped vision media resources are released."""

    return all(
        getattr(snapshot, field_name) == 0
        for field_name in _VISION_MEDIA_LIVE_GAUGE_FIELDS
    )


def _vision_media_counter_delta(
    before: VisionMediaDiagnosticsSnapshot,
    after: VisionMediaDiagnosticsSnapshot,
) -> dict[str, int]:
    """Subtract cumulative vision media counters from same-process snapshots."""

    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in _VISION_MEDIA_COUNTER_FIELDS
    }


def _score_vision_media_baseline(
    *,
    model_id: str,
    test_name: str,
    owners: list[ClusterApiOwner],
    serving_node_id: str | None,
    serving_node_ids: set[str],
    snapshots: dict[str, VisionMediaDiagnosticsSnapshot],
) -> list[Issue]:
    """Reject a non-idle vision baseline whose request ownership is unknown."""

    issues: list[Issue] = []
    for owner_index, owner in enumerate(owners, start=1):
        snapshot = snapshots.get(owner.node_id)
        if snapshot is None:
            continue
        role = _vision_media_owner_role(
            owner.node_id,
            serving_node_id=serving_node_id,
            serving_node_ids=serving_node_ids,
        )
        live_gauges = {
            field_name: getattr(snapshot, field_name)
            for field_name in _VISION_MEDIA_LIVE_GAUGE_FIELDS
            if getattr(snapshot, field_name) > 0
        }
        if live_gauges:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=(
                        "Vision media diagnostics baseline is not idle; routing "
                        "attribution would be inconclusive"
                    ),
                    evidence={
                        "owner": f"owner-{owner_index}-{role}",
                        "live_gauges": live_gauges,
                    },
                )
            )
    return issues


def _sanitized_vision_media_snapshot(
    snapshot: VisionMediaDiagnosticsSnapshot,
) -> dict[str, int]:
    """Return vision diagnostics without node identity or route details."""

    values = asdict(snapshot)
    values.pop("node_id")
    return {key: value for key, value in values.items() if isinstance(value, int)}


def _vision_media_owner_role(
    node_id: str,
    *,
    serving_node_id: str | None,
    serving_node_ids: set[str],
) -> str:
    """Return a sanitized topology role for one diagnostics owner."""

    if node_id == serving_node_id:
        return "serving_local"
    if node_id in serving_node_ids:
        return "serving_participant"
    return "remote_owner"


def _score_vision_media_diagnostics(
    *,
    model_id: str,
    test_name: str,
    owners: list[ClusterApiOwner],
    request_owners: list[ClusterApiOwner],
    serving_node_id: str | None,
    serving_node_ids: set[str],
    before: dict[str, VisionMediaDiagnosticsSnapshot],
    after: dict[str, VisionMediaDiagnosticsSnapshot],
    successful_requests: int,
) -> tuple[list[Issue], list[dict[str, object]]]:
    """Prove local/remote vision routing, bounded cleanup, and clean outcomes."""

    issues: list[Issue] = []
    records: list[dict[str, object]] = []
    deltas: dict[str, dict[str, int]] = {}
    for owner_index, owner in enumerate(owners, start=1):
        before_snapshot = before.get(owner.node_id)
        after_snapshot = after.get(owner.node_id)
        role = _vision_media_owner_role(
            owner.node_id,
            serving_node_id=serving_node_id,
            serving_node_ids=serving_node_ids,
        )
        label = f"owner-{owner_index}-{role}"
        if before_snapshot is None or after_snapshot is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Vision media diagnostics missing for selected owner",
                    evidence={"owner": label},
                )
            )
            continue
        delta = _vision_media_counter_delta(before_snapshot, after_snapshot)
        deltas[owner.node_id] = delta
        records.append(
            {
                "owner": label,
                "before": _sanitized_vision_media_snapshot(before_snapshot),
                "after": _sanitized_vision_media_snapshot(after_snapshot),
                "delta": delta,
            }
        )
        negative = {key: value for key, value in delta.items() if value < 0}
        if negative:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Vision media counters reset during qualification",
                    evidence={"owner": label, "negative_deltas": negative},
                )
            )
        live_gauges = {
            field_name: getattr(after_snapshot, field_name)
            for field_name in _VISION_MEDIA_LIVE_GAUGE_FIELDS
            if getattr(after_snapshot, field_name) > 0
        }
        if live_gauges:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Vision media resources did not drain after requests",
                    evidence={"owner": label, "live_gauges": live_gauges},
                )
            )
        anomalies = {
            key: delta[key]
            for key in _VISION_MEDIA_ANOMALY_FIELDS
            if delta[key] > 0
        }
        if anomalies:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Vision media diagnostics recorded transfer anomalies",
                    evidence={"owner": label, "anomalies": anomalies},
                )
            )

    local_serving_delta = deltas.get(serving_node_id or "")
    if local_serving_delta is None:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="Serving-node vision media diagnostics were unavailable",
            )
        )
    else:
        if local_serving_delta["local_short_circuits"] <= 0:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Local VLM request did not use the vision fast path",
                )
            )

    for serving_participant_id in sorted(serving_node_ids):
        serving_delta = deltas.get(serving_participant_id)
        if serving_delta is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=(
                        "Vision media diagnostics missing for a serving participant"
                    ),
                )
            )
            continue
        if serving_delta["completed_streams"] < successful_requests:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Vision ingress completion counters missed requests",
                    evidence={
                        "successful_requests": successful_requests,
                        "completed_streams_delta": serving_delta[
                            "completed_streams"
                        ],
                    },
                )
            )

    for owner in request_owners:
        if owner.node_id == serving_node_id:
            continue
        remote_delta = deltas.get(owner.node_id)
        if remote_delta is None:
            continue
        if remote_delta["remote_frames_published"] <= 0:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Remote VLM request did not publish vision media frames",
                )
            )
        elif (
            remote_delta["remote_frames_enqueued"]
            != remote_delta["remote_frames_published"]
        ):
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Remote vision media egress did not publish every frame",
                    evidence={
                        "enqueued": remote_delta["remote_frames_enqueued"],
                        "published": remote_delta["remote_frames_published"],
                    },
                )
            )
    return issues, records


def _vision_media_diagnostics_artifact_path(
    artifact_dir: Path,
    model_id: str,
    test_name: str,
    repetition: int,
    records: list[dict[str, object]],
) -> Path:
    """Persist sanitized local/remote vision transport evidence."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / (
        f"{slugify(model_id)}--{slugify(test_name)}--rep-{repetition}"
        ".vision-media.json"
    )
    path.write_text(json.dumps({"owners": records}, indent=2, sort_keys=True))
    return path


def _vision_data_plane_result(
    *,
    model_id: str,
    test: PromptTest,
    repetition: int,
    issues: list[Issue],
) -> TestResult:
    """Build a pre-workload failure result for vision DATA qualification."""

    return TestResult(
        model_id=model_id,
        test_name=test.name,
        repetition=repetition,
        passed=False,
        output_text="vision DATA qualification did not start",
        metrics=_empty_metrics(),
        issues=issues,
    )


_DATA_PLANE_COUNTER_FIELDS = (
    "started_frames",
    "completed_frames",
    "failed_frames",
    "cancelled_frames",
    "duplicate_frames",
    "out_of_order_frames",
    "skipped_sequences",
    "late_frames",
    "missing_started_streams",
    "missing_terminal_streams",
    "idle_timeouts",
    "transport_failures",
    "local_short_circuits",
    "remote_frames_enqueued",
    "remote_frames_published",
    "remote_frames_dropped",
    "remote_publish_failures",
    "idle_stream_reclaims",
)

_DATA_PLANE_ANOMALY_FIELDS = (
    "failed_frames",
    "cancelled_frames",
    "duplicate_frames",
    "out_of_order_frames",
    "skipped_sequences",
    "late_frames",
    "missing_started_streams",
    "missing_terminal_streams",
    "idle_timeouts",
    "transport_failures",
    "remote_frames_dropped",
    "remote_publish_failures",
    "idle_stream_reclaims",
)


def _data_plane_counter_delta(
    before: DataPlaneDiagnosticsSnapshot,
    after: DataPlaneDiagnosticsSnapshot,
) -> dict[str, int]:
    """Subtract cumulative DATA counters from two same-process snapshots."""

    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in _DATA_PLANE_COUNTER_FIELDS
    }


def _score_data_plane_baseline(
    *,
    model_id: str,
    test_name: str,
    owners: list[ClusterApiOwner],
    serving_node_id: str | None,
    snapshots: dict[str, DataPlaneDiagnosticsSnapshot],
) -> list[Issue]:
    """Reject a contaminated baseline whose workload ownership is unknowable."""

    issues: list[Issue] = []
    for owner_index, owner in enumerate(owners, start=1):
        snapshot = snapshots.get(owner.node_id)
        role = "serving_local" if owner.node_id == serving_node_id else "remote_owner"
        label = f"owner-{owner_index}-{role}"
        if snapshot is None:
            continue
        live_gauges = {
            "active_streams": snapshot.active_streams,
            "active_stream_queues": snapshot.active_stream_queues,
            "queue_depth": snapshot.queue_depth,
        }
        live_gauges = {key: value for key, value in live_gauges.items() if value > 0}
        if not live_gauges:
            continue
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=(
                    "DATA diagnostics baseline is not idle; pressure attribution "
                    "would be inconclusive"
                ),
                evidence={"owner": label, "live_gauges": live_gauges},
            )
        )
    return issues


def _sanitized_data_plane_snapshot(
    snapshot: DataPlaneDiagnosticsSnapshot,
) -> dict[str, int]:
    """Return diagnostics without node identity or environment route details."""

    values = asdict(snapshot)
    values.pop("node_id")
    return {key: value for key, value in values.items() if isinstance(value, int)}


def _score_data_plane_diagnostics(
    *,
    model_id: str,
    test_name: str,
    owners: list[ClusterApiOwner],
    serving_node_id: str | None,
    before: dict[str, DataPlaneDiagnosticsSnapshot],
    after: dict[str, DataPlaneDiagnosticsSnapshot],
    successful_streams: int,
    require_local_remote: bool,
) -> tuple[list[Issue], list[dict[str, object]]]:
    """Require clean lifecycle deltas and prove configured routing activity."""

    issues: list[Issue] = []
    records: list[dict[str, object]] = []
    deltas: dict[str, dict[str, int]] = {}
    for owner_index, owner in enumerate(owners, start=1):
        before_snapshot = before.get(owner.node_id)
        after_snapshot = after.get(owner.node_id)
        role = "serving_local" if owner.node_id == serving_node_id else "remote_owner"
        label = f"owner-{owner_index}-{role}"
        if before_snapshot is None or after_snapshot is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="DATA diagnostics snapshot missing for selected owner",
                    evidence={"owner": label},
                )
            )
            continue
        delta = _data_plane_counter_delta(before_snapshot, after_snapshot)
        deltas[owner.node_id] = delta
        records.append(
            {
                "owner": label,
                "before": _sanitized_data_plane_snapshot(before_snapshot),
                "after": _sanitized_data_plane_snapshot(after_snapshot),
                "delta": delta,
            }
        )
        negative = {key: value for key, value in delta.items() if value < 0}
        if negative:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="DATA diagnostics counters reset during pressure test",
                    evidence={"owner": label, "negative_deltas": negative},
                )
            )
        if (
            after_snapshot.active_streams
            or after_snapshot.active_stream_queues
            or after_snapshot.queue_depth
        ):
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="DATA streams or egress queues did not drain after pressure",
                    evidence={
                        "owner": label,
                        "active_streams": after_snapshot.active_streams,
                        "active_stream_queues": after_snapshot.active_stream_queues,
                        "queue_depth": after_snapshot.queue_depth,
                    },
                )
            )
        anomalies = {
            key: delta[key] for key in _DATA_PLANE_ANOMALY_FIELDS if delta[key] > 0
        }
        if anomalies:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="DATA diagnostics recorded anomalies during successful pressure",
                    evidence={"owner": label, "anomalies": anomalies},
                )
            )

    started = sum(delta["started_frames"] for delta in deltas.values())
    terminal = sum(
        delta["completed_frames"] + delta["failed_frames"] + delta["cancelled_frames"]
        for delta in deltas.values()
    )
    if started < successful_streams or terminal < successful_streams:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message="DATA lifecycle counters did not cover every successful request",
                evidence={
                    "successful_requests": successful_streams,
                    "started_delta": started,
                    "terminal_delta": terminal,
                },
            )
        )

    if require_local_remote:
        serving_delta = deltas.get(serving_node_id or "")
        if serving_delta is None:
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message="Serving-node DATA diagnostics were unavailable",
                )
            )
        else:
            if serving_delta["local_short_circuits"] <= 0:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test_name,
                        message="Deterministic local owner did not use the DATA fast path",
                    )
                )
            if serving_delta["remote_frames_published"] <= 0:
                issues.append(
                    Issue(
                        severity="error",
                        model_id=model_id,
                        test_name=test_name,
                        message="Deterministic remote owner did not use DATA egress",
                    )
                )
    return issues, records


def _data_plane_diagnostics_artifact_path(
    artifact_dir: Path,
    model_id: str,
    test_name: str,
    repetition: int,
    records: list[dict[str, object]],
) -> Path:
    """Persist sanitized pre/post DATA evidence for one pressure result."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / (
        f"{slugify(model_id)}--{slugify(test_name)}--rep-{repetition}.data-plane.json"
    )
    path.write_text(json.dumps({"owners": records}, indent=2, sort_keys=True))
    return path


def _speech_pressure_result(
    *,
    model_id: str,
    test: PromptTest,
    repetition: int,
    issues: list[Issue],
    elapsed_s: float,
) -> TestResult:
    """Build a failed pressure result for pre-workload validation errors."""

    return TestResult(
        model_id=model_id,
        test_name=test.name,
        repetition=repetition,
        passed=False,
        output_text="speech pressure did not start",
        metrics=GenerationMetrics(elapsed_s=elapsed_s),
        issues=issues,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return the ``percentile`` (0-100) value via linear interpolation.

    Used to summarize per-request decode rate and TTFT distributions for the
    concurrent benchmark. Returns ``None`` for an empty sample so callers leave
    the corresponding optional metric unset rather than reporting a fake zero.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _empty_metrics() -> GenerationMetrics:
    return GenerationMetrics(elapsed_s=0.0)


def _run_id(spec: RunSpec) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = spec.run_name or f"{spec.model_set}-{spec.test_set}"
    return f"{stamp}-{slugify(name)}"
