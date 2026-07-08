"""Main planning and execution engine for the Skulk harness."""

from __future__ import annotations

import mimetypes
import re
import time
from collections.abc import Mapping
from pathlib import Path

import httpx

from skulk_test_harness.client import ChatExecution, SkulkApiError, SkulkClient
from skulk_test_harness.fingerprint import gather_fingerprint
from skulk_test_harness.models import (
    ExpectedToolCall,
    GenerationMetrics,
    HarnessConfig,
    Issue,
    ModelRef,
    ModelSelector,
    ModelSet,
    PlacementPolicy,
    PlacementResult,
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
            fingerprint, fp_issues = gather_fingerprint(client, spec, run_reason=spec.mode)
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
            fingerprint, fp_issues = gather_fingerprint(client, spec, run_reason=spec.mode)
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
                # ready is either a cluster-side teardown (recovery killed it;
                # the 30-minute ready-wait then polls a vanished instance) or
                # a genuine load failure. The first pooled-rpc runs dropped
                # this on the floor (no issue, no placement recorded) and read
                # as empty green runs. A retryable give-up defers with a
                # visible warning; a final failure is an error issue.
                retryable = _is_retryable_placement_giveup(placement)
                report.issues.append(
                    Issue(
                        severity=(
                            "warning" if retryable and not deferred_retry else "error"
                        ),
                        model_id=model.model_id,
                        message=(
                            "Instance was placed but never became ready (torn "
                            "down by cluster recovery, or load failed); see "
                            "master logs"
                        ),
                        evidence={"instance_id": placement.instance_id or ""},
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
                    report.results.append(result)
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
                    client, model.model_id, placement.instance_id, report
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
    ) -> bool:
        """Delete every live instance the harness owns for ``model_id``.

        Teardown deletes by instance_id, but the cluster can re-place an
        instance under a *new* id mid-run (failover / re-placement carry-over).
        When that happens, deleting the id we were handed at creation 404s while
        the re-IDed instance is orphaned -- it then starves the next cell and
        reads as "the harness left the old instance running". So we delete the
        original id AND sweep the current state for any instance still serving
        this model. This branch only runs when the harness *created* the
        lineage (``created_by_harness``), so every live instance for the model
        is ours to reap; a pre-existing/reused instance never reaches here.
        """
        target_ids: list[str] = []
        if primary_instance_id:
            target_ids.append(primary_instance_id)
        all_deletes_succeeded = bool(target_ids)
        try:
            for live in client.find_placements_for_model(model_id):
                if live.instance_id and live.instance_id not in target_ids:
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
        store_entries = _store_registry_entries(client.get_store_registry())
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
        # An excluded node must not be reused: without this, a cell that asks to
        # exclude kite4 (to force a GGUF onto kite5) would silently reuse a prior
        # kite4 placement and never exercise the target node.
        if spec.placement.excluded_nodes:
            excluded = set(spec.placement.excluded_nodes)
            existing = [p for p in existing if not excluded.intersection(p.node_ids)]
        if existing and spec.reuse_existing_instances:
            placement = existing[0]
            if placement.instance_id and not placement.ready:
                placement = client.wait_for_instance_ready(
                    placement.instance_id,
                    timeout_s=self.config.placement_ready_timeout_s,
                    poll_interval_s=self.config.poll_interval_s,
                )
                placement = placement.model_copy(update={"reused_existing": True})
            return placement

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

        appear_deadline = time.monotonic() + min(
            self.config.placement_appearance_timeout_s,
            self.config.placement_ready_timeout_s,
        )
        while time.monotonic() < appear_deadline:
            placements = client.find_placements_for_model(model_id)
            if placements:
                placement = placements[0].model_copy(
                    update={"created_by_harness": True, "reused_existing": False}
                )
                if placement.instance_id and not placement.ready:
                    ready_deadline = time.monotonic() + self.config.placement_ready_timeout_s
                    placement = client.wait_for_instance_ready(
                        placement.instance_id,
                        timeout_s=max(0.1, ready_deadline - time.monotonic()),
                        poll_interval_s=self.config.poll_interval_s,
                    ).model_copy(update={"created_by_harness": True})
                return placement
            time.sleep(self.config.poll_interval_s)

        report.issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                message=(
                    "Timed out waiting for placed model to appear in cluster "
                    "state; treating as a placement refusal/give-up"
                ),
                evidence={
                    "placement_appearance_timeout_s": (
                        self.config.placement_appearance_timeout_s
                    ),
                    "placement_ready_timeout_s": self.config.placement_ready_timeout_s,
                },
            )
        )
        return PlacementResult(model_id=model_id, created_by_harness=True, ready=False)

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
        if test.kind == "audio_speech":
            return self._run_audio_speech_test(
                client, model_id=model_id, test=test, repetition=repetition
            )
        if test.kind == "audio_transcription":
            return self._run_audio_transcription_test(
                client, model_id=model_id, test=test, repetition=repetition
            )
        if test.kind == "speech_roundtrip":
            return self._run_speech_roundtrip_test(
                client,
                model_id=model_id,
                test=test,
                repetition=repetition,
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
            if (
                test.expected_embedding_dimensions is not None
                and any(
                    dim != test.expected_embedding_dimensions
                    for dim in execution.dimensions
                )
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
    ) -> TestResult:
        """Run a TTS request and assert Skulk returns plausible audio bytes."""

        issues: list[Issue] = []
        output = ""
        elapsed = 0.0
        try:
            execution = client.audio_speech(
                model_id=model_id,
                input_text=_expanded_prompt(test),
                response_format=test.audio_response_format,
                voice=test.speech_voice,
                speed=test.speech_speed,
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
            output = (
                f"audio_bytes={len(execution.audio)} "
                f"media_type={execution.media_type} "
                f"format={execution.response_format}"
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

    def _run_speech_roundtrip_test(
        self,
        client: SkulkClient,
        *,
        model_id: str,
        test: PromptTest,
        repetition: int,
        spec: RunSpec | None,
        report: RunReport | None,
        writer: ReportWriter | None,
    ) -> TestResult:
        """Generate speech with a TTS model, then transcribe it with an STT model."""

        issues: list[Issue] = []
        output = ""
        elapsed = 0.0
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
                model_id, test.name, repetition, output, elapsed, issues
            )

        transcription_model_id: str | None = None
        stt_placement: PlacementResult | None = None
        try:
            transcription_model_id = test.transcription_model_id or _first_stt_model_id(
                client.list_models(), exclude_model_id=model_id
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
                    model_id, test.name, repetition, output, elapsed, issues
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
                    model_id, test.name, repetition, output, elapsed, issues
                )
            _append_unique_placement(report, stt_placement)

            speech = client.audio_speech(
                model_id=model_id,
                input_text=_expanded_prompt(test),
                response_format=test.audio_response_format,
                voice=test.speech_voice,
                speed=test.speech_speed,
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
            transcript = client.audio_transcription(
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
                )
                if spec.delete_staged_models and torn_down:
                    self._evict_staged_model(client, transcription_model_id, report)
                if writer is not None:
                    writer.write(report)
        return _speech_result(model_id, test.name, repetition, output, elapsed, issues)

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
        image_url: dict[str, object] = {"url": image.url}
        if image.detail is not None:
            image_url["detail"] = image.detail
        content.append({"type": "image_url", "image_url": image_url})
    messages.append({"role": "user", "content": content})
    return messages


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
    if response_format == "wav" and audio and not (
        audio.startswith(b"RIFF") and b"WAVE" in audio[:16]
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


def _speech_result(
    model_id: str,
    test_name: str,
    repetition: int,
    output: str,
    elapsed: float,
    issues: list[Issue],
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
        ),
        issues=issues,
    )


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


def _empty_metrics() -> GenerationMetrics:
    return GenerationMetrics(elapsed_s=0.0)


def _run_id(spec: RunSpec) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = spec.run_name or f"{spec.model_set}-{spec.test_set}"
    return f"{stamp}-{slugify(name)}"
