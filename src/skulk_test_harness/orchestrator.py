"""Main planning and execution engine for the Skulk harness."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path

from skulk_test_harness.client import ChatExecution, SkulkApiError, SkulkClient
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
                    report.placements.append(_placement_from_preview(model.model_id, preview))
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
                self._run_model_lifecycle(
                    client,
                    model,
                    spec,
                    report,
                    test_set,
                    writer,
                    thinking_toggles,
                    deferred_retry=True,
                )
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
                return True
            report.placements.append(placement)
            # Dashboard parity: when the model exposes a thinking toggle and the
            # test does not pin enable_thinking, default it OFF so the model
            # answers instead of emitting an all-reasoning, length-capped reply.
            thinking_default = (
                False if thinking_toggles.get(model.model_id) else None
            )
            for test in test_set.tests:
                for repetition in range(1, test.repetitions + 1):
                    result = self._run_test(
                        client,
                        model_id=model.model_id,
                        test=test,
                        repetition=repetition,
                        artifact_dir=writer.run_dir(report.run_id) / "artifacts",
                        thinking_default=thinking_default,
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
            if (
                placement is not None
                and placement.created_by_harness
                and not spec.retain_instances
            ):
                self._teardown_harness_instances(
                    client, model.model_id, placement.instance_id, report
                )

    def _teardown_harness_instances(
        self,
        client: SkulkClient,
        model_id: str,
        primary_instance_id: str | None,
        report: RunReport,
    ) -> None:
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
        try:
            for live in client.find_placements_for_model(model_id):
                if live.instance_id and live.instance_id not in target_ids:
                    target_ids.append(live.instance_id)
        except Exception as exc:  # noqa: BLE001 - sweep is best-effort
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
                report.issues.append(
                    Issue(
                        severity="warning",
                        model_id=model_id,
                        message="Failed to delete harness-created instance",
                        evidence={"error": str(exc), "instance_id": instance_id},
                    )
                )

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
            if not seed.require_mlx_community or seed.model_id.startswith("mlx-community/"):
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
        return sorted(previews, key=lambda p: (_preview_node_count(p), str(p.get("sharding"))))[0]

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

        deadline = time.monotonic() + self.config.placement_ready_timeout_s
        while time.monotonic() < deadline:
            placements = client.find_placements_for_model(model_id)
            if placements:
                placement = placements[0].model_copy(
                    update={"created_by_harness": True, "reused_existing": False}
                )
                if placement.instance_id and not placement.ready:
                    placement = client.wait_for_instance_ready(
                        placement.instance_id,
                        timeout_s=max(0.1, deadline - time.monotonic()),
                        poll_interval_s=self.config.poll_interval_s,
                    ).model_copy(update={"created_by_harness": True})
                return placement
            time.sleep(self.config.poll_interval_s)

        report.issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                message="Timed out waiting for placed model to appear in cluster state",
            )
        )
        return None

    def _ensure_model_card(
        self, client: SkulkClient, model_id: str, report: RunReport
    ) -> None:
        catalog_ids = {_model_id_from_catalog_entry(item) for item in client.list_models()}
        if model_id in catalog_ids:
            return
        try:
            client.add_model_card(model_id)
        except SkulkApiError as exc:
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
        except SkulkApiError as exc:
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
            except SkulkApiError:
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
    ) -> TestResult:
        messages = []
        if test.system:
            messages.append({"role": "system", "content": test.system})
        messages.append({"role": "user", "content": test.prompt})
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
        )
        issues.extend(roundtrip_issues)
        artifact_path = _artifact_path(artifact_dir, model_id, test, repetition, execution)
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
        if selector.family and str(model.get("family") or "").lower() != selector.family.lower():
            continue
        if selector.id_contains and selector.id_contains.lower() not in model_id.lower():
            continue
        if regex and regex.search(model_id) is None:
            continue
        if selector.tags_any and not _has_any(model.get("tags"), selector.tags_any):
            continue
        if selector.tasks_any and not _has_any(model.get("tasks"), selector.tasks_any):
            continue
        if selector.capabilities_any and not _has_any(
            model.get("capabilities"), selector.capabilities_any
        ):
            continue
        selected.append(model)
        if selector.max_models is not None and len(selected) >= selector.max_models:
            break
    return selected


def _model_id_from_catalog_entry(model: dict[str, object]) -> str:
    for key in ("model_id", "hugging_face_id", "id", "name"):
        value = model.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


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


def _placement_from_preview(model_id: str, preview: dict[str, object]) -> PlacementResult:
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
) -> list[Issue]:
    issues: list[Issue] = []
    tool_calls = tool_calls or []
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
    if len(text) < criteria.min_chars:
        issues.append(
            Issue(
                severity="error",
                model_id=model_id,
                test_name=test_name,
                message=f"Output shorter than required minimum ({len(text)} < {criteria.min_chars})",
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
    for substring in criteria.forbidden_substrings:
        if substring.lower() in text.lower():
            issues.append(
                Issue(
                    severity="error",
                    model_id=model_id,
                    test_name=test_name,
                    message=f"Output contained forbidden substring {substring!r}",
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
    if criteria.min_code_block_chars and len(code_block or "") < criteria.min_code_block_chars:
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
            # answered something else) — that is a content failure, not a
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
    extension = "html" if "<html" in code.lower() or "<canvas" in code.lower() else "txt"
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
