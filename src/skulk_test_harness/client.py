"""HTTP client for a live Skulk API node."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from skulk_test_harness.models import (
    GenerationMetrics,
    Issue,
    PlacementResult,
    ToolCallRecord,
)
from skulk_test_harness.utils import unwrap_tagged

QueryParams = dict[str, str | int | float | bool | list[str]]


class SkulkApiError(RuntimeError):
    """Raised when Skulk returns an unsuccessful response."""

    def __init__(self, method: str, path: str, status_code: int, body: str) -> None:
        super().__init__(f"{method} {path} failed with HTTP {status_code}: {body[:500]}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ChatExecution:
    """Text and timing collected from one chat completion request."""

    text: str
    reasoning_text: str
    tool_calls: list[ToolCallRecord]
    metrics: GenerationMetrics
    command_id: str | None
    raw_events: list[dict[str, object]]


class SkulkClient:
    """Small synchronous Skulk API client."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float = 30.0,
        generation_timeout_s: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.generation_timeout_s = generation_timeout_s
        self._client = httpx.Client(base_url=self.base_url, timeout=request_timeout_s)

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "SkulkClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        params: QueryParams | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, object] | list[object] | str | None:
        response = self._client.request(
            method,
            path,
            json=json_body,
            params=params,
            timeout=timeout_s or self.request_timeout_s,
        )
        if response.status_code >= 400:
            raise SkulkApiError(method, path, response.status_code, response.text)
        if not response.content:
            return None
        return response.json()

    def get_node_id(self) -> str:
        """Return this API node's libp2p node ID."""

        payload = self._request_json("GET", "/node_id")
        if not isinstance(payload, str):
            raise TypeError(f"Unexpected /node_id payload: {payload!r}")
        return payload

    def get_state(self) -> dict[str, object]:
        """Return current replicated cluster state as seen by this API node."""

        payload = self._request_json("GET", "/state")
        if not isinstance(payload, dict):
            raise TypeError("Expected /state to return an object")
        return payload

    def get_diagnostics_node(self) -> dict[str, object]:
        """Return this API node's diagnostics payload.

        The ``runtime`` block carries ``masterNodeId``, ``isMaster``, ``nodeId``,
        and ``friendlyName`` for the node serving this request, which the
        stability suites use to identify the current master before crashing it.
        """

        payload = self._request_json("GET", "/v1/diagnostics/node")
        if not isinstance(payload, dict):
            raise TypeError("Expected /v1/diagnostics/node to return an object")
        return payload

    def get_master_node_id(self) -> str:
        """Return the current master node ID as seen by this API node."""

        diagnostics = self.get_diagnostics_node()
        runtime = diagnostics.get("runtime")
        if not isinstance(runtime, dict):
            return ""
        master = runtime.get("masterNodeId")
        return master if isinstance(master, str) else ""

    def list_models(self) -> list[dict[str, object]]:
        """Return the Skulk model catalog."""

        payload = self._request_json("GET", "/models")
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def add_model_card(self, model_id: str) -> dict[str, object] | None:
        """Ask Skulk to add/fetch a custom model card for ``model_id``."""

        payload = self._request_json("POST", "/models/add", json_body={"model_id": model_id})
        return payload if isinstance(payload, dict) else None

    def get_store_registry(self) -> dict[str, object] | None:
        """Return the model-store registry when available."""

        payload = self._request_json("GET", "/store/registry")
        return payload if isinstance(payload, dict) else None

    def request_store_download(self, model_id: str) -> dict[str, object] | None:
        """Request a shared-store download for ``model_id``."""

        path_model = quote(model_id, safe="/")
        payload = self._request_json(
            "POST", f"/store/models/{path_model}/download", timeout_s=60.0
        )
        return payload if isinstance(payload, dict) else None

    def get_store_download_status(self, model_id: str) -> dict[str, object] | None:
        """Return the store download status for ``model_id`` if available."""

        path_model = quote(model_id, safe="/")
        payload = self._request_json(
            "GET", f"/store/models/{path_model}/download/status"
        )
        return payload if isinstance(payload, dict) else None

    def get_placement_previews(
        self,
        model_id: str,
        *,
        excluded_node_ids: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """Return placement previews for one model."""

        params: QueryParams = {"model_id": model_id}
        if excluded_node_ids:
            params["excluded_node_ids"] = excluded_node_ids
        payload = self._request_json("GET", "/instance/previews", params=params)
        if not isinstance(payload, dict):
            return []
        previews = payload.get("previews")
        if not isinstance(previews, list):
            return []
        return [item for item in previews if isinstance(item, dict)]

    def place_model(
        self,
        *,
        model_id: str,
        sharding: str,
        instance_meta: str,
        min_nodes: int,
        excluded_nodes: list[str],
    ) -> dict[str, object] | None:
        """Request a model placement."""

        payload = self._request_json(
            "POST",
            "/place_instance",
            json_body={
                "model_id": model_id,
                "sharding": sharding,
                "instance_meta": instance_meta,
                "min_nodes": min_nodes,
                "excluded_nodes": excluded_nodes,
            },
            timeout_s=60.0,
        )
        return payload if isinstance(payload, dict) else None

    def delete_instance(self, instance_id: str) -> dict[str, object] | None:
        """Delete one Skulk instance."""

        path_instance = quote(instance_id, safe="")
        payload = self._request_json("DELETE", f"/instance/{path_instance}")
        return payload if isinstance(payload, dict) else None

    def find_placements_for_model(self, model_id: str) -> list[PlacementResult]:
        """Return instances currently placed for ``model_id``."""

        state = self.get_state()
        instances = state.get("instances")
        if not isinstance(instances, dict):
            return []
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
            placements.append(
                PlacementResult(
                    model_id=model_id,
                    instance_id=str(instance_id),
                    node_ids=list(node_to_runner) if isinstance(node_to_runner, dict) else [],
                    runner_ids=list(runner_to_shard) if isinstance(runner_to_shard, dict) else [],
                    instance_meta=tag,
                    reused_existing=True,
                    ready=self.instance_is_ready(str(instance_id)),
                )
            )
        return placements

    def instance_is_ready(self, instance_id: str) -> bool:
        """Return whether all runners for an instance are ready or running."""

        state = self.get_state()
        instance = _get_instance_body(state, instance_id)
        if instance is None:
            return False
        assignments = instance.get("shardAssignments")
        if not isinstance(assignments, dict):
            return False
        runner_to_shard = assignments.get("runnerToShard")
        if not isinstance(runner_to_shard, dict) or not runner_to_shard:
            return False
        runners = state.get("runners")
        if not isinstance(runners, dict):
            return False
        for runner_id in runner_to_shard:
            raw_status = runners.get(runner_id)
            parsed = unwrap_tagged(raw_status)
            if parsed is None:
                return False
            tag, _body = parsed
            if tag not in {"RunnerReady", "RunnerRunning"}:
                return False
        return True

    def wait_for_instance_ready(
        self,
        instance_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> PlacementResult:
        """Wait until an instance reaches a dispatchable runner status."""

        deadline = time.monotonic() + timeout_s
        last_result: PlacementResult | None = None
        while time.monotonic() < deadline:
            result = self.describe_instance(instance_id)
            if result is not None:
                last_result = result
                if result.ready:
                    return result
            time.sleep(poll_interval_s)
        if last_result is not None:
            return last_result
        return PlacementResult(model_id="", instance_id=instance_id, ready=False)

    def describe_instance(self, instance_id: str) -> PlacementResult | None:
        """Return a compact placement summary for one instance."""

        state = self.get_state()
        body = _get_instance_body(state, instance_id)
        if body is None:
            return None
        assignments = body.get("shardAssignments")
        if not isinstance(assignments, dict):
            return None
        node_to_runner = assignments.get("nodeToRunner")
        runner_to_shard = assignments.get("runnerToShard")
        model_id = str(assignments.get("modelId") or "")
        tag = _get_instance_tag(state, instance_id)
        return PlacementResult(
            model_id=model_id,
            instance_id=instance_id,
            node_ids=list(node_to_runner) if isinstance(node_to_runner, dict) else [],
            runner_ids=list(runner_to_shard) if isinstance(runner_to_shard, dict) else [],
            instance_meta=tag,
            ready=self.instance_is_ready(instance_id),
        )

    def stream_chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, object]],
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatExecution:
        """Run one streaming chat completion and measure wall-clock metrics."""

        payload: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

        start = time.monotonic()
        first_token_at: float | None = None
        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        raw_events: list[dict[str, object]] = []
        command_id: str | None = None
        chunks = 0

        with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=self.generation_timeout_s,
        ) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise SkulkApiError(
                    "POST", "/v1/chat/completions", response.status_code, body
                )
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(": command_id"):
                    command_id = line.rsplit(" ", 1)[-1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = _safe_json_object(data)
                if event is None:
                    continue
                raw_events.append(event)
                content, reasoning, event_tool_calls = _extract_stream_delta(event)
                text_for_timing = content or reasoning
                if text_for_timing:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    chunks += 1
                if event_tool_calls and first_token_at is None:
                    first_token_at = time.monotonic()
                if content:
                    output_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                tool_calls.extend(event_tool_calls)

        elapsed = time.monotonic() - start
        text = "".join(output_parts)
        approx_tokens = max(1, round(len(text) / 4)) if text else 0
        active_decode_s = (
            elapsed - (first_token_at - start)
            if first_token_at is not None
            else elapsed
        )
        wall_tps = approx_tokens / active_decode_s if active_decode_s > 0 else None
        metrics = GenerationMetrics(
            elapsed_s=elapsed,
            ttft_s=(first_token_at - start) if first_token_at is not None else None,
            output_chars=len(text),
            chunks=chunks,
            approx_output_tokens=approx_tokens,
            wall_tps=wall_tps,
        )
        return ChatExecution(
            text=text,
            reasoning_text="".join(reasoning_parts),
            tool_calls=tool_calls,
            metrics=metrics,
            command_id=command_id,
            raw_events=raw_events,
        )

    def bench_chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, object]],
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatExecution:
        """Run Skulk's non-streaming bench endpoint and collect reported metrics."""

        payload: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

        start = time.monotonic()
        response = self._request_json(
            "POST",
            "/bench/chat/completions",
            json_body=payload,
            timeout_s=self.generation_timeout_s,
        )
        elapsed = time.monotonic() - start
        event = response if isinstance(response, dict) else {}
        text = _extract_non_stream_text(event)
        tool_calls = _extract_non_stream_tool_calls(event)
        metrics = GenerationMetrics(
            elapsed_s=elapsed,
            ttft_s=None,
            output_chars=len(text),
            chunks=1 if text else 0,
            approx_output_tokens=max(1, round(len(text) / 4)) if text else 0,
        )
        stats = event.get("generation_stats")
        if isinstance(stats, dict):
            metrics = metrics.model_copy(
                update={
                    "skulk_prompt_tps": _float_or_none(stats.get("prompt_tps")),
                    "skulk_generation_tps": _float_or_none(
                        stats.get("generation_tps")
                    ),
                    "skulk_prompt_tokens": _int_or_none(stats.get("prompt_tokens")),
                    "skulk_generation_tokens": _int_or_none(
                        stats.get("generation_tokens")
                    ),
                }
            )
        return ChatExecution(
            text=text,
            reasoning_text="",
            tool_calls=tool_calls,
            metrics=metrics,
            command_id=str(event.get("id")) if event.get("id") is not None else None,
            raw_events=[event],
        )

    def detect_runner_state_drift(self) -> list[Issue]:
        """Detect divergence between replicated state and local diagnostics."""

        issues: list[Issue] = []
        try:
            diagnostics = self._request_json("GET", "/v1/diagnostics/node")
            state = self.get_state()
        except Exception as exc:
            return [
                Issue(
                    severity="warning",
                    message="Unable to inspect runner-state drift",
                    evidence={"error": repr(exc)},
                )
            ]
        if not isinstance(diagnostics, dict):
            return issues
        supervisor_runners = diagnostics.get("supervisorRunners")
        state_runners = state.get("runners")
        if not isinstance(supervisor_runners, list) or not isinstance(state_runners, dict):
            return issues
        active_runner_ids = _active_runner_ids(state)
        stale_runner_ids = [
            str(runner_id)
            for runner_id in state_runners
            if str(runner_id) not in active_runner_ids
        ]
        if stale_runner_ids:
            issues.append(
                Issue(
                    severity="warning",
                    message="Replicated state contains runner entries not referenced by any live instance",
                    evidence={
                        "count": len(stale_runner_ids),
                        "sample_runner_ids": stale_runner_ids[:10],
                    },
                )
            )

        for runner in supervisor_runners:
            if not isinstance(runner, dict):
                continue
            runner_id = str(runner.get("runnerId") or "")
            local_status = str(runner.get("statusKind") or "")
            parsed = unwrap_tagged(state_runners.get(runner_id))
            state_status = parsed[0] if parsed is not None else None
            if runner_id and state_status and local_status and state_status != local_status:
                issues.append(
                    Issue(
                        severity="warning",
                        message="Replicated runner state differs from local supervisor diagnostics",
                        model_id=str(runner.get("modelId") or ""),
                        evidence={
                            "runner_id": runner_id,
                            "state_status": state_status,
                            "local_status": local_status,
                            "phase": runner.get("phase"),
                        },
                    )
                )
        return issues


def _safe_json_object(data: str) -> dict[str, object] | None:
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_stream_delta(
    event: dict[str, object],
) -> tuple[str, str, list[ToolCallRecord]]:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", "", []
    first = choices[0]
    if not isinstance(first, dict):
        return "", "", []
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return "", "", []
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else "",
        _tool_call_records(delta.get("tool_calls")),
    )


def _extract_non_stream_text(event: dict[str, object]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_non_stream_tool_calls(event: dict[str, object]) -> list[ToolCallRecord]:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message")
    if not isinstance(message, dict):
        return []
    return _tool_call_records(message.get("tool_calls"))


def _tool_call_records(value: object) -> list[ToolCallRecord]:
    if not isinstance(value, list):
        return []
    records: list[ToolCallRecord] = []
    for index, raw_tool_call in enumerate(value):
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            continue
        parsed_arguments = _safe_json_object(arguments)
        records.append(
            ToolCallRecord(
                id=str(raw_tool_call.get("id") or function.get("id") or ""),
                name=name,
                arguments_text=arguments,
                arguments=parsed_arguments,
                index=_int_or_none(raw_tool_call.get("index")) or index,
            )
        )
    return records


def _get_instance_tag(state: dict[str, object], instance_id: str) -> str | None:
    instances = state.get("instances")
    if not isinstance(instances, dict):
        return None
    parsed = unwrap_tagged(instances.get(instance_id))
    return parsed[0] if parsed is not None else None


def _get_instance_body(
    state: dict[str, object], instance_id: str
) -> dict[str, object] | None:
    instances = state.get("instances")
    if not isinstance(instances, dict):
        return None
    parsed = unwrap_tagged(instances.get(instance_id))
    return parsed[1] if parsed is not None else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _active_runner_ids(state: dict[str, object]) -> set[str]:
    instances = state.get("instances")
    if not isinstance(instances, dict):
        return set()
    active: set[str] = set()
    for raw_instance in instances.values():
        parsed = unwrap_tagged(raw_instance)
        if parsed is None:
            continue
        _tag, body = parsed
        assignments = body.get("shardAssignments")
        if not isinstance(assignments, dict):
            continue
        runner_to_shard = assignments.get("runnerToShard")
        if isinstance(runner_to_shard, dict):
            active.update(str(runner_id) for runner_id in runner_to_shard)
    return active
