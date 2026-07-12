"""HTTP client for a live Skulk API node."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from websockets.exceptions import WebSocketException
from websockets.sync import client as websocket_client
from websockets.sync.connection import Connection

from skulk_test_harness.models import (
    GenerationMetrics,
    Issue,
    PlacementResult,
    ToolCallRecord,
)
from skulk_test_harness.utils import unwrap_tagged

QueryParams = dict[str, str | int | float | bool | list[str]]
_REALTIME_MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def _required_int(payload: Mapping[str, object], key: str) -> int:
    """Read one required integer counter without accepting booleans."""

    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected diagnostics field {key!r} to be an integer")
    return value


def _replace_url_host(url: str, host: str) -> str:
    """Replace a URL host while preserving its scheme and explicit port."""

    parsed = urlsplit(url)
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", "", "", ""))


def _realtime_url(base_url: str, model_id: str, *, fabric_chain: bool = False) -> str:
    """Build a same-owner realtime or Fabric-chain WebSocket URL."""

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported Skulk API URL for realtime STT: {base_url!r}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = "/v1/fabric/chains/speech" if fabric_chain else "/v1/realtime"
    parameter = "stt_model" if fabric_chain else "model"
    return urlunsplit(
        (scheme, parsed.netloc, path, f"{parameter}={quote(model_id, safe='')}", "")
    )


class SkulkApiError(RuntimeError):
    """Raised when Skulk returns an unsuccessful response."""

    def __init__(self, method: str, path: str, status_code: int, body: str) -> None:
        super().__init__(
            f"{method} {path} failed with HTTP {status_code}: {body[:500]}"
        )
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
    # Per-token logprob coverage observed in the stream (0 unless logprobs were
    # requested). ``logprob_tokens`` counts tokens that carried a logprob;
    # ``top_logprob_tokens`` counts those that also carried ranked alternatives.
    logprob_tokens: int = 0
    top_logprob_tokens: int = 0
    canceled: bool = False


@dataclass(frozen=True)
class EmbeddingExecution:
    """Embedding vectors and timing collected from one embeddings request."""

    dimensions: list[int]
    norms: list[float]
    elapsed_s: float
    raw_response: dict[str, object]


@dataclass(frozen=True)
class AudioSpeechExecution:
    """Encoded audio and timing collected from one speech synthesis request."""

    audio: bytes
    media_type: str
    elapsed_s: float
    response_format: str
    chunks: int = 0
    first_byte_s: float | None = None
    chunk_sizes: list[int] = field(default_factory=list)
    chunk_arrival_s: list[float] = field(default_factory=list)
    streaming: bool = False


@dataclass(frozen=True)
class AudioTranscriptionExecution:
    """Transcript text and timing collected from one audio transcription request."""

    text: str
    media_type: str
    elapsed_s: float
    response_format: str
    raw_response: dict[str, object] | str


@dataclass(frozen=True)
class StreamingAudioTranscriptionExecution:
    """Typed SSE transcript lifecycle and timing from one uploaded audio clip."""

    text: str
    elapsed_s: float
    first_transcript_s: float | None
    input_bytes: int
    transcript_deltas: int
    event_types: list[str]
    event_arrival_s: list[float]
    events: list[dict[str, object]]
    canceled: bool = False


@dataclass(frozen=True)
class RealtimeTranscriptionExecution:
    """Transcript lifecycle and timing from one realtime WebSocket session."""

    text: str
    elapsed_s: float
    first_transcript_s: float | None
    input_bytes: int
    input_frames: int
    transcript_deltas: int
    event_types: list[str]
    canceled: bool = False
    assistant_text: str = ""
    response_audio: bytes = b""
    response_audio_chunks: int = 0
    response_status: str | None = None


@dataclass(frozen=True)
class ClusterApiOwner:
    """One controller-reachable API route paired with its fabric node identity."""

    node_id: str
    base_url: str


@dataclass(frozen=True)
class DataPlaneDiagnosticsSnapshot:
    """Counters and live gauges read from one node's DATA diagnostics."""

    node_id: str
    active_streams: int
    started_frames: int
    completed_frames: int
    failed_frames: int
    cancelled_frames: int
    duplicate_frames: int
    out_of_order_frames: int
    skipped_sequences: int
    late_frames: int
    missing_started_streams: int
    missing_terminal_streams: int
    idle_timeouts: int
    transport_failures: int
    active_stream_queues: int
    queue_depth: int
    local_short_circuits: int
    remote_frames_enqueued: int
    remote_frames_published: int
    remote_frames_dropped: int
    remote_publish_failures: int

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "DataPlaneDiagnosticsSnapshot":
        """Parse the DATA subset of ``GET /v1/diagnostics/node`` strictly."""

        runtime = payload.get("runtime")
        data_plane = payload.get("dataPlane")
        if not isinstance(runtime, dict) or not isinstance(data_plane, dict):
            raise TypeError(
                "Node diagnostics did not include runtime/dataPlane objects"
            )
        egress = data_plane.get("egress")
        if not isinstance(egress, dict):
            raise TypeError("Node diagnostics did not include dataPlane.egress")
        node_id = runtime.get("nodeId")
        if not isinstance(node_id, str) or not node_id:
            raise TypeError("Node diagnostics did not include runtime.nodeId")

        return cls(
            node_id=node_id,
            active_streams=_required_int(data_plane, "activeStreams"),
            started_frames=_required_int(data_plane, "startedFrames"),
            completed_frames=_required_int(data_plane, "completedFrames"),
            failed_frames=_required_int(data_plane, "failedFrames"),
            cancelled_frames=_required_int(data_plane, "cancelledFrames"),
            duplicate_frames=_required_int(data_plane, "duplicateFrames"),
            out_of_order_frames=_required_int(data_plane, "outOfOrderFrames"),
            skipped_sequences=_required_int(data_plane, "skippedSequences"),
            late_frames=_required_int(data_plane, "lateFrames"),
            missing_started_streams=_required_int(data_plane, "missingStartedStreams"),
            missing_terminal_streams=_required_int(
                data_plane, "missingTerminalStreams"
            ),
            idle_timeouts=_required_int(data_plane, "idleTimeouts"),
            transport_failures=_required_int(data_plane, "transportFailures"),
            active_stream_queues=_required_int(egress, "activeStreamQueues"),
            queue_depth=_required_int(egress, "queueDepth"),
            local_short_circuits=_required_int(egress, "localShortCircuits"),
            remote_frames_enqueued=_required_int(egress, "remoteFramesEnqueued"),
            remote_frames_published=_required_int(egress, "remoteFramesPublished"),
            remote_frames_dropped=_required_int(egress, "remoteFramesDropped"),
            remote_publish_failures=_required_int(egress, "remotePublishFailures"),
        )


@dataclass(frozen=True)
class ProviderCapabilityDiagnosticsSnapshot:
    """Lifecycle and media counters for one provider capability on one node."""

    node_id: str
    capability_id: str
    active_streams: int
    stream_slots_in_use: int
    admitted_streams: int
    input_queue_depth: int
    input_frames: int
    input_media_bytes: int
    output_frames: int
    output_media_bytes: int
    completed_streams: int
    failed_streams: int
    cancelled_streams: int
    missing_terminal_streams: int
    cancellation_requests: int

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        capability_id: str,
    ) -> "ProviderCapabilityDiagnosticsSnapshot":
        """Parse one capability from ``GET /v1/diagnostics/node`` strictly."""

        runtime = payload.get("runtime")
        provider = payload.get("provider")
        if not isinstance(runtime, dict) or not isinstance(provider, dict):
            raise TypeError("Node diagnostics did not include runtime/provider objects")
        node_id = runtime.get("nodeId")
        if not isinstance(node_id, str) or not node_id:
            raise TypeError("Node diagnostics did not include runtime.nodeId")
        capabilities = provider.get("capabilities")
        if not isinstance(capabilities, dict):
            raise TypeError("Node diagnostics did not include provider.capabilities")
        capability = capabilities.get(capability_id)
        if capability is None:
            capability = {
                "activeStreams": 0,
                "admittedStreams": 0,
                "inputQueueDepth": 0,
                "inputFrames": 0,
                "inputMediaBytes": 0,
                "outputFrames": 0,
                "outputMediaBytes": 0,
                "completedStreams": 0,
                "failedStreams": 0,
                "cancelledStreams": 0,
                "missingTerminalStreams": 0,
                "cancellationRequests": 0,
            }
        if not isinstance(capability, dict):
            raise TypeError(
                f"Provider diagnostics for {capability_id!r} were not an object"
            )
        return cls(
            node_id=node_id,
            capability_id=capability_id,
            active_streams=_required_int(capability, "activeStreams"),
            stream_slots_in_use=_required_int(provider, "streamSlotsInUse"),
            admitted_streams=_required_int(capability, "admittedStreams"),
            input_queue_depth=_required_int(capability, "inputQueueDepth"),
            input_frames=_required_int(capability, "inputFrames"),
            input_media_bytes=_required_int(capability, "inputMediaBytes"),
            output_frames=_required_int(capability, "outputFrames"),
            output_media_bytes=_required_int(capability, "outputMediaBytes"),
            completed_streams=_required_int(capability, "completedStreams"),
            failed_streams=_required_int(capability, "failedStreams"),
            cancelled_streams=_required_int(capability, "cancelledStreams"),
            missing_terminal_streams=_required_int(
                capability, "missingTerminalStreams"
            ),
            cancellation_requests=_required_int(capability, "cancellationRequests"),
        )


def _receive_realtime_event(
    connection: Connection,
    *,
    timeout_s: float,
) -> dict[str, object]:
    """Receive and parse one bounded JSON text event from Skulk realtime."""

    message = connection.recv(timeout=timeout_s, decode=True)
    if not isinstance(message, str):
        raise TypeError("Realtime transcription returned a binary event")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise TypeError("Realtime transcription returned a non-object JSON event")
    event_type = payload.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise TypeError("Realtime transcription event did not include a type")
    return payload


def _realtime_error_message(event: dict[str, object]) -> str:
    """Extract a stable failure message from a realtime error event."""

    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return f"Realtime transcription failed with event {event.get('type')!r}"


class SkulkClient:
    """Small synchronous Skulk API client."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float = 30.0,
        generation_timeout_s: float = 1800.0,
        stream_read_timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.generation_timeout_s = generation_timeout_s
        self.stream_read_timeout_s = stream_read_timeout_s
        self._client = httpx.Client(base_url=self.base_url, timeout=request_timeout_s)

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "SkulkClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # Transport-level failures that are safe to retry for read-only calls: a
    # long-lived client reusing a keep-alive connection the server has already
    # closed surfaces as RemoteProtocolError ("Server disconnected without
    # sending a response") on the next request, even though the server is
    # healthy. Timeouts under cluster load are likewise transient for idempotent
    # reads. Mutating calls use the narrower tuple below so a placement/download
    # request that may already have reached Skulk is not replayed.
    _READ_RETRYABLE_TRANSPORT_ERRORS = (
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
    )
    _MUTATION_RETRYABLE_TRANSPORT_ERRORS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
    )
    _MAX_TRANSPORT_RETRIES = 4

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        params: QueryParams | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, object] | list[object] | str | None:
        method_upper = method.upper()
        retryable_errors = (
            self._READ_RETRYABLE_TRANSPORT_ERRORS
            if method_upper in {"GET", "HEAD", "OPTIONS"}
            else self._MUTATION_RETRYABLE_TRANSPORT_ERRORS
        )
        last_exc: Exception | None = None
        for attempt in range(self._MAX_TRANSPORT_RETRIES):
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    timeout=timeout_s or self.request_timeout_s,
                )
                break
            except retryable_errors as exc:
                # Back off briefly and let httpx open a fresh connection.
                # Mutations only retry pre-request connection/pool failures.
                last_exc = exc
                if attempt == self._MAX_TRANSPORT_RETRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:  # pragma: no cover - loop always breaks or raises
            assert last_exc is not None
            raise last_exc
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

    def get_cluster_api_owners(self) -> list[ClusterApiOwner]:
        """Return controller-reachable API routes with stable live node IDs.

        Peer URLs in cluster diagnostics are selected from the serving node's
        routes and may not be reachable from the harness controller. Prefer a
        node's advertised overlay DNS or friendly hostname when available, and
        probe each candidate before assigning pressure traffic to it.
        """

        payload = self._request_json("GET", "/v1/diagnostics/cluster")
        local_node_id = (
            payload.get("localNodeId") if isinstance(payload, dict) else None
        )
        if not isinstance(local_node_id, str) or not local_node_id:
            local_node_id = self.get_node_id()
        owners = [ClusterApiOwner(node_id=local_node_id, base_url=self.base_url)]
        if not isinstance(payload, dict):
            return owners
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            return owners
        seen_node_ids = {local_node_id}
        for node in nodes:
            if not isinstance(node, dict) or node.get("ok") is not True:
                continue
            node_id = node.get("nodeId")
            if not isinstance(node_id, str) or not node_id or node_id in seen_node_ids:
                continue
            for candidate in self._cluster_node_api_candidates(node):
                if any(owner.base_url == candidate for owner in owners):
                    break
                if self._api_url_reachable(candidate):
                    owners.append(ClusterApiOwner(node_id=node_id, base_url=candidate))
                    seen_node_ids.add(node_id)
                    break
        return owners

    def get_cluster_api_urls(self) -> list[str]:
        """Return controller-reachable API URLs for distinct cluster nodes."""

        return [owner.base_url for owner in self.get_cluster_api_owners()]

    def _cluster_node_api_candidates(self, node: dict[str, object]) -> list[str]:
        """Build preferred API routes for one cluster-diagnostics node."""

        reported_url = node.get("url")
        route_template = (
            reported_url
            if isinstance(reported_url, str) and reported_url
            else self.base_url
        )
        diagnostics = node.get("diagnostics")
        typed_diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        tailscale = typed_diagnostics.get("tailscale")
        typed_tailscale = tailscale if isinstance(tailscale, dict) else {}
        identity = typed_diagnostics.get("identity")
        typed_identity = identity if isinstance(identity, dict) else {}

        hosts = (
            typed_tailscale.get("dnsName"),
            typed_tailscale.get("hostname"),
            typed_identity.get("friendlyName"),
        )
        candidates: list[str] = []
        for host in hosts:
            if isinstance(host, str) and host:
                candidate = _replace_url_host(route_template, host)
                if candidate not in candidates:
                    candidates.append(candidate)
        if isinstance(reported_url, str) and reported_url:
            normalized_reported_url = reported_url.rstrip("/")
            if normalized_reported_url not in candidates:
                candidates.append(normalized_reported_url)
        return candidates

    def _api_url_reachable(self, base_url: str) -> bool:
        """Return whether the controller can reach a candidate Skulk API."""

        timeout_seconds = min(5.0, self.request_timeout_s)
        try:
            response = httpx.get(
                f"{base_url}/config",
                timeout=httpx.Timeout(timeout_seconds),
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    def resolve_node_ids(self, names: list[str]) -> list[str]:
        """Resolve friendly node names (e.g. ``kite5``) to live libp2p node IDs.

        Placement exclusion/pinning is by node ID, but a node's libp2p ID is
        ephemeral (regenerated every process start), so a battery cell can only
        refer to a node by its stable friendly name. This maps each name through
        the current ``/state`` ``nodeIdentities`` (``friendlyName`` -> node id).
        An entry that is already a known node ID passes through unchanged. Raises
        ``ValueError`` listing any name that matches no live node, so a cell that
        asks to exclude a node that is not in the cluster fails loudly rather than
        silently placing on it.
        """
        if not names:
            return []
        identities = self.get_state().get("nodeIdentities")
        if not isinstance(identities, dict):
            identities = {}
        friendly_to_id: dict[str, str] = {}
        for node_id, info in identities.items():
            if isinstance(info, dict):
                friendly = info.get("friendlyName")
                if isinstance(friendly, str):
                    friendly_to_id[friendly] = node_id
        resolved: list[str] = []
        unknown: list[str] = []
        for name in names:
            if name in friendly_to_id:
                resolved.append(friendly_to_id[name])
            elif name in identities:
                resolved.append(name)
            else:
                unknown.append(name)
        if unknown:
            raise ValueError(
                f"Unknown node name(s) {unknown!r}; live nodes are "
                f"{sorted(friendly_to_id)}"
            )
        return resolved

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

    def get_data_plane_diagnostics(self) -> DataPlaneDiagnosticsSnapshot:
        """Return typed DATA lifecycle and egress counters for this API node."""

        return DataPlaneDiagnosticsSnapshot.from_payload(self.get_diagnostics_node())

    def get_provider_capability_diagnostics(
        self,
        capability_id: str,
    ) -> ProviderCapabilityDiagnosticsSnapshot:
        """Return provider counters for one qualified capability on this node."""

        return ProviderCapabilityDiagnosticsSnapshot.from_payload(
            self.get_diagnostics_node(),
            capability_id,
        )

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

    def resolved_thinking_toggle_by_model(self) -> dict[str, bool]:
        """Map each catalog model id to whether it supports a thinking toggle.

        Read from ``/models`` -> ``resolved_capabilities.supports_thinking_toggle``.
        The harness uses this to mirror the dashboard's off-by-default thinking
        behavior: a model whose toggle is on but whose request omits
        ``enable_thinking`` lets the chat template default thinking ON, which on
        models like GLM produces an all-reasoning, ``finish_reason=length``
        response with empty content. Sending ``enable_thinking=false`` explicitly
        (the dashboard default) avoids that.
        """
        toggles: dict[str, bool] = {}
        for entry in self.list_models():
            model_id = entry.get("id")
            resolved = entry.get("resolved_capabilities")
            if not isinstance(model_id, str) or not isinstance(resolved, dict):
                continue
            toggles[model_id] = bool(resolved.get("supports_thinking_toggle"))
        return toggles

    def add_model_card(self, model_id: str) -> dict[str, object] | None:
        """Ask Skulk to add/fetch a custom model card for ``model_id``."""

        payload = self._request_json(
            "POST", "/models/add", json_body={"model_id": model_id}
        )
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

    def audio_voices(self, model_id: str) -> list[str]:
        """Return stable built-in voice identifiers for a mounted TTS model."""

        payload = self._request_json(
            "GET", "/v1/audio/voices", params={"model": model_id}
        )
        if not isinstance(payload, dict) or payload.get("object") != "list":
            raise TypeError("Expected /v1/audio/voices to return a list object")
        data = payload.get("data")
        if not isinstance(data, list):
            raise TypeError("Expected /v1/audio/voices data to be a list")
        voices: list[str] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise TypeError("Expected every audio voice to have a string id")
            voices.append(item["id"])
        return voices

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

    def delete_store_model(
        self, model_id: str, *, timeout_s: float | None = None
    ) -> dict[str, object] | None:
        """Evict a model's staged weights from the model store (DELETE).

        Used to clean up staged GGUF/MLX weights after a benchmark run so test
        models do not accumulate on disk. Maps to ``DELETE /store/models/{id}``.
        """
        path_model = quote(model_id, safe="/")
        payload = self._request_json(
            "DELETE", f"/store/models/{path_model}", timeout_s=timeout_s
        )
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
                    node_ids=list(node_to_runner)
                    if isinstance(node_to_runner, dict)
                    else [],
                    runner_ids=list(runner_to_shard)
                    if isinstance(runner_to_shard, dict)
                    else [],
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
            runner_ids=list(runner_to_shard)
            if isinstance(runner_to_shard, dict)
            else [],
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
        top_logprobs: int | None = None,
        cancel_after_chunks: int | None = None,
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
        if top_logprobs is not None:
            # OpenAI semantics: logprobs must be on for top_logprobs to apply.
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        start = time.monotonic()
        first_token_at: float | None = None
        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        raw_events: list[dict[str, object]] = []
        command_id: str | None = None
        chunks = 0
        logprob_tokens = 0
        top_logprob_tokens = 0
        canceled = False

        try:
            with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(
                    timeout=None,
                    connect=self.request_timeout_s,
                    read=self.stream_read_timeout_s,
                    write=self.request_timeout_s,
                    pool=self.request_timeout_s,
                ),
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
                    event_logprobs, event_top_logprobs = _extract_stream_logprobs(event)
                    logprob_tokens += event_logprobs
                    top_logprob_tokens += event_top_logprobs
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
                    if cancel_after_chunks and chunks >= cancel_after_chunks:
                        canceled = True
                        break
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SkulkApiError(
                "POST",
                "/v1/chat/completions",
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc

        elapsed = time.monotonic() - start
        text = "".join(output_parts)
        reasoning_text = "".join(reasoning_parts)
        generated_chars = len(text) + len(reasoning_text)
        approx_tokens = max(1, round(generated_chars / 4)) if generated_chars else 0
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
            generated_chars=generated_chars,
            chunks=chunks,
            approx_output_tokens=approx_tokens,
            wall_tps=wall_tps,
        )
        return ChatExecution(
            text=text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            metrics=metrics,
            command_id=command_id,
            raw_events=raw_events,
            logprob_tokens=logprob_tokens,
            top_logprob_tokens=top_logprob_tokens,
            canceled=canceled,
        )

    def embeddings(
        self,
        *,
        model_id: str,
        input_text: str | list[str],
        encoding_format: str = "float",
    ) -> EmbeddingExecution:
        """Run one OpenAI-compatible embeddings request and summarize vectors."""

        start = time.monotonic()
        payload = {
            "model": model_id,
            "input": input_text,
            "encoding_format": encoding_format,
        }
        response = self._request_json(
            "POST",
            "/v1/embeddings",
            json_body=payload,
            timeout_s=self.generation_timeout_s,
        )
        elapsed = time.monotonic() - start
        if not isinstance(response, dict):
            raise TypeError(f"Unexpected /v1/embeddings payload: {response!r}")
        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise TypeError("Expected /v1/embeddings to return non-empty data")
        dimensions: list[int] = []
        norms: list[float] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Embedding data item was not an object")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise TypeError("Embedding vector was absent or empty")
            values = [float(value) for value in embedding]
            dimensions.append(len(values))
            norms.append(sum(value * value for value in values) ** 0.5)
        return EmbeddingExecution(
            dimensions=dimensions,
            norms=norms,
            elapsed_s=elapsed,
            raw_response=response,
        )

    def audio_speech(
        self,
        *,
        model_id: str,
        input_text: str,
        response_format: str = "wav",
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        streaming_interval: float | None = None,
        read_delay_s: float = 0.0,
        reference_audio: bytes | None = None,
        reference_audio_filename: str = "reference.wav",
        reference_audio_media_type: str = "audio/wav",
        reference_text: str | None = None,
    ) -> AudioSpeechExecution:
        """Generate speech audio with OpenAI's speech endpoint."""

        if streaming_interval is not None and not stream:
            raise ValueError("streaming_interval requires stream=True")
        if read_delay_s < 0:
            raise ValueError("read_delay_s must be non-negative")
        if stream and reference_audio is not None:
            raise ValueError("reference_audio is not supported with stream=True")
        payload: dict[str, object] = {
            "model": model_id,
            "input": input_text,
            "response_format": response_format,
        }
        if stream:
            payload["stream"] = True
        if streaming_interval is not None:
            payload["streaming_interval"] = streaming_interval
        if voice is not None:
            payload["voice"] = voice
        if speed is not None:
            payload["speed"] = speed

        start = time.monotonic()
        if stream:
            first_byte_at: float | None = None
            audio_parts: list[bytes] = []
            chunk_sizes: list[int] = []
            chunk_arrival_s: list[float] = []
            intentional_read_delay_s = 0.0
            try:
                with self._client.stream(
                    "POST",
                    "/v1/audio/speech",
                    json=payload,
                    timeout=httpx.Timeout(
                        timeout=None,
                        connect=self.request_timeout_s,
                        read=self.stream_read_timeout_s,
                        write=self.request_timeout_s,
                        pool=self.request_timeout_s,
                    ),
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise SkulkApiError(
                            "POST", "/v1/audio/speech", response.status_code, body
                        )
                    media_type = _base_media_type(
                        response.headers.get("content-type", "")
                    )
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        arrived_at = max(
                            0.0,
                            time.monotonic() - start - intentional_read_delay_s,
                        )
                        if first_byte_at is None:
                            first_byte_at = arrived_at
                        audio_parts.append(chunk)
                        chunk_sizes.append(len(chunk))
                        chunk_arrival_s.append(arrived_at)
                        if read_delay_s > 0:
                            time.sleep(read_delay_s)
                            intentional_read_delay_s += read_delay_s
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise SkulkApiError(
                    "POST",
                    "/v1/audio/speech",
                    0,
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            elapsed = time.monotonic() - start
            return AudioSpeechExecution(
                audio=b"".join(audio_parts),
                media_type=media_type,
                elapsed_s=elapsed,
                response_format=response_format,
                chunks=len(chunk_sizes),
                first_byte_s=first_byte_at,
                chunk_sizes=chunk_sizes,
                chunk_arrival_s=chunk_arrival_s,
                streaming=True,
            )

        try:
            if reference_audio is None:
                response = self._client.post(
                    "/v1/audio/speech",
                    json=payload,
                    timeout=self.generation_timeout_s,
                )
            else:
                form = {key: str(value) for key, value in payload.items()}
                if reference_text is not None:
                    form["reference_text"] = reference_text
                response = self._client.post(
                    "/v1/audio/speech",
                    data=form,
                    files={
                        "reference_audio": (
                            reference_audio_filename,
                            reference_audio,
                            reference_audio_media_type,
                        )
                    },
                    timeout=self.generation_timeout_s,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SkulkApiError(
                "POST",
                "/v1/audio/speech",
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        elapsed = time.monotonic() - start
        if response.status_code >= 400:
            raise SkulkApiError(
                "POST", "/v1/audio/speech", response.status_code, response.text
            )
        return AudioSpeechExecution(
            audio=response.content,
            media_type=_base_media_type(response.headers.get("content-type", "")),
            elapsed_s=elapsed,
            response_format=response_format,
            chunks=1 if response.content else 0,
        )

    def audio_transcription(
        self,
        *,
        model_id: str,
        audio: bytes,
        filename: str,
        media_type: str,
        response_format: str = "json",
        language: str | None = None,
        prompt: str | None = None,
    ) -> AudioTranscriptionExecution:
        """Transcribe one audio payload with OpenAI's transcriptions endpoint."""

        data: dict[str, str] = {
            "model": model_id,
            "response_format": response_format,
        }
        if language is not None:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        start = time.monotonic()
        try:
            response = self._client.post(
                "/v1/audio/transcriptions",
                data=data,
                files={"file": (filename, audio, media_type)},
                timeout=self.generation_timeout_s,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SkulkApiError(
                "POST",
                "/v1/audio/transcriptions",
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        elapsed = time.monotonic() - start
        if response.status_code >= 400:
            raise SkulkApiError(
                "POST",
                "/v1/audio/transcriptions",
                response.status_code,
                response.text,
            )
        raw, text = _transcription_payload_text(response, response_format)
        return AudioTranscriptionExecution(
            text=text,
            media_type=_base_media_type(response.headers.get("content-type", "")),
            elapsed_s=elapsed,
            response_format=response_format,
            raw_response=raw,
        )

    def audio_translation(
        self,
        *,
        model_id: str,
        audio: bytes,
        filename: str,
        media_type: str,
        response_format: str = "json",
        language: str | None = None,
        prompt: str | None = None,
    ) -> AudioTranscriptionExecution:
        """Translate one audio payload to English with the audio translations API."""

        data: dict[str, str] = {
            "model": model_id,
            "response_format": response_format,
        }
        if language is not None:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        start = time.monotonic()
        try:
            response = self._client.post(
                "/v1/audio/translations",
                data=data,
                files={"file": (filename, audio, media_type)},
                timeout=self.generation_timeout_s,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SkulkApiError(
                "POST",
                "/v1/audio/translations",
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        elapsed = time.monotonic() - start
        if response.status_code >= 400:
            raise SkulkApiError(
                "POST",
                "/v1/audio/translations",
                response.status_code,
                response.text,
            )
        raw, text = _transcription_payload_text(response, response_format)
        return AudioTranscriptionExecution(
            text=text,
            media_type=_base_media_type(response.headers.get("content-type", "")),
            elapsed_s=elapsed,
            response_format=response_format,
            raw_response=raw,
        )

    def streaming_audio_transcription(
        self,
        *,
        model_id: str,
        audio: bytes,
        filename: str,
        media_type: str,
        language: str | None = None,
        prompt: str | None = None,
        cancel_after_deltas: int = 0,
    ) -> StreamingAudioTranscriptionExecution:
        """Stream typed SSE transcript events for one bounded audio upload."""

        if cancel_after_deltas < 0:
            raise ValueError("cancel_after_deltas must be non-negative")
        data: dict[str, str] = {"model": model_id, "stream": "true"}
        if language is not None:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        start = time.monotonic()
        events: list[dict[str, object]] = []
        event_types: list[str] = []
        event_arrival_s: list[float] = []
        text_parts: list[str] = []
        first_transcript_s: float | None = None
        canceled = False
        try:
            with self._client.stream(
                "POST",
                "/v1/audio/transcriptions",
                data=data,
                files={"file": (filename, audio, media_type)},
                timeout=httpx.Timeout(
                    timeout=None,
                    connect=self.request_timeout_s,
                    read=self.stream_read_timeout_s,
                    write=self.request_timeout_s,
                    pool=self.request_timeout_s,
                ),
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise SkulkApiError(
                        "POST", "/v1/audio/transcriptions", response.status_code, body
                    )
                if _base_media_type(response.headers.get("content-type", "")) != "text/event-stream":
                    raise TypeError("Streaming transcription did not return SSE")
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line.removeprefix("data: "))
                    if not isinstance(payload, dict):
                        raise TypeError("Streaming transcription event was not an object")
                    event = cast(dict[str, object], payload)
                    event_type = event.get("type")
                    if not isinstance(event_type, str):
                        raise TypeError("Streaming transcription event omitted type")
                    arrival = time.monotonic() - start
                    events.append(event)
                    event_types.append(event_type)
                    event_arrival_s.append(arrival)
                    if event_type == "transcription.error":
                        raise SkulkApiError(
                            "POST",
                            "/v1/audio/transcriptions",
                            200,
                            str(event.get("message") or event),
                        )
                    if event_type == "transcription.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise TypeError("Transcription delta omitted text")
                        if first_transcript_s is None:
                            first_transcript_s = arrival
                        text_parts.append(delta)
                        if cancel_after_deltas and len(text_parts) >= cancel_after_deltas:
                            canceled = True
                            break
                    if event_type == "transcription.completed":
                        completed_text = event.get("text")
                        if isinstance(completed_text, str):
                            text_parts = [completed_text]
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SkulkApiError(
                "POST",
                "/v1/audio/transcriptions",
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return StreamingAudioTranscriptionExecution(
            text="".join(text_parts),
            elapsed_s=time.monotonic() - start,
            first_transcript_s=first_transcript_s,
            input_bytes=len(audio),
            transcript_deltas=sum(
                event_type == "transcription.delta" for event_type in event_types
            ),
            event_types=event_types,
            event_arrival_s=event_arrival_s,
            events=events,
            canceled=canceled,
        )

    def realtime_transcription(
        self,
        *,
        model_id: str,
        pcm16: bytes,
        sample_rate: int,
        frame_duration_ms: int = 100,
        pace_audio: bool = True,
        cancel_after_frames: int = 0,
        fabric_chain: bool = False,
        response_model_id: str | None = None,
        response_tts_model_id: str | None = None,
        response_voice: str | None = None,
    ) -> RealtimeTranscriptionExecution:
        """Transcribe PCM16 through realtime STT or a typed Fabric speech chain.

        Args:
            model_id: Mounted realtime STT model selected for the session.
            pcm16: Signed little-endian mono PCM16 bytes without a container.
            sample_rate: Input sample rate in hertz.
            frame_duration_ms: Duration represented by each append event.
            pace_audio: Sleep between frames to reproduce microphone cadence.
            cancel_after_frames: Close without commit after this many frames.
            fabric_chain: Use the explicit Fabric speech-chain endpoint.
            response_model_id: Optional mounted chat participant.
            response_tts_model_id: Optional mounted TTS participant.
            response_voice: Optional voice selected for response TTS.

        Returns:
            Realtime transcript lifecycle, timing, and input counters.

        Raises:
            ValueError: If the PCM or framing inputs are invalid.
            SkulkApiError: If the WebSocket or realtime protocol fails.
        """

        if not pcm16 or len(pcm16) % 2 != 0:
            raise ValueError("realtime PCM16 input must contain complete samples")
        if not 8_000 <= sample_rate <= 96_000:
            raise ValueError("realtime sample rate must be between 8000 and 96000 Hz")
        if not 20 <= frame_duration_ms <= 1000:
            raise ValueError("realtime frame duration must be between 20 and 1000 ms")
        if cancel_after_frames < 0:
            raise ValueError("cancel_after_frames must be non-negative")
        if fabric_chain and not response_model_id:
            raise ValueError("fabric speech chain requires response_model_id")
        if response_tts_model_id and not response_model_id:
            raise ValueError("response TTS requires response_model_id")

        samples_per_frame = max(1, sample_rate * frame_duration_ms // 1000)
        bytes_per_frame = samples_per_frame * 2
        frames = [
            pcm16[offset : offset + bytes_per_frame]
            for offset in range(0, len(pcm16), bytes_per_frame)
        ]
        started_at = time.monotonic()
        first_transcript_s: float | None = None
        transcript_deltas = 0
        event_types: list[str] = []
        path = "/v1/fabric/chains/speech" if fabric_chain else "/v1/realtime"
        url = _realtime_url(self.base_url, model_id, fabric_chain=fabric_chain)
        transcript = ""
        assistant_parts: list[str] = []
        response_audio_parts: list[bytes] = []
        response_audio_chunks = 0

        try:
            with websocket_client.connect(
                url,
                open_timeout=self.request_timeout_s,
                close_timeout=self.request_timeout_s,
                ping_interval=20.0,
                ping_timeout=self.stream_read_timeout_s,
                max_size=_REALTIME_MAX_MESSAGE_BYTES,
                proxy=None,
            ) as connection:
                created = _receive_realtime_event(
                    connection,
                    timeout_s=self.stream_read_timeout_s,
                )
                created_type = str(created["type"])
                event_types.append(created_type)
                if created_type in {
                    "error",
                    "conversation.item.input_audio_transcription.failed",
                }:
                    raise SkulkApiError("WS", path, 0, _realtime_error_message(created))
                if created_type != "session.created":
                    raise TypeError(
                        f"Expected session.created, received {created_type!r}"
                    )

                session: dict[str, object] = {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": sample_rate,
                            },
                            "transcription": {"model": model_id},
                            "turn_detection": None,
                            "noise_reduction": None,
                        }
                    },
                    "include": [],
                }
                if response_model_id is not None:
                    response: dict[str, object] = {"model": response_model_id}
                    if response_tts_model_id is not None:
                        response["tts_model"] = response_tts_model_id
                    if response_voice is not None:
                        response["voice"] = response_voice
                    session["response"] = response
                connection.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": session,
                        },
                        separators=(",", ":"),
                    )
                )
                updated = _receive_realtime_event(
                    connection,
                    timeout_s=self.stream_read_timeout_s,
                )
                updated_type = str(updated["type"])
                event_types.append(updated_type)
                if updated_type != "session.updated":
                    if updated_type in {
                        "error",
                        "conversation.item.input_audio_transcription.failed",
                    }:
                        raise SkulkApiError(
                            "WS", path, 0, _realtime_error_message(updated)
                        )
                    raise TypeError(
                        f"Expected session.updated, received {updated_type!r}"
                    )

                for sent_frames, frame in enumerate(frames, start=1):
                    connection.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(frame).decode("ascii"),
                            },
                            separators=(",", ":"),
                        )
                    )
                    if cancel_after_frames and sent_frames >= cancel_after_frames:
                        connection.close(1000, "harness cancellation probe")
                        return RealtimeTranscriptionExecution(
                            text="",
                            elapsed_s=time.monotonic() - started_at,
                            first_transcript_s=None,
                            input_bytes=sum(len(item) for item in frames[:sent_frames]),
                            input_frames=sent_frames,
                            transcript_deltas=0,
                            event_types=event_types,
                            canceled=True,
                        )
                    if pace_audio:
                        time.sleep(len(frame) / (sample_rate * 2))

                connection.send(
                    json.dumps(
                        {"type": "input_audio_buffer.commit"},
                        separators=(",", ":"),
                    )
                )
                while True:
                    event = _receive_realtime_event(
                        connection,
                        timeout_s=self.stream_read_timeout_s,
                    )
                    event_type = str(event["type"])
                    event_types.append(event_type)
                    if event_type == "input_audio_buffer.committed":
                        continue
                    if (
                        event_type
                        == "conversation.item.input_audio_transcription.delta"
                    ):
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise TypeError("Realtime transcript delta was not text")
                        transcript_deltas += 1
                        if first_transcript_s is None:
                            first_transcript_s = time.monotonic() - started_at
                        continue
                    if event_type == "conversation.item.input_audio_transcription.completed":
                        final_transcript = event.get("transcript")
                        if not isinstance(final_transcript, str):
                            raise TypeError("Realtime final transcript was not text")
                        transcript = final_transcript
                        if first_transcript_s is None:
                            first_transcript_s = time.monotonic() - started_at
                        if response_model_id is None:
                            return RealtimeTranscriptionExecution(
                                text=transcript,
                                elapsed_s=time.monotonic() - started_at,
                                first_transcript_s=first_transcript_s,
                                input_bytes=len(pcm16),
                                input_frames=len(frames),
                                transcript_deltas=transcript_deltas,
                                event_types=event_types,
                            )
                        continue
                    if event_type in {"response.created", "response.audio.done"}:
                        continue
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise TypeError("Realtime assistant delta was not text")
                        assistant_parts.append(delta)
                        continue
                    if event_type == "response.output_text.done":
                        text = event.get("text")
                        if not isinstance(text, str):
                            raise TypeError("Realtime assistant final text was not text")
                        assistant_parts = [text]
                        continue
                    if event_type == "response.audio.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise TypeError("Realtime response audio delta was not base64")
                        response_audio_parts.append(base64.b64decode(delta, validate=True))
                        response_audio_chunks += 1
                        continue
                    if event_type == "response.done":
                        response_payload = event.get("response")
                        if not isinstance(response_payload, dict):
                            raise TypeError("Realtime response.done omitted response status")
                        status = response_payload.get("status")
                        if not isinstance(status, str):
                            raise TypeError("Realtime response status was not text")
                        return RealtimeTranscriptionExecution(
                            text=transcript,
                            elapsed_s=time.monotonic() - started_at,
                            first_transcript_s=first_transcript_s,
                            input_bytes=len(pcm16),
                            input_frames=len(frames),
                            transcript_deltas=transcript_deltas,
                            event_types=event_types,
                            assistant_text="".join(assistant_parts),
                            response_audio=b"".join(response_audio_parts),
                            response_audio_chunks=response_audio_chunks,
                            response_status=status,
                        )
                    if event_type in {
                        "error",
                        "conversation.item.input_audio_transcription.failed",
                    }:
                        raise SkulkApiError(
                            "WS", path, 0, _realtime_error_message(event)
                        )
                    raise TypeError(f"Unexpected realtime event {event_type!r}")
        except SkulkApiError:
            raise
        except (
            WebSocketException,
            OSError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            raise SkulkApiError(
                "WS",
                path,
                0,
                f"{type(exc).__name__}: {exc}",
            ) from exc

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
            generated_chars=len(text),
            chunks=1 if text else 0,
            approx_output_tokens=max(1, round(len(text) / 4)) if text else 0,
        )
        stats = event.get("generation_stats")
        if isinstance(stats, dict):
            metrics = metrics.model_copy(
                update={
                    "skulk_prompt_tps": _float_or_none(stats.get("prompt_tps")),
                    "skulk_generation_tps": _float_or_none(stats.get("generation_tps")),
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
        if not isinstance(supervisor_runners, list) or not isinstance(
            state_runners, dict
        ):
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
            if (
                runner_id
                and state_status
                and local_status
                and state_status != local_status
            ):
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


def _base_media_type(value: str) -> str:
    """Return only the MIME type portion of a Content-Type header."""

    return value.split(";", 1)[0].strip().lower()


def _transcription_payload_text(
    response: httpx.Response, response_format: str
) -> tuple[dict[str, object] | str, str]:
    """Normalize the many transcription response formats to text for scoring."""

    if response_format in {"json", "verbose_json"}:
        raw = response.json()
        if not isinstance(raw, dict):
            raise TypeError(f"Unexpected transcription JSON payload: {raw!r}")
        text = raw.get("text")
        return raw, text if isinstance(text, str) else ""
    if response_format == "ndjson":
        parts: list[str] = []
        for line in response.text.splitlines():
            item = _safe_json_object(line)
            if item is None:
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return response.text, " ".join(parts).strip()
    return response.text, response.text


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


def _extract_stream_logprobs(event: dict[str, object]) -> tuple[int, int]:
    """Count per-token logprob entries in one streaming chunk.

    Skulk (OpenAI-compatible) puts logprobs at the choice level, as a sibling of
    ``delta``: ``choices[0].logprobs.content`` is a list of per-token entries,
    each optionally carrying ``top_logprobs`` alternatives. Returns
    ``(logprob_tokens, top_logprob_tokens)`` for this chunk; ``(0, 0)`` when the
    chunk carries no logprobs (the common case when they were not requested).
    """
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0, 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0, 0
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, dict):
        return 0, 0
    content = logprobs.get("content")
    if not isinstance(content, list):
        return 0, 0
    logprob_tokens = 0
    top_logprob_tokens = 0
    for entry in content:
        if not isinstance(entry, dict) or "logprob" not in entry:
            continue
        logprob_tokens += 1
        top = entry.get("top_logprobs")
        if isinstance(top, list) and top:
            top_logprob_tokens += 1
    return logprob_tokens, top_logprob_tokens


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
