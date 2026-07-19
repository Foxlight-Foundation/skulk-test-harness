"""Typed configuration and report models for the Skulk harness."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PlacementStrategy = Literal["minimum", "single", "exact"]
ShardingMode = Literal["Pipeline", "Tensor"]
InstanceMeta = Literal["MlxRing", "MlxJaccl", "LlamaRpc"]
AudioResponseFormat = Literal["mp3", "wav", "flac", "ogg", "opus"]
TranscriptionResponseFormat = Literal[
    "json", "text", "verbose_json", "srt", "vtt", "ndjson"
]
OwnerTopology = Literal["any", "local_remote"]
SpeechOwnerTopology = OwnerTopology
TestKind = Literal[
    "chat",
    "code",
    "artifact",
    "tool",
    "cancel",
    "concurrent",
    "error",
    "embedding",
    "audio_speech",
    "audio_speech_streaming",
    "audio_speech_pressure",
    "audio_voices",
    "audio_transcription",
    "audio_transcription_streaming",
    "realtime_transcription",
    "realtime_conversation",
    "fabric_speech_chain",
    "speech_roundtrip",
    "speech_translation_roundtrip",
    "speech_reference_roundtrip",
    "vision_data_plane",
]
RunMode = Literal["plan", "execute"]
IssueSeverity = Literal["info", "warning", "error"]


class HarnessBaseModel(BaseModel):
    """Base model with strict extra-field handling for harness config."""

    model_config = ConfigDict(extra="forbid")


class ClusterNode(HarnessBaseModel):
    """SSH-reachable control surface for one physical cluster node.

    The harness uses this to crash and relaunch a node's skulk process during
    the stability suites (failover/churn). It is intentionally separate from the
    libp2p node identity: ``ssh_host`` is an SSH alias/hostname, while the live
    libp2p ``node_id`` is ephemeral and discovered from cluster state at runtime.
    """

    ssh_host: str = Field(
        description="SSH host or alias used to reach this node (passed to `ssh`)."
    )
    repo_path: str | None = Field(
        default=None,
        description=(
            "Backward-compatible Skulk checkout path used to build a default "
            "relaunch command when relaunch_command is not set."
        ),
    )
    kill_command: str | None = Field(
        default=None,
        description="Optional shell command used to stop Skulk on this node.",
    )
    relaunch_command: str | None = Field(
        default=None,
        description="Optional shell command used to relaunch Skulk on this node.",
    )


class FleetLock(HarnessBaseModel):
    """Config for the git-backed fleet lease (exclusive shared-fleet access).

    When two agents work the same codebase and both deploy branches to one
    shared test fleet, two end-to-end runs at once corrupt each other because
    Skulk does not support mixed-version clusters. The fleet lease is a mutex
    over the fleet, backed by a small JSON file in a shared git repo: acquiring
    means committing a claim and pushing, and a rejected non-fast-forward push
    means another agent won the race.

    This section is optional. With no ``fleet_lock`` config the lease is disabled
    and every fleet-lock operation is a no-op, so community users of the public
    harness are unaffected.
    """

    remote: str = Field(
        description=(
            "Git remote URL of the coordination repo that holds the lock file "
            "(e.g. the private foxlight-docs repo). The harness clones it into "
            "cache_dir and pushes lock updates to it."
        )
    )
    holder: str = Field(
        description=(
            "This agent's stable name (e.g. 'claude' or 'codex'). Identifies who "
            "holds the fleet so the other agent's runs are refused, and so only "
            "the holder can extend or release without --force."
        )
    )
    branch: str = Field(
        default="main",
        description="Branch in the coordination repo that carries the lock file.",
    )
    path: str = Field(
        default="coordination/fleet-lock.json",
        description="Path to the lock JSON within the coordination repo.",
    )
    default_ttl_s: float = Field(
        default=1800.0,
        gt=0,
        description=(
            "Default lease lifetime in seconds. A lock past its expiry is treated "
            "as free, so a crashed run that never releases cannot wedge the fleet. "
            "Long batteries should extend the lease to stay live."
        ),
    )
    cache_dir: Path | None = Field(
        default=None,
        description=(
            "Local directory for the coordination-repo clone. Defaults to "
            "~/.cache/skulk-test-harness/fleet-lock when unset."
        ),
    )


class HarnessConfig(HarnessBaseModel):
    """Top-level local coordinator settings."""

    api_base_url: str = "http://localhost:52415"
    request_timeout_s: float = 30.0
    generation_timeout_s: float = 1800.0
    stream_read_timeout_s: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Maximum seconds to wait for the next streaming response byte before "
            "treating the request as stalled. This is intentionally separate "
            "from generation_timeout_s, which bounds long healthy generations."
        ),
    )
    placement_ready_timeout_s: float = 1800.0
    placement_ready_total_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Hard wall-clock ceiling on one model's ENTIRE readiness wait, "
            "counted from the first placement appearing and spanning every "
            "re-anchored replacement instance. Each replacement still gets a "
            "fresh placement_ready_timeout_s allowance, but placement churn "
            "(the cluster re-placing over and over) would otherwise extend the "
            "wait without bound. Unset (the default) derives the ceiling as "
            "2 * placement_ready_timeout_s + placement_appearance_timeout_s: "
            "a full allowance for the original placement, a full allowance for "
            "one replacement, and one appearance window for the gap between "
            "them. Hitting the ceiling fails the wait loudly with "
            "unavailable_reason 'churn'."
        ),
    )
    placement_appearance_timeout_s: float = Field(
        default=300.0,
        gt=0,
        description=(
            "Maximum seconds to wait for a newly requested placement to appear "
            "in cluster state. Runner readiness can still take "
            "placement_ready_timeout_s, but a placement that never appears is a "
            "refusal/give-up signal. Keep this long enough for large cached "
            "models to surface under placement/teardown contention."
        ),
    )
    store_download_timeout_s: float = 14400.0
    store_delete_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Maximum seconds to spend on best-effort staged-model eviction. "
            "Eviction is hygiene, not correctness, so it must not wedge a run."
        ),
    )
    poll_interval_s: float = 2.0
    preview_settle_attempts: int = Field(
        default=8,
        ge=1,
        description=(
            "How many times to re-request placement previews when none are yet "
            "viable, polling at poll_interval_s. Bridges the transient where a "
            "just-torn-down instance's freed memory has not yet reflected in "
            "gossiped telemetry (the tear-down-then-place matrix loop), which the "
            "cluster clears within a few seconds."
        ),
    )
    output_dir: Path = Path("runs")
    model_sets_path: Path = Path("configs/model_sets.yaml")
    test_sets_path: Path = Path("configs/test_sets.yaml")
    cluster_nodes: dict[str, ClusterNode] = Field(
        default_factory=dict,
        description=(
            "Map of friendly node name to its SSH control surface. Required for "
            "the failover and churn stability suites; keyed by the friendly name "
            "reported in cluster state (nodeIdentities[<nodeId>].friendlyName)."
        ),
    )
    fleet_lock: FleetLock | None = Field(
        default=None,
        description=(
            "Optional git-backed fleet lease for coordinating exclusive access "
            "to a shared test fleet across multiple agents. Disabled (no-op) when "
            "absent, so single-operator use is unaffected."
        ),
    )


class HuggingFaceSeed(HarnessBaseModel):
    """Optional seed that can add a model card from Hugging Face."""

    model_id: str
    reason: str = ""
    require_mlx_community: bool = True


class ModelSelector(HarnessBaseModel):
    """Catalog selector used to expand a model set from Skulk's model list."""

    source: Literal["catalog", "store", "both"] = "catalog"
    family: str | None = None
    id_contains: str | None = None
    id_regex: str | None = None
    tags_any: list[str] = Field(default_factory=list)
    tasks_any: list[str] = Field(default_factory=list)
    capabilities_any: list[str] = Field(default_factory=list)
    served_spec_types_any: list[str] = Field(
        default_factory=list,
        description=(
            "Optional runtime.served_spec_type values to match, such as "
            "draft_mtp, draft_simple, or draft_eagle3."
        ),
    )
    require_audio_streaming: bool = Field(
        default=False,
        description=(
            "When true, select only catalog/store entries whose audio metadata "
            "declares supports_streaming=true."
        ),
    )
    require_audio_realtime: bool = Field(
        default=False,
        description=(
            "When true, select only catalog/store entries whose audio metadata "
            "declares both supports_streaming=true and supports_realtime=true."
        ),
    )
    max_models: int | None = Field(default=None, ge=1)


class ModelSet(HarnessBaseModel):
    """Named collection of explicit model IDs and/or catalog selectors."""

    name: str
    description: str = ""
    models: list[str] = Field(default_factory=list)
    selectors: list[ModelSelector] = Field(default_factory=list)
    huggingface_seeds: list[HuggingFaceSeed] = Field(default_factory=list)


class ModelSetFile(HarnessBaseModel):
    """YAML file containing named model sets."""

    model_sets: dict[str, ModelSet]

    @field_validator("model_sets")
    @classmethod
    def _names_match_keys(cls, value: dict[str, ModelSet]) -> dict[str, ModelSet]:
        for key, model_set in value.items():
            if model_set.name != key:
                raise ValueError(f"model set key {key!r} must match name field")
        return value


class SuccessCriteria(HarnessBaseModel):
    """Heuristic output checks for one prompt-style test."""

    min_chars: int = Field(default=1, ge=0)
    min_code_block_chars: int = Field(default=0, ge=0)
    min_list_items: int = Field(
        default=0,
        ge=0,
        description=(
            "When > 0, require at least this many structured list items. "
            "Accepts Markdown bullets, common unicode bullets/dashes, and "
            "numbered/lettered list markers so the test checks structure instead "
            "of one exact glyph."
        ),
    )
    min_generated_chars: int = Field(
        default=0,
        ge=0,
        description=(
            "When > 0, require at least this many characters across visible "
            "content plus separated reasoning. Use for reasoning-model gates "
            "where a healthy response may be entirely in reasoning_content."
        ),
    )
    min_tool_calls: int = Field(default=0, ge=0)
    in_order_integers: int = Field(
        default=0,
        ge=0,
        description=(
            "When > 0, assert the integers 1..N this size that appear in the "
            "output arrive in strictly ascending emission order. Catches "
            "token/sub-word transposition (e.g. a data-plane delivery reorder) "
            "that presence-only checks are blind to."
        ),
    )
    required_substrings: list[str] = Field(default_factory=list)
    forbidden_substrings: list[str] = Field(default_factory=list)
    required_regexes: list[str] = Field(default_factory=list)
    expected_tool_calls: list["ExpectedToolCall"] = Field(default_factory=list)
    require_html_artifact: bool = False
    require_logprobs: bool = Field(
        default=False,
        description=(
            "When true, assert the stream returned per-token logprobs (i.e. the "
            "test must also set top_logprobs). Verifies the logprobs capability "
            "end-to-end through the API and the serving runner -- the key check "
            "for llama.cpp logprob parity, where a build that cannot serve "
            "logprobs would yield none."
        ),
    )
    min_reasoning_chars: int = Field(
        default=0,
        ge=0,
        description=(
            "When > 0, assert the model emitted at least this many characters of "
            "SEPARATED reasoning (the reasoning_content channel), not just visible "
            "content. Verifies a reasoning model's thinking was parsed into its "
            "own channel rather than swallowed into content or dropped -- the key "
            "check for the Gemma 4 served channel parser, where a regression would "
            "either lose the split or leak the thought text into content."
        ),
    )
    forbid_in_reasoning: bool = Field(
        default=True,
        description=(
            "Also apply forbidden_substrings to the reasoning channel, not only "
            "the visible content. On by default so a forbidden marker that leaks "
            "into reasoning is caught too; the Gemma channel-marker check "
            "(forbidden '<|channel>' / '<channel|>') relies on this to catch a "
            "leak in either channel."
        ),
    )
    min_wall_tps: float | None = Field(
        default=None,
        ge=0,
        description=(
            "When set, assert steady-state decode throughput (wall_tps, tokens / "
            "decode time excluding TTFT) is at least this value. The token "
            "estimate includes visible content and separated reasoning so "
            "reasoning-only generations do not read as zero throughput. A floor "
            "calibrated ABOVE the model's non-speculative decode rate makes a "
            "SILENT speculative/MTP fallback visible. Hardware- and model-specific; "
            "set per benchmark cell for the target node, and keep it conservative."
        ),
    )
    min_audio_bytes: int = Field(
        default=0,
        ge=0,
        description=(
            "For speech synthesis tests, require at least this many encoded "
            "audio bytes. Use this instead of min_chars for binary audio output."
        ),
    )
    max_word_error_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "For speech roundtrip tests, require the STT transcript's word error "
            "rate against the synthesized prompt to be no greater than this value."
        ),
    )
    min_stream_chunks: int = Field(
        default=0,
        ge=0,
        description=(
            "For streaming tests, require at least this many response chunks. "
            "For TTS streaming this is measured from HTTP audio byte chunks."
        ),
    )
    max_first_byte_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional maximum time to first streamed byte/token in seconds. "
            "Leave unset for cold-load speech tests where model startup dominates."
        ),
    )
    min_stream_span_s: float = Field(
        default=0.0,
        ge=0,
        description=(
            "For streaming tests, require at least this many seconds between the "
            "first and last streamed response chunk. This distinguishes genuine "
            "incremental delivery from a finished response that was merely split "
            "into multiple HTTP chunks."
        ),
    )
    min_transcript_deltas: int = Field(
        default=0,
        ge=0,
        description=(
            "For realtime transcription tests, require at least this many "
            "incremental transcript delta events before the final transcript."
        ),
    )


class ExpectedToolCall(HarnessBaseModel):
    """Tool-call expectation used to score model-emitted function calls."""

    name: str
    required_arguments: list[str] = Field(default_factory=list)
    arguments_contains: dict[str, object] = Field(default_factory=dict)
    argument_substrings: dict[str, str] = Field(default_factory=dict)


class ToolMock(HarnessBaseModel):
    """Static tool result returned by the harness during tool-call round trips."""

    name: str
    content: str


class PromptImage(HarnessBaseModel):
    """OpenAI-style image input attached to a chat prompt."""

    url: str = Field(
        description=(
            "Image URL or data URL sent as an OpenAI `image_url` content part."
        )
    )
    detail: Literal["auto", "low", "high"] | None = None


class PromptTest(HarnessBaseModel):
    """One prompt-style test case."""

    name: str
    kind: TestKind = "chat"
    description: str = ""
    system: str | None = None
    prompt: str
    max_tokens: int = Field(default=512, ge=1)
    temperature: float | None = Field(default=0.2, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    enable_thinking: bool | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    prompt_repetitions: int = Field(
        default=1,
        ge=1,
        description=(
            "Repeat the prompt text before sending it. Used by admission tests "
            "to build oversized requests without embedding huge YAML blobs."
        ),
    )
    images: list[PromptImage] = Field(default_factory=list)
    tools: list[dict[str, object]] = Field(default_factory=list)
    tool_choice: str | dict[str, object] | None = None
    parallel_tool_calls: bool | None = None
    tool_mocks: list[ToolMock] = Field(default_factory=list)
    cancel_after_chunks: int = Field(
        default=0,
        ge=0,
        description=(
            "For `kind: cancel`, close the stream after this many content or "
            "reasoning chunks, then verify a follow-up request still succeeds."
        ),
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        le=256,
        description=(
            "For `kind: concurrent`, the number of simultaneous in-flight "
            "requests (worker threads, each with its own client and connection "
            "pool). This is the batch pressure the placed model sees at once."
        ),
    )
    concurrent_requests_per_worker: int = Field(
        default=1,
        ge=1,
        le=1000,
        description=(
            "For `kind: concurrent`, requests each worker issues sequentially. "
            "Total requests = `concurrency * concurrent_requests_per_worker`; "
            "raising this sustains the load past a single wave so the engine "
            "reaches steady-state batching instead of a transient burst."
        ),
    )
    followup_prompt: str | None = Field(
        default=None,
        description=(
            "Optional health-check prompt for cancellation and expected-error tests."
        ),
    )
    expected_error_statuses: list[int] = Field(
        default_factory=list,
        description=(
            "For `kind: error`, acceptable HTTP status codes. Empty means any "
            "SkulkApiError is acceptable."
        ),
    )
    expected_error_substrings: list[str] = Field(
        default_factory=list,
        description="For `kind: error`, substrings that must appear in the error body.",
    )
    embedding_input: str | list[str] | None = Field(
        default=None,
        description=(
            "For `kind: embedding`, request input. Defaults to `prompt` when unset."
        ),
    )
    expected_embedding_dimensions: int | None = Field(
        default=None,
        ge=1,
        description="For `kind: embedding`, expected vector dimensionality.",
    )
    min_embedding_norm: float = Field(
        default=0.0,
        ge=0,
        description="For `kind: embedding`, minimum L2 norm for every vector.",
    )
    audio_response_format: AudioResponseFormat = Field(
        default="wav",
        description="For speech tests, encoded audio format requested from TTS.",
    )
    speech_voice: str | None = Field(
        default=None,
        description="Optional TTS voice name passed to `/v1/audio/speech`.",
    )
    speech_speed: float | None = Field(
        default=None,
        gt=0,
        description="Optional TTS speed multiplier passed to `/v1/audio/speech`.",
    )
    reference_model_id: str | None = Field(
        default=None,
        description=(
            "Donor TTS model used to synthesize the conditioning clip for "
            "`kind: speech_reference_roundtrip`."
        ),
    )
    reference_text: str | None = Field(
        default=None,
        description=(
            "Transcript spoken by the donor model for reference conditioning; "
            "defaults to the test prompt."
        ),
    )
    expected_voice_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Voice identifiers required from `GET /v1/audio/voices` for "
            "`kind: audio_voices`."
        ),
    )
    speech_streaming_interval: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional streaming_interval hint passed to `/v1/audio/speech` for "
            "`kind: audio_speech_streaming`."
        ),
    )
    speech_concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Concurrent workers for `kind: audio_speech_pressure`.",
    )
    speech_requests_per_worker: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Streaming TTS requests issued sequentially by each worker.",
    )
    speech_owner_count: int = Field(
        default=1,
        ge=1,
        description=(
            "Distinct reachable API owners used by a pressure test. Owners are "
            "discovered from cluster diagnostics and assigned round-robin."
        ),
    )
    speech_owner_topology: SpeechOwnerTopology = Field(
        default="any",
        description=(
            "Owner selection policy for speech pressure. `local_remote` chooses "
            "one API owner on the TTS serving node and the remaining owners away "
            "from it, proving both DATA routing paths deterministically."
        ),
    )
    speech_assert_data_plane_diagnostics: bool = Field(
        default=False,
        description=(
            "Capture DATA diagnostics before and after pressure, require drained "
            "stream/egress gauges, reject new anomaly counters, and persist a "
            "sanitized diagnostics sidecar."
        ),
    )
    speech_chat_model_id: str | None = Field(
        default=None,
        description=(
            "Optional secondary text-generation model mounted for concurrent "
            "chat-plus-TTS DATA pressure."
        ),
    )
    speech_chat_concurrency: int = Field(
        default=0,
        ge=0,
        le=32,
        description=(
            "Concurrent streaming chat workers run beside TTS pressure. Zero "
            "keeps the pressure test speech-only."
        ),
    )
    speech_chat_prompt: str | None = Field(
        default=None,
        description="Prompt used by concurrent chat workers in mixed pressure tests.",
    )
    speech_slow_workers: int = Field(
        default=0,
        ge=0,
        description="Leading pressure workers that intentionally read audio slowly.",
    )
    speech_slow_reader_delay_s: float = Field(
        default=0.0,
        ge=0,
        description="Delay after each received audio chunk for slow workers.",
    )
    input_audio_path: Path | None = Field(
        default=None,
        description=(
            "For `kind: audio_transcription`, path to an audio fixture. Relative "
            "paths resolve from the current harness working directory."
        ),
    )
    input_audio_mime_type: str | None = Field(
        default=None,
        description=(
            "Optional MIME type sent for `input_audio_path` multipart uploads. "
            "When omitted, the harness infers it from the fixture extension."
        ),
    )
    transcription_model_id: str | None = Field(
        default=None,
        description=(
            "For speech transcription/translation roundtrips, optional explicit "
            "STT model. When "
            "unset, the harness selects the first live catalog model advertising "
            "STT support."
        ),
    )
    speech_synthesis_model_id: str | None = Field(
        default=None,
        description=(
            "For `kind: realtime_transcription`, optional TTS model used to "
            "generate the semantic PCM16 fixture. When unset, the harness "
            "selects the first live catalog model advertising TTS support."
        ),
    )
    realtime_response_model_id: str | None = Field(
        default=None,
        description=(
            "For conversational realtime or Fabric-chain tests, mounted chat "
            "participant that receives each final transcript."
        ),
    )
    realtime_response_tts_model_id: str | None = Field(
        default=None,
        description=(
            "For conversational realtime or Fabric-chain tests, mounted TTS "
            "participant that speaks assistant responses."
        ),
    )
    realtime_frame_duration_ms: int = Field(
        default=100,
        ge=20,
        le=1000,
        description=(
            "PCM16 frame duration sent to `/v1/realtime`. The dashboard uses "
            "100 ms frames."
        ),
    )
    realtime_pace_audio: bool = Field(
        default=True,
        description=(
            "Send realtime PCM frames at their media cadence instead of as a burst."
        ),
    )
    realtime_cancel_after_frames: int = Field(
        default=0,
        ge=0,
        description=(
            "When positive, run a disconnect probe after this many PCM frames "
            "before the successful realtime transcription requests."
        ),
    )
    realtime_assert_provider_diagnostics: bool = Field(
        default=False,
        description=(
            "Require stt.realtime provider lifecycle/media counters to cover "
            "successful and cancelled sessions and drain active gauges to zero."
        ),
    )
    realtime_turn_count: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Number of server-VAD utterances sent over one persistent realtime "
            "conversation socket."
        ),
    )
    realtime_server_vad: bool = Field(
        default=False,
        description=(
            "Enable server-owned VAD and automatic input commit for realtime "
            "conversation tests."
        ),
    )
    realtime_barge_in: bool = Field(
        default=False,
        description=(
            "Send the next utterance after response audio begins and require the "
            "superseded response to cancel. Requires at least two turns."
        ),
    )
    transcription_response_format: TranscriptionResponseFormat = Field(
        default="json",
        description="Response format requested from `/v1/audio/transcriptions`.",
    )
    transcription_language: str | None = Field(
        default=None,
        description="Optional language hint for transcription requests.",
    )
    transcription_cancel_after_deltas: int = Field(
        default=1,
        ge=0,
        description=(
            "For streaming audio transcription, close a secondary probe after "
            "this many transcript deltas. Zero disables the cancellation probe."
        ),
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description=(
            "Request this many ranked logprob alternatives per token (sets "
            "logprobs=true on the request). Pair with success.require_logprobs."
        ),
    )
    repetitions: int = Field(default=1, ge=1)
    success: SuccessCriteria = Field(default_factory=SuccessCriteria)


class TestSet(HarnessBaseModel):
    """Named set of tests that can be run against any compatible model set."""

    name: str
    description: str = ""
    tests: list[PromptTest]


class TestSetFile(HarnessBaseModel):
    """YAML file containing named test sets."""

    test_sets: dict[str, TestSet]

    @field_validator("test_sets")
    @classmethod
    def _names_match_keys(cls, value: dict[str, TestSet]) -> dict[str, TestSet]:
        for key, test_set in value.items():
            if test_set.name != key:
                raise ValueError(f"test set key {key!r} must match name field")
        return value


class PlacementPolicy(HarnessBaseModel):
    """How the harness should place each model before running tests."""

    strategy: PlacementStrategy = "minimum"
    sharding: ShardingMode = "Pipeline"
    instance_meta: InstanceMeta = "MlxRing"
    min_nodes: int | None = Field(default=None, ge=1)
    excluded_nodes: list[str] = Field(default_factory=list)


class RunSpec(HarnessBaseModel):
    """Concrete run request produced by CLI flags or a natural-language goal."""

    model_set: str
    test_set: str
    mode: RunMode = "plan"
    placement: PlacementPolicy = Field(default_factory=PlacementPolicy)
    ensure_model_cards: bool = True
    ensure_store_downloads: bool = False
    reuse_existing_instances: bool = True
    retain_instances: bool = True
    delete_staged_models: bool = False
    """Evict each model's staged weights from the store after its run (after
    instance teardown). Off by default so normal runs keep the store warm; set
    for benchmark batteries so test models do not accumulate on disk."""
    run_name: str | None = None


class ModelRef(HarnessBaseModel):
    """Resolved model selected for a run."""

    model_id: str
    source: Literal["explicit", "selector", "huggingface_seed"]
    detail: str = ""


class Issue(HarnessBaseModel):
    """Problem or notable condition discovered by the harness."""

    severity: IssueSeverity
    message: str
    model_id: str | None = None
    test_name: str | None = None
    evidence: dict[str, object] = Field(default_factory=dict)


class PlacementResult(HarnessBaseModel):
    """Placement selected or reused for one model."""

    model_id: str
    instance_id: str | None = None
    node_ids: list[str] = Field(default_factory=list)
    runner_ids: list[str] = Field(default_factory=list)
    sharding: str | None = None
    instance_meta: str | None = None
    reused_existing: bool = False
    created_by_harness: bool = False
    ready: bool = False
    terminal_failure: bool = Field(
        default=False,
        description=(
            "Whether at least one assigned runner entered a terminal failed state."
        ),
    )
    runner_failure_messages: list[str] = Field(
        default_factory=list,
        description="Runner failure messages observed while waiting for readiness.",
    )
    unavailable_reason: str | None = Field(
        default=None,
        description=(
            "Why the model has no ready instance, when ``ready`` is False: "
            "``never_appeared`` (the requested placement never surfaced in "
            "cluster state), ``disappeared_without_replacement`` (an instance was "
            "seen and then torn down with no re-placement), or ``ready_timeout`` "
            "(an instance was present the whole time but never reached a "
            "dispatchable runner). ``None`` when ``ready`` is True. Lets the "
            "caller report a precise cause instead of a generic 'never became "
            "ready', and distinguishes a re-placeable give-up from a hard load "
            "failure."
        ),
    )
    readiness_transitions: list[dict[str, object]] = Field(
        default_factory=list,
        description=(
            "Ordered record of every observed change in the model's placement "
            "while waiting for readiness: each entry has the elapsed seconds and "
            "the instances then serving the model (id, ready, terminal_failure). "
            "A ready wait that silently burns its full timeout is diagnosable "
            "from this history alone, without re-deriving it from master logs."
        ),
    )
    protected_instance_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Instance ids that existed for the model BEFORE a forced placement "
            "(reuse_existing_instances=False). They are operator-owned, never "
            "adopted by the readiness wait and never deleted by teardown, so a "
            "harness cell for a model the operator is already running cannot tear "
            "down that live instance."
        ),
    )


class GenerationMetrics(HarnessBaseModel):
    """Wall-clock and Skulk-reported generation metrics."""

    elapsed_s: float
    ttft_s: float | None = None
    output_chars: int = 0
    generated_chars: int = 0
    chunks: int = 0
    approx_output_tokens: int | None = None
    wall_tps: float | None = None
    skulk_prompt_tps: float | None = None
    skulk_generation_tps: float | None = None
    skulk_prompt_tokens: int | None = None
    skulk_generation_tokens: int | None = None
    word_error_rate: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Levenshtein word error rate between the source prompt and the STT "
            "transcript for a non-translation speech roundtrip."
        ),
    )
    # Concurrency-benchmark aggregates (populated only for `kind: concurrent`;
    # None everywhere else, so single-request reports and the ledger are
    # unaffected). For a concurrent run `skulk_generation_tps` above carries the
    # aggregate throughput so existing readers still see one headline number.
    concurrency: int | None = Field(
        default=None,
        description="Simultaneous in-flight requests driven for a concurrent test.",
    )
    concurrent_total_requests: int | None = Field(
        default=None,
        description="Total requests issued across all workers in a concurrent test.",
    )
    concurrent_succeeded: int | None = Field(
        default=None,
        description="Requests that succeeded and met the success criteria under load.",
    )
    concurrent_failed: int | None = Field(
        default=None,
        description="Requests that errored or failed scoring under load.",
    )
    aggregate_generation_tps: float | None = Field(
        default=None,
        description=(
            "Total generated tokens across all concurrent requests divided by "
            "the wall span from first request start to last request end. The "
            "headline concurrency number: where a batching engine on a large "
            "GPU pulls away from its single-stream decode rate."
        ),
    )
    per_request_generation_tps_mean: float | None = Field(
        default=None,
        description="Mean per-request decode tok/s across successful concurrent requests.",
    )
    per_request_generation_tps_p50: float | None = Field(
        default=None,
        description="Median per-request decode tok/s under concurrent load.",
    )
    per_request_generation_tps_p90: float | None = Field(
        default=None,
        description="90th-percentile per-request decode tok/s under concurrent load.",
    )
    ttft_p50_s: float | None = Field(
        default=None,
        description="Median time-to-first-token across concurrent requests.",
    )
    ttft_p90_s: float | None = Field(
        default=None,
        description="90th-percentile time-to-first-token across concurrent requests.",
    )
    wall_span_s: float | None = Field(
        default=None,
        description=(
            "Wall-clock seconds from the first concurrent request starting to "
            "the last one finishing (the denominator for aggregate throughput)."
        ),
    )


class ToolCallRecord(HarnessBaseModel):
    """Normalized function-call payload emitted by Skulk."""

    id: str
    name: str
    arguments_text: str
    arguments: dict[str, object] | None = None
    index: int | None = None


class TestResult(HarnessBaseModel):
    """Result for one test execution against one model."""

    model_id: str
    test_name: str
    kind: TestKind = Field(
        default="chat",
        description=(
            "The test's kind (chat, code, tool, audio_speech, ...), copied from "
            "its PromptTest so a report is self-describing: a downstream reader "
            "(the results ledger) can explain what a suite measures without the "
            "harness config. Defaults to 'chat' for reports predating this field."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Human explanation of what this test checks, copied from its "
            "PromptTest. Empty when the test declares none."
        ),
    )
    repetition: int
    passed: bool
    output_text: str
    reasoning_text: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metrics: GenerationMetrics
    issues: list[Issue] = Field(default_factory=list)
    artifact_path: Path | None = None


class RepoRef(HarnessBaseModel):
    """Git provenance for one repository involved in a run."""

    name: str
    path: str | None = None
    branch: str | None = None
    commit: str | None = None
    dirty: bool | None = None


class SourceContext(HarnessBaseModel):
    """Why the run happened and from which code."""

    run_reason: str = "unspecified"
    visibility: Literal["private", "public"] = "private"
    operator_note: str | None = None
    repositories: list[RepoRef] = Field(default_factory=list)


class RuntimeFingerprint(HarnessBaseModel):
    """The harness process's own runtime, plus the cluster's Skulk version.

    Package versions here are the HARNESS process's (it is an HTTP client, so
    mlx / skulk run on the nodes). ``skulk_version`` / ``skulk_commit`` are
    read from the API node's diagnostics. Any probe that fails is recorded as
    ``"unknown"`` (with an issue), never a report-write failure.
    """

    python: str | None = None
    platform: str | None = None
    harness_packages: dict[str, str] = Field(default_factory=dict)
    skulk_version: str | None = None
    skulk_commit: str | None = None


class ClusterNodeFingerprint(HarnessBaseModel):
    """One node as seen in ``/state`` at run time."""

    node_id: str
    friendly_name: str | None = None
    ram_total_bytes: int | None = None
    # Accelerator vendor ("apple", "amd", ...) from nodeSystem telemetry. This
    # is the only per-node heterogeneity marker an HTTP client can observe:
    # nodeResources (participation backends) is deliberately NOT merged into
    # /state, so vendor is our proxy for "MLX node vs llama.cpp node".
    accelerator_vendor: str | None = None
    # Accelerator marketing name from nodeSystem telemetry ("M4", "Radeon
    # 8060S", ...). Feeds chip-level hardware classes in the results ledger;
    # None when the node's telemetry has not landed or predates the field.
    accelerator_name: str | None = None
    # Accelerator VRAM and GTT aperture from nodeSystem telemetry, in bytes.
    # For a discrete GPU vram is separate device memory; for a unified-memory
    # APU (e.g. AMD Strix Halo) vram is the BIOS carve-out of the SAME physical
    # DIMMs, and gtt is the host RAM the GPU can additionally map. ram_total_bytes
    # is only the OS-visible slice after that carve, so the node's true unified
    # capacity is ram_total + vram_total -- the ledger needs vram to tier such a
    # node correctly (a 128GB Strix with a 64GB carve reports ~61GiB of RAM).
    # None on nodes whose telemetry has not landed or predates the field.
    vram_total_bytes: int | None = None
    gtt_total_bytes: int | None = None
    skulk_version: str | None = None
    system_telemetry_present: bool = False
    memory_telemetry_present: bool = False


class ClusterFingerprint(HarnessBaseModel):
    """The cluster the run executed against."""

    api_base_url: str | None = None
    api_node_id: str | None = None
    master_node_id: str | None = None
    node_count: int = 0
    nodes: list[ClusterNodeFingerprint] = Field(default_factory=list)
    topology_label: str | None = None


class CacheState(HarnessBaseModel):
    """Store/instance cache conditions, from the run spec's own flags.

    Approximate by nature: distinguishes what the spec REQUESTED from any
    claim of controlled warm/cold conditions.
    """

    ensure_store_downloads: bool = False
    reuse_existing_instances: bool = True
    retain_instances: bool = True
    delete_staged_models: bool = False
    classification: Literal["unknown", "cold", "warm", "mixed"] = "unknown"


class ReportFingerprint(HarnessBaseModel):
    """Durable self-description of a run (results-ledger schema 2.0)."""

    # 2.2 adds per-node accelerator vram_total_bytes / gtt_total_bytes (the
    # unified-APU carve, so a consumer can report true capacity). 2.1 added
    # accelerator_name. Bump on any additive durable fingerprint field so
    # downstream consumers can select parsing/compatibility by version.
    schema_version: str = "2.2"
    source_context: SourceContext = Field(default_factory=SourceContext)
    runtime: RuntimeFingerprint = Field(default_factory=RuntimeFingerprint)
    cluster: ClusterFingerprint = Field(default_factory=ClusterFingerprint)
    cache_state: CacheState = Field(default_factory=CacheState)


class RunReport(HarnessBaseModel):
    """Complete machine-readable report for one harness run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    spec: RunSpec
    test_set_description: str = Field(
        default="",
        description=(
            "Human explanation of what the run's test set measures, resolved "
            "from the loaded TestSet. Lets a report describe its own suite "
            "instead of relying on a name lookup. Empty for older reports or a "
            "test set that declares no description."
        ),
    )
    models: list[ModelRef] = Field(default_factory=list)
    placements: list[PlacementResult] = Field(default_factory=list)
    results: list[TestResult] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    fingerprint: ReportFingerprint | None = None

    @classmethod
    def start(cls, run_id: str, spec: RunSpec, models: list[ModelRef]) -> "RunReport":
        """Create a report with the current UTC start time."""

        return cls(
            run_id=run_id,
            started_at=datetime.now(tz=UTC),
            spec=spec,
            models=models,
        )

    def finish(self) -> "RunReport":
        """Return a copy marked with the current UTC finish time."""

        return self.model_copy(update={"finished_at": datetime.now(tz=UTC)})


ComparisonGuardKind = Literal[
    "node_set_mismatch",
    "cache_mismatch",
    "low_sample",
    "short_output_dominant",
    "issue_marked",
    "decode_tps_unavailable",
    "missing_fingerprint",
    "model_only_one_side",
]


class MetricAggregate(HarnessBaseModel):
    """One metric aggregated over a population of like results for a model.

    Median is the headline (robust to the short-output outliers that make raw
    wall throughput meaningless). ``sample_count`` counts the substantive
    samples the median is built from; ``short_sample_count`` counts the
    outputs excluded as too short to time honestly. Both travel with the number
    so a reader can never mistake a 1-sample median for a measured result.
    """

    metric: str
    unit: str
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    sample_count: int = 0
    short_sample_count: int = 0


class ModelMetricSummary(HarnessBaseModel):
    """All headline metrics for one model within one run set."""

    model_id: str
    run_ids: list[str] = Field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    issue_count: int = 0
    node_count_observed: list[int] = Field(default_factory=list)
    topology_labels: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricAggregate] = Field(default_factory=dict)


class MetricDelta(HarnessBaseModel):
    """Candidate-vs-baseline change for one metric of one model."""

    metric: str
    unit: str
    baseline: float | None = None
    candidate: float | None = None
    absolute_delta: float | None = None
    percent_delta: float | None = None
    higher_is_better: bool = True


class ModelComparison(HarnessBaseModel):
    """Like-for-like comparison of one model across two run sets."""

    model_id: str
    deltas: list[MetricDelta] = Field(default_factory=list)
    guards: list[ComparisonGuardKind] = Field(default_factory=list)
    baseline_summary: ModelMetricSummary | None = None
    candidate_summary: ModelMetricSummary | None = None


class ComparisonRecord(HarnessBaseModel):
    """A full baseline-vs-candidate comparison over matched run sets.

    Deliberately records the guards that make a comparison NOT like-for-like
    (differing node set, cache warmth, low sample count, short-output noise,
    issue-marked runs) rather than hiding them. A downstream reader or the
    results site must be able to see why a delta may not be trustworthy.
    """

    schema_version: str = "1.0"
    baseline_label: str
    candidate_label: str
    baseline_run_ids: list[str] = Field(default_factory=list)
    candidate_run_ids: list[str] = Field(default_factory=list)
    models: list[ModelComparison] = Field(default_factory=list)
    guards: list[ComparisonGuardKind] = Field(default_factory=list)


StabilitySuite = Literal["failover", "churn", "soak", "refusal"]


class LatencySummary(HarnessBaseModel):
    """Aggregate latency statistics over a population of completions."""

    count: int = 0
    failures: int = 0
    p50_s: float | None = None
    p95_s: float | None = None
    max_s: float | None = None
    min_s: float | None = None
    mean_s: float | None = None


class StabilityReport(HarnessBaseModel):
    """Machine-readable result of one stability suite run.

    Unlike :class:`RunReport`, stability suites assert cluster *properties*
    (master election, instance continuity, refusal behavior) rather than scoring
    model output, so they carry a free-form ``observations`` map alongside the
    shared :class:`Issue` list. ``passed`` is true only when no error-severity
    issue was recorded.
    """

    run_id: str
    suite: StabilitySuite
    model_id: str
    started_at: datetime
    finished_at: datetime | None = None
    passed: bool = True
    issues: list[Issue] = Field(default_factory=list)
    latency: LatencySummary | None = None
    observations: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def start(
        cls, run_id: str, suite: StabilitySuite, model_id: str
    ) -> "StabilityReport":
        """Create a stability report stamped with the current UTC start time."""

        return cls(
            run_id=run_id,
            suite=suite,
            model_id=model_id,
            started_at=datetime.now(tz=UTC),
        )

    def add_issue(self, issue: Issue) -> None:
        """Record an issue and clear ``passed`` on any error severity."""

        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    def finish(self) -> "StabilityReport":
        """Return a copy marked with the current UTC finish time."""

        return self.model_copy(update={"finished_at": datetime.now(tz=UTC)})
