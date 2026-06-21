"""Typed configuration and report models for the Skulk harness."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PlacementStrategy = Literal["minimum", "single", "exact"]
ShardingMode = Literal["Pipeline", "Tensor"]
InstanceMeta = Literal["MlxRing", "MlxJaccl"]
TestKind = Literal["chat", "code", "artifact", "tool"]
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
    repo_path: str = Field(
        description="Absolute path to the Skulk checkout on the node, used to relaunch."
    )


class HarnessConfig(HarnessBaseModel):
    """Top-level local coordinator settings."""

    api_base_url: str = "http://localhost:52415"
    request_timeout_s: float = 30.0
    generation_timeout_s: float = 1800.0
    placement_ready_timeout_s: float = 1800.0
    store_download_timeout_s: float = 14400.0
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


class PromptTest(HarnessBaseModel):
    """One text-generation test case."""

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
    tools: list[dict[str, object]] = Field(default_factory=list)
    tool_choice: str | dict[str, object] | None = None
    parallel_tool_calls: bool | None = None
    tool_mocks: list[ToolMock] = Field(default_factory=list)
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


class GenerationMetrics(HarnessBaseModel):
    """Wall-clock and Skulk-reported generation metrics."""

    elapsed_s: float
    ttft_s: float | None = None
    output_chars: int = 0
    chunks: int = 0
    approx_output_tokens: int | None = None
    wall_tps: float | None = None
    skulk_prompt_tps: float | None = None
    skulk_generation_tps: float | None = None
    skulk_prompt_tokens: int | None = None
    skulk_generation_tokens: int | None = None


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
    repetition: int
    passed: bool
    output_text: str
    reasoning_text: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metrics: GenerationMetrics
    issues: list[Issue] = Field(default_factory=list)
    artifact_path: Path | None = None


class RunReport(HarnessBaseModel):
    """Complete machine-readable report for one harness run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    spec: RunSpec
    models: list[ModelRef] = Field(default_factory=list)
    placements: list[PlacementResult] = Field(default_factory=list)
    results: list[TestResult] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)

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
    def start(cls, run_id: str, suite: StabilitySuite, model_id: str) -> "StabilityReport":
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
