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


class HarnessConfig(HarnessBaseModel):
    """Top-level local coordinator settings."""

    api_base_url: str = "http://localhost:52415"
    request_timeout_s: float = 30.0
    generation_timeout_s: float = 1800.0
    placement_ready_timeout_s: float = 1800.0
    store_download_timeout_s: float = 14400.0
    poll_interval_s: float = 2.0
    output_dir: Path = Path("runs")
    model_sets_path: Path = Path("configs/model_sets.yaml")
    test_sets_path: Path = Path("configs/test_sets.yaml")


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
