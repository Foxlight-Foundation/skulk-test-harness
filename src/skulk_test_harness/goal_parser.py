"""Small deterministic parser for agent-authored natural language goals."""

from __future__ import annotations

from skulk_test_harness.models import PlacementPolicy, RunSpec


def parse_goal(
    text: str,
    *,
    model_set_names: list[str],
    test_set_names: list[str],
    execute: bool = False,
) -> RunSpec:
    """Convert a constrained natural-language request into a run spec.

    This is intentionally deterministic and local. It recognizes named model
    sets and test sets by substring, plus a few placement phrases such as
    "minimum node placement" or "single node".
    """

    lowered = text.lower()
    model_set = _find_name(lowered, model_set_names)
    test_set = _find_name(lowered, test_set_names)
    if model_set is None:
        raise ValueError(
            "Could not find a named model set in the goal. Known sets: "
            + ", ".join(sorted(model_set_names))
        )
    if test_set is None:
        raise ValueError(
            "Could not find a named test set in the goal. Known sets: "
            + ", ".join(sorted(test_set_names))
        )

    strategy = "minimum"
    min_nodes: int | None = None
    if "single node" in lowered or "1 node" in lowered or "one node" in lowered:
        strategy = "single"
        min_nodes = 1
    elif "minimum" in lowered or "min node" in lowered or "minimum node" in lowered:
        strategy = "minimum"

    sharding = "Tensor" if "tensor" in lowered else "Pipeline"
    instance_meta = "MlxJaccl" if "jaccl" in lowered or "rdma" in lowered else "MlxRing"

    return RunSpec(
        model_set=model_set,
        test_set=test_set,
        mode="execute" if execute else "plan",
        placement=PlacementPolicy(
            strategy=strategy,  # type: ignore[arg-type]
            sharding=sharding,  # type: ignore[arg-type]
            instance_meta=instance_meta,  # type: ignore[arg-type]
            min_nodes=min_nodes,
        ),
        run_name=text[:80],
    )


def _find_name(goal_text: str, names: list[str]) -> str | None:
    goal_variants = {goal_text, goal_text.replace("-", " ")}
    normalized = {
        name.lower(): name for name in names
    } | {
        name.lower().replace("-", " "): name for name in names
    }
    for lowered, original in sorted(
        normalized.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if any(lowered in variant for variant in goal_variants):
            return original
    return None
