from skulk_test_harness.goal_parser import parse_goal


def test_parse_goal_handles_hyphenated_model_and_test_sets() -> None:
    spec = parse_goal(
        "step through catalog small text with minimum node placement and run code tests",
        model_set_names=["catalog-small-text", "store-smoke"],
        test_set_names=["code-tests", "chat-tests"],
    )

    assert spec.model_set == "catalog-small-text"
    assert spec.test_set == "code-tests"
    assert spec.placement.strategy == "minimum"
    assert spec.mode == "plan"


def test_parse_goal_detects_single_node_and_execute() -> None:
    spec = parse_goal(
        "run store smoke on chat-tests using a single node",
        model_set_names=["store-smoke"],
        test_set_names=["chat-tests"],
        execute=True,
    )

    assert spec.mode == "execute"
    assert spec.placement.strategy == "single"
    assert spec.placement.min_nodes == 1


def test_parse_goal_handles_tool_suite_aliases() -> None:
    spec = parse_goal(
        "run tool tests against store smoke with minimum placement",
        model_set_names=["store-smoke", "catalog-small-text"],
        test_set_names=["tool-tests", "chat-tests"],
    )

    assert spec.model_set == "store-smoke"
    assert spec.test_set == "tool-tests"
