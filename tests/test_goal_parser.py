from skulk_test_harness.goal_parser import parse_goal


def test_parse_goal_handles_hyphenated_model_and_test_sets() -> None:
    spec = parse_goal(
        "step through gemma4-family with minimum node placement and run asteroids-challenge",
        model_set_names=["gemma4-family", "smoke"],
        test_set_names=["asteroids-challenge", "chat-tests"],
    )

    assert spec.model_set == "gemma4-family"
    assert spec.test_set == "asteroids-challenge"
    assert spec.placement.strategy == "minimum"
    assert spec.mode == "plan"


def test_parse_goal_detects_single_node_and_execute() -> None:
    spec = parse_goal(
        "run smoke on chat-tests using a single node",
        model_set_names=["smoke"],
        test_set_names=["chat-tests"],
        execute=True,
    )

    assert spec.mode == "execute"
    assert spec.placement.strategy == "single"
    assert spec.placement.min_nodes == 1


def test_parse_goal_handles_gpt_oss_complete_suite() -> None:
    spec = parse_goal(
        "run gpt oss 20b complete with minimum placement",
        model_set_names=["gpt-oss-20b", "smoke"],
        test_set_names=["gpt-oss-20b-complete", "chat-tests"],
    )

    assert spec.model_set == "gpt-oss-20b"
    assert spec.test_set == "gpt-oss-20b-complete"
