"""Tests for privacy-preserving effective request identities."""

from skulk_test_harness.models import PromptTest, SuccessCriteria
from skulk_test_harness.protocol import protocol_ids


def _test(**updates: object) -> PromptTest:
    base = PromptTest(name="ordered", prompt="Write a long answer.")
    return base.model_copy(update=updates)


def test_protocol_id_is_deterministic_and_sha256() -> None:
    first = protocol_ids(_test(), thinking_default=False)
    second = protocol_ids(_test(), thinking_default=False)

    assert first == second
    assert all(len(value) == 64 for value in first)
    assert all(set(value) <= set("0123456789abcdef") for value in first)


def test_labels_scoring_and_repetitions_do_not_change_protocol() -> None:
    baseline = protocol_ids(_test(), thinking_default=False)
    relabeled = protocol_ids(
        _test(
            name="renamed",
            description="New explanation",
            repetitions=9,
            success=SuccessCriteria(min_chars=999),
        ),
        thinking_default=False,
    )

    assert relabeled == baseline


def test_request_inputs_and_resolved_defaults_change_protocol() -> None:
    baseline = protocol_ids(_test(), thinking_default=False)

    assert protocol_ids(_test(max_tokens=1024), thinking_default=False) != baseline
    assert protocol_ids(_test(), thinking_default=True) != baseline


def test_only_exact_id_changes_across_concurrency_levels() -> None:
    one = protocol_ids(
        _test(kind="concurrent", concurrency=1), thinking_default=False
    )
    sixteen = protocol_ids(
        _test(kind="concurrent", concurrency=16), thinking_default=False
    )

    assert one[0] != sixteen[0]
    assert one[1] == sixteen[1]
