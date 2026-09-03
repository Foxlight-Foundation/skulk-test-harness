"""Unit coverage for resident intelligent-fabric release checks."""

from types import SimpleNamespace

import pytest

from skulk_test_harness import steward_qualification
from skulk_test_harness.steward_qualification import qualify_steward


@pytest.fixture(autouse=True)
def _clean_harness_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        steward_qualification,
        "_harness_commit",
        lambda: "1234567890abcdef1234567890abcdef12345678",
    )


class _Client:
    """Small deterministic client stand-in for the black-box qualifier."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_steward_status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "present": True,
            "ready": True,
            "state": "ready",
            "transition": "idle",
            "steward_model": "org/brain",
            "desired_model": "org/brain",
        }

    def get_diagnostics_node(self) -> dict[str, object]:
        return {"runtime": {"skulkCommit": "abc1234"}}

    def get_cluster_node_diagnostics(self, _node_id: str) -> dict[str, object]:
        return {"doctor": []}

    def list_models(self) -> list[dict[str, object]]:
        return [{"id": "skulk/steward", "system_role": "steward"}]

    def get_state(self) -> dict[str, object]:
        return {
            "nodeIdentities": {
                "peer-api": {
                    "friendlyName": "api-node",
                    "modelId": "API Model",
                    "chipId": "API Chip",
                    "skulkCommit": "abc1234",
                },
                "peer-worker": {
                    "friendlyName": "worker-node",
                    "modelId": "Worker Model_42",
                    "chipId": "Worker_Chip 9",
                    "skulkCommit": "abc1234",
                },
            },
            "nodeResources": {
                "peer-api": {"apiAvailable": True},
                "peer-worker": {"apiAvailable": False},
            },
        }

    def get_node_id(self) -> str:
        return "peer-api"

    def stream_chat(self, **kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        message = messages[0]
        assert isinstance(message, dict)
        prompt = message["content"]
        assert isinstance(prompt, str)
        self.prompts.append(prompt)
        if prompt.startswith("What system or service"):
            return SimpleNamespace(text="I am Skulk.")
        if "api-node" in prompt:
            return SimpleNamespace(
                text=(
                    "api-node is an API Model with API Chip; it has 0 doctor findings."
                ),
                reasoning_text=(
                    'get_node_diagnostics {"node_name":"api-node"}\n'
                    'run_doctor {"node_name":"api-node"}\n'
                ),
            )
        return SimpleNamespace(
            text=(
                "worker-node is a Worker Model 42 with Worker Chip 9; "
                "it has 0 doctor findings."
            ),
            reasoning_text=(
                'get_node_diagnostics {"node_name":"worker-node"}\n'
                'run_doctor {"node_name":"worker-node"}\n'
            ),
        )


def test_qualification_prefers_no_api_worker_for_named_diagnostics() -> None:
    client = _Client()

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is True
    assert evidence.target_node_name == "worker-node"
    assert len(evidence.checks) == 4
    assert evidence.skulk_commit == "abc1234"
    assert evidence.node_commits == {
        "api-node": "abc1234",
        "worker-node": "abc1234",
    }
    assert "Skulk" not in client.prompts[0]
    assert "peer-worker" not in evidence.diagnostics_response
    assert "worker-node" in client.prompts[1]


def test_qualification_rejects_non_ready_transition_and_identity_leak() -> None:
    client = _Client()
    client.get_steward_status = lambda: {  # type: ignore[method-assign]
        "enabled": True,
        "present": True,
        "ready": False,
        "state": "starting",
        "transition": "prestaging",
        "steward_model": "org/brain",
        "desired_model": "org/better",
    }
    client.stream_chat = lambda **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        text="worker-node peer-worker",
        reasoning_text='run_doctor {"node_name":"worker-node"}\n',
    )

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any("ready=False" in failure for failure in evidence.failures)
    assert any("leaked" in failure for failure in evidence.failures)


def test_qualification_rejects_negated_identity_and_unproven_diagnostics() -> None:
    client = _Client()
    responses = iter(
        [
            SimpleNamespace(text="I am not Skulk."),
            SimpleNamespace(
                text="worker-node could not be inspected.",
                reasoning_text='run_doctor {"node_name":"worker-node"}\n',
            ),
        ]
    )
    client.stream_chat = lambda **_kwargs: next(responses)  # type: ignore[method-assign]

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any("identity" in failure for failure in evidence.failures)
    assert any("API-observed" in failure for failure in evidence.failures)


def test_qualification_restricts_diagnostics_to_eligible_fleet_scope() -> None:
    client = _Client()

    evidence = qualify_steward(
        client,  # pyright: ignore[reportArgumentType]
        eligible_node_ids={"peer-api"},
    )

    assert evidence.passed is True
    assert evidence.target_node_name == "api-node"
    assert "api-node" in client.prompts[1]


def test_qualification_rejects_unattributable_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(steward_qualification, "_harness_commit", lambda: "unknown")
    client.get_diagnostics_node = lambda: {  # type: ignore[method-assign]
        "runtime": {"skulkCommit": "unknown"}
    }

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any("harness source provenance" in failure for failure in evidence.failures)
    assert any("Skulk source provenance" in failure for failure in evidence.failures)


def test_qualification_rejects_mixed_node_commits_and_unproven_doctor() -> None:
    client = _Client()
    state = client.get_state()
    identities = state["nodeIdentities"]
    assert isinstance(identities, dict)
    worker = identities["peer-worker"]
    assert isinstance(worker, dict)
    worker["skulkCommit"] = "def5678"
    client.get_state = lambda: state  # type: ignore[method-assign]
    client.get_cluster_node_diagnostics = lambda _node_id: {  # type: ignore[method-assign]
        "doctor": [{"checkId": "models-storage", "verdict": "degraded"}]
    }

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any("runs def5678" in failure for failure in evidence.failures)
    assert any("doctor finding count" in failure for failure in evidence.failures)


def test_qualification_requires_structured_doctor_tool_trace() -> None:
    client = _Client()
    responses = iter(
        [
            SimpleNamespace(text="I am Skulk."),
            SimpleNamespace(
                text=(
                    "worker-node is a Worker Model 42 with Worker Chip 9; "
                    "it has 0 doctor findings."
                ),
                reasoning_text=('get_node_diagnostics {"node_name":"worker-node"}\n'),
            ),
        ]
    )
    client.stream_chat = lambda **_kwargs: next(responses)  # type: ignore[method-assign]

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any(
        "structured run_doctor trace" in failure for failure in evidence.failures
    )


def test_qualification_rejects_unidentified_live_resource_node() -> None:
    client = _Client()
    state = client.get_state()
    resources = state["nodeResources"]
    assert isinstance(resources, dict)
    resources["peer-unidentified"] = {"apiAvailable": False}
    client.get_state = lambda: state  # type: ignore[method-assign]

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert evidence.node_commits["Unidentified node 1"] == "unknown"
    assert any(
        "Unidentified node 1" in failure and "unavailable" in failure
        for failure in evidence.failures
    )
