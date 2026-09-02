"""Unit coverage for resident intelligent-fabric release checks."""

from types import SimpleNamespace

from skulk_test_harness.steward_qualification import qualify_steward


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

    def list_models(self) -> list[dict[str, object]]:
        return [{"id": "skulk/steward", "system_role": "steward"}]

    def get_state(self) -> dict[str, object]:
        return {
            "nodeIdentities": {
                "peer-api": {"friendlyName": "api-node"},
                "peer-worker": {"friendlyName": "worker-node"},
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
        if "name" in prompt:
            return SimpleNamespace(text="I am Skulk.")
        return SimpleNamespace(
            text="worker-node is healthy; its doctor checks report no problems."
        )


def test_qualification_prefers_no_api_worker_for_named_diagnostics() -> None:
    client = _Client()

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is True
    assert evidence.target_node_name == "worker-node"
    assert len(evidence.checks) == 4
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
        text="worker-node peer-worker"
    )

    evidence = qualify_steward(client)  # type: ignore[arg-type]

    assert evidence.passed is False
    assert any("ready=False" in failure for failure in evidence.failures)
    assert any("leaked" in failure for failure in evidence.failures)
