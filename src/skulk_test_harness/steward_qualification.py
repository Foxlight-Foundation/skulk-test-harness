"""Black-box release checks for Skulk's resident intelligent fabric."""

from __future__ import annotations

from dataclasses import dataclass

from skulk_test_harness.client import SkulkClient


@dataclass(frozen=True)
class StewardQualificationEvidence:
    """Bounded evidence from one resident-intelligence qualification pass."""

    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    status: dict[str, object]
    target_node_name: str
    identity_response: str
    diagnostics_response: str


def _friendly_node_names(state: dict[str, object]) -> dict[str, str]:
    """Return stable friendly names keyed by live node identity."""

    identities = state.get("nodeIdentities")
    if not isinstance(identities, dict):
        return {}
    names: dict[str, str] = {}
    for node_id, raw_identity in identities.items():
        if not isinstance(node_id, str) or not isinstance(raw_identity, dict):
            continue
        friendly_name = raw_identity.get("friendlyName")
        if isinstance(friendly_name, str) and friendly_name.strip():
            names[node_id] = friendly_name.strip()
    return names


def _diagnostic_target(
    client: SkulkClient,
    state: dict[str, object],
    requested_name: str | None,
) -> tuple[str, str]:
    """Choose one named node, preferring a worker that does not expose an API."""

    names = _friendly_node_names(state)
    if not names:
        raise RuntimeError("/state has no friendly node identities")
    if requested_name is not None:
        matches = [
            (node_id, name) for node_id, name in names.items() if name == requested_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"diagnostic node {requested_name!r} is absent or ambiguous"
            )
        return matches[0]

    resources = state.get("nodeResources")
    typed_resources = resources if isinstance(resources, dict) else {}
    no_api = sorted(
        (
            (name, node_id)
            for node_id, name in names.items()
            if isinstance(typed_resources.get(node_id), dict)
            and typed_resources[node_id].get("apiAvailable") is False
        ),
        key=lambda item: item[0],
    )
    if no_api:
        name, node_id = no_api[0]
        return node_id, name

    local_node_id = client.get_node_id()
    remote = sorted(
        (
            (name, node_id)
            for node_id, name in names.items()
            if node_id != local_node_id
        ),
        key=lambda item: item[0],
    )
    if remote:
        name, node_id = remote[0]
        return node_id, name
    node_id, name = next(iter(names.items()))
    return node_id, name


def qualify_steward(
    client: SkulkClient,
    *,
    diagnostic_node_name: str | None = None,
) -> StewardQualificationEvidence:
    """Exercise the enabled resident, discovery entry, and named-node tools.

    This is observe-only with respect to cluster configuration and placement.
    It does create ordinary transient text-generation tasks through the reserved
    ``skulk/steward`` model, exactly as an operator conversation does.
    """

    checks: list[str] = []
    failures: list[str] = []
    status = client.get_steward_status()
    expected_status = {
        "enabled": True,
        "present": True,
        "ready": True,
        "state": "ready",
        "transition": "idle",
    }
    for field, expected in expected_status.items():
        if status.get(field) != expected:
            failures.append(
                f"steward status {field}={status.get(field)!r}, expected {expected!r}"
            )
    steward_model = status.get("steward_model", status.get("stewardModel"))
    desired_model = status.get("desired_model", status.get("desiredModel"))
    if not isinstance(steward_model, str) or not steward_model:
        failures.append("steward status has no serving model")
    elif desired_model != steward_model:
        failures.append("idle steward desired model does not match its serving model")
    else:
        checks.append("ready idle steward status is internally consistent")

    virtual_models = [
        entry
        for entry in client.list_models()
        if entry.get("id") == "skulk/steward"
        and entry.get("system_role", entry.get("systemRole")) == "steward"
    ]
    if len(virtual_models) != 1:
        failures.append(
            "model discovery did not expose exactly one skulk/steward system entry"
        )
    else:
        checks.append("virtual steward model is discoverable exactly once")

    state = client.get_state()
    target_node_id, target_node_name = _diagnostic_target(
        client, state, diagnostic_node_name
    )
    identity = client.stream_chat(
        model_id="skulk/steward",
        messages=[{"role": "user", "content": "What is your name?"}],
        max_tokens=96,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ).text.strip()
    if "skulk" not in identity.casefold():
        failures.append("resident identity response did not identify itself as Skulk")
    elif not identity:
        failures.append("resident identity response was empty")
    else:
        checks.append("resident identifies itself as Skulk")

    diagnostics = client.stream_chat(
        model_id="skulk/steward",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Inspect node {target_node_name} using its complete node "
                    "diagnostics and doctor findings. Name the node and summarize "
                    "the observed result in plain language."
                ),
            }
        ],
        max_tokens=384,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ).text.strip()
    if not diagnostics:
        failures.append("named-node diagnostic response was empty")
    elif target_node_name.casefold() not in diagnostics.casefold():
        failures.append("named-node diagnostic response omitted the friendly node name")
    elif target_node_id in diagnostics:
        failures.append(
            "named-node diagnostic response leaked the internal node identity"
        )
    else:
        checks.append("named-node diagnostics completed without identity leakage")

    return StewardQualificationEvidence(
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        status=status,
        target_node_name=target_node_name,
        identity_response=identity,
        diagnostics_response=diagnostics,
    )
