"""Black-box release checks for Skulk's resident intelligent fabric."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from skulk_test_harness import __version__
from skulk_test_harness.client import SkulkClient


@dataclass(frozen=True)
class StewardQualificationEvidence:
    """Bounded evidence from one resident-intelligence qualification pass."""

    generated_at: str
    harness_version: str
    harness_commit: str
    skulk_commit: str
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    status: dict[str, object]
    target_node_name: str
    identity_response: str
    diagnostics_response: str


def _harness_commit() -> str:
    """Return the source commit when running from a checkout."""

    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _skulk_commit(diagnostics: dict[str, object]) -> str:
    """Read the serving Skulk commit from node diagnostics."""

    runtime = diagnostics.get("runtime")
    if not isinstance(runtime, dict):
        return "unknown"
    commit = runtime.get("skulkCommit")
    return commit if isinstance(commit, str) and commit else "unknown"


def _target_hardware_facts(
    state: dict[str, object], target_node_id: str
) -> tuple[str, ...]:
    """Return API-observed facts that a successful diagnostic tool must recover."""

    identities = state.get("nodeIdentities")
    if not isinstance(identities, dict):
        return ()
    identity = identities.get(target_node_id)
    if not isinstance(identity, dict):
        return ()
    return tuple(
        value.strip()
        for key in ("modelId", "chipId")
        if isinstance((value := identity.get(key)), str) and value.strip()
    )


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
    eligible_node_ids: set[str] | None,
) -> tuple[str, str]:
    """Choose one named node, preferring a worker that does not expose an API."""

    names = _friendly_node_names(state)
    if eligible_node_ids is not None:
        names = {
            node_id: name
            for node_id, name in names.items()
            if node_id in eligible_node_ids
        }
    if not names:
        raise RuntimeError("eligible fleet scope has no friendly node identities")
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
    eligible_node_ids: set[str] | None = None,
) -> StewardQualificationEvidence:
    """Exercise the enabled resident, discovery entry, and named-node tools.

    This is observe-only with respect to cluster configuration and placement.
    It does create ordinary transient text-generation tasks through the reserved
    ``skulk/steward`` model, exactly as an operator conversation does.
    """

    checks: list[str] = []
    failures: list[str] = []
    local_diagnostics = client.get_diagnostics_node()
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
        client, state, diagnostic_node_name, eligible_node_ids
    )
    expected_hardware_facts = _target_hardware_facts(state, target_node_id)
    if len(expected_hardware_facts) < 2:
        failures.append(
            "target node lacks model and chip facts needed to prove diagnostics"
        )
    identity = client.stream_chat(
        model_id="skulk/steward",
        messages=[
            {
                "role": "user",
                "content": (
                    "State your identity in one sentence. Begin exactly with "
                    "'I am Skulk'."
                ),
            }
        ],
        max_tokens=96,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ).text.strip()
    if not identity.casefold().startswith("i am skulk"):
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
                    "the observed result in plain language, including its exact "
                    "hardware model and chip."
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
    elif any(
        fact.casefold() not in diagnostics.casefold()
        for fact in expected_hardware_facts
    ):
        failures.append(
            "named-node diagnostic response did not reproduce API-observed "
            "hardware facts"
        )
    else:
        checks.append(
            "named-node diagnostics reproduced API facts without identity leakage"
        )

    return StewardQualificationEvidence(
        generated_at=datetime.now(UTC).isoformat(),
        harness_version=__version__,
        harness_commit=_harness_commit(),
        skulk_commit=_skulk_commit(local_diagnostics),
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        status=status,
        target_node_name=target_node_name,
        identity_response=identity,
        diagnostics_response=diagnostics,
    )
