"""Black-box release checks for Skulk's resident intelligent fabric."""

from __future__ import annotations

import re
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
    node_commits: dict[str, str]
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    status: dict[str, object]
    target_node_name: str
    identity_response: str
    diagnostics_response: str


def _harness_commit() -> str:
    """Return the source commit, marking a checkout with local changes dirty."""

    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    if not commit:
        return "unknown"
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _skulk_commit(diagnostics: dict[str, object]) -> str:
    """Read the serving Skulk commit from node diagnostics."""

    runtime = diagnostics.get("runtime")
    if not isinstance(runtime, dict):
        return "unknown"
    commit = runtime.get("skulkCommit")
    return commit if isinstance(commit, str) and commit else "unknown"


def _is_attributable_commit(value: str) -> bool:
    """Return whether a provenance value identifies one clean Git commit."""

    return re.fullmatch(r"[0-9a-fA-F]{7,64}", value) is not None


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


def _normalized_fact_text(value: str) -> str:
    """Normalize harmless presentation differences without dropping words."""

    expanded = value.casefold().replace("w/", "with")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", expanded).split())


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


def _node_commits(
    state: dict[str, object], eligible_node_ids: set[str] | None
) -> dict[str, str]:
    """Return source provenance for every node in the exercised fleet scope."""

    identities = state.get("nodeIdentities")
    typed_identities = identities if isinstance(identities, dict) else {}
    names = _friendly_node_names(state)
    selected_ids = set(names) if eligible_node_ids is None else eligible_node_ids
    commits: dict[str, str] = {}
    for node_id in sorted(selected_ids):
        identity = typed_identities.get(node_id)
        commit = identity.get("skulkCommit") if isinstance(identity, dict) else None
        commits[names.get(node_id, f"unresolved:{node_id}")] = (
            commit if isinstance(commit, str) and commit else "unknown"
        )
    return commits


def _doctor_reference(
    diagnostics: dict[str, object],
) -> tuple[int, str | None, str | None]:
    """Return the finding count and first stable doctor result identifiers."""

    doctor = diagnostics.get("doctor")
    if not isinstance(doctor, list):
        raise RuntimeError("target node diagnostics omitted the doctor result list")
    if not doctor:
        return 0, None, None
    first = doctor[0]
    if not isinstance(first, dict):
        raise RuntimeError("target node diagnostics returned an invalid doctor result")
    check_id = first.get("checkId")
    verdict = first.get("verdict")
    if not isinstance(check_id, str) or not isinstance(verdict, str):
        raise RuntimeError("target node doctor result omitted its check ID or verdict")
    return len(doctor), check_id, verdict


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
    harness_commit = _harness_commit()
    skulk_commit = _skulk_commit(local_diagnostics)
    if not _is_attributable_commit(harness_commit):
        failures.append(
            f"harness source provenance is not a clean commit: {harness_commit!r}"
        )
    if not _is_attributable_commit(skulk_commit):
        failures.append(
            f"serving Skulk source provenance is unavailable: {skulk_commit!r}"
        )
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
    node_commits = _node_commits(state, eligible_node_ids)
    if not node_commits:
        failures.append("exercised fleet scope has no node source provenance")
    for node_name, commit in node_commits.items():
        if not _is_attributable_commit(commit):
            failures.append(
                f"node {node_name!r} source provenance is unavailable: {commit!r}"
            )
        elif commit.casefold() != skulk_commit.casefold():
            failures.append(
                f"node {node_name!r} runs {commit}, expected {skulk_commit}"
            )
    target_node_id, target_node_name = _diagnostic_target(
        client, state, diagnostic_node_name, eligible_node_ids
    )
    expected_hardware_facts = _target_hardware_facts(state, target_node_id)
    if len(expected_hardware_facts) < 2:
        failures.append(
            "target node lacks model and chip facts needed to prove diagnostics"
        )
    target_diagnostics = client.get_cluster_node_diagnostics(target_node_id)
    doctor_count, doctor_check_id, doctor_verdict = _doctor_reference(
        target_diagnostics
    )
    identity = client.stream_chat(
        model_id="skulk/steward",
        messages=[
            {
                "role": "user",
                "content": "What system or service are you? Answer in one sentence.",
            }
        ],
        max_tokens=96,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ).text.strip()
    if (
        re.search(r"\b(?:i am|i'm|this is)\s+(?:the\s+)?skulk\b", identity.casefold())
        is None
    ):
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
                    "hardware model and chip. Report the exact doctor finding "
                    f"count using the phrase '{doctor_count} doctor findings'. "
                    + (
                        "Also include the first finding's exact check ID "
                        f"'{doctor_check_id}' and verdict '{doctor_verdict}'."
                        if doctor_check_id is not None and doctor_verdict is not None
                        else ""
                    )
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
        _normalized_fact_text(fact) not in _normalized_fact_text(diagnostics)
        for fact in expected_hardware_facts
    ):
        failures.append(
            "named-node diagnostic response did not reproduce API-observed "
            "hardware facts"
        )
    elif f"{doctor_count} doctor findings" not in diagnostics.casefold():
        failures.append(
            "named-node diagnostic response did not reproduce the doctor finding count"
        )
    elif (
        doctor_check_id is not None
        and doctor_verdict is not None
        and (
            _normalized_fact_text(doctor_check_id)
            not in _normalized_fact_text(diagnostics)
            or _normalized_fact_text(doctor_verdict)
            not in _normalized_fact_text(diagnostics)
        )
    ):
        failures.append(
            "named-node diagnostic response did not reproduce the first doctor result"
        )
    else:
        checks.append(
            "named-node diagnostics reproduced API facts without identity leakage"
        )

    return StewardQualificationEvidence(
        generated_at=datetime.now(UTC).isoformat(),
        harness_version=__version__,
        harness_commit=harness_commit,
        skulk_commit=skulk_commit,
        node_commits=node_commits,
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        status=status,
        target_node_name=target_node_name,
        identity_response=identity,
        diagnostics_response=diagnostics,
    )
