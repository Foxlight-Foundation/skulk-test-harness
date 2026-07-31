"""Direct-API and fresh-runtime acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from skulk_test_harness.client import SkulkClient
from skulk_test_harness.echo_phrase import echo_matched, echo_phrase, echo_prompt
from skulk_test_harness.models import (
    DashboardContract,
    DataTransport,
    InstallProvenance,
    VisionFixtureEvidence,
)
from skulk_test_harness.vision_fixture import VisionFixture


class UnexpectedFreshInstallPeerError(RuntimeError):
    """Signal that an allegedly isolated fresh node discovered another node."""


def assert_fresh_single_node(
    client: SkulkClient,
    *,
    expected_node_id: str | None = None,
) -> str:
    """Require the fresh runtime to remain the same isolated one-node cluster.

    Args:
        client: Client connected directly to the temporary fresh runtime.
        expected_node_id: Temporary runtime identity captured before user
            journeys begin. When supplied, a process restart or identity swap
            is also a qualification failure.

    Returns:
        The sole node identity observed in cluster state.

    Raises:
        UnexpectedFreshInstallPeerError: Another node joined the temporary
            runtime.
        RuntimeError: No node is present or the temporary runtime identity
            changed.
    """

    return _assert_fresh_single_node_state(
        client.get_state(),
        expected_node_id=expected_node_id,
    )


def assert_fresh_cluster(
    client: SkulkClient,
    *,
    expected_node_count: int,
    expected_node_ids: frozenset[str] | None = None,
) -> frozenset[str]:
    """Require an exact, stable fresh-install cluster membership."""

    return _assert_fresh_cluster_state(
        client.get_state(),
        expected_node_count=expected_node_count,
        expected_node_ids=expected_node_ids,
    )


def _assert_fresh_cluster_state(
    state: dict[str, object],
    *,
    expected_node_count: int,
    expected_node_ids: frozenset[str] | None = None,
) -> frozenset[str]:
    """Require exact membership in an already-fetched state snapshot."""

    resources = _object(state.get("nodeResources"))
    identities = _object(state.get("nodeIdentities"))
    node_ids = frozenset(resources.keys() | identities.keys())
    if len(node_ids) > expected_node_count:
        raise UnexpectedFreshInstallPeerError(
            "fresh install discovered an unexpected node; observed "
            f"{len(node_ids)}, expected {expected_node_count}"
        )
    if len(node_ids) != expected_node_count:
        raise RuntimeError(
            f"fresh install must form exactly {expected_node_count} nodes; "
            f"observed {len(node_ids)}"
        )
    if expected_node_ids is not None and node_ids != expected_node_ids:
        raise RuntimeError("fresh install cluster identity changed during qualification")
    return node_ids


def _assert_fresh_single_node_state(
    state: dict[str, object],
    *,
    expected_node_id: str | None = None,
) -> str:
    """Require one stable node in an already-fetched state snapshot."""

    expected_ids = (
        frozenset({expected_node_id}) if expected_node_id is not None else None
    )
    node_ids = _assert_fresh_cluster_state(
        state,
        expected_node_count=1,
        expected_node_ids=expected_ids,
    )
    node_id = next(iter(node_ids))
    if expected_node_id is not None and node_id != expected_node_id:
        raise RuntimeError("fresh install runtime identity changed during qualification")
    return node_id


@dataclass(frozen=True)
class DirectTextQualification:
    """Result and bounded diagnostic evidence for a direct text request."""

    passed: bool
    response: str


def assert_fresh_runtime_contract(
    client: SkulkClient,
    *,
    expected_backends: list[str],
    expected_transport: DataTransport,
    expected_commit: str | None,
    dashboard_contract: DashboardContract = "required",
    expected_node_count: int = 1,
    expected_node_ids: frozenset[str] | None = None,
) -> InstallProvenance:
    """Validate topology, backend, transport, dashboard, and commit truth."""

    state = client.get_state()
    resources = _object(state.get("nodeResources"))
    identities = _object(state.get("nodeIdentities"))
    _assert_fresh_cluster_state(
        state,
        expected_node_count=expected_node_count,
        expected_node_ids=expected_node_ids,
    )
    detected_backends: set[str] = set()
    transports: set[str] = set()
    for raw in resources.values():
        resource = _object(raw)
        backends = resource.get("backends")
        if isinstance(backends, list):
            detected_backends.update(
                item for item in backends if isinstance(item, str)
            )
        transport = resource.get("dataTransport")
        if isinstance(transport, str):
            transports.add(transport)
    missing_backends = sorted(set(expected_backends) - detected_backends)
    if missing_backends:
        raise RuntimeError(
            f"fresh install did not detect expected backends: {missing_backends}"
        )
    if transports != {expected_transport}:
        raise RuntimeError(
            f"fresh install DATA transport mismatch: observed {sorted(transports)}"
        )
    diagnostics = client.get_diagnostics_node()
    runtime = _object(diagnostics.get("runtime"))
    resolved_commit = runtime.get("skulkCommit", runtime.get("skulk_commit"))
    if not isinstance(resolved_commit, str):
        resolved_commit = None
    reported_commits: list[str] = []
    for raw_identity in identities.values():
        identity = _object(raw_identity)
        reported = identity.get("skulkCommit", identity.get("skulk_commit"))
        if isinstance(reported, str):
            reported_commits.append(reported)
    # Older and single-node runtimes expose commit provenance only through the
    # local diagnostics endpoint. Multi-node qualification requires one commit
    # value per member from replicated identity state so a mixed build cannot
    # hide behind the entrypoint's correct value.
    if expected_node_count == 1 and not reported_commits and resolved_commit:
        reported_commits.append(resolved_commit)
    if expected_commit:
        mismatched = sorted(
            commit
            for commit in reported_commits
            if not _commit_matches(expected_commit, commit)
        )
        if len(reported_commits) != expected_node_count or mismatched:
            raise RuntimeError(
                "fresh install runtime commit did not match the pinned "
                f"candidate: observed {sorted(reported_commits)}"
            )
    response = httpx.get(client.base_url, timeout=client.request_timeout_s)
    dashboard_present = (
        response.status_code == 200
        and "<html" in response.text.lower()
        and 'id="root"' in response.text
    )
    # A node with no Node toolchain is a shipped shape, not a degraded one: the
    # installer skips the dashboard build and the API serves without the web UI.
    # The target declares which of the two it is, and both are asserted, so an
    # unexpectedly missing dashboard still fails and "absent" never becomes a
    # quiet skip that would also pass on a broken build.
    if dashboard_contract == "required" and not dashboard_present:
        raise RuntimeError("fresh install did not serve the production dashboard build")
    if dashboard_contract == "absent" and dashboard_present:
        raise RuntimeError(
            "fresh install served a dashboard on a target declared headless"
        )
    return InstallProvenance(
        mode="fresh_install",
        environment="fresh_install",
        expected_commit=expected_commit,
        resolved_commit=resolved_commit,
        environment_override_names=[],
        detected_backends=sorted(detected_backends),
        data_transport=expected_transport,
        node_count=expected_node_count,
        dashboard_build_present=dashboard_present,
    )


def qualify_direct_text(
    client: SkulkClient,
    *,
    model_id: str,
    enable_thinking: bool | None,
) -> DirectTextQualification:
    """Require the direct API response to retain all randomized prompt items."""

    phrase = echo_phrase()
    execution = client.stream_chat(
        model_id=model_id,
        messages=[{"role": "user", "content": echo_prompt(phrase)}],
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=enable_thinking,
    )
    return DirectTextQualification(
        passed=echo_matched(phrase, execution.text),
        response=execution.text.strip()[:400],
    )


def qualify_direct_vision(
    client: SkulkClient,
    *,
    model_id: str,
    fixture: VisionFixture,
    enable_thinking: bool | None,
) -> VisionFixtureEvidence:
    """Require exact hidden-code and visual-attribute recognition via the API."""

    execution = client.stream_chat(
        model_id=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": fixture.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": fixture.data_url, "detail": "high"},
                    },
                ],
            }
        ],
        # Small vision models often explain each observed element before
        # giving the requested compact answer. A 128-token cap can stop after
        # the code and shape but before the color, turning successful image
        # understanding into a false gate failure.
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=enable_thinking,
    )
    code_matched, color_matched, shape_matched = (
        fixture.response_match_details(execution.text)
    )
    format_matched = fixture.response_format_matches(execution.text)
    attribute_matched = color_matched and shape_matched
    return VisionFixtureEvidence(
        channel="api",
        fixture_sha256=fixture.sha256,
        code_sha256=fixture.code_sha256,
        expected_shape=fixture.shape,
        expected_color=fixture.color,
        response_matched_code=code_matched,
        response_matched_attribute=attribute_matched,
        response_matched_color=color_matched,
        response_matched_shape=shape_matched,
        response_matched_format=format_matched,
        response_excerpt=execution.text.replace(
            fixture.code, "<hidden-code>"
        ).strip()[:400],
        request_image_sha256=fixture.sha256,
        passed=code_matched and attribute_matched and format_matched,
    )


_MINIMUM_ABBREVIATED_COMMIT_LENGTH = 7


def _commit_matches(expected: str, resolved: str | None) -> bool:
    """Return whether a runtime commit identifies the pinned candidate build.

    A qualification pins a full 40-character SHA, but the node reports
    ``git rev-parse --short HEAD``, so an equality test can never succeed.
    Skulk itself compares builds by abbreviation, so this applies the same
    contract: the shorter identifier must be a prefix of the longer one and
    at least as long as git's minimum abbreviation. A node that cannot read
    its own commit reports ``unknown``, which never matches.
    """

    if resolved is None:
        return False
    pinned = expected.strip().lower()
    reported = resolved.strip().lower()
    if not pinned or not reported or reported == "unknown":
        return False
    shorter, longer = sorted((pinned, reported), key=len)
    if len(shorter) < _MINIMUM_ABBREVIATED_COMMIT_LENGTH:
        return False
    return longer.startswith(shorter)


def _object(value: object) -> dict[str, object]:
    """Return a typed dictionary or an empty object."""

    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
