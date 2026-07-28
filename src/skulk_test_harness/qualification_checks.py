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
) -> InstallProvenance:
    """Validate topology, backend, transport, dashboard, and commit truth."""

    state = client.get_state()
    resources = _object(state.get("nodeResources"))
    identities = _object(state.get("nodeIdentities"))
    node_ids = resources.keys() | identities.keys()
    if len(node_ids) > 1:
        raise UnexpectedFreshInstallPeerError(
            f"fresh install discovered another node; observed {len(node_ids)}"
        )
    if len(node_ids) != 1:
        raise RuntimeError(
            f"fresh install must form exactly one node; observed {len(node_ids)}"
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
    if expected_commit and not _commit_matches(expected_commit, resolved_commit):
        raise RuntimeError(
            "fresh install runtime commit did not match the pinned candidate: "
            f"pinned {expected_commit}, runtime reported {resolved_commit}"
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
        node_count=1,
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
        response_excerpt=execution.text.replace(
            fixture.code, "<hidden-code>"
        ).strip()[:400],
        request_image_sha256=fixture.sha256,
        passed=code_matched and attribute_matched,
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
