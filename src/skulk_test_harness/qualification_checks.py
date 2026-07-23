"""Direct-API and fresh-runtime acceptance checks."""

from __future__ import annotations

import secrets

import httpx

from skulk_test_harness.client import SkulkClient
from skulk_test_harness.models import (
    DataTransport,
    InstallProvenance,
    VisionFixtureEvidence,
)
from skulk_test_harness.vision_fixture import VisionFixture


def assert_fresh_runtime_contract(
    client: SkulkClient,
    *,
    expected_backends: list[str],
    expected_transport: DataTransport,
    expected_commit: str | None,
) -> InstallProvenance:
    """Validate topology, backend, transport, dashboard, and commit truth."""

    state = client.get_state()
    resources = _object(state.get("nodeResources"))
    identities = _object(state.get("nodeIdentities"))
    node_ids = resources.keys() | identities.keys()
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
    if expected_commit and resolved_commit != expected_commit:
        raise RuntimeError(
            "fresh install runtime commit did not match the pinned candidate"
        )
    response = httpx.get(client.base_url, timeout=client.request_timeout_s)
    dashboard_present = (
        response.status_code == 200
        and "<html" in response.text.lower()
        and 'id="root"' in response.text
    )
    if not dashboard_present:
        raise RuntimeError("fresh install did not serve the production dashboard build")
    return InstallProvenance(
        mode="fresh_install",
        environment="fresh_install",
        expected_commit=expected_commit,
        resolved_commit=resolved_commit,
        environment_override_names=[],
        detected_backends=sorted(detected_backends),
        data_transport=expected_transport,
        node_count=1,
        dashboard_build_present=True,
    )


def qualify_direct_text(client: SkulkClient, *, model_id: str) -> bool:
    """Require the direct API to echo an unpredictable token."""

    token = f"API-{secrets.token_hex(4).upper()}"
    execution = client.stream_chat(
        model_id=model_id,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with this token exactly once and nothing else: " + token
                ),
            }
        ],
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
    )
    return token in execution.text


def qualify_direct_vision(
    client: SkulkClient,
    *,
    model_id: str,
    fixture: VisionFixture,
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
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    code_matched, attribute_matched = fixture.response_matches(
        execution.text
    )
    return VisionFixtureEvidence(
        channel="api",
        fixture_sha256=fixture.sha256,
        code_sha256=fixture.code_sha256,
        expected_shape=fixture.shape,
        expected_color=fixture.color,
        response_matched_code=code_matched,
        response_matched_attribute=attribute_matched,
        request_image_sha256=fixture.sha256,
        passed=code_matched and attribute_matched,
    )


def _object(value: object) -> dict[str, object]:
    """Return a typed dictionary or an empty object."""

    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
