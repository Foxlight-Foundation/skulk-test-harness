"""Unit tests for the run-fingerprint gatherer (results-ledger schema 2.0).

These exercise the pure parsing + classification paths against a fake client;
no live cluster and no git dependency on the checkout's real state.
"""

from __future__ import annotations

from skulk_test_harness.fingerprint import _classify_cache, gather_fingerprint
from skulk_test_harness.models import RunSpec


class _FakeClient:
    """Minimal SkulkClient stand-in returning canned /state + diagnostics."""

    base_url = "http://kite1:52415"

    def __init__(self, *, state: dict[str, object], diag: dict[str, object]) -> None:
        self._state = state
        self._diag = diag

    def get_state(self) -> dict[str, object]:
        return self._state

    def get_diagnostics_node(self) -> dict[str, object]:
        return self._diag


def _state() -> dict[str, object]:
    return {
        "lastSeen": {"nodeA": "2026-07-07T00:00:00Z", "nodeB": "2026-07-07T00:00:00Z"},
        "nodeIdentities": {
            "nodeA": {"friendlyName": "kite1", "skulkVersion": "1.4.2"},
            "nodeB": {"friendlyName": "kite4", "skulkVersion": "1.4.2"},
        },
        "nodeMemory": {
            "nodeA": {"ramTotal": {"inBytes": 17179869184}},
            "nodeB": {"ramTotal": {"inBytes": 137438953472}},
        },
        "nodeSystem": {
            "nodeA": {"accelerator": {"vendor": "apple", "name": "M4"}},
            "nodeB": {
                "accelerator": {
                    "vendor": "amd",
                    "vramTotalBytes": 68719476736,
                    "gttTotalBytes": 133143986176,
                }
            },
        },
    }


def _diag() -> dict[str, object]:
    return {
        "runtime": {
            "nodeId": "nodeA",
            "masterNodeId": "nodeB",
            "skulkVersion": "1.4.2",
            "skulkCommit": "984179e2",
        }
    }


def test_gather_fingerprint_populates_cluster_and_runtime() -> None:
    client = _FakeClient(state=_state(), diag=_diag())
    spec = RunSpec(model_set="m", test_set="t", mode="plan")

    fp, issues = gather_fingerprint(client, spec, run_reason="plan")  # type: ignore[arg-type]

    assert issues == []
    assert fp.schema_version == "2.1"
    assert fp.runtime.skulk_version == "1.4.2"
    assert fp.runtime.skulk_commit == "984179e2"
    assert fp.runtime.python  # populated from the running interpreter

    cluster = fp.cluster
    assert cluster.api_base_url == "http://kite1:52415"
    assert cluster.api_node_id == "nodeA"
    assert cluster.master_node_id == "nodeB"
    assert cluster.node_count == 2
    assert cluster.topology_label == "kite1-kite4"

    by_name = {n.friendly_name: n for n in cluster.nodes}
    assert by_name["kite1"].accelerator_vendor == "apple"
    assert by_name["kite1"].accelerator_name == "M4"
    assert by_name["kite4"].accelerator_vendor == "amd"
    # No name in telemetry stays None, never a guess.
    assert by_name["kite4"].accelerator_name is None
    assert by_name["kite4"].ram_total_bytes == 137438953472
    # VRAM carve + GTT aperture captured for the AMD node so the ledger can
    # report a unified APU's true capacity (ram_total + vram); Apple has no
    # separate carve field, so it stays None.
    assert by_name["kite4"].vram_total_bytes == 68719476736
    assert by_name["kite4"].gtt_total_bytes == 133143986176
    assert by_name["kite1"].vram_total_bytes is None
    assert by_name["kite1"].gtt_total_bytes is None
    # Uniform version across nodes is the mixed-version detector's clean case.
    assert {n.skulk_version for n in cluster.nodes} == {"1.4.2"}


def test_gather_fingerprint_flags_mixed_versions_in_data() -> None:
    state = _state()
    state["nodeIdentities"]["nodeB"]["skulkVersion"] = "1.4.1"  # type: ignore[index]
    client = _FakeClient(state=state, diag=_diag())
    spec = RunSpec(model_set="m", test_set="t", mode="plan")

    fp, _ = gather_fingerprint(client, spec, run_reason="plan")  # type: ignore[arg-type]

    versions = {n.friendly_name: n.skulk_version for n in fp.cluster.nodes}
    # The fingerprint records per-node versions verbatim; divergence is visible
    # to any downstream reader without the harness making a judgement call.
    assert versions == {"kite1": "1.4.2", "kite4": "1.4.1"}


def test_cluster_probe_failure_is_a_warning_not_a_raise() -> None:
    class _Boom(_FakeClient):
        def get_state(self) -> dict[str, object]:
            raise RuntimeError("no /state")

    client = _Boom(state={}, diag=_diag())
    spec = RunSpec(model_set="m", test_set="t", mode="plan")

    fp, issues = gather_fingerprint(client, spec, run_reason="plan")  # type: ignore[arg-type]

    assert fp.cluster.node_count == 0
    assert any("state" in i.message for i in issues)
    assert all(i.severity == "warning" for i in issues)


def test_classify_cache_warm_and_mixed() -> None:
    warm = RunSpec(
        model_set="m",
        test_set="t",
        mode="plan",
        reuse_existing_instances=True,
        ensure_store_downloads=False,
    )
    mixed = RunSpec(
        model_set="m", test_set="t", mode="plan", delete_staged_models=True
    )
    unknown = RunSpec(
        model_set="m", test_set="t", mode="plan", ensure_store_downloads=True
    )

    assert _classify_cache(warm) == "warm"
    assert _classify_cache(mixed) == "mixed"
    assert _classify_cache(unknown) == "unknown"
