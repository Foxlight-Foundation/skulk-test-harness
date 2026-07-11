"""Unit tests for community submission (skulk-harness submit). Fully offline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from skulk_test_harness import submit


def _raw_report() -> dict:
    return {
        "run_id": "20260710-120000-suite-set",
        "spec": {"model_set": "m", "test_set": "t", "mode": "run", "run_name": "my-secret-box"},
        "results": [
            {
                "model_id": "org/model",
                "test_name": "t1",
                "repetition": 0,
                "passed": True,
                "output_text": "PRIVATE GENERATED TEXT",
                "reasoning_text": "PRIVATE REASONING",
                "tool_calls": [{"name": "x"}],
                "artifact_path": "/Users/someone/runs/x",
                "metrics": {"ttft_s": 0.4, "skulk_generation_tps": 40.0},
            }
        ],
        "placements": [{"model_id": "org/model", "node_ids": ["nodeA"]}],
        "fingerprint": {
            "schema_version": "2.1",
            "source_context": {
                "operator_note": "ran from the attic box",
                "repositories": [{"name": "skulk", "path": "/Users/someone/skulk", "commit": "abc"}],
            },
            "cluster": {
                "api_base_url": "http://10.0.0.5:52415",
                "topology_label": "attic1-attic2",
                "node_count": 1,
                "nodes": [
                    {
                        "node_id": "nodeA",
                        "friendly_name": "attic1",
                        "ram_total_bytes": 17179869184,
                        "accelerator_vendor": "apple",
                        "accelerator_name": "M4",
                    }
                ],
            },
        },
    }


def test_slim_and_redact_strips_text_and_identity_keeps_attribution_facts() -> None:
    payload = submit.slim_and_redact_report(_raw_report())

    result = payload["results"][0]
    for gone in ("output_text", "reasoning_text", "tool_calls", "artifact_path"):
        assert gone not in result
    # Metrics and pass/fail survive: they ARE the submission.
    assert result["metrics"]["skulk_generation_tps"] == 40.0
    assert result["passed"] is True

    fp = payload["fingerprint"]
    assert "operator_note" not in fp["source_context"]
    assert "path" not in fp["source_context"]["repositories"][0]
    cluster = fp["cluster"]
    assert "api_base_url" not in cluster
    assert "topology_label" not in cluster
    node = cluster["nodes"][0]
    assert "friendly_name" not in node
    # node_id is KEPT: the ledger joins placements for exact hardware
    # attribution and hashes ids before publishing.
    assert node["node_id"] == "nodeA"
    assert node["accelerator_name"] == "M4"
    assert "run_name" not in payload["spec"]


def test_redact_run_id_keeps_default_shape_and_hashes_custom_labels() -> None:
    default = _raw_report()
    default["run_id"] = "20260710-120000-m-t"  # slug(model_set-test_set)
    assert submit.slim_and_redact_report(default)["run_id"] == "20260710-120000-m-t"

    labeled = _raw_report()
    labeled["run_id"] = "20260710-120000-acme-lab-3-smoketest"
    out1 = submit.slim_and_redact_report(labeled)["run_id"]
    out2 = submit.slim_and_redact_report(labeled)["run_id"]
    assert "acme" not in out1
    assert re.match(r"^20260710-120000-submitted-[0-9a-f]{10}$", out1)
    # Deterministic: resubmission dedups server-side.
    assert out1 == out2


def test_slim_and_redact_never_mutates_the_input() -> None:
    raw = _raw_report()
    submit.slim_and_redact_report(raw)
    assert raw["results"][0]["output_text"] == "PRIVATE GENERATED TEXT"
    assert raw["fingerprint"]["cluster"]["nodes"][0]["friendly_name"] == "attic1"


def test_locate_report_accepts_dir_and_file(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert submit.locate_report(tmp_path) == report
    assert submit.locate_report(report) == report
    with pytest.raises(submit.SubmitError):
        submit.locate_report(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(submit.SubmitError):
        submit.locate_report(empty)


def test_resolve_github_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "env-token")
    assert submit.resolve_github_token("explicit") == "explicit"
    assert submit.resolve_github_token(None) == "env-token"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.setenv("GITHUB_TOKEN", "gha-token")
    assert submit.resolve_github_token(None) == "gha-token"


def test_resolve_github_token_fails_loud_without_any_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Force the gh-CLI fallback to fail regardless of the machine's setup.
    monkeypatch.setenv("PATH", "")
    with pytest.raises(submit.SubmitError, match="no GitHub token"):
        submit.resolve_github_token(None)


def test_post_submission_success_and_gate_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["auth"] = kwargs["headers"]["authorization"]
        return httpx.Response(201, json={"accepted": True, "status": "pending"})

    monkeypatch.setattr(submit.httpx, "post", fake_post)
    body = submit.post_submission({"run_id": "x"}, "tok", "https://ingest.example")
    assert body["accepted"] is True
    assert captured["url"] == "https://ingest.example/v1/submissions"
    assert captured["auth"] == "Bearer tok"

    def fake_reject(url: str, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        return httpx.Response(422, json={"error": "submission rejected by gates", "details": ["bad"]})

    monkeypatch.setattr(submit.httpx, "post", fake_reject)
    with pytest.raises(submit.SubmitError, match="422.*bad"):
        submit.post_submission({"run_id": "x"}, "tok", "https://ingest.example")
