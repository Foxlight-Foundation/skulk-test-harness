"""Community submission of harness runs to the Foxlight open ledger.

Pure logic for ``skulk-harness submit``: load a local ``report.json``, slim
and redact it CLIENT-side (the operator can inspect the exact payload with
``--dry-run`` before anything leaves the machine), resolve a GitHub token for
attribution, and POST to the ingest API. Never contacts a Skulk cluster.

Redaction philosophy: strip everything operator-identifying that the ledger
does not need (friendly node names, API URLs, operator notes, local repo
paths) and all generated text (prompt/output/reasoning/tool calls). Node ids
are KEPT: the ledger joins placements to fingerprint nodes for exact
hardware attribution, then hashes ids before anything is published.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from skulk_test_harness.utils import slugify

DEFAULT_INGEST_URL = "https://skulk-ledger-ingest.thomastupper92618.workers.dev"

#: Per-result fields that never leave the machine (generated text + local paths).
_RESULT_STRIP_FIELDS = ("output_text", "reasoning_text", "tool_calls", "artifact_path")


class SubmitError(RuntimeError):
    """A submission problem the operator must resolve (bad input, no token)."""


def slim_and_redact_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a submission payload: slimmed of text, redacted of identity.

    Works on the raw ``report.json`` dict so it never depends on the report
    having been produced by this exact harness version.
    """
    report = json.loads(json.dumps(raw))  # deep copy; payload must not alias input

    for result in report.get("results") or []:
        if isinstance(result, dict):
            for field in _RESULT_STRIP_FIELDS:
                result.pop(field, None)
            _strip_issue_evidence(result.get("issues"))
    _strip_issue_evidence(report.get("issues"))

    fingerprint = report.get("fingerprint")
    if isinstance(fingerprint, dict):
        source = fingerprint.get("source_context")
        if isinstance(source, dict):
            source.pop("operator_note", None)
            for repo in source.get("repositories") or []:
                if isinstance(repo, dict):
                    repo.pop("path", None)
        cluster = fingerprint.get("cluster")
        if isinstance(cluster, dict):
            cluster.pop("api_base_url", None)
            for node in cluster.get("nodes") or []:
                if isinstance(node, dict):
                    node.pop("friendly_name", None)
            cluster.pop("topology_label", None)

    spec = report.get("spec")
    if isinstance(spec, dict):
        spec.pop("run_name", None)

    _redact_run_id(report)

    return report


def _strip_issue_evidence(issues: Any) -> None:
    """Drop issue evidence wholesale: it can embed generated content (e.g.
    ``actual_tool_calls`` with argument text on failed tool-call tests), and
    the ledger only ever renders an issue's severity and message."""
    if not isinstance(issues, list):
        return
    for issue in issues:
        if isinstance(issue, dict):
            issue.pop("evidence", None)


def _redact_run_id(report: dict[str, Any]) -> None:
    """Strip operator labels from run_id while preserving dedup identity.

    ``_run_id`` embeds the slugified ``--run-name`` when one was given
    (default runs embed ``<model-set>-<test-set>``), so a custom label can
    carry a host/lab/customer name into the very field the ledger keys on.
    Default-shaped suffixes are kept (they are harness vocabulary, not
    operator identity); anything else becomes a deterministic hash of the
    original id, so re-submitting the same run still dedups server-side.
    """
    run_id = report.get("run_id")
    if not isinstance(run_id, str):
        return
    match = re.match(r"^(\d{8}-\d{6})-(.+)$", run_id)
    if not match:
        return
    stamp, suffix = match.groups()
    spec = report.get("spec")
    model_set = spec.get("model_set") if isinstance(spec, dict) else None
    test_set = spec.get("test_set") if isinstance(spec, dict) else None
    if (
        isinstance(model_set, str)
        and isinstance(test_set, str)
        and suffix == slugify(f"{model_set}-{test_set}")
    ):
        return
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:10]
    report["run_id"] = f"{stamp}-submitted-{digest}"


def locate_report(path: Path) -> Path:
    """Accept either a run directory or a report.json path."""
    if path.is_dir():
        candidate = path / "report.json"
        if not candidate.is_file():
            raise SubmitError(f"no report.json under {path}")
        return candidate
    if path.is_file():
        return path
    raise SubmitError(f"{path} is neither a run directory nor a report file")


def resolve_github_token(explicit: str | None = None) -> str:
    """Token precedence: --github-token, GH_TOKEN, GITHUB_TOKEN, `gh auth token`.

    Device-flow login needs a Foxlight OAuth app and is a planned follow-up;
    every early submitter realistically has one of these already.
    """
    for candidate in (explicit, os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_TOKEN")):
        if candidate:
            return candidate
    try:
        token = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        if token:
            return token
    except (OSError, subprocess.SubprocessError):
        pass
    raise SubmitError(
        "no GitHub token: pass --github-token, set GH_TOKEN, or log in with the gh CLI"
    )


def post_submission(
    payload: dict[str, Any], token: str, ingest_url: str = DEFAULT_INGEST_URL
) -> dict[str, Any]:
    """POST one report to the ingest API; return its JSON response.

    Raises SubmitError with the server's explanation on any non-2xx (the
    ingest returns structured errors for gate rejections, duplicates, and
    quota).
    """
    response = httpx.post(
        f"{ingest_url.rstrip('/')}/v1/submissions",
        json=payload,
        headers={"authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    body: dict[str, Any]
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text[:300]}
    if response.status_code >= 300:
        detail = body.get("details") or body.get("error") or response.status_code
        raise SubmitError(f"ingest rejected the submission ({response.status_code}): {detail}")
    return body
