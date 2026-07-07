"""Runtime and cluster fingerprint gathering for durable harness reports.

Populates a :class:`ReportFingerprint` (results-ledger schema 2.0) so a run's
report.json is self-describing: which harness code ran it, on what Python and
packages, against which Skulk version and cluster, under which cache flags.

Every probe is best-effort. A failure yields ``"unknown"`` (and, for probes
tied to the cluster, an :class:`Issue`), never an exception that would fail the
report write.
"""

from __future__ import annotations

import platform as _platform
import subprocess
import sys
from importlib import metadata as _metadata
from pathlib import Path
from typing import Literal

from .models import (
    CacheState,
    ClusterFingerprint,
    ClusterNodeFingerprint,
    Issue,
    RepoRef,
    ReportFingerprint,
    RunSpec,
    RuntimeFingerprint,
    SourceContext,
)

# Harness-side packages worth recording. mlx / skulk are NOT here: the harness
# is an HTTP client and does not import them; they live on the nodes and come
# from the API diagnostics instead.
_HARNESS_PACKAGES = ("skulk-test-harness", "httpx", "pydantic")


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command in ``repo``; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _repo_ref(name: str, repo: Path) -> RepoRef | None:
    """Git provenance for ``repo``, or None if it is not a git checkout."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo, "status", "--porcelain")
    return RepoRef(
        name=name,
        path=str(repo),
        branch=branch,
        commit=commit[:12],
        dirty=bool(status) if status is not None else None,
    )


def _harness_packages() -> dict[str, str]:
    """Best-effort version map for the harness's own runtime packages."""
    versions: dict[str, str] = {}
    for pkg in _HARNESS_PACKAGES:
        try:
            versions[pkg] = _metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def _classify_cache(spec: RunSpec) -> Literal["unknown", "cold", "warm", "mixed"]:
    """Coarse cache classification from the spec's flags.

    Honest, not a controlled-benchmark claim: if the run does not force
    downloads and reuses instances, it is "warm"; if it evicts staged models,
    "mixed"; otherwise "unknown". Never asserts "cold" from flags alone.
    """
    if spec.delete_staged_models:
        return "mixed"
    if spec.reuse_existing_instances and not spec.ensure_store_downloads:
        return "warm"
    return "unknown"


def _cluster_fingerprint(client: object) -> tuple[ClusterFingerprint, list[Issue]]:
    """Build the cluster fingerprint from ``/state`` + node diagnostics.

    ``client`` is a ``SkulkClient``; typed as object to avoid an import cycle.
    """
    issues: list[Issue] = []
    fp = ClusterFingerprint(api_base_url=getattr(client, "base_url", None))

    try:
        diag = client.get_diagnostics_node()  # type: ignore[attr-defined]
        runtime = diag.get("runtime", {}) if isinstance(diag, dict) else {}
        if isinstance(runtime, dict):
            fp.api_node_id = _as_str(runtime.get("nodeId"))
            fp.master_node_id = _as_str(runtime.get("masterNodeId"))
    except Exception as exc:
        issues.append(
            Issue(
                severity="warning",
                message="fingerprint: node diagnostics probe failed",
                evidence={"error": str(exc)},
            )
        )

    try:
        state = client.get_state()  # type: ignore[attr-defined]
    except Exception as exc:
        issues.append(
            Issue(
                severity="warning",
                message="fingerprint: /state probe failed for cluster fingerprint",
                evidence={"error": str(exc)},
            )
        )
        return fp, issues

    identities = state.get("nodeIdentities") if isinstance(state, dict) else None
    node_memory = state.get("nodeMemory") if isinstance(state, dict) else None
    node_system = state.get("nodeSystem") if isinstance(state, dict) else None
    last_seen = state.get("lastSeen") if isinstance(state, dict) else None

    node_ids = list(last_seen.keys()) if isinstance(last_seen, dict) else []
    fp.node_count = len(node_ids)
    names: list[str] = []
    for nid in node_ids:
        ident = identities.get(nid, {}) if isinstance(identities, dict) else {}
        name = _as_str(ident.get("friendlyName")) if isinstance(ident, dict) else None
        version = _as_str(ident.get("skulkVersion")) if isinstance(ident, dict) else None
        if name:
            names.append(name)
        mem = node_memory.get(nid, {}) if isinstance(node_memory, dict) else {}
        ram = None
        if isinstance(mem, dict):
            rt = mem.get("ramTotal") or mem.get("ram_total")
            if isinstance(rt, dict):
                ram = rt.get("inBytes")
            elif isinstance(rt, int):
                ram = rt
        sysprof = node_system.get(nid, {}) if isinstance(node_system, dict) else {}
        accel = sysprof.get("accelerator") if isinstance(sysprof, dict) else None
        vendor = _as_str(accel.get("vendor")) if isinstance(accel, dict) else None
        fp.nodes.append(
            ClusterNodeFingerprint(
                node_id=nid,
                friendly_name=name,
                ram_total_bytes=ram if isinstance(ram, int) else None,
                accelerator_vendor=vendor,
                skulk_version=version,
                system_telemetry_present=isinstance(node_system, dict) and nid in node_system,
                memory_telemetry_present=isinstance(node_memory, dict) and nid in node_memory,
            )
        )
    if names:
        fp.topology_label = "-".join(sorted(names))
    return fp, issues


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def gather_fingerprint(
    client: object,
    spec: RunSpec,
    harness_repo: Path | None = None,
    run_reason: str | None = None,
    operator_note: str | None = None,
) -> tuple[ReportFingerprint, list[Issue]]:
    """Assemble a run's fingerprint; return it plus any probe issues.

    Best-effort throughout: this never raises. Cluster probes may add warning
    issues that the caller folds into the report.
    """
    issues: list[Issue] = []

    harness_repo = harness_repo or Path(__file__).resolve().parents[2]
    repos: list[RepoRef] = []
    self_ref = _repo_ref("Foxlight-Foundation/skulk-test-harness", harness_repo)
    if self_ref is not None:
        repos.append(self_ref)

    cluster, cluster_issues = _cluster_fingerprint(client)
    issues.extend(cluster_issues)

    skulk_version = skulk_commit = None
    try:
        diag = client.get_diagnostics_node()  # type: ignore[attr-defined]
        runtime = diag.get("runtime", {}) if isinstance(diag, dict) else {}
        if isinstance(runtime, dict):
            skulk_version = _as_str(runtime.get("skulkVersion"))
            skulk_commit = _as_str(runtime.get("skulkCommit"))
        if skulk_commit or skulk_version:
            repos.append(
                RepoRef(
                    name="Foxlight-Foundation/Skulk",
                    branch=None,
                    commit=skulk_commit,
                    dirty=None,
                )
            )
    except Exception:
        # already surfaced via _cluster_fingerprint's diagnostics probe
        pass

    fingerprint = ReportFingerprint(
        source_context=SourceContext(
            run_reason=run_reason or "unspecified",
            visibility="private",
            operator_note=operator_note or spec.run_name,
            repositories=repos,
        ),
        runtime=RuntimeFingerprint(
            python=_platform.python_version(),
            platform=f"{_platform.system()} {_platform.release()} {_platform.machine()}",
            harness_packages=_harness_packages(),
            skulk_version=skulk_version,
            skulk_commit=skulk_commit,
        ),
        cluster=cluster,
        cache_state=CacheState(
            ensure_store_downloads=spec.ensure_store_downloads,
            reuse_existing_instances=spec.reuse_existing_instances,
            retain_instances=spec.retain_instances,
            delete_staged_models=spec.delete_staged_models,
            classification=_classify_cache(spec),
        ),
    )
    _ = sys  # reserved for future entrypoint capture
    return fingerprint, issues
