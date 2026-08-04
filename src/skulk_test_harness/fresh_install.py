"""Fail-safe fresh-install qualification lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import BinaryIO, Literal, TypeVar

import httpx
import yaml

from skulk_test_harness.client import (
    SkulkApiError,
    SkulkClient,
    concurrent_benchmark_client,
    stream_chat_async,
)
from skulk_test_harness.dashboard_qualification import DashboardQualifier
from skulk_test_harness.fleet_lock import FleetLockStore
from skulk_test_harness.lease_heartbeat import (
    AuthoritativeLeaseHeartbeat,
    LeaseHeartbeatError,
)
from skulk_test_harness.models import (
    FreshInstallE2EBatteryEvidence,
    FreshInstallE2EResumptionEvidence,
    FreshInstallLifecycleStage,
    FreshInstallMemberEvidence,
    FreshInstallPhysicalFleet,
    FreshInstallPlatform,
    FreshInstallProfile,
    FreshInstallQualificationReport,
    FreshInstallTarget,
    HarnessConfig,
    InstallProvenance,
    Issue,
    ReleaseQualificationLegEvidence,
    ReleaseQualificationReport,
    RunPodFreshInstallConfig,
    RunReport,
    ServedEngineContract,
    ServedEngineEvidence,
)
from skulk_test_harness.qualification_checks import (
    UnexpectedFreshInstallPeerError,
    assert_fresh_cluster,
    assert_fresh_runtime_contract,
    assert_fresh_single_node,
    commit_matches,
    qualify_direct_text,
    qualify_direct_vision,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.runpod import RunPodClient, RunPodSshEndpoint
from skulk_test_harness.target_control import (
    RecoverySnapshot,
    SshTargetController,
)
from skulk_test_harness.vision_fixture import generate_vision_fixture

ResultT = TypeVar("ResultT")


@dataclass
class _PhysicalFleetMemberRuntime:
    """Private mutable lifecycle state for one whole-fleet member."""

    ordinal: int
    target_name: str
    target: FreshInstallTarget
    controller: SshTargetController
    local_port: int | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    snapshot: RecoverySnapshot | None = None
    service_stopped: bool = False
    temporary_root: str | None = None
    skulk_process: subprocess.Popen[bytes] | None = None
    skulk_log_handle: BinaryIO | None = None


@dataclass(frozen=True)
class _FreshE2EResumptionSource:
    """Validated predecessor artifacts eligible for one resumed provenance gate."""

    report: FreshInstallQualificationReport
    root: Path
    battery_log: Path
    report_paths: tuple[Path, ...]
    reports: tuple[RunReport, ...]
    expected_node_ids: frozenset[str]
    predecessor_harness_root: Path
    predecessor_harness_commit: str
    predecessor_harness_tree: str
    current_harness_commit: str
    current_harness_tree: str


class QualificationInterruptedError(BaseException):
    """Raised when a termination signal requests orderly restoration.

    Deliberately a ``BaseException``, for the same reason ``KeyboardInterrupt``
    is one. This is raised from a signal handler, so it lands wherever the
    interpreter happens to be executing, and the qualification is full of
    ``except Exception`` boundaries that exist to turn a browser or subprocess
    failure into a reported outcome. Inheriting from ``Exception`` let those
    boundaries swallow an operator's termination request and carry on: a
    signal arriving inside the consent-dialog probe was read as "no dialog",
    and one arriving anywhere in the browser journey was reported as a failed
    leg before the run continued to the next one. On a path that provisions
    billable cloud machines and destroys local state, an ignored stop request
    is not acceptable. ``finally`` blocks still run, so orderly restoration is
    unaffected.
    """


class QualificationSignalGuard(AbstractContextManager["QualificationSignalGuard"]):
    """Convert work signals into exceptions and defer recovery-time signals."""

    def __init__(self) -> None:
        self._previous: dict[int, object] = {}
        self._recovery_started = False
        self._interrupted_signum: int | None = None

    def __enter__(self) -> "QualificationSignalGuard":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)  # pyright: ignore[reportArgumentType]

    @property
    def interrupted_signum(self) -> int | None:
        """Return the first termination signal observed by this guard."""

        return self._interrupted_signum

    def begin_recovery(self) -> None:
        """Defer termination until mandatory restoration has completed.

        Signals received while product work is running still raise immediately
        so the run stops at the next Python boundary. Once a lifecycle enters
        its ``finally`` block, however, raising from the handler can interrupt
        the service restart, tunnel teardown, provider deletion, or lease
        release itself. Recovery records the stop request and completes before
        the report returns a blocking interrupted verdict.
        """

        self._recovery_started = True

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self._interrupted_signum is None:
            self._interrupted_signum = signum
        if self._recovery_started:
            return
        raise QualificationInterruptedError(
            f"fresh-install qualification interrupted by signal {signum}"
        )


class _FreshRuntimeMonitor(AbstractContextManager["_FreshRuntimeMonitor"]):
    """Remember topology violations that occur inside blocking user actions."""

    def __init__(
        self,
        *,
        api_base_url: str,
        expected_node_id: str | None = None,
        expected_node_ids: frozenset[str] | None = None,
        expected_node_count: int = 1,
        poll_interval_s: float,
        request_timeout_s: float,
    ) -> None:
        self.expected_node_id = expected_node_id
        self.expected_node_ids = expected_node_ids
        self.expected_node_count = expected_node_count
        self._api_base_url = api_base_url
        self._poll_interval_s = max(poll_interval_s, 1.0)
        self._request_timeout_s = request_timeout_s
        self._stop = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="fresh-install-runtime-monitor",
            daemon=True,
        )

    def __enter__(self) -> "_FreshRuntimeMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self._request_timeout_s + 1.0, 5.0))
        if self._thread.is_alive():
            self._record_failure(
                RuntimeError("fresh-install runtime monitor did not stop")
            )
        # Preserve an active product/recovery exception. Otherwise a violation
        # observed only by the monitor must still fail the enclosing journey.
        if not _exc or _exc[0] is None:
            self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Raise a previously observed topology or identity violation."""

        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def _record_failure(self, failure: Exception) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure

    def _run(self) -> None:
        try:
            with SkulkClient(
                self._api_base_url,
                request_timeout_s=self._request_timeout_s,
            ) as client:
                while not self._stop.is_set():
                    if self.expected_node_count == 1:
                        assert_fresh_single_node(
                            client,
                            expected_node_id=self.expected_node_id,
                        )
                    else:
                        assert_fresh_cluster(
                            client,
                            expected_node_count=self.expected_node_count,
                            expected_node_ids=self.expected_node_ids,
                        )
                    if self._stop.wait(self._poll_interval_s):
                        return
        except Exception as exception:  # noqa: BLE001 - monitored failure boundary
            self._record_failure(exception)
            self._stop.set()


class _LifecycleJournal:
    """Append durable stage transitions to a report as work progresses."""

    def __init__(
        self,
        report: FreshInstallQualificationReport,
        writer: ReportWriter,
    ) -> None:
        self.report = report
        self.writer = writer

    def stage(self, name: str) -> "_StageContext":
        """Create one running stage context."""

        return _StageContext(self, name)

    def persist(self) -> None:
        """Write the report after every externally meaningful transition."""

        self.writer.write_fresh_install(self.report)


class _StageContext(AbstractContextManager[FreshInstallLifecycleStage]):
    def __init__(self, journal: _LifecycleJournal, name: str) -> None:
        self._journal = journal
        self._stage = FreshInstallLifecycleStage(
            name=name,
            status="running",
            started_at=datetime.now(UTC),
        )

    def __enter__(self) -> FreshInstallLifecycleStage:
        self._journal.report.lifecycle.append(self._stage)
        self._journal.persist()
        return self._stage

    def __exit__(self, exc_type: object, exc: object, _traceback: object) -> None:
        self._stage.finished_at = datetime.now(UTC)
        if exc is None:
            self._stage.status = "passed"
            if self._stage.message is None:
                self._stage.message = "completed"
        else:
            self._stage.status = "failed"
            self._stage.message = str(exc)
        self._journal.persist()


class FreshInstallQualifier:
    """Orchestrate physical and RunPod fresh-install qualification legs."""

    def __init__(self, config: HarnessConfig) -> None:
        if config.fresh_install is None:
            raise ValueError("fresh_install configuration is required")
        self.config = config
        self.fresh = config.fresh_install
        self.writer = ReportWriter(config.output_dir)

    def qualify_release_matrix(
        self,
        *,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        resume_from: Path | None = None,
    ) -> ReleaseQualificationReport:
        """Run one atomic fresh physical E2E plus RunPod release gate.

        ``resume_from`` is accepted only for a predecessor that completed every
        E2E cell successfully and then failed the harness provenance gate. The
        new run still performs a normal whole-fleet fresh install before
        replaying that failed gate.
        """

        with QualificationSignalGuard() as signal_guard:
            return self._qualify_release_matrix(
                profile=profile,
                expected_commit=expected_commit,
                resume_from=resume_from,
                signal_guard=signal_guard,
            )

    def _qualify_release_matrix(
        self,
        *,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        resume_from: Path | None,
        signal_guard: QualificationSignalGuard,
    ) -> ReleaseQualificationReport:
        """Execute the composite gate under one outer interruption guard."""

        _require_commit_sha(expected_commit)
        assert expected_commit is not None
        if self.config.fleet_lock is None:
            raise ValueError("release qualification requires fleet_lock")
        physical_fleets = self.fresh.eligible_physical_fleets()
        fleet_member_names = {
            target_name
            for _fleet_name, fleet in physical_fleets
            for target_name in fleet.member_targets
        }
        targets = [
            (name, target)
            for name, target in self.fresh.eligible_targets()
            if name not in fleet_member_names
        ]
        self.fresh.assert_complete_release_matrix(targets, physical_fleets)
        if not physical_fleets:
            raise ValueError(
                "release qualification requires an eligible whole physical fleet"
            )
        if len(physical_fleets) != 1:
            raise ValueError(
                "release qualification requires exactly one eligible whole "
                "physical fleet containing every release node"
            )
        unexpected_physical = [
            name for name, target in targets if target.kind == "physical"
        ]
        if unexpected_physical:
            raise ValueError(
                "release qualification cannot run physical targets separately: "
                f"{unexpected_physical}"
            )
        runpod_targets = [
            (name, target) for name, target in targets if target.kind == "runpod"
        ]
        if not runpod_targets:
            raise ValueError("release qualification requires an eligible RunPod target")
        repository_root = Path(__file__).resolve().parents[2]
        _qualification_source_path(
            repository_root,
            physical_fleets[0][1].e2e_battery_script,
            label="full E2E battery script",
        )
        _qualification_source_path(
            repository_root,
            self.config.model_sets_path,
            label="model-set matrix",
        )
        _qualification_source_path(
            repository_root,
            self.config.test_sets_path,
            label="test-set matrix",
        )
        resumption_source = (
            _prepare_e2e_resumption_source(
                resume_from=resume_from,
                repository_root=repository_root,
                fleet=physical_fleets[0][1],
                model_sets_path=self.config.model_sets_path,
                test_sets_path=self.config.test_sets_path,
                profile=profile,
                expected_commit=expected_commit,
            )
            if resume_from is not None
            else None
        )

        qualification_id = (
            f"release-{profile}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        report = ReleaseQualificationReport(
            qualification_id=qualification_id,
            profile=profile,
            expected_commit=expected_commit,
            required_platforms=list(self.fresh.required_platforms),
            started_at=datetime.now(UTC),
        )
        self.writer.write_release_qualification(report)
        store = FleetLockStore(self.config.fleet_lock)
        heartbeat = AuthoritativeLeaseHeartbeat(
            store,
            holder=self.config.fleet_lock.holder,
            ttl_s=self.fresh.lease_ttl_s,
            interval_s=self.fresh.resolved_lease_heartbeat_s,
            on_verified_expiry=report.lease_renewal_expiries.append,
        )
        acquired = False
        heartbeat_failed = False
        release_failed = False
        try:
            outcome = store.acquire(
                branch=expected_commit,
                host=socket.gethostname(),
                battery="fresh-install-release-qualification",
                ttl_s=self.fresh.lease_ttl_s,
                note=f"{profile} complete physical E2E plus RunPod",
            )
            if not outcome.ok:
                raise RuntimeError(outcome.message)
            acquired = True
            heartbeat.start()
            self.writer.write_release_qualification(report)

            for fleet_name, fleet in physical_fleets:
                child = self.qualify_physical_fleet(
                    fleet_name=fleet_name,
                    fleet=fleet,
                    profile=profile,
                    expected_commit=expected_commit,
                    shared_heartbeat=heartbeat,
                    e2e_resumption_source=resumption_source,
                )
                report.legs.append(_release_leg_evidence(child))
                self.writer.write_release_qualification(report)
                if not child.passed:
                    if child.critical_recovery_required:
                        report.critical_recovery_required = True
                    raise RuntimeError(
                        f"mandatory physical fleet leg failed: {child.qualification_id}"
                    )

            for target_name, target in runpod_targets:
                heartbeat.raise_if_failed()
                child = self.qualify_target(
                    target_name=target_name,
                    target=target,
                    profile=profile,
                    expected_commit=expected_commit,
                    heartbeat=heartbeat,
                )
                report.legs.append(_release_leg_evidence(child))
                self.writer.write_release_qualification(report)
                if not child.passed:
                    if child.critical_recovery_required:
                        report.critical_recovery_required = True
                    raise RuntimeError(
                        f"mandatory RunPod leg failed: {child.qualification_id}"
                    )

            heartbeat.raise_if_failed()
            try:
                _assert_restored_fleet_clean_through_target(
                    self.fresh.targets[physical_fleets[0][1].entrypoint_target],
                    expected_node_count=len(physical_fleets[0][1].member_targets),
                    remote_port=self.fresh.remote_port,
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                )
            except Exception as exception:
                # The child lifecycle has already attempted restoration. A
                # failed independent audit means that recovery is not proven,
                # so the authoritative lease must remain held for an operator.
                report.critical_recovery_required = True
                raise RuntimeError(
                    f"restored fleet composite audit failed: {exception}"
                ) from exception
        except QualificationInterruptedError as exception:
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"release qualification interrupted: {exception}",
                )
            )
        except Exception as exception:  # noqa: BLE001 - composite gate boundary
            heartbeat_failed = isinstance(exception, LeaseHeartbeatError)
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"release qualification failed: {exception}",
                )
            )
        finally:
            # Signals during product work stop the gate immediately. From here
            # onward they are recorded but deferred so provider teardown, fleet
            # restoration auditing, and lease disposition cannot be interrupted.
            signal_guard.begin_recovery()
            try:
                heartbeat.raise_if_failed()
            except LeaseHeartbeatError as exception:
                heartbeat_failed = True
                report.issues.append(
                    Issue(
                        severity="error", message=f"lease heartbeat failed: {exception}"
                    )
                )
            heartbeat.stop()
            try:
                heartbeat.raise_if_failed()
            except LeaseHeartbeatError as exception:
                heartbeat_failed = True
                if not any(
                    "lease heartbeat failed" in issue.message for issue in report.issues
                ):
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"lease heartbeat failed: {exception}",
                        )
                    )

            if (
                acquired
                and not report.critical_recovery_required
                and not heartbeat_failed
            ):
                release = store.release()
                if release.ok:
                    report = report.model_copy(update={"lease_released": True})
                else:
                    release_failed = True
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"fleet lease release failed: {release.message}",
                        )
                    )
            if acquired and (
                report.critical_recovery_required or heartbeat_failed or release_failed
            ):
                report.critical_recovery_required = True
                try:
                    heartbeat.emergency_extend(ttl_s=self.fresh.emergency_lease_ttl_s)
                except LeaseHeartbeatError as exception:
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"emergency lease extension failed: {exception}",
                        )
                    )
                report.issues.append(
                    Issue(
                        severity="error",
                        message=(
                            "fleet lease intentionally remains held pending "
                            "operator recovery"
                        ),
                    )
                )
            covered_platforms = {
                platform
                for leg in report.legs
                if leg.passed
                for platform in leg.covered_platforms
            }
            physical_e2e_passed = any(
                leg.platform == "mixed" and leg.complete_e2e_passed is True
                for leg in report.legs
            )
            _record_deferred_interruption(report, signal_guard)
            passed = (
                not report.issues
                and set(report.required_platforms).issubset(covered_platforms)
                and physical_e2e_passed
                and report.lease_released
                and not report.critical_recovery_required
            )
            report = report.finish(
                passed=passed,
                lease_released=report.lease_released,
            )
            self.writer.write_release_qualification(report)
        return report

    def qualify_target(
        self,
        *,
        target_name: str,
        target: FreshInstallTarget,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        heartbeat: AuthoritativeLeaseHeartbeat | None = None,
    ) -> FreshInstallQualificationReport:
        """Run one explicitly eligible target leg."""

        if not target.eligible:
            raise ValueError(f"fresh-install target {target_name!r} is not eligible")
        _require_commit_sha(expected_commit)
        qualification_id = (
            f"fresh-{profile}-{target.platform}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        artifact_directory = self.writer.run_dir(qualification_id)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        artifact_directory.chmod(0o700)
        report = FreshInstallQualificationReport(
            qualification_id=qualification_id,
            profile=profile,
            platform=target.platform,
            hardware_class=target.hardware_class,
            started_at=datetime.now(UTC),
            install=InstallProvenance(
                mode="fresh_install",
                environment="fresh_install",
                profile=profile,
                platform=target.platform,
                hardware_class=target.hardware_class,
                expected_commit=expected_commit,
                environment_override_names=[],
            ),
            artifact_directory=artifact_directory,
        )
        journal = _LifecycleJournal(report, self.writer)
        journal.persist()
        with QualificationSignalGuard() as signal_guard:
            if target.kind == "runpod":
                return self._qualify_runpod(
                    target=target,
                    profile=profile,
                    expected_commit=expected_commit,
                    report=report,
                    journal=journal,
                    artifact_directory=artifact_directory,
                    signal_guard=signal_guard,
                    heartbeat=heartbeat,
                )
            return self._qualify_physical(
                target=target,
                profile=profile,
                expected_commit=expected_commit,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                signal_guard=signal_guard,
            )

    def qualify_physical_fleet(
        self,
        *,
        fleet_name: str,
        fleet: FreshInstallPhysicalFleet,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        shared_heartbeat: AuthoritativeLeaseHeartbeat | None = None,
        e2e_resumption_source: _FreshE2EResumptionSource | None = None,
    ) -> FreshInstallQualificationReport:
        """Install and qualify every member of one physical topology together."""

        if not fleet.eligible:
            raise ValueError(
                f"fresh-install physical fleet {fleet_name!r} is not eligible"
            )
        _require_commit_sha(expected_commit)
        members = [
            _PhysicalFleetMemberRuntime(
                ordinal=index,
                target_name=target_name,
                target=target,
                controller=SshTargetController(target),
            )
            for index, (target_name, target) in enumerate(
                self.fresh.physical_fleet_targets(fleet),
                start=1,
            )
        ]
        entrypoint = next(
            member
            for member in members
            if member.target_name == fleet.entrypoint_target
        )
        e2e_entrypoint = next(
            member
            for member in members
            if member.target_name
            == (fleet.e2e_entrypoint_target or fleet.entrypoint_target)
        )
        contract_members = [
            next(member for member in members if member.target_name == target_name)
            for target_name in fleet.qualification_targets
        ]
        qualification_id = (
            f"fresh-{profile}-mixed-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        artifact_directory = self.writer.run_dir(qualification_id)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        artifact_directory.chmod(0o700)
        report = FreshInstallQualificationReport(
            qualification_id=qualification_id,
            profile=profile,
            platform="mixed",
            hardware_class=fleet.hardware_class,
            started_at=datetime.now(UTC),
            install=InstallProvenance(
                mode="fresh_install",
                environment="fresh_install",
                profile=profile,
                platform="mixed",
                hardware_class=fleet.hardware_class,
                expected_commit=expected_commit,
                environment_override_names=[],
            ),
            members=[
                FreshInstallMemberEvidence(
                    ordinal=member.ordinal,
                    platform=member.target.platform,
                    hardware_class=member.target.hardware_class,
                    requested_ref=(
                        expected_commit if profile == "candidate" else "main"
                    ),
                    expected_backends=member.target.expected_backends,
                )
                for member in members
            ],
            artifact_directory=artifact_directory,
        )
        journal = _LifecycleJournal(report, self.writer)
        journal.persist()
        with QualificationSignalGuard() as signal_guard:
            return self._qualify_physical_fleet(
                fleet=fleet,
                members=members,
                entrypoint=entrypoint,
                e2e_entrypoint=e2e_entrypoint,
                contract_members=contract_members,
                profile=profile,
                expected_commit=expected_commit,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                signal_guard=signal_guard,
                shared_heartbeat=shared_heartbeat,
                e2e_resumption_source=e2e_resumption_source,
            )

    def _qualify_physical(
        self,
        *,
        target: FreshInstallTarget,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        signal_guard: QualificationSignalGuard,
    ) -> FreshInstallQualificationReport:
        if self.config.fleet_lock is None:
            raise ValueError("physical fresh-install qualification requires fleet_lock")
        if not target.isolation_enter_command or not target.isolation_exit_command:
            raise ValueError(
                "single-node physical qualification requires reversible "
                "Skulk-network isolation; use a physical_fleet for normal networking"
            )
        store = FleetLockStore(self.config.fleet_lock)
        owns_lease = True
        controller = SshTargetController(target)
        heartbeat = AuthoritativeLeaseHeartbeat(
            store,
            holder=self.config.fleet_lock.holder,
            ttl_s=self.fresh.lease_ttl_s,
            interval_s=self.fresh.resolved_lease_heartbeat_s,
            on_verified_expiry=report.lease_renewal_expiries.append,
        )
        acquired = False
        service_stopped = False
        isolation_entered = False
        restoration_succeeded = False
        heartbeat_failed = False
        snapshot: RecoverySnapshot | None = None
        tunnel: subprocess.Popen[bytes] | None = None
        local_port: int | None = None
        original_diagnostics: dict[str, object] = {}
        try:
            with journal.stage("acquire authoritative fleet lease"):
                outcome = store.acquire(
                    branch=expected_commit or "main",
                    host=socket.gethostname(),
                    battery="fresh-install-qualification",
                    ttl_s=self.fresh.lease_ttl_s,
                    note=f"{profile} {target.platform}",
                )
                if not outcome.ok:
                    raise RuntimeError(outcome.message)
                acquired = True
                heartbeat.start()

            with journal.stage("open target API tunnel"):
                local_port, tunnel = controller.open_tunnel(
                    remote_port=self.fresh.remote_port
                )
                original_node_id, original_node_count = _wait_for_api_identity(
                    f"http://127.0.0.1:{local_port}",
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                )
                with SkulkClient(f"http://127.0.0.1:{local_port}") as client:
                    original_diagnostics = client.get_diagnostics_node()

            with journal.stage("capture dual recovery snapshots"):
                snapshot = controller.capture_recovery_snapshot(
                    qualification_id=report.qualification_id,
                    controller_root=self.fresh.snapshot_root,
                    retention_days=self.fresh.snapshot_retention_days,
                    api_node_id=original_node_id,
                    cluster_node_count=original_node_count,
                    api_diagnostics=original_diagnostics,
                )
                report.snapshot_target_sha256 = snapshot.remote_sha256
                report.snapshot_controller_sha256 = snapshot.controller_sha256
                journal.persist()
                heartbeat.raise_if_failed()

            with journal.stage("stop selected target service"):
                assert target.service_stop_command is not None
                # A stop command can mutate service state before returning a
                # failure code. From this point onward restoration is mandatory.
                service_stopped = True
                controller.run(target.service_stop_command, timeout_s=120)
                heartbeat.raise_if_failed()

            with journal.stage("isolate temporary node from the existing fabric"):
                assert target.isolation_enter_command is not None
                # The reversal is mandatory even when the enter command mutates
                # state and then reports failure.
                isolation_entered = True
                controller.run(target.isolation_enter_command, timeout_s=120)
                heartbeat.raise_if_failed()

            assert local_port is not None
            self._execute_clean_install(
                controller=controller,
                api_base_url=f"http://127.0.0.1:{local_port}",
                target=target,
                profile=profile,
                expected_commit=expected_commit,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                heartbeat=heartbeat,
                signal_guard=signal_guard,
            )
        except QualificationInterruptedError as exception:
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"fresh-install leg interrupted: {exception}",
                )
            )
        except Exception as exception:  # noqa: BLE001 - lifecycle failure boundary
            heartbeat_failed = isinstance(exception, LeaseHeartbeatError)
            report.issues.append(
                Issue(
                    severity="error", message=f"fresh-install leg failed: {exception}"
                )
            )
        finally:
            signal_guard.begin_recovery()
            isolation_restored = not isolation_entered
            if isolation_entered:
                try:
                    with journal.stage("remove temporary fabric isolation"):
                        assert target.isolation_exit_command is not None
                        controller.run(target.isolation_exit_command, timeout_s=120)
                    isolation_restored = True
                except Exception as exception:  # noqa: BLE001 - recovery boundary
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"critical isolation restoration failure: {exception}",
                        )
                    )
                    isolation_restored = False
            if service_stopped and snapshot is not None and local_port is not None:
                try:
                    service_restored = self._restore_physical(
                        controller=controller,
                        target=target,
                        snapshot=snapshot,
                        api_base_url=f"http://127.0.0.1:{local_port}",
                        report=report,
                        journal=journal,
                    )
                    restoration_succeeded = isolation_restored and service_restored
                except Exception as exception:  # noqa: BLE001 - recovery boundary
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"critical restoration failure: {exception}",
                        )
                    )
                    restoration_succeeded = False
            elif acquired:
                restoration_succeeded = isolation_restored
            report.restoration_succeeded = restoration_succeeded
            report.teardown_succeeded = restoration_succeeded
            try:
                heartbeat.raise_if_failed()
            except LeaseHeartbeatError as exception:
                heartbeat_failed = True
                report.issues.append(
                    Issue(
                        severity="error", message=f"lease heartbeat failed: {exception}"
                    )
                )
            if owns_lease:
                heartbeat.stop()
                try:
                    heartbeat.raise_if_failed()
                except LeaseHeartbeatError as exception:
                    if not heartbeat_failed:
                        report.issues.append(
                            Issue(
                                severity="error",
                                message=f"lease heartbeat failed: {exception}",
                            )
                        )
                    heartbeat_failed = True
            if tunnel is not None:
                _terminate_process(tunnel)
            release_failed = False
            if (
                owns_lease
                and acquired
                and restoration_succeeded
                and not heartbeat_failed
            ):
                try:
                    with journal.stage("release restored fleet lease"):
                        release = store.release()
                        if not release.ok:
                            raise RuntimeError(release.message)
                except Exception as exception:  # noqa: BLE001 - keep lease held
                    release_failed = True
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"fleet lease release failed: {exception}",
                        )
                    )
            if acquired and (
                not restoration_succeeded or heartbeat_failed or release_failed
            ):
                report.critical_recovery_required = True
                if owns_lease:
                    try:
                        heartbeat.emergency_extend(
                            ttl_s=self.fresh.emergency_lease_ttl_s
                        )
                    except LeaseHeartbeatError as exception:
                        report.issues.append(
                            Issue(
                                severity="error",
                                message=f"emergency lease extension failed: {exception}",
                            )
                        )
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=(
                                "fleet lease intentionally remains held pending "
                                "operator recovery"
                            ),
                        )
                    )
            _record_deferred_interruption(report, signal_guard)
            report = report.finish(
                passed=(
                    not _blocking_issues(report)
                    and not _failed_lifecycle_stages(report)
                    and restoration_succeeded
                    and not release_failed
                    and not report.critical_recovery_required
                )
            )
            journal.report = report
            journal.persist()
        return report

    def _qualify_physical_fleet(
        self,
        *,
        fleet: FreshInstallPhysicalFleet,
        members: list[_PhysicalFleetMemberRuntime],
        entrypoint: _PhysicalFleetMemberRuntime,
        e2e_entrypoint: _PhysicalFleetMemberRuntime,
        contract_members: list[_PhysicalFleetMemberRuntime],
        profile: FreshInstallProfile,
        expected_commit: str | None,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        signal_guard: QualificationSignalGuard,
        shared_heartbeat: AuthoritativeLeaseHeartbeat | None,
        e2e_resumption_source: _FreshE2EResumptionSource | None,
    ) -> FreshInstallQualificationReport:
        """Run one fail-safe unsandboxed whole-fleet qualification lifecycle."""

        if self.config.fleet_lock is None:
            raise ValueError("physical fresh-install qualification requires fleet_lock")
        if any(member.target.runtime_isolation_prefix for member in members):
            raise ValueError(
                "whole-fleet qualification must not declare runtime isolation prefixes"
            )
        store = FleetLockStore(self.config.fleet_lock)
        owns_lease = shared_heartbeat is None
        heartbeat = shared_heartbeat or AuthoritativeLeaseHeartbeat(
            store,
            holder=self.config.fleet_lock.holder,
            ttl_s=self.fresh.lease_ttl_s,
            interval_s=self.fresh.resolved_lease_heartbeat_s,
            on_verified_expiry=report.lease_renewal_expiries.append,
        )
        acquired = not owns_lease
        restoration_succeeded = False
        cleanup_succeeded = True
        heartbeat_failed = False
        release_failed = False
        original_api: dict[int, tuple[str, int, dict[str, object]]] = {}
        observed_member_node_ids: dict[int, frozenset[str]] = {}
        installer_url: str | None = None
        installer_digest: str | None = None
        expected_node_count = len(members)
        try:
            if owns_lease:
                with journal.stage("acquire authoritative fleet lease"):
                    outcome = store.acquire(
                        branch=expected_commit or "main",
                        host=socket.gethostname(),
                        battery="fresh-install-whole-fleet-qualification",
                        ttl_s=self.fresh.lease_ttl_s,
                        note=f"{profile} mixed {expected_node_count} nodes",
                    )
                    if not outcome.ok:
                        raise RuntimeError(outcome.message)
                    acquired = True
                    heartbeat.start()
            else:
                with journal.stage("verify composite release fleet lease"):
                    heartbeat.verify_current()

            with journal.stage("open API tunnels to every physical member"):

                def open_member(
                    member: _PhysicalFleetMemberRuntime,
                ) -> tuple[
                    int,
                    subprocess.Popen[bytes],
                    str,
                    int,
                    dict[str, object],
                    frozenset[str],
                ]:
                    local_port, tunnel = member.controller.open_tunnel(
                        remote_port=self.fresh.remote_port
                    )
                    member.local_port = local_port
                    member.tunnel = tunnel
                    node_id, node_count = _wait_for_api_identity(
                        f"http://127.0.0.1:{local_port}",
                        timeout_s=self.fresh.readiness_timeout_s,
                        poll_interval_s=self.fresh.poll_interval_s,
                    )
                    with SkulkClient(f"http://127.0.0.1:{local_port}") as client:
                        diagnostics = client.get_diagnostics_node()
                        observed_node_ids = assert_fresh_cluster(
                            client,
                            expected_node_count=expected_node_count,
                        )
                    return (
                        local_port,
                        tunnel,
                        node_id,
                        node_count,
                        diagnostics,
                        observed_node_ids,
                    )

                opened = _run_member_operations(members, open_member)
                for member, result in zip(members, opened, strict=True):
                    (
                        local_port,
                        tunnel,
                        node_id,
                        node_count,
                        diagnostics,
                        observed_node_ids,
                    ) = result
                    member.local_port = local_port
                    member.tunnel = tunnel
                    original_api[member.ordinal] = (
                        node_id,
                        node_count,
                        diagnostics,
                    )
                    observed_member_node_ids[member.ordinal] = observed_node_ids
                _assert_declared_member_topologies(
                    expected_node_count=expected_node_count,
                    local_node_ids=(
                        node_id
                        for node_id, _node_count, _diagnostics in original_api.values()
                    ),
                    member_observed_node_ids=(
                        observed_member_node_ids[member.ordinal] for member in members
                    ),
                )
                assert entrypoint.local_port is not None
                with SkulkClient(f"http://127.0.0.1:{entrypoint.local_port}") as client:
                    original_state = client.get_state()
                instances = original_state.get("instances")
                runners = original_state.get("runners")
                if (
                    isinstance(instances, dict)
                    and instances
                    or isinstance(runners, dict)
                    and runners
                ):
                    raise RuntimeError(
                        "physical fleet must be idle before fresh-install "
                        "qualification; active instances or runners were observed"
                    )
                heartbeat.raise_if_failed()

            with journal.stage("capture recovery snapshots for every member"):
                for member in members:
                    node_id, node_count, diagnostics = original_api[member.ordinal]
                    snapshot = member.controller.capture_recovery_snapshot(
                        qualification_id=(
                            f"{report.qualification_id}-member-{member.ordinal:02d}"
                        ),
                        controller_root=self.fresh.snapshot_root,
                        retention_days=self.fresh.snapshot_retention_days,
                        api_node_id=node_id,
                        cluster_node_count=node_count,
                        api_diagnostics=diagnostics,
                    )
                    member.snapshot = snapshot
                    evidence = report.members[member.ordinal - 1]
                    report.members[member.ordinal - 1] = evidence.model_copy(
                        update={
                            "snapshot_target_sha256": snapshot.remote_sha256,
                            "snapshot_controller_sha256": snapshot.controller_sha256,
                        }
                    )
                    journal.persist()
                    heartbeat.raise_if_failed()
                report.snapshot_target_sha256 = _aggregate_digests(
                    snapshot.remote_sha256
                    for snapshot in (member.snapshot for member in members)
                    if snapshot is not None
                )
                report.snapshot_controller_sha256 = _aggregate_digests(
                    snapshot.controller_sha256
                    for snapshot in (member.snapshot for member in members)
                    if snapshot is not None
                )
                journal.persist()

            with journal.stage("stop existing Skulk service on every member"):
                for member in members:
                    assert member.target.service_stop_command is not None
                    member.service_stopped = True
                    member.controller.run(
                        member.target.service_stop_command,
                        timeout_s=120,
                    )
                    heartbeat.raise_if_failed()

            with journal.stage("create empty temporary HOME on every member"):

                def create_member_home(
                    member: _PhysicalFleetMemberRuntime,
                ) -> None:
                    root = self._create_temporary_home(member.controller)
                    member.temporary_root = root

                _run_member_operations(members, create_member_home)

            with journal.stage("run official installer on every member"):
                installer_url, installer_digest = _installer_provenance(
                    self.fresh.installer_url,
                    profile=profile,
                    expected_commit=expected_commit,
                    shipping_ref=self.fresh.shipping_installer_ref,
                )

                def install_member(
                    member: _PhysicalFleetMemberRuntime,
                ) -> tuple[str, str]:
                    assert member.temporary_root is not None
                    member_artifacts = (
                        artifact_directory / "members" / f"{member.ordinal:02d}"
                    )
                    member_artifacts.mkdir(parents=True, exist_ok=True)
                    member_artifacts.chmod(0o700)
                    return self._install_member(
                        member=member,
                        profile=profile,
                        expected_commit=expected_commit,
                        installer_url=installer_url,
                        artifact_directory=member_artifacts,
                        heartbeat=heartbeat,
                    )

                installed = _run_member_operations(members, install_member)
                for member, (resolved_commit, config_digest) in zip(
                    members, installed, strict=True
                ):
                    evidence = report.members[member.ordinal - 1]
                    report.members[member.ordinal - 1] = evidence.model_copy(
                        update={
                            "resolved_commit": resolved_commit,
                            "generated_config_sha256": config_digest,
                        }
                    )
                report.install = report.install.model_copy(
                    update={
                        "installer_url": installer_url,
                        "installer_sha256": installer_digest,
                        "requested_ref": (
                            expected_commit if profile == "candidate" else "main"
                        ),
                        "generated_config_sha256": _aggregate_digests(
                            evidence.generated_config_sha256
                            for evidence in report.members
                            if evidence.generated_config_sha256 is not None
                        ),
                    }
                )
                journal.persist()
                heartbeat.raise_if_failed()

            with journal.stage("start fresh Skulk on every member without isolation"):

                def start_member(member: _PhysicalFleetMemberRuntime) -> None:
                    assert member.temporary_root is not None
                    assert member.local_port is not None
                    member_artifacts = (
                        artifact_directory / "members" / f"{member.ordinal:02d}"
                    )
                    command = _clean_environment_command(
                        member.temporary_root,
                        'cd "$HOME/skulk" && exec uv run skulk',
                    )
                    process, log_handle = member.controller.start(
                        command,
                        log_path=member_artifacts / "skulk.log",
                    )
                    member.skulk_process = process
                    member.skulk_log_handle = log_handle
                    _wait_for_http(
                        f"http://127.0.0.1:{member.local_port}/state",
                        timeout_s=self.fresh.readiness_timeout_s,
                        poll_interval_s=self.fresh.poll_interval_s,
                        heartbeat=heartbeat,
                    )

                _run_member_operations(members, start_member)

            assert entrypoint.local_port is not None
            api_base_url = f"http://127.0.0.1:{entrypoint.local_port}"
            expected_backends = sorted(
                {
                    backend
                    for member in members
                    for backend in member.target.expected_backends
                }
            )
            with journal.stage("assert fresh whole-fleet runtime contract"):
                provenance = _wait_for_runtime_contract(
                    api_base_url,
                    target=entrypoint.target,
                    expected_commit=expected_commit,
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                    stability_s=self.fresh.runtime_contract_stability_s,
                    heartbeat=heartbeat,
                    expected_node_count=expected_node_count,
                    expected_backends=expected_backends,
                )
                with SkulkClient(api_base_url) as client:
                    fresh_node_ids = assert_fresh_cluster(
                        client,
                        expected_node_count=expected_node_count,
                    )
                report.install = report.install.model_copy(
                    update={
                        **provenance.model_dump(),
                        "profile": profile,
                        "platform": "mixed",
                        "hardware_class": fleet.hardware_class,
                        "installer_url": installer_url,
                        "installer_sha256": installer_digest,
                        "requested_ref": (
                            expected_commit if profile == "candidate" else "main"
                        ),
                        "expected_commit": expected_commit,
                        "generated_config_sha256": (
                            report.install.generated_config_sha256
                        ),
                    }
                )
                self._record_member_runtime_evidence(
                    members=members,
                    expected_node_count=expected_node_count,
                    report=report,
                )
                journal.persist()
                heartbeat.raise_if_failed()

            self._qualify_fleet_models(
                api_base_url=api_base_url,
                members=members,
                contract_members=contract_members,
                expected_node_ids=fresh_node_ids,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                heartbeat=heartbeat,
            )
            if e2e_resumption_source is None:
                with journal.stage("run complete E2E battery on fresh physical fleet"):
                    assert e2e_entrypoint.local_port is not None
                    report.e2e_battery = self._run_complete_e2e_battery(
                        api_base_url=f"http://127.0.0.1:{e2e_entrypoint.local_port}",
                        fleet=fleet,
                        expected_node_ids=fresh_node_ids,
                        expected_commit=expected_commit,
                        report=report,
                        journal=journal,
                        artifact_directory=artifact_directory,
                        heartbeat=heartbeat,
                    )
                    journal.persist()
            else:
                with journal.stage("resume failed complete E2E provenance gate"):
                    heartbeat.raise_if_failed()
                    battery, resumption = _seal_and_replay_e2e_resumption(
                        source=e2e_resumption_source,
                        repository_root=Path(__file__).resolve().parents[2],
                        fleet=fleet,
                        expected_commit=expected_commit,
                        artifact_directory=artifact_directory,
                    )
                    report.e2e_battery = battery
                    report.e2e_resumption = resumption
                    journal.persist()
        except QualificationInterruptedError as exception:
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"fresh-install fleet interrupted: {exception}",
                )
            )
        except Exception as exception:  # noqa: BLE001 - lifecycle failure boundary
            heartbeat_failed = isinstance(exception, LeaseHeartbeatError)
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"fresh-install fleet failed: {exception}",
                )
            )
        finally:
            signal_guard.begin_recovery()
            try:
                with journal.stage("stop and remove every temporary installation"):
                    _run_member_operations(
                        members,
                        self._cleanup_physical_fleet_member,
                    )
            except Exception as exception:  # noqa: BLE001 - recovery boundary
                cleanup_succeeded = False
                report.issues.append(
                    Issue(
                        severity="error",
                        message=f"critical temporary-fleet cleanup failure: {exception}",
                    )
                )
            if any(member.service_stopped for member in members):
                try:
                    service_restoration_succeeded = self._restore_physical_fleet(
                        members=members,
                        report=report,
                        journal=journal,
                        heartbeat=heartbeat,
                    )
                    restoration_succeeded = (
                        cleanup_succeeded and service_restoration_succeeded
                    )
                except Exception as exception:  # noqa: BLE001 - recovery boundary
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"critical whole-fleet restoration failure: {exception}",
                        )
                    )
                    restoration_succeeded = False
            elif acquired:
                restoration_succeeded = cleanup_succeeded
            report.restoration_succeeded = restoration_succeeded
            report.teardown_succeeded = restoration_succeeded
            try:
                heartbeat.raise_if_failed()
            except LeaseHeartbeatError as exception:
                heartbeat_failed = True
                report.issues.append(
                    Issue(
                        severity="error", message=f"lease heartbeat failed: {exception}"
                    )
                )
            if owns_lease:
                heartbeat.stop()
                try:
                    heartbeat.raise_if_failed()
                except LeaseHeartbeatError as exception:
                    if not heartbeat_failed:
                        report.issues.append(
                            Issue(
                                severity="error",
                                message=f"lease heartbeat failed: {exception}",
                            )
                        )
                    heartbeat_failed = True
            for member in members:
                if member.tunnel is not None:
                    _terminate_process(member.tunnel)
            if (
                owns_lease
                and acquired
                and restoration_succeeded
                and not heartbeat_failed
            ):
                try:
                    with journal.stage("release restored fleet lease"):
                        release = store.release()
                        if not release.ok:
                            raise RuntimeError(release.message)
                except Exception as exception:  # noqa: BLE001 - keep lease held
                    release_failed = True
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=f"fleet lease release failed: {exception}",
                        )
                    )
            if acquired and (
                not restoration_succeeded or heartbeat_failed or release_failed
            ):
                report.critical_recovery_required = True
                if owns_lease:
                    try:
                        heartbeat.emergency_extend(
                            ttl_s=self.fresh.emergency_lease_ttl_s
                        )
                    except LeaseHeartbeatError as exception:
                        report.issues.append(
                            Issue(
                                severity="error",
                                message=f"emergency lease extension failed: {exception}",
                            )
                        )
                    report.issues.append(
                        Issue(
                            severity="error",
                            message=(
                                "fleet lease intentionally remains held pending "
                                "operator recovery"
                            ),
                        )
                    )
            _record_deferred_interruption(report, signal_guard)
            report = report.finish(
                passed=(
                    not _blocking_issues(report)
                    and not _failed_lifecycle_stages(report)
                    and restoration_succeeded
                    and not release_failed
                    and not report.critical_recovery_required
                )
            )
            journal.report = report
            journal.persist()
        return report

    def _run_complete_e2e_battery(
        self,
        *,
        api_base_url: str,
        fleet: FreshInstallPhysicalFleet,
        expected_node_ids: frozenset[str],
        expected_commit: str | None,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat,
    ) -> FreshInstallE2EBatteryEvidence:
        """Run and verify every configured E2E cell before fresh-fleet teardown."""

        if expected_commit is None:
            raise ValueError("fresh-fleet E2E requires an exact expected commit")
        repository_root = Path(__file__).resolve().parents[2]
        script_path = _qualification_source_path(
            repository_root,
            fleet.e2e_battery_script,
            label="full E2E battery script",
        )
        model_sets_source = _qualification_source_path(
            repository_root,
            self.config.model_sets_path,
            label="model-set matrix",
        )
        test_sets_source = _qualification_source_path(
            repository_root,
            self.config.test_sets_path,
            label="test-set matrix",
        )

        e2e_root = artifact_directory / "complete-e2e"
        e2e_root.mkdir(parents=True, exist_ok=True)
        e2e_root.chmod(0o700)
        model_sets_snapshot = e2e_root / "model-sets.yaml"
        test_sets_snapshot = e2e_root / "test-sets.yaml"
        shutil.copyfile(model_sets_source, model_sets_snapshot)
        shutil.copyfile(test_sets_source, test_sets_snapshot)
        model_sets_snapshot.chmod(0o600)
        test_sets_snapshot.chmod(0o600)
        active_report_path = artifact_directory / "fresh-install-report.json"
        journal.persist()
        generated_config = self.config.model_copy(
            update={
                "api_base_url": api_base_url,
                "output_dir": e2e_root / "runs",
                "fresh_install_report_path": active_report_path,
                "model_sets_path": model_sets_snapshot,
                "test_sets_path": test_sets_snapshot,
            }
        )
        config_path = e2e_root / "qualification-config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                json.loads(generated_config.model_dump_json()),
                sort_keys=False,
            )
        )
        config_path.chmod(0o600)
        battery_log = e2e_root / "battery.log"
        controller_log = e2e_root / "controller.log"

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SKULK_")
        }
        environment.update(
            {
                "SKULK_HARNESS_CONFIG": str(config_path),
                "SKULK_E2E_BATTERY_LOG": str(battery_log),
                "SKULK_E2E_DELETE_STAGED_MODELS": "1",
                "SKULK_E2E_FAIL_FAST": "1",
                "SKULK_PUBLISH_RESULTS": "0",
            }
        )
        heartbeat.raise_if_failed()
        with (
            controller_log.open("wb") as log_handle,
            _FreshRuntimeMonitor(
                api_base_url=api_base_url,
                expected_node_ids=expected_node_ids,
                expected_node_count=len(expected_node_ids),
                poll_interval_s=self.fresh.poll_interval_s,
                request_timeout_s=self.config.request_timeout_s,
            ) as runtime_monitor,
        ):
            process = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=repository_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + fleet.e2e_battery_timeout_s
            try:
                while process.poll() is None:
                    heartbeat.raise_if_failed()
                    runtime_monitor.raise_if_failed()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "complete fresh-fleet E2E battery exceeded its timeout"
                        )
                    time.sleep(self.fresh.poll_interval_s)
            finally:
                if process.poll() is None:
                    _terminate_process_group(process)
        heartbeat.raise_if_failed()
        assert process.returncode is not None
        evidence = _summarize_fresh_e2e_battery(
            script_path=script_path,
            battery_log=battery_log,
            report_root=e2e_root / "runs",
            expected_commit=expected_commit,
            expected_node_ids=expected_node_ids,
            process_returncode=process.returncode,
        )
        if not evidence.passed:
            raise RuntimeError(
                "complete fresh-fleet E2E battery failed its result or provenance gate"
            )
        return evidence

    @staticmethod
    def _create_temporary_home(controller: SshTargetController) -> str:
        """Create an empty HOME on the target user's real home filesystem."""

        result = controller.run(
            "umask 077; "
            'root=$(mktemp -d "$HOME/.skulk-fresh.XXXXXX"); '
            'case "$root" in "$HOME"/.skulk-fresh.*) ;; *) exit 97 ;; esac; '
            'mkdir -p "$root/home" "$root/tmp"; '
            'printf "%s" "$root"',
            timeout_s=30,
        )
        temporary_root = result.stdout.strip()
        remote_path = PurePosixPath(temporary_root)
        if (
            "\n" in temporary_root
            or "\r" in temporary_root
            or not remote_path.is_absolute()
            or len(remote_path.parts) < 3
            or re.fullmatch(
                r"\.skulk-fresh\.[A-Za-z0-9_-]+",
                remote_path.name,
            )
            is None
        ):
            raise RuntimeError("target returned an unsafe temporary root")
        return temporary_root

    def _install_member(
        self,
        *,
        member: _PhysicalFleetMemberRuntime,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        installer_url: str,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat,
    ) -> tuple[str, str]:
        """Install one fleet member from the same official installer bytes."""

        assert member.temporary_root is not None
        command = _installer_command(
            installer_url=installer_url,
            profile=profile,
            expected_commit=expected_commit,
        )
        returncode = _run_remote_logged_command(
            controller=member.controller,
            command=_clean_environment_command(member.temporary_root, command),
            log_path=artifact_directory / "installer.log",
            timeout_s=14400,
            poll_interval_s=self.fresh.poll_interval_s,
            heartbeat=heartbeat,
        )
        if returncode != 0:
            raise RuntimeError(f"official installer exited {returncode}")
        checkout = shlex.quote(member.temporary_root + "/home/skulk")
        resolved_commit = member.controller.run(
            f"git -C {checkout} rev-parse HEAD",
            timeout_s=30,
        ).stdout.strip()
        if expected_commit and resolved_commit != expected_commit:
            raise RuntimeError("installer resolved a different candidate commit")
        config_path = member.temporary_root + "/home/skulk/skulk.yaml"
        config_digest = _remote_sha256(member.controller, config_path)
        if config_digest is None:
            raise RuntimeError("installer did not generate skulk.yaml")
        member.controller.copy_from(
            config_path,
            artifact_directory / "generated-skulk.yaml",
        )
        return resolved_commit, config_digest

    def _record_member_runtime_evidence(
        self,
        *,
        members: list[_PhysicalFleetMemberRuntime],
        expected_node_count: int,
        report: FreshInstallQualificationReport,
    ) -> None:
        """Assert and retain each member's own backend and dashboard contract."""

        local_node_ids: list[str] = []
        member_observed_node_ids: list[frozenset[str]] = []
        for member in members:
            assert member.local_port is not None
            api_base_url = f"http://127.0.0.1:{member.local_port}"
            observed_node_ids = _wait_for_exact_cluster(
                api_base_url,
                expected_node_count=expected_node_count,
                timeout_s=self.fresh.readiness_timeout_s,
                poll_interval_s=self.fresh.poll_interval_s,
            )
            node_id, node_count = _wait_for_api_identity(
                api_base_url,
                timeout_s=self.fresh.readiness_timeout_s,
                poll_interval_s=self.fresh.poll_interval_s,
            )
            if node_count != expected_node_count:
                raise RuntimeError(
                    f"member {member.ordinal} observed {node_count} nodes; "
                    f"expected {expected_node_count}"
                )
            local_node_ids.append(node_id)
            member_observed_node_ids.append(observed_node_ids)
            with SkulkClient(api_base_url) as client:
                state = client.get_state()
            resources = state.get("nodeResources")
            raw_resource = (
                resources.get(node_id) if isinstance(resources, dict) else None
            )
            resource = raw_resource if isinstance(raw_resource, dict) else {}
            raw_backends = resource.get("backends")
            detected_backends = sorted(
                backend
                for backend in (raw_backends if isinstance(raw_backends, list) else [])
                if isinstance(backend, str)
            )
            missing = sorted(
                set(member.target.expected_backends) - set(detected_backends)
            )
            if missing:
                raise RuntimeError(
                    f"member {member.ordinal} did not detect expected backends: "
                    f"{missing}"
                )
            transport = resource.get("dataTransport")
            if transport != member.target.expected_data_transport:
                raise RuntimeError(
                    f"member {member.ordinal} DATA transport mismatch: {transport!r}"
                )
            response = httpx.get(
                api_base_url,
                timeout=self.config.request_timeout_s,
            )
            dashboard_present = (
                response.status_code == 200
                and "<html" in response.text.lower()
                and 'id="root"' in response.text
            )
            expected_dashboard = member.target.dashboard_contract
            if expected_dashboard == "required" and not dashboard_present:
                raise RuntimeError(
                    f"member {member.ordinal} did not serve the dashboard build"
                )
            if expected_dashboard == "absent" and dashboard_present:
                raise RuntimeError(
                    f"member {member.ordinal} unexpectedly served a dashboard"
                )
            evidence = report.members[member.ordinal - 1]
            report.members[member.ordinal - 1] = evidence.model_copy(
                update={
                    "detected_backends": detected_backends,
                    "data_transport": transport,
                    "dashboard_build_present": dashboard_present,
                }
            )
        _assert_declared_member_topologies(
            expected_node_count=expected_node_count,
            local_node_ids=local_node_ids,
            member_observed_node_ids=member_observed_node_ids,
        )

    def _qualify_fleet_models(
        self,
        *,
        api_base_url: str,
        members: list[_PhysicalFleetMemberRuntime],
        contract_members: list[_PhysicalFleetMemberRuntime],
        expected_node_ids: frozenset[str],
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat,
    ) -> None:
        """Exercise each platform contract through the formed mixed fleet."""

        model_contracts: list[tuple[str, _PhysicalFleetMemberRuntime]] = []
        seen_models: set[str] = set()
        for contract_member in contract_members:
            models = [
                *contract_member.target.text_models,
                *contract_member.target.vision_models,
            ]
            for model_id in models:
                if model_id in seen_models:
                    continue
                seen_models.add(model_id)
                model_contracts.append((model_id, contract_member))
        if not model_contracts:
            raise ValueError("fresh-install physical fleet has no qualification models")
        primary_chat_model = model_contracts[0][0]
        audio_contracts = [
            member.target.dashboard_audio
            for member in contract_members
            if member.target.dashboard_audio is not None
        ]
        if len(audio_contracts) > 1 and any(
            contract != audio_contracts[0] for contract in audio_contracts[1:]
        ):
            raise ValueError(
                "fresh-install physical fleet declares conflicting dashboard audio contracts"
            )
        audio_contract = audio_contracts[0] if audio_contracts else None
        with (
            SkulkClient(
                api_base_url,
                request_timeout_s=self.config.request_timeout_s,
                generation_timeout_s=self.config.generation_timeout_s,
                stream_read_timeout_s=self.config.stream_read_timeout_s,
            ) as client,
            _FreshRuntimeMonitor(
                api_base_url=api_base_url,
                expected_node_ids=expected_node_ids,
                expected_node_count=len(members),
                poll_interval_s=self.fresh.poll_interval_s,
                request_timeout_s=self.config.request_timeout_s,
            ) as runtime_monitor,
        ):

            def check_fresh_runtime() -> None:
                """Fail immediately if lease health or fleet membership changes."""

                _check_heartbeat(heartbeat)
                runtime_monitor.raise_if_failed()
                assert_fresh_cluster(
                    client,
                    expected_node_count=len(members),
                    expected_node_ids=expected_node_ids,
                )
                runtime_monitor.raise_if_failed()

            dashboard = DashboardQualifier(
                api_base_url=api_base_url,
                artifact_directory=artifact_directory / "playwright",
                poll_interval_s=self.fresh.poll_interval_s,
                model_ready_timeout_s=self.fresh.model_ready_timeout_s,
                abort_check=check_fresh_runtime,
            )
            thinking_toggles = client.resolved_thinking_toggle_by_model()
            card_image_input = client.resolved_image_input_by_model()
            for model_id, contract_member in model_contracts:
                target = contract_member.target
                check_fresh_runtime()
                enable_thinking = (
                    False if thinking_toggles.get(model_id, False) else None
                )
                with journal.stage(f"dashboard fleet journey: {model_id}"):
                    expectation = _browser_vision_expectation(
                        model_id,
                        vision_models=target.vision_models,
                        card_image_input=card_image_input.get(model_id),
                    )
                    browser_fixture = (
                        generate_vision_fixture() if expectation == "positive" else None
                    )
                    outcome = dashboard.qualify(
                        model_id=model_id,
                        vision_contract=expectation,
                        fixture=browser_fixture,
                    )
                    report.browser.append(outcome)
                    journal.persist()
                    if not outcome.passed:
                        raise RuntimeError(
                            outcome.message
                            or f"dashboard journey failed for {model_id}"
                        )
                    check_fresh_runtime()

                if target.served_engine_contract is not None:
                    with journal.stage(f"served engine fleet contract: {model_id}"):
                        evidence = _qualify_served_engine_fleet(
                            members=[
                                member
                                for member in members
                                if member.target.platform == target.platform
                            ],
                            api_base_url=api_base_url,
                            model_id=model_id,
                            contract=target.served_engine_contract,
                            request_timeout_s=self.config.request_timeout_s,
                            stream_read_timeout_s=self.config.stream_read_timeout_s,
                        )
                        report.served_engines.append(evidence)
                        journal.persist()
                        if not evidence.passed:
                            raise RuntimeError(
                                "fresh served-engine fleet contract failed for "
                                f"{model_id}: expected --parallel "
                                f"{evidence.expected_parallel}, observed "
                                f"{evidence.observed_parallel}; expected "
                                f"kv-unified={evidence.kv_unified_required}, "
                                f"observed={evidence.kv_unified_observed}; "
                                "maximum live concurrency "
                                f"{evidence.maximum_observed_active}"
                            )
                        check_fresh_runtime()

                with journal.stage(f"direct fleet API parity: {model_id}"):
                    text_outcome = qualify_direct_text(
                        client,
                        model_id=model_id,
                        enable_thinking=enable_thinking,
                    )
                    if not text_outcome.passed:
                        raise RuntimeError(
                            f"direct text API parity failed for {model_id}; "
                            f"the model replied: {text_outcome.response!r}"
                        )
                    if model_id in target.vision_models:
                        api_fixture = generate_vision_fixture()
                        api_fixture.write(
                            artifact_directory
                            / "api-fixtures"
                            / f"{_safe_model_name(model_id)}.png"
                        )
                        vision_outcome = qualify_direct_vision(
                            client,
                            model_id=model_id,
                            fixture=api_fixture,
                            enable_thinking=enable_thinking,
                        )
                        report.api_vision.append(vision_outcome)
                        journal.persist()
                        if not vision_outcome.passed:
                            raise RuntimeError(
                                f"direct vision API parity failed for {model_id}; "
                                "matches: "
                                f"code={vision_outcome.response_matched_code}, "
                                f"color={vision_outcome.response_matched_color}, "
                                f"shape={vision_outcome.response_matched_shape}, "
                                f"format={vision_outcome.response_matched_format}; "
                                "redacted reply: "
                                f"{vision_outcome.response_excerpt!r}"
                            )
                    check_fresh_runtime()

                if model_id == primary_chat_model:
                    continue
                with journal.stage(f"stop temporary fleet placement: {model_id}"):
                    check_fresh_runtime()
                    for placement in client.find_placements_for_model(model_id):
                        if placement.instance_id:
                            client.delete_instance(placement.instance_id)
                    _wait_for_no_placement(
                        client,
                        model_id=model_id,
                        timeout_s=180,
                        poll_interval_s=self.fresh.poll_interval_s,
                        heartbeat=heartbeat,
                        runtime_check=check_fresh_runtime,
                    )
                    check_fresh_runtime()

            with journal.stage("dashboard release experience"):
                experience = dashboard.qualify_experience(
                    model_id=primary_chat_model,
                    expected_node_count=len(members),
                )
                report.dashboard_experience = experience
                journal.persist()
                if not experience.passed:
                    raise RuntimeError(
                        experience.message or "dashboard release experience failed"
                    )
                check_fresh_runtime()

            temporary_models = [primary_chat_model]
            if audio_contract is not None:
                with journal.stage("dashboard audio experience"):
                    audio_evidence = dashboard.qualify_audio(
                        chat_model_id=primary_chat_model,
                        speech_synthesis_model=(audio_contract.speech_synthesis_model),
                        transcription_model=audio_contract.transcription_model,
                    )
                    report.dashboard_audio = audio_evidence
                    journal.persist()
                    if not audio_evidence.passed:
                        raise RuntimeError(
                            audio_evidence.message
                            or "dashboard audio experience failed"
                        )
                    check_fresh_runtime()
                temporary_models.extend(
                    [
                        audio_contract.speech_synthesis_model,
                        audio_contract.transcription_model,
                    ]
                )

            for model_id in dict.fromkeys(temporary_models):
                with journal.stage(f"stop temporary fleet placement: {model_id}"):
                    check_fresh_runtime()
                    for placement in client.find_placements_for_model(model_id):
                        if placement.instance_id:
                            client.delete_instance(placement.instance_id)
                    _wait_for_no_placement(
                        client,
                        model_id=model_id,
                        timeout_s=180,
                        poll_interval_s=self.fresh.poll_interval_s,
                        heartbeat=heartbeat,
                        runtime_check=check_fresh_runtime,
                    )
                    check_fresh_runtime()

    def _cleanup_physical_fleet_member(
        self,
        member: _PhysicalFleetMemberRuntime,
    ) -> None:
        """Stop one temporary runtime and remove only its private HOME."""

        if member.skulk_process is not None:
            _terminate_process(member.skulk_process)
            member.skulk_process = None
        if member.skulk_log_handle is not None:
            member.skulk_log_handle.close()
            member.skulk_log_handle = None
        if member.temporary_root is None:
            return
        process_pattern = _self_safe_process_pattern(
            member.temporary_root + "/home/skulk"
        )
        member.controller.run(
            f"pkill -TERM -f {shlex.quote(process_pattern)} 2>/dev/null || true",
            check=False,
            timeout_s=30,
        )
        member.controller.run(
            f"rm -rf -- {shlex.quote(member.temporary_root)}",
            timeout_s=300,
        )
        cleanup_deadline = time.monotonic() + 60
        cleanup_probe = (
            f"test ! -e {shlex.quote(member.temporary_root)} && "
            f"! pgrep -f {shlex.quote(process_pattern)} >/dev/null"
        )
        last_error: Exception | None = None
        while time.monotonic() < cleanup_deadline:
            try:
                member.controller.run(cleanup_probe, timeout_s=30)
                break
            except Exception as exception:  # noqa: BLE001 - cleanup boundary
                last_error = exception
                time.sleep(2)
        else:
            assert last_error is not None
            raise last_error
        member.temporary_root = None

    def _restore_physical_fleet(
        self,
        *,
        members: list[_PhysicalFleetMemberRuntime],
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        heartbeat: AuthoritativeLeaseHeartbeat,
    ) -> bool:
        """Restore and verify every original service and the complete topology."""

        with journal.stage("restore original service on every member"):

            def start_original(member: _PhysicalFleetMemberRuntime) -> None:
                if not member.service_stopped:
                    return
                if member.snapshot is None:
                    raise RuntimeError("recovery snapshot is missing")
                assert member.target.service_start_command is not None
                _prepare_original_service_start(
                    controller=member.controller,
                    target=member.target,
                )
                member.controller.run(
                    member.target.service_start_command,
                    timeout_s=120,
                )

            failures: list[str] = []
            restored_local_node_ids: list[str] = []
            restored_member_node_ids: list[frozenset[str]] = []
            restored_observations: list[
                tuple[_PhysicalFleetMemberRuntime, str, int, frozenset[str]]
            ] = []
            recovery_start_succeeded = False
            try:
                _run_member_operations(members, start_original)
                restored_observations = _wait_for_restored_declared_topology(
                    members,
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                    heartbeat=heartbeat,
                )
                recovery_start_succeeded = True
            finally:

                def stop_failed_start(member: _PhysicalFleetMemberRuntime) -> None:
                    if not member.service_stopped:
                        return
                    assert member.target.service_stop_command is not None
                    member.controller.run(
                        member.target.service_stop_command,
                        check=False,
                        timeout_s=120,
                    )

                def restore_configs(member: _PhysicalFleetMemberRuntime) -> None:
                    assert member.snapshot is not None
                    member.controller.restore_original_config_files(member.snapshot)

                try:
                    if not recovery_start_succeeded:
                        _run_member_operations(members, stop_failed_start)
                finally:
                    _run_member_operations(
                        [member for member in members if member.snapshot is not None],
                        restore_configs,
                    )
            for member, node_id, node_count, observed_node_ids in restored_observations:
                assert member.snapshot is not None
                mismatches = member.controller.verify_restored_state(
                    member.snapshot.original,
                    api_node_id=node_id,
                    cluster_node_count=node_count,
                )
                if mismatches:
                    failures.append(f"member {member.ordinal}: {'; '.join(mismatches)}")
                    continue
                restored_local_node_ids.append(node_id)
                restored_member_node_ids.append(observed_node_ids)
            if not failures:
                _assert_declared_member_topologies(
                    expected_node_count=len(members),
                    local_node_ids=restored_local_node_ids,
                    member_observed_node_ids=restored_member_node_ids,
                )
            if failures:
                raise RuntimeError("; ".join(failures))
            for member in members:
                evidence = report.members[member.ordinal - 1]
                report.members[member.ordinal - 1] = evidence.model_copy(
                    update={"restored": True}
                )
                member.service_stopped = False
            journal.persist()
        return True

    def _qualify_runpod(
        self,
        *,
        target: FreshInstallTarget,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        signal_guard: QualificationSignalGuard,
        heartbeat: AuthoritativeLeaseHeartbeat | None,
    ) -> FreshInstallQualificationReport:
        if self.fresh.runpod is None:
            raise ValueError("runpod target selected without fresh_install.runpod")
        pod_id: str | None = None
        teardown_succeeded = False
        deadline_timer: threading.Timer | None = None
        deadline_fired = threading.Event()
        deadline_cancelled = threading.Event()
        deadline_errors: list[Exception] = []
        teardown_lock = threading.Lock()
        runpod = RunPodClient(self.fresh.runpod)
        try:
            _check_heartbeat(heartbeat)
            with journal.stage("provision clean cost-bounded RunPod"):
                pod = runpod.provision(qualification_id=report.qualification_id)
                pod_id = pod.pod_id
                deadline_timer = threading.Timer(
                    self.fresh.runpod.maximum_runtime_s,
                    _runpod_deadline_teardown,
                    kwargs={
                        "client": runpod,
                        "pod_id": pod_id,
                        "fired": deadline_fired,
                        "cancelled": deadline_cancelled,
                        "errors": deadline_errors,
                        "teardown_lock": teardown_lock,
                    },
                )
                deadline_timer.daemon = True
                deadline_timer.start()
                endpoint = runpod.wait_for_ssh(pod_id)
                _check_heartbeat(heartbeat)
            ephemeral_target = _runpod_ephemeral_target(
                target=target,
                runpod_config=self.fresh.runpod,
                endpoint=endpoint,
            )
            controller = SshTargetController(ephemeral_target)
            local_port, tunnel = controller.open_tunnel(
                remote_port=self.fresh.remote_port
            )
            try:
                self._execute_clean_install(
                    controller=controller,
                    api_base_url=f"http://127.0.0.1:{local_port}",
                    target=ephemeral_target,
                    profile=profile,
                    expected_commit=expected_commit,
                    report=report,
                    journal=journal,
                    artifact_directory=artifact_directory,
                    heartbeat=heartbeat,
                    signal_guard=signal_guard,
                )
            finally:
                _terminate_process(tunnel)
        except QualificationInterruptedError as exception:
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"RunPod qualification interrupted: {exception}",
                )
            )
        except Exception as exception:  # noqa: BLE001 - provider lifecycle boundary
            report.issues.append(
                Issue(
                    severity="error",
                    message=f"RunPod qualification failed: {exception}",
                )
            )
        finally:
            signal_guard.begin_recovery()
            deadline_cancelled.set()
            if deadline_timer is not None:
                deadline_timer.cancel()
            try:
                # Serialize the normal finally path with a cost-deadline teardown
                # that may already be in flight. The HTTP client must not close
                # while its timer thread is still polling provider state.
                with teardown_lock:
                    if pod_id is not None:
                        with journal.stage("terminate RunPod and confirm deletion"):
                            runpod.terminate_and_confirm(pod_id)
                teardown_succeeded = True
            except Exception as exception:  # noqa: BLE001 - teardown must be reported
                report.issues.append(
                    Issue(
                        severity="error",
                        message=f"critical RunPod teardown failure: {exception}",
                    )
                )
                report.critical_recovery_required = True
            if deadline_timer is not None:
                # The timer shares the provider client. Wait for its bounded
                # teardown path to exit before closing that client.
                deadline_timer.join()
            if deadline_fired.is_set():
                report.issues.append(
                    Issue(
                        severity="error",
                        message="RunPod qualification exceeded its maximum runtime",
                    )
                )
            for deadline_error in deadline_errors:
                report.issues.append(
                    Issue(
                        severity="error",
                        message=f"RunPod deadline teardown failed: {deadline_error}",
                    )
                )
            runpod.close()
            report.restoration_succeeded = None
            report.teardown_succeeded = teardown_succeeded
            _record_deferred_interruption(report, signal_guard)
            report = report.finish(
                passed=(
                    not _blocking_issues(report)
                    and not _failed_lifecycle_stages(report)
                    and teardown_succeeded
                )
            )
            journal.report = report
            journal.persist()
        return report

    def _execute_clean_install(
        self,
        *,
        controller: SshTargetController,
        api_base_url: str,
        target: FreshInstallTarget,
        profile: FreshInstallProfile,
        expected_commit: str | None,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat | None,
        signal_guard: QualificationSignalGuard,
    ) -> None:
        temporary_root: str | None = None
        skulk_process: subprocess.Popen[bytes] | None = None
        skulk_log_handle: BinaryIO | None = None
        try:
            with journal.stage("create empty temporary HOME"):
                temporary_root = self._create_temporary_home(controller)

            with journal.stage("run official installer"):
                installer_url, installer_digest = _installer_provenance(
                    self.fresh.installer_url,
                    profile=profile,
                    expected_commit=expected_commit,
                    shipping_ref=self.fresh.shipping_installer_ref,
                )
                command = _installer_command(
                    installer_url=installer_url,
                    profile=profile,
                    expected_commit=expected_commit,
                )
                installer_log = artifact_directory / "installer.log"
                installer_returncode = _run_remote_logged_command(
                    controller=controller,
                    command=_clean_environment_command(temporary_root, command),
                    log_path=installer_log,
                    timeout_s=14400,
                    poll_interval_s=self.fresh.poll_interval_s,
                    heartbeat=heartbeat,
                )
                if installer_returncode != 0:
                    raise RuntimeError(
                        f"official installer exited {installer_returncode}"
                    )
                resolved_commit = controller.run(
                    "git -C "
                    f"{shlex.quote(temporary_root + '/home/skulk')} "
                    "rev-parse HEAD",
                    timeout_s=30,
                ).stdout.strip()
                if expected_commit and resolved_commit != expected_commit:
                    raise RuntimeError(
                        "installer resolved a different candidate commit"
                    )
                config_path = temporary_root + "/home/skulk/skulk.yaml"
                generated_config_digest = _remote_sha256(controller, config_path)
                if generated_config_digest is None:
                    raise RuntimeError("installer did not generate skulk.yaml")
                controller.copy_from(
                    config_path,
                    artifact_directory / "generated-skulk.yaml",
                )
                report.install = report.install.model_copy(
                    update={
                        "installer_url": installer_url,
                        "installer_sha256": installer_digest,
                        "requested_ref": (
                            expected_commit if profile == "candidate" else "main"
                        ),
                        "resolved_commit": resolved_commit,
                        "generated_config_sha256": generated_config_digest,
                    }
                )
                journal.persist()
                _check_heartbeat(heartbeat)

            with journal.stage("start installer-printed Skulk command"):
                start_command = _runtime_start_command(
                    temporary_root=temporary_root,
                    target=target,
                )
                skulk_process, skulk_log_handle = controller.start(
                    start_command,
                    log_path=artifact_directory / "skulk.log",
                )
                _wait_for_http(
                    api_base_url + "/state",
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                    heartbeat=heartbeat,
                )

            with journal.stage("assert fresh runtime contract"):
                provenance = _wait_for_runtime_contract(
                    api_base_url,
                    target=target,
                    expected_commit=expected_commit,
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                    stability_s=self.fresh.runtime_contract_stability_s,
                    heartbeat=heartbeat,
                )
                report.install = report.install.model_copy(
                    update={
                        **provenance.model_dump(),
                        "profile": profile,
                        "platform": target.platform,
                        "hardware_class": target.hardware_class,
                        "installer_url": report.install.installer_url,
                        "installer_sha256": report.install.installer_sha256,
                        "requested_ref": report.install.requested_ref,
                        "expected_commit": expected_commit,
                        "generated_config_sha256": (
                            report.install.generated_config_sha256
                        ),
                    }
                )
                journal.persist()
                _check_heartbeat(heartbeat)

            self._qualify_models(
                api_base_url=api_base_url,
                controller=controller,
                installation_root=temporary_root,
                target=target,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                heartbeat=heartbeat,
            )
        finally:
            # Cleanup of the temporary process and HOME is already recovery:
            # allowing a signal to raise here can skip the remote kill/removal
            # and then restore the original service beside an orphan runtime.
            signal_guard.begin_recovery()
            if skulk_process is not None:
                _terminate_process(skulk_process)
            if skulk_log_handle is not None:
                skulk_log_handle.close()
            if temporary_root is not None:
                process_pattern = _self_safe_process_pattern(
                    temporary_root + "/home/skulk"
                )
                controller.run(
                    f"pkill -TERM -f {shlex.quote(process_pattern)} "
                    "2>/dev/null || true",
                    check=False,
                    timeout_s=30,
                )
                controller.run(
                    f"rm -rf -- {shlex.quote(temporary_root)}",
                    check=False,
                    timeout_s=300,
                )

    def _qualify_models(
        self,
        *,
        api_base_url: str,
        controller: SshTargetController,
        installation_root: str,
        target: FreshInstallTarget,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat | None,
    ) -> None:
        models = list(dict.fromkeys([*target.text_models, *target.vision_models]))
        if not models:
            raise ValueError("fresh-install target has no qualification models")
        primary_chat_model = models[0]
        with (
            SkulkClient(
                api_base_url,
                request_timeout_s=self.config.request_timeout_s,
                generation_timeout_s=self.config.generation_timeout_s,
                stream_read_timeout_s=self.config.stream_read_timeout_s,
            ) as client,
            _FreshRuntimeMonitor(
                api_base_url=api_base_url,
                expected_node_id=assert_fresh_single_node(client),
                poll_interval_s=self.fresh.poll_interval_s,
                request_timeout_s=self.config.request_timeout_s,
            ) as runtime_monitor,
        ):
            expected_node_id = runtime_monitor.expected_node_id

            def check_fresh_runtime() -> None:
                """Fail immediately if lease health or isolation changes."""

                _check_heartbeat(heartbeat)
                runtime_monitor.raise_if_failed()
                assert_fresh_single_node(
                    client,
                    expected_node_id=expected_node_id,
                )
                runtime_monitor.raise_if_failed()

            dashboard = DashboardQualifier(
                api_base_url=api_base_url,
                artifact_directory=artifact_directory / "playwright",
                poll_interval_s=self.fresh.poll_interval_s,
                model_ready_timeout_s=self.fresh.model_ready_timeout_s,
                abort_check=check_fresh_runtime,
            )
            thinking_toggles = client.resolved_thinking_toggle_by_model()
            card_image_input = client.resolved_image_input_by_model()
            for model_id in models:
                check_fresh_runtime()
                enable_thinking = (
                    False if thinking_toggles.get(model_id, False) else None
                )
                # The browser journey is what provisions the model on a target
                # that serves the UI: it finds, downloads, launches, and then
                # chats. A headless node has no UI to drive, so it provisions
                # the same model through the API the dashboard itself calls.
                # Skipping provisioning entirely would leave the parity check
                # below asking a node to serve a model it never mounted.
                if target.dashboard_contract == "required":
                    with journal.stage(f"dashboard user journey: {model_id}"):
                        expectation = _browser_vision_expectation(
                            model_id,
                            vision_models=target.vision_models,
                            card_image_input=card_image_input.get(model_id),
                        )
                        browser_fixture = (
                            generate_vision_fixture()
                            if expectation == "positive"
                            else None
                        )
                        outcome = dashboard.qualify(
                            model_id=model_id,
                            vision_contract=expectation,
                            fixture=browser_fixture,
                        )
                        report.browser.append(outcome)
                        journal.persist()
                        if not outcome.passed:
                            raise RuntimeError(
                                outcome.message
                                or f"dashboard journey failed for {model_id}"
                            )
                        check_fresh_runtime()
                else:
                    with journal.stage(f"headless model provisioning: {model_id}"):
                        _provision_model_over_api(
                            client,
                            model_id=model_id,
                            model_ready_timeout_s=self.fresh.model_ready_timeout_s,
                            poll_interval_s=self.fresh.poll_interval_s,
                            heartbeat=heartbeat,
                            runtime_check=check_fresh_runtime,
                        )
                        check_fresh_runtime()

                if target.served_engine_contract is not None:
                    with journal.stage(f"served engine runtime contract: {model_id}"):
                        evidence = _qualify_served_engine(
                            controller=controller,
                            installation_root=installation_root,
                            api_base_url=api_base_url,
                            model_id=model_id,
                            contract=target.served_engine_contract,
                            request_timeout_s=self.config.request_timeout_s,
                            stream_read_timeout_s=self.config.stream_read_timeout_s,
                        )
                        report.served_engines.append(evidence)
                        journal.persist()
                        if not evidence.passed:
                            raise RuntimeError(
                                "fresh served-engine contract failed for "
                                f"{model_id}: expected --parallel "
                                f"{evidence.expected_parallel}, observed "
                                f"{evidence.observed_parallel}; expected "
                                f"kv-unified={evidence.kv_unified_required}, "
                                f"observed={evidence.kv_unified_observed}; "
                                "maximum live concurrency "
                                f"{evidence.maximum_observed_active}"
                            )
                        check_fresh_runtime()

                with journal.stage(f"direct API parity: {model_id}"):
                    check_fresh_runtime()
                    text_outcome = qualify_direct_text(
                        client,
                        model_id=model_id,
                        enable_thinking=enable_thinking,
                    )
                    if not text_outcome.passed:
                        raise RuntimeError(
                            f"direct text API parity failed for {model_id}; "
                            f"the model replied: {text_outcome.response!r}"
                        )
                    if model_id in target.vision_models:
                        api_fixture = generate_vision_fixture()
                        api_fixture.write(
                            artifact_directory
                            / "api-fixtures"
                            / f"{_safe_model_name(model_id)}.png"
                        )
                        evidence = qualify_direct_vision(
                            client,
                            model_id=model_id,
                            fixture=api_fixture,
                            enable_thinking=enable_thinking,
                        )
                        report.api_vision.append(evidence)
                        journal.persist()
                        if not evidence.passed:
                            raise RuntimeError(
                                f"direct vision API parity failed for {model_id}; "
                                "matches: "
                                f"code={evidence.response_matched_code}, "
                                f"color={evidence.response_matched_color}, "
                                f"shape={evidence.response_matched_shape}, "
                                f"format={evidence.response_matched_format}; "
                                f"redacted reply: {evidence.response_excerpt!r}"
                            )
                    check_fresh_runtime()

                if (
                    model_id == primary_chat_model
                    and target.dashboard_contract == "required"
                ):
                    continue
                with journal.stage(f"stop temporary model placement: {model_id}"):
                    check_fresh_runtime()
                    for placement in client.find_placements_for_model(model_id):
                        if placement.instance_id:
                            client.delete_instance(placement.instance_id)
                    _wait_for_no_placement(
                        client,
                        model_id=model_id,
                        timeout_s=180,
                        poll_interval_s=self.fresh.poll_interval_s,
                        heartbeat=heartbeat,
                        runtime_check=check_fresh_runtime,
                    )
                    check_fresh_runtime()

            if target.dashboard_contract == "required":
                with journal.stage("dashboard release experience"):
                    experience = dashboard.qualify_experience(
                        model_id=primary_chat_model,
                        expected_node_count=1,
                    )
                    report.dashboard_experience = experience
                    journal.persist()
                    if not experience.passed:
                        raise RuntimeError(
                            experience.message or "dashboard release experience failed"
                        )
                    check_fresh_runtime()

                temporary_models = [primary_chat_model]
                if target.dashboard_audio is not None:
                    with journal.stage("dashboard audio experience"):
                        audio_evidence = dashboard.qualify_audio(
                            chat_model_id=primary_chat_model,
                            speech_synthesis_model=(
                                target.dashboard_audio.speech_synthesis_model
                            ),
                            transcription_model=(
                                target.dashboard_audio.transcription_model
                            ),
                        )
                        report.dashboard_audio = audio_evidence
                        journal.persist()
                        if not audio_evidence.passed:
                            raise RuntimeError(
                                audio_evidence.message
                                or "dashboard audio experience failed"
                            )
                        check_fresh_runtime()
                    temporary_models.extend(
                        [
                            target.dashboard_audio.speech_synthesis_model,
                            target.dashboard_audio.transcription_model,
                        ]
                    )

                for model_id in dict.fromkeys(temporary_models):
                    with journal.stage(f"stop temporary model placement: {model_id}"):
                        check_fresh_runtime()
                        for placement in client.find_placements_for_model(model_id):
                            if placement.instance_id:
                                client.delete_instance(placement.instance_id)
                        _wait_for_no_placement(
                            client,
                            model_id=model_id,
                            timeout_s=180,
                            poll_interval_s=self.fresh.poll_interval_s,
                            heartbeat=heartbeat,
                            runtime_check=check_fresh_runtime,
                        )
                        check_fresh_runtime()

    def _restore_physical(
        self,
        *,
        controller: SshTargetController,
        target: FreshInstallTarget,
        snapshot: RecoverySnapshot,
        api_base_url: str,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
    ) -> bool:
        with journal.stage("restore original selected-target service") as stage:
            original = snapshot.original
            assert target.service_start_command is not None
            recovery_start_succeeded = False
            try:
                _prepare_original_service_start(
                    controller=controller,
                    target=target,
                )
                controller.run(target.service_start_command, timeout_s=120)
                # Readiness is the target answering again, which is exactly what
                # restarting its service controls. It deliberately does not wait for
                # the fleet to return to its pre-run size: that size counts nodes
                # this leg never touched, so a peer that was rebooting, pruning, or
                # deliberately quiet made the wait unsatisfiable and timed out on a
                # machine that had already recovered.
                node_id, node_count = _wait_for_api_identity(
                    api_base_url,
                    timeout_s=self.fresh.readiness_timeout_s,
                    poll_interval_s=self.fresh.poll_interval_s,
                )
                recovery_start_succeeded = True
            finally:
                try:
                    if not recovery_start_succeeded:
                        assert target.service_stop_command is not None
                        controller.run(
                            target.service_stop_command,
                            check=False,
                            timeout_s=120,
                        )
                finally:
                    controller.restore_original_config_files(snapshot)
            mismatches = controller.verify_restored_state(
                original,
                api_node_id=node_id,
                cluster_node_count=node_count,
            )
            if mismatches:
                raise RuntimeError("; ".join(mismatches))
            # A restarted node always carries a new identity because Skulk never
            # persists node_id, so both are recorded rather than compared. An
            # operator reading the report can still see exactly which identity
            # left and which one rejoined.
            stage.message = (
                f"restored: node identity {original.api_node_id} -> {node_id}, "
                f"fleet {node_count} nodes"
            )
            if (
                original.cluster_node_count is not None
                and node_count < original.cluster_node_count
            ):
                # Recorded, not fatal. This leg stops and starts exactly one
                # node, so a smaller fleet is a statement about peers outside
                # the experiment. Failing here would hold the fleet lease and
                # declare a fully recovered machine broken because someone
                # else's node was down.
                report.issues.append(
                    Issue(
                        severity="warning",
                        message=(
                            "restored fleet is smaller than before the run: "
                            f"{original.cluster_node_count} -> {node_count} "
                            "nodes; the target itself restored cleanly"
                        ),
                    )
                )
        return True


def _runpod_ephemeral_target(
    *,
    target: FreshInstallTarget,
    runpod_config: RunPodFreshInstallConfig,
    endpoint: RunPodSshEndpoint,
) -> FreshInstallTarget:
    """Convert one declared RunPod contract into its ephemeral SSH target."""

    return FreshInstallTarget(
        kind="physical",
        platform="nvidia",
        hardware_class=target.hardware_class,
        eligible=True,
        ssh_host=endpoint.host,
        ssh_user="root",
        ssh_port=endpoint.port,
        ssh_identity_file=runpod_config.ssh_private_key_file,
        # The pod generates its host key at boot, so there is nothing to have
        # pinned in advance. Inventory hardware never gets this exception.
        accept_unknown_host_key=True,
        service_manager="command",
        service_stop_command="true",
        service_start_command="true",
        isolation_enter_command="true",
        isolation_exit_command="true",
        expected_backends=target.expected_backends,
        expected_data_transport=target.expected_data_transport,
        served_engine_contract=target.served_engine_contract,
        vision_contract=target.vision_contract,
        dashboard_contract=target.dashboard_contract,
        text_models=target.text_models,
        vision_models=target.vision_models,
    )


def _run_member_operations(
    members: Sequence[_PhysicalFleetMemberRuntime],
    operation: Callable[[_PhysicalFleetMemberRuntime], ResultT],
) -> list[ResultT]:
    """Run every member operation sequentially and report all ordinary failures.

    Sequential execution is deliberate. A signal raised in the main thread can
    immediately enter mandatory recovery; a thread-pool context would wait for
    every in-flight remote installer before unwinding.
    """

    results: list[ResultT] = []
    failures: list[str] = []
    for member in members:
        try:
            results.append(operation(member))
        except LeaseHeartbeatError:
            raise
        except Exception as exception:  # noqa: BLE001 - aggregate member failures
            failures.append(f"member {member.ordinal}: {exception}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return results


def _prepare_original_service_start(
    *,
    controller: SshTargetController,
    target: FreshInstallTarget,
) -> None:
    """Prevent a supervised recovery start from mutating the saved checkout.

    The shipped service wrapper may auto-update before launching Skulk. That is
    desirable on an ordinary boot but violates qualification's byte-for-byte
    recovery contract. Suppress it only for the recovery start; the caller must
    reapply the archived configuration after API readiness, which also repairs
    any runtime config normalization or cluster sync.
    """

    for config_path in target.original_config_paths:
        if PurePosixPath(config_path).name != "skulk.env":
            continue
        controller.run(
            _suppress_auto_update_command(config_path),
            timeout_s=30,
        )


def _suppress_auto_update_command(config_path: str) -> str:
    """Build an atomic, portable command for one transient recovery override."""

    quoted_path = shlex.quote(config_path)
    awk_program = (
        "BEGIN { found=0 } "
        "/^[[:space:]]*(export[[:space:]]+)?SKULK_AUTO_UPDATE=/ { "
        'if (!found) print "SKULK_AUTO_UPDATE=0"; found=1; next } '
        '{ print } END { if (!found) print "SKULK_AUTO_UPDATE=0" }'
    )
    return (
        f"target={quoted_path}; "
        'if [ -f "$target" ]; then '
        'if mode=$(stat -f %Lp "$target" 2>/dev/null); then :; '
        'else mode=$(stat -c %a "$target"); fi; '
        'tmp=$(mktemp "${target}.restore-start.XXXXXX"); '
        "trap 'rm -f \"$tmp\"' EXIT; "
        f'awk {shlex.quote(awk_program)} "$target" > "$tmp"; '
        'chmod "$mode" "$tmp"; mv "$tmp" "$target"; trap - EXIT; '
        "fi"
    )


def _assert_declared_member_topologies(
    *,
    expected_node_count: int,
    local_node_ids: Iterable[str],
    member_observed_node_ids: Iterable[frozenset[str]],
) -> frozenset[str]:
    """Require every declared member to observe exactly the declared fleet."""

    declared_node_ids = frozenset(local_node_ids)
    observed_topologies = tuple(member_observed_node_ids)
    mismatched_members = [
        ordinal
        for ordinal, observed_node_ids in enumerate(observed_topologies, start=1)
        if observed_node_ids != declared_node_ids
    ]
    if (
        len(declared_node_ids) != expected_node_count
        or len(observed_topologies) != expected_node_count
        or mismatched_members
    ):
        raise RuntimeError(
            "declared physical fleet does not match one complete live topology: "
            f"expected {expected_node_count} members, observed "
            f"{len(declared_node_ids)} unique local identities, mismatched "
            f"member views {mismatched_members}"
        )
    return declared_node_ids


def _aggregate_digests(digests: Iterable[str]) -> str:
    """Return an order-stable digest over anonymous member digests."""

    normalized = sorted(str(digest) for digest in digests)
    payload = "\n".join(normalized).encode()
    return hashlib.sha256(payload).hexdigest()


def _release_leg_evidence(
    report: FreshInstallQualificationReport,
) -> ReleaseQualificationLegEvidence:
    """Reduce one private leg report to the atomic release-gate evidence."""

    covered_platforms: list[FreshInstallPlatform] = sorted(
        {member.platform for member in report.members}
        or ({report.platform} if report.platform != "mixed" else set())
    )
    return ReleaseQualificationLegEvidence(
        qualification_id=report.qualification_id,
        platform=report.platform,
        covered_platforms=covered_platforms,
        hardware_class=report.hardware_class,
        passed=report.passed,
        critical_recovery_required=report.critical_recovery_required,
        complete_e2e_passed=(
            report.e2e_battery.passed if report.e2e_battery is not None else None
        ),
        e2e_resumption=report.e2e_resumption,
    )


def _assert_restored_fleet_clean(
    api_base_url: str,
    *,
    expected_node_count: int,
) -> None:
    """Require the restored physical fleet to be complete, idle, and drift-free."""

    with SkulkClient(api_base_url) as client:
        state = client.get_state()
        node_identities = state.get("nodeIdentities")
        if (
            not isinstance(node_identities, dict)
            or len(node_identities) != expected_node_count
        ):
            observed = len(node_identities) if isinstance(node_identities, dict) else 0
            raise RuntimeError(
                "restored fleet node count mismatch: "
                f"expected {expected_node_count}, observed {observed}"
            )
        for field in ("instances", "runners"):
            value = state.get(field)
            if isinstance(value, dict) and value:
                raise RuntimeError(f"restored fleet retained active {field}")
        drift = client.detect_runner_state_drift()
        if drift:
            raise RuntimeError(
                f"restored fleet retained {len(drift)} runner-state drift issue(s)"
            )


def _assert_restored_fleet_clean_through_target(
    target: FreshInstallTarget,
    *,
    expected_node_count: int,
    remote_port: int,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Reopen the physical entrypoint tunnel and audit the restored fleet."""

    controller = SshTargetController(target)
    local_port, tunnel = controller.open_tunnel(remote_port=remote_port)
    try:
        api_base_url = f"http://127.0.0.1:{local_port}"
        _wait_for_api_identity(
            api_base_url,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        _assert_restored_fleet_clean(
            api_base_url,
            expected_node_count=expected_node_count,
        )
    finally:
        _terminate_process(tunnel)


def _clean_git_checkout_identity(
    repository_root: Path,
    *,
    expected_commit: str | None = None,
) -> tuple[str, str]:
    """Return commit and tree IDs only for a clean, pinned Git checkout."""

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError) as exception:
            raise ValueError(
                "resumption harness source checkout could not be verified"
            ) from exception
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if expected_commit is not None and not commit_matches(expected_commit, commit):
        raise ValueError("resumption harness source commit does not match its report")
    if git("status", "--porcelain"):
        raise ValueError("resumption requires a clean harness source checkout")
    tree = git("rev-parse", "HEAD^{tree}")
    if not commit or not tree:
        raise ValueError("resumption harness source identity is incomplete")
    return commit, tree


def _prepare_e2e_resumption_source(
    *,
    resume_from: Path,
    repository_root: Path,
    fleet: FreshInstallPhysicalFleet,
    model_sets_path: Path,
    test_sets_path: Path,
    profile: FreshInstallProfile,
    expected_commit: str,
) -> _FreshE2EResumptionSource:
    """Validate a predecessor that may resume only the failed provenance gate."""

    report_path = resume_from.expanduser().resolve()
    if report_path.is_dir():
        report_path = report_path / "fresh-install-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"resumption report is missing: {report_path}")
    predecessor = FreshInstallQualificationReport.model_validate_json(
        report_path.read_text()
    )
    if predecessor.passed:
        raise ValueError("resumption predecessor already passed qualification")
    if predecessor.profile != profile or predecessor.platform != "mixed":
        raise ValueError("resumption predecessor profile or platform does not match")
    if predecessor.install.expected_commit != expected_commit or not commit_matches(
        expected_commit, predecessor.install.resolved_commit
    ):
        raise ValueError("resumption predecessor does not prove the candidate commit")
    if (
        predecessor.install.mode != "fresh_install"
        or predecessor.install.environment != "fresh_install"
    ):
        raise ValueError("resumption predecessor is not a fresh-install run")
    if (
        predecessor.restoration_succeeded is not True
        or predecessor.teardown_succeeded is not True
        or predecessor.critical_recovery_required
    ):
        raise ValueError("resumption predecessor did not restore and tear down cleanly")
    failed_stages = [stage for stage in predecessor.lifecycle if stage.status == "failed"]
    provenance_failures = [
        stage
        for stage in failed_stages
        if stage.name == "run complete E2E battery on fresh physical fleet"
        and stage.message is not None
        and "result or provenance gate" in stage.message
    ]
    if len(failed_stages) != 1 or len(provenance_failures) != 1:
        raise ValueError(
            "resumption is allowed only after the complete E2E provenance gate failed"
        )

    e2e_root = report_path.parent / "complete-e2e"
    battery_log = e2e_root / "battery.log"
    report_paths = tuple(sorted((e2e_root / "runs").glob("*/report.json")))
    if not battery_log.is_file() or not report_paths:
        raise ValueError("resumption predecessor is missing complete E2E artifacts")
    reports = tuple(
        RunReport.model_validate_json(path.read_text()) for path in report_paths
    )
    node_sets = {
        frozenset(node.node_id for node in report.fingerprint.cluster.nodes)
        for report in reports
        if report.fingerprint is not None
    }
    if len(node_sets) != 1:
        raise ValueError("resumption predecessor reports do not share one topology")
    expected_node_ids = next(iter(node_sets))
    if len(expected_node_ids) != len(fleet.member_targets):
        raise ValueError("resumption predecessor topology size does not match the fleet")

    script_path = _qualification_source_path(
        repository_root,
        fleet.e2e_battery_script,
        label="full E2E battery script",
    )
    current_model_sets = _qualification_source_path(
        repository_root,
        model_sets_path,
        label="model-set matrix",
    )
    current_test_sets = _qualification_source_path(
        repository_root,
        test_sets_path,
        label="test-set matrix",
    )
    for snapshot_name, current_path in (
        ("model-sets.yaml", current_model_sets),
        ("test-sets.yaml", current_test_sets),
    ):
        snapshot_path = e2e_root / snapshot_name
        if not snapshot_path.is_file() or snapshot_path.read_bytes() != current_path.read_bytes():
            raise ValueError(
                f"resumption predecessor {snapshot_name} does not match the current matrix"
            )

    log_cells = re.findall(
        r"==== CELL\s+model-set=(\S+)\s+test-set=(\S+)\s+START ====",
        battery_log.read_text(),
    )
    script_cells = re.findall(
        r"^\s*cell\s+(\S+)\s+(\S+)(?:\s|$)",
        script_path.read_text(),
        flags=re.MULTILINE,
    )
    report_cells = [(report.spec.model_set, report.spec.test_set) for report in reports]
    if not log_cells or log_cells != script_cells or report_cells != log_cells:
        raise ValueError(
            "resumption predecessor cell manifest does not match the current battery"
        )

    evidence = _summarize_fresh_e2e_battery(
        script_path=script_path,
        battery_log=battery_log,
        report_root=e2e_root / "runs",
        expected_commit=expected_commit,
        expected_node_ids=expected_node_ids,
        process_returncode=0,
    )
    if not evidence.passed:
        raise ValueError(
            "resumption predecessor does not pass every result and provenance gate"
        )
    harness_references = [
        repository
        for report in reports
        if report.fingerprint is not None
        for repository in report.fingerprint.source_context.repositories
        if repository.name.endswith("/skulk-test-harness")
    ]
    if len(harness_references) != len(reports):
        raise ValueError("resumption predecessor has incomplete harness provenance")
    if any(reference.dirty is True for reference in harness_references):
        raise ValueError("resumption predecessor used a dirty harness checkout")
    harness_commits = {
        reference.commit for reference in harness_references if reference.commit
    }
    harness_paths = {
        reference.path for reference in harness_references if reference.path
    }
    if len(harness_commits) != 1 or len(harness_paths) != 1:
        raise ValueError("resumption predecessor has ambiguous harness provenance")
    predecessor_harness_root = Path(next(iter(harness_paths))).expanduser().resolve()
    predecessor_harness_commit = next(iter(harness_commits))
    predecessor_harness_commit, predecessor_harness_tree = (
        _clean_git_checkout_identity(
            predecessor_harness_root,
            expected_commit=predecessor_harness_commit,
        )
    )
    current_harness_commit, current_harness_tree = _clean_git_checkout_identity(
        repository_root
    )
    return _FreshE2EResumptionSource(
        report=predecessor,
        root=e2e_root,
        battery_log=battery_log,
        report_paths=report_paths,
        reports=reports,
        expected_node_ids=expected_node_ids,
        predecessor_harness_root=predecessor_harness_root,
        predecessor_harness_commit=predecessor_harness_commit,
        predecessor_harness_tree=predecessor_harness_tree,
        current_harness_commit=current_harness_commit,
        current_harness_tree=current_harness_tree,
    )


def _seal_and_replay_e2e_resumption(
    *,
    source: _FreshE2EResumptionSource,
    repository_root: Path,
    fleet: FreshInstallPhysicalFleet,
    expected_commit: str | None,
    artifact_directory: Path,
) -> tuple[FreshInstallE2EBatteryEvidence, FreshInstallE2EResumptionEvidence]:
    """Seal predecessor reports and rerun the failed provenance gate."""

    if expected_commit is None:
        raise ValueError("fresh E2E resumption requires an exact expected commit")
    sealed_root = artifact_directory / "complete-e2e-resumption"
    sealed_reports = sealed_root / "runs"
    sealed_reports.mkdir(parents=True, exist_ok=False)
    sealed_root.chmod(0o700)
    sealed_reports.chmod(0o700)
    script_path = _qualification_source_path(
        repository_root,
        fleet.e2e_battery_script,
        label="full E2E battery script",
    )
    for source_path, target_name in (
        (source.battery_log, "battery.log"),
        (source.root / "model-sets.yaml", "model-sets.yaml"),
        (source.root / "test-sets.yaml", "test-sets.yaml"),
        (script_path, "run_e2e_battery.sh"),
    ):
        target_path = sealed_root / target_name
        shutil.copy2(source_path, target_path)
        target_path.chmod(0o600)

    cells: list[dict[str, object]] = []
    for ordinal, (source_path, report) in enumerate(
        zip(source.report_paths, source.reports, strict=True),
        start=1,
    ):
        target_directory = sealed_reports / f"{ordinal:02d}"
        target_directory.mkdir(mode=0o700)
        target_path = target_directory / "report.json"
        shutil.copy2(source_path, target_path)
        target_path.chmod(0o600)
        cells.append(
            {
                "ordinal": ordinal,
                "run_id": report.run_id,
                "model_set": report.spec.model_set,
                "test_set": report.spec.test_set,
                "report_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                "result_count": len(report.results),
            }
        )

    predecessor_identity = _clean_git_checkout_identity(
        source.predecessor_harness_root,
        expected_commit=source.predecessor_harness_commit,
    )
    if predecessor_identity != (
        source.predecessor_harness_commit,
        source.predecessor_harness_tree,
    ):
        raise RuntimeError("predecessor harness source changed during resumption")
    current_identity = _clean_git_checkout_identity(
        repository_root,
        expected_commit=source.current_harness_commit,
    )
    if current_identity != (
        source.current_harness_commit,
        source.current_harness_tree,
    ):
        raise RuntimeError("current harness source changed during resumption")
    current_harness_commit, current_harness_tree = current_identity
    manifest = {
        "schema_version": "1.0",
        "predecessor_qualification_id": source.report.qualification_id,
        "predecessor_expected_commit": expected_commit,
        "predecessor_harness_commit": source.predecessor_harness_commit,
        "predecessor_harness_tree": source.predecessor_harness_tree,
        "current_harness_commit": current_harness_commit,
        "current_harness_tree": current_harness_tree,
        "battery_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "model_sets_sha256": hashlib.sha256(
            (sealed_root / "model-sets.yaml").read_bytes()
        ).hexdigest(),
        "test_sets_sha256": hashlib.sha256(
            (sealed_root / "test-sets.yaml").read_bytes()
        ).hexdigest(),
        "completed_cells": cells,
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path = sealed_root / "resumption-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o600)
    evidence = _summarize_fresh_e2e_battery(
        script_path=script_path,
        battery_log=sealed_root / "battery.log",
        report_root=sealed_reports,
        expected_commit=expected_commit,
        expected_node_ids=source.expected_node_ids,
        process_returncode=0,
    )
    if not evidence.passed:
        raise RuntimeError("sealed predecessor failed the resumed provenance gate")
    return evidence, FreshInstallE2EResumptionEvidence(
        predecessor_qualification_id=source.report.qualification_id,
        predecessor_expected_commit=expected_commit,
        predecessor_harness_commit=source.predecessor_harness_commit,
        predecessor_harness_tree=source.predecessor_harness_tree,
        current_harness_commit=current_harness_commit,
        current_harness_tree=current_harness_tree,
        completed_cell_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        completed_cell_count=evidence.completed_cell_count,
        passed_result_count=evidence.passed_result_count,
        passed=True,
    )


def _summarize_fresh_e2e_battery(
    *,
    script_path: Path,
    battery_log: Path,
    report_root: Path,
    expected_commit: str,
    expected_node_ids: frozenset[str],
    process_returncode: int,
) -> FreshInstallE2EBatteryEvidence:
    """Build a strict composite verdict from every full-battery report."""

    log_text = battery_log.read_text() if battery_log.is_file() else ""
    cell_count = len(re.findall(r"==== CELL .* START ====", log_text))
    completed_cell_count = len(re.findall(r"==== CELL .* END \(rc=0\) ====", log_text))
    reports = [
        RunReport.model_validate_json(path.read_text())
        for path in sorted(report_root.rglob("report.json"))
    ]
    results = [result for report in reports for result in report.results]
    issues = [issue for report in reports for issue in report.issues]
    fresh_reports = [
        report
        for report in reports
        if report.fingerprint is not None
        and report.fingerprint.install.mode == "fresh_install"
        and report.fingerprint.install.environment == "fresh_install"
    ]
    expected_commit_reports = [
        report
        for report in fresh_reports
        if report.fingerprint is not None
        and report.fingerprint.install.expected_commit == expected_commit
        and commit_matches(
            expected_commit, report.fingerprint.install.resolved_commit
        )
    ]
    live_commit_reports = [
        report
        for report in reports
        if report.fingerprint is not None
        and commit_matches(expected_commit, report.fingerprint.runtime.skulk_commit)
    ]
    exact_topology_reports = [
        report
        for report in reports
        if report.fingerprint is not None
        and report.fingerprint.cluster.node_count == len(expected_node_ids)
        and {node.node_id for node in report.fingerprint.cluster.nodes}
        == expected_node_ids
    ]
    passed_results = [result for result in results if result.passed]
    failed_results = [result for result in results if not result.passed]
    passed = (
        process_returncode == 0
        and "BATTERY COMPLETE (rc=0)" in log_text
        and cell_count > 0
        and completed_cell_count == cell_count
        and len(reports) == cell_count
        and bool(results)
        and not failed_results
        and not issues
        and len(fresh_reports) == len(reports)
        and len(expected_commit_reports) == len(reports)
        and len(live_commit_reports) == len(reports)
        and len(exact_topology_reports) == len(reports)
    )
    return FreshInstallE2EBatteryEvidence(
        script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
        cell_count=cell_count,
        completed_cell_count=completed_cell_count,
        report_count=len(reports),
        result_count=len(results),
        passed_result_count=len(passed_results),
        failed_result_count=len(failed_results),
        issue_count=len(issues),
        fresh_provenance_report_count=len(fresh_reports),
        expected_commit_report_count=len(expected_commit_reports),
        live_commit_report_count=len(live_commit_reports),
        exact_topology_report_count=len(exact_topology_reports),
        passed=passed,
    )


def _wait_for_exact_cluster(
    api_base_url: str,
    *,
    expected_node_count: int,
    timeout_s: float,
    poll_interval_s: float,
) -> frozenset[str]:
    """Wait until one API observes the complete exact physical topology."""

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    with SkulkClient(api_base_url) as client:
        while time.monotonic() < deadline:
            try:
                return assert_fresh_cluster(
                    client,
                    expected_node_count=expected_node_count,
                )
            except UnexpectedFreshInstallPeerError:
                raise
            except Exception as exception:  # noqa: BLE001 - topology converges
                last_error = exception
                time.sleep(poll_interval_s)
    raise TimeoutError(
        f"fresh physical fleet did not form {expected_node_count} nodes: {last_error}"
    )


def _wait_for_restored_declared_topology(
    members: Sequence[_PhysicalFleetMemberRuntime],
    *,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat,
) -> list[tuple[_PhysicalFleetMemberRuntime, str, int, frozenset[str]]]:
    """Wait until every restored member reports one identical complete topology.

    Restored services can expose their local API before election and discovery
    have converged on the replacement runtime identities. Sampling each member
    with an independent readiness wait can therefore combine individually valid
    but temporally inconsistent views. A release recovery verdict must instead
    come from one complete sweep in which every declared member agrees.
    """

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        _observe_heartbeat_during_recovery(heartbeat)
        observations: list[
            tuple[_PhysicalFleetMemberRuntime, str, int, frozenset[str]]
        ] = []
        try:
            for member in members:
                _observe_heartbeat_during_recovery(heartbeat)
                if member.local_port is None:
                    raise RuntimeError(
                        f"member {member.ordinal} lacks a recovery API tunnel"
                    )
                api_base_url = f"http://127.0.0.1:{member.local_port}"
                with SkulkClient(api_base_url) as client:
                    node_id = client.get_node_id()
                    observed_node_ids = assert_fresh_cluster(
                        client,
                        expected_node_count=len(members),
                    )
                observations.append(
                    (
                        member,
                        node_id,
                        len(observed_node_ids),
                        observed_node_ids,
                    )
                )
            _assert_declared_member_topologies(
                expected_node_count=len(members),
                local_node_ids=(node_id for _, node_id, _, _ in observations),
                member_observed_node_ids=(
                    observed_node_ids
                    for _, _, _, observed_node_ids in observations
                ),
            )
            return observations
        except Exception as exception:  # noqa: BLE001 - restored peers converge
            last_error = exception
        time.sleep(poll_interval_s)
    raise TimeoutError(
        "restored physical fleet did not converge on one complete declared "
        f"topology: {last_error}"
    )


def _qualify_served_engine_fleet(
    *,
    members: Sequence[_PhysicalFleetMemberRuntime],
    api_base_url: str,
    model_id: str,
    contract: ServedEngineContract,
    request_timeout_s: float,
    stream_read_timeout_s: float,
) -> ServedEngineEvidence:
    """Verify one served runner wherever placement selected it in the fleet."""

    before: list[tuple[int, int, int, bool]] = []
    for member in members:
        assert member.temporary_root is not None
        listing = member.controller.run(
            "ps -axo pid=,command=",
            timeout_s=30,
        ).stdout
        for pid, parallel, kv_unified in _llama_server_process_candidates(
            listing,
            installation_root=member.temporary_root,
        ):
            before.append((member.ordinal, pid, parallel, kv_unified))
    if len(before) != 1:
        raise RuntimeError(
            "expected exactly one fresh llama-server child across compatible "
            f"fleet members, observed {len(before)}"
        )
    member_ordinal, before_pid, observed_parallel, kv_unified_observed = before[0]
    _run_served_engine_overlap_probe(
        api_base_url=api_base_url,
        model_id=model_id,
        concurrency=contract.probe_concurrency,
        request_timeout_s=request_timeout_s,
        stream_read_timeout_s=stream_read_timeout_s,
    )
    maximum_observed_active, batching_reported = _served_engine_envelope(
        api_base_url=api_base_url,
        model_id=model_id,
        backend=contract.backend,
        request_timeout_s=request_timeout_s,
    )
    after: list[tuple[int, int, int, bool]] = []
    for member in members:
        assert member.temporary_root is not None
        listing = member.controller.run(
            "ps -axo pid=,command=",
            timeout_s=30,
        ).stdout
        for pid, parallel, kv_unified in _llama_server_process_candidates(
            listing,
            installation_root=member.temporary_root,
        ):
            after.append((member.ordinal, pid, parallel, kv_unified))
    runner_survived = after == [
        (
            member_ordinal,
            before_pid,
            observed_parallel,
            kv_unified_observed,
        )
    ]
    passed = (
        observed_parallel == contract.parallel
        and kv_unified_observed == contract.kv_unified
        and maximum_observed_active >= 2
        and batching_reported
        and runner_survived
    )
    return ServedEngineEvidence(
        model_id=model_id,
        backend=contract.backend,
        expected_parallel=contract.parallel,
        observed_parallel=observed_parallel,
        kv_unified_required=contract.kv_unified,
        kv_unified_observed=kv_unified_observed,
        probe_concurrency=contract.probe_concurrency,
        maximum_observed_active=maximum_observed_active,
        batching_reported=batching_reported,
        runner_survived=runner_survived,
        passed=passed,
    )


def _qualify_served_engine(
    *,
    controller: SshTargetController,
    installation_root: str,
    api_base_url: str,
    model_id: str,
    contract: ServedEngineContract,
    request_timeout_s: float,
    stream_read_timeout_s: float,
) -> ServedEngineEvidence:
    """Verify effective served-engine flags, overlap, and post-load survival."""

    before = controller.run("ps -axo pid=,command=", timeout_s=30).stdout
    before_pid, observed_parallel, kv_unified_observed = _llama_server_process_contract(
        before,
        installation_root=installation_root,
    )
    _run_served_engine_overlap_probe(
        api_base_url=api_base_url,
        model_id=model_id,
        concurrency=contract.probe_concurrency,
        request_timeout_s=request_timeout_s,
        stream_read_timeout_s=stream_read_timeout_s,
    )
    maximum_observed_active, batching_reported = _served_engine_envelope(
        api_base_url=api_base_url,
        model_id=model_id,
        backend=contract.backend,
        request_timeout_s=request_timeout_s,
    )
    after = controller.run("ps -axo pid=,command=", timeout_s=30).stdout
    after_pid, after_parallel, after_kv_unified = _llama_server_process_contract(
        after,
        installation_root=installation_root,
    )
    runner_survived = (
        after_pid == before_pid
        and after_parallel == observed_parallel
        and after_kv_unified == kv_unified_observed
    )
    passed = (
        observed_parallel == contract.parallel
        and kv_unified_observed == contract.kv_unified
        and maximum_observed_active >= 2
        and batching_reported
        and runner_survived
    )
    return ServedEngineEvidence(
        model_id=model_id,
        backend=contract.backend,
        expected_parallel=contract.parallel,
        observed_parallel=observed_parallel,
        kv_unified_required=contract.kv_unified,
        kv_unified_observed=kv_unified_observed,
        probe_concurrency=contract.probe_concurrency,
        maximum_observed_active=maximum_observed_active,
        batching_reported=batching_reported,
        runner_survived=runner_survived,
        passed=passed,
    )


def _llama_server_process_contract(
    process_listing: str,
    *,
    installation_root: str,
) -> tuple[int, int, bool]:
    """Return process identity, parallel slots, and unified-KV state."""

    candidates = _llama_server_process_candidates(
        process_listing,
        installation_root=installation_root,
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one fresh llama-server child, observed {len(candidates)}"
        )
    return candidates[0]


def _llama_server_process_candidates(
    process_listing: str,
    *,
    installation_root: str,
) -> list[tuple[int, int, bool]]:
    """Return every served child owned by one temporary installation."""

    raw_candidates: list[tuple[int, list[str]]] = []
    for line in process_listing.splitlines():
        if installation_root not in line:
            continue
        raw_pid, separator, raw_command = line.strip().partition(" ")
        if not separator:
            continue
        try:
            pid = int(raw_pid)
            arguments = shlex.split(raw_command.strip())
        except ValueError:
            continue
        if any(Path(argument).name == "llama-server" for argument in arguments):
            raw_candidates.append((pid, arguments))
    parsed: list[tuple[int, int, bool]] = []
    for pid, arguments in raw_candidates:
        parallel: int | None = None
        for index, argument in enumerate(arguments):
            if argument == "--parallel" and index + 1 < len(arguments):
                try:
                    parallel = int(arguments[index + 1])
                except ValueError as exception:
                    raise RuntimeError(
                        "fresh llama-server --parallel value was not an integer"
                    ) from exception
            elif argument.startswith("--parallel="):
                try:
                    parallel = int(argument.partition("=")[2])
                except ValueError as exception:
                    raise RuntimeError(
                        "fresh llama-server --parallel value was not an integer"
                    ) from exception
        if parallel is None:
            raise RuntimeError("fresh llama-server child omitted --parallel")
        parsed.append((pid, parallel, "--kv-unified" in arguments))
    return parsed


def _run_served_engine_overlap_probe(
    *,
    api_base_url: str,
    model_id: str,
    concurrency: int,
    request_timeout_s: float,
    stream_read_timeout_s: float,
) -> None:
    """Drive a bounded simultaneous burst through the ordinary chat endpoint."""

    async def run() -> None:
        async with concurrent_benchmark_client(
            api_base_url,
            concurrency=concurrency,
        ) as client:
            executions = await asyncio.gather(
                *(
                    stream_chat_async(
                        client,
                        model_id=model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Write the lowercase letter x exactly 96 "
                                    "times. Do not add spaces or commentary."
                                ),
                            }
                        ],
                        max_tokens=128,
                        temperature=0.0,
                        top_p=1.0,
                        request_timeout_s=request_timeout_s,
                        stream_read_timeout_s=stream_read_timeout_s,
                    )
                    for _ in range(concurrency)
                )
            )
        if any(not execution.text.strip() for execution in executions):
            raise RuntimeError("served-engine overlap probe returned an empty response")

    asyncio.run(run())


def _served_engine_envelope(
    *,
    api_base_url: str,
    model_id: str,
    backend: str,
    request_timeout_s: float,
) -> tuple[int, bool]:
    """Return maximum observed live concurrency and batching truth."""

    response = httpx.get(
        api_base_url.rstrip("/") + "/v1/diagnostics/performance-envelopes",
        timeout=request_timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("performance-envelope endpoint returned a non-object")
    raw_envelopes = body.get("envelopes")
    if not isinstance(raw_envelopes, list):
        raise RuntimeError("performance-envelope endpoint omitted envelopes")
    maximum = 0
    batches = False
    for raw_envelope in raw_envelopes:
        if not isinstance(raw_envelope, dict):
            continue
        envelope_model = raw_envelope.get(
            "modelId",
            raw_envelope.get("model_id"),
        )
        envelope_backend = raw_envelope.get("backend")
        if envelope_model != model_id or envelope_backend != backend:
            continue
        batches = batches or raw_envelope.get("batches") is True
        raw_buckets = raw_envelope.get("buckets")
        if not isinstance(raw_buckets, list):
            continue
        for raw_bucket in raw_buckets:
            if not isinstance(raw_bucket, dict):
                continue
            concurrency = raw_bucket.get("concurrency")
            if isinstance(concurrency, int) and not isinstance(concurrency, bool):
                maximum = max(maximum, concurrency)
    return maximum, batches


def _blocking_issues(report: FreshInstallQualificationReport) -> list[Issue]:
    """Return only the issues that should fail a leg.

    A qualification records two different kinds of finding. An ``error`` is a
    release gate saying the candidate or the machine is not acceptable. A
    ``warning`` is a condition worth an operator's attention that does not
    make the leg's verdict false, such as the fleet coming back with fewer
    members than it had before the run because peers outside the experiment
    were down the whole time.

    Failing on any issue at all collapsed that distinction: a leg whose every
    stage passed, whose target restored cleanly, and whose only finding was
    that exact fleet-size warning was reported as a failed release gate. The
    warning had already been deliberately downgraded from a fatal restoration
    failure for this reason, so counting it here undid the downgrade and left
    only the label changed.
    """

    return [issue for issue in report.issues if issue.severity == "error"]


def _record_deferred_interruption(
    report: FreshInstallQualificationReport | ReleaseQualificationReport,
    signal_guard: QualificationSignalGuard,
) -> None:
    """Record a signal deferred until mandatory recovery had completed."""

    signum = signal_guard.interrupted_signum
    if signum is None:
        return
    if any("interrupted" in issue.message.lower() for issue in report.issues):
        return
    report.issues.append(
        Issue(
            severity="error",
            message=(
                "fresh-install qualification interrupted by signal "
                f"{signum}; mandatory recovery completed before stopping"
            ),
        )
    )


def _failed_lifecycle_stages(
    report: FreshInstallQualificationReport,
) -> list[FreshInstallLifecycleStage]:
    """Return failed stages so no incomplete lifecycle can publish a pass."""

    return [stage for stage in report.lifecycle if stage.status == "failed"]


def _browser_vision_expectation(
    model_id: str,
    *,
    vision_models: Sequence[str],
    card_image_input: bool | None,
) -> Literal["positive", "unavailable"]:
    """Return what the browser journey must prove about vision for one model.

    The expectation is per model, not per target. The dashboard enables its
    attachment control from the selected model's own image-input support, so a
    text model must offer no image path even on a target whose platform can
    serve vision. Passing a target's ``positive`` contract straight through for
    a text model demanded a vision check on a model that has no vision, and
    failed the leg on a fixture that could not exist for it.

    The shipped card is the authority. ``card_image_input`` is what the release
    itself declares for this model, and the inventory's ``vision_models`` list
    is the operator's statement of intent; a disagreement means one of the two
    is wrong and is raised as its own error. Deriving the expectation from an
    inventory list alone let a hand-maintained list drift from the card, and
    the leg then failed on a UI assertion that was itself incorrect instead of
    naming the real problem.
    """

    declared_vision = model_id in vision_models
    if card_image_input is not None and card_image_input != declared_vision:
        raise RuntimeError(
            f"vision classification disagrees for {model_id}: the shipped card "
            f"reports supports_image_input={card_image_input} but the "
            f"qualification inventory lists it as a "
            f"{'vision' if declared_vision else 'text'} model. Fix whichever is "
            "wrong before qualifying this model."
        )
    return "positive" if declared_vision else "unavailable"


def _installer_provenance(
    url_template: str,
    *,
    profile: FreshInstallProfile,
    expected_commit: str | None,
    shipping_ref: str,
) -> tuple[str, str]:
    """Fetch the exact installer bytes so their digest is retained."""

    ref = expected_commit if profile == "candidate" else shipping_ref
    assert ref is not None
    url = url_template.format(ref=ref)
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return url, hashlib.sha256(response.content).hexdigest()


def _installer_command(
    *,
    installer_url: str,
    profile: FreshInstallProfile,
    expected_commit: str | None,
) -> str:
    """Return the candidate or literal public shipping installer command."""

    quoted_url = shlex.quote(installer_url)
    if profile == "shipping":
        return f"curl -fsSL {quoted_url} | bash"
    _require_commit_sha(expected_commit)
    assert expected_commit is not None
    return f"curl -fsSL {quoted_url} | bash -s -- --ref {expected_commit}"


def _clean_environment_command(temporary_root: str, command: str) -> str:
    """Run with an empty HOME and no inherited SKULK environment overrides."""

    home = shlex.quote(temporary_root + "/home")
    temporary = shlex.quote(temporary_root + "/tmp")
    path = (
        f"{temporary_root}/home/.local/bin:{temporary_root}/home/.cargo/bin:"
        "/opt/homebrew/bin:"
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )
    return (
        f"env -i HOME={home} USER=$(id -un) TMPDIR={temporary} "
        f"LANG=C.UTF-8 PATH={shlex.quote(path)} "
        f"bash -c {shlex.quote(command)}"
    )


def _runtime_start_command(
    *,
    temporary_root: str,
    target: FreshInstallTarget,
) -> str:
    """Wrap the literal clean runtime command only in declared OS isolation."""

    command = _clean_environment_command(
        temporary_root,
        'cd "$HOME/skulk" && exec uv run skulk',
    )
    if target.runtime_isolation_prefix:
        return f"{target.runtime_isolation_prefix} {command}"
    return command


def _remote_sha256(
    controller: SshTargetController,
    path: str,
) -> str | None:
    """Return a portable remote file hash."""

    quoted = shlex.quote(path)
    result = controller.run(
        f"if [ -f {quoted} ]; then "
        "if command -v shasum >/dev/null 2>&1; "
        f"then shasum -a 256 {quoted}; else sha256sum {quoted}; fi; fi",
        check=False,
        timeout_s=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def _wait_for_runtime_contract(
    api_base_url: str,
    *,
    target: FreshInstallTarget,
    expected_commit: str | None,
    timeout_s: float,
    poll_interval_s: float,
    stability_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
    expected_node_count: int = 1,
    expected_node_ids: frozenset[str] | None = None,
    expected_backends: list[str] | None = None,
) -> InstallProvenance:
    """Require every shipped runtime invariant to remain continuously stable."""

    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    stable_provenance: InstallProvenance | None = None
    last_error: Exception | None = None
    with SkulkClient(api_base_url) as client:
        while time.monotonic() < deadline:
            _check_heartbeat(heartbeat)
            try:
                stable_provenance = assert_fresh_runtime_contract(
                    client,
                    expected_backends=expected_backends or target.expected_backends,
                    expected_transport=target.expected_data_transport,
                    expected_commit=expected_commit,
                    dashboard_contract=target.dashboard_contract,
                    expected_node_count=expected_node_count,
                    expected_node_ids=expected_node_ids,
                )
                now = time.monotonic()
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= stability_s:
                    return stable_provenance
            except UnexpectedFreshInstallPeerError:
                # Once another node appears, the clean install is not isolated.
                # Waiting for it to disappear would turn a real qualification
                # failure into a timing-dependent pass.
                raise
            except Exception as exception:  # noqa: BLE001 - startup is eventually consistent
                last_error = exception
                stable_since = None
                stable_provenance = None
                time.sleep(poll_interval_s)
                continue
            time.sleep(poll_interval_s)
    raise TimeoutError(f"fresh runtime contract did not settle: {last_error}")


def _wait_for_api_identity(
    api_base_url: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[str, int]:
    """Wait for the API to answer with an identity and report its fleet size.

    Readiness is deliberately the target answering, not the fleet reaching a
    given size. A leg starts exactly one node, so gating this wait on a
    pre-run fleet count made it depend on peers the leg never touched and
    timed out on targets that had already recovered.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with SkulkClient(api_base_url) as client:
                node_id = client.get_node_id()
                state = client.get_state()
            identities = state.get("nodeIdentities")
            resources = state.get("nodeResources")
            ids: set[object] = set()
            if isinstance(identities, dict):
                ids.update(identities)
            if isinstance(resources, dict):
                ids.update(resources)
            return node_id, len(ids)
        except Exception:  # noqa: BLE001 - service is starting
            pass
        time.sleep(poll_interval_s)
    raise TimeoutError("target API did not become ready")


def _wait_for_http(
    url: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
) -> None:
    """Wait for one successful HTTP response."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_heartbeat(heartbeat)
        try:
            if httpx.get(url, timeout=5).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(poll_interval_s)
    raise TimeoutError(f"HTTP endpoint did not become ready: {url}")


_COMPLETED_DOWNLOAD_STATES = frozenset({"complete", "completed", "ready", "succeeded"})
_FAILED_DOWNLOAD_STATES = frozenset({"failed", "error"})


def _provision_model_over_api(
    client: SkulkClient,
    *,
    model_id: str,
    model_ready_timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
    runtime_check: Callable[[], None] | None = None,
) -> None:
    """Download and mount one model through the API the dashboard itself calls.

    A target that ships without the web UI still has to prove that a brand-new
    machine can go from nothing to a served model. This walks the same
    endpoints the browser journey drives (download into the node's store, place
    an instance, wait for a ready runner) so the headless leg exercises the
    identical product path rather than a weaker one.

    A model the release already ships a card for is never re-added. ``POST
    /models/add`` fetches a card from Hugging Face and stores it as a *custom*
    card, which then overrides the shipped one, so adding it here would
    qualify a card the release does not ship. This mirrors what the dashboard
    does: it offers "Download" for a model already in the catalog and "Add and
    download" only for one that is not.

    Every step is fatal. This is a release gate, so a model that cannot be
    fetched or mounted must fail the leg rather than degrade into a parity
    check against a model that was never there.
    """

    _check_optional_runtime(runtime_check)
    card_error: str | None = None
    catalog = {
        str(entry.get("id"))
        for entry in client.list_models()
        if isinstance(entry.get("id"), str)
    }
    if model_id not in catalog:
        try:
            client.add_model_card(model_id)
        except (SkulkApiError, httpx.HTTPError) as exception:
            # Only fatal if the download below also fails: the catalog read
            # could be stale, and the download is the step that actually
            # decides whether this model can be provisioned.
            card_error = str(exception)
    try:
        client.request_store_download(model_id)
    except (SkulkApiError, httpx.HTTPError) as exception:
        detail = str(exception)
        if card_error:
            detail = f"{detail} (after card add failed: {card_error})"
        raise RuntimeError(
            f"could not request a download for {model_id}: {detail}"
        ) from exception

    deadline = time.monotonic() + model_ready_timeout_s
    while True:
        _check_heartbeat(heartbeat)
        _check_optional_runtime(runtime_check)
        status = client.get_store_download_status(model_id) or {}
        state = str(status.get("status") or status.get("state") or "").lower()
        if state in _COMPLETED_DOWNLOAD_STATES:
            break
        if state in _FAILED_DOWNLOAD_STATES:
            raise RuntimeError(f"store download failed for {model_id}: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"store download did not complete for {model_id}: last state {state!r}"
            )
        time.sleep(poll_interval_s)

    offered = client.get_placement_previews(model_id)
    _check_optional_runtime(runtime_check)
    # A preview carrying an error is an option the planner examined and
    # rejected (it does not fit, or the node cannot serve it), and the API is
    # free to list one ahead of a viable option. Taking the first entry blindly
    # would place against a rejected option and fail a target that was in fact
    # offering a working placement further down the list. Every other placement
    # path in this harness filters the same way.
    previews = [preview for preview in offered if preview.get("error") in (None, "")]
    if not previews:
        raise RuntimeError(
            f"no viable placement preview was offered for {model_id}: {offered}"
        )
    preview = previews[0]
    placed = client.place_model(
        model_id=model_id,
        sharding=str(preview.get("sharding") or "auto"),
        instance_meta=str(preview.get("instance_meta") or "TextInstance"),
        min_nodes=1,
        excluded_nodes=[],
    )
    if placed is None:
        raise RuntimeError(f"placement request was refused for {model_id}")

    ready_deadline = time.monotonic() + model_ready_timeout_s
    while True:
        _check_heartbeat(heartbeat)
        _check_optional_runtime(runtime_check)
        placements = client.find_placements_for_model(model_id)
        if any(placement.ready for placement in placements):
            return
        failures = [
            message
            for placement in placements
            if placement.terminal_failure
            for message in placement.runner_failure_messages
        ]
        if failures:
            raise RuntimeError(f"runner failed for {model_id}: {'; '.join(failures)}")
        if time.monotonic() >= ready_deadline:
            raise TimeoutError(f"placement never became ready for {model_id}")
        time.sleep(poll_interval_s)


def _wait_for_no_placement(
    client: SkulkClient,
    *,
    model_id: str,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
    runtime_check: Callable[[], None] | None = None,
) -> None:
    """Wait until one temporary model has no remaining instances."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_heartbeat(heartbeat)
        _check_optional_runtime(runtime_check)
        if not client.find_placements_for_model(model_id):
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"temporary placement did not stop: {model_id}")


def _check_optional_runtime(runtime_check: Callable[[], None] | None) -> None:
    """Run an optional qualification invariant callback."""

    if runtime_check is not None:
        runtime_check()


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child process without allowing cleanup to hang."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate a locally created process session, including its children."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _run_remote_logged_command(
    *,
    controller: SshTargetController,
    command: str,
    log_path: Path,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
) -> int:
    """Run a remote command while polling the authoritative lease heartbeat."""

    process, log_handle = controller.start(command, log_path=log_path)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            _check_heartbeat(heartbeat)
            if time.monotonic() >= deadline:
                raise TimeoutError("remote command exceeded its lifecycle timeout")
            time.sleep(poll_interval_s)
    finally:
        if process.poll() is None:
            _terminate_process(process)
        log_handle.close()


def _check_heartbeat(
    heartbeat: AuthoritativeLeaseHeartbeat | None,
) -> None:
    """Abort at the next lifecycle boundary after a renewal failure."""

    if heartbeat is not None:
        heartbeat.raise_if_failed()


def _observe_heartbeat_during_recovery(
    heartbeat: AuthoritativeLeaseHeartbeat,
) -> None:
    """Poll lease health without allowing failure to interrupt restoration.

    The heartbeat retains its first failure, so the lifecycle reports it and
    performs the emergency extension immediately after recovery. Original
    services must still finish restoring even when lease renewal has failed.
    """

    with suppress(LeaseHeartbeatError):
        heartbeat.raise_if_failed()


def _qualification_source_path(
    repository_root: Path,
    configured_path: Path,
    *,
    label: str,
) -> Path:
    """Resolve and require one local input before qualification mutates state."""

    source_path = configured_path.expanduser()
    if not source_path.is_absolute():
        source_path = repository_root / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"{label} is missing: {source_path}")
    return source_path


def _require_commit_sha(value: str | None) -> None:
    """Require a full SHA so candidate qualification cannot race a branch."""

    if value is None or len(value) != 40:
        raise ValueError("candidate qualification requires a full 40-character SHA")
    try:
        int(value, 16)
    except ValueError as exception:
        raise ValueError(
            "candidate qualification requires a hexadecimal SHA"
        ) from exception


def _safe_model_name(model_id: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in model_id)


def _self_safe_process_pattern(value: str) -> str:
    """Return a regex that matches ``value`` without matching its own argv."""

    if not value:
        raise ValueError("process pattern cannot be empty")
    escaped = re.escape(value)
    return f"[{escaped[0]}]{escaped[1:]}"


def _runpod_deadline_teardown(
    *,
    client: RunPodClient,
    pod_id: str,
    fired: threading.Event,
    cancelled: threading.Event,
    errors: list[Exception],
    teardown_lock: threading.Lock,
) -> None:
    """Terminate a cost-bearing pod when the configured wall clock expires."""

    fired.set()
    try:
        with teardown_lock:
            if cancelled.is_set():
                return
            client.terminate_and_confirm(pod_id)
    except Exception as exception:  # noqa: BLE001 - relayed to the main report
        errors.append(exception)
