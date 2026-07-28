"""Fail-safe fresh-install qualification lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Literal

import httpx

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
    FreshInstallLifecycleStage,
    FreshInstallProfile,
    FreshInstallQualificationReport,
    FreshInstallTarget,
    HarnessConfig,
    InstallProvenance,
    Issue,
    RunPodFreshInstallConfig,
    ServedEngineContract,
    ServedEngineEvidence,
)
from skulk_test_harness.qualification_checks import (
    UnexpectedFreshInstallPeerError,
    assert_fresh_runtime_contract,
    assert_fresh_single_node,
    qualify_direct_text,
    qualify_direct_vision,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.runpod import RunPodClient, RunPodSshEndpoint
from skulk_test_harness.target_control import (
    OriginalTargetState,
    RecoverySnapshot,
    SshTargetController,
)
from skulk_test_harness.vision_fixture import generate_vision_fixture


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
        expected_node_id: str,
        poll_interval_s: float,
        request_timeout_s: float,
    ) -> None:
        self.expected_node_id = expected_node_id
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
                    assert_fresh_single_node(
                        client,
                        expected_node_id=self.expected_node_id,
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

    def qualify_target(
        self,
        *,
        target_name: str,
        target: FreshInstallTarget,
        profile: FreshInstallProfile,
        expected_commit: str | None,
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
        store = FleetLockStore(self.config.fleet_lock)
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
                Issue(severity="error", message=f"fresh-install leg failed: {exception}")
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
                        original=snapshot.original,
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
                    Issue(severity="error", message=f"lease heartbeat failed: {exception}")
                )
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
            if acquired and restoration_succeeded and not heartbeat_failed:
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
                    heartbeat=None,
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
                Issue(severity="error", message=f"RunPod qualification failed: {exception}")
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
                result = controller.run(
                    "umask 077; "
                    "root=$(mktemp -d /tmp/skulk-fresh.XXXXXX); "
                    'mkdir -p "$root/home" "$root/tmp"; '
                    'printf "%s" "$root"',
                    timeout_s=30,
                )
                temporary_root = result.stdout.strip()
                if not temporary_root.startswith("/tmp/skulk-fresh."):
                    raise RuntimeError("target returned an unsafe temporary root")

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
                    raise RuntimeError("installer resolved a different candidate commit")
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
        with SkulkClient(
            api_base_url,
            request_timeout_s=self.config.request_timeout_s,
            generation_timeout_s=self.config.generation_timeout_s,
            stream_read_timeout_s=self.config.stream_read_timeout_s,
        ) as client, _FreshRuntimeMonitor(
            api_base_url=api_base_url,
            expected_node_id=assert_fresh_single_node(client),
            poll_interval_s=self.fresh.poll_interval_s,
            request_timeout_s=self.config.request_timeout_s,
        ) as runtime_monitor:
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
                    with journal.stage(
                        f"served engine runtime contract: {model_id}"
                    ):
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
                                f"the model replied: {evidence.response_excerpt!r}"
                            )
                    check_fresh_runtime()

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
        original: OriginalTargetState,
        api_base_url: str,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
    ) -> bool:
        with journal.stage("restore original selected-target service") as stage:
            assert target.service_start_command is not None
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
    before_pid, observed_parallel, kv_unified_observed = (
        _llama_server_process_contract(
        before,
        installation_root=installation_root,
        )
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

    candidates: list[tuple[int, list[str]]] = []
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
            candidates.append((pid, arguments))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one fresh llama-server child, observed "
            f"{len(candidates)}"
        )
    pid, arguments = candidates[0]
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
    return pid, parallel, "--kv-unified" in arguments


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
    report: FreshInstallQualificationReport,
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
                    expected_backends=target.expected_backends,
                    expected_transport=target.expected_data_transport,
                    expected_commit=expected_commit,
                    dashboard_contract=target.dashboard_contract,
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


_COMPLETED_DOWNLOAD_STATES = frozenset(
    {"complete", "completed", "ready", "succeeded"}
)
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
        raise RuntimeError(f"no viable placement preview was offered for {model_id}: {offered}")
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


def _require_commit_sha(value: str | None) -> None:
    """Require a full SHA so candidate qualification cannot race a branch."""

    if value is None or len(value) != 40:
        raise ValueError("candidate qualification requires a full 40-character SHA")
    try:
        int(value, 16)
    except ValueError as exception:
        raise ValueError("candidate qualification requires a hexadecimal SHA") from exception


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
