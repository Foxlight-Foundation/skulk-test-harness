"""Fail-safe fresh-install qualification lifecycle."""

from __future__ import annotations

import hashlib
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import BinaryIO

import httpx

from skulk_test_harness.client import SkulkClient
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
)
from skulk_test_harness.qualification_checks import (
    assert_fresh_runtime_contract,
    qualify_direct_text,
    qualify_direct_vision,
)
from skulk_test_harness.reporting import ReportWriter
from skulk_test_harness.runpod import RunPodClient
from skulk_test_harness.target_control import (
    OriginalTargetState,
    RecoverySnapshot,
    SshTargetController,
)
from skulk_test_harness.vision_fixture import generate_vision_fixture


class QualificationInterruptedError(RuntimeError):
    """Raised when a termination signal requests orderly restoration."""


class QualificationSignalGuard(AbstractContextManager["QualificationSignalGuard"]):
    """Convert SIGINT/SIGTERM into exceptions so lifecycle ``finally`` runs."""

    def __init__(self) -> None:
        self._previous: dict[int, object] = {}

    def __enter__(self) -> "QualificationSignalGuard":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)  # pyright: ignore[reportArgumentType]

    @staticmethod
    def _handle(signum: int, _frame: FrameType | None) -> None:
        raise QualificationInterruptedError(
            f"fresh-install qualification interrupted by signal {signum}"
        )


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
        with QualificationSignalGuard():
            if target.kind == "runpod":
                return self._qualify_runpod(
                    target=target,
                    profile=profile,
                    expected_commit=expected_commit,
                    report=report,
                    journal=journal,
                    artifact_directory=artifact_directory,
                )
            return self._qualify_physical(
                target=target,
                profile=profile,
                expected_commit=expected_commit,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
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
            )
        except Exception as exception:  # noqa: BLE001 - lifecycle failure boundary
            heartbeat_failed = isinstance(exception, LeaseHeartbeatError)
            report.issues.append(
                Issue(severity="error", message=f"fresh-install leg failed: {exception}")
            )
        finally:
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
            report = report.finish(
                passed=(
                    not report.issues
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
    ) -> FreshInstallQualificationReport:
        if self.fresh.runpod is None:
            raise ValueError("runpod target selected without fresh_install.runpod")
        pod_id: str | None = None
        teardown_succeeded = False
        deadline_timer: threading.Timer | None = None
        deadline_fired = threading.Event()
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
                        "errors": deadline_errors,
                        "teardown_lock": teardown_lock,
                    },
                )
                deadline_timer.daemon = True
                deadline_timer.start()
                endpoint = runpod.wait_for_ssh(pod_id)
            ephemeral_target = FreshInstallTarget(
                kind="physical",
                platform="nvidia",
                hardware_class=target.hardware_class,
                eligible=True,
                ssh_host=endpoint.host,
                ssh_user="root",
                ssh_port=endpoint.port,
                ssh_identity_file=self.fresh.runpod.ssh_private_key_file,
                service_manager="command",
                service_stop_command="true",
                service_start_command="true",
                isolation_enter_command="true",
                isolation_exit_command="true",
                expected_backends=target.expected_backends,
                expected_data_transport=target.expected_data_transport,
                vision_contract=target.vision_contract,
                text_models=target.text_models,
                vision_models=target.vision_models,
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
                )
            finally:
                _terminate_process(tunnel)
        except Exception as exception:  # noqa: BLE001 - provider lifecycle boundary
            report.issues.append(
                Issue(severity="error", message=f"RunPod qualification failed: {exception}")
            )
        finally:
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
                deadline_timer.join(timeout=1)
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
            report = report.finish(
                passed=not report.issues and teardown_succeeded
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
                start_command = _clean_environment_command(
                    temporary_root,
                    'cd "$HOME/skulk" && exec uv run skulk',
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
                target=target,
                report=report,
                journal=journal,
                artifact_directory=artifact_directory,
                heartbeat=heartbeat,
            )
        finally:
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
        target: FreshInstallTarget,
        report: FreshInstallQualificationReport,
        journal: _LifecycleJournal,
        artifact_directory: Path,
        heartbeat: AuthoritativeLeaseHeartbeat | None,
    ) -> None:
        models = list(dict.fromkeys([*target.text_models, *target.vision_models]))
        if not models:
            raise ValueError("fresh-install target has no qualification models")
        dashboard = DashboardQualifier(
            api_base_url=api_base_url,
            artifact_directory=artifact_directory / "playwright",
            poll_interval_s=self.fresh.poll_interval_s,
            model_ready_timeout_s=self.fresh.model_ready_timeout_s,
            abort_check=lambda: _check_heartbeat(heartbeat),
        )
        with SkulkClient(
            api_base_url,
            request_timeout_s=self.config.request_timeout_s,
            generation_timeout_s=self.config.generation_timeout_s,
            stream_read_timeout_s=self.config.stream_read_timeout_s,
        ) as client:
            for model_id in models:
                with journal.stage(f"dashboard user journey: {model_id}"):
                    browser_fixture = (
                        generate_vision_fixture()
                        if model_id in target.vision_models
                        else None
                    )
                    outcome = dashboard.qualify(
                        model_id=model_id,
                        vision_contract=(
                            "positive"
                            if model_id in target.vision_models
                            else target.vision_contract
                        ),
                        fixture=browser_fixture,
                    )
                    report.browser.append(outcome)
                    journal.persist()
                    if not outcome.passed:
                        raise RuntimeError(
                            outcome.message
                            or f"dashboard journey failed for {model_id}"
                        )
                    _check_heartbeat(heartbeat)

                with journal.stage(f"direct API parity: {model_id}"):
                    if not qualify_direct_text(client, model_id=model_id):
                        raise RuntimeError(
                            f"direct text API parity failed for {model_id}"
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
                        )
                        report.api_vision.append(evidence)
                        journal.persist()
                        if not evidence.passed:
                            raise RuntimeError(
                                f"direct vision API parity failed for {model_id}"
                            )
                    _check_heartbeat(heartbeat)

                with journal.stage(f"stop temporary model placement: {model_id}"):
                    for placement in client.find_placements_for_model(model_id):
                        if placement.instance_id:
                            client.delete_instance(placement.instance_id)
                    _wait_for_no_placement(
                        client,
                        model_id=model_id,
                        timeout_s=180,
                        poll_interval_s=self.fresh.poll_interval_s,
                        heartbeat=heartbeat,
                    )

    def _restore_physical(
        self,
        *,
        controller: SshTargetController,
        target: FreshInstallTarget,
        original: OriginalTargetState,
        api_base_url: str,
        journal: _LifecycleJournal,
    ) -> bool:
        with journal.stage("restore original selected-target service"):
            assert target.service_start_command is not None
            controller.run(target.service_start_command, timeout_s=120)
            node_id, node_count = _wait_for_api_identity(
                api_base_url,
                timeout_s=self.fresh.readiness_timeout_s,
                poll_interval_s=self.fresh.poll_interval_s,
                minimum_node_count=original.cluster_node_count,
            )
            mismatches = controller.verify_restored_state(
                original,
                api_node_id=node_id,
                cluster_node_count=node_count,
            )
            if mismatches:
                raise RuntimeError("; ".join(mismatches))
        return True


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
    heartbeat: AuthoritativeLeaseHeartbeat | None,
) -> InstallProvenance:
    """Poll until startup telemetry proves every shipped runtime invariant."""

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    with SkulkClient(api_base_url) as client:
        while time.monotonic() < deadline:
            _check_heartbeat(heartbeat)
            try:
                return assert_fresh_runtime_contract(
                    client,
                    expected_backends=target.expected_backends,
                    expected_transport=target.expected_data_transport,
                    expected_commit=expected_commit,
                )
            except Exception as exception:  # noqa: BLE001 - startup is eventually consistent
                last_error = exception
                time.sleep(poll_interval_s)
    raise TimeoutError(f"fresh runtime contract did not settle: {last_error}")


def _wait_for_api_identity(
    api_base_url: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    minimum_node_count: int | None = None,
) -> tuple[str, int]:
    """Wait for API identity and return its observed cluster size."""

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
            node_count = len(ids)
            if minimum_node_count is None or node_count >= minimum_node_count:
                return node_id, node_count
        except Exception:  # noqa: BLE001 - service is starting
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


def _wait_for_no_placement(
    client: SkulkClient,
    *,
    model_id: str,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat: AuthoritativeLeaseHeartbeat | None,
) -> None:
    """Wait until one temporary model has no remaining instances."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_heartbeat(heartbeat)
        if not client.find_placements_for_model(model_id):
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"temporary placement did not stop: {model_id}")


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
    errors: list[Exception],
    teardown_lock: threading.Lock,
) -> None:
    """Terminate a cost-bearing pod when the configured wall clock expires."""

    fired.set()
    try:
        with teardown_lock:
            client.terminate_and_confirm(pod_id)
    except Exception as exception:  # noqa: BLE001 - relayed to the main report
        errors.append(exception)
