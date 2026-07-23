"""Unit coverage for the release-blocking fresh-install lifecycle."""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, cast

import httpx
import pytest
from pydantic import ValidationError

import skulk_test_harness.fresh_install as fresh_install_module
from skulk_test_harness.dashboard_qualification import (
    _captured_image_digest,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.fleet_lock import FleetLease, LeaseOutcome
from skulk_test_harness.fresh_install import (
    QualificationInterruptedError,
    QualificationSignalGuard,
    _clean_environment_command,  # pyright: ignore[reportPrivateUsage]
    _installer_command,  # pyright: ignore[reportPrivateUsage]
    _run_remote_logged_command,  # pyright: ignore[reportPrivateUsage]
    _self_safe_process_pattern,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.lease_heartbeat import (
    AuthoritativeLeaseHeartbeat,
    LeaseHeartbeatError,
)
from skulk_test_harness.models import (
    FleetLock,
    FreshInstallConfig,
    FreshInstallQualificationReport,
    FreshInstallTarget,
    HarnessConfig,
    InstallProvenance,
    RunPodFreshInstallConfig,
)
from skulk_test_harness.runpod import RunPodClient
from skulk_test_harness.target_control import (
    OriginalTargetState,
    RecoverySnapshot,
    SshTargetController,
    _snapshot_command,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.vision_fixture import (
    data_url_sha256,
    generate_vision_fixture,
)


def _physical_target(*, eligible: bool = True) -> FreshInstallTarget:
    return FreshInstallTarget(
        kind="physical",
        platform="apple",
        hardware_class="apple-silicon-32gb",
        eligible=eligible,
        ssh_host="private-alias",
        service_stop_command="stop selected service",
        service_start_command="start selected service",
        isolation_enter_command="isolate selected target",
        isolation_exit_command="restore selected target network",
        expected_backends=["mlx"],
        vision_contract="positive",
        text_models=["mlx-community/Qwen3.5-2B-4bit"],
        vision_models=["mlx-community/Qwen3.5-2B-4bit"],
    )


def test_target_selection_uses_only_explicit_eligibility() -> None:
    config = FreshInstallConfig(
        targets={
            "eligible": _physical_target(),
            "excluded": _physical_target(eligible=False),
        }
    )

    assert [name for name, _target in config.eligible_targets()] == ["eligible"]
    with pytest.raises(ValueError, match="not eligible"):
        config.eligible_targets(["excluded"])
    with pytest.raises(ValueError, match="unknown"):
        config.eligible_targets(["incidental-fabric-node"])


def test_complete_release_matrix_requires_every_blocking_platform() -> None:
    config = FreshInstallConfig(targets={"apple": _physical_target()})
    selected = config.eligible_targets()

    with pytest.raises(ValueError, match="amd.*nvidia"):
        config.assert_complete_release_matrix(selected)


def test_target_contract_rejects_adaptive_vision_skip() -> None:
    with pytest.raises(ValidationError, match="positive vision"):
        FreshInstallTarget(
            kind="physical",
            platform="apple",
            hardware_class="apple-silicon",
            eligible=True,
            ssh_host="alias",
            service_stop_command="stop",
            service_start_command="start",
            isolation_enter_command="isolate",
            isolation_exit_command="restore network",
            expected_backends=["mlx"],
            vision_contract="positive",
        )
    with pytest.raises(ValidationError, match="cannot list vision_models"):
        FreshInstallTarget(
            kind="runpod",
            platform="nvidia",
            hardware_class="cuda",
            eligible=True,
            expected_backends=["llama_server", "llama_server-cuda"],
            vision_contract="unavailable",
            vision_models=["not-allowed"],
        )
    with pytest.raises(ValidationError, match="reversible Skulk-network isolation"):
        FreshInstallTarget(
            kind="physical",
            platform="amd",
            hardware_class="amd-linux",
            eligible=True,
            ssh_host="alias",
            service_stop_command="stop",
            service_start_command="start",
            expected_backends=["llama_server"],
            vision_contract="unavailable",
            text_models=["unsloth/Llama-3.2-1B-Instruct-GGUF"],
        )


def test_heartbeat_must_not_exceed_one_third_of_ttl() -> None:
    with pytest.raises(ValidationError, match="one third"):
        FreshInstallConfig(lease_ttl_s=90, lease_heartbeat_s=31)
    assert FreshInstallConfig(lease_ttl_s=90).resolved_lease_heartbeat_s == 30


def test_random_vision_fixture_has_exact_judge_free_contract(tmp_path: Path) -> None:
    first = generate_vision_fixture()
    second = generate_vision_fixture()

    assert first.sha256 != second.sha256
    assert first.code != second.code
    assert first.code not in first.prompt
    assert data_url_sha256(first.data_url) == first.sha256
    assert first.response_matches(
        f"{first.code}\n{first.color} {first.shape}"
    ) == (True, True)
    assert first.response_matches("a plausible blue bedroom") == (False, False)
    path = tmp_path / "fixture.png"
    first.write(path)
    assert path.stat().st_mode & 0o777 == 0o600


def test_captured_dashboard_request_must_contain_exact_fixture() -> None:
    fixture = generate_vision_fixture()
    digest = _captured_image_digest(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": fixture.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": fixture.data_url},
                            },
                        ],
                    }
                ]
            }
        ]
    )
    assert digest == fixture.sha256


class _LeaseStore:
    def __init__(self) -> None:
        self.current = _held_lease(seconds=60)

    def read(self) -> FleetLease:
        return self.current

    def extend(self, *, ttl_s: float | None = None) -> LeaseOutcome:
        self.current = _held_lease(seconds=ttl_s or 60)
        return LeaseOutcome(True, self.current, "extended")


class _StaleAuthoritativeLeaseStore(_LeaseStore):
    def extend(self, *, ttl_s: float | None = None) -> LeaseOutcome:
        written = _held_lease(seconds=ttl_s or 60)
        self.current = _held_lease(seconds=5)
        return LeaseOutcome(True, written, "locally extended")


def _held_lease(*, seconds: float) -> FleetLease:
    now = datetime.now(UTC)
    return FleetLease(
        state="held",
        holder="codex",
        acquired_at=now.isoformat(),
        heartbeat_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=seconds)).isoformat(),
    )


def test_lease_renewal_rereads_authoritative_expiry() -> None:
    store = _LeaseStore()
    observed: list[datetime] = []
    heartbeat = AuthoritativeLeaseHeartbeat(
        store,
        holder="codex",
        ttl_s=120,
        interval_s=40,
        on_verified_expiry=observed.append,
    )

    renewed = heartbeat.renew_once()

    assert renewed == store.current
    assert observed[-1] == store.current.expiry()


def test_lease_renewal_rejects_stale_authoritative_record() -> None:
    heartbeat = AuthoritativeLeaseHeartbeat(
        _StaleAuthoritativeLeaseStore(),
        holder="codex",
        ttl_s=120,
        interval_s=40,
    )

    with pytest.raises(LeaseHeartbeatError, match="did not reflect"):
        heartbeat.renew_once()


def test_install_commands_pin_candidate_and_preserve_literal_shipping() -> None:
    sha = "a" * 40
    shipping = _installer_command(
        installer_url=(
            "https://raw.githubusercontent.com/"
            "Foxlight-Foundation/Skulk/main/install.sh"
        ),
        profile="shipping",
        expected_commit=None,
    )
    candidate = _installer_command(
        installer_url=f"https://example.invalid/{sha}/install.sh",
        profile="candidate",
        expected_commit=sha,
    )
    clean = _clean_environment_command("/tmp/skulk-fresh.abc123", candidate)

    assert shipping == (
        "curl -fsSL https://raw.githubusercontent.com/"
        "Foxlight-Foundation/Skulk/main/install.sh | bash"
    )
    assert candidate.endswith(f"| bash -s -- --ref {sha}")
    assert "env -i" in clean
    assert "SKULK_" not in clean


def test_cleanup_process_pattern_cannot_match_its_own_command_text() -> None:
    temporary_checkout = "/tmp/skulk-fresh.abc123/home/skulk"
    pattern = _self_safe_process_pattern(temporary_checkout)

    assert re.fullmatch(pattern, temporary_checkout)
    assert temporary_checkout not in pattern


def test_remote_installer_aborts_immediately_after_heartbeat_failure(
    tmp_path: Path,
) -> None:
    processes: list[subprocess.Popen[bytes]] = []

    class FailingHeartbeat:
        def raise_if_failed(self) -> None:
            raise LeaseHeartbeatError("authoritative renewal failed")

    class FakeController:
        def start(
            self,
            _command: str,
            *,
            log_path: Path,
        ) -> tuple[subprocess.Popen[bytes], BinaryIO]:
            log_handle = log_path.open("wb")
            log_path.chmod(0o600)
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            return process, log_handle

    log_path = tmp_path / "installer.log"
    with pytest.raises(LeaseHeartbeatError, match="renewal failed"):
        _run_remote_logged_command(
            controller=cast(SshTargetController, FakeController()),
            command="install",
            log_path=log_path,
            timeout_s=30,
            poll_interval_s=0.001,
            heartbeat=cast(
                AuthoritativeLeaseHeartbeat,
                FailingHeartbeat(),
            ),
        )

    assert processes[0].poll() is not None
    assert log_path.stat().st_mode & 0o777 == 0o600


def test_install_provenance_has_no_private_inventory_fields() -> None:
    payload = InstallProvenance(
        mode="fresh_install",
        environment="fresh_install",
        profile="candidate",
        platform="apple",
        hardware_class="apple-silicon-32gb",
        environment_override_names=[],
    ).model_dump()

    serialized = json.dumps(payload)
    assert "ssh" not in serialized.lower()
    assert "node_name" not in serialized
    assert "private_path" not in serialized


def test_signal_guard_turns_sigterm_into_recoverable_exception() -> None:
    with pytest.raises(QualificationInterruptedError, match=str(signal.SIGTERM)):
        QualificationSignalGuard._handle(signal.SIGTERM, None)  # pyright: ignore[reportPrivateUsage]


def test_recovery_snapshot_is_mode_600_and_contains_manifest_and_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("model_store: {}\n")
    manifest = base64.b64encode(b'{"git_commit":"abc"}').decode()
    command = _snapshot_command(
        qualification_id="qualification",
        encoded_manifest=manifest,
        config_paths=json.dumps([str(config_path)]),
        retention_days=30,
    )

    result = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    archive_path = Path(json.loads(result.stdout)["path"])

    assert archive_path.stat().st_mode & 0o777 == 0o600
    with tarfile.open(archive_path) as archive:
        assert sorted(archive.getnames()) == [
            "recovery",
            "recovery/config-0",
            "recovery/manifest.json",
        ]


def test_restoration_verification_detects_every_changed_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={"/config": "digest-a"},
        process_arguments=["uv run skulk"],
        service_status="exit=0\nrunning",
        api_node_id="node-a",
        cluster_node_count=3,
    )
    changed = dataclasses.replace(
        original,
        git_commit="commit-b",
        git_status="dirty",
        config_sha256={"/config": "digest-b"},
        process_arguments=["different command"],
        service_status="exit=1\nstopped",
        api_node_id="node-b",
        cluster_node_count=2,
    )
    controller = SshTargetController(_physical_target())
    monkeypatch.setattr(
        controller,
        "capture_original_state",
        lambda **_kwargs: changed,
    )

    mismatches = controller.verify_restored_state(
        original,
        api_node_id="node-b",
        cluster_node_count=2,
    )

    assert mismatches == [
        "original checkout commit changed",
        "original checkout status changed",
        "original configuration hash changed",
        "original process arguments were not restored",
        "original API identity was not restored",
        "original fleet membership did not rejoin",
        "original service manager state was not restored",
    ]


def _run_failed_physical_lifecycle(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restoration_mismatches: list[str],
) -> tuple[FreshInstallQualificationReport, list[str], list[float], list[bool]]:
    lease = _held_lease(seconds=3600)
    extensions: list[float] = []
    releases: list[bool] = []

    class FakeStore:
        current = lease

        def acquire(self, **_kwargs: object) -> LeaseOutcome:
            return LeaseOutcome(True, self.current, "acquired")

        def read(self) -> FleetLease:
            return self.current

        def extend(self, *, ttl_s: float | None = None) -> LeaseOutcome:
            extensions.append(ttl_s or 0)
            self.current = _held_lease(seconds=ttl_s or 60)
            return LeaseOutcome(True, self.current, "extended")

        def release(self) -> LeaseOutcome:
            releases.append(True)
            return LeaseOutcome(True, FleetLease(), "released")

    commands: list[str] = []
    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={"/config": "digest"},
        process_arguments=["uv run skulk"],
        service_status="exit=0\nrunning",
        api_node_id="node-a",
        cluster_node_count=3,
    )

    class FakeController:
        def __init__(self, _target: FreshInstallTarget) -> None:
            pass

        def open_tunnel(self, *, remote_port: int) -> tuple[int, object]:
            assert remote_port == 52415
            return 12345, object()

        def capture_recovery_snapshot(self, **_kwargs: object) -> RecoverySnapshot:
            return RecoverySnapshot(
                remote_path="/private/recovery.tar.gz",
                remote_sha256="digest",
                controller_path=tmp_path / "recovery.tar.gz",
                controller_sha256="digest",
                original=original,
            )

        def run(
            self,
            command: str,
            *,
            timeout_s: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            del timeout_s, check
            commands.append(command)
            return subprocess.CompletedProcess([], 0, "", "")

        def verify_restored_state(
            self,
            _original: OriginalTargetState,
            *,
            api_node_id: str | None,
            cluster_node_count: int | None,
        ) -> list[str]:
            assert api_node_id == "node-a"
            assert cluster_node_count == 3
            return restoration_mismatches

    store = FakeStore()
    monkeypatch.setattr(fresh_install_module, "FleetLockStore", lambda _config: store)
    monkeypatch.setattr(fresh_install_module, "SshTargetController", FakeController)

    class FakeSkulkClient:
        def __init__(self, _base_url: str) -> None:
            pass

        def __enter__(self) -> "FakeSkulkClient":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def get_diagnostics_node(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(fresh_install_module, "SkulkClient", FakeSkulkClient)
    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_api_identity",
        lambda *_args, **_kwargs: ("node-a", 3),
    )
    monkeypatch.setattr(
        fresh_install_module,
        "_terminate_process",
        lambda _process: None,
    )

    def fail_browser_boundary(
        _self: object,
        **_kwargs: object,
    ) -> None:
        raise RuntimeError("forced browser boundary failure")

    monkeypatch.setattr(
        fresh_install_module.FreshInstallQualifier,
        "_execute_clean_install",
        fail_browser_boundary,
    )
    target = _physical_target()
    config = HarnessConfig(
        output_dir=tmp_path / "runs",
        fleet_lock=FleetLock(remote="private", holder="codex"),
        fresh_install=FreshInstallConfig(
            targets={"apple": target},
            snapshot_root=tmp_path / "snapshots",
            lease_ttl_s=90,
            lease_heartbeat_s=30,
            emergency_lease_ttl_s=600,
        ),
    )
    report = fresh_install_module.FreshInstallQualifier(config).qualify_target(
        target_name="apple",
        target=target,
        profile="candidate",
        expected_commit="a" * 40,
    )
    return report, commands, extensions, releases


def test_browser_failure_restores_service_then_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, commands, extensions, releases = _run_failed_physical_lifecycle(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        restoration_mismatches=[],
    )

    assert commands == [
        "stop selected service",
        "isolate selected target",
        "restore selected target network",
        "start selected service",
    ]
    assert report.restoration_succeeded is True
    assert report.critical_recovery_required is False
    assert report.passed is False
    assert extensions == []
    assert releases == [True]


def test_restore_failure_emergency_extends_and_leaves_lease_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, commands, extensions, releases = _run_failed_physical_lifecycle(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        restoration_mismatches=["original process arguments were not restored"],
    )

    assert commands == [
        "stop selected service",
        "isolate selected target",
        "restore selected target network",
        "start selected service",
    ]
    assert report.restoration_succeeded is False
    assert report.critical_recovery_required is True
    assert report.passed is False
    assert extensions == [600]
    assert releases == []


def test_runpod_is_clean_cost_bounded_and_teardown_is_confirmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_key = tmp_path / "id.pub"
    private_key = tmp_path / "id"
    public_key.write_text("ssh-ed25519 AAAATEST qualification")
    private_key.write_text("private")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["volumeInGb"] == 0
            assert "networkVolumeId" not in body
            assert body["imageName"] == "nvidia/cuda-node-neutral"
            return httpx.Response(
                201,
                json={
                    "id": "pod-1",
                    "adjustedCostPerHr": 1.25,
                    "networkVolume": None,
                },
            )
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(204)
        return httpx.Response(404)

    config = RunPodFreshInstallConfig(
        ssh_public_key_file=public_key,
        ssh_private_key_file=private_key,
        image_name="nvidia/cuda-node-neutral",
        gpu_type_ids=["NVIDIA L4"],
        maximum_hourly_cost_usd=2,
        poll_interval_s=0.001,
        readiness_timeout_s=0.01,
    )
    http_client = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        transport=httpx.MockTransport(handler),
    )
    client = RunPodClient(config, client=http_client)

    lease = client.provision(qualification_id="qualification")
    client.terminate_and_confirm(lease.pod_id)

    assert lease.hourly_cost_usd == 1.25
    assert deleted == ["/v1/pods/pod-1"]


def test_runpod_rejects_over_ceiling_pod_only_after_confirmed_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_key = tmp_path / "id.pub"
    private_key = tmp_path / "id"
    public_key.write_text("ssh-ed25519 AAAATEST qualification")
    private_key.write_text("private")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    probes = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal probes
        if request.method == "POST":
            return httpx.Response(
                201,
                json={"id": "pod-costly", "adjustedCostPerHr": 5.0},
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        probes += 1
        if probes == 1:
            return httpx.Response(200, json={"desiredStatus": "TERMINATED"})
        return httpx.Response(404)

    config = RunPodFreshInstallConfig(
        ssh_public_key_file=public_key,
        ssh_private_key_file=private_key,
        image_name="nvidia/cuda-node-neutral",
        gpu_type_ids=["NVIDIA L4"],
        maximum_hourly_cost_usd=2,
        poll_interval_s=0.001,
        readiness_timeout_s=0.01,
    )
    http_client = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        transport=httpx.MockTransport(handler),
    )
    client = RunPodClient(config, client=http_client)

    with pytest.raises(RuntimeError, match="ceiling"):
        client.provision(qualification_id="qualification")

    assert probes == 2
