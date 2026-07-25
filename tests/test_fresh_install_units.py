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
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import httpx
import pytest
from playwright.sync_api import Page
from pydantic import ValidationError

import skulk_test_harness.dashboard_qualification as dashboard_qualification_module
import skulk_test_harness.fresh_install as fresh_install_module
import skulk_test_harness.qualification_checks as qualification_checks_module
from skulk_test_harness import runpod as runpod_module
from skulk_test_harness.client import SkulkClient
from skulk_test_harness.dashboard_qualification import (
    DashboardQualifier,
    _captured_image_digest,  # pyright: ignore[reportPrivateUsage]
    _JourneyProgress,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.echo_phrase import echo_matched, echo_phrase, echo_prompt
from skulk_test_harness.fleet_lock import FleetLease, LeaseOutcome
from skulk_test_harness.fresh_install import (
    QualificationInterruptedError,
    QualificationSignalGuard,
    _browser_vision_expectation,  # pyright: ignore[reportPrivateUsage]
    _clean_environment_command,  # pyright: ignore[reportPrivateUsage]
    _installer_command,  # pyright: ignore[reportPrivateUsage]
    _provision_model_over_api,  # pyright: ignore[reportPrivateUsage]
    _run_remote_logged_command,  # pyright: ignore[reportPrivateUsage]
    _self_safe_process_pattern,  # pyright: ignore[reportPrivateUsage]
    _wait_for_api_identity,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.lease_heartbeat import (
    AuthoritativeLeaseHeartbeat,
    LeaseHeartbeatError,
)
from skulk_test_harness.models import (
    DashboardContract,
    FleetLock,
    FreshInstallConfig,
    FreshInstallQualificationReport,
    FreshInstallTarget,
    HarnessConfig,
    InstallProvenance,
    PlacementResult,
    RunPodFreshInstallConfig,
)
from skulk_test_harness.qualification_checks import qualify_direct_text
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


def test_direct_text_check_matches_dashboard_thinking_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def stream_chat(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(text="amber harbor 4821")

    monkeypatch.setattr(
        qualification_checks_module,
        "echo_phrase",
        lambda: "amber harbor 4821",
    )

    assert qualify_direct_text(
        cast(SkulkClient, FakeClient()),
        model_id="org/toggle-model",
        enable_thinking=False,
    )
    assert calls == [
        {
            "model_id": "org/toggle-model",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Repeat this phrase back exactly and say nothing else: "
                        "amber harbor 4821"
                    ),
                }
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
        }
    ]


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
        expected_commit=sha,
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
        # A node that never answered at all reports no identity. That is the
        # identity failure worth catching; a *different* identity is what a
        # healthy restart always produces and is asserted separately.
        api_node_id=None,
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
        api_node_id=None,
        cluster_node_count=2,
    )

    # The smaller fleet is deliberately absent: it counts nodes this leg never
    # touched, so it is reported as a warning by the caller rather than as a
    # restoration failure. See the shrunk-fleet test below.
    assert mismatches == [
        "original checkout commit changed",
        "original checkout status changed",
        "original configuration hash changed",
        "original process arguments were not restored",
        "restored node did not report an API identity",
        "original service manager state was not restored",
    ]


def test_restoration_passes_when_only_the_surrounding_fleet_shrank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered target must not fail because a peer outside the leg is down.

    A leg stops and starts exactly one node, so every other member is outside
    the experiment. A peer that is rebooting, still pruning out of the pre-run
    reading, or deliberately quiet for the duration lowers the count while this
    target recovered perfectly. Failing here held the fleet lease on a machine
    whose service, checkout, configuration, and arguments were all restored.
    """

    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={"/config": "digest-a"},
        process_arguments=["uv run skulk"],
        service_status="exit=0\nrunning",
        api_node_id="node-a",
        cluster_node_count=2,
    )
    restored = dataclasses.replace(
        original,
        api_node_id="node-b",
        cluster_node_count=1,
    )
    controller = SshTargetController(_physical_target())
    monkeypatch.setattr(
        controller,
        "capture_original_state",
        lambda **_kwargs: restored,
    )

    mismatches = controller.verify_restored_state(
        original,
        api_node_id="node-b",
        cluster_node_count=1,
    )

    assert mismatches == []


def test_restoration_accepts_the_new_identity_a_restart_always_produces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully restored node reports a different node id and must still pass.

    Skulk regenerates its node identity on every process start and never
    persists it, so stopping and starting the service guarantees a new one.
    Comparing identities for equality failed every physical leg on a machine
    that had actually recovered.
    """

    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={"/config": "digest-a"},
        process_arguments=["uv run skulk"],
        service_status="exit=0\nrunning",
        api_node_id="node-before-restart",
        cluster_node_count=5,
    )
    restored = dataclasses.replace(original, api_node_id="node-after-restart")
    controller = SshTargetController(_physical_target())
    monkeypatch.setattr(
        controller,
        "capture_original_state",
        lambda **_kwargs: restored,
    )

    mismatches = controller.verify_restored_state(
        original,
        api_node_id="node-after-restart",
        cluster_node_count=5,
    )

    assert mismatches == []


def test_identity_wait_returns_as_soon_as_the_target_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness is the target answering, not the fleet reaching a given size.

    A leg starts exactly one node, so waiting for a pre-run fleet count made
    readiness depend on peers the leg never touched. On a run where the rest of
    the fabric was deliberately quiet, that wait could never be satisfied and
    spent its whole window before failing a target that was already serving.
    """

    sleeps: list[float] = []
    attempts: list[int] = []

    class FakeClient:
        def __init__(self, _base_url: str) -> None:
            attempts.append(1)

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def get_node_id(self) -> str:
            if len(attempts) == 1:
                raise RuntimeError("service is still starting")
            return "node-a"

        def get_state(self) -> dict[str, object]:
            # One node, well below the size this fleet had before the run.
            return {"nodeIdentities": {"node-a": {}}, "nodeResources": {}}

    monkeypatch.setattr(fresh_install_module, "SkulkClient", FakeClient)
    monkeypatch.setattr(
        fresh_install_module.time,
        "sleep",
        sleeps.append,
    )

    assert _wait_for_api_identity(
        "http://127.0.0.1:52415",
        timeout_s=1,
        poll_interval_s=0.25,
    ) == ("node-a", 1)
    # Exactly one retry: it polled through the unavailable API and stopped the
    # moment an identity came back, without waiting on fleet size.
    assert sleeps == [0.25]


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


def test_runpod_deletes_created_pod_when_cost_metadata_is_missing(
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
            return httpx.Response(201, json={"id": "pod-unknown-cost"})
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

    with pytest.raises(RuntimeError, match="omitted hourly cost"):
        client.provision(qualification_id="qualification")

    assert deleted == ["/v1/pods/pod-unknown-cost"]


def test_cancelled_runpod_deadline_does_not_reenter_teardown() -> None:
    terminated: list[str] = []

    class FakeRunPodClient:
        def terminate_and_confirm(self, pod_id: str) -> None:
            terminated.append(pod_id)

    fired = threading.Event()
    cancelled = threading.Event()
    cancelled.set()
    errors: list[Exception] = []

    fresh_install_module._runpod_deadline_teardown(
        client=cast(RunPodClient, FakeRunPodClient()),
        pod_id="pod-already-terminating",
        fired=fired,
        cancelled=cancelled,
        errors=errors,
        teardown_lock=threading.Lock(),
    )

    assert fired.is_set()
    assert terminated == []
    assert errors == []


def test_runpod_ssh_readiness_waits_for_a_real_sshd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RUNNING pod whose sshd is still booting must not be handed back.

    RunPod publishes the port mapping as soon as the container starts, while
    the bootstrap is still installing and launching sshd. Trusting provider
    metadata alone returned an endpoint that refused the controller's tunnel
    and failed qualification on a pod that was merely slow to boot.
    """

    public_key = tmp_path / "id.pub"
    private_key = tmp_path / "id"
    public_key.write_text("ssh-ed25519 AAAATEST qualification")
    private_key.write_text("private")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "203.0.113.10",
                "portMappings": {"22": 22198},
            },
        )

    config = RunPodFreshInstallConfig(
        ssh_public_key_file=public_key,
        ssh_private_key_file=private_key,
        image_name="nvidia/cuda-node-neutral",
        gpu_type_ids=["NVIDIA L4"],
        poll_interval_s=0.001,
        readiness_timeout_s=0.05,
    )
    http_client = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        transport=httpx.MockTransport(handler),
    )
    client = RunPodClient(config, client=http_client)

    banners: list[bool] = [False, False, True]
    attempts = 0

    def fake_banner(host: str, port: int, **_kwargs: object) -> bool:
        nonlocal attempts
        assert (host, port) == ("203.0.113.10", 22198)
        result = banners[min(attempts, len(banners) - 1)]
        attempts += 1
        return result

    monkeypatch.setattr(runpod_module, "_ssh_banner_ready", fake_banner)

    endpoint = client.wait_for_ssh("pod-1")

    assert attempts == 3, "readiness must keep polling while sshd is refusing"
    assert (endpoint.host, endpoint.port) == ("203.0.113.10", 22198)


def test_runpod_ssh_readiness_times_out_when_sshd_never_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pod that never serves SSH fails on the deadline, not on first probe."""

    public_key = tmp_path / "id.pub"
    private_key = tmp_path / "id"
    public_key.write_text("ssh-ed25519 AAAATEST qualification")
    private_key.write_text("private")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "203.0.113.10",
                "portMappings": {"22": 22198},
            },
        )

    config = RunPodFreshInstallConfig(
        ssh_public_key_file=public_key,
        ssh_private_key_file=private_key,
        image_name="nvidia/cuda-node-neutral",
        gpu_type_ids=["NVIDIA L4"],
        poll_interval_s=0.001,
        readiness_timeout_s=0.02,
    )
    http_client = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        transport=httpx.MockTransport(handler),
    )
    client = RunPodClient(config, client=http_client)
    monkeypatch.setattr(
        runpod_module, "_ssh_banner_ready", lambda *_args, **_kwargs: False
    )

    with pytest.raises(TimeoutError, match="readiness deadline"):
        client.wait_for_ssh("pod-1")


def test_ssh_host_key_policy_is_strict_for_inventory_and_lenient_for_pods(
    tmp_path: Path,
) -> None:
    """Only an ephemeral pod may trust an unknown host key.

    A provider pod's host key is generated at boot and its address/port pairs
    are recycled between pods, so strict checking rejected the controller's
    very first connection. Real fleet hardware keeps strict checking, because
    there a changed host key is a signal worth stopping on.
    """

    inventory = _physical_target()
    ephemeral = FreshInstallTarget(
        kind="physical",
        platform="nvidia",
        hardware_class="nvidia-cuda",
        eligible=True,
        ssh_host="203.0.113.10",
        ssh_user="root",
        ssh_port=22198,
        ssh_identity_file=tmp_path / "pod-key",
        accept_unknown_host_key=True,
        service_manager="command",
        service_stop_command="true",
        service_start_command="true",
        isolation_enter_command="true",
        isolation_exit_command="true",
        expected_backends=["llama_server", "llama_server-cuda"],
        vision_contract="unavailable",
        text_models=["unsloth/Llama-3.2-1B-Instruct-GGUF"],
    )

    strict_prefix = SshTargetController(inventory)._ssh_prefix()  # pyright: ignore[reportPrivateUsage]
    lenient_prefix = SshTargetController(ephemeral)._ssh_prefix()  # pyright: ignore[reportPrivateUsage]
    lenient_scp = SshTargetController(ephemeral)._scp_prefix()  # pyright: ignore[reportPrivateUsage]

    assert "StrictHostKeyChecking=accept-new" not in strict_prefix
    assert "UserKnownHostsFile=/dev/null" not in strict_prefix
    assert "StrictHostKeyChecking=accept-new" in lenient_prefix
    assert "UserKnownHostsFile=/dev/null" in lenient_prefix
    assert "UserKnownHostsFile=/dev/null" in lenient_scp


class _StubContractClient:
    """Minimal API stand-in for the fresh-runtime contract assertions."""

    def __init__(self, reported_commit: str | None = None) -> None:
        self.base_url = "http://127.0.0.1:52415"
        self.request_timeout_s = 5.0
        self.reported_commit = reported_commit

    def get_state(self) -> dict[str, object]:
        """Return one node advertising the expected backend and transport."""

        return {
            "nodeResources": {
                "node-a": {
                    "backends": ["llama_server", "llama_server-vulkan"],
                    "dataTransport": "zenoh",
                }
            },
            "nodeIdentities": {"node-a": {}},
        }

    def get_diagnostics_node(self) -> dict[str, object]:
        """Return runtime provenance carrying this stub's reported commit."""

        return {"runtime": {"skulkCommit": self.reported_commit}}


@pytest.mark.parametrize(
    ("contract", "served", "expected_failure"),
    [
        ("required", True, None),
        ("required", False, "did not serve the production dashboard build"),
        ("absent", False, None),
        ("absent", True, "declared headless"),
    ],
)
def test_dashboard_contract_asserts_both_shipped_shapes(
    monkeypatch: pytest.MonkeyPatch,
    contract: str,
    served: bool,
    expected_failure: str | None,
) -> None:
    """A headless node is a shipped shape, and both outcomes are asserted.

    The installer skips the dashboard build on a target with no Node toolchain,
    so demanding the web UI everywhere fails a supported install. Declaring the
    absence keeps it a contract rather than a skip: an unexpectedly missing
    dashboard still fails, and so does one that appears where none should.
    """

    body = '<html><body><div id="root"></div></body></html>' if served else "not found"

    def fake_get(url: str, timeout: float | None = None) -> httpx.Response:
        return httpx.Response(200 if served else 404, text=body)

    monkeypatch.setattr(qualification_checks_module.httpx, "get", fake_get)
    client = cast(SkulkClient, _StubContractClient())

    if expected_failure is None:
        provenance = qualification_checks_module.assert_fresh_runtime_contract(
            client,
            expected_backends=["llama_server"],
            expected_transport="zenoh",
            expected_commit=None,
            dashboard_contract=cast(DashboardContract, contract),
        )
        assert provenance.dashboard_build_present is served
        return

    with pytest.raises(RuntimeError, match=expected_failure):
        qualification_checks_module.assert_fresh_runtime_contract(
            client,
            expected_backends=["llama_server"],
            expected_transport="zenoh",
            expected_commit=None,
            dashboard_contract=cast(DashboardContract, contract),
        )


@pytest.mark.parametrize(
    ("reported_commit", "matches"),
    [
        ("32fffb7", True),
        ("32fffb7a36f9872b361c20ef47888f452211a8b6", True),
        ("32FFFB7", True),
        ("32fffb8", False),
        ("32fff", False),
        ("unknown", False),
        (None, False),
    ],
)
def test_pinned_commit_matches_the_runtime_abbreviation(
    monkeypatch: pytest.MonkeyPatch,
    reported_commit: str | None,
    matches: bool,
) -> None:
    """A pinned full SHA must match the node's abbreviated commit.

    Qualification pins a 40-character SHA, but a node reports
    `git rev-parse --short HEAD`, so an equality test could never succeed and
    every leg stalled until the readiness deadline. Skulk compares builds by
    abbreviation, and this applies the same contract: prefix match, no shorter
    than git's minimum abbreviation, and never a match on an unknown commit.
    """

    pinned = "32fffb7a36f9872b361c20ef47888f452211a8b6"

    def fake_get(url: str, timeout: float | None = None) -> httpx.Response:
        return httpx.Response(
            200, text='<html><body><div id="root"></div></body></html>'
        )

    monkeypatch.setattr(qualification_checks_module.httpx, "get", fake_get)
    client = cast(SkulkClient, _StubContractClient(reported_commit))

    if matches:
        provenance = qualification_checks_module.assert_fresh_runtime_contract(
            client,
            expected_backends=["llama_server"],
            expected_transport="zenoh",
            expected_commit=pinned,
        )
        # The report records what the node actually said, not the pinned value.
        assert provenance.resolved_commit == reported_commit
        return

    with pytest.raises(RuntimeError, match="did not match the pinned candidate"):
        qualification_checks_module.assert_fresh_runtime_contract(
            client,
            expected_backends=["llama_server"],
            expected_transport="zenoh",
            expected_commit=pinned,
        )


class _StubProvisionClient:
    """Record the store and placement calls a headless provisioning leg makes."""

    def __init__(
        self,
        *,
        download_states: list[str],
        placement_ready: bool = True,
        card_add_error: Exception | None = None,
        download_request_error: Exception | None = None,
        previews: list[dict[str, object]] | None = None,
        catalog_model_ids: list[str] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._catalog_model_ids = catalog_model_ids or []
        self._download_states = iter(download_states)
        self._placement_ready = placement_ready
        self._card_add_error = card_add_error
        self._download_request_error = download_request_error
        default_preview: dict[str, object] = {
            "sharding": "single",
            "instance_meta": "TextInstance",
        }
        self._previews = previews if previews is not None else [default_preview]
        self.placement_request: dict[str, object] | None = None

    def list_models(self) -> list[dict[str, object]]:
        self.calls.append("list_models")
        return [{"id": model_id} for model_id in self._catalog_model_ids]

    def add_model_card(self, model_id: str) -> dict[str, object] | None:
        self.calls.append("add_model_card")
        if self._card_add_error is not None:
            raise self._card_add_error
        return {"model_id": model_id}

    def request_store_download(self, model_id: str) -> dict[str, object] | None:
        self.calls.append("request_store_download")
        if self._download_request_error is not None:
            raise self._download_request_error
        return {"model_id": model_id}

    def get_store_download_status(self, model_id: str) -> dict[str, object] | None:
        self.calls.append("get_store_download_status")
        return {"status": next(self._download_states)}

    def get_placement_previews(self, model_id: str) -> list[dict[str, object]]:
        self.calls.append("get_placement_previews")
        return self._previews

    def place_model(self, **kwargs: object) -> dict[str, object] | None:
        self.calls.append("place_model")
        self.placement_request = dict(kwargs)
        return {"instance_id": "instance-a"}

    def find_placements_for_model(self, model_id: str) -> list[PlacementResult]:
        self.calls.append("find_placements_for_model")
        return [
            PlacementResult(
                model_id=model_id,
                instance_id="instance-a",
                ready=self._placement_ready,
                terminal_failure=not self._placement_ready,
                runner_failure_messages=(
                    [] if self._placement_ready else ["runner exited"]
                ),
            )
        ]


def test_headless_provisioning_walks_the_same_path_the_dashboard_drives() -> None:
    """A target with no web UI must still download, place, and mount the model.

    The browser journey is what provisions the model on a target that serves
    the UI. A headless node has no UI to drive, so it walks the identical
    store-download-then-place endpoints rather than skipping provisioning and
    asking the direct-API parity check to serve a model that was never mounted.
    """

    model_id = "unsloth/Llama-3.2-1B-Instruct-GGUF"
    client = _StubProvisionClient(
        download_states=["downloading", "complete"],
        catalog_model_ids=[model_id],
    )

    _provision_model_over_api(
        cast(SkulkClient, client),
        model_id=model_id,
        model_ready_timeout_s=30,
        poll_interval_s=0,
        heartbeat=None,
    )

    assert client.calls == [
        "list_models",
        "request_store_download",
        "get_store_download_status",
        "get_store_download_status",
        "get_placement_previews",
        "place_model",
        "find_placements_for_model",
    ]
    assert client.placement_request == {
        "model_id": model_id,
        "sharding": "single",
        "instance_meta": "TextInstance",
        "min_nodes": 1,
        "excluded_nodes": [],
    }


def test_headless_provisioning_never_overrides_a_shipped_card() -> None:
    """A model the release ships a card for must not be re-added from the hub.

    ``POST /models/add`` stores a Hugging Face card as a *custom* card, which
    then overrides the shipped one. Adding it here would qualify a card the
    release does not ship. This mirrors the dashboard, which offers "Download"
    for a catalog model and "Add and download" only for one that is not.
    """

    shipped = _StubProvisionClient(
        download_states=["complete"],
        catalog_model_ids=["unsloth/Llama-3.2-1B-Instruct-GGUF"],
    )
    _provision_model_over_api(
        cast(SkulkClient, shipped),
        model_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
        model_ready_timeout_s=30,
        poll_interval_s=0,
        heartbeat=None,
    )
    assert "add_model_card" not in shipped.calls

    uncarded = _StubProvisionClient(
        download_states=["complete"],
        catalog_model_ids=["some/other-model"],
    )
    _provision_model_over_api(
        cast(SkulkClient, uncarded),
        model_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
        model_ready_timeout_s=30,
        poll_interval_s=0,
        heartbeat=None,
    )
    assert "add_model_card" in uncarded.calls


def test_headless_provisioning_failures_fail_the_leg() -> None:
    """Provisioning is a release gate, so no step may degrade into a warning."""

    unreachable = httpx.ConnectError("store unreachable")
    download_refused = _StubProvisionClient(
        download_states=[],
        card_add_error=unreachable,
        download_request_error=unreachable,
    )
    with pytest.raises(RuntimeError, match="after card add failed"):
        _provision_model_over_api(
            cast(SkulkClient, download_refused),
            model_id="model",
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
        )

    download_failed = _StubProvisionClient(download_states=["failed"])
    with pytest.raises(RuntimeError, match="store download failed"):
        _provision_model_over_api(
            cast(SkulkClient, download_failed),
            model_id="model",
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
        )

    runner_failed = _StubProvisionClient(
        download_states=["complete"], placement_ready=False
    )
    with pytest.raises(RuntimeError, match="runner failed"):
        _provision_model_over_api(
            cast(SkulkClient, runner_failed),
            model_id="model",
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
        )

    no_preview = _StubProvisionClient(download_states=["complete"], previews=[])
    with pytest.raises(RuntimeError, match="no viable placement preview"):
        _provision_model_over_api(
            cast(SkulkClient, no_preview),
            model_id="model",
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
        )

    only_rejected = _StubProvisionClient(
        download_states=["complete"],
        previews=[{"sharding": "single", "error": "does not fit"}],
    )
    with pytest.raises(RuntimeError, match="no viable placement preview"):
        _provision_model_over_api(
            cast(SkulkClient, only_rejected),
            model_id="model",
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
        )


def test_headless_provisioning_skips_previews_the_planner_rejected() -> None:
    """A rejected preview listed first must not shadow a viable one behind it.

    ``/instance/previews`` lists options the planner examined, and an option it
    rejected carries an error. Taking the first entry blindly would place
    against a rejected option and fail a target that was in fact offering a
    working placement, which is the exact false-failure class this gate exists
    to avoid.
    """

    rejected: dict[str, object] = {
        "sharding": "pipeline",
        "instance_meta": "TextInstance",
        "error": "model does not fit on the available nodes",
    }
    viable: dict[str, object] = {
        "sharding": "single",
        "instance_meta": "TextInstance",
    }
    client = _StubProvisionClient(
        download_states=["complete"], previews=[rejected, viable]
    )

    _provision_model_over_api(
        cast(SkulkClient, client),
        model_id="model",
        model_ready_timeout_s=30,
        poll_interval_s=0,
        heartbeat=None,
    )

    assert client.placement_request is not None
    assert client.placement_request["sharding"] == "single"


class _StubConsentDialog:
    """Stand in for the Playwright locator of the first-run consent modal."""

    def __init__(self, *, visible: bool) -> None:
        self._visible = visible
        self._pending_click = ""
        self.clicked: list[str] = []
        self.waited_states: list[str] = []

    def wait_for(self, *, state: str, timeout: float) -> None:
        self.waited_states.append(state)
        if state == "visible" and not self._visible:
            raise TimeoutError("consent dialog never appeared")

    def get_by_role(self, role: str, *, name: str, exact: bool) -> "_StubConsentDialog":
        self._pending_click = f"{role}:{name}"
        return self

    def click(self) -> None:
        self.clicked.append(self._pending_click)


class _StubConsentPage:
    def __init__(self, dialog: _StubConsentDialog) -> None:
        self._dialog = dialog

    def get_by_role(self, role: str) -> "_StubConsentPage":
        return self

    def get_by_text(self, text: str, *, exact: bool) -> str:
        return text

    def filter(self, *, has: object) -> _StubConsentDialog:
        return self._dialog


def test_first_run_consent_modal_is_answered_not_bypassed() -> None:
    """A clean machine shows a blocking consent modal that the journey answers.

    The modal covers the page and intercepts every pointer event until it is
    answered, so a fresh install fails at the first click without this. A
    long-lived operator browser stamped its marker long ago and never sees it,
    which is exactly the fleet-versus-new-user delta this records. "Not now"
    is the deliberate answer: it leaves fleet consent unasked, so a throwaway
    qualification node never enables collection or publishes anything.
    """

    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )

    prompted_dialog = _StubConsentDialog(visible=True)
    prompted = qualifier._dismiss_first_run_consent(  # pyright: ignore[reportPrivateUsage]
        cast(Page, _StubConsentPage(prompted_dialog))
    )
    assert prompted is True
    assert prompted_dialog.clicked == ["button:Not now"]
    assert prompted_dialog.waited_states == ["visible", "hidden"]

    absent_dialog = _StubConsentDialog(visible=False)
    absent = qualifier._dismiss_first_run_consent(  # pyright: ignore[reportPrivateUsage]
        cast(Page, _StubConsentPage(absent_dialog))
    )
    assert absent is False
    assert absent_dialog.clicked == []


class _StubConversationLocator:
    """Stand in for a Playwright locator used by the conversation reset."""

    def __init__(self, *, matches: int = 1) -> None:
        self.matches = matches
        self.clicks = 0
        self.selected: list[str] = []
        self.waited_states: list[str] = []

    def count(self) -> int:
        return self.matches

    def click(self) -> None:
        self.clicks += 1

    def select_option(self, value: str) -> None:
        self.selected.append(value)

    def wait_for(self, *, state: str, timeout: float) -> None:
        self.waited_states.append(state)


class _StubConversationPage:
    """A dashboard page whose prior thread clears, or does not."""

    def __init__(self, *, stale_replies: int, clears: bool) -> None:
        self.new_button = _StubConversationLocator()
        self.model_selector = _StubConversationLocator()
        self.message_box = _StubConversationLocator()
        self.assistant = _StubConversationLocator(matches=stale_replies)
        self._clears = clears
        self.idle_polls = 0

    def get_by_role(
        self, role: str, *, name: str, exact: bool
    ) -> _StubConversationLocator:
        assert (role, name, exact) == ("button", "+ New", True)
        return self.new_button

    def get_by_label(self, label: str, *, exact: bool) -> _StubConversationLocator:
        assert exact is True
        if label == "Select chat model":
            return self.model_selector
        if label == "Chat message":
            return self.message_box
        if label == "Assistant message":
            return self.assistant
        raise AssertionError(f"unexpected label {label!r}")

    def wait_for_timeout(self, milliseconds: float) -> None:
        self.idle_polls += 1
        if self._clears:
            self.assistant.matches = 0


def test_vision_turn_starts_its_own_conversation() -> None:
    """Each capability check must be judged on its own answer.

    Stacking the vision turn onto the text turn's thread left the earlier
    instruction ("repeat this phrase back exactly and say nothing else")
    standing, and a 4B model obeyed it: shown the image fixture, it answered
    with the previous turn's echo phrase. The leg failed reporting a vision
    defect on a run whose image bytes had provably arrived intact. Resetting
    through the control a user would click keeps the check honest, and
    re-selecting the model covers a new conversation coming up unselected.
    """

    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )

    page = _StubConversationPage(stale_replies=2, clears=True)
    message = qualifier._start_new_conversation(  # pyright: ignore[reportPrivateUsage]
        cast(Page, page), model_id="org/vision-model"
    )

    assert page.new_button.clicks == 1
    assert page.model_selector.selected == ["org/vision-model"]
    assert cast(object, message) is page.message_box
    assert page.message_box.waited_states == ["visible"]


def test_a_conversation_that_never_clears_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reset that silently kept the old thread would measure the wrong thing.

    Without this the vision turn would run against the text turn's context
    again and the failure would once more be reported as a vision defect.
    """

    ticks = iter(range(0, 600, 10))
    monkeypatch.setattr(
        dashboard_qualification_module.time, "monotonic", lambda: float(next(ticks))
    )

    page = _StubConversationPage(stale_replies=1, clears=False)
    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )

    with pytest.raises(RuntimeError, match="prior assistant messages"):
        qualifier._start_new_conversation(  # pyright: ignore[reportPrivateUsage]
            cast(Page, page), model_id="org/vision-model"
        )


def test_browser_vision_expectation_is_per_model_not_per_target() -> None:
    """A text model must be checked for the absence of a vision path.

    The dashboard enables its attachment control from the selected model's own
    image-input support, so a text model offers no image path even on a target
    whose platform can serve vision. Treating the target's positive contract as
    the per-model expectation demanded a vision fixture for a text model and
    failed the whole leg on a model that had already found, downloaded,
    launched, and chatted successfully.
    """

    vision_models = ["org/vision-model"]

    assert (
        _browser_vision_expectation(
            "org/vision-model",
            vision_models=vision_models,
            card_image_input=True,
        )
        == "positive"
    )
    assert (
        _browser_vision_expectation(
            "org/text-model",
            vision_models=vision_models,
            card_image_input=False,
        )
        == "unavailable"
    )
    # A model the catalog did not report on falls back to the inventory.
    assert (
        _browser_vision_expectation(
            "org/text-model", vision_models=[], card_image_input=None
        )
        == "unavailable"
    )


def test_vision_classification_disagreement_is_named_not_asserted_around() -> None:
    """A card that disagrees with the inventory must fail by saying so.

    The shipped card is the authority for what a model can accept, and the
    inventory list is the operator's statement of intent. When they disagreed,
    deriving the expectation from the list alone made the journey assert
    something untrue about the interface and fail on the assertion, pointing
    diagnosis at the dashboard instead of at the misclassification. This is a
    real case: a small model whose shipped card declares native multimodal
    support was listed as a text model, and the leg failed claiming the
    dashboard offered a false vision path when the card says it is a vision
    model.
    """

    with pytest.raises(RuntimeError, match="vision classification disagrees"):
        _browser_vision_expectation(
            "org/actually-a-vision-model",
            vision_models=[],
            card_image_input=True,
        )
    with pytest.raises(RuntimeError, match="vision classification disagrees"):
        _browser_vision_expectation(
            "org/actually-a-text-model",
            vision_models=["org/actually-a-text-model"],
            card_image_input=False,
        )


def test_failed_browser_journey_reports_the_progress_it_actually_made() -> None:
    """A journey that broke at the last step must not report zero progress.

    The failure path used to return an untouched outcome, so a journey that
    found, downloaded, launched, and chatted successfully and then failed on a
    later assertion was reported as if the very first click had failed. That
    sent diagnosis to the wrong end of the journey entirely.
    """

    progress = _JourneyProgress(model_id="org/model")
    progress.first_run_consent_prompted = True
    progress.found = True
    progress.download_started = True
    progress.launched = True
    progress.selected = True
    progress.text_chat_passed = True

    outcome = progress.outcome(passed=False, message="something later broke")

    assert outcome.model_id == "org/model"
    assert outcome.found is True
    assert outcome.download_started is True
    assert outcome.launched is True
    assert outcome.selected is True
    assert outcome.text_chat_passed is True
    assert outcome.first_run_consent_prompted is True
    assert outcome.passed is False
    assert outcome.message == "something later broke"


def test_untouched_browser_journey_reports_no_progress() -> None:
    """A journey that failed before its first step still reports nothing done."""

    outcome = _JourneyProgress(model_id="org/model").outcome(
        passed=False, message="first click failed"
    )

    assert outcome.found is False
    assert outcome.download_started is False
    assert outcome.launched is False
    assert outcome.selected is False
    assert outcome.text_chat_passed is False
    assert outcome.vision is None
    assert outcome.false_vision_path_offered is None
    assert outcome.passed is False


def test_echo_phrase_does_not_look_like_a_credential() -> None:
    """The echo phrase must not read as a secret to a safety-tuned model.

    A hex nonce (``FRESH-3D8C32F7``) was refused outright by a 1B instruct
    model, which failed a qualification leg on an install that was working
    perfectly. The phrase is ordinary language for that reason, so this
    asserts the property rather than the exact wording.
    """

    for _ in range(50):
        phrase = echo_phrase()
        words = phrase.split(" ")
        assert len(words) == 3
        assert words[0].isalpha() and words[1].isalpha()
        assert words[0] != words[1]
        assert words[2].isdigit() and len(words[2]) == 4
        # No hex-nonce run, and none of the words that cue "secret".
        assert not re.search(r"[0-9a-f]{8}", phrase, re.IGNORECASE)
        assert "-" not in phrase
        assert not re.search(r"token|key|secret|code", phrase, re.IGNORECASE)

    assert "token" not in echo_prompt("amber harbor 1234").lower()


def test_echo_phrase_is_unpredictable() -> None:
    """A stale or replayed response must not be able to satisfy the check."""

    assert len({echo_phrase() for _ in range(200)}) > 150


def test_echo_match_accepts_a_recapitalized_reply() -> None:
    """A model that capitalizes its reply has still proven the chat path works.

    The browser waiter already returns on a case-insensitive match, so a
    case-sensitive assertion afterwards would reject a response the wait had
    declared good.
    """

    assert echo_matched("amber harbor 4821", "Amber Harbor 4821")
    assert echo_matched("amber harbor 4821", "Sure! amber harbor 4821")
    assert not echo_matched("amber harbor 4821", "amber harbor 4822")


def test_failed_text_chat_reports_what_the_model_actually_said() -> None:
    """A refused prompt must be distinguishable from a broken chat path.

    Both produce ``text_chat_passed: false``. Only the response text says
    which one happened, and reading it out of a screenshot is not diagnosis.
    """

    progress = _JourneyProgress(model_id="org/model")
    progress.text_chat_response = "I can't assist with that request."

    message = progress.failure_message()

    assert message is not None
    assert "I can't assist with that request." in message
    assert progress.outcome(passed=False, message=message).message == message


def test_passing_text_chat_carries_no_failure_message() -> None:
    """A journey that met its assertions reports no explanation."""

    assert _JourneyProgress(model_id="org/model").failure_message() is None


def test_failed_vision_reports_what_the_model_actually_said() -> None:
    """A vision miss must be distinguishable from a broken image path.

    The image digest already proves the bytes arrived, so what remains
    unexplained is the answer itself. Reading it out of a screenshot is not
    diagnosis: shown the fixture, a model once replied with the previous
    turn's echo phrase, and only the response text revealed that the two
    checks were sharing a conversation.
    """

    progress = _JourneyProgress(model_id="org/model")
    progress.vision_response = "quartz cobalt 2911 square"

    message = progress.failure_message()

    assert message is not None
    assert "quartz cobalt 2911 square" in message
    assert "vision response" in message


def test_both_chat_and_vision_failures_are_reported_together() -> None:
    """One failing check must not hide the other."""

    progress = _JourneyProgress(model_id="org/model")
    progress.text_chat_response = "I can't assist with that."
    progress.vision_response = "a blue triangle"

    message = progress.failure_message()

    assert message is not None
    assert "I can't assist with that." in message
    assert "a blue triangle" in message
