"""Unit coverage for the release-blocking fresh-install lifecycle."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, Literal, cast

import httpx
import pytest
from PIL import Image
from playwright.sync_api import Browser, BrowserContext, Page
from pydantic import ValidationError

import skulk_test_harness.dashboard_qualification as dashboard_qualification_module
import skulk_test_harness.fresh_install as fresh_install_module
import skulk_test_harness.qualification_checks as qualification_checks_module
import skulk_test_harness.target_control as target_control_module
from skulk_test_harness import runpod as runpod_module
from skulk_test_harness.client import SkulkClient
from skulk_test_harness.dashboard_qualification import (
    DashboardQualifier,
    _capture_and_close_browser,  # pyright: ignore[reportPrivateUsage]
    _captured_image_digest,  # pyright: ignore[reportPrivateUsage]
    _fake_microphone_recording_ms,  # pyright: ignore[reportPrivateUsage]
    _JourneyProgress,  # pyright: ignore[reportPrivateUsage]
    _pcm_wav_duration_and_rms,  # pyright: ignore[reportPrivateUsage]
    _transcript_matches,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.echo_phrase import echo_matched, echo_phrase, echo_prompt
from skulk_test_harness.fleet_lock import FleetLease, LeaseOutcome
from skulk_test_harness.fresh_install import (
    QualificationInterruptedError,
    QualificationSignalGuard,
    _assert_declared_member_topologies,  # pyright: ignore[reportPrivateUsage]
    _blocking_issues,  # pyright: ignore[reportPrivateUsage]
    _browser_vision_expectation,  # pyright: ignore[reportPrivateUsage]
    _clean_environment_command,  # pyright: ignore[reportPrivateUsage]
    _failed_lifecycle_stages,  # pyright: ignore[reportPrivateUsage]
    _installer_command,  # pyright: ignore[reportPrivateUsage]
    _llama_server_process_contract,  # pyright: ignore[reportPrivateUsage]
    _PhysicalFleetMemberRuntime,  # pyright: ignore[reportPrivateUsage]
    _provision_model_over_api,  # pyright: ignore[reportPrivateUsage]
    _qualify_served_engine,  # pyright: ignore[reportPrivateUsage]
    _run_member_operations,  # pyright: ignore[reportPrivateUsage]
    _run_remote_logged_command,  # pyright: ignore[reportPrivateUsage]
    _runpod_ephemeral_target,  # pyright: ignore[reportPrivateUsage]
    _runtime_start_command,  # pyright: ignore[reportPrivateUsage]
    _self_safe_process_pattern,  # pyright: ignore[reportPrivateUsage]
    _served_engine_envelope,  # pyright: ignore[reportPrivateUsage]
    _wait_for_api_identity,  # pyright: ignore[reportPrivateUsage]
    _wait_for_runtime_contract,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.lease_heartbeat import (
    AuthoritativeLeaseHeartbeat,
    LeaseHeartbeatError,
)
from skulk_test_harness.models import (
    DashboardAudioContract,
    DashboardAudioEvidence,
    DashboardContract,
    DashboardExperienceEvidence,
    FleetLock,
    FreshInstallConfig,
    FreshInstallLifecycleStage,
    FreshInstallMemberEvidence,
    FreshInstallPhysicalFleet,
    FreshInstallQualificationReport,
    FreshInstallTarget,
    HarnessConfig,
    InstallProvenance,
    Issue,
    PlacementResult,
    RunPodFreshInstallConfig,
    ServedEngineContract,
    ServedEngineEvidence,
)
from skulk_test_harness.qualification_checks import (
    UnexpectedFreshInstallPeerError,
    assert_fresh_cluster,
    assert_fresh_runtime_contract,
    assert_fresh_single_node,
    qualify_direct_text,
    qualify_direct_vision,
)
from skulk_test_harness.reporting import (
    ReportWriter,
    _fresh_install_markdown,  # pyright: ignore[reportPrivateUsage]
)
from skulk_test_harness.runpod import RunPodClient, RunPodSshEndpoint
from skulk_test_harness.target_control import (
    OriginalTargetState,
    RecoverySnapshot,
    SshTargetController,
    _restore_config_files_command,  # pyright: ignore[reportPrivateUsage]
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


def _whole_fleet_target(
    platform: Literal["apple", "amd"],
    *,
    eligible: bool = True,
) -> FreshInstallTarget:
    common: dict[str, object] = {
        "kind": "physical",
        "platform": platform,
        "hardware_class": f"{platform}-hardware",
        "eligible": eligible,
        "ssh_host": f"{platform}-alias",
        "service_stop_command": "stop",
        "service_start_command": "start",
        "whole_fleet_member": True,
        "expected_data_transport": "zenoh",
    }
    if platform == "apple":
        common.update(
            {
                "expected_backends": ["mlx"],
                "vision_contract": "positive",
                "text_models": ["mlx-community/Qwen3.5-2B-4bit"],
                "vision_models": ["mlx-community/Qwen3.5-2B-4bit"],
            }
        )
    else:
        common.update(
            {
                "expected_backends": ["llama_server"],
                "vision_contract": "unavailable",
                "text_models": ["unsloth/Llama-3.2-1B-Instruct-GGUF"],
            }
        )
    return FreshInstallTarget.model_validate(common)


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


@pytest.mark.parametrize("service_manager", ["launchd", "systemd"])
def test_eligible_supervised_target_requires_archived_service_environment(
    service_manager: Literal["launchd", "systemd"],
) -> None:
    target_payload = _whole_fleet_target("apple").model_dump()
    target_payload.update(
        {
            "service_manager": service_manager,
            "original_config_paths": ["/private/skulk.yaml"],
        }
    )

    with pytest.raises(
        ValueError,
        match="must include skulk.env in original_config_paths",
    ):
        FreshInstallTarget.model_validate(target_payload)

    target_payload["original_config_paths"] = [
        "/private/skulk.yaml",
        "/private/skulk.env",
    ]
    validated = FreshInstallTarget.model_validate(target_payload)

    assert validated.original_config_paths[-1] == "/private/skulk.env"


def test_complete_release_matrix_requires_every_blocking_platform() -> None:
    config = FreshInstallConfig(targets={"apple": _physical_target()})
    selected = config.eligible_targets()

    with pytest.raises(ValueError, match="amd.*nvidia"):
        config.assert_complete_release_matrix(selected)


def test_physical_fleet_accepts_normal_networking_and_composes_platforms() -> None:
    config = FreshInstallConfig(
        required_platforms=["apple", "amd"],
        targets={
            "apple-1": _whole_fleet_target("apple"),
            "apple-2": _whole_fleet_target("apple"),
            "amd-1": _whole_fleet_target("amd"),
        },
        physical_fleets={
            "release-fleet": FreshInstallPhysicalFleet(
                hardware_class="mixed-three-node",
                eligible=True,
                member_targets=["apple-1", "apple-2", "amd-1"],
                entrypoint_target="apple-1",
                qualification_targets=["apple-1", "amd-1"],
            )
        },
    )

    selected_fleets = config.eligible_physical_fleets()
    assert [name for name, _fleet in selected_fleets] == ["release-fleet"]
    assert [
        name for name, _target in config.physical_fleet_targets(selected_fleets[0][1])
    ] == ["apple-1", "apple-2", "amd-1"]
    config.assert_complete_release_matrix([], selected_fleets)


def test_physical_fleet_release_coverage_uses_only_qualification_targets() -> None:
    config = FreshInstallConfig(
        required_platforms=["apple", "amd"],
        targets={
            "apple-contract": _whole_fleet_target("apple"),
            "amd-capacity-only": _whole_fleet_target("amd"),
        },
        physical_fleets={
            "release-fleet": FreshInstallPhysicalFleet(
                hardware_class="mixed-two-node",
                eligible=True,
                member_targets=["apple-contract", "amd-capacity-only"],
                entrypoint_target="apple-contract",
                qualification_targets=["apple-contract"],
            )
        },
    )

    with pytest.raises(ValueError, match="amd"):
        config.assert_complete_release_matrix([], config.eligible_physical_fleets())


def test_physical_fleet_rejects_nonmember_contract_and_isolated_member() -> None:
    with pytest.raises(ValidationError, match="qualification_targets must be members"):
        FreshInstallPhysicalFleet(
            hardware_class="mixed",
            eligible=True,
            member_targets=["apple-1", "amd-1"],
            entrypoint_target="apple-1",
            qualification_targets=["not-a-member"],
        )

    isolated = _physical_target()
    with pytest.raises(ValidationError, match="whole_fleet_member=true"):
        FreshInstallConfig(
            targets={
                "apple-1": isolated,
                "amd-1": _whole_fleet_target("amd"),
            },
            physical_fleets={
                "release-fleet": FreshInstallPhysicalFleet(
                    hardware_class="mixed",
                    eligible=True,
                    member_targets=["apple-1", "amd-1"],
                    entrypoint_target="apple-1",
                    qualification_targets=["apple-1", "amd-1"],
                )
            },
        )


def test_member_operations_do_not_swallow_recovery_interrupt() -> None:
    target = _whole_fleet_target("apple")
    member = _PhysicalFleetMemberRuntime(
        ordinal=1,
        target_name="apple-1",
        target=target,
        controller=SshTargetController(target),
    )

    def interrupted(_member: _PhysicalFleetMemberRuntime) -> None:
        raise QualificationInterruptedError("stop now")

    with pytest.raises(QualificationInterruptedError, match="stop now"):
        _run_member_operations([member], interrupted)


def test_member_operations_propagate_lease_failure_immediately() -> None:
    target = _whole_fleet_target("apple")
    members = [
        _PhysicalFleetMemberRuntime(
            ordinal=ordinal,
            target_name=f"apple-{ordinal}",
            target=target,
            controller=SshTargetController(target),
        )
        for ordinal in (1, 2)
    ]
    attempted: list[int] = []

    def heartbeat_failed(member: _PhysicalFleetMemberRuntime) -> None:
        attempted.append(member.ordinal)
        raise LeaseHeartbeatError("lease ownership lost")

    with pytest.raises(LeaseHeartbeatError, match="lease ownership lost"):
        _run_member_operations(members, heartbeat_failed)
    assert attempted == [1]


def test_declared_topology_rejects_substituted_member_view() -> None:
    declared = frozenset({"node-a", "node-b", "node-c"})

    assert (
        _assert_declared_member_topologies(
            expected_node_count=3,
            local_node_ids=declared,
            member_observed_node_ids=(declared, declared, declared),
        )
        == declared
    )
    with pytest.raises(RuntimeError, match=r"mismatched member views \[2\]"):
        _assert_declared_member_topologies(
            expected_node_count=3,
            local_node_ids=declared,
            member_observed_node_ids=(
                declared,
                frozenset({"node-a", "node-b", "incidental-node"}),
                declared,
            ),
        )


def test_temporary_home_uses_the_target_home_filesystem() -> None:
    commands: list[str] = []

    class FakeController:
        def run(
            self,
            command: str,
            *,
            timeout_s: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            assert timeout_s == 30
            commands.append(command)
            return subprocess.CompletedProcess(
                [],
                0,
                "/home/operator/.skulk-fresh.a1B2c3",
                "",
            )

    temporary_root = fresh_install_module.FreshInstallQualifier._create_temporary_home(  # pyright: ignore[reportPrivateUsage]
        cast(SshTargetController, FakeController())
    )

    assert temporary_root == "/home/operator/.skulk-fresh.a1B2c3"
    assert 'mktemp -d "$HOME/.skulk-fresh.XXXXXX"' in commands[0]
    assert "/tmp/skulk-fresh" not in commands[0]


@pytest.mark.parametrize(
    "unsafe_root",
    [
        "/tmp/skulk-fresh.a1B2c3",
        "/home/operator/not-skulk.a1B2c3",
        "relative/.skulk-fresh.a1B2c3",
        "/home/operator/.skulk-fresh.a1B2c3\n/untrusted",
    ],
)
def test_temporary_home_rejects_an_unsafe_remote_root(unsafe_root: str) -> None:
    class FakeController:
        def run(
            self,
            _command: str,
            *,
            timeout_s: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            assert timeout_s == 30
            return subprocess.CompletedProcess([], 0, unsafe_root, "")

    with pytest.raises(RuntimeError, match="unsafe temporary root"):
        fresh_install_module.FreshInstallQualifier._create_temporary_home(  # pyright: ignore[reportPrivateUsage]
            cast(SshTargetController, FakeController())
        )


def test_restoration_rejects_asymmetric_member_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={},
        process_arguments=["uv run skulk"],
        service_status="running",
        api_node_id="old-node",
        cluster_node_count=2,
    )

    class FakeController:
        def run(self, *_args: object, **_kwargs: object) -> None:
            pass

        def restore_original_config_files(
            self,
            _snapshot: RecoverySnapshot,
        ) -> None:
            pass

        def verify_restored_state(
            self,
            _original: OriginalTargetState,
            *,
            api_node_id: str | None,
            cluster_node_count: int | None,
        ) -> list[str]:
            assert api_node_id is not None
            assert cluster_node_count == 2
            return []

    members = [
        _PhysicalFleetMemberRuntime(
            ordinal=ordinal,
            target_name=f"apple-{ordinal}",
            target=_whole_fleet_target("apple"),
            controller=cast(SshTargetController, FakeController()),
            local_port=52000 + ordinal,
            snapshot=RecoverySnapshot(
                remote_path="/private/recovery.tar.gz",
                remote_sha256="remote-digest",
                controller_path=tmp_path / f"recovery-{ordinal}.tar.gz",
                controller_sha256="controller-digest",
                original=original,
            ),
            service_stopped=True,
        )
        for ordinal in (1, 2)
    ]
    local_ids = {
        "http://127.0.0.1:52001": ("node-a", 2),
        "http://127.0.0.1:52002": ("node-b", 2),
    }
    member_views = {
        "http://127.0.0.1:52001": frozenset({"node-a", "node-b"}),
        "http://127.0.0.1:52002": frozenset({"node-a", "incidental-node"}),
    }
    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_api_identity",
        lambda api_base_url, **_kwargs: local_ids[api_base_url],
    )
    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_exact_cluster",
        lambda api_base_url, **_kwargs: member_views[api_base_url],
    )

    class FakeJournal:
        def stage(self, _name: str) -> object:
            return nullcontext()

        def persist(self) -> None:
            pass

    report = _report_with([]).model_copy(
        update={
            "members": [
                FreshInstallMemberEvidence(
                    ordinal=ordinal,
                    platform="apple",
                    hardware_class="apple-hardware",
                )
                for ordinal in (1, 2)
            ]
        }
    )
    qualifier = fresh_install_module.FreshInstallQualifier(
        HarnessConfig(fresh_install=FreshInstallConfig())
    )

    with pytest.raises(RuntimeError, match=r"mismatched member views \[2\]"):
        qualifier._restore_physical_fleet(  # pyright: ignore[reportPrivateUsage]
            members=members,
            report=report,
            journal=cast(fresh_install_module._LifecycleJournal, FakeJournal()),  # pyright: ignore[reportPrivateUsage]
        )
    assert [member.restored for member in report.members] == [None, None]


def test_fresh_runtime_evidence_rejects_asymmetric_member_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        _PhysicalFleetMemberRuntime(
            ordinal=ordinal,
            target_name=f"apple-{ordinal}",
            target=_whole_fleet_target("apple"),
            controller=SshTargetController(_whole_fleet_target("apple")),
            local_port=53000 + ordinal,
        )
        for ordinal in (1, 2)
    ]
    local_ids = {
        "http://127.0.0.1:53001": ("node-a", 2),
        "http://127.0.0.1:53002": ("node-b", 2),
    }
    member_views = {
        "http://127.0.0.1:53001": frozenset({"node-a", "node-b"}),
        "http://127.0.0.1:53002": frozenset({"node-a", "incidental-node"}),
    }
    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_api_identity",
        lambda api_base_url, **_kwargs: local_ids[api_base_url],
    )
    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_exact_cluster",
        lambda api_base_url, **_kwargs: member_views[api_base_url],
    )

    class FakeSkulkClient:
        def __init__(self, api_base_url: str) -> None:
            self.api_base_url = api_base_url

        def __enter__(self) -> "FakeSkulkClient":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def get_state(self) -> dict[str, object]:
            node_id = local_ids[self.api_base_url][0]
            return {
                "nodeResources": {
                    node_id: {
                        "backends": ["mlx"],
                        "dataTransport": "zenoh",
                    }
                }
            }

    monkeypatch.setattr(fresh_install_module, "SkulkClient", FakeSkulkClient)
    monkeypatch.setattr(
        fresh_install_module.httpx,
        "get",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            text='<html><div id="root"></div></html>',
        ),
    )
    report = _report_with([]).model_copy(
        update={
            "members": [
                FreshInstallMemberEvidence(
                    ordinal=ordinal,
                    platform="apple",
                    hardware_class="apple-hardware",
                )
                for ordinal in (1, 2)
            ]
        }
    )
    qualifier = fresh_install_module.FreshInstallQualifier(
        HarnessConfig(fresh_install=FreshInstallConfig())
    )

    with pytest.raises(RuntimeError, match=r"mismatched member views \[2\]"):
        qualifier._record_member_runtime_evidence(  # pyright: ignore[reportPrivateUsage]
            members=members,
            expected_node_count=2,
            report=report,
        )


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


def test_dashboard_audio_requires_a_real_dashboard_and_mlx_audio() -> None:
    contract = DashboardAudioContract(
        speech_synthesis_model="org/tts",
        transcription_model="org/stt",
    )
    with pytest.raises(ValidationError, match="requires mlx_audio"):
        FreshInstallTarget(
            kind="runpod",
            platform="nvidia",
            hardware_class="cuda",
            eligible=True,
            expected_backends=["llama_server", "llama_server-cuda"],
            vision_contract="unavailable",
            text_models=["org/chat"],
            dashboard_audio=contract,
        )
    with pytest.raises(ValidationError, match="dashboard_contract='required'"):
        FreshInstallTarget(
            kind="physical",
            platform="apple",
            hardware_class="apple",
            eligible=True,
            ssh_host="alias",
            service_stop_command="stop",
            service_start_command="start",
            isolation_enter_command="isolate",
            isolation_exit_command="restore",
            expected_backends=["mlx", "mlx_audio"],
            vision_contract="unavailable",
            text_models=["org/chat"],
            dashboard_contract="absent",
            dashboard_audio=contract,
        )

    target = FreshInstallTarget(
        kind="physical",
        platform="apple",
        hardware_class="apple",
        eligible=True,
        ssh_host="alias",
        service_stop_command="stop",
        service_start_command="start",
        isolation_enter_command="isolate",
        isolation_exit_command="restore",
        expected_backends=["mlx", "mlx_audio"],
        vision_contract="unavailable",
        text_models=["org/chat"],
        dashboard_audio=contract,
    )
    assert target.dashboard_audio == contract


def test_served_engine_contract_must_name_an_expected_backend() -> None:
    with pytest.raises(
        ValidationError,
        match="served_engine_contract backend must be listed",
    ):
        FreshInstallTarget(
            kind="runpod",
            platform="nvidia",
            hardware_class="cuda",
            eligible=True,
            expected_backends=["llama_server", "llama_server-cuda"],
            served_engine_contract=ServedEngineContract(backend="llama_server-vulkan"),
            vision_contract="unavailable",
            text_models=["unsloth/Llama-3.2-1B-Instruct-GGUF"],
        )


def test_llama_server_process_contract_reads_effective_flags() -> None:
    listing = (
        "  4321 /tmp/skulk-fresh.abc/home/.cache/skulk/llama-server "
        "--model /tmp/skulk-fresh.abc/home/model.gguf "
        "--parallel 16 --kv-unified\n"
    )

    assert _llama_server_process_contract(
        listing,
        installation_root="/tmp/skulk-fresh.abc",
    ) == (4321, 16, True)


def test_llama_server_process_contract_rejects_ambiguous_children() -> None:
    listing = "\n".join(
        [
            "4321 /tmp/skulk-fresh.abc/home/bin/llama-server --parallel 16",
            "4322 /tmp/skulk-fresh.abc/home/bin/llama-server --parallel 16",
        ]
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        _llama_server_process_contract(
            listing,
            installation_root="/tmp/skulk-fresh.abc",
        )


def test_served_engine_probe_detects_runner_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listings = [
        ("4321 /tmp/skulk-fresh.abc/home/bin/llama-server --parallel 16 --kv-unified"),
        ("9876 /tmp/skulk-fresh.abc/home/bin/llama-server --parallel 16 --kv-unified"),
    ]

    class FakeController:
        def run(
            self,
            _command: str,
            *,
            timeout_s: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            del timeout_s
            return subprocess.CompletedProcess([], 0, listings.pop(0), "")

    monkeypatch.setattr(
        fresh_install_module,
        "_run_served_engine_overlap_probe",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        fresh_install_module,
        "_served_engine_envelope",
        lambda **_kwargs: (4, True),
    )

    evidence = _qualify_served_engine(
        controller=cast(SshTargetController, FakeController()),
        installation_root="/tmp/skulk-fresh.abc",
        api_base_url="http://example.invalid",
        model_id="model",
        contract=ServedEngineContract(
            backend="llama_server-vulkan",
            parallel=16,
            kv_unified=True,
            probe_concurrency=4,
        ),
        request_timeout_s=1,
        stream_read_timeout_s=1,
    )

    assert evidence.runner_survived is False
    assert evidence.passed is False


def test_served_engine_envelope_requires_matching_backend_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        200,
        json={
            "envelopes": [
                {
                    "modelId": "model",
                    "backend": "llama_server-vulkan",
                    "batches": True,
                    "buckets": [
                        {"concurrency": 1},
                        {"concurrency": 4},
                    ],
                },
                {
                    "modelId": "different",
                    "backend": "llama_server-vulkan",
                    "batches": True,
                    "buckets": [{"concurrency": 99}],
                },
            ]
        },
        request=httpx.Request("GET", "http://example.invalid"),
    )
    monkeypatch.setattr(
        fresh_install_module.httpx, "get", lambda *_args, **_kw: response
    )

    assert _served_engine_envelope(
        api_base_url="http://example.invalid",
        model_id="model",
        backend="llama_server-vulkan",
        request_timeout_s=1,
    ) == (4, True)


def test_fresh_summary_includes_publishable_served_engine_evidence() -> None:
    report = _report_with([])
    report.served_engines.append(
        ServedEngineEvidence(
            model_id="model",
            backend="llama_server-vulkan",
            expected_parallel=16,
            observed_parallel=16,
            kv_unified_required=True,
            kv_unified_observed=True,
            probe_concurrency=4,
            maximum_observed_active=4,
            batching_reported=True,
            runner_survived=True,
            passed=True,
        )
    )

    summary = _fresh_install_markdown(report)

    assert "parallel `16` (expected `16`)" in summary
    assert "maximum active `4`" in summary
    assert "survived `True`" in summary


def test_fresh_summary_includes_dashboard_release_and_audio_evidence() -> None:
    report = _report_with([]).model_copy(
        update={
            "dashboard_experience": DashboardExperienceEvidence(
                model_id="org/chat",
                settings_opened=True,
                settings_saved=True,
                topology_expected_nodes=5,
                topology_visible_nodes=5,
                request_failure_visible=True,
                request_retry_passed=True,
                webkit_loaded=True,
                webkit_text_chat_passed=True,
                passed=True,
            ),
            "dashboard_audio": DashboardAudioEvidence(
                speech_synthesis_model="org/tts",
                transcription_model="org/stt",
                synthesis_audio_bytes=4096,
                transcription_request_observed=True,
                transcript_matched=True,
                passed=True,
            ),
        }
    )

    summary = _fresh_install_markdown(report)

    assert "Settings opened/saved `True/True`" in summary
    assert "topology `5/5`" in summary
    assert "WebKit load/chat `True/True`" in summary
    assert "Audio `org/tts` -> `org/stt`" in summary
    assert "TTS bytes `4096`" in summary


def test_heartbeat_must_not_exceed_one_third_of_ttl() -> None:
    with pytest.raises(ValidationError, match="one third"):
        FreshInstallConfig(lease_ttl_s=90, lease_heartbeat_s=31)
    assert FreshInstallConfig(lease_ttl_s=90).resolved_lease_heartbeat_s == 30


def test_fresh_runtime_contract_requires_a_stable_one_node_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer arriving just after startup must fail before model work begins."""

    class PeerJoiningClient(_StubContractClient):
        state_reads = 0

        def __enter__(self) -> "PeerJoiningClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_state(self) -> dict[str, object]:
            state = super().get_state()
            self.state_reads += 1
            if self.state_reads > 1:
                resources = cast(dict[str, object], state["nodeResources"])
                identities = cast(dict[str, object], state["nodeIdentities"])
                resources["node-b"] = {
                    "backends": ["llama_server"],
                    "dataTransport": "zenoh",
                }
                identities["node-b"] = {}
            return state

    client = PeerJoiningClient()
    monkeypatch.setattr(fresh_install_module, "SkulkClient", lambda *_args: client)
    monkeypatch.setattr(
        qualification_checks_module.httpx,
        "get",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            text='<html><body><div id="root"></div></body></html>',
        ),
    )

    with pytest.raises(UnexpectedFreshInstallPeerError, match="observed 2"):
        _wait_for_runtime_contract(
            "http://127.0.0.1:52415",
            target=FreshInstallTarget(
                kind="runpod",
                platform="nvidia",
                hardware_class="nvidia-cuda",
                eligible=True,
                expected_backends=["llama_server"],
                vision_contract="unavailable",
            ),
            expected_commit=None,
            timeout_s=1,
            poll_interval_s=0.001,
            stability_s=0.1,
            heartbeat=None,
        )


def test_recovery_tunnel_isolated_from_terminal_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SSH recovery tunnel must survive a signal to the caller's session."""

    captured: dict[str, object] = {}

    class FakeTunnel:
        stderr = None

        def poll(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeTunnel:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeTunnel()

    monkeypatch.setattr(target_control_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(target_control_module.time, "sleep", lambda _seconds: None)

    controller = SshTargetController(_physical_target())
    _local_port, process = controller.open_tunnel(remote_port=52415)

    assert isinstance(process, FakeTunnel)
    assert cast(dict[str, object], captured["kwargs"])["start_new_session"] is True


def test_runtime_isolation_wraps_the_literal_clean_command() -> None:
    payload = _physical_target().model_dump()
    payload["runtime_isolation_prefix"] = "sandbox-exec -f isolation.sb"
    target = FreshInstallTarget.model_validate(payload)

    command = _runtime_start_command(
        temporary_root="/tmp/fresh",
        target=target,
    )

    assert command.startswith("sandbox-exec -f isolation.sb env -i ")
    assert 'cd "$HOME/skulk" && exec uv run skulk' in command
    assert "SKULK_" not in command


def test_runtime_isolation_rejects_product_overrides() -> None:
    payload = _physical_target().model_dump()
    payload["runtime_isolation_prefix"] = "env SKULK_DATA_TRANSPORT=zenoh"

    with pytest.raises(ValidationError, match="cannot add Skulk"):
        FreshInstallTarget.model_validate(payload)


def test_random_vision_fixture_has_exact_judge_free_contract(tmp_path: Path) -> None:
    first = generate_vision_fixture()
    second = generate_vision_fixture()

    assert first.sha256 != second.sha256
    assert first.code != second.code
    assert first.code not in first.prompt
    assert "COLOR SHAPE | CODE" in first.prompt
    assert "COLOR from red, blue, green, or orange" in first.prompt
    assert "SHAPE from circle, diamond, or triangle" in first.prompt
    assert data_url_sha256(first.data_url) == first.sha256
    assert first.response_matches(f"{first.code}\n{first.color} {first.shape}") == (
        True,
        True,
    )
    assert first.response_matches(
        f"{first.color.upper()} {first.shape.upper()} | {first.code}"
    ) == (True, True)
    assert first.response_match_details(
        f"{first.code}\n{first.color} {first.shape}"
    ) == (True, True, True)
    wrong_color = next(
        color for color in ("red", "blue", "green", "orange") if color != first.color
    )
    assert first.response_match_details(
        f"{first.code}\n{wrong_color} {first.shape}"
    ) == (True, False, True)
    grouped_code = f"{first.code[:4]} {first.code[4:]}"
    assert first.response_matches(f"{first.color} {first.shape} | {grouped_code}") == (
        True,
        True,
    )
    hyphenated_code = "-".join(first.code)
    assert first.response_matches(
        f"{first.color} {first.shape} | {hyphenated_code}"
    ) == (True, True)
    assert first.response_matches(f"{first.color} {first.shape} | {first.code}A") == (
        False,
        True,
    )
    assert first.response_matches(
        f"{first.color} {first.shape} | {first.code[:-1]}"
    ) == (False, True)
    assert first.response_matches(
        f"{first.color} {first.shape} | {first.code}{first.code[-1]}"
    ) == (False, True)
    assert first.response_matches("a plausible blue bedroom") == (False, False)
    exact_response = f"{first.color} {first.shape} | {first.code}"
    assert first.response_format_matches(exact_response)
    assert first.response_format_matches(f"**{exact_response}.**")
    assert first.response_format_matches(
        f"{first.color} {first.shape} | {grouped_code}"
    )
    assert first.response_format_matches(
        "I checked the visible attributes.\n\n" + exact_response
    )
    assert first.response_format_matches(
        f"I considered {exact_response} while checking the card.\n"
        "The visible attributes are consistent.\n" + exact_response
    )
    assert not first.response_format_matches(f"I see {exact_response}")
    assert not first.response_format_matches(f"{exact_response}\n{exact_response}")
    assert not first.response_format_matches(
        f"{exact_response} " + "repeated output " * 100
    )
    path = tmp_path / "fixture.png"
    first.write(path)
    assert path.stat().st_mode & 0o777 == 0o600


def test_circle_fixture_is_geometrically_circular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model must not be penalized for calling a stretched circle an oval."""

    def choose_fixture_value(options: str | tuple[str, ...]) -> str:
        if isinstance(options, tuple) and "circle" in options:
            return "circle"
        if isinstance(options, tuple) and "orange" in options:
            return "orange"
        return options[0]

    monkeypatch.setattr(
        "skulk_test_harness.vision_fixture.secrets.choice",
        choose_fixture_value,
    )
    fixture = generate_vision_fixture()
    image = Image.open(io.BytesIO(fixture.png)).convert("RGB")
    orange_pixels = Image.new("1", image.size)
    orange_pixels.putdata(
        [pixel == (255, 128, 0) for pixel in image.get_flattened_data()]
    )
    bounds = orange_pixels.getbbox()

    assert fixture.shape == "circle"
    assert bounds is not None
    assert bounds[2] - bounds[0] == bounds[3] - bounds[1]


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

    outcome = qualify_direct_text(
        cast(SkulkClient, FakeClient()),
        model_id="org/toggle-model",
        enable_thinking=False,
    )
    assert outcome.passed is True
    assert outcome.response == "amber harbor 4821"
    assert calls == [
        {
            "model_id": "org/toggle-model",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write one short, friendly welcome sentence for a hotel "
                        "guest. The hotel is named Amber Harbor, and the guest's "
                        "room number is 4821. Mention both Amber Harbor and room "
                        "4821 in your sentence."
                    ),
                }
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
        }
    ]


def test_direct_vision_requires_concise_response_and_redacts_code() -> None:
    """A concise VLM answer passes without leaking the hidden code."""

    calls: list[dict[str, object]] = []
    fixture = generate_vision_fixture()

    class FakeClient:
        def stream_chat(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                text=f"{fixture.color} {fixture.shape} | {fixture.code}"
            )

    evidence = qualify_direct_vision(
        cast(SkulkClient, FakeClient()),
        model_id="org/vision-model",
        fixture=fixture,
        enable_thinking=False,
    )

    assert evidence.passed is True
    assert evidence.response_matched_format is True
    assert calls[0]["max_tokens"] == 512
    assert evidence.response_excerpt is not None
    assert "<hidden-code>" in evidence.response_excerpt
    assert fixture.code not in evidence.response_excerpt


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
    guard = QualificationSignalGuard()
    with pytest.raises(QualificationInterruptedError, match=str(signal.SIGTERM)):
        guard._handle(signal.SIGTERM, None)  # pyright: ignore[reportPrivateUsage]


def test_interruption_survives_the_broad_handlers_that_report_outcomes() -> None:
    """An operator's stop request must outrank every report-the-failure boundary.

    The qualification wraps browser work, subprocess work, and probes in
    ``except Exception`` so a failure becomes a reported outcome rather than a
    traceback. An interruption caught by one of those would let a billable,
    destructive run continue past the stop request.
    """

    assert not issubclass(QualificationInterruptedError, Exception)

    caught_by_boundary = False
    guard = QualificationSignalGuard()
    try:
        try:
            guard._handle(signal.SIGINT, None)  # pyright: ignore[reportPrivateUsage]
        except Exception:  # noqa: BLE001 - the boundary shape under test
            caught_by_boundary = True
    except QualificationInterruptedError:
        pass
    assert not caught_by_boundary


def test_interruption_still_lets_finally_restore_state() -> None:
    """BaseException does not skip cleanup, so orderly restoration is unaffected."""

    restored = False
    guard = QualificationSignalGuard()
    try:
        try:
            guard._handle(signal.SIGTERM, None)  # pyright: ignore[reportPrivateUsage]
        finally:
            restored = True
    except QualificationInterruptedError:
        pass
    assert restored


def test_dashboard_cleanup_does_not_mask_an_operator_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed Playwright page cannot replace the signal that closed it."""

    cleanup_events: list[str] = []

    class FakePage:
        def on(self, _event: str, _handler: object) -> None:
            return None

        def screenshot(self, **_kwargs: object) -> None:
            cleanup_events.append("screenshot")
            raise RuntimeError("page has been closed")

    class FakeTracing:
        def start(self, **_kwargs: object) -> None:
            return None

        def stop(self, **_kwargs: object) -> None:
            cleanup_events.append("trace")

    class FakeContext:
        tracing = FakeTracing()

        def new_page(self) -> FakePage:
            return FakePage()

    class FakeBrowser:
        def new_context(self, **_kwargs: object) -> FakeContext:
            return FakeContext()

        def close(self) -> None:
            cleanup_events.append("browser")

    class FakeChromium:
        def launch(self, **_kwargs: object) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywrightContext:
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(chromium=FakeChromium())

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        dashboard_qualification_module,
        "sync_playwright",
        lambda: FakePlaywrightContext(),
    )
    qualifier = DashboardQualifier(
        api_base_url="http://127.0.0.1:52415",
        artifact_directory=tmp_path,
        poll_interval_s=0.01,
        model_ready_timeout_s=1,
    )

    def interrupt(*_args: object, **_kwargs: object) -> object:
        raise QualificationInterruptedError("signal 2")

    monkeypatch.setattr(qualifier, "_run_journey", interrupt)

    with pytest.raises(QualificationInterruptedError, match="signal 2"):
        qualifier.qualify(
            model_id="model",
            vision_contract="unavailable",
            fixture=None,
        )

    assert cleanup_events == ["screenshot", "trace", "browser"]


def test_signal_guard_defers_interruptions_during_mandatory_recovery() -> None:
    """A second stop request must not interrupt service or provider cleanup."""

    guard = QualificationSignalGuard()
    guard.begin_recovery()
    report = _report_with([])

    guard._handle(signal.SIGINT, None)  # pyright: ignore[reportPrivateUsage]
    fresh_install_module._record_deferred_interruption(  # pyright: ignore[reportPrivateUsage]
        report,
        guard,
    )

    assert guard.interrupted_signum == signal.SIGINT
    assert report.issues == [
        Issue(
            severity="error",
            message=(
                "fresh-install qualification interrupted by signal "
                f"{signal.SIGINT}; mandatory recovery completed before stopping"
            ),
        )
    ]


def test_temporary_install_cleanup_enters_signal_safe_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt during remote cleanup cannot skip removal of the temp HOME."""

    guard = QualificationSignalGuard()
    commands: list[str] = []

    class CleanupController:
        def run(
            self,
            command: str,
            *,
            timeout_s: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            del timeout_s, check
            commands.append(command)
            if command.startswith("umask 077"):
                return subprocess.CompletedProcess(
                    [],
                    0,
                    "/Users/operator/.skulk-fresh.cleanup-test",
                    "",
                )
            if command.startswith("pkill "):
                guard._handle(  # pyright: ignore[reportPrivateUsage]
                    signal.SIGINT,
                    None,
                )
            return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(
        fresh_install_module,
        "_installer_provenance",
        lambda *_args, **_kwargs: ("https://example.invalid/install.sh", "digest"),
    )
    monkeypatch.setattr(
        fresh_install_module,
        "_run_remote_logged_command",
        lambda **_kwargs: 1,
    )
    target = _physical_target()
    config = HarnessConfig(
        output_dir=tmp_path / "runs",
        fresh_install=FreshInstallConfig(targets={"apple": target}),
    )
    qualifier = fresh_install_module.FreshInstallQualifier(config)
    report = _report_with([])
    journal = fresh_install_module._LifecycleJournal(  # pyright: ignore[reportPrivateUsage]
        report,
        ReportWriter(tmp_path / "runs"),
    )
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()

    with pytest.raises(RuntimeError, match="official installer exited 1"):
        qualifier._execute_clean_install(  # pyright: ignore[reportPrivateUsage]
            controller=cast(SshTargetController, CleanupController()),
            api_base_url="http://127.0.0.1:52415",
            target=target,
            profile="candidate",
            expected_commit="a" * 40,
            report=report,
            journal=journal,
            artifact_directory=artifact_directory,
            heartbeat=None,
            signal_guard=guard,
        )

    assert guard.interrupted_signum == signal.SIGINT
    assert any(command.startswith("pkill ") for command in commands)
    assert commands[-1] == "rm -rf -- /Users/operator/.skulk-fresh.cleanup-test"


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


def test_archived_config_bytes_are_atomically_restored_after_restart(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("staging_keep_recent_gb: 40\n")
    config_path.chmod(0o640)
    recovery_root = tmp_path / "archive-root"
    recovery = recovery_root / "recovery"
    recovery.mkdir(parents=True)
    (recovery / "config-0").write_bytes(config_path.read_bytes())
    archive_path = tmp_path / "recovery.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(recovery, arcname="recovery")

    config_path.write_text("staging_keep_recent_gb: 40.0\n")
    result = subprocess.run(
        _restore_config_files_command(
            archive_path=str(archive_path),
            config_paths=[str(config_path)],
        ),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert config_path.read_bytes() == b"staging_keep_recent_gb: 40\n"
    assert config_path.stat().st_mode & 0o777 == 0o640


def test_recovery_start_suppresses_auto_update_atomically(tmp_path: Path) -> None:
    """A service restart must not advance the checkout before verification."""

    environment_path = tmp_path / "skulk.env"
    environment_path.write_text(
        "SKULK_AUTO_UPDATE=1\nSKULK_LOG_LEVEL=INFO\n",
    )
    environment_path.chmod(0o640)

    result = subprocess.run(
        fresh_install_module._suppress_auto_update_command(  # pyright: ignore[reportPrivateUsage]
            str(environment_path)
        ),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert environment_path.read_text() == (
        "SKULK_AUTO_UPDATE=0\nSKULK_LOG_LEVEL=INFO\n"
    )
    assert environment_path.stat().st_mode & 0o777 == 0o640


def test_failed_recovery_readiness_still_reapplies_archived_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed service start must not strand the transient override on disk."""

    target = _physical_target().model_copy(
        update={"original_config_paths": ["/private/skulk.env"]}
    )
    original = OriginalTargetState(
        git_commit="commit-a",
        git_status="clean",
        config_sha256={"/private/skulk.env": "digest"},
        process_arguments=["uv run skulk"],
        service_status="exit=0\nrunning",
        api_node_id="node-a",
        cluster_node_count=1,
    )
    snapshot = RecoverySnapshot(
        remote_path="/private/recovery.tar.gz",
        remote_sha256="remote-digest",
        controller_path=tmp_path / "recovery.tar.gz",
        controller_sha256="controller-digest",
        original=original,
    )
    commands: list[str] = []
    restored: list[RecoverySnapshot] = []

    class FakeController:
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

        def restore_original_config_files(
            self,
            recovery_snapshot: RecoverySnapshot,
        ) -> None:
            restored.append(recovery_snapshot)

    monkeypatch.setattr(
        fresh_install_module,
        "_wait_for_api_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("API never became ready")
        ),
    )
    report = _report_with([])
    qualifier = fresh_install_module.FreshInstallQualifier(
        HarnessConfig(
            output_dir=tmp_path / "runs",
            fresh_install=FreshInstallConfig(),
        )
    )
    journal = fresh_install_module._LifecycleJournal(  # pyright: ignore[reportPrivateUsage]
        report,
        ReportWriter(tmp_path / "runs"),
    )

    with pytest.raises(RuntimeError, match="API never became ready"):
        qualifier._restore_physical(  # pyright: ignore[reportPrivateUsage]
            controller=cast(SshTargetController, FakeController()),
            target=target,
            snapshot=snapshot,
            api_base_url="http://127.0.0.1:52415",
            report=report,
            journal=journal,
        )

    assert "SKULK_AUTO_UPDATE=0" in commands[0]
    assert commands[1] == "start selected service"
    assert commands[2] == "stop selected service"
    assert restored == [snapshot]


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
    lifecycle_failure: BaseException | None = None,
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

        def restore_original_config_files(
            self,
            snapshot: RecoverySnapshot,
        ) -> None:
            assert snapshot.original == original

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
        if lifecycle_failure is not None:
            raise lifecycle_failure
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


def test_interrupt_restores_service_but_can_never_publish_a_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed stop is a blocking verdict even when restoration is perfect."""

    report, commands, extensions, releases = _run_failed_physical_lifecycle(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        restoration_mismatches=[],
        lifecycle_failure=QualificationInterruptedError("signal 2"),
    )

    assert commands == [
        "stop selected service",
        "isolate selected target",
        "restore selected target network",
        "start selected service",
    ]
    assert report.restoration_succeeded is True
    assert report.teardown_succeeded is True
    assert report.passed is False
    assert releases == [True]
    assert extensions == []
    assert any(
        issue.severity == "error" and "interrupted" in issue.message
        for issue in report.issues
    )


def test_failed_lifecycle_stage_is_release_blocking_without_an_issue() -> None:
    report = _report_with([])
    report.lifecycle.append(
        FreshInstallLifecycleStage(
            name="start fresh runtime",
            status="failed",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            message="operator interrupted",
        )
    )

    assert [stage.name for stage in _failed_lifecycle_stages(report)] == [
        "start fresh runtime"
    ]


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


def test_runpod_ephemeral_target_preserves_served_engine_contract(
    tmp_path: Path,
) -> None:
    contract = ServedEngineContract(
        backend="llama_server-cuda",
        parallel=16,
        kv_unified=True,
        probe_concurrency=4,
    )
    target = FreshInstallTarget(
        kind="runpod",
        platform="nvidia",
        hardware_class="nvidia-cuda",
        eligible=True,
        expected_backends=["llama_server", "llama_server-cuda"],
        served_engine_contract=contract,
        vision_contract="unavailable",
        text_models=["unsloth/Llama-3.2-1B-Instruct-GGUF"],
    )
    runpod_config = RunPodFreshInstallConfig(
        ssh_public_key_file=tmp_path / "id.pub",
        ssh_private_key_file=tmp_path / "id",
        image_name="nvidia/cuda-node-neutral",
        gpu_type_ids=["NVIDIA L4"],
    )

    ephemeral = _runpod_ephemeral_target(
        target=target,
        runpod_config=runpod_config,
        endpoint=RunPodSshEndpoint(host="203.0.113.10", port=22198),
    )

    assert ephemeral.served_engine_contract == contract
    assert ephemeral.expected_backends == target.expected_backends
    assert ephemeral.text_models == target.text_models


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


def test_fresh_contract_checks_topology_and_backends_from_one_snapshot() -> None:
    """A transient peer cannot disappear between topology and backend checks."""

    class DepartingPeerClient(_StubContractClient):
        state_reads = 0

        def get_state(self) -> dict[str, object]:
            self.state_reads += 1
            state = super().get_state()
            if self.state_reads == 1:
                resources = cast(dict[str, object], state["nodeResources"])
                identities = cast(dict[str, object], state["nodeIdentities"])
                resources["departing-peer"] = {}
                identities["departing-peer"] = {}
            return state

    client = DepartingPeerClient()

    with pytest.raises(UnexpectedFreshInstallPeerError, match="observed 2"):
        qualification_checks_module.assert_fresh_runtime_contract(
            cast(SkulkClient, client),
            expected_backends=["llama_server"],
            expected_transport="zenoh",
            expected_commit=None,
        )

    assert client.state_reads == 1


def test_fresh_single_node_rejects_a_peer_joining_after_startup() -> None:
    """Isolation is a continuous invariant, not only a startup assertion."""

    class PeerJoiningClient(_StubContractClient):
        peer_joined = False

        def get_state(self) -> dict[str, object]:
            state = super().get_state()
            if self.peer_joined:
                resources = cast(dict[str, object], state["nodeResources"])
                identities = cast(dict[str, object], state["nodeIdentities"])
                resources["node-b"] = {}
                identities["node-b"] = {}
            return state

    client = PeerJoiningClient()
    expected_node_id = assert_fresh_single_node(cast(SkulkClient, client))
    client.peer_joined = True

    with pytest.raises(UnexpectedFreshInstallPeerError, match="observed 2"):
        assert_fresh_single_node(
            cast(SkulkClient, client),
            expected_node_id=expected_node_id,
        )


def test_fresh_single_node_rejects_runtime_identity_replacement() -> None:
    """A transparent fresh-process restart cannot satisfy the same leg."""

    class ReplacementClient(_StubContractClient):
        node_id = "node-a"

        def get_state(self) -> dict[str, object]:
            return {
                "nodeResources": {self.node_id: {}},
                "nodeIdentities": {self.node_id: {}},
            }

    client = ReplacementClient()
    expected_node_id = assert_fresh_single_node(cast(SkulkClient, client))
    client.node_id = "node-replacement"

    with pytest.raises(RuntimeError, match="identity changed"):
        assert_fresh_single_node(
            cast(SkulkClient, client),
            expected_node_id=expected_node_id,
        )


def test_fresh_cluster_requires_exact_stable_membership() -> None:
    class FleetClient(_StubContractClient):
        node_ids = ["node-a", "node-b", "node-c"]

        def get_state(self) -> dict[str, object]:
            return {
                "nodeResources": {
                    node_id: {
                        "backends": ["mlx"],
                        "dataTransport": "zenoh",
                    }
                    for node_id in self.node_ids
                },
                "nodeIdentities": {node_id: {} for node_id in self.node_ids},
            }

    client = FleetClient()
    expected = assert_fresh_cluster(
        cast(SkulkClient, client),
        expected_node_count=3,
    )
    assert expected == frozenset({"node-a", "node-b", "node-c"})

    client.node_ids = ["node-a", "node-b", "node-replacement"]
    with pytest.raises(RuntimeError, match="identity changed"):
        assert_fresh_cluster(
            cast(SkulkClient, client),
            expected_node_count=3,
            expected_node_ids=expected,
        )

    client.node_ids.append("incidental-node")
    with pytest.raises(UnexpectedFreshInstallPeerError, match="expected 3"):
        assert_fresh_cluster(
            cast(SkulkClient, client),
            expected_node_count=3,
            expected_node_ids=expected,
        )


def test_fresh_runtime_contract_requires_every_member_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = "32fffb7a36f9872b361c20ef47888f452211a8b6"

    class FleetContractClient(_StubContractClient):
        commits = {"node-a": "32fffb7", "node-b": "32fffb7"}

        def get_state(self) -> dict[str, object]:
            return {
                "nodeResources": {
                    "node-a": {
                        "backends": ["mlx"],
                        "dataTransport": "zenoh",
                    },
                    "node-b": {
                        "backends": ["llama_server"],
                        "dataTransport": "zenoh",
                    },
                },
                "nodeIdentities": {
                    node_id: {"skulkCommit": commit}
                    for node_id, commit in self.commits.items()
                },
            }

        def get_diagnostics_node(self) -> dict[str, object]:
            return {"runtime": {"skulkCommit": "32fffb7"}}

    monkeypatch.setattr(
        qualification_checks_module.httpx,
        "get",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            text='<html><body><div id="root"></div></body></html>',
        ),
    )
    client = FleetContractClient()
    provenance = assert_fresh_runtime_contract(
        cast(SkulkClient, client),
        expected_backends=["mlx", "llama_server"],
        expected_transport="zenoh",
        expected_commit=pinned,
        expected_node_count=2,
    )
    assert provenance.node_count == 2

    client.commits["node-b"] = "deadbee"
    with pytest.raises(RuntimeError, match="did not match the pinned candidate"):
        assert_fresh_runtime_contract(
            cast(SkulkClient, client),
            expected_backends=["mlx", "llama_server"],
            expected_transport="zenoh",
            expected_commit=pinned,
            expected_node_count=2,
        )


def test_runtime_monitor_remembers_a_transient_peer_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer that joins and leaves inside one inference still fails the leg."""

    class PeerJoiningMonitorClient(_StubContractClient):
        state_reads = 0

        def __enter__(self) -> "PeerJoiningMonitorClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_state(self) -> dict[str, object]:
            state = super().get_state()
            self.state_reads += 1
            if self.state_reads == 2:
                resources = cast(dict[str, object], state["nodeResources"])
                identities = cast(dict[str, object], state["nodeIdentities"])
                resources["transient-peer"] = {}
                identities["transient-peer"] = {}
            return state

    monkeypatch.setattr(
        fresh_install_module,
        "SkulkClient",
        lambda *_args, **_kwargs: PeerJoiningMonitorClient(),
    )
    monitor = fresh_install_module._FreshRuntimeMonitor(  # pyright: ignore[reportPrivateUsage]
        api_base_url="http://127.0.0.1:52415",
        expected_node_id="node-a",
        poll_interval_s=1,
        request_timeout_s=1,
    )
    monitor._poll_interval_s = 0.001  # pyright: ignore[reportPrivateUsage]

    with (
        pytest.raises(UnexpectedFreshInstallPeerError, match="observed 2"),
        monitor,
    ):
        time.sleep(0.05)


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


def test_headless_provisioning_checks_runtime_during_long_polls() -> None:
    """A peer arriving during a download must abort before placement."""

    model_id = "unsloth/Llama-3.2-1B-Instruct-GGUF"
    client = _StubProvisionClient(
        download_states=["downloading", "complete"],
        catalog_model_ids=[model_id],
    )
    checks = 0

    def fail_after_download_starts() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise UnexpectedFreshInstallPeerError("peer joined")

    with pytest.raises(UnexpectedFreshInstallPeerError, match="peer joined"):
        _provision_model_over_api(
            cast(SkulkClient, client),
            model_id=model_id,
            model_ready_timeout_s=30,
            poll_interval_s=0,
            heartbeat=None,
            runtime_check=fail_after_download_starts,
        )

    assert "place_model" not in client.calls


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


def _report_with(issues: list[Issue]) -> FreshInstallQualificationReport:
    """Build a minimal finished report carrying the given issues."""

    return FreshInstallQualificationReport(
        qualification_id="fresh-candidate-test",
        profile="candidate",
        platform="apple",
        hardware_class="test-class",
        started_at=datetime.now(UTC),
        install=InstallProvenance(
            mode="fresh_install",
            environment="fresh_install",
            data_transport="zenoh",
            node_count=1,
        ),
        issues=issues,
    )


def test_a_warning_does_not_fail_a_leg_that_otherwise_passed() -> None:
    """Only an error is a release gate; a warning is operator information.

    The fleet-size comparison was deliberately downgraded from a fatal
    restoration failure to a warning so a healthy target would stop being
    declared broken by the state of machines outside the experiment. Failing
    on any issue at all undid that: a leg whose every stage passed, whose
    target restored cleanly, and whose only finding was that warning was
    still reported as a failed release gate. This pins the distinction the
    downgrade was supposed to make.
    """

    warned = _report_with(
        [
            Issue(
                severity="warning",
                message=(
                    "restored fleet is smaller than before the run: 2 -> 1 "
                    "nodes; the target itself restored cleanly"
                ),
            )
        ]
    )
    assert _blocking_issues(warned) == []

    informational = _report_with([Issue(severity="info", message="noted")])
    assert _blocking_issues(informational) == []

    failed = _report_with(
        [
            Issue(severity="warning", message="fleet came back smaller"),
            Issue(severity="error", message="fresh-install leg failed"),
        ]
    )
    blocking = _blocking_issues(failed)
    assert [issue.severity for issue in blocking] == ["error"]


class _StubConversationLocator:
    """Stand in for a Playwright locator used by the conversation reset."""

    def __init__(self, *, matches: int = 1) -> None:
        self.matches = matches
        self.clicks = 0
        self.selected: list[str] = []
        self.waited_states: list[str] = []

    def count(self) -> int:
        return self.matches

    def filter(self, *, visible: bool) -> "_StubConversationLocator":
        assert visible is True
        return self

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


class _StubSelectedComposer:
    """Composer proving the sole ready model is already selected."""

    def __init__(self, *, count: int, placeholder: str = "") -> None:
        self.matches = count
        self.placeholder = placeholder
        self.selected: list[str] = []

    def count(self) -> int:
        return self.matches

    def wait_for(self, *, state: str, timeout: float) -> None:
        assert state == "visible"
        assert timeout == 30_000

    def select_option(self, value: str) -> None:
        self.selected.append(value)

    def get_attribute(self, name: str) -> str:
        assert name == "placeholder"
        return self.placeholder


class _StubSelectedModelPage:
    """Chat page with either the explicit select or the sole-model composer."""

    def __init__(self, *, selector_count: int, placeholder: str) -> None:
        self.selector = _StubSelectedComposer(count=selector_count)
        self.message = _StubSelectedComposer(count=1, placeholder=placeholder)

    def get_by_label(self, label: str, *, exact: bool) -> _StubSelectedComposer:
        assert exact is True
        if label == "Select chat model":
            return self.selector
        if label == "Chat message":
            return self.message
        raise AssertionError(f"unexpected label {label!r}")


def test_chat_model_selection_accepts_the_sole_ready_model_composer() -> None:
    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )
    custom_selector_page = _StubSelectedModelPage(
        selector_count=0,
        placeholder="Message Qwen3.5-2B-4bit...",
    )
    qualifier._select_chat_model(  # pyright: ignore[reportPrivateUsage]
        cast(Page, custom_selector_page),
        model_id="mlx-community/Qwen3.5-2B-4bit",
    )

    explicit_selector_page = _StubSelectedModelPage(
        selector_count=1,
        placeholder="",
    )
    qualifier._select_chat_model(  # pyright: ignore[reportPrivateUsage]
        cast(Page, explicit_selector_page),
        model_id="org/model",
    )
    assert explicit_selector_page.selector.selected == ["org/model"]


class _StubPersistedMessage:
    """One visible user or assistant message after a dashboard reload."""

    def __init__(self, text: str, *, attachment_name: str | None = None) -> None:
        self.text = text
        self.attachment_name = attachment_name

    def filter(self, *, visible: bool) -> "_StubPersistedMessage":
        assert visible is True
        return self

    @property
    def last(self) -> "_StubPersistedMessage":
        return self

    def wait_for(self, *, state: str, timeout: float) -> None:
        assert state == "visible"
        assert timeout == 30_000

    def inner_text(self) -> str:
        return self.text

    def get_by_alt_text(self, name: str) -> "_StubPersistedAttachment":
        return _StubPersistedAttachment(name == self.attachment_name)


class _StubPersistedAttachment:
    """Attachment locator whose visibility records persisted preview bytes."""

    def __init__(self, visible: bool) -> None:
        self.visible = visible

    def count(self) -> int:
        return int(self.visible)

    def is_visible(self) -> bool:
        return self.visible


class _StubPersistencePage:
    """Dashboard page exposing one persisted active conversation."""

    def __init__(self, *, attachment_name: str | None) -> None:
        self.user = _StubPersistedMessage(
            "Read qualification.png and report a green circle.",
            attachment_name=attachment_name,
        )
        self.assistant = _StubPersistedMessage("AB12CD green circle")
        self.reloads = 0

    def reload(self, *, wait_until: str) -> None:
        assert wait_until == "networkidle"
        self.reloads += 1

    def get_by_label(self, label: str, *, exact: bool) -> _StubPersistedMessage:
        assert exact is True
        if label == "User message":
            return self.user
        if label == "Assistant message":
            return self.assistant
        raise AssertionError(f"unexpected label {label!r}")


def test_dashboard_reload_requires_text_and_attachment_persistence() -> None:
    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )
    page = _StubPersistencePage(attachment_name="qualification.png")

    conversation, attachment = qualifier._verify_conversation_persistence(  # pyright: ignore[reportPrivateUsage]
        cast(Page, page),
        expected_user_text="green circle",
        expected_assistant_text="AB12CD",
        attachment_name="qualification.png",
    )

    assert page.reloads == 1
    assert conversation is True
    assert attachment is True

    missing_attachment = _StubPersistencePage(attachment_name=None)
    conversation, attachment = qualifier._verify_conversation_persistence(  # pyright: ignore[reportPrivateUsage]
        cast(Page, missing_attachment),
        expected_user_text="green circle",
        expected_assistant_text="AB12CD",
        attachment_name="qualification.png",
    )
    assert conversation is True
    assert attachment is False


class _StubStreamingAssistant:
    """Assistant locator whose text grows while generation is active."""

    def __init__(self, page: "_StubStreamingPage") -> None:
        self.page = page

    def count(self) -> int:
        return 1

    def filter(self, *, visible: bool) -> "_StubStreamingAssistant":
        assert visible is True
        return self

    def nth(self, index: int) -> "_StubStreamingAssistant":
        assert index == 0
        return self

    def inner_text(self) -> str:
        if self.page.polls < 2:
            return "UJUEUC"
        return "UJUEUC cyan circle"


class _StubStreamingCancel:
    """Cancel control that disappears only after the final stream update."""

    def __init__(self, page: "_StubStreamingPage") -> None:
        self.page = page

    def count(self) -> int:
        return 1 if self.page.polls < 2 else 0


class _StubStreamingPage:
    """Dashboard page exposing a partial answer before generation completes."""

    def __init__(self) -> None:
        self.polls = 0
        self.assistant = _StubStreamingAssistant(self)
        self.cancel = _StubStreamingCancel(self)

    def get_by_label(self, label: str, *, exact: bool) -> _StubStreamingAssistant:
        assert (label, exact) == ("Assistant message", True)
        return self.assistant

    def get_by_role(self, role: str, *, name: str, exact: bool) -> _StubStreamingCancel:
        assert (role, name, exact) == ("button", "Cancel generation", True)
        return self.cancel

    def wait_for_timeout(self, milliseconds: float) -> None:
        assert milliseconds == 500
        self.polls += 1


class _StubHiddenHistoryAssistant:
    """Assistant locator with a current reply plus hidden conversation history."""

    def __init__(self, *, current_only: bool = False) -> None:
        self.current_only = current_only

    def filter(self, *, visible: bool) -> "_StubHiddenHistoryAssistant":
        assert visible is True
        return _StubHiddenHistoryAssistant(current_only=True)

    def count(self) -> int:
        return 1 if self.current_only else 2

    def nth(self, index: int) -> "_StubHiddenHistoryAssistantMessage":
        if self.current_only:
            assert index == 0
            return _StubHiddenHistoryAssistantMessage("PVNA7M amber circle")
        assert index in (0, 1)
        return _StubHiddenHistoryAssistantMessage(
            "PVNA7M amber circle" if index == 0 else "old unrelated reply"
        )


class _StubHiddenHistoryAssistantMessage:
    """One assistant message from the visible or hidden conversation."""

    def __init__(self, text: str) -> None:
        self.text = text

    def inner_text(self) -> str:
        return self.text


class _StubHiddenHistoryPage:
    """Page that keeps a prior conversation mounted but hidden."""

    def __init__(self) -> None:
        self.assistant = _StubHiddenHistoryAssistant()
        self.cancel = _StubAbsentCancel()

    def get_by_label(self, label: str, *, exact: bool) -> _StubHiddenHistoryAssistant:
        assert (label, exact) == ("Assistant message", True)
        return self.assistant

    def get_by_role(self, role: str, *, name: str, exact: bool) -> "_StubAbsentCancel":
        assert (role, name, exact) == ("button", "Cancel generation", True)
        return self.cancel

    def wait_for_timeout(self, milliseconds: float) -> None:
        assert milliseconds == 500


class _StubAbsentCancel:
    """Cancel control after the current response has completed."""

    def count(self) -> int:
        return 0


def test_assistant_wait_requires_stream_completion() -> None:
    """Seeing the hidden code must not capture a partial vision response."""

    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )
    page = _StubStreamingPage()

    response = qualifier._wait_for_assistant(  # pyright: ignore[reportPrivateUsage]
        cast(Page, page),
        expected="UJUEUC",
    )

    assert page.polls == 2
    assert response == "UJUEUC cyan circle"


def test_assistant_wait_ignores_hidden_conversation_history() -> None:
    """The active reply must win over mounted cards from hidden threads."""

    qualifier = DashboardQualifier(
        api_base_url="http://example.invalid",
        artifact_directory=Path("unused"),
        poll_interval_s=1,
        model_ready_timeout_s=1,
    )
    page = _StubHiddenHistoryPage()

    response = qualifier._wait_for_assistant(  # pyright: ignore[reportPrivateUsage]
        cast(Page, page),
        expected="PVNA7M",
    )

    assert response == "PVNA7M amber circle"


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

    prompt = echo_prompt("amber harbor 1234")
    assert "hotel guest" in prompt
    assert "hotel is named Amber Harbor" in prompt
    assert "room number is 1234" in prompt
    assert prompt.endswith("Mention both Amber Harbor and room 1234 in your sentence.")
    assert "token" not in prompt.lower()
    assert "exactly" not in prompt.lower()
    assert "say nothing else" not in prompt.lower()
    assert "complete text" not in prompt.lower()


def test_echo_phrase_is_unpredictable() -> None:
    """A stale or replayed response must not be able to satisfy the check."""

    assert len({echo_phrase() for _ in range(200)}) > 150


def test_dashboard_transcript_match_tolerates_punctuation_not_wrong_words() -> None:
    assert _transcript_matches(
        "Hello world from the Skulk dashboard.",
        "hello, world from the skulk dashboard",
    )
    assert _transcript_matches(
        "Hello world from the Skulk dashboard.",
        "Hello world from Skulk dashboard.",
    )
    assert not _transcript_matches(
        "Hello world from the Skulk dashboard.",
        "The weather is warm at the hotel.",
    )
    assert not _transcript_matches(
        "release audio bravo hotel seven cedar",
        "release audio bravo hotel seven cedar release audio bravo hotel seven cedar",
    )


def test_dashboard_stt_fixture_is_non_silent_and_stops_before_looping() -> None:
    fixture = (
        Path(dashboard_qualification_module.__file__).parent
        / "fixtures"
        / "dashboard-stt-release.wav"
    )

    duration_s, rms = _pcm_wav_duration_and_rms(fixture.read_bytes())

    assert duration_s == pytest.approx(2.793, abs=0.001)
    assert rms > 100
    assert 2_500 <= _fake_microphone_recording_ms(fixture) < 2_793


def test_dashboard_cleanup_closes_browser_when_artifact_capture_fails(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakePage:
        def screenshot(self, **_kwargs: object) -> None:
            calls.append("screenshot")
            raise RuntimeError("page already closed")

    class FakeTracing:
        def stop(self, **_kwargs: object) -> None:
            calls.append("trace")

    class FakeContext:
        tracing = FakeTracing()

    class FakeBrowser:
        def close(self) -> None:
            calls.append("browser")

    error = _capture_and_close_browser(
        cast(Page, FakePage()),
        cast(BrowserContext, FakeContext()),
        cast(Browser, FakeBrowser()),
        screenshot_path=tmp_path / "final.png",
        trace_path=tmp_path / "trace.zip",
    )

    assert isinstance(error, RuntimeError)
    assert calls == ["screenshot", "trace", "browser"]


def test_echo_match_accepts_a_recapitalized_reply() -> None:
    """A model that capitalizes its reply has still proven the chat path works.

    The browser waiter already returns on a case-insensitive match, so a
    case-sensitive assertion afterwards would reject a response the wait had
    declared good.
    """

    assert echo_matched("amber harbor 4821", "Amber Harbor 4821")
    assert echo_matched("amber harbor 4821", "Sure! amber harbor 4821")
    assert echo_matched(
        "amber harbor 4821",
        "The harbor looked amber when flight 4821 arrived.",
    )
    assert not echo_matched("amber harbor 4821", "amber harbor 4822")
    assert not echo_matched("amber harbor 4821", "amber harbor 14821")


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
