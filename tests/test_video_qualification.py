"""LTX qualification evidence must describe the actual scoped render."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError
from typer.testing import CliRunner

from skulk_test_harness import video_qualification as video
from skulk_test_harness.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
JSON = TypeAdapter(JsonValue)


def _load(name: str) -> dict[str, JsonValue]:
    result = JSON.validate_json((FIXTURES / name).read_bytes())
    assert isinstance(result, dict)
    return result


def _section(parent: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    result = parent[key]
    assert isinstance(result, dict)
    return result


def _task(state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _section(_section(_section(state, "tasks"), "task"), "VideoGeneration")


def _shard(state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    instance = _section(
        _section(_section(state, "instances"), "placement"), "MlxRingInstance"
    )
    assignments = _section(instance, "shardAssignments")
    return _section(
        _section(_section(assignments, "runnerToShard"), "runner"),
        "PipelineShardMetadata",
    )


def _profile(state: dict[str, JsonValue]) -> video.VideoProfile:
    card = _section(_shard(state), "modelCard")
    digest = hashlib.sha256(
        json.dumps(card, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return video.VideoProfile(
        api_url="http://candidate.test",
        core_commit="a" * 40,
        node_ids=("api", "worker"),
        api_node_id="api",
        worker_node_id="worker",
        model="Lightricks/LTX-2.5-distilled-comfy-int8",
        card_sha256=digest,
        backend="comfy-rocm",
        engine_build="fixture-build",
    )


def test_actual_core_wire_and_resolved_schedule_are_accepted() -> None:
    state = _load("ltx25-core-state.json")
    profile = _profile(state)
    proof = video.video_execution_proof(state, profile, "job", video.BASELINE[0], {})
    assert proof is not None
    _task(state)["taskStatus"] = "Running"
    assert (
        video.video_execution_proof(state, profile, "job", video.BASELINE[0], {})
        == proof
    )
    assert (
        video.verify_video_result(_load("ltx25-core-result.json"), video.BASELINE[0])[
            "frame_count"
        ]
        == 49
    )


@pytest.mark.parametrize(
    "fault",
    [
        "peer",
        "commit",
        "engine",
        "transport",
        "card",
        "owner",
        "seed",
        "backend",
        "duplicate",
    ],
)
def test_wrong_execution_is_refused(fault: str) -> None:
    state = _load("ltx25-core-state.json")
    profile = _profile(state)
    if fault == "peer":
        _section(state, "nodeIdentities")["foreign"] = {}
    elif fault == "commit":
        _section(_section(state, "nodeIdentities"), "api")["skulkCommit"] = "unknown"
    elif fault == "engine":
        _section(_section(_section(state, "nodeResources"), "worker"), "engineBuilds")[
            "comfy-rocm"
        ] = "other-build"
    elif fault == "transport":
        _section(_section(state, "nodeResources"), "worker")["dataTransport"] = "libp2p"
    elif fault == "card":
        _section(_shard(state), "modelCard")["sourceRevision"] = "b" * 40
    elif fault == "owner":
        _task(state)["ownerNode"] = "worker"
    elif fault == "seed":
        _section(_task(state), "taskParams")["seed"] = 99
    elif fault == "backend":
        _shard(state)["resolvedBackend"] = "comfy-cuda"
    else:
        _section(state, "tasks")["duplicate"] = _section(state, "tasks")["task"]
    with pytest.raises(ValueError):
        video.video_execution_proof(state, profile, "job", video.BASELINE[0], {})


def test_no_task_cannot_be_inferred_from_current_placement() -> None:
    state = _load("ltx25-core-state.json")
    profile = _profile(state)
    state["tasks"] = {}
    assert (
        video.video_execution_proof(state, profile, "job", video.BASELINE[0], {})
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [("width", 768), ("frame_count", 73), ("audio_channels", 1), ("size_bytes", 0)],
)
def test_wrong_actual_media_is_refused(field: str, value: int) -> None:
    job = _load("ltx25-core-result.json")
    _section(job, "output")[field] = value
    with pytest.raises(ValueError):
        video.verify_video_result(job, video.BASELINE[0])


def test_cached_render_is_not_a_timing_measurement() -> None:
    job = _load("ltx25-core-result.json")
    _section(job, "stats")["seconds_per_step"] = 0.0
    with pytest.raises(ValueError, match="cache hit"):
        video.verify_video_result(job, video.BASELINE[0])


class _Api:
    """Wire-level API fixture using the serialized real Core task/shard shapes."""

    def __init__(self, *, fault: str = "") -> None:
        self.state = _load("ltx25-core-state.json")
        self.profile = _profile(self.state)
        self.fault = fault
        self.requests: list[tuple[str, str]] = []
        self.jobs = 0
        self.fields: dict[str, JsonValue] = {}
        self.references: list[JsonValue] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer actual submitted fields using Core's serialized wire shapes."""
        self.requests.append((request.method, request.url.path))
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"cancelled": True})
        if request.method == "POST":
            assert request.url.path == "/v1/videos"
            self.jobs += 1
            self.references = []
            if request.headers.get("content-type", "").startswith("multipart/"):
                message = BytesParser(policy=policy.default).parsebytes(
                    (
                        "Content-Type: " + request.headers["content-type"] + "\r\n\r\n"
                    ).encode()
                    + request.read()
                )
                self.fields = {}
                for part in message.walk():
                    if part.is_multipart():
                        continue
                    name = part.get_param("name", header="content-disposition")
                    assert isinstance(name, str)
                    payload = part.get_payload(decode=True)
                    assert isinstance(payload, bytes)
                    if part.get_filename():
                        self.references.append(
                            {
                                "role": name,
                                "kind": "image",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "sizeBytes": len(payload),
                                "atSeconds": 1.0 if name == "keyframe" else None,
                            }
                        )
                    else:
                        self.fields[name] = (
                            JSON.validate_json(payload)
                            if name in {"seconds", "steps", "seed", "audio"}
                            else payload.decode()
                        )
            else:
                fields = JSON.validate_json(request.content)
                assert isinstance(fields, dict)
                self.fields = fields
            return httpx.Response(
                200, json={"id": f"job-{self.jobs}", "model": self.profile.model}
            )
        if request.url.path == "/v1/diagnostics/node":
            return httpx.Response(200, json={"runtime": {"nodeId": "api"}})
        if request.url.path == "/state":
            if self.jobs:
                task = _task(self.state)
                task["commandId"] = f"job-{self.jobs}"
                params = _section(task, "taskParams")
                params.update(
                    {
                        key: value
                        for key, value in self.fields.items()
                        if key != "keyframe_at"
                    }
                )
                params["references"] = self.references
                if self.fault == "references" and self.references:
                    first = self.references[0]
                    assert isinstance(first, dict)
                    first["sha256"] = "0" * 64
                if self.fault == "keyframe" and len(self.references) == 3:
                    third = self.references[2]
                    assert isinstance(third, dict)
                    third["atSeconds"] = 0.0
                if self.fault == "scope":
                    _section(self.state, "nodeIdentities")["unexpected"] = {}
                if self.fault == "proof":
                    return httpx.Response(200, json={**self.state, "tasks": {}})
            return httpx.Response(200, json=self.state)
        if request.url.path.endswith("/content"):
            return httpx.Response(
                200,
                content=b"wrong-content"
                if self.fault == "digest"
                else b"fixture-video",
            )
        job = _load("ltx25-core-result.json")
        job["id"] = "foreign-job" if self.fault == "job" else f"job-{self.jobs}"
        _section(_section(job, "stats"), "engine")["seed"] = self.fields["seed"]
        if self.fault == "timeout":
            job["status"] = "in_progress"
        return httpx.Response(200, json=job)


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda name: f"/fixture/{name}")
    monkeypatch.setattr(
        video,
        "probe_video",
        lambda path, **_kwargs: {"checked_bytes": len(path.read_bytes())},
    )


def test_all_cells_submit_real_json_and_multipart_and_retain_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(monkeypatch)
    api = _Api()
    with httpx.Client(
        base_url=api.profile.api_url, transport=httpx.MockTransport(api.handle)
    ) as client:
        video.VideoApiQualification(api.profile, client).run(tmp_path / "evidence")
    receipt = json.loads((tmp_path / "evidence/receipt.json").read_text())
    assert receipt["status"] == "api_checks_passed"
    assert receipt["remote_media"] is True
    assert receipt["completed_cells"] == [cell.name for cell in video.BASELINE]
    assert api.jobs == 5
    assert {path for method, path in api.requests if method != "GET"} == {"/v1/videos"}
    assert (
        json.loads((tmp_path / "evidence/text/checks.json").read_text())[
            "external_peak_memory"
        ]
        == "unmeasured"
    )
    assert (
        json.loads((tmp_path / "evidence/timed-keyframe/execution.json").read_text())[
            "parameters"
        ]["references"][2]["atSeconds"]
        == 1.0
    )


@pytest.mark.parametrize(
    "fault", ["digest", "job", "timeout", "references", "keyframe", "scope", "proof"]
)
def test_failures_keep_partial_evidence_and_cancel_only_owned_active_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _prepare(monkeypatch)
    api = _Api(fault=fault)
    elapsed = [0.0]

    def sleep(seconds: float) -> None:
        elapsed[0] += seconds

    profile = api.profile.model_copy(update={"timeout_seconds": 1})
    with (
        httpx.Client(
            base_url=profile.api_url, transport=httpx.MockTransport(api.handle)
        ) as client,
        pytest.raises((ValueError, TimeoutError)),
    ):
        video.VideoApiQualification(
            profile, client, clock=lambda: elapsed[0], sleep=sleep
        ).run(tmp_path / "evidence")
    receipt = json.loads((tmp_path / "evidence/receipt.json").read_text())
    assert receipt["status"] == "failed"
    cancellations = [
        path
        for method, path in api.requests
        if method == "POST" and path.endswith("/cancel")
    ]
    assert cancellations == (
        [] if fault in {"digest", "proof"} else [f"/v1/videos/job-{api.jobs}/cancel"]
    )
    assert not any(path.endswith("foreign-job/cancel") for path in cancellations)


def test_existing_evidence_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(monkeypatch)
    api = _Api()
    with (
        httpx.Client(
            base_url=api.profile.api_url, transport=httpx.MockTransport(api.handle)
        ) as client,
        pytest.raises(FileExistsError),
    ):
        video.VideoApiQualification(api.profile, client).run(tmp_path)
    assert api.requests == []


def test_plan_is_offline_and_rejects_cpu_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(_load("ltx25-core-state.json"))
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json())
    result = CliRunner().invoke(app, ["video", "plan", "--profile", str(path)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["cells"]) == 5
    with pytest.raises(ValidationError):
        video.VideoProfile.model_validate_json(
            profile.model_dump_json().replace('"comfy-rocm"', '"comfy-cpu"')
        )


@pytest.mark.skipif(
    not (os.environ.get("SKULK_TEST_FFMPEG") or shutil.which("ffmpeg"))
    or not (os.environ.get("SKULK_TEST_MEDIA_PYTHON") or shutil.which("ffprobe")),
    reason="media tools unavailable",
)
@pytest.mark.parametrize("channels", ["stereo", "mono"])
def test_real_full_decode_and_audio_contract(tmp_path: Path, channels: str) -> None:
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            os.environ.get("SKULK_TEST_FFMPEG", "ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=512x512:r=24",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl={channels}",
            "-frames:v",
            "49",
            "-t",
            str(49 / 24),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    if channels == "stereo":
        assert video.probe_video(
            path, interpreter=os.environ.get("SKULK_TEST_MEDIA_PYTHON")
        )["streams"]
    else:
        with pytest.raises(ValueError, match="sample rate/channels"):
            video.probe_video(
                path, interpreter=os.environ.get("SKULK_TEST_MEDIA_PYTHON")
            )
