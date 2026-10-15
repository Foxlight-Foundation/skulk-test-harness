"""Reproducible LTX baseline API checks, independent of hardware qualification.

This operator tool consumes an already placed, isolated candidate. It never
installs engines, downloads weights, changes placements or lowers admission.
Passing these checks is not perceptual, lifecycle or peak-memory acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, final
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image, ImageDraw
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

_JSON = TypeAdapter(JsonValue)
_MAX_JSON_BYTES = 16 * 1024**2
_MAX_VIDEO_BYTES = 512 * 1024**2
_PROMPT = (
    "A red ball rolls slowly from left to right across a blue tabletop. "
    "The camera stays fixed. Soft rolling sounds and three gentle wooden taps."
)
_PYAV_PROBE = """
import av,json,sys
with av.open(sys.argv[1]) as container:
    if len(container.streams) != 2 or len(container.streams.video) != 1 or len(container.streams.audio) != 1:
        raise ValueError('expected one video and one audio stream')
    video, audio = container.streams.video[0], container.streams.audio[0]
    frames = samples = 0
    for packet in container.demux(video, audio):
        for frame in packet.decode():
            if isinstance(frame, av.VideoFrame):
                frames += 1
                if frames > 49:
                    raise ValueError('video exceeds baseline frame count')
            elif isinstance(frame, av.AudioFrame):
                samples += frame.samples
                if samples > 3 * 48000:
                    raise ValueError('audio exceeds baseline duration')
    if frames != 49 or abs(samples / 48000 - 49 / 24) > 1 / 24:
        raise ValueError('decoded media does not match baseline duration')
    if not video.duration or not video.time_base or not audio.duration or not audio.time_base:
        raise ValueError('missing stream durations')
    print(json.dumps({'format': {'format_name': container.format.name}, 'streams': [
        {'codec_type': 'video', 'codec_name': video.codec_context.codec.canonical_name,
         'width': video.codec_context.width, 'height': video.codec_context.height,
         'nb_read_frames': str(frames), 'avg_frame_rate': str(video.average_rate),
         'duration': str(float(video.duration * video.time_base))},
        {'codec_type': 'audio', 'codec_name': audio.codec_context.codec.canonical_name,
         'sample_rate': str(audio.codec_context.sample_rate), 'channels': audio.codec_context.channels,
         'duration': str(float(audio.duration * audio.time_base)), 'decoded_samples': samples}
    ]}))
"""


@final
class VideoProfile(BaseModel):
    """An operator-reviewed exact candidate and isolated API/worker scope."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    api_url: str = Field(description="Direct URL of the isolated owning API.")
    core_commit: str = Field(
        pattern=r"^[a-f0-9]{40}$", description="Accepted Core commit."
    )
    node_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
        description="Exact isolated cluster membership, including the API and worker.",
    )
    worker_node_id: str = Field(
        description="Only node permitted to render this baseline."
    )
    api_node_id: str = Field(description="Expected identity of the owning API node.")
    model: str = Field(description="Exact selected distilled LTX-2.5 card identifier.")
    card_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 of the complete card's canonical JSON as serialized on /state.",
    )
    backend: Literal["comfy-cuda", "comfy-rocm", "mlx_video-metal"] = Field(
        description="GPU runner lane to exercise."
    )
    engine_build: str = Field(
        min_length=1,
        description="Exact installed engine identity expected on the worker.",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=1,
        le=10800,
        description="Deadline for each job, including ordinary admission.",
    )
    media_python: str | None = Field(
        default=None,
        description="Optional local verified engine Python carrying PyAV for full media inspection; otherwise ffmpeg/ffprobe must be on PATH. Never used for rendering.",
    )

    @model_validator(mode="after")
    def _validate_scope(self) -> VideoProfile:
        if (
            self.worker_node_id not in self.node_ids
            or self.api_node_id not in self.node_ids
            or len(set(self.node_ids)) != len(self.node_ids)
        ):
            raise ValueError("worker must belong to a unique, explicit node scope")
        parsed = urlsplit(self.api_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "api_url must be a direct HTTP(S) origin without credentials"
            )
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("invalid API port")
        return self


@dataclass(frozen=True)
class VideoCell:
    """One fixed 512-square, 49-frame distilled baseline render."""

    name: str
    seed: int
    parts: tuple[tuple[str, str], ...] = ()

    def body(self, model: str) -> dict[str, JsonValue]:
        """Return LTX request fields without borrowing H3 sampler defaults."""
        body: dict[str, JsonValue] = {
            "model": model,
            "prompt": _PROMPT,
            "seconds": 2,
            "size": "512x512",
            "seed": self.seed,
            "audio": True,
            "steps": 8,
            "pipeline": "distilled_two_stage",
            "decoder": "convolutional",
            "mode": "fl2va" if self.parts else "t2va",
            "codec": "h264",
        }
        if any(role == "keyframe" for role, _ in self.parts):
            body["keyframe_at"] = "1.0"
        return body


BASELINE = (
    VideoCell("text", 7),
    # A distinct seed forces actual sampling instead of a Comfy cache hit.
    VideoCell("text-repeat", 8),
    VideoCell("first-frame", 9, (("first_frame", "first.png"),)),
    VideoCell(
        "first-last", 10, (("first_frame", "first.png"), ("last_frame", "last.png"))
    ),
    VideoCell(
        "timed-keyframe",
        11,
        (
            ("first_frame", "first.png"),
            ("last_frame", "last.png"),
            ("keyframe", "middle.png"),
        ),
    ),
)


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: JsonValue, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _save(path: Path, value: JsonValue) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_video_scope(state: dict[str, JsonValue], profile: VideoProfile) -> None:
    """Refuse unexpected peers, unknown builds or an uninstalled target engine."""
    identities = _object(state.get("nodeIdentities"), "identities")
    resources = _object(state.get("nodeResources"), "resources")
    if set(identities) != set(profile.node_ids) or set(resources) != set(
        profile.node_ids
    ):
        raise ValueError("isolated cluster membership differs from the reviewed scope")
    for identity in identities.values():
        commit = _text(_object(identity, "identity").get("skulkCommit"), "commit")
        if not re.fullmatch(
            r"[a-f0-9]{8,40}", commit
        ) or not profile.core_commit.startswith(commit):
            raise ValueError("node does not report the accepted Core commit")
    if any(
        _object(item, "node resources").get("dataTransport") != "zenoh"
        for item in resources.values()
    ):
        raise ValueError("baseline requires the shipping Zenoh data transport")
    worker = _object(resources[profile.worker_node_id], "worker resources")
    builds = _object(worker.get("engineBuilds"), "engine builds")
    if builds.get(profile.backend) != profile.engine_build:
        raise ValueError("worker installed engine differs from the reviewed build")


def _instance_proof(instance: JsonValue, profile: VideoProfile) -> dict[str, JsonValue]:
    tagged = _object(instance, "instance")
    if len(tagged) != 1:
        raise ValueError("ambiguous instance tag")
    body = _object(next(iter(tagged.values())), "instance body")
    shards = _object(body.get("shardAssignments"), "shards")
    if shards.get("modelId") != profile.model:
        raise ValueError("task is bound to another model")
    mapping = _object(shards.get("nodeToRunner"), "node-to-runner mapping")
    if set(mapping) != {profile.worker_node_id}:
        raise ValueError("render did not use the intended single worker")
    runner_id = _text(mapping[profile.worker_node_id], "runner id")
    runners = _object(shards.get("runnerToShard"), "runner shards")
    if set(runners) != {runner_id}:
        raise ValueError("render includes unexpected shards")
    tagged_shard = _object(runners[runner_id], "shard tag")
    if len(tagged_shard) != 1:
        raise ValueError("ambiguous shard tag")
    shard = _object(next(iter(tagged_shard.values())), "shard")
    stamped_build = shard.get("resolvedEngineBuild")
    # Core stamps native sidecar builds. Comfy's installed build is instead
    # proven by live node resources, checked again on every observation.
    if (
        shard.get("resolvedBackend") != profile.backend
        or (stamped_build is not None and stamped_build != profile.engine_build)
        or (
            profile.backend == "mlx_video-metal"
            and stamped_build != profile.engine_build
        )
    ):
        raise ValueError("bound shard does not name the reviewed engine")
    card = _object(shard.get("modelCard"), "bound card")
    if hashlib.sha256(_canonical(card).encode()).hexdigest() != profile.card_sha256:
        raise ValueError("bound complete card differs from the reviewed artifact")
    video = _object(card.get("video"), "video card")
    if (
        card.get("family") != "ltx-2.5"
        or video.get("transformerVariant") != "distilled"
    ):
        raise ValueError("baseline requires an exact distilled LTX-2.5 card")
    return {
        "worker_node_id": profile.worker_node_id,
        "runner_id": runner_id,
        "installed_build": profile.engine_build,
        "shard": shard,
    }


def video_execution_proof(
    state: dict[str, JsonValue],
    profile: VideoProfile,
    job_id: str,
    cell: VideoCell,
    inputs: dict[str, tuple[str, int]],
) -> dict[str, JsonValue] | None:
    """Bind an observed command to its actual task, parameters and complete shard."""
    verify_video_scope(state, profile)
    tasks = _object(state.get("tasks"), "tasks")
    matches: list[dict[str, JsonValue]] = []
    for tagged in tasks.values():
        task_tag = _object(tagged, "task tag")
        if set(task_tag) == {"VideoGeneration"}:
            task = _object(task_tag["VideoGeneration"], "video task")
            if task.get("commandId") == job_id:
                matches.append(task)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("ambiguous generation command")
    task = matches[0]
    if task.get("ownerNode") != profile.api_node_id:
        raise ValueError("task is not owned by the reviewed API node")
    params = _object(task.get("taskParams"), "task parameters")
    for key, value in cell.body(profile.model).items():
        if key != "keyframe_at" and params.get(key) != value:
            raise ValueError(f"actual task differs at {key}")
    references = params.get("references", [])
    if not isinstance(references, list):
        raise ValueError("actual references must be a list")
    roles = [_object(item, "reference").get("role") for item in references]
    if sorted(str(role) for role in roles) != sorted(role for role, _ in cell.parts):
        raise ValueError("actual reference roles differ from the request")
    for role, name in cell.parts:
        reference = next(
            _object(item, "reference")
            for item in references
            if _object(item, "reference").get("role") == role
        )
        digest, size = inputs[name]
        if reference.get("sha256") != digest or reference.get("sizeBytes") != size:
            raise ValueError("actual conditioning bytes differ from the fixtures")
        if role == "keyframe" and reference.get("atSeconds") != 1.0:
            raise ValueError("actual keyframe time differs from the request")
    instance_id = _text(task.get("instanceId"), "instance id")
    instances = _object(state.get("instances"), "instances")
    proof = _instance_proof(instances.get(instance_id), profile)
    # Task status changes during normal dispatch. Only immutable execution
    # inputs belong in the identity compared across observations.
    return {
        **proof,
        "parameters": params,
        "owner_node": task.get("ownerNode"),
        "instance_id": instance_id,
    }


def make_video_fixtures(directory: Path) -> dict[str, str]:
    """Create deterministic conditioning images and return their SHA-256 digests."""
    directory.mkdir()
    digests: dict[str, str] = {}
    for name, center in (("first", 96), ("middle", 256), ("last", 416)):
        image = Image.new("RGB", (512, 512), (32, 72, 160))
        ImageDraw.Draw(image).ellipse(
            (center - 48, 208, center + 48, 304), fill=(225, 38, 38)
        )
        path = directory / f"{name}.png"
        image.save(path)
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def verify_video_result(
    job: dict[str, JsonValue],
    cell: VideoCell,
    *,
    backend: Literal["comfy-cuda", "comfy-rocm", "mlx_video-metal"],
) -> dict[str, JsonValue]:
    """Require a sampled 8+3 render and the requested output/audio geometry."""
    if job.get("status") != "completed":
        raise ValueError(f"render did not complete: {job.get('error')}")
    output = _object(job.get("output"), "output")
    stats = _object(job.get("stats"), "stats")
    engine = _object(stats.get("engine"), "engine settings")
    wanted = {
        "width": 512,
        "height": 512,
        "frame_count": 49,
        "fps": 24,
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    }
    for key, value in wanted.items():
        if _integer(output.get(key), key) != value:
            raise ValueError(f"output differs at {key}")
    for key, value in {
        "width": 512,
        "height": 512,
        "frame_count": 49,
        "seed": cell.seed,
        "steps": 11,
        "pipeline": "distilled_two_stage",
        "decoder": "convolutional",
        "codec": "h264",
        "sampler": "euler_ancestral",
        "scheduler": "ltx-2.5" if backend == "mlx_video-metal" else "manual",
    }.items():
        if engine.get(key) != value:
            raise ValueError(f"executed settings differ at {key}")
    if (
        _integer(stats.get("steps"), "sampling steps") != 11
        or _number(stats.get("seconds_per_step"), "sampling time") <= 0
    ):
        raise ValueError(
            "missing actual 8+3 sampling; a cache hit is not a measurement"
        )
    size = _integer(output.get("size_bytes"), "output size")
    if not 1 <= size <= _MAX_VIDEO_BYTES:
        raise ValueError("output exceeds the baseline download bound")
    if not re.fullmatch(r"[a-f0-9]{64}", _text(output.get("sha256"), "digest")):
        raise ValueError("invalid output digest")
    return output


def probe_video(path: Path, *, interpreter: str | None = None) -> dict[str, JsonValue]:
    """Decode the complete container and check stream geometry and mux alignment.

    Stream duration agreement is a container check, not semantic audio sync or
    perceptual quality. The original media remains available for those reviews.
    """
    command = (
        [interpreter, "-I", "-B", "-c", _PYAV_PROBE, str(path)]
        if interpreter is not None
        else [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
        timeout=120,
    )
    if len(result.stdout) > 16 * 1024:
        raise ValueError("media probe exceeded its bounded response")
    probe = _object(_JSON.validate_json(result.stdout), "media probe")
    if "mp4" not in _text(
        _object(probe.get("format"), "container format").get("format_name"),
        "format name",
    ).split(","):
        raise ValueError("decoded container is not MP4")
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise ValueError("media probe returned no streams")
    videos = [
        _object(item, "stream")
        for item in streams
        if _object(item, "stream").get("codec_type") == "video"
    ]
    audios = [
        _object(item, "stream")
        for item in streams
        if _object(item, "stream").get("codec_type") == "audio"
    ]
    if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
        raise ValueError("expected exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    if (
        video.get("width") != 512
        or video.get("height") != 512
        or video.get("codec_name") != "h264"
        or video.get("nb_read_frames") != "49"
        or Fraction(_text(video.get("avg_frame_rate"), "frame rate")) != 24
    ):
        raise ValueError("decoded video differs from the requested geometry or codec")
    if (
        audio.get("sample_rate") != "48000"
        or audio.get("channels") != 2
        or audio.get("codec_name") != "aac"
    ):
        raise ValueError("decoded audio differs from the required sample rate/channels")
    video_seconds = float(_text(video.get("duration"), "video duration"))
    audio_seconds = float(_text(audio.get("duration"), "audio duration"))
    if (
        not math.isfinite(video_seconds)
        or not math.isfinite(audio_seconds)
        or abs(video_seconds - 49 / 24) > 1 / 24
        or abs(audio_seconds - video_seconds) > 1 / 24
    ):
        raise ValueError("stream durations do not align within one video frame")
    if interpreter is None:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
    return probe


@final
class VideoApiQualification:
    """Run baseline jobs on an explicitly scoped isolated API, retaining evidence."""

    def __init__(
        self,
        profile: VideoProfile,
        client: httpx.Client,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind the reviewed profile to a caller-owned client and polling clock."""
        self.profile = profile
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.active_job: str | None = None

    def _json(self, path: str) -> dict[str, JsonValue]:
        with self.client.stream("GET", path) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_JSON_BYTES:
                    raise ValueError("API JSON exceeds the evidence bound")
        return _object(_JSON.validate_json(body), "API response")

    def _cell(self, cell: VideoCell, fixtures: Path, directory: Path) -> None:
        directory.mkdir()
        body = cell.body(self.profile.model)
        _save(directory / "request.json", body)
        started = self.clock()
        if cell.parts:
            files = [
                (role, (name, (fixtures / name).read_bytes(), "image/png"))
                for role, name in cell.parts
            ]
            fields = {
                key: value if isinstance(value, str) else json.dumps(value)
                for key, value in body.items()
            }
            response = self.client.post("/v1/videos", data=fields, files=files)
        else:
            response = self.client.post("/v1/videos", json=body)
        response.raise_for_status()
        submitted = _object(_JSON.validate_json(response.content), "submitted job")
        job_id = _text(submitted.get("id"), "job id")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", job_id):
            raise ValueError("API returned an invalid job id")
        self.active_job = job_id
        _save(directory / "submitted.json", submitted)
        proof: dict[str, JsonValue] | None = None
        inputs = {
            name: (
                hashlib.sha256((fixtures / name).read_bytes()).hexdigest(),
                (fixtures / name).stat().st_size,
            )
            for _, name in cell.parts
        }
        while True:
            state = self._json("/state")
            observed = video_execution_proof(state, self.profile, job_id, cell, inputs)
            if observed is not None:
                if proof is not None and _canonical(proof) != _canonical(observed):
                    raise ValueError(
                        "actual task or placement changed during rendering"
                    )
                proof = observed
                _save(directory / "execution.json", proof)
            job = self._json(f"/v1/videos/{quote(job_id, safe='')}")
            _save(directory / "job.json", job)
            if job.get("id") != job_id or job.get("model") != self.profile.model:
                raise ValueError("API answered with another job or model")
            with (directory / "observations.jsonl").open("a") as stream:
                stream.write(
                    _canonical(
                        {
                            "elapsed_seconds": self.clock() - started,
                            "status": job.get("status"),
                            "stage": job.get("stage"),
                            "progress": job.get("progress"),
                            "worker_memory": _object(
                                state.get("nodeMemory", {}), "node memory"
                            ).get(self.profile.worker_node_id),
                            "worker_system": _object(
                                state.get("nodeSystem", {}), "node system"
                            ).get(self.profile.worker_node_id),
                        }
                    )
                    + "\n"
                )
            if job.get("status") in ("completed", "failed", "cancelled"):
                self.active_job = None
                break
            if self.clock() - started >= self.profile.timeout_seconds:
                raise TimeoutError("render exceeded its deadline")
            self.sleep(1.0)
        output = verify_video_result(job, cell, backend=self.profile.backend)
        if proof is None:
            raise ValueError("no actual task/shard evidence was observed")
        path = directory / "output.mp4"
        digest = hashlib.sha256()
        size = 0
        with (
            self.client.stream(
                "GET", f"/v1/videos/{quote(job_id, safe='')}/content"
            ) as download,
            path.open("xb") as target,
        ):
            download.raise_for_status()
            for chunk in download.iter_bytes():
                size += len(chunk)
                if size > _integer(output.get("size_bytes"), "output size"):
                    raise ValueError("download exceeds its manifest")
                digest.update(chunk)
                target.write(chunk)
        if size != output.get("size_bytes") or digest.hexdigest() != output.get(
            "sha256"
        ):
            raise ValueError("download does not match its verified manifest")
        _save(
            directory / "media-probe.json",
            probe_video(path, interpreter=self.profile.media_python),
        )
        _save(
            directory / "checks.json",
            {
                "api_checks": "passed",
                "wall_seconds": self.clock() - started,
                "quality": "unreviewed",
                "semantic_audio_sync": "unreviewed",
                "external_peak_memory": "unmeasured",
                "telemetry": "sampled node resources; not an allocator or process high-water counter",
            },
        )

    def run(self, directory: Path) -> None:
        """Retain five baseline jobs and cancel only this run's active job on exit.

        A new directory is required, preventing accidental evidence replacement.
        Failed and interrupted runs retain their partial evidence and fail closed.
        """
        if self.profile.media_python is None:
            if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
                raise ValueError(
                    "ffmpeg/ffprobe or a verified PyAV interpreter are required before submitting work"
                )
        else:
            subprocess.run(
                [self.profile.media_python, "-I", "-B", "-c", "import av"],
                check=True,
                capture_output=True,
                timeout=10,
            )
        directory.mkdir(parents=True, mode=0o700)
        receipt: dict[str, JsonValue] = {
            "status": "interrupted",
            "profile": _JSON.validate_json(self.profile.model_dump_json()),
            "harness_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "scope": "LTX baseline API/media checks; not complete qualification",
            "remote_media": self.profile.api_node_id != self.profile.worker_node_id,
            "completed_cells": [],
        }
        completed: list[JsonValue] = []
        try:
            state = self._json("/state")
            _save(directory / "preflight-state.json", state)
            verify_video_scope(state, self.profile)
            diagnostics = self._json("/v1/diagnostics/node")
            _save(directory / "api-diagnostics.json", diagnostics)
            if (
                _object(diagnostics.get("runtime"), "API runtime").get("nodeId")
                != self.profile.api_node_id
            ):
                raise ValueError("URL does not serve the reviewed owning API node")
            instances = _object(state.get("instances"), "instances")
            # Refuse before submitting if no exact reviewed placement is present.
            matches = []
            for instance in instances.values():
                tagged = _object(instance, "instance tag")
                if len(tagged) == 1:
                    body = _object(next(iter(tagged.values())), "instance")
                    shards = body.get("shardAssignments")
                    if (
                        isinstance(shards, dict)
                        and shards.get("modelId") == self.profile.model
                    ):
                        matches.append(_instance_proof(instance, self.profile))
            if len(matches) != 1:
                raise ValueError(
                    "expected one exact candidate placement before submission"
                )
            fixtures = directory / "fixtures"
            receipt["fixtures"] = {
                name: digest for name, digest in make_video_fixtures(fixtures).items()
            }
            for cell in BASELINE:
                self._cell(cell, fixtures, directory / cell.name)
                completed.append(cell.name)
                receipt["completed_cells"] = list(completed)
                _save(directory / "receipt.json", receipt)
            receipt["status"] = "api_checks_passed"
        except (
            ValueError,
            RuntimeError,
            httpx.HTTPError,
            subprocess.SubprocessError,
            OSError,
        ) as error:
            receipt["status"] = "failed"
            receipt["error"] = str(error)
            raise
        finally:
            if self.active_job is not None:
                receipt["cancel_requested_for"] = self.active_job
                try:
                    response = self.client.post(
                        f"/v1/videos/{quote(self.active_job, safe='')}/cancel"
                    )
                    response.raise_for_status()
                except httpx.HTTPError as error:
                    receipt["cancel_error"] = str(error)
            _save(directory / "receipt.json", receipt)
