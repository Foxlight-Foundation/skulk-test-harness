"""Image generation and editing test kinds: client requests and runner checks."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from skulk_test_harness.client import ImageExecution, SkulkClient
from skulk_test_harness.models import (
    HarnessConfig,
    PromptImage,
    PromptTest,
    RunReport,
    RunSpec,
    TestResult,
)
from skulk_test_harness.orchestrator import HarnessRunner


def _png(width: int, height: int) -> bytes:
    """Build a small but valid RGB PNG of the given size."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _client(handler: httpx.MockTransport) -> SkulkClient:
    client = SkulkClient("http://skulk.test")
    client._client.close()  # pyright: ignore[reportPrivateUsage]
    client._client = httpx.Client(  # pyright: ignore[reportPrivateUsage]
        base_url="http://skulk.test", transport=handler
    )
    return client


def test_images_generate_requests_inline_pngs() -> None:
    image = _png(512, 512)
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/generations"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"created": 7, "data": [{"b64_json": base64.b64encode(image).decode()}]},
        )

    client = _client(httpx.MockTransport(handler))
    try:
        execution = client.images_generate(
            model_id="org/flux",
            prompt="a fox",
            size="512x512",
            advanced_params={"seed": 7, "num_inference_steps": 4},
        )
    finally:
        client.close()

    assert seen == [
        {
            "model": "org/flux",
            "prompt": "a fox",
            "n": 1,
            "size": "512x512",
            "response_format": "b64_json",
            "advanced_params": {"seed": 7, "num_inference_steps": 4},
        }
    ]
    assert execution.images == [image]
    assert execution.created == 7


def test_images_edit_sends_the_image_as_a_multipart_form() -> None:
    source = _png(64, 32)
    edited = _png(64, 32)
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/edits"
        assert request.headers["content-type"].startswith("multipart/form-data")
        bodies.append(request.content)
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(edited).decode()}]}
        )

    client = _client(httpx.MockTransport(handler))
    try:
        execution = client.images_edit(
            model_id="org/kontext",
            prompt="make it snow",
            image=source,
            filename="card.png",
            media_type="image/png",
            advanced_params={"seed": 7},
        )
    finally:
        client.close()

    body = bodies[0]
    assert source in body
    assert b'name="model"' in body and b"org/kontext" in body
    assert b'name="prompt"' in body and b"make it snow" in body
    assert b'name="response_format"' in body and b"b64_json" in body
    assert b'"seed": 7' in body
    assert execution.images == [edited]
    assert execution.created is None


def test_image_response_without_inline_bytes_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"url": "http://skulk.test/images/1"}]})

    client = _client(httpx.MockTransport(handler))
    try:
        with pytest.raises(TypeError, match="no b64_json"):
            client.images_generate(model_id="org/flux", prompt="a fox")
    finally:
        client.close()


class _ImageClient:
    """SkulkClient stand-in that returns fixed images and records requests."""

    def __init__(self, images: list[bytes]) -> None:
        self.images = images
        self.generations: list[dict[str, object]] = []
        self.edits: list[dict[str, object]] = []

    def images_generate(self, **kwargs: object) -> ImageExecution:
        self.generations.append(kwargs)
        return ImageExecution(images=self.images, elapsed_s=1.5, created=1)

    def images_edit(self, **kwargs: object) -> ImageExecution:
        self.edits.append(kwargs)
        return ImageExecution(images=self.images, elapsed_s=2.5, created=1)


def _run(
    client: _ImageClient, test: PromptTest, artifact_dir: Path, model_id: str
) -> TestResult:
    runner = HarnessRunner(config=HarnessConfig(), model_sets={}, test_sets={})
    spec = RunSpec(model_set="m", test_set="t", mode="execute")
    return runner._run_test(  # pyright: ignore[reportPrivateUsage]
        client,  # pyright: ignore[reportArgumentType]
        model_id=model_id,
        test=test,
        repetition=1,
        artifact_dir=artifact_dir,
        spec=spec,
        report=RunReport.start("run-1", spec, []),
    )


def test_image_generation_saves_each_png_as_an_artifact(tmp_path: Path) -> None:
    image = _png(512, 512)
    client = _ImageClient([image])
    test = PromptTest(
        name="text-to-image-512",
        kind="image_generation",
        prompt="a fox",
        image_advanced_params={"seed": 7},
    )

    result = _run(client, test, tmp_path, "org/FLUX")

    assert result.passed is True
    assert result.artifact_path == tmp_path / "org-flux--text-to-image-512--rep-1--0.png"
    assert (tmp_path / "org-flux--text-to-image-512--rep-1--0.png").read_bytes() == image
    request: Mapping[str, object] = client.generations[0]
    assert request["size"] == "512x512"
    assert request["advanced_params"] == {"seed": 7}


def test_image_generation_fails_on_the_wrong_size(tmp_path: Path) -> None:
    client = _ImageClient([_png(256, 256)])
    test = PromptTest(name="text-to-image-512", kind="image_generation", prompt="a fox")

    result = _run(client, test, tmp_path, "org/FLUX")

    assert result.passed is False
    messages = [issue.message for issue in result.issues]
    assert "Image was not a PNG of the expected size" in messages


def test_image_generation_fails_on_bytes_that_are_not_png(tmp_path: Path) -> None:
    client = _ImageClient([b"\xff\xd8\xff\xe0 not a png"])
    test = PromptTest(name="text-to-image-512", kind="image_generation", prompt="a fox")

    result = _run(client, test, tmp_path, "org/FLUX")

    assert result.passed is False


def test_image_edit_sends_the_fixture_and_accepts_its_size(tmp_path: Path) -> None:
    fixture = tmp_path / "card.png"
    fixture.write_bytes(_png(96, 48))
    client = _ImageClient([_png(96, 48)])
    test = PromptTest(
        name="image-edit-512",
        kind="image_edit",
        prompt="make it snow",
        images=[PromptImage(input_path=fixture, media_type="image/png")],
    )

    result = _run(client, test, tmp_path / "artifacts", "org/Kontext")

    assert result.passed is True
    request = client.edits[0]
    assert request["image"] == fixture.read_bytes()
    assert request["filename"] == "card.png"
    assert request["media_type"] == "image/png"


def test_image_edit_requires_exactly_one_input_image() -> None:
    with pytest.raises(ValidationError, match="exactly one image with input_path"):
        PromptTest(name="edit", kind="image_edit", prompt="make it snow")
