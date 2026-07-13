import base64
import json

import httpx
import pytest

from skulk_test_harness import client as client_module
from skulk_test_harness.client import (
    ClusterApiOwner,
    DataPlaneDiagnosticsSnapshot,
    ProviderCapabilityDiagnosticsSnapshot,
    SkulkApiError,
    SkulkClient,
)


class _FakeWebSocket:
    """Synchronous websocket stand-in with a fixed server event sequence."""

    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = [json.dumps(event) for event in events]
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    def __enter__(self) -> "_FakeWebSocket":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def recv(self, *, timeout: float | None = None, decode: bool | None = None) -> str:
        del timeout, decode
        return self.events.pop(0)

    def send(self, message: str) -> None:
        self.sent.append(message)

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


def test_cluster_api_urls_include_local_and_reachable_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://local.test/")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *_args, **_kwargs: {
            "localNodeId": "local",
            "nodes": [
                {
                    "nodeId": "local",
                    "url": None,
                    "ok": True,
                    "diagnostics": {"identity": {"friendlyName": "local.test"}},
                },
                {"nodeId": "peer-a", "url": "http://peer-a.test/", "ok": True},
                {"nodeId": "peer-b", "url": "http://peer-b.test", "ok": False},
            ],
        },
    )
    monkeypatch.setattr(client, "_api_url_reachable", lambda _url: True)
    try:
        assert client.get_cluster_api_urls() == [
            "http://local.test",
            "http://peer-a.test",
        ]
    finally:
        client.close()


def test_cluster_api_urls_prefers_controller_reachable_node_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://controller.test:52415")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *_args, **_kwargs: {
            "localNodeId": "controller",
            "nodes": [
                {
                    "nodeId": "peer-a",
                    "url": "http://node-local-route.test:52415",
                    "ok": True,
                    "diagnostics": {
                        "identity": {"friendlyName": "peer-a"},
                        "tailscale": {
                            "dnsName": "peer-a.overlay.test",
                            "hostname": "peer-a",
                        },
                    },
                }
            ],
        },
    )
    attempted: list[str] = []

    def reachable(url: str) -> bool:
        attempted.append(url)
        return url == "http://peer-a.overlay.test:52415"

    monkeypatch.setattr(client, "_api_url_reachable", reachable)
    try:
        assert client.get_cluster_api_urls() == [
            "http://controller.test:52415",
            "http://peer-a.overlay.test:52415",
        ]
    finally:
        client.close()

    assert attempted == ["http://peer-a.overlay.test:52415"]


def test_cluster_api_owners_preserve_node_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://controller.test:52415")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *_args, **_kwargs: {
            "localNodeId": "node-local",
            "nodes": [
                {"nodeId": "node-local", "ok": True},
                {
                    "nodeId": "node-remote",
                    "url": "http://remote.test:52415",
                    "ok": True,
                },
            ],
        },
    )
    monkeypatch.setattr(client, "_api_url_reachable", lambda _url: True)
    try:
        assert client.get_cluster_api_owners() == [
            ClusterApiOwner("node-local", "http://controller.test:52415"),
            ClusterApiOwner("node-remote", "http://remote.test:52415"),
        ]
    finally:
        client.close()


def test_data_plane_diagnostics_snapshot_parses_camel_case_payload() -> None:
    payload: dict[str, object] = {
        "runtime": {"nodeId": "node-a"},
        "dataPlane": {
            "activeStreams": 0,
            "startedFrames": 4,
            "completedFrames": 4,
            "failedFrames": 0,
            "cancelledFrames": 0,
            "duplicateFrames": 0,
            "outOfOrderFrames": 0,
            "skippedSequences": 0,
            "lateFrames": 0,
            "missingStartedStreams": 0,
            "missingTerminalStreams": 0,
            "idleTimeouts": 0,
            "transportFailures": 0,
            "egress": {
                "activeStreamQueues": 0,
                "queueDepth": 0,
                "localShortCircuits": 8,
                "remoteFramesEnqueued": 8,
                "remoteFramesPublished": 8,
                "remoteFramesDropped": 0,
                "remotePublishFailures": 0,
            },
        },
    }

    snapshot = DataPlaneDiagnosticsSnapshot.from_payload(payload)

    assert snapshot.node_id == "node-a"
    assert snapshot.started_frames == 4
    assert snapshot.local_short_circuits == 8
    assert snapshot.remote_frames_published == 8


def test_get_store_registry_maps_storeless_503_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store-less node's 503 means "no store here", not a fatal error."""

    client = SkulkClient("http://skulk.test")

    def request(method: str, path: str, **_kwargs: object) -> httpx.Response:
        assert (method, path) == ("GET", "/store/registry")
        return httpx.Response(503, json={"error": {"message": "Store not configured"}})

    monkeypatch.setattr(client._client, "request", request)
    try:
        assert client.get_store_registry() is None
    finally:
        client.close()


def test_get_store_registry_still_raises_other_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the explicit storeless body maps to None; other 503s stay fatal."""

    client = SkulkClient("http://skulk.test")

    def request(method: str, path: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(503, text="registry rebuilding, try again")

    monkeypatch.setattr(client._client, "request", request)
    try:
        with pytest.raises(SkulkApiError):
            client.get_store_registry()
    finally:
        client.close()


def test_request_json_retries_read_timeout_for_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    calls = 0
    monkeypatch.setattr("skulk_test_harness.client.time.sleep", lambda _seconds: None)

    def request(method: str, path: str, **_kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("stalled read")
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(client._client, "request", request)
    try:
        assert client._request_json("GET", "/state") == {"ok": True}
    finally:
        client.close()
    assert calls == 2


def test_request_json_does_not_retry_post_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    calls = 0

    def request(method: str, path: str, **_kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(f"stalled {method} {path}")

    monkeypatch.setattr(client._client, "request", request)
    try:
        with pytest.raises(httpx.ReadTimeout):
            client._request_json(
                "POST",
                "/place_instance",
                json_body={"model_id": "m/Foo"},
            )
    finally:
        client.close()
    assert calls == 1


def test_audio_speech_posts_openai_payload_and_returns_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 32)

    def post(path: str, **kwargs: object) -> httpx.Response:
        seen["path"] = path
        seen["json"] = kwargs["json"]
        return httpx.Response(
            200,
            content=wav,
            headers={"content-type": "audio/wav; charset=binary"},
        )

    monkeypatch.setattr(client._client, "post", post)
    try:
        execution = client.audio_speech(
            model_id="org/TTS",
            input_text="hello",
            response_format="wav",
            voice="af_heart",
            speed=1.1,
        )
    finally:
        client.close()

    assert seen == {
        "path": "/v1/audio/speech",
        "json": {
            "model": "org/TTS",
            "input": "hello",
            "response_format": "wav",
            "voice": "af_heart",
            "speed": 1.1,
        },
    }
    assert execution.audio == wav
    assert execution.media_type == "audio/wav"
    assert execution.response_format == "wav"


def test_audio_speech_streams_bytes_and_records_chunk_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}
    times = iter([100.0, 100.2, 100.6, 101.0])
    monkeypatch.setattr("skulk_test_harness.client.time.monotonic", lambda: next(times))

    class _Stream:
        status_code = 200
        headers = {"content-type": "audio/mpeg"}

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def iter_bytes(self):
            yield b"abc"
            yield b"def"

    def stream(method: str, path: str, **kwargs: object) -> _Stream:
        seen["method"] = method
        seen["path"] = path
        seen["json"] = kwargs["json"]
        return _Stream()

    monkeypatch.setattr(client._client, "stream", stream)
    try:
        execution = client.audio_speech(
            model_id="org/TTS",
            input_text="hello",
            response_format="mp3",
            stream=True,
            streaming_interval=0.25,
        )
    finally:
        client.close()

    assert seen == {
        "method": "POST",
        "path": "/v1/audio/speech",
        "json": {
            "model": "org/TTS",
            "input": "hello",
            "response_format": "mp3",
            "stream": True,
            "streaming_interval": 0.25,
        },
    }
    assert execution.audio == b"abcdef"
    assert execution.media_type == "audio/mpeg"
    assert execution.elapsed_s == pytest.approx(1.0)
    assert execution.first_byte_s == pytest.approx(0.2)
    assert execution.chunks == 2
    assert execution.chunk_sizes == [3, 3]
    assert execution.chunk_arrival_s == pytest.approx([0.2, 0.6])
    assert execution.streaming is True


def test_audio_speech_posts_reference_audio_as_multipart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}
    reference = b"RIFF-reference"

    def post(path: str, **kwargs: object) -> httpx.Response:
        seen.update(path=path, **kwargs)
        return httpx.Response(
            200,
            content=b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 2048),
            headers={"content-type": "audio/wav"},
        )

    monkeypatch.setattr(client._client, "post", post)
    try:
        client.audio_speech(
            model_id="org/QwenBase",
            input_text="conditioned output",
            reference_audio=reference,
            reference_text="reference words",
        )
    finally:
        client.close()

    assert "json" not in seen
    assert seen["data"] == {
        "model": "org/QwenBase",
        "input": "conditioned output",
        "response_format": "wav",
        "reference_text": "reference words",
    }
    assert seen["files"] == {
        "reference_audio": ("reference.wav", reference, "audio/wav")
    }


def test_audio_voices_returns_strict_voice_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *_args, **_kwargs: {
            "object": "list",
            "data": [{"id": "ryan"}, {"id": "aiden"}],
        },
    )
    try:
        assert client.audio_voices("org/TTS") == ["ryan", "aiden"]
    finally:
        client.close()


def test_audio_speech_can_intentionally_delay_stream_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    delays: list[float] = []
    times = iter([100.0, 100.2, 100.7, 101.2])
    monkeypatch.setattr("skulk_test_harness.client.time.monotonic", lambda: next(times))

    class _Stream:
        status_code = 200
        headers = {"content-type": "audio/mpeg"}

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def iter_bytes(self):
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(client._client, "stream", lambda *_args, **_kwargs: _Stream())
    monkeypatch.setattr(
        "skulk_test_harness.client.time.sleep", lambda seconds: delays.append(seconds)
    )
    try:
        execution = client.audio_speech(
            model_id="org/TTS",
            input_text="hello",
            response_format="mp3",
            stream=True,
            read_delay_s=0.25,
        )
    finally:
        client.close()

    assert execution.audio == b"abcdef"
    assert delays == [0.25, 0.25]
    assert execution.chunk_arrival_s == pytest.approx([0.2, 0.45])
    assert execution.elapsed_s == pytest.approx(1.2)


def test_audio_speech_rejects_streaming_interval_without_stream() -> None:
    client = SkulkClient("http://skulk.test")
    try:
        with pytest.raises(ValueError, match="streaming_interval requires stream=True"):
            client.audio_speech(
                model_id="org/TTS",
                input_text="hello",
                response_format="mp3",
                streaming_interval=0.25,
            )
    finally:
        client.close()


def test_audio_speech_wraps_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")

    def post(_path: str, **_kwargs: object) -> httpx.Response:
        raise httpx.ReadTimeout("speech stalled")

    monkeypatch.setattr(client._client, "post", post)
    try:
        with pytest.raises(SkulkApiError) as exc_info:
            client.audio_speech(
                model_id="org/TTS",
                input_text="hello",
                response_format="wav",
            )
    finally:
        client.close()

    assert exc_info.value.status_code == 0
    assert exc_info.value.path == "/v1/audio/speech"
    assert "ReadTimeout: speech stalled" in exc_info.value.body


def test_audio_transcription_posts_multipart_and_extracts_json_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}

    def post(path: str, **kwargs: object) -> httpx.Response:
        seen["path"] = path
        seen["data"] = kwargs["data"]
        seen["files"] = kwargs["files"]
        return httpx.Response(
            200,
            json={"text": "hello world"},
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(client._client, "post", post)
    try:
        execution = client.audio_transcription(
            model_id="org/STT",
            audio=b"RIFF....WAVE",
            filename="sample.wav",
            media_type="audio/wav",
            response_format="json",
            language="en",
            prompt="hint",
        )
    finally:
        client.close()

    assert seen["path"] == "/v1/audio/transcriptions"
    assert seen["data"] == {
        "model": "org/STT",
        "response_format": "json",
        "language": "en",
        "prompt": "hint",
    }
    assert seen["files"] == {"file": ("sample.wav", b"RIFF....WAVE", "audio/wav")}
    assert execution.text == "hello world"
    assert execution.media_type == "application/json"


def test_audio_transcription_wraps_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")

    def post(_path: str, **_kwargs: object) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr(client._client, "post", post)
    try:
        with pytest.raises(SkulkApiError) as exc_info:
            client.audio_transcription(
                model_id="org/STT",
                audio=b"RIFF....WAVE",
                filename="sample.wav",
                media_type="audio/wav",
            )
    finally:
        client.close()

    assert exc_info.value.status_code == 0
    assert exc_info.value.path == "/v1/audio/transcriptions"
    assert "RemoteProtocolError: server disconnected" in exc_info.value.body


def test_audio_translation_posts_multipart_to_translation_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}

    def post(path: str, **kwargs: object) -> httpx.Response:
        seen["path"] = path
        seen["data"] = kwargs["data"]
        seen["files"] = kwargs["files"]
        return httpx.Response(
            200,
            json={"text": "hello project"},
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(client._client, "post", post)
    try:
        execution = client.audio_translation(
            model_id="org/Canary",
            audio=b"RIFF....WAVE",
            filename="french.wav",
            media_type="audio/wav",
            response_format="json",
            language="fr",
        )
    finally:
        client.close()

    assert seen["path"] == "/v1/audio/translations"
    assert seen["data"] == {
        "model": "org/Canary",
        "response_format": "json",
        "language": "fr",
    }
    assert seen["files"] == {"file": ("french.wav", b"RIFF....WAVE", "audio/wav")}
    assert execution.text == "hello project"


def test_audio_transcription_extracts_ndjson_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")

    def post(_path: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"text":"hello"}\n{"text":"world"}\n',
            headers={"content-type": "application/x-ndjson"},
        )

    monkeypatch.setattr(client._client, "post", post)
    try:
        execution = client.audio_transcription(
            model_id="org/STT",
            audio=b"audio",
            filename="sample.wav",
            media_type="audio/wav",
            response_format="ndjson",
        )
    finally:
        client.close()

    assert execution.text == "hello world"


def test_streaming_audio_transcription_records_typed_sse_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    seen: dict[str, object] = {}
    times = iter([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])
    monkeypatch.setattr(
        "skulk_test_harness.client.time.monotonic", lambda: next(times)
    )

    class _Stream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            yield ': keep-alive'
            yield 'data: {"type":"transcription.delta","delta":"hello "}'
            yield 'data: {"type":"transcription.delta","delta":"world"}'
            yield 'data: {"type":"transcription.completed","text":"hello world"}'
            yield 'data: {"type":"transcription.usage","input_bytes":5}'

    def stream(method: str, path: str, **kwargs: object) -> _Stream:
        seen["method"] = method
        seen["path"] = path
        seen["data"] = kwargs["data"]
        seen["files"] = kwargs["files"]
        return _Stream()

    monkeypatch.setattr(client._client, "stream", stream)
    try:
        execution = client.streaming_audio_transcription(
            model_id="org/STT",
            audio=b"audio",
            filename="sample.wav",
            media_type="audio/wav",
            language="en",
            prompt="hint",
        )
    finally:
        client.close()

    assert seen == {
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "data": {
            "model": "org/STT",
            "stream": "true",
            "language": "en",
            "prompt": "hint",
        },
        "files": {"file": ("sample.wav", b"audio", "audio/wav")},
    }
    assert execution.text == "hello world"
    assert execution.first_transcript_s == pytest.approx(0.1)
    assert execution.transcript_deltas == 2
    assert execution.event_types[-2:] == [
        "transcription.completed",
        "transcription.usage",
    ]
    assert execution.event_arrival_s == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert execution.elapsed_s == pytest.approx(0.5)


def test_realtime_transcription_maps_pcm_to_websocket_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeWebSocket(
        [
            {"type": "session.created", "session": {"type": "transcription"}},
            {"type": "session.updated"},
            {"type": "input_audio_buffer.committed"},
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "hello ",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hello world",
            },
        ]
    )
    connected: dict[str, object] = {}

    def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        connected["url"] = url
        connected["kwargs"] = kwargs
        return socket

    monkeypatch.setattr(client_module.websocket_client, "connect", connect)
    client = SkulkClient("https://skulk.test:52415")
    pcm16 = bytes(range(256)) * 3
    try:
        execution = client.realtime_transcription(
            model_id="org/realtime stt",
            pcm16=pcm16,
            sample_rate=8_000,
            frame_duration_ms=20,
            pace_audio=False,
        )
    finally:
        client.close()

    assert connected["url"] == (
        "wss://skulk.test:52415/v1/realtime?model=org%2Frealtime%20stt"
    )
    assert isinstance(connected["kwargs"], dict)
    assert connected["kwargs"]["proxy"] is None
    assert execution.text == "hello world"
    assert execution.input_bytes == len(pcm16)
    assert execution.input_frames == 3
    assert execution.transcript_deltas == 1
    assert execution.canceled is False
    payloads = [json.loads(message) for message in socket.sent]
    assert payloads[0]["type"] == "session.update"
    append_payloads = [
        payload
        for payload in payloads
        if payload["type"] == "input_audio_buffer.append"
    ]
    assert (
        b"".join(base64.b64decode(payload["audio"]) for payload in append_payloads)
        == pcm16
    )
    assert payloads[-1] == {"type": "input_audio_buffer.commit"}


def test_fabric_speech_chain_collects_transcript_text_and_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit Fabric endpoint must complete the full speech composition."""

    socket = _FakeWebSocket(
        [
            {"type": "session.created", "session": {"type": "transcription"}},
            {"type": "session.updated"},
            {"type": "input_audio_buffer.committed"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hello world",
            },
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "hello "},
            {"type": "response.output_text.done", "text": "hello back"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"audio-one").decode("ascii"),
            },
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"audio-two").decode("ascii"),
            },
            {"type": "response.audio.done"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    connected: dict[str, object] = {}

    def connect(url: str, **_kwargs: object) -> _FakeWebSocket:
        connected["url"] = url
        return socket

    monkeypatch.setattr(client_module.websocket_client, "connect", connect)
    client = SkulkClient("https://skulk.test:52415")
    try:
        execution = client.realtime_transcription(
            model_id="org/realtime stt",
            pcm16=b"\x01\x00" * 160,
            sample_rate=8_000,
            frame_duration_ms=20,
            pace_audio=False,
            fabric_chain=True,
            response_model_id="org/chat",
            response_tts_model_id="org/tts",
            response_voice="coral",
            response_max_output_tokens=96,
            response_enable_thinking=False,
        )
    finally:
        client.close()

    assert connected["url"] == (
        "wss://skulk.test:52415/v1/fabric/chains/speech"
        "?stt_model=org%2Frealtime%20stt"
    )
    session = json.loads(socket.sent[0])["session"]
    assert session["response"] == {
        "model": "org/chat",
        "tts_model": "org/tts",
        "voice": "coral",
        "max_output_tokens": 96,
        "enable_thinking": False,
    }
    assert execution.text == "hello world"
    assert execution.assistant_text == "hello back"
    assert execution.response_audio == b"audio-oneaudio-two"
    assert execution.response_audio_chunks == 2
    assert execution.response_status == "completed"


def test_realtime_conversation_uses_vad_multi_turn_and_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One socket should retain VAD turns and interrupt an active response."""

    socket = _FakeWebSocket(
        [
            {"type": "session.created", "session": {"type": "transcription"}},
            {"type": "session.updated"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "input_audio_buffer.committed"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "first turn",
            },
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "long response"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"partial-audio").decode("ascii"),
            },
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "input_audio_buffer.committed"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "second turn",
            },
            {"type": "response.created"},
            {"type": "response.output_text.done", "text": "final answer"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"final-audio").decode("ascii"),
            },
            {"type": "response.audio.done"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    monkeypatch.setattr(
        client_module.websocket_client,
        "connect",
        lambda *_args, **_kwargs: socket,
    )
    client = SkulkClient("http://skulk.test")
    try:
        execution = client.realtime_transcription(
            model_id="org/STT",
            pcm16=b"\x01\x00" * 160,
            sample_rate=8_000,
            frame_duration_ms=20,
            pace_audio=False,
            response_model_id="org/chat",
            response_tts_model_id="org/tts",
            server_vad=True,
            turn_count=2,
            barge_in=True,
        )
    finally:
        client.close()

    session = json.loads(socket.sent[0])["session"]
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert all(
        json.loads(message)["type"] != "input_audio_buffer.commit"
        for message in socket.sent
    )
    assert execution.transcripts == ["first turn", "second turn"]
    assert execution.response_statuses == ["cancelled", "completed"]
    assert execution.assistant_turns[-1] == "final answer"
    assert execution.response_audio_turns[-1] == b"final-audio"
    assert execution.speech_started_events == 2
    assert execution.speech_stopped_events == 2
    assert execution.provider_sessions == 2
    assert execution.barge_in_sent is True
    audio_events = [
        event for event in execution.events if event["type"] == "response.audio.delta"
    ]
    assert audio_events[0]["delta_bytes"] == len(b"partial-audio")
    assert "delta" not in audio_events[0]


def test_realtime_transcription_disconnect_probe_closes_without_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
        ]
    )
    monkeypatch.setattr(
        client_module.websocket_client,
        "connect",
        lambda *_args, **_kwargs: socket,
    )
    client = SkulkClient("http://skulk.test")
    try:
        execution = client.realtime_transcription(
            model_id="org/STT",
            pcm16=b"\x00\x00" * 640,
            sample_rate=8_000,
            frame_duration_ms=20,
            pace_audio=False,
            cancel_after_frames=2,
        )
    finally:
        client.close()

    assert execution.canceled is True
    assert execution.input_frames == 2
    assert socket.closed == (1000, "harness cancellation probe")
    assert all(
        json.loads(message)["type"] != "input_audio_buffer.commit"
        for message in socket.sent
    )


def test_realtime_transcription_surfaces_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeWebSocket(
        [{"type": "error", "error": {"message": "realtime unavailable"}}]
    )
    monkeypatch.setattr(
        client_module.websocket_client,
        "connect",
        lambda *_args, **_kwargs: socket,
    )
    client = SkulkClient("http://skulk.test")
    try:
        with pytest.raises(SkulkApiError, match="realtime unavailable") as exc_info:
            client.realtime_transcription(
                model_id="org/STT",
                pcm16=b"\x00\x00" * 160,
                sample_rate=8_000,
                pace_audio=False,
            )
    finally:
        client.close()

    assert exc_info.value.method == "WS"
    assert exc_info.value.path == "/v1/realtime"


def test_realtime_transcription_wraps_handshake_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_handshake(*_args: object, **_kwargs: object) -> None:
        raise client_module.WebSocketException("handshake rejected")

    monkeypatch.setattr(client_module.websocket_client, "connect", reject_handshake)
    client = SkulkClient("http://skulk.test")
    try:
        with pytest.raises(SkulkApiError, match="handshake rejected") as exc_info:
            client.realtime_transcription(
                model_id="org/STT",
                pcm16=b"\x00\x00" * 160,
                sample_rate=8_000,
                pace_audio=False,
            )
    finally:
        client.close()

    assert exc_info.value.method == "WS"
    assert exc_info.value.path == "/v1/realtime"
    assert exc_info.value.status_code == 0


@pytest.mark.parametrize("max_output_tokens", [0, 4097])
def test_realtime_transcription_rejects_invalid_response_token_limit(
    max_output_tokens: int,
) -> None:
    client = SkulkClient("http://skulk.test")
    try:
        with pytest.raises(
            ValueError,
            match="response_max_output_tokens must be between 1 and 4096",
        ):
            client.realtime_transcription(
                model_id="org/STT",
                pcm16=b"\x00\x00" * 160,
                sample_rate=8_000,
                pace_audio=False,
                response_model_id="org/chat",
                response_max_output_tokens=max_output_tokens,
            )
    finally:
        client.close()


def test_provider_capability_diagnostics_parses_realtime_counters() -> None:
    snapshot = ProviderCapabilityDiagnosticsSnapshot.from_payload(
        {
            "runtime": {"nodeId": "node-a"},
            "provider": {
                "streamSlotsInUse": 1,
                "capabilities": {
                    "stt.realtime@1.0.0": {
                        "activeStreams": 1,
                        "admittedStreams": 4,
                        "inputQueueDepth": 2,
                        "inputFrames": 20,
                        "inputMediaBytes": 6400,
                        "outputFrames": 8,
                        "outputMediaBytes": 0,
                        "completedStreams": 3,
                        "failedStreams": 0,
                        "cancelledStreams": 1,
                        "missingTerminalStreams": 0,
                        "cancellationRequests": 1,
                    }
                },
            },
        },
        "stt.realtime@1.0.0",
    )

    assert snapshot.node_id == "node-a"
    assert snapshot.active_streams == 1
    assert snapshot.input_media_bytes == 6400
    assert snapshot.completed_streams == 3
    assert snapshot.cancellation_requests == 1


def test_stream_chat_counts_reasoning_for_generated_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    reasoning = "reasoning token stream " * 12

    class _Stream:
        status_code = 200

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            event = {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": reasoning,
                        }
                    }
                ]
            }
            yield f"data: {json.dumps(event)}"
            yield "data: [DONE]"

    def stream(*_args: object, **_kwargs: object) -> _Stream:
        return _Stream()

    monkeypatch.setattr(client._client, "stream", stream)
    try:
        execution = client.stream_chat(
            model_id="m/Reasoning",
            messages=[{"role": "user", "content": "think"}],
            max_tokens=64,
            temperature=0,
            top_p=None,
        )
    finally:
        client.close()

    assert execution.text == ""
    assert execution.reasoning_text == reasoning
    assert execution.metrics.output_chars == 0
    assert execution.metrics.generated_chars == len(reasoning)
    assert execution.metrics.approx_output_tokens == round(len(reasoning) / 4)
    assert execution.metrics.wall_tps is not None
    assert execution.metrics.wall_tps > 0


def _state_response(_method: str, _path: str, **_kwargs: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "nodeIdentities": {
                "12D3KooAAA": {"friendlyName": "kite4"},
                "12D3KooBBB": {"friendlyName": "kite5"},
            }
        },
    )


def test_resolve_node_ids_maps_friendly_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    monkeypatch.setattr(client._client, "request", _state_response)
    try:
        # friendly name -> node id; an already-resolved node id passes through.
        assert client.resolve_node_ids(["kite4"]) == ["12D3KooAAA"]
        assert client.resolve_node_ids(["12D3KooBBB"]) == ["12D3KooBBB"]
        assert client.resolve_node_ids([]) == []
    finally:
        client.close()


def test_resolve_node_ids_raises_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SkulkClient("http://skulk.test")
    monkeypatch.setattr(client._client, "request", _state_response)
    try:
        # An unknown node name must fail loudly, not silently drop the exclusion
        # (which would place on the node the cell meant to avoid).
        with pytest.raises(ValueError, match="kite9"):
            client.resolve_node_ids(["kite9"])
    finally:
        client.close()


def test_streaming_chat_parses_generation_stats_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine stats ride the final chunk as an SSE comment; streamed tests
    must surface them as skulk_* metrics (previously null on every engine)."""

    lines = [
        ": command_id abc-123",
        'data: {"choices": [{"delta": {"content": "hello"}}]}',
        ': generation_stats {"prompt_tps": 187.4, "generation_tps": 182.7,'
        ' "prompt_tokens": 17, "generation_tokens": 120,'
        ' "peak_memory_usage": {"inBytes": 1}}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ]

    class _FakeStreamResponse:
        status_code = 200

        def iter_lines(self):
            yield from lines

        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    client = SkulkClient("http://skulk.test")
    monkeypatch.setattr(
        client._client, "stream", lambda *a, **k: _FakeStreamResponse()
    )
    try:
        result = client.stream_chat(
            model_id="m/x",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            temperature=None,
            top_p=None,
        )
    finally:
        client.close()

    assert result.text == "hello"
    assert result.metrics.skulk_generation_tps == 182.7
    assert result.metrics.skulk_prompt_tps == 187.4
    assert result.metrics.skulk_prompt_tokens == 17
    assert result.metrics.skulk_generation_tokens == 120
