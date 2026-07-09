import json

import httpx
import pytest

from skulk_test_harness.client import SkulkApiError, SkulkClient


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
    monkeypatch.setattr(
        "skulk_test_harness.client.time.monotonic", lambda: next(times)
    )

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
