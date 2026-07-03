import json

import httpx
import pytest

from skulk_test_harness.client import SkulkClient


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
