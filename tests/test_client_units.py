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
