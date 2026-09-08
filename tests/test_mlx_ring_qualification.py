from __future__ import annotations

import json

import pytest

from skulk_test_harness import mlx_ring_qualification as ring


def test_probe_config_derives_bfloat16_element_count() -> None:
    config = ring.RingProbeConfig(size_mb=8.0, dtype_bytes=2)

    assert config.payload_bytes == 8 * 1024 * 1024
    assert config.element_count == 4 * 1024 * 1024


def test_probe_config_derives_float32_element_count() -> None:
    config = ring.RingProbeConfig(size_mb=8.0, dtype_bytes=4)

    assert config.element_count == 2 * 1024 * 1024


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ring.RingProbeConfig(size_mb=0), "size_mb"),
        (ring.RingProbeConfig(iterations=0), "iterations"),
        (ring.RingProbeConfig(delay_rank=-1), "delay_rank"),
        (ring.RingProbeConfig(delay_seconds=-1), "delay_seconds"),
        (ring.RingProbeConfig(delay_iteration=0), "delay_iteration"),
        (ring.RingProbeConfig(stall_timeout_seconds=0), "stall_timeout_seconds"),
        (ring.RingProbeConfig(dtype_bytes=0), "dtype_bytes"),
    ],
)
def test_probe_config_rejects_invalid_values(
    config: ring.RingProbeConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_iteration_observation_serializes_machine_readable_record() -> None:
    observation = ring.IterationObservation(
        rank=1,
        world_size=3,
        iteration=7,
        stream="cpu",
        payload_bytes=64 * 1024 * 1024,
        elapsed_seconds=0.125,
        delayed_seconds=0.0,
        max_abs_error=0.0,
    )

    assert observation.as_dict() == {
        "event": "iteration",
        "rank": 1,
        "world_size": 3,
        "iteration": 7,
        "stream": "cpu",
        "payload_bytes": 64 * 1024 * 1024,
        "elapsed_seconds": 0.125,
        "delayed_seconds": 0.0,
        "max_abs_error": 0.0,
    }


def test_main_builds_expected_probe_config(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[ring.RingProbeConfig] = []

    def fake_run_probe(config: ring.RingProbeConfig) -> int:
        observed.append(config)
        return 0

    monkeypatch.setattr(ring, "run_probe", fake_run_probe)

    rc = ring.main(
        [
            "--stream",
            "cpu",
            "--size-mb",
            "32",
            "--iterations",
            "5000",
            "--delay-rank",
            "2",
            "--delay-seconds",
            "7",
            "--delay-iteration",
            "3",
            "--stall-timeout-seconds",
            "45",
            "--dtype-bytes",
            "4",
            "--pre-barrier",
        ]
    )

    assert rc == 0
    assert observed == [
        ring.RingProbeConfig(
            stream="cpu",
            size_mb=32.0,
            iterations=5000,
            delay_rank=2,
            delay_seconds=7.0,
            delay_iteration=3,
            stall_timeout_seconds=45.0,
            dtype_bytes=4,
            pre_barrier=True,
        )
    ]


def test_main_reports_runtime_error_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_probe(_config: ring.RingProbeConfig) -> int:
        raise RuntimeError("mlx unavailable")

    monkeypatch.setattr(ring, "run_probe", fail_probe)

    assert ring.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"event": "error", "error": "mlx unavailable"}
