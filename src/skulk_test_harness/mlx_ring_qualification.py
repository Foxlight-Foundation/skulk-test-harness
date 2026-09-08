"""Standalone MLX ring collective qualification probe.

This module is intentionally independent from the running Skulk service. It is
meant to answer a narrower substrate question before Skulk-level regression
work starts: does the version of MLX installed on a real Apple-Silicon fleet
survive repeated ring ``all_sum`` collectives under realistic payload sizes and
rank skew?

It targets the two failure modes described in upstream MLX issue #4475:

* default GPU-stream collectives can trip the Metal watchdog when one rank is
  delayed for several seconds;
* CPU-stream collectives avoid that watchdog but were reported to deadlock
  probabilistically on three or more ranks under repeated larger reductions.

The module imports MLX lazily so the normal harness controller and offline unit
test suite do not require macOS or MLX.

Typical use from a checkout available at the same path on each Mac::

    mlx.launch --hosts host-a,host-b,host-c --backend ring -- \
      uv run python -m skulk_test_harness.mlx_ring_qualification \
      --stream gpu --size-mb 64 --iterations 20 \
      --delay-rank 1 --delay-seconds 7 --delay-iteration 1

Then test the CPU-stream workaround without artificial skew::

    mlx.launch --hosts host-a,host-b,host-c --backend ring -- \
      uv run python -m skulk_test_harness.mlx_ring_qualification \
      --stream cpu --size-mb 64 --iterations 5000

A per-iteration process watchdog exits with status 124 when a native collective
stops making progress, preventing a test from hanging forever. A successful run
prints one JSON record per rank/iteration plus a final summary record per rank.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

StreamKind = Literal["gpu", "cpu"]


class _ArrayLike(Protocol):
    """Minimal dynamic surface used from MLX arrays."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def item(self) -> object: ...


@dataclass(frozen=True, slots=True)
class RingProbeConfig:
    """Configuration for one distributed collective probe process."""

    stream: StreamKind = "gpu"
    size_mb: float = 64.0
    iterations: int = 1000
    delay_rank: int | None = None
    delay_seconds: float = 0.0
    delay_iteration: int = 1
    stall_timeout_seconds: float = 30.0
    dtype_bytes: int = 2
    pre_barrier: bool = False

    def validate(self) -> None:
        if self.stream not in {"gpu", "cpu"}:
            raise ValueError(f"unsupported stream: {self.stream}")
        if not math.isfinite(self.size_mb) or self.size_mb <= 0:
            raise ValueError("size_mb must be a finite value greater than zero")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.delay_rank is not None and self.delay_rank < 0:
            raise ValueError("delay_rank must be >= 0")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("delay_seconds must be finite and >= 0")
        if self.delay_iteration < 1:
            raise ValueError("delay_iteration must be at least 1")
        if (
            not math.isfinite(self.stall_timeout_seconds)
            or self.stall_timeout_seconds <= 0
        ):
            raise ValueError("stall_timeout_seconds must be finite and > 0")
        if self.dtype_bytes < 1:
            raise ValueError("dtype_bytes must be at least 1")

    @property
    def payload_bytes(self) -> int:
        return max(1, int(self.size_mb * 1024 * 1024))

    @property
    def element_count(self) -> int:
        return max(1, self.payload_bytes // self.dtype_bytes)


@dataclass(frozen=True, slots=True)
class IterationObservation:
    rank: int
    world_size: int
    iteration: int
    stream: StreamKind
    payload_bytes: int
    elapsed_seconds: float
    delayed_seconds: float
    max_abs_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "event": "iteration",
            "rank": self.rank,
            "world_size": self.world_size,
            "iteration": self.iteration,
            "stream": self.stream,
            "payload_bytes": self.payload_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "delayed_seconds": self.delayed_seconds,
            "max_abs_error": self.max_abs_error,
        }


class _IterationWatchdog:
    """Hard process watchdog for native collectives that can deadlock forever."""

    def __init__(self, *, timeout_seconds: float, rank: int, iteration: int) -> None:
        self._timeout_seconds = timeout_seconds
        self._rank = rank
        self._iteration = iteration
        self._completed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"mlx-ring-watchdog-r{rank}-i{iteration}",
            daemon=True,
        )

    def _run(self) -> None:
        if self._completed.wait(self._timeout_seconds):
            return
        record = {
            "event": "stall_timeout",
            "rank": self._rank,
            "iteration": self._iteration,
            "timeout_seconds": self._timeout_seconds,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        os._exit(124)

    def __enter__(self) -> "_IterationWatchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._completed.set()


def _mlx_module() -> Any:
    """Import ``mlx.core`` lazily and fail with a useful operator message."""

    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exception:
        raise RuntimeError(
            "MLX is required for this probe. Run it on an Apple-Silicon host "
            "with the same MLX environment used by Skulk."
        ) from exception
    return mx


def _max_abs_error(mx: Any, value: Any, expected: float) -> float:
    """Materialize one scalar correctness check after a collective."""

    error = mx.max(mx.abs(value - expected))
    mx.eval(error)
    return float(cast(_ArrayLike, error).item())


def _all_sum(mx: Any, value: Any, group: Any, stream: StreamKind) -> Any:
    """Issue one all-sum on the requested MLX stream."""

    if stream == "cpu":
        with mx.stream(mx.cpu):
            reduced = mx.distributed.all_sum(value, group=group)
            mx.eval(reduced)
        mx.synchronize(mx.cpu)
        return reduced

    reduced = mx.distributed.all_sum(value, group=group)
    mx.eval(reduced)
    return reduced


def _scalar_barrier(mx: Any, group: Any, stream: StreamKind) -> None:
    """Optional tiny all-sum rendezvous used to rule out arrival-skew effects."""

    scalar = mx.array([1.0], dtype=mx.float32)
    reduced = _all_sum(mx, scalar, group, stream)
    error = _max_abs_error(mx, reduced, float(group.size()))
    if error != 0.0:
        raise RuntimeError(f"pre-barrier produced incorrect result: error={error}")


def run_probe(config: RingProbeConfig) -> int:
    """Run the distributed qualification loop in the current MLX rank process."""

    config.validate()
    mx = _mlx_module()
    group = mx.distributed.init(backend="ring", strict=True)
    rank = int(group.rank())
    world_size = int(group.size())

    if world_size < 2:
        raise RuntimeError("ring qualification requires at least two ranks")
    if config.delay_rank is not None and config.delay_rank >= world_size:
        raise ValueError(
            f"delay_rank {config.delay_rank} is outside world size {world_size}"
        )

    dtype = mx.bfloat16 if config.dtype_bytes == 2 else mx.float32
    value = mx.ones((config.element_count,), dtype=dtype)
    mx.eval(value)

    print(
        json.dumps(
            {
                "event": "start",
                "rank": rank,
                "world_size": world_size,
                "stream": config.stream,
                "payload_bytes": config.payload_bytes,
                "iterations": config.iterations,
                "delay_rank": config.delay_rank,
                "delay_seconds": config.delay_seconds,
                "delay_iteration": config.delay_iteration,
                "stall_timeout_seconds": config.stall_timeout_seconds,
                "pre_barrier": config.pre_barrier,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    worst_error = 0.0
    max_elapsed = 0.0
    total_elapsed = 0.0

    for iteration in range(1, config.iterations + 1):
        if config.pre_barrier:
            _scalar_barrier(mx, group, config.stream)

        delayed_seconds = 0.0
        if (
            config.delay_rank is not None
            and rank == config.delay_rank
            and iteration == config.delay_iteration
            and config.delay_seconds > 0
        ):
            delayed_seconds = config.delay_seconds
            print(
                json.dumps(
                    {
                        "event": "rank_delay",
                        "rank": rank,
                        "iteration": iteration,
                        "seconds": delayed_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delayed_seconds)

        started = time.perf_counter()
        with _IterationWatchdog(
            timeout_seconds=config.stall_timeout_seconds,
            rank=rank,
            iteration=iteration,
        ):
            reduced = _all_sum(mx, value, group, config.stream)
            error = _max_abs_error(mx, reduced, float(world_size))
        elapsed = time.perf_counter() - started

        worst_error = max(worst_error, error)
        max_elapsed = max(max_elapsed, elapsed)
        total_elapsed += elapsed

        observation = IterationObservation(
            rank=rank,
            world_size=world_size,
            iteration=iteration,
            stream=config.stream,
            payload_bytes=config.payload_bytes,
            elapsed_seconds=elapsed,
            delayed_seconds=delayed_seconds,
            max_abs_error=error,
        )
        print(json.dumps(observation.as_dict(), sort_keys=True), flush=True)

        if error != 0.0:
            print(
                json.dumps(
                    {
                        "event": "incorrect_collective",
                        "rank": rank,
                        "iteration": iteration,
                        "max_abs_error": error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2

    print(
        json.dumps(
            {
                "event": "summary",
                "rank": rank,
                "world_size": world_size,
                "stream": config.stream,
                "payload_bytes": config.payload_bytes,
                "iterations_completed": config.iterations,
                "mean_elapsed_seconds": total_elapsed / config.iterations,
                "max_elapsed_seconds": max_elapsed,
                "worst_max_abs_error": worst_error,
                "passed": worst_error == 0.0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify MLX ring all_sum reliability across multiple Macs."
    )
    parser.add_argument("--stream", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--size-mb", type=float, default=64.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--delay-rank", type=int)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--delay-iteration", type=int, default=1)
    parser.add_argument("--stall-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        choices=(2, 4),
        default=2,
        help="2 uses bfloat16, 4 uses float32.",
    )
    parser.add_argument(
        "--pre-barrier",
        action="store_true",
        help="Run a tiny all_sum before every measured collective.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RingProbeConfig(
        stream=cast(StreamKind, args.stream),
        size_mb=args.size_mb,
        iterations=args.iterations,
        delay_rank=args.delay_rank,
        delay_seconds=args.delay_seconds,
        delay_iteration=args.delay_iteration,
        stall_timeout_seconds=args.stall_timeout_seconds,
        dtype_bytes=args.dtype_bytes,
        pre_barrier=args.pre_barrier,
    )
    try:
        return run_probe(config)
    except (RuntimeError, ValueError) as exception:
        print(
            json.dumps(
                {"event": "error", "error": str(exception)}, sort_keys=True
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
