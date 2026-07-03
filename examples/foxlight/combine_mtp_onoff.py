#!/usr/bin/env python3
"""Combine two harness runs of the ``mtp-benchmark`` test set into an MTP
on-vs-off throughput table (the served / llama_server engine equivalent of the
MLX speculative-decoding table).

Both runs must use the same model set and the ``mtp-benchmark`` test set on the
same node build, differing only by the node env
``SKULK_LLAMA_SERVER_FORCE_NO_SPEC``:

- the MTP-ON run leaves it unset (speculation active),
- the MTP-OFF run sets it to ``1`` (plain decode of the identical GGUF).

For each model we take the median ``wall_tps`` across the run's repetitions
(matching the MLX protocol's median-of-N), then emit a Markdown table with
Plain / With MTP / Gain columns. Median, not mean, so a single slow rep (a
cold cache, a GC pause) does not skew the number.

Usage:
    combine_mtp_onoff.py <on_run_dir> <off_run_dir> [--out table.md]

Each run dir is a harness run directory containing ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Human-facing class label per model, mirroring the prior served-MTP benchmark
# note so the published table reads the same way. Unlisted models fall back to
# an empty class cell rather than failing.
_MODEL_CLASS: dict[str, str] = {
    "unsloth/Qwen3.5-9B-MTP-GGUF": "dense, small",
    "unsloth/Qwen3.6-27B-MTP-GGUF": "dense, mid",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "MoE (A3B)",
    "google/gemma-4-31B-it-qat-q4_0-gguf": "dense, draft-model",
    "unsloth/Qwen3.5-122B-A10B-MTP-GGUF": "MoE (A10B), large",
}

# The single benchmark test in the mtp-benchmark test set. Kept explicit so a
# run that also contains other tests (a combined test set) still measures the
# right one.
_BENCH_TEST = "greedy-200"


def _median_wall_tps(report: dict, test_name: str) -> dict[str, float]:
    """Return {model_id: median wall_tps} over the given test's repetitions.

    Reps that did not pass or that carry no ``wall_tps`` are ignored; a model
    with no usable rep is omitted from the result (the caller renders it as a
    gap rather than inventing a number).
    """
    by_model: dict[str, list[float]] = {}
    for result in report.get("results", []):
        if result.get("test_name") != test_name:
            continue
        if not result.get("passed"):
            continue
        wall_tps = (result.get("metrics") or {}).get("wall_tps")
        if wall_tps is None:
            continue
        by_model.setdefault(result["model_id"], []).append(float(wall_tps))
    return {m: statistics.median(v) for m, v in by_model.items() if v}


def _load_report(run_dir: Path) -> dict:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"no report.json in {run_dir}")
    return json.loads(report_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("on_run_dir", type=Path, help="harness run dir, MTP on")
    parser.add_argument("off_run_dir", type=Path, help="harness run dir, MTP off")
    parser.add_argument("--out", type=Path, default=None, help="write table here")
    args = parser.parse_args()

    on = _median_wall_tps(_load_report(args.on_run_dir), _BENCH_TEST)
    off = _median_wall_tps(_load_report(args.off_run_dir), _BENCH_TEST)

    # Union of models, preserving the on-run order where possible so the table
    # reads small -> large as the model set is authored.
    models = list(on) + [m for m in off if m not in on]

    lines = [
        "| Model | Class | Plain (tok/s) | With MTP (tok/s) | Gain |",
        "|---|---|---:|---:|---:|",
    ]
    for model in models:
        cls = _MODEL_CLASS.get(model, "")
        plain = off.get(model)
        mtp = on.get(model)
        short = model.split("/")[-1].removesuffix("-GGUF")
        if plain and mtp:
            gain = f"**{mtp / plain:.2f}x**"
            lines.append(
                f"| `{short}` | {cls} | {plain:.2f} | {mtp:.2f} | {gain} |"
            )
        else:
            # Surface a missing arm rather than dropping the row silently.
            plain_s = f"{plain:.2f}" if plain else "n/a"
            mtp_s = f"{mtp:.2f}" if mtp else "n/a"
            lines.append(f"| `{short}` | {cls} | {plain_s} | {mtp_s} | n/a |")

    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
