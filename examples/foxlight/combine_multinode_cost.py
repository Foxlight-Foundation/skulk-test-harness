#!/usr/bin/env python3
"""Combine a SOLO and a POOLED harness run into a multi-node cost table.

Both runs use the ``multinode-cost`` model set and ``sustained-300`` test on
the same fleet, differing only by placement width:

- SOLO: single node (kite4), ``--min-nodes 1``,
- POOLED: forced across two nodes (kite4+kite5) as a llama.cpp RPC pair,
  ``--min-nodes 2 --instance-meta LlamaRpc``.

The model fits the solo node, so it never *needs* the second node; the delta is
the cost of pooling (interconnect + weight-split + cross-node KV). For each
model we take the median across repetitions (median, not mean, so one cold rep
does not skew it) of decode throughput (``wall_tps``) and time-to-first-token
(``ttft_s``), and emit a Markdown table:

    Model | Solo tok/s | Pooled tok/s | Decode cost | Solo TTFT | Pooled TTFT

Decode cost is ``(solo - pooled) / solo`` as a percentage: how much decode
throughput pooling costs when the model did not need it. Positive = pooling is
slower (the expected case).

Usage:
    combine_multinode_cost.py <solo_run_dir> <pooled_run_dir> [--out table.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_BENCH_TEST = "sustained-300"


def _median_metric(report: dict, test_name: str, metric: str) -> dict[str, float]:
    """Return {model_id: median <metric>} over the test's passing reps."""
    by_model: dict[str, list[float]] = {}
    for result in report.get("results", []):
        if result.get("test_name") != test_name or not result.get("passed"):
            continue
        value = (result.get("metrics") or {}).get(metric)
        if value is None:
            continue
        by_model.setdefault(result["model_id"], []).append(float(value))
    return {m: statistics.median(v) for m, v in by_model.items() if v}


def _placement_width(report: dict) -> dict[str, int]:
    """Return {model_id: number of nodes} from the run's placements."""
    widths: dict[str, int] = {}
    for placement in report.get("placements", []):
        model_id = placement.get("model_id")
        node_ids = placement.get("node_ids") or []
        if model_id:
            widths[model_id] = len(node_ids)
    return widths


def _load_report(run_dir: Path) -> dict:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"no report.json in {run_dir}")
    return json.loads(report_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solo_run_dir", type=Path, help="harness run dir, SOLO")
    parser.add_argument("pooled_run_dir", type=Path, help="harness run dir, POOLED")
    parser.add_argument("--out", type=Path, default=None, help="write table here")
    args = parser.parse_args()

    solo_report = _load_report(args.solo_run_dir)
    pooled_report = _load_report(args.pooled_run_dir)

    solo_tps = _median_metric(solo_report, _BENCH_TEST, "wall_tps")
    pooled_tps = _median_metric(pooled_report, _BENCH_TEST, "wall_tps")
    solo_ttft = _median_metric(solo_report, _BENCH_TEST, "ttft_s")
    pooled_ttft = _median_metric(pooled_report, _BENCH_TEST, "ttft_s")
    solo_width = _placement_width(solo_report)
    pooled_width = _placement_width(pooled_report)

    models = list(solo_tps) + [m for m in pooled_tps if m not in solo_tps]

    lines = [
        "| Model | Solo tok/s | Pooled tok/s | Decode cost | Solo TTFT s | Pooled TTFT s | Widths (solo/pooled) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        short = model.split("/")[-1].removesuffix("-GGUF")
        s_tps = solo_tps.get(model)
        p_tps = pooled_tps.get(model)
        s_ttft = solo_ttft.get(model)
        p_ttft = pooled_ttft.get(model)
        widths = f"{solo_width.get(model, '?')}/{pooled_width.get(model, '?')}"
        if s_tps and p_tps:
            cost = (s_tps - p_tps) / s_tps * 100.0
            lines.append(
                f"| `{short}` | {s_tps:.1f} | {p_tps:.1f} | **{cost:+.0f}%** | "
                f"{_fmt(s_ttft)} | {_fmt(p_ttft)} | {widths} |"
            )
        else:
            lines.append(
                f"| `{short}` | {_fmt(s_tps)} | {_fmt(p_tps)} | n/a | "
                f"{_fmt(s_ttft)} | {_fmt(p_ttft)} | {widths} |"
            )

    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}")


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


if __name__ == "__main__":
    main()
