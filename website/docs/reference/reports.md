---
title: Reports
---

Every normal harness plan or run writes a report directory under `output_dir`.
The default output directory is `runs/`.

## Directory Shape

```text
runs/<run-id>/
  report.json
  events.jsonl
  summary.md
  artifacts/
```

Stability suites write:

```text
runs/<run-id>/
  report.json
  summary.md
```

## Files

| File | Audience | Purpose |
| --- | --- | --- |
| `summary.md` | Humans | Fast reading and pull request comments |
| `report.json` | Automation | Full structured report |
| `events.jsonl` | Scripts and logs | One event per line |
| `artifacts/` | Humans and scripts | Generated files such as HTML/code outputs and speech audio |

## Summary Sections

| Section | Meaning |
| --- | --- |
| Models | Resolved model ids and how they were selected |
| Placements | Instances, nodes, reuse, and readiness |
| Results | Pass or fail for each model, test, and repetition |
| Issues | Run-level and result-level problems |

Speech synthesis, streaming speech synthesis, and speech roundtrip results
include an artifact path when they generate audio. Open the recorded path to
inspect or listen to the exact bytes the harness scored. Streaming speech
synthesis also writes a `.stream.json` sidecar next to the audio with chunk
count, first-byte latency, stream span, per-chunk byte sizes, and per-chunk
arrival offsets. Realtime transcription writes a `.realtime.json` sidecar
containing the selected TTS fixture model, PCM frame shape, protocol event
types, first-transcript timing, cancellation outcome, and sanitized provider
counter deltas. It contains no audio payload, route, or node identifier.
Non-translation speech roundtrips also record `word_error_rate` in their result
metrics, including successful runs, so semantic quality remains comparable over
time instead of being reduced to a pass/fail threshold.

## Important Metrics

| Metric | Meaning |
| --- | --- |
| `ttft_s` | Time to first token in seconds |
| `wall_tps` | Approximate decode tokens per second from wall-clock timing |
| `skulk_prompt_tps` | Prompt throughput reported by Skulk when available |
| `skulk_generation_tps` | Generation throughput reported by Skulk when available |
| `output_chars` | Visible output character count |
| `generated_chars` | Visible plus separated reasoning character count |
| `chunks` | Number of streamed chunks observed |
| `word_error_rate` | STT transcript edit distance divided by source-prompt word count for a speech roundtrip |

For a `kind: concurrent` test the same block also carries the load aggregates
below (all `null` for single-request tests). The aggregate throughput is also
copied into `skulk_generation_tps` so a reader that knows only the single
headline field still sees the concurrency number.

| Metric | Meaning |
| --- | --- |
| `concurrency` | Simultaneous in-flight requests driven |
| `concurrent_total_requests` | Total requests issued across all workers |
| `concurrent_succeeded` / `concurrent_failed` | Requests that passed or failed scoring under load |
| `aggregate_generation_tps` | Total generated tokens divided by the wall span from first request start to last request end |
| `per_request_generation_tps_mean` / `_p50` / `_p90` | Per-request decode-rate distribution under load |
| `ttft_p50_s` / `ttft_p90_s` | Time-to-first-token distribution under load |
| `wall_span_s` | Wall-clock span used as the aggregate-throughput denominator |

## The Fingerprint

Every `report.json` from a plan or run carries a top-level `fingerprint`
block (current `schema_version`: `2.2`). It exists for one reason: **a number
should never be separated from what produced it**. A "45 tok/s" without the
Skulk version, the nodes, and the cache conditions behind it is not a
measurement; the fingerprint makes each report self-describing, so it stays
meaningful in a comparison next month or on the public ledger next year.

Every fingerprint probe is best-effort: a probe that fails records `null` or
`"unknown"` (plus a warning issue) rather than failing the report write.

The four sections:

### `source_context`

Why the run happened and from which code.

| Field | Meaning |
| --- | --- |
| `run_reason` | Free-form reason string (`"unspecified"` unless provided) |
| `visibility` | `private` or `public` |
| `operator_note` | Optional free-form note (the run name, when one was given) |
| `repositories` | Git provenance: the harness checkout (name, path, branch, short commit, dirty flag) and, when the cluster reports it, the Skulk commit |

### `runtime`

The harness process's own runtime plus the cluster's Skulk version.

| Field | Meaning |
| --- | --- |
| `python` | Python version running the harness |
| `platform` | Harness OS, kernel release, and machine architecture |
| `harness_packages` | Versions of the harness and its key client packages |
| `skulk_version`, `skulk_commit` | Read from the API node's diagnostics (the harness is an HTTP client; Skulk and MLX run on the nodes, not here) |

### `cluster`

The cluster the run executed against, as seen in `/state` at run time.

| Field | Meaning |
| --- | --- |
| `api_base_url` | The API the harness talked to |
| `api_node_id`, `master_node_id` | Which node answered, and which was master |
| `node_count` | Nodes visible in cluster state |
| `nodes` | One entry per node, see below |
| `topology_label` | Sorted, joined friendly names, a human-readable cluster shape |

Each entry in `nodes` describes one machine:

| Field | Meaning |
| --- | --- |
| `node_id` | The node's cluster identifier |
| `friendly_name` | The operator-given node name, when known |
| `ram_total_bytes` | Total RAM from memory telemetry |
| `accelerator_vendor` | GPU/accelerator vendor from system telemetry (`apple`, `amd`, ...) |
| `accelerator_name` | Accelerator marketing name (`M4`, `Radeon 8060S`, ...), `null` when telemetry has not landed or predates the field |
| `skulk_version` | That node's reported Skulk version |
| `system_telemetry_present` | Whether system telemetry existed for this node at fingerprint time |
| `memory_telemetry_present` | Whether memory telemetry existed for this node at fingerprint time |

The report's `placements[].node_ids` list the exact nodes that served each
model. Joining those ids against `fingerprint.cluster.nodes` attributes every
result to the machines that produced it; this is how the
[community ledger](../guides/submit-to-the-ledger.md) labels a submitted run
with its real hardware.

### `cache_state`

The store and instance cache conditions, recorded from the run's own flags.

| Field | Meaning |
| --- | --- |
| `ensure_store_downloads` | Whether the run forced store downloads |
| `reuse_existing_instances` | Whether existing instances could be reused |
| `retain_instances` | Whether created instances were left running |
| `delete_staged_models` | Whether staged weights were evicted afterwards |
| `classification` | One of `cold`, `warm`, `mixed`, `unknown` |

The classification is deliberately conservative: it distinguishes what the
run REQUESTED from any claim of controlled conditions, and it never asserts
`cold` from flags alone. `compare` uses it
to flag [cache-mismatched comparisons](../guides/compare-runs.md#cache_mismatch).

## A Small Result Example

```json
{
  "model_id": "mlx-community/Qwen3.5-9B-4bit",
  "test_name": "concise-factual-answer",
  "repetition": 1,
  "passed": true,
  "output_text": "Paris",
  "metrics": {
    "elapsed_s": 1.42,
    "ttft_s": 0.81,
    "output_chars": 5,
    "generated_chars": 5,
    "chunks": 2,
    "wall_tps": 18.0
  },
  "issues": [],
  "artifact_path": null
}
```

## How To Read Failures

Use this order:

1. Open `summary.md`.
2. Read `Issues`.
3. Find the failing result row.
4. Open `report.json` if the summary is not enough.
5. Compare the test's `success` rules with `output_text`, `reasoning_text`, and
   `tool_calls`.

Common failure patterns:

| Symptom | Likely next step |
| --- | --- |
| No models selected | Check the model set selector and model store |
| Placement not ready | Check cluster capacity and runner logs |
| Required substring missing | Check whether the prompt is too open-ended |
| Tool call missing | Confirm the model and backend support tool calls |
| Throughput below floor | Compare with a recent known-good benchmark |
| Context error expected but absent | Check Skulk context admission behavior |

## Using Reports Beyond One Run

- [Compare runs](../guides/compare-runs.md): like-for-like deltas between two
  run sets, with trust guards.
- [Submit to the ledger](../guides/submit-to-the-ledger.md): publish a run to
  the community benchmarks ledger, redacted client-side.
