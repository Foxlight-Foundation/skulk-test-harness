---
title: Compare Runs
---

`compare` answers whether a performance change survives a like-for-like
comparison. It never builds a model-wide median.

```bash
uv run skulk-harness compare -b <baseline> -n <candidate>
```

| Flag | Meaning |
| --- | --- |
| `--baseline`, `-b` | Before-side run directory or substring under `output_dir` |
| `--candidate`, `-n` | After-side run directory or substring under `output_dir` |
| `--out <path>` | Write the machine-readable schema 2.0 comparison record |
| `--config`, `-c` | Harness config used to locate `output_dir` |

Selectors may match multiple run directories. Stability-suite reports are
skipped automatically.

## What Is Compared

Each row is one exact execution series defined by:

- model, suite, and test;
- `protocol_id`;
- metric source (`client_exact`, `engine_reported`, or `client_approx`);
- exact placement hardware and memory facts;
- resolved backend set;
- instance type, sharding family, and shard types.

Different tests, protocols, hardware, backends, sources, or placement shapes
produce different rows. A mismatch is reported as `series_only_one_side`; it
does not produce a percentage.

Within one run, valid repetitions are reduced to a median. Across a selector,
the comparator summarizes those distinct run-level medians. Three repetitions
in one run improve that run's estimate but do not pretend to be three
longitudinal runs.

Only `chat`, `code`, and `artifact` results enter text-decode comparison.
Failed results and outputs below 20 tokens do not feed throughput points. Each
source uses its own token basis: exact and engine measurements use
`skulk_generation_tokens`; approximate measurements use
`approx_output_tokens`. Tool, concurrency, cancellation, expected-error,
speech, and embedding tests remain in their reports but are not mixed into
ordinary decode rows.

The terminal prints decode TPS for every matching source. The JSON record also
contains TTFT from the same result population. No source silently falls back to
another.

## Guards

| Guard | Meaning |
| --- | --- |
| `series_only_one_side` | No full series identity exists on the other side |
| `series_identity_incomplete` | Protocol, exact hardware, backend, or placement metadata is missing; no delta is computed |
| `low_sample` | Fewer than three valid run-level points exist on a side |
| `short_output_dominant` | Short repetitions outnumber the valid run-level evidence |
| `issue_marked` | A contributing result recorded an issue |
| `decode_tps_unavailable` | A matched complete series has no valid TPS point on a side |
| `cache_mismatch` | Run-set cache classifications differ |
| `missing_fingerprint` | At least one report lacks an environment fingerprint |

Incomplete legacy reports remain visible in the comparison record, but they
cannot yield a performance delta. Re-run with the current harness to record the
protocol, exact placement, and backend.

## Examples

```bash
# Explicit run directories
uv run skulk-harness compare \
  -b runs/20260701-090000-throughput \
  -n runs/20260709-141530-throughput

# Date or battery-name selectors
uv run skulk-harness compare -b 20260701 -n 20260709
uv run skulk-harness compare -b dense-before -n dense-after

# Durable machine-readable record
uv run skulk-harness compare -b dense-before -n dense-after \
  --out comparison.json
```

## Related Pages

- [Reports](../reference/reports.md): metric, protocol, placement, and fingerprint fields
- [Submit to the ledger](submit-to-the-ledger.md): publishing a run
