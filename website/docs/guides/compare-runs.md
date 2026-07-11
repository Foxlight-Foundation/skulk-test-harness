---
title: Compare Runs
---

You ran a benchmark last week. You changed something (a Skulk version, a
cable, a model quant) and ran it again today. Is it actually faster?

`compare` answers that question the reproducible way: it aggregates two run
sets, prints per-model deltas for the headline metrics, and, crucially, tells
you when the comparison is **not fair**, instead of letting an unfair delta
look like a result.

## The command

```bash
uv run skulk-harness compare -b <baseline> -n <candidate>
```

| Flag | Meaning |
| --- | --- |
| `--baseline`, `-b` | The "before" side: a run directory path, or a substring matched against run-directory names under `runs/` |
| `--candidate`, `-n` | The "after" side, same matching rules |
| `--out <path>` | Also write the machine-readable comparison record as JSON |
| `--config`, `-c` | Harness config (used to find your `output_dir`) |

Selectors are flexible on purpose. All of these work:

```bash
# Two explicit run directories
uv run skulk-harness compare -b runs/20260701-090000-store-smoke-chat-tests \
                             -n runs/20260709-141530-store-smoke-chat-tests

# Every run whose directory name contains a date prefix
uv run skulk-harness compare -b 20260701 -n 20260709

# Every run of a named battery, before and after
uv run skulk-harness compare -b dense-singles-before -n dense-singles-after
```

A substring selector can match several run directories; that is a feature.
Comparing three baseline runs against three candidate runs gives each side
more samples, and medians over more samples are worth more.

Stability-suite reports are skipped automatically: a comparison only operates
over model-scoring runs.

## What the numbers mean

For each model that appears on both sides, `compare` prints the median of
each metric, the delta, and any guards:

| Metric | Meaning | Better is |
| --- | --- | --- |
| `decode_tps` | Steady-state decode tokens per second. Uses Skulk's own generation throughput when the run recorded it (it excludes prompt processing time); falls back to wall-clock throughput otherwise | Higher |
| `ttft_s` | Time to first token, in seconds | Lower |
| `wall_tps` | Tokens per second over the whole request wall clock, including prompt processing | Higher |

Two aggregation rules keep the medians honest:

- **Medians, not means.** One stalled request should not drag the headline
  number; one lucky one should not inflate it.
- **Short outputs are excluded, but counted.** A five-character answer
  produces a meaningless throughput figure (the timing is all overhead), so
  outputs under 20 generated tokens do not enter the throughput medians. They
  are counted separately so the exclusion is visible, never silent.

## The trust guards

A guard is `compare` telling you: this delta may not mean what it looks like
it means. Guards appear per model and for the run set as a whole.

### `low_sample`

**What it means:** one side has fewer than 3 substantive samples for the
model. A median of one number is just that number.

**What to do:** run the benchmark again (or use a broader selector that
matches more runs) so each side has at least 3 real samples. Test sets can
also set `repetitions` on a test to gather samples in one run.

### `short_output_dominant`

**What it means:** more than half of a side's outputs were too short to time
honestly, so the median rests on a minority of the data. This usually means
the test set is a correctness smoke, not a benchmark.

**What to do:** benchmark with a test set that generates real output (the
built-in `throughput` set exists for this), then compare those runs.

### `node_set_mismatch`

**What it means:** the model ran on a different number of nodes on each side.
A model served by one node and the same model sharded across three are
different systems; their throughput numbers are not comparable.

**What to do:** re-run one side so both use the same placement. Pin the shape
with `--min-nodes` or `--exclude-nodes` if the planner keeps choosing
differently.

### `cache_mismatch`

**What it means:** the two sides ran under different cache conditions as
recorded in their fingerprints (for example, one side forced fresh store
downloads while the other reused warm instances). Cold starts measure
loading; warm runs measure serving.

**What to do:** re-run with matching flags. For like-for-like serving
benchmarks, both sides should reuse instances the same way and neither should
be doing first-time downloads.

### `issue_marked`

**What it means:** at least one side recorded issues (failures, warnings)
for this model. A run that struggled is not a clean measurement.

**What to do:** open the run's `summary.md`, resolve the issues, and re-run
before trusting the delta.

### `decode_tps_unavailable`

**What it means:** at least one side had no usable decode throughput median,
so the headline metric fell back or is missing. Common on older runs where
Skulk did not report generation throughput.

**What to do:** prefer `wall_tps` for that comparison, or re-run the baseline
on a current Skulk so both sides report real decode numbers.

### `missing_fingerprint`

**What it means:** one side has no fingerprint at all, so `compare` cannot
verify the cluster, Skulk version, or cache conditions were comparable. This
is normal for runs recorded by old harness versions.

**What to do:** treat the delta as indicative, not conclusive. If the
baseline matters, reproduce it on the current harness so it carries a
fingerprint.

### `model_only_one_side`

**What it means:** a model appears in only one run set, so there is nothing
to compare it against. It is listed so you can see it was not silently
dropped.

**What to do:** nothing, unless you expected it on both sides; then check the
model set the other run used.

## Reading a result

A delta with **no guards** on a model whose sides each have several samples is
a real measurement: you can act on it.

A delta **with guards** is a hint. The guards are not errors and `compare`
still prints the numbers; they are the reasons a careful person would
hesitate. Fix what the guard points at and compare again.

To keep the comparison for automation or records:

```bash
uv run skulk-harness compare -b <baseline> -n <candidate> --out comparison.json
```

The JSON record contains every per-model summary, delta, and guard.

## Related pages

- [Reports](../reference/reports.md): where the metrics and fingerprints come from
- [Submit to the ledger](submit-to-the-ledger.md): sharing a run once you trust it
