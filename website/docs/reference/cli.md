---
title: CLI Reference
---

Run commands with:

```bash
uv run skulk-harness <command>
```

Most commands accept `--config` or `-c` (default: `skulk-harness.yaml` in the
current directory, falling back to built-in defaults if absent).

## Top-Level Commands

| Command | Needs live cluster | Mutates cluster | Purpose |
| --- | --- | --- | --- |
| `doctor` | Yes | No | Print API and cluster summary |
| `plan` | Yes | No | Resolve models and preview run work |
| `run` without `--execute` | Yes | No | Dry-run form of `run` |
| `run --execute` | Yes | Yes | Place models and run tests |
| `goal` without `--execute` | Yes | No | Parse a constrained natural-language goal into a plan |
| `goal --execute` | Yes | Yes | Execute the parsed goal |
| `compare` | No | No | Compare two local run sets like-for-like |
| `submit` | No | No | Submit a local run to the community benchmarks ledger |

`compare` and `submit` work entirely from local report files under
`output_dir`; neither contacts a Skulk cluster (`submit` contacts only the
ledger's ingest API).

## Model Commands

| Command | Needs live cluster | Purpose |
| --- | --- | --- |
| `models sets` | No | List named model sets |
| `models catalog` | Yes | List live Skulk catalog models |
| `models store` | Yes | List models in the Skulk model store |
| `models add <model-id>` | Yes | Ask Skulk to add or fetch a model card |
| `models download <model-id>` | Yes | Request a model-store download |
| `models download <model-id> --wait` | Yes | Wait for the download to finish |

## Test Commands

| Command | Needs live cluster | Purpose |
| --- | --- | --- |
| `tests sets` | No | List named test sets |

## Stability Commands

| Command | Needs live cluster | Destructive opt-in | Purpose |
| --- | --- | --- | --- |
| `stability soak` | Yes | No | Sustained concurrent load |
| `stability failover` | Yes | `--execute-destructive` | Kill and relaunch during failover coverage |
| `stability churn` | Yes | `--execute-destructive` | Repeated crash and relaunch rounds |
| `stability refusal` | Yes | `--execute-destructive` | Impossible placement behavior coverage |

Stability suites also accept `--model`/`-m` (the model to exercise), and
per-suite knobs: `--min-nodes` (failover), `--rounds` (churn), and
`--concurrency` plus `--duration-s` (soak).

## Common Flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--config`, `-c` | Most commands | Path to harness config YAML |
| `--model-set`, `-m` | `plan`, `run` | Named model set |
| `--test-set`, `-t` | `plan`, `run` | Named test set |
| `--execute` | `run`, `goal` | Actually run live requests |
| `--dry-run` | `run`, `goal` | Plan only (the default) |
| `--ensure-store-downloads` | `run` | Request model-store downloads before placement |
| `--retain-instances` | `run` | Leave created instances running (the default) |
| `--delete-created-instances` | `run` | Delete instances the harness created |
| `--delete-staged-models` | `run` | Evict staged model weights after a run |
| `--sharding` | `plan`, `run` | `Pipeline` or `Tensor` |
| `--instance-meta` | `plan`, `run` | `MlxRing`, `MlxJaccl`, or `LlamaRpc` |
| `--min-nodes` | `plan`, `run` | Override minimum node count |
| `--exclude-nodes` | `run` | Comma-separated friendly node names to exclude from placement |
| `--fail-on-issue` | `run` | Exit non-zero on failed results or error issues (the default; disable with `--no-fail-on-issue`) |

## Compare Flags

`compare` aggregates two run sets from `output_dir` and prints per-model
deltas with trust guards. See [Compare runs](../guides/compare-runs.md).

| Flag | Required | Meaning |
| --- | --- | --- |
| `--baseline`, `-b` | Yes | Baseline run selector: a run directory, or a substring matched against run-directory names |
| `--candidate`, `-n` | Yes | Candidate run selector, same matching rules |
| `--out` | No | Write the machine-readable comparison record JSON to this path |
| `--config`, `-c` | No | Harness config YAML (locates `output_dir`) |

## Submit Flags

`submit` takes one argument (a run directory or `report.json` path), redacts
it client-side, and sends it to the community benchmarks ledger. See
[Submit to the ledger](../guides/submit-to-the-ledger.md).

| Flag | Meaning |
| --- | --- |
| `--dry-run` | Print the exact payload instead of sending |
| `--github-token` | GitHub token for attribution (also read from `GH_TOKEN`, `GITHUB_TOKEN`, or the `gh` CLI) |
| `--ingest-url` | Override the ingest API base URL (also read from the `SKULK_INGEST_URL` environment variable; defaults to the public ledger) |

## Examples

List local sets:

```bash
uv run skulk-harness models sets
uv run skulk-harness tests sets
```

Plan a run:

```bash
uv run skulk-harness plan --model-set store-smoke --test-set chat-tests
```

Execute and clean up:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

Compare two run sets:

```bash
uv run skulk-harness compare -b 20260701 -n 20260709 --out comparison.json
```

Inspect, then submit a run to the ledger:

```bash
uv run skulk-harness submit runs/<run-id> --dry-run
uv run skulk-harness submit runs/<run-id>
```

Use Foxlight production config:

```bash
uv run skulk-harness tests sets --config examples/foxlight/skulk-harness.yaml
```
