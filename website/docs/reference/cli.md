---
title: CLI Reference
---

Run commands with:

```bash
uv run skulk-harness <command>
```

Most commands accept `--config` or `-c`.

## Top-Level Commands

| Command | Needs live cluster | Mutates cluster | Purpose |
| --- | --- | --- | --- |
| `doctor` | Yes | No | Print API and cluster summary |
| `plan` | Yes | No | Resolve models and preview run work |
| `run` without `--execute` | Yes | No | Dry-run form of `run` |
| `run --execute` | Yes | Yes | Place models and run tests |
| `goal` without `--execute` | Yes | No | Parse a constrained natural-language goal into a plan |
| `goal --execute` | Yes | Yes | Execute the parsed goal |

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

## Common Flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--config`, `-c` | Most commands | Path to harness config YAML |
| `--model-set`, `-m` | `plan`, `run` | Named model set |
| `--test-set`, `-t` | `plan`, `run` | Named test set |
| `--execute` | `run`, `goal` | Actually run live requests |
| `--dry-run` | `run`, `goal` | Plan only |
| `--ensure-store-downloads` | `run` | Request model-store downloads before placement |
| `--retain-instances` | `run` | Leave created instances running |
| `--delete-created-instances` | `run` | Delete instances the harness created |
| `--delete-staged-models` | `run` | Evict staged model weights after a run |
| `--sharding` | `plan`, `run` | `Pipeline` or `Tensor` |
| `--instance-meta` | `plan`, `run` | `MlxRing` or `MlxJaccl` |
| `--min-nodes` | `plan`, `run` | Override minimum node count |
| `--fail-on-issue` | `run` | Exit non-zero on failed results or error issues |

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

Use Foxlight production config:

```bash
uv run skulk-harness tests sets --config examples/foxlight/skulk-harness.yaml
```
