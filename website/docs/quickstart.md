---
title: Quickstart
---

This quickstart gets you from a fresh checkout to your first safe harness
commands. It assumes you are new to both Skulk and cluster testing.

## Requirements

| Requirement | Why you need it |
| --- | --- |
| Python 3.13 or newer | The harness package targets Python 3.13 |
| [`uv`](https://docs.astral.sh/uv/) | Installs dependencies and runs the CLI |
| A local checkout of this repository | Contains the configs, tests, and examples |
| Node.js 20 or newer | Only needed if you edit or build this docs site |
| A reachable Skulk API | Only needed for `doctor`, `plan`, `run`, model store commands, and stability suites |

## 1. Install The Harness

From the repository root:

```bash
uv sync
```

This installs the local package into the project environment. You will run the
CLI with `uv run skulk-harness ...`.

## 2. Inspect Local YAML Without A Cluster

These commands do not call Skulk. They only load the public example config and
print the model and test set names it points at:

```bash
uv run skulk-harness models sets --config skulk-harness.example.yaml
uv run skulk-harness tests sets --config skulk-harness.example.yaml
```

You should see tables named `Model Sets` and `Test Sets`.

:::tip
If these commands fail, fix your local Python environment before thinking about
Skulk. The cluster is not involved yet.
:::

## 3. Make A Private Local Config

Copy the example config:

```bash
cp skulk-harness.example.yaml skulk-harness.yaml
```

`skulk-harness.yaml` is ignored by git. Put your private cluster URL or local
paths there.

For a local Skulk API, the default is already correct:

```yaml
api_base_url: http://localhost:52415
model_sets_path: configs/model_sets.yaml
test_sets_path: configs/test_sets.yaml
output_dir: runs
cluster_nodes: {}
```

If your Skulk API runs somewhere else, change only `api_base_url` first.

## 4. Check A Live Skulk API

Now you need a reachable Skulk cluster:

```bash
uv run skulk-harness doctor
```

`doctor` prints a compact table with the configured API, node id, model count,
memory node count, instance count, runner state count, and runner drift warning
count.

If your cluster is down, skip to [Troubleshooting](troubleshooting.md). The
harness cannot start Skulk for you.

## 5. Plan A Run

Planning asks Skulk what is available and writes a report, but it does not run
the prompts:

```bash
uv run skulk-harness plan \
  --model-set store-smoke \
  --test-set chat-tests
```

The `store-smoke` model set chooses one model from the Skulk model store. The
`chat-tests` test set checks basic chat behavior.

You will get a report directory like this:

```text
runs/harness-2026-06-30t12-00-00/
  report.json
  events.jsonl
  summary.md
```

## 6. Run Real Requests

When the plan looks sane, execute the same pair:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

The key flag is `--execute`. Without it, `run` behaves like a dry-run plan.

`--delete-created-instances` tells the harness to clean up instances it created
for the run. That is usually the right first choice because it leaves the
cluster closer to how it started.

## 7. Read The Summary

Open the printed report directory and read `summary.md` first. It is the
human-readable result:

| Section | What it means |
| --- | --- |
| Models | Which model ids were selected |
| Placements | Which instances and nodes were used |
| Results | One row per model, test, and repetition |
| Issues | Failures, warnings, and useful evidence |

Use `report.json` when you need structured data for automation.

## The Short Version

```bash
uv sync
cp skulk-harness.example.yaml skulk-harness.yaml
uv run skulk-harness models sets
uv run skulk-harness tests sets
uv run skulk-harness doctor
uv run skulk-harness plan --model-set store-smoke --test-set chat-tests
uv run skulk-harness run --model-set store-smoke --test-set chat-tests --execute --delete-created-instances
```
