---
title: First Local Run
---

This guide walks through a first real run against a Skulk API. It uses the
public defaults and the smallest useful test set.

## Before You Start

You need:

- the harness installed with `uv sync`
- a private `skulk-harness.yaml`
- a Skulk API reachable from your machine
- at least one model available in the Skulk model store for `store-smoke`

If you do not know whether your cluster has a model in the store, that is fine.
The guide checks that before running prompts.

## 1. Confirm Your Config

Open `skulk-harness.yaml` and make sure the first line points at your Skulk API:

```yaml
api_base_url: http://localhost:52415
```

For a remote or forwarded API, it might look like:

```yaml
api_base_url: http://kite1:52415
```

or:

```yaml
api_base_url: http://127.0.0.1:52415
```

## 2. Check The Cluster

Run:

```bash
uv run skulk-harness doctor
```

Look for these fields:

| Field | Good sign |
| --- | --- |
| API | It matches your config |
| Known models | Greater than zero |
| Cluster memory nodes | Greater than zero |
| State drift issues | Usually zero |

If `doctor` cannot connect, check the API URL and whether Skulk is running.

## 3. Check The Store

List the current model store:

```bash
uv run skulk-harness models store
```

If at least one model appears, you can use `store-smoke`.

If the store is empty, you have two options:

| Option | Command |
| --- | --- |
| Add a model card | `uv run skulk-harness models add mlx-community/Qwen3.5-9B-4bit` |
| Request a store download | `uv run skulk-harness models download mlx-community/Qwen3.5-9B-4bit --wait` |

The exact model you choose should match what your Skulk cluster can serve.

:::warning
Large model downloads can take a long time and use a lot of disk space. Ask your
cluster operator before downloading unfamiliar models on shared hardware.
:::

## 4. Plan The Run

Run:

```bash
uv run skulk-harness plan \
  --model-set store-smoke \
  --test-set chat-tests
```

The command writes a report directory and prints a compact summary. A plan can
still fail if no model matches the set or if placement preview cannot find a
usable option.

## 5. Execute The Run

When the plan is sensible, run:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

The run will:

1. Resolve the selected model.
2. Reuse an existing compatible instance if one is available.
3. Otherwise ask Skulk to place a new instance.
4. Wait for readiness.
5. Send each chat test.
6. Score the responses.
7. Delete created instances because you passed `--delete-created-instances`.
8. Write report files.

## 6. Read The Result

Open `summary.md` in the printed report directory.

A small passing run might show:

```md
# Skulk Harness Run: harness-2026-06-30t12-00-00

- Model set: `store-smoke`
- Test set: `chat-tests`
- Mode: `execute`

## Results

| Model | Test | Rep | Pass | TTFT s | Wall TPS | Content Chars | Generated Chars |
|---|---|---:|---:|---:|---:|---:|---:|
| `mlx-community/Qwen3.5-9B-4bit` | `concise-factual-answer` | 1 | yes | 1.20 | 42.50 | 5 | 5 |
```

Read failures from the bottom up. The `Issues` section usually has the most
direct clue.

## 7. Try A Different Test Set

Once chat works, try one of these:

| Test set | What it checks |
| --- | --- |
| `code-tests` | Simple code generation |
| `tool-tests` | OpenAI-style tool calls with static mock results |
| `cancellation` | Mid-stream cancellation and follow-up health |
| `throughput` | Longer generation and throughput metrics |

Example:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set tool-tests \
  --execute \
  --delete-created-instances
```

Tool tests need a model and serving path that support tool calling. If they
fail on a plain chat model, that may be expected.
