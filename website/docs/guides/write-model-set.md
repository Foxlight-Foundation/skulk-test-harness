---
title: Write A Model Set
---

A model set is a named group of models. You can list exact model ids, select
models from Skulk metadata, or ask the harness to seed a model card from Hugging
Face.

## Where Model Sets Live

The public file is:

```text
configs/model_sets.yaml
```

Your local config points at that file:

```yaml
model_sets_path: configs/model_sets.yaml
```

For private work, make your own file and point `skulk-harness.yaml` at it:

```yaml
model_sets_path: local/model_sets.yaml
```

Keep private model ids out of public files if they should not be published.

## The Smallest Model Set

This set names one exact model:

```yaml
model_sets:
  my-first-model:
    name: my-first-model
    description: One explicit model for my first smoke test.
    models:
      - mlx-community/Qwen3.5-9B-4bit
```

The map key and the `name` field must match. If they do not match, config
loading fails.

## Select From The Store

Use a selector when you want the harness to choose from models Skulk already
knows about:

```yaml
model_sets:
  one-store-model:
    name: one-store-model
    description: First model currently present in the Skulk model store.
    selectors:
      - source: store
        id_regex: ".*"
        max_models: 1
```

This is how `store-smoke` works.

## Select From Catalog And Store

This set looks in both the catalog and the store, filters by model id, and caps
the result count:

```yaml
model_sets:
  small-chat-models:
    name: small-chat-models
    description: Small chat-like models discovered from Skulk metadata.
    selectors:
      - source: both
        id_regex: "(qwen|llama|gemma|mistral|phi)"
        max_models: 5
```

## Select By Served Speculation Type

If model cards expose served speculation metadata, you can select by it:

```yaml
model_sets:
  draft-simple-models:
    name: draft-simple-models
    description: Models that declare draft_simple served speculation.
    selectors:
      - source: both
        id_regex: ".*"
        served_spec_types_any:
          - draft_simple
```

## Seed A Hugging Face Model Card

Seeds tell the harness about models you want available as model cards:

```yaml
model_sets:
  seeded-smoke:
    name: seeded-smoke
    description: Add one model card before resolving tests.
    huggingface_seeds:
      - model_id: mlx-community/Qwen3.5-9B-4bit
        reason: Small public model for smoke testing.
        require_mlx_community: true
```

Seeds do not magically make a huge model cheap to run. They only help the
harness make sure Skulk knows about the model card.

## Selector Fields

| Field | Meaning |
| --- | --- |
| `source` | `catalog`, `store`, or `both` |
| `family` | Optional exact family match |
| `id_contains` | Optional substring match on model id |
| `id_regex` | Optional regular expression match on model id |
| `tags_any` | Match any listed tag |
| `tasks_any` | Match any listed task |
| `capabilities_any` | Match any listed capability |
| `served_spec_types_any` | Match any listed served speculation type |
| `require_audio_streaming` | Require streaming-capable audio metadata |
| `require_audio_realtime` | Require truthful streaming plus realtime audio metadata |
| `max_models` | Stop after this many matches |

## Check Your Set

After editing, run:

```bash
uv run skulk-harness models sets --config skulk-harness.yaml
```

Then plan a run:

```bash
uv run skulk-harness plan \
  --model-set my-first-model \
  --test-set chat-tests
```

If the model cannot be placed, start by checking whether Skulk can see it in
the catalog or store:

```bash
uv run skulk-harness models catalog
uv run skulk-harness models store
```
