---
title: Model Sets
---

Model sets live in a YAML file with one top-level key: `model_sets`.

## File Shape

```yaml
model_sets:
  store-smoke:
    name: store-smoke
    description: First model currently present in the configured Skulk model store.
    selectors:
      - source: store
        id_regex: ".*"
        max_models: 1
```

Each map key must match the model set's `name`.

## Model Set Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Name used by CLI flags |
| `description` | string | Human explanation shown in tables |
| `models` | list of strings | Exact model ids |
| `selectors` | list | Rules that expand from Skulk catalog or store metadata |
| `huggingface_seeds` | list | Optional model cards to seed through Skulk |

## Selector Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `source` | `catalog`, `store`, or `both` | Where to search |
| `family` | string | Exact model family match |
| `id_contains` | string | Simple substring match |
| `id_regex` | string | Regular expression match |
| `tags_any` | list of strings | Match any listed tag |
| `tasks_any` | list of strings | Match any listed task |
| `capabilities_any` | list of strings | Match any listed capability |
| `served_spec_types_any` | list of strings | Match served speculation metadata |
| `max_models` | integer | Stop after this many matches |

## Hugging Face Seed Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `model_id` | string | Hugging Face model id |
| `reason` | string | Why this seed exists |
| `require_mlx_community` | boolean | Require `mlx-community/` by default |

## Public Built-In Sets

| Name | Purpose |
| --- | --- |
| `store-smoke` | First model currently present in the store |
| `store-all` | Every model currently present in the store |
| `catalog-small-text` | Small text-generation candidates |
| `embeddings` | Sentence transformer embedding smoke target |
| `speech-tts` | Text-to-speech targets resolved through Skulk's store |
| `speech-stt` | Speech-to-text targets resolved through Skulk's store |
| `vision` | Vision-capable models discovered from metadata |
| `served-spec-draft-simple` | Models with `draft_simple` served speculation |
| `served-spec-draft-eagle3` | Models with `draft_eagle3` served speculation |

## Validation

List sets after editing:

```bash
uv run skulk-harness models sets --config skulk-harness.yaml
```

Then plan a run with the new name:

```bash
uv run skulk-harness plan --model-set my-model-set --test-set chat-tests
```
