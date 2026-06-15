# skulk-test-harness

Agent-controlled test harness workbench for Skulk.

This repository is intended to hold integration, scenario, and compatibility
testing tools around the Skulk runtime without coupling experimental harness
code directly into the main Skulk repository.

## V1 Capabilities

- Named model sets in `configs/model_sets.yaml`
- Named test sets in `configs/test_sets.yaml`
- Live Skulk API discovery through `skulk-harness.yaml`
- Store-aware model selection through `/store/registry`
- Model-card addition through `/models/add`
- Optional model-store download requests through `/store/models/{model}/download`
- Placement planning through `/instance/previews`
- Placement execution through `/place_instance`
- Streaming chat execution with wall-clock TTFT and approximate TPS
- OpenAI-style tool-call tests with expected function and argument checks
- JSON, JSONL, Markdown reports under `runs/`
- Deterministic local natural-language goal parser for agent use

## Quick Start

```bash
uv sync
uv run skulk-harness doctor
uv run skulk-harness models sets
uv run skulk-harness tests sets
uv run skulk-harness plan --model-set qwen35-small --test-set chat-tests
```

Cluster-mutating work is opt-in. `run` defaults to dry-run unless `--execute`
is passed:

```bash
uv run skulk-harness run \
  --model-set gpt-oss-20b \
  --test-set gpt-oss-20b-complete \
  --execute \
  --retain-instances
```

Agent-oriented natural language goals are supported for named sets:

```bash
uv run skulk-harness goal \
  "step through the models in gemma4-family, make minimum node placements, run asteroids-challenge"
```

Add `--execute` to actually place models and run tests.

## Built-In Sets

Model sets:

- `smoke`
- `qwen35-small`
- `gemma4-family`
- `moe-family`
- `mtp-tests`
- `gpt-oss-20b`
- `store-all`

Test sets:

- `chat-tests`
- `code-tests`
- `asteroids-challenge`
- `gpt-oss-20b-complete`

## Safety Notes

This harness is designed to coexist with other operators using the same Skulk
cluster. Dry-run planning is the default. Executions reuse existing placements
by default and retain harness-created instances unless explicitly asked to
delete them.
