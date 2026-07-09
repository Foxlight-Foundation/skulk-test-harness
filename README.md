# skulk-test-harness

Cluster-neutral test harness tooling for [Skulk](https://github.com/Foxlight-Foundation/Skulk).

The harness provides named model sets, named test sets, placement planning,
OpenAI-compatible chat/tool-call execution, and JSON/Markdown reports without
coupling experimental test logic into the main Skulk repository.

## Quick Start

```bash
uv sync
cp skulk-harness.example.yaml skulk-harness.yaml
uv run skulk-harness doctor
uv run skulk-harness models sets
uv run skulk-harness tests sets
uv run skulk-harness plan --model-set store-smoke --test-set chat-tests
```

For a more careful walkthrough, see the Docusaurus docs in `website/docs/`.
To preview them locally:

```bash
cd website
npm ci
npm run start
```

`run` defaults to dry-run. Pass `--execute` only when you want the harness to
place models and issue live requests:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

If `skulk-harness.yaml` is absent, the CLI falls back to safe defaults pointed
at `http://localhost:52415`.

## Public Defaults

The default configs are intentionally generic:

- model sets in `configs/model_sets.yaml`
- test sets in `configs/test_sets.yaml`
- example local config in `skulk-harness.example.yaml`

Built-in model sets include:

- `store-smoke`
- `store-all`
- `catalog-small-text`
- `embeddings`
- `speech-tts`
- `speech-stt`
- `vision`
- `served-spec-draft-simple`
- `served-spec-draft-eagle3`

Built-in test sets include:

- `chat-tests`
- `code-tests`
- `tool-tests`
- `throughput`
- `cancellation`
- `context-admission`
- `embeddings`
- `speech-synthesis`
- `speech-roundtrip`
- `vision`
- `served-speculation`

## What Requires a Live Cluster?

Commands that list local config, model sets, or test sets do not need a live
Skulk cluster. Commands that call `doctor`, inspect live catalog/store models,
plan placements, run tests, or request downloads need a reachable Skulk API.

The stability commands are advanced operational checks. `failover`, `churn`,
and `refusal` require `--execute-destructive` before they perform any API or SSH
side effects. Configure `cluster_nodes` with SSH hosts and explicit
`kill_command` / `relaunch_command` values before using them.

## Reports

Runs write JSON, JSONL, Markdown summaries, and artifacts under `runs/` by
default. Speech synthesis and speech roundtrip tests persist generated audio
under the run's `artifacts/` directory. The `runs/` directory is ignored by git.

## Foxlight Profile

Foxlight's production e2e matrix lives under `examples/foxlight/`. The root
`run_e2e_battery.sh`, `run_mtp_battery.sh`, `run_throughput_battery.sh`, and
`run_stability_battery.sh` entrypoints are compatibility wrappers for existing
Foxlight automation.

Foxlight operators can also invoke the profile directly:

```bash
uv run skulk-harness tests sets --config examples/foxlight/skulk-harness.yaml
uv run skulk-harness models sets --config examples/foxlight/skulk-harness.yaml
./run_e2e_battery.sh
./run_stability_battery.sh              # soaks (MLX + AMD engines), non-destructive
./run_stability_battery.sh --destructive  # + failover/churn/refusal
```

`run_stability_battery.sh` orchestrates the stability suites where
`run_e2e_battery.sh` proves each model works once: it soaks the MLX engine, the
AMD llama.cpp engine, and the AMD served-MTP engine under sustained concurrent
load, then (with `--destructive`) runs failover/churn/refusal. The AMD soaks are
the point for AMD Strix Halo deployments, whose single GPU node takes the load a
Mac cluster spreads across many. It pre-stages each model into the store first,
so a soak spends its window on load, not a first download. See
[Stability Suites](website/docs/guides/stability-suites.md).

The Foxlight stability example is intentionally separate at
`examples/foxlight/skulk-harness.stability.example.yaml` because it contains
destructive SSH process-control settings that each operator must adapt.

## Documentation Site

The user-facing documentation site lives under `website/` and is built with
Docusaurus. Pull requests build the site and upload an artifact. Pushes publish
to the `gh-pages` branch, with branch previews for non-main pushes.

```bash
cd website
npm ci
npm run build
```

## License

MIT
