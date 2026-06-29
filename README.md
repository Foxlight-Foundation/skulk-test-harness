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
- Expected-error, cancellation, image-input, and embedding test kinds
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
- `mtp-served`
- `gpt-oss-20b`
- `gguf-llama-cpp`
- `tensor-sharding`
- `context-admission`
- `embeddings`
- `vision`
- `served-spec-draft-simple`
- `served-spec-draft-eagle3`
- `store-all`

Test sets:

- `chat-tests`
- `code-tests`
- `asteroids-challenge`
- `gpt-oss-20b-complete`
- `llama-cpp`
- `throughput`
- `mtp-correctness`
- `cancellation`
- `context-admission`
- `embeddings`
- `vision`
- `served-speculation`

### AMD / llama.cpp (GGUF) node

`gguf-llama-cpp` + `llama-cpp` exercise the non-MLX engine on a GPU node (e.g. an
AMD Strix Halo box). The GGUF card's `compatible_backends` are llama.cpp tags, so
Skulk's placement filter routes the model only to a `llama_cpp` node; no
harness-side node pinning is needed (point `api_base_url` at any cluster node).

```bash
uv run skulk-harness run \
  --model-set gguf-llama-cpp \
  --test-set llama-cpp \
  --execute --delete-created-instances
```

The `llama-cpp` suite covers basic generation, streamed-order coherence
(`in_order_integers`), the tool-calling code path, and a harmony-marker leak
guard for the GGUF gpt-oss path. Per-token logprob parity is intentionally not
part of the default suite: llama.cpp requires a dedicated `logits_all`-enabled
placement for that, and a normal GGUF placement correctly returns no logprobs.

### Native MTP (served / llama.cpp `draft-mtp`)

`mtp-served` is the GGUF set served via the `llama_server` engine with native
multi-token prediction on a GPU node (kite4). It spans both MTP shapes: baked-in
heads (the Qwen `draft_mtp` cards) and a separate draft assistant via
`--model-draft` (Gemma 4 31B, which is also a reasoning model with a literal
`<|channel>` parser). `run_mtp_battery.sh` runs two passes against it:

```bash
./run_mtp_battery.sh   # pass 1: mtp-correctness (gates), pass 2: throughput (benchmark)
```

- **`mtp-correctness`** asserts the two failure modes that presence/coherence
  checks miss:
  - *reasoning split + no marker leak* -- `min_reasoning_chars` proves the
    thinking landed in its own channel, and `forbidden_substrings` (applied to
    content **and** reasoning via `forbid_in_reasoning`) proves no `<|channel>`
    control marker leaked into either. Guards the Gemma 4 served channel parser.
  - *MTP-on throughput floor* -- `min_wall_tps` fails a run whose decode rate
    dropped below a floor calibrated above the model's non-speculative rate, so a
    **silent** `draft-mtp` fallback (correct text, just slower) shows red instead
    of passing. Hardware/model-specific; the floor is kite4-Vulkan calibrated and
    is a reliable MTP-off detector for the mid/large dense + Gemma cards (the 9B
    is decode-bound-fast, the A3B MoE has less margin -- noted in the cell).
- **`throughput`** then records steady-state `wall_tps` as the benchmark number.

These new `SuccessCriteria` keys (`min_reasoning_chars`, `forbid_in_reasoning`,
`min_wall_tps`) are general and usable by any test set.

## Coverage Suites

- `tensor-sharding` + `chat-tests` with `--sharding Tensor --min-nodes 2`
  exercises the TP placement path separately from the default Pipeline cells.
- `smoke` + `cancellation` starts a streaming generation, closes the stream
  after a few chunks, then verifies the instance can still serve a follow-up.
- `context-admission` + `context-admission` sends an oversized request and
  expects a clean `context_length_exceeded` 400 plus a healthy follow-up.
- `embeddings` + `embeddings` calls `/v1/embeddings` and checks vector shape and
  non-zero norm.
- `vision` + `vision` sends an OpenAI-style `image_url` content part to a VLM
  and checks the answer.
- `served-spec-draft-simple` / `served-spec-draft-eagle3` + `served-speculation`
  select live catalog/store models by `runtime.served_spec_type` when cards for
  those llama_server modes are present.

## Transport Matrix

The data-plane transport matrix requires relaunching the Skulk fleet with
different node env vars, so it is an operator procedure rather than a normal
single-process harness flag. Run the same coherence cell twice:

```bash
# pass 1: gossipsub data plane
SKULK_ZENOH_DATA_PLANE=0 uv run skulk   # on each node, or via your service manager
uv run skulk-harness run -m multinode-large -t chat-tests --execute --delete-created-instances

# pass 2: Zenoh data plane
SKULK_ZENOH_DATA_PLANE=1 uv run skulk   # on each node with the normal Zenoh listen config
uv run skulk-harness run -m multinode-large -t chat-tests --execute --delete-created-instances
```

The coherence gate is `ordered-integers-coherence`; it catches token/sub-word
reordering on either transport path.

## Safety Notes

This harness is designed to coexist with other operators using the same Skulk
cluster. Dry-run planning is the default. Executions reuse existing placements
by default and retain harness-created instances unless explicitly asked to
delete them.
