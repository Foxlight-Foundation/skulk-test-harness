#!/usr/bin/env bash
# Full e2e + benchmark battery against the live multi-node cluster. Each cell
# places from the store, runs its test set, then tears the instance down to free
# memory. The MLX/llama.cpp cells capture wall-clock TTFT + approx TPS; the MTP
# cell is a pass/fail correctness gate. A failing cell fails the whole battery
# (battery_rc), so a regression cannot look green.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="examples/foxlight/skulk-harness.yaml"
LOG="${SKULK_E2E_BATTERY_LOG:-runs/e2e_battery.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

stop_battery() {
  local rc="$1"
  trap - INT TERM
  say "BATTERY INTERRUPTED (rc=$rc)"
  exit "$rc"
}
trap 'stop_battery 130' INT
trap 'stop_battery 143' TERM

battery_rc=0
cell() {
  local mset="$1" tset="$2" extra="${3:-}"
  say "==== CELL  model-set=$mset  test-set=$tset  START ===="
  # shellcheck disable=SC2086 -- $extra is an intentional optional flag list
  uv run skulk-harness run \
    --config "$CONFIG" \
    --model-set "$mset" \
    --test-set "$tset" \
    --execute \
    --ensure-store-downloads \
    --delete-created-instances $extra >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
    stop_battery "$rc"
  fi
  [ "$rc" -ne 0 ] && battery_rc=$rc
  say "==== CELL  model-set=$mset  test-set=$tset  END (rc=$rc) ===="
}

say "BATTERY START on $(uv run skulk-harness doctor --config "$CONFIG" 2>/dev/null | grep -m1 API || echo cluster)"

# --- MLX matrix (kite1/2/3): correctness + TTFT/TPS benchmarks ---
cell dense-singles    chat-tests
cell moe              chat-tests
cell multinode-large  chat-tests
cell tensor-sharding  chat-tests        "--sharding Tensor --min-nodes 2"
cell smoke            cancellation
cell context-admission context-admission
cell embeddings       embeddings
cell speech-tts       speech-synthesis-semantic
cell speech-tts-streaming speech-data-pressure
cell speech-roundtrip-tts speech-roundtrip
cell speech-stt-realtime realtime-transcription
cell speech-stt-realtime conversational-realtime
cell speech-translation-tts speech-translation
cell speech-reference-tts speech-reference-conditioning
cell speech-voice-catalog-tts speech-voice-catalog
cell speech-stt-realtime streaming-transcription
cell speech-stt-realtime fabric-speech-chain
cell vision           vision
cell vision           vision-data-plane "--min-nodes 2"

# --- AMD / llama.cpp leg (kite4): GGUF coherence + the big models Bug A unlocked ---
cell gguf-llama-cpp   llama-cpp
cell gguf-big         llama-cpp

# --- Native MTP (served, GPU node e.g. kite4): correctness GATES -- reasoning
# split, no channel-marker leak, and an MTP-on throughput floor (so a silent
# draft-mtp fallback shows red). Pass/fail, and it evicts the staged GGUFs after
# (benchmark hygiene). It places via compatible_backends=llama_server-*, so it
# needs a llama-server node in the cluster; on an MLX-only sub-cluster placement
# finds no viable node and this cell reports a failure -- run the full battery
# where the GPU node is present. The long throughput benchmark stays in
# run_mtp_battery.sh.
cell mtp-served       mtp-correctness  --delete-staged-models

# --- Served tool calling (kite4, llama_server): a strict agentic round-trip on
# the served/GGUF path. Qwen3-Coder-30B (defaults to llama_server post-#607)
# must emit a real parsed tool_call with the right arguments AND use the mocked
# tool result in its final answer -- not merely complete coherently. Guards the
# agentic-GGUF-on-served path that a card routing change activated; needs a
# llama-server node present (kite4), so it fails loudly on an MLX-only cluster.
cell tool-served      tool-served-check

# --- Multi-node GGUF pooling (kite4 driver + kite5 donor, Skulk #328): the
# pooled 120B is forced onto the RPC shape with min_nodes 2 + LlamaRpc meta
# (smallest-cycle preference would otherwise single-node it on kite4). Needs
# BOTH AMD nodes present; with either absent, placement fails loudly and the
# cell reports red rather than silently skipping. Store re-download of the
# 60GB model is expected after a fleet cold restart (staging-orphan
# reconciliation), so this cell can take ~10 minutes on first run.
cell pooled-rpc       llama-cpp        "--min-nodes 2 --instance-meta LlamaRpc"

# --- 2nd AMD node coverage (kite5): the planner prefers the larger AMD node
# (kite4, 128GB) for every GGUF/served placement, so without excluding it the
# smaller Strix (kite5, 32GB VRAM) never serves inference and its llama.cpp +
# served-MTP paths go untested. --exclude-nodes kite4 forces these small cells
# onto kite5. If kite5 is absent, placement finds no viable node and the cell
# fails (visible, not silently skipped). ---
cell gguf-llama-cpp   llama-cpp        "--exclude-nodes kite4"
cell mtp-served-9b    mtp-correctness  "--exclude-nodes kite4 --delete-staged-models"

# --- Throughput-vs-concurrency sweep (non-MTP text) --------------------------
# The concurrency leg: same cells as run_concurrency_battery.sh, folded into the
# e2e so every run traces the throughput-vs-concurrency curve per model x engine
# x hardware, single-rank and multi-rank. MLX stops at the fleet policy
# SKULK_MAX_CONCURRENT_REQUESTS=16 (the Skulk code default is 8); 32/64 only
# measure queue latency on that path. The continuously batching GGUF path
# retains 32/64. Non-toggleable reasoning cards use a larger output
# budget so the sweep measures completed visible output instead of truncated
# analysis. A level that saturates (admission refuses) fails its cell, which is
# the finding, not a flake. Set SKULK_E2E_CONCURRENCY=0 to run the correctness/
# benchmark battery without this long leg.
if [ "${SKULK_E2E_CONCURRENCY:-1}" = "1" ]; then
  cell concurrency-mlx            concurrency-16
  cell concurrency-mlx-reasoning  concurrency-reasoning-16
  cell concurrency-mlx-multinode  concurrency-16  "--sharding Tensor --min-nodes 2"
  cell concurrency-gguf           concurrency
  cell concurrency-120b           concurrency-reasoning
  cell concurrency-gguf-pooled    concurrency-reasoning  "--min-nodes 2 --instance-meta LlamaRpc"
else
  say "==== CONCURRENCY LEG SKIPPED (SKULK_E2E_CONCURRENCY=0) ===="
fi

say "BATTERY COMPLETE (rc=$battery_rc)"

# --- Publish results to the ledger + prune published local runs (ON by
# default; hold with a .autopublish-results-off marker at the repo root or
# SKULK_PUBLISH_RESULTS=0). Non-fatal, runs regardless of pass/fail so failed
# cells still land in the ledger history. Output is tee'd into the battery log
# so an unattended run leaves a visible publish/skip record. ---
"$SCRIPT_DIR/publish_results.sh" 2>&1 | tee -a "$LOG" || true

exit "$battery_rc"
