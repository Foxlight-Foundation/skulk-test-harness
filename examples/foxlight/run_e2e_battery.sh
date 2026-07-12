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
LOG=runs/e2e_battery.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

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
cell speech-tts       speech-synthesis
cell speech-tts-streaming speech-data-pressure
cell speech-roundtrip-tts speech-roundtrip
cell speech-stt-realtime realtime-transcription
cell speech-translation-tts speech-translation
cell speech-reference-tts speech-reference-conditioning
cell speech-voice-catalog-tts speech-voice-catalog
cell speech-stt-realtime streaming-transcription
cell speech-stt-realtime fabric-speech-chain
cell vision           vision

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

say "BATTERY COMPLETE (rc=$battery_rc)"

# --- Publish results to the ledger + prune published local runs (ON by
# default; hold with a .autopublish-results-off marker at the repo root or
# SKULK_PUBLISH_RESULTS=0). Non-fatal, runs regardless of pass/fail so failed
# cells still land in the ledger history. Output is tee'd into the battery log
# so an unattended run leaves a visible publish/skip record. ---
"$SCRIPT_DIR/publish_results.sh" 2>&1 | tee -a "$LOG" || true

exit "$battery_rc"
