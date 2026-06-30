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

say "BATTERY COMPLETE (rc=$battery_rc)"
exit "$battery_rc"
