#!/usr/bin/env bash
# Full e2e + benchmark battery against the live 4-node cluster on dev (98e09b6d:
# #364 send-queue + Bug B n_ctx + Bug A VRAM-aware placement all merged).
# Each cell places from the store, runs its test set (chat-tests captures
# wall-clock TTFT + approx TPS), then tears the instance down to free memory.
set -u
cd "$(dirname "$0")"
LOG=runs/e2e_battery.log
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cell() {
  local mset="$1" tset="$2"
  say "==== CELL  model-set=$mset  test-set=$tset  START ===="
  uv run skulk-harness run \
    --model-set "$mset" \
    --test-set "$tset" \
    --execute \
    --ensure-store-downloads \
    --delete-created-instances >>"$LOG" 2>&1
  say "==== CELL  model-set=$mset  test-set=$tset  END (rc=$?) ===="
}

say "BATTERY START on $(uv run skulk-harness doctor 2>/dev/null | grep -m1 API || echo cluster)"

# --- MLX matrix (kite1/2/3): correctness + TTFT/TPS benchmarks ---
cell dense-singles    chat-tests
cell moe              chat-tests
cell multinode-large  chat-tests

# --- AMD / llama.cpp leg (kite4): GGUF coherence + the big models Bug A unlocked ---
cell gguf-llama-cpp   llama-cpp
cell gguf-big         llama-cpp

say "BATTERY COMPLETE"
