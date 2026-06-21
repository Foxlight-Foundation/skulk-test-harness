#!/usr/bin/env bash
# Real decode-throughput pass: the `throughput` test set (sustained 512-token
# generation, 3 reps) against the model matrix, so wall_tps is a meaningful
# steady-state tok/s. Each cell places from the store and tears down after.
set -u
cd "$(dirname "$0")"
LOG=runs/throughput_battery.log
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cell() {
  say "==== THROUGHPUT  model-set=$1  START ===="
  uv run skulk-harness run \
    --model-set "$1" \
    --test-set throughput \
    --execute \
    --ensure-store-downloads \
    --delete-created-instances >>"$LOG" 2>&1
  say "==== THROUGHPUT  model-set=$1  END (rc=$?) ===="
}

say "THROUGHPUT BATTERY START"
cell dense-singles
cell moe
cell multinode-large
cell gguf-big
say "THROUGHPUT BATTERY COMPLETE"
