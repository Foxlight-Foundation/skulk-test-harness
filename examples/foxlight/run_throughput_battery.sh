#!/usr/bin/env bash
# Real decode-throughput pass: the `throughput` test set (sustained 512-token
# generation, 3 reps) against the model matrix, so wall_tps is a meaningful
# steady-state tok/s. Each cell places from the store and tears down after.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="examples/foxlight/skulk-harness.yaml"
LOG=runs/throughput_battery.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cell() {
  say "==== THROUGHPUT  model-set=$1  START ===="
  uv run skulk-harness run \
    --config "$CONFIG" \
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

# --- Publish results to the ledger + prune published local runs (ON by
# default; hold with a .autopublish-results-off marker at the repo root or
# SKULK_PUBLISH_RESULTS=0). Non-fatal, runs regardless of pass/fail so failed
# cells still land in the ledger history. Output is tee'd into the battery log
# so an unattended run leaves a visible publish/skip record. ---
"$SCRIPT_DIR/publish_results.sh" 2>&1 | tee -a "$LOG" || true
