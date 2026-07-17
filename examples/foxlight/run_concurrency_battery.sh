#!/usr/bin/env bash
# Throughput-vs-concurrency battery: non-MTP text generators across a range of
# model sizes (small -> large), on both engine families, single-rank and
# multi-rank, swept through concurrency 1, 4, 8, 32, 64. Each cell places from
# the store, runs the `concurrency` test set (aggregate tok/s + per-request
# decode p50/p90 + TTFT p50/p90 per level, keyed by model x engine x hardware),
# then tears the instance down. A failing cell fails the whole battery
# (battery_rc) so a regression cannot look green. Results publish to the ledger.
#
# Run standalone:  ./run_concurrency_battery.sh
# It is also invoked as the concurrency leg of run_e2e_battery.sh.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="examples/foxlight/skulk-harness.yaml"
LOG="${SKULK_CONCURRENCY_BATTERY_LOG:-runs/concurrency_battery.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

stop_battery() {
  local rc="$1"
  trap - INT TERM
  say "CONCURRENCY BATTERY INTERRUPTED (rc=$rc)"
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

say "CONCURRENCY BATTERY START (levels 1/4/8/32/64)"

# --- MLX, single rank (rank 0), small -> large ------------------------------
cell concurrency-mlx            concurrency

# --- MLX, multiple ranks (large dense forced across Apple nodes) -------------
cell concurrency-mlx-multinode  concurrency  "--sharding Tensor --min-nodes 2"

# --- llama.cpp / AMD GPU node, single rank, small -> large ------------------
cell concurrency-gguf           concurrency

# --- llama.cpp / AMD, multiple ranks (RPC memory pooling, driver + donor) ---
cell concurrency-gguf-pooled    concurrency  "--min-nodes 2 --instance-meta LlamaRpc"

say "CONCURRENCY BATTERY COMPLETE (rc=$battery_rc)"

# Publish + prune (same as the e2e battery; non-fatal, runs regardless of
# pass/fail so failed cells still land in the ledger history).
"$SCRIPT_DIR/publish_results.sh" 2>&1 | tee -a "$LOG" || true

exit "$battery_rc"
