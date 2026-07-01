#!/usr/bin/env bash
# Post-deploy performance regression: single-node spread across engines, then a
# forced 2-node MLX pipeline. Non-destructive (no failover/crash); tears down
# and evicts staged weights after each cell.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="examples/foxlight/skulk-harness.yaml"
LOG=runs/perf_check.log; mkdir -p runs; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "PERF CHECK START"
say "cell 1: perf-check x throughput (single-node: MLX dense, MLX MoE, AMD served)"
uv run skulk-harness run --config "$CONFIG" --model-set perf-check --test-set throughput \
  --execute --ensure-store-downloads --delete-created-instances --delete-staged-models >>"$LOG" 2>&1
rc1=$?; say "cell 1 END rc=$rc1"

say "cell 2: perf-multi x throughput (2-node MLX pipeline)"
uv run skulk-harness run --config "$CONFIG" --model-set perf-multi --test-set throughput \
  --execute --min-nodes 2 --ensure-store-downloads --delete-created-instances --delete-staged-models >>"$LOG" 2>&1
rc2=$?; say "cell 2 END rc=$rc2"

rc=0; [ "$rc1" -ne 0 ] && rc=$rc1; [ "$rc2" -ne 0 ] && rc=$rc2
say "PERF CHECK END (rc=$rc; cell1=$rc1 cell2=$rc2)"
exit "$rc"
