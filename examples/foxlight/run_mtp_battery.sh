#!/usr/bin/env bash
# Native-MTP battery for the `mtp-served` model set (llama.cpp draft-mtp via the
# llama_server engine) on the GPU node (kite4). Two passes:
#   1. mtp-correctness -- reasoning split + no channel-marker leak + an MTP-on
#      throughput floor, all pass/fail, so a parser regression or a silent
#      speculative fallback fails the battery instead of slipping through green.
#   2. throughput      -- steady-state wall_tps benchmark (served-MTP decode rate).
# Each model: --ensure-store-downloads stages the GGUF from the model store (not
# hand-placed); --delete-created-instances tears the served instance down after its
# run; --delete-staged-models evicts the staged weights so test models do not
# accumulate on disk (benchmark hygiene).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="examples/foxlight/skulk-harness.yaml"
LOG=runs/mtp_battery.log
mkdir -p "$(dirname "$LOG")"  # runs/ is only gitignored; create it for a clean checkout
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "MTP BATTERY START"

say "MTP correctness pass (mtp-correctness)"
uv run skulk-harness run \
  --config "$CONFIG" \
  --model-set mtp-served \
  --test-set mtp-correctness \
  --execute \
  --ensure-store-downloads \
  --delete-created-instances \
  --delete-staged-models >>"$LOG" 2>&1
rc_correctness=$?
say "MTP correctness pass END (rc=$rc_correctness)"

say "MTP throughput pass (throughput)"
uv run skulk-harness run \
  --config "$CONFIG" \
  --model-set mtp-served \
  --test-set throughput \
  --execute \
  --ensure-store-downloads \
  --delete-created-instances \
  --delete-staged-models >>"$LOG" 2>&1
rc_throughput=$?
say "MTP throughput pass END (rc=$rc_throughput)"

# Battery fails if either pass failed. Capture rc explicitly; without this the
# script's exit code becomes the say/tee status (usually 0) and a failed battery
# looks green to CI.
rc=0
[ "$rc_correctness" -ne 0 ] && rc=$rc_correctness
[ "$rc_throughput" -ne 0 ] && rc=$rc_throughput
say "MTP BATTERY END (rc=$rc; correctness=$rc_correctness throughput=$rc_throughput)"

# --- Publish results to the ledger + prune published local runs (ON by
# default; hold with a .autopublish-results-off marker at the repo root or
# SKULK_PUBLISH_RESULTS=0). Non-fatal, runs regardless of pass/fail so failed
# cells still land in the ledger history. Output is tee'd into the battery log
# so an unattended run leaves a visible publish/skip record. ---
"$SCRIPT_DIR/publish_results.sh" 2>&1 | tee -a "$LOG" || true

exit "$rc"
