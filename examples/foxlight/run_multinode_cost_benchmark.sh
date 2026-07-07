#!/usr/bin/env bash
# Multi-node GGUF cost benchmark: what does RPC memory pooling cost when the
# model did not need it?
#
# For every model in the multinode-cost set (a size ladder that all fit kite4
# alone), run the sustained-300 test set TWICE:
#   SOLO   -- forced single-node on kite4 (--min-nodes 1, peer excluded).
#   POOLED -- forced across kite4+kite5 as a llama.cpp RPC driver+donor
#             (--min-nodes 2 --instance-meta LlamaRpc).
# The model never *needs* the second node, so the throughput and TTFT delta is
# pure interconnect + weight-split overhead. combine_multinode_cost.py diffs the
# two runs into a Solo / Pooled / cost table the AMD community can read as a
# curve (small models = worst case, large = amortized).
#
# The two arms differ ONLY by placement flags (no node restart, unlike the MTP
# on/off benchmark). The node build under test must already be deployed.
#
# Config via env:
#   BENCH_CONFIG     harness config (default examples/foxlight/skulk-harness.yaml)
#   BENCH_MODEL_SET  default multinode-cost
#   BENCH_TEST_SET   default multinode-cost
#   SOLO_NODE        the single node the SOLO arm runs on (default kite4)
#   PEER_NODE        the node excluded from SOLO / added by POOLED (default kite5)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${BENCH_CONFIG:-examples/foxlight/skulk-harness.yaml}"
MODEL_SET="${BENCH_MODEL_SET:-multinode-cost}"
TEST_SET="${BENCH_TEST_SET:-multinode-cost}"
SOLO_NODE="${SOLO_NODE:-kite4}"
PEER_NODE="${PEER_NODE:-kite5}"

LOG=runs/multinode_cost_benchmark.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
# Progress to log + stderr, never stdout (run_arm's stdout returns the run dir).
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" >&2; }

# Run one arm (whole model set), echo the run dir on stdout.
run_arm() {
  local label="$1"; shift
  say "benchmark arm: $label ($*)"
  # shellcheck disable=SC2068 -- $@ is an intentional flag list per arm
  uv run skulk-harness run \
    --config "$CONFIG" \
    --model-set "$MODEL_SET" \
    --test-set "$TEST_SET" \
    --sharding Pipeline \
    --execute \
    --ensure-store-downloads \
    --delete-created-instances \
    $@ >>"$LOG" 2>&1
  local rc=$?
  say "benchmark arm END: $label (rc=$rc)"
  [ "$rc" -ne 0 ] && return "$rc"
  ls -dt runs/*-"${MODEL_SET}"-"${TEST_SET}"/ 2>/dev/null | head -1
}

say "MULTI-NODE COST BENCHMARK START (solo=$SOLO_NODE peer=$PEER_NODE model_set=$MODEL_SET)"

# SOLO: single node on SOLO_NODE. Exclude the peer so the only GGUF-capable
# cycle is the solo node (Macs are auto-excluded by the common-engine rule).
# MlxRing is the generic single-node container (a solo served instance reports
# MlxRing; only pooled reports LlamaRpc).
SOLO_DIR="$(run_arm "SOLO" --min-nodes 1 --exclude-nodes "$PEER_NODE" --instance-meta MlxRing)" \
  || { say "SOLO arm failed"; exit 1; }
say "SOLO run dir: $SOLO_DIR"

# POOLED: force width 2 so single-node cycles are filtered out; the only 2-node
# GGUF cycle is solo+peer, minted as a llama.cpp RPC driver+donor.
POOLED_DIR="$(run_arm "POOLED" --min-nodes 2 --instance-meta LlamaRpc)" \
  || { say "POOLED arm failed"; exit 1; }
say "POOLED run dir: $POOLED_DIR"

if [ -z "${SOLO_DIR:-}" ] || [ -z "${POOLED_DIR:-}" ]; then
  say "ERROR: missing a run dir (SOLO='$SOLO_DIR' POOLED='$POOLED_DIR')"
  exit 1
fi

TABLE="runs/multinode_cost_table.md"
say "combining -> $TABLE"
uv run python examples/foxlight/combine_multinode_cost.py \
  "$SOLO_DIR" "$POOLED_DIR" --out "$TABLE" | tee -a "$LOG"
say "MULTI-NODE COST BENCHMARK END (table: $TABLE)"
