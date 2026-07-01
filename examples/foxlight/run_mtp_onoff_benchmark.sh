#!/usr/bin/env bash
# MTP on-vs-off throughput benchmark for the served (llama_server) engine.
#
# Runs the mtp-served model set through the mtp-benchmark test set (200-token
# greedy, median of 3, production API) TWICE on the same GPU node build:
#   1. MTP ON  -- node launched normally, speculation active.
#   2. MTP OFF -- node relaunched with SKULK_LLAMA_SERVER_FORCE_NO_SPEC=1, which
#      forces plain decode of the IDENTICAL GGUF (see Skulk PR #434).
# Then combine_mtp_onoff.py diffs the two runs into a Plain / With-MTP / Gain
# table -- the served-engine equivalent of the MLX speculative-decoding table.
#
# The only difference between the two arms is the one env var, so the gain is
# attributable to speculation alone (same weights, same node, same protocol).
#
# The node toggle is done over SSH by editing the node's EnvironmentFile
# (~/.skulk/skulk.env, loaded by the systemd user service) and restarting the
# service. Configure the node with BENCH_NODE (ssh alias) and BENCH_NODE_ENV_FILE.
# The node must already run the build under test (deploy first); this script does
# not deploy code, only flips the spec env and restarts.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${BENCH_CONFIG:-examples/foxlight/skulk-harness.yaml}"
MODEL_SET="${BENCH_MODEL_SET:-mtp-served}"
TEST_SET="${BENCH_TEST_SET:-mtp-benchmark}"
BENCH_NODE="${BENCH_NODE:-kite4}"
BENCH_NODE_ENV_FILE="${BENCH_NODE_ENV_FILE:-.skulk/skulk.env}"  # relative to node $HOME
SPEC_VAR="SKULK_LLAMA_SERVER_FORCE_NO_SPEC"

LOG=runs/mtp_onoff_benchmark.log
mkdir -p "$(dirname "$LOG")"  # runs/ is gitignored; create it on a clean checkout
: > "$LOG"
# Progress goes to the log and to stderr, NOT stdout: run_arm's stdout is captured
# by command substitution to return the run dir, so any stray stdout would corrupt it.
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" >&2; }

# Run systemctl --user over ssh with the runtime dir set, so a non-interactive
# session can still reach the user service manager (requires linger enabled on
# the node, which the skulk user service already needs to survive logout).
node_ssh() { ssh "$BENCH_NODE" "XDG_RUNTIME_DIR=/run/user/\$(id -u) $*"; }

# Set the node's spec mode to "on" (remove the force-off var) or "off" (set it),
# then restart the skulk user service and wait for its API to come back. Editing
# is idempotent: the var appears at most once in the env file.
set_node_spec() {
  local mode="$1"  # on | off
  say "node $BENCH_NODE: set MTP $mode (restart skulk)"
  # Strip any existing line, then append when forcing off.
  node_ssh "sed -i '/^${SPEC_VAR}=/d' \"\$HOME/${BENCH_NODE_ENV_FILE}\" 2>/dev/null; \
            touch \"\$HOME/${BENCH_NODE_ENV_FILE}\"; \
            $( [ "$mode" = off ] && echo "echo '${SPEC_VAR}=1' >> \"\$HOME/${BENCH_NODE_ENV_FILE}\";" ) \
            systemctl --user restart skulk" >>"$LOG" 2>&1
  wait_node_ready
}

# Poll the node's API until it answers, then a short settle so the worker has
# re-advertised its served backend before the harness tries to place a model.
# The poll runs ON the node (ssh -> localhost) rather than from here: a headless
# worker's API binds 0.0.0.0:52415 but that port is typically only reachable
# inside the cluster LAN, not from the operator host (which drives placements
# through the cluster entry node, not this worker directly). The timeout is
# generous because a restart with SKULK_AUTO_UPDATE on runs git pull + uv sync
# before the API binds.
wait_node_ready() {
  local deadline=$(( $(date +%s) + 360 ))
  until node_ssh "curl -fsS -o /dev/null --max-time 5 http://localhost:52415/state" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      say "ERROR: $BENCH_NODE API did not come up within 360s after restart"
      exit 1
    fi
    sleep 5
  done
  sleep 12
  say "node $BENCH_NODE: API ready"
}

# Run one benchmark arm and echo (on stdout) the run dir the harness created, so
# the caller can hand the two dirs to the combiner. Harness chatter goes to LOG.
run_arm() {
  local label="$1"
  say "benchmark arm: $label"
  uv run skulk-harness run \
    --config "$CONFIG" \
    --model-set "$MODEL_SET" \
    --test-set "$TEST_SET" \
    --execute \
    --ensure-store-downloads \
    --delete-created-instances \
    --delete-staged-models >>"$LOG" 2>&1
  local rc=$?
  say "benchmark arm END: $label (rc=$rc)"
  [ "$rc" -ne 0 ] && return "$rc"
  # Newest run dir for this model set + test set.
  ls -dt runs/*-"${MODEL_SET}"-"${TEST_SET}"/ 2>/dev/null | head -1
}

say "MTP ON/OFF BENCHMARK START (node=$BENCH_NODE model_set=$MODEL_SET)"

set_node_spec on
ON_DIR="$(run_arm "MTP-ON")" || { say "ON arm failed"; set_node_spec on; exit 1; }
say "ON run dir: $ON_DIR"

set_node_spec off
OFF_DIR="$(run_arm "MTP-OFF")" || { say "OFF arm failed"; set_node_spec on; exit 1; }
say "OFF run dir: $OFF_DIR"

# Always restore the node to normal (speculation on) before reporting.
set_node_spec on

if [ -z "${ON_DIR:-}" ] || [ -z "${OFF_DIR:-}" ]; then
  say "ERROR: missing a run dir (ON='$ON_DIR' OFF='$OFF_DIR')"
  exit 1
fi

TABLE="runs/mtp_onoff_table.md"
say "combining -> $TABLE"
uv run python examples/foxlight/combine_mtp_onoff.py "$ON_DIR" "$OFF_DIR" --out "$TABLE" | tee -a "$LOG"
say "MTP ON/OFF BENCHMARK END (table: $TABLE)"
