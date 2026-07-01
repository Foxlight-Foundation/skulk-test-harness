#!/usr/bin/env bash
# Stability battery: sustained concurrent soak on the MLX and AMD engines, plus
# an optional destructive failover/churn/refusal pass. Where run_e2e_battery.sh
# proves each model works *once*, this proves the cluster holds up under
# *sustained* load and node loss.
#
# The AMD cells are the point. This release targets AMD Strix Halo owners, who
# have no MLX-style fallback and will hammer the single GPU node harder than Mac
# users did. So we soak the AMD llama.cpp engine and the AMD served-MTP
# (llama-server --spec-type draft-mtp) engine under concurrency and watch for
# wedge, memory creep, or an orphaned llama-server subprocess — not just a
# single happy-path request.
#
# Usage:
#   ./run_stability_battery.sh              # soaks only (non-destructive, safe)
#   ./run_stability_battery.sh --destructive  # + failover/churn/refusal (kills nodes over SSH)
#
# The soaks are non-destructive and self-contained (they pre-stage each model
# into the store first). The destructive trio requires a stability config whose
# cluster_nodes carry real ssh_host + kill/relaunch commands for THIS fleet; see
# skulk-harness.stability.example.yaml. Point STABILITY_CONFIG at it.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="examples/foxlight/skulk-harness.yaml"
STABILITY_CONFIG="${STABILITY_CONFIG:-examples/foxlight/skulk-harness.stability.yaml}"
# Store host API for pre-staging models before a soak. Defaults to the store
# host; override for a different fleet.
STORE_URL="${SKULK_STORE_URL:-http://kite3:52415}"

# Soak knobs (env-overridable). Concurrency is simultaneous completion workers;
# duration is wall-seconds of sustained load per cell.
SOAK_CONCURRENCY="${SOAK_CONCURRENCY:-4}"
MLX_SOAK_S="${MLX_SOAK_S:-120}"
AMD_SOAK_S="${AMD_SOAK_S:-240}"

DESTRUCTIVE=0
[ "${1:-}" = "--destructive" ] && DESTRUCTIVE=1

LOG=runs/stability_battery.log
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

battery_rc=0

# Pre-stage a model into the store so a soak doesn't spend its window on the
# first download. Idempotent: a model already in the store completes instantly.
ensure_store() {
  local model="$1"
  say "ensure-store: $model"
  curl -s -X POST --max-time 30 "$STORE_URL/store/models/$model/download" >/dev/null 2>&1 || true
  for _ in $(seq 1 480); do
    local st
    st=$(curl -s --max-time 10 "$STORE_URL/store/models/$model/download/status" 2>/dev/null \
      | python3 -c "import sys,json;print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    case "$st" in
      complete) return 0 ;;
      failed) say "ensure-store: $model FAILED"; return 1 ;;
    esac
    sleep 15
  done
  say "ensure-store: $model timed out"
  return 1
}

soak_cell() {
  local model="$1" conc="$2" dur="$3" label="$4"
  say "==== SOAK  $label  model=$model  concurrency=$conc  duration=${dur}s  START ===="
  ensure_store "$model" || { battery_rc=1; say "==== SOAK  $label  END (rc=store-download-failed) ===="; return; }
  uv run skulk-harness stability soak \
    --config "$CONFIG" \
    --model "$model" \
    --concurrency "$conc" \
    --duration-s "$dur" >>"$LOG" 2>&1
  local rc=$?
  [ "$rc" -ne 0 ] && battery_rc=$rc
  say "==== SOAK  $label  END (rc=$rc) ===="
}

destructive_cell() {
  local suite="$1"; shift
  say "==== DESTRUCTIVE  $suite  START ===="
  # shellcheck disable=SC2086 -- $* is an intentional optional flag list
  uv run skulk-harness stability "$suite" \
    --config "$STABILITY_CONFIG" \
    --execute-destructive "$@" >>"$LOG" 2>&1
  local rc=$?
  [ "$rc" -ne 0 ] && battery_rc=$rc
  say "==== DESTRUCTIVE  $suite  END (rc=$rc) ===="
}

say "STABILITY BATTERY START (destructive=$DESTRUCTIVE)"

# --- Soaks: sustained concurrent load, non-destructive ---
# MLX baseline: confirms the mature engine holds under load (regression anchor).
soak_cell mlx-community/Qwen3.5-9B-4bit "$SOAK_CONCURRENCY" "$MLX_SOAK_S" "mlx-baseline"
# AMD llama.cpp (kite4): sustained in-process GGUF decode on the GPU node.
soak_cell Qwen/Qwen2.5-7B-Instruct-GGUF "$SOAK_CONCURRENCY" "$AMD_SOAK_S" "amd-llama-cpp"
# AMD served-MTP (kite4): sustained load on llama-server --spec-type draft-mtp.
# The headline AMD path; watch for wedge / memory creep / orphan llama-server.
soak_cell unsloth/Qwen3.5-9B-MTP-GGUF "$SOAK_CONCURRENCY" "$AMD_SOAK_S" "amd-served-mtp"

# --- Destructive: node loss + placement refusal (opt-in) ---
if [ "$DESTRUCTIVE" = 1 ]; then
  destructive_cell failover
  destructive_cell churn
  destructive_cell refusal
else
  say "SKIP destructive suites (pass --destructive to run failover/churn/refusal)"
fi

say "STABILITY BATTERY COMPLETE (rc=$battery_rc)"
exit "$battery_rc"
